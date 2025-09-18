import os
from pathlib import Path
from typing import List
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Image Toolkit UI", page_icon="🧰", layout="wide")
st.title("🧰 Image Toolkit via FastAPI")

# Initialize session state
if 'preview_image' not in st.session_state:
    st.session_state.preview_image = None
if 'bg_selected' not in st.session_state:
    st.session_state.bg_selected = set()
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'merge_params' not in st.session_state:
    st.session_state.merge_params = {
        'foreground_scale': 1.0,
        'background_scale': 1.0,
        'position_x': 0,
        'position_y': 0
    }

# Settings sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_base = st.text_input("FastAPI Base URL", value="http://127.0.0.1:8000", help="Root URL where FastAPI is running")
    health_btn = st.button("Check API Health")
    st.caption("Endpoints used: /api/scrapers/*, /api/background/*, /api/optimization/*, /api/generation/*, /api/tasks")

if health_btn:
    try:
        resp1 = requests.get(f"{api_base}/api/background/health", timeout=5)
        resp2 = requests.get(f"{api_base}/api/optimization/health", timeout=5)
        resp3 = requests.get(f"{api_base}/api/generation/health", timeout=5)
        responses = [resp1, resp2, resp3]
        for resp in responses:
            if not resp.ok:
                st.sidebar.error(f"Health check failed for {resp.url}: {resp.status_code} - {resp.text[:200]}")
        if all(resp.ok for resp in responses):
            st.sidebar.success("All APIs are healthy")
        else:
            st.sidebar.warning("Some API health checks failed")
    except Exception as e:
        st.sidebar.error(f"Failed to reach API: {e}")

# Helpers
base_dir = Path("images")

def discover_images(root: Path) -> List[Path]:
    results: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                results.append(Path(dirpath) / f)
    return sorted(results)

# Tabs
scrapers_tab, bg_tab, opt_tab, gen_tab, merge_tab, tasks_tab = st.tabs(
    ["🕷️ Scrapers", "🎨 Background Removal", "🛠️ Optimization", "🖌️ Generation", "🔗 Merge", "📋 Task Report"]
)

with scrapers_tab:
    st.subheader("Scrape Product Images")
    platform = st.selectbox("Platform", ["amazon", "ebay", "shopify"], index=0)
    url = st.text_input("Store/Product URL", value="")
    max_products = st.number_input("Max products (0 for all)", min_value=0, value=0)
    col1, col2 = st.columns(2)
    with col1:
        start_scrape = st.button("Start Scrape", type="primary")
    with col2:
        status_placeholder = st.empty()

    if start_scrape:
        if not url.strip():
            st.warning("Provide a valid URL")
            st.stop()
        payload = {
            "url": url.strip(),
            "max_products": (None if max_products == 0 else int(max_products)),
            "headless": True,
            "driver_type": "edge"
        }
        endpoint = f"{api_base}/api/scrapers/{platform}"
        try:
            with st.spinner(f"Scraping {platform}..."):
                resp = requests.post(endpoint, json=payload, timeout=600)
                if not resp.ok:
                    status_placeholder.error(f"API error: {resp.status_code} - {resp.text[:200]}")
                    st.stop()
                data = resp.json()
                status_placeholder.success(f"Products: {data.get('total_products',0)} | Images: {data.get('total_images',0)} | Downloads: {data.get('successful_downloads',0)}")
                
                # Display local images for all platforms
                if base_dir.exists():
                    recent = sorted(discover_images(base_dir), key=lambda p: p.stat().st_mtime, reverse=True)[:12]
                    if recent:
                        st.caption(f"Recent images ({platform})")
                        cols = st.columns(4)
                        for i, p in enumerate(recent):
                            with cols[i % 4]:
                                try:
                                    if p.exists():
                                        st.image(str(p), caption=p.name, width=200)
                                    else:
                                        st.write(f"Image not found: {p.name}")
                                except Exception as e:
                                    st.write(f"Error displaying {p.name}: {e}")
                    else:
                        st.info(f"No images found for {platform}.")
                else:
                    st.info("No images directory found.")
        except Exception as e:
            status_placeholder.error(f"Request failed: {e}")

with bg_tab:
    st.subheader("Background Removal")
    if not base_dir.exists():
        st.info("No images directory found yet. Run a scrape first.")
    else:
        all_images = discover_images(base_dir)
        if not all_images:
            st.info("No images found.")
        else:
            rel_paths = [str(p.relative_to(base_dir)) for p in all_images]
            count_placeholder = st.empty()
            colA, colB = st.columns(2)
            with colA:
                if st.button("Select All", key="bg_select_all"):
                    for rp in rel_paths:
                        st.session_state.bg_selected.add(rp)
                    st.rerun()
            with colB:
                if st.button("Clear Selection", key="bg_clear_selection"):
                    st.session_state.bg_selected.clear()
                    st.rerun()

            cols = st.columns(4)
            for i, p in enumerate(all_images):
                rel = str(p.relative_to(base_dir))
                with cols[i % 4]:
                    checked = rel in st.session_state.bg_selected
                    new_state = st.checkbox("✓ Select", value=checked, key=f"bg_{rel}")
                    if new_state:
                        st.session_state.bg_selected.add(rel)
                    else:
                        st.session_state.bg_selected.discard(rel)
                    try:
                        if p.exists():
                            st.image(str(p), caption=p.name, width=220)
                        else:
                            st.write(f"Image not found: {p.name}")
                    except Exception as e:
                        st.write(f"Error displaying {p.name}: {e}")

            count_placeholder.caption(f"Total images: {len(rel_paths)} | Selected: {len(st.session_state.bg_selected)}")

            add_white_bg = st.toggle("Add White Background", value=False)
            start_btn = st.button("Start Background Removal", type="primary")

            result_placeholder = st.empty()
            if start_btn:
                if not st.session_state.bg_selected:
                    st.warning("Select at least one image.")
                    st.stop()
                selected_abs = [str((base_dir / rp).resolve()) for rp in sorted(st.session_state.bg_selected)]
                payload = {
                    "image_paths": selected_abs,
                    "add_white_bg": bool(add_white_bg),
                }
                try:
                    with st.spinner("Calling FastAPI to remove backgrounds..."):
                        resp = requests.post(f"{api_base}/api/background/remove", json=payload, timeout=300)
                        if not resp.ok:
                            result_placeholder.error(f"API error: {resp.status_code} - {resp.text[:200]}")
                            st.stop()
                        data = resp.json()
                        result_placeholder.success(f"Processed: {data.get('processed', 0)} | Successful: {data.get('successful', 0)} | Failed: {data.get('failed', 0)}")
                        results = data.get("results", [])
                        if results:
                            st.caption("Showing up to first 12 outputs")
                            cols = st.columns(4)
                            for i, out_path in enumerate(results[:12]):
                                with cols[i % 4]:
                                    try:
                                        if Path(out_path).exists():
                                            st.image(out_path, caption=Path(out_path).name, width=200)
                                        else:
                                            st.write(f"Image not found: {Path(out_path).name}")
                                    except Exception as e:
                                        st.write(f"Error displaying {Path(out_path).name}: {e}")
                        else:
                            st.info("No result files returned.")
                except Exception as e:
                    result_placeholder.error(f"Request failed: {e}")

with opt_tab:
    st.subheader("Image Optimization")
    if not base_dir.exists():
        st.info("No images directory found yet. Run a scrape first.")
    else:
        all_images = discover_images(base_dir)
        if not all_images:
            st.info("No images found.")
        else:
            rel_paths = [str(p.relative_to(base_dir)) for p in all_images]
            selected_rel_opt = st.multiselect("Choose images to optimize", rel_paths, key="opt_select")
            upscale = st.toggle("Upscale", value=True)
            denoise = st.toggle("Denoise", value=True)
            opt_btn = st.button("Run Optimization", type="primary")
            opt_result = st.empty()
            if opt_btn:
                if not selected_rel_opt:
                    st.warning("Select at least one image.")
                    st.stop()
                selected_abs = [str((base_dir / rp).resolve()) for rp in selected_rel_opt]
                payload = {"image_paths": selected_abs, "upscale": bool(upscale), "denoise": bool(denoise)}
                try:
                    with st.spinner("Calling FastAPI to optimize images..."):
                        resp = requests.post(f"{api_base}/api/optimization/optimize", json=payload, timeout=300)
                        if not resp.ok:
                            opt_result.error(f"API error: {resp.status_code} - {resp.text[:200]}")
                            st.stop()
                        data = resp.json()
                        paths = data.get("optimized", [])
                        opt_result.success(f"Optimized {len(paths)} images.")
                        if paths:
                            cols = st.columns(4)
                            for i, p in enumerate(paths[:12]):
                                with cols[i % 4]:
                                    try:
                                        if Path(p).exists():
                                            st.image(p, caption=Path(p).name, width=200)
                                        else:
                                            st.write(f"Image not found: {Path(p).name}")
                                    except Exception as e:
                                        st.write(f"Error displaying {Path(p).name}: {e}")
                except Exception as e:
                    opt_result.error(f"Request failed: {e}")

with gen_tab:
    st.subheader("Background Generation")
    prompt = st.text_input("Prompt", value="studio product background, soft light")
    negative = st.text_input("Negative prompt", value="")
    num_images = st.number_input("Number of images", min_value=1, max_value=10, value=1)
    size = st.selectbox("Image size", ["1024x1024", "1792x1024", "1024x1792"], index=0)
    quality = st.selectbox("Quality", ["standard", "hd"], index=1)
    gen_btn = st.button("Generate Backgrounds", type="primary")
    gen_result = st.empty()
    if gen_btn:
        payload = {
            "prompt": prompt,
            "negative_prompt": (negative or None),
            "num_images": int(num_images),
            "size": size,
            "quality": quality
        }
        try:
            with st.spinner("Calling FastAPI to generate backgrounds..."):
                resp = requests.post(f"{api_base}/api/generation/generate", json=payload, timeout=300)
                if not resp.ok:
                    gen_result.error(f"API error: {resp.status_code} - {resp.text[:200]}")
                    st.stop()
                data = resp.json()
                images = data.get("images", [])
                st.session_state.generated_images = images
                if images:
                    cols = st.columns(4)
                    for i, p in enumerate(images[:12]):
                        with cols[i % 4]:
                            try:
                                if Path(p).exists():
                                    st.image(p, caption=Path(p).name, width=200)
                                else:
                                    st.write(f"Image not found: {Path(p).name}")
                            except Exception as e:
                                st.write(f"Error displaying {Path(p).name}: {e}")
                    gen_result.success(f"Generated {len(images)} images")
                else:
                    gen_result.info("No images returned")
        except Exception as e:
            gen_result.error(f"Request failed: {e}")

with merge_tab:
    st.subheader("Merge Background-Removed or Optimized Image with Generated Background")
    st.info("Select a background-removed or optimized image as the product (foreground) and a generated background. Adjust sliders to customize the merge.")
    if not base_dir.exists():
        st.info("No images directory found yet. Run a scrape, background removal, or optimization first.")
    else:
        # Discover images
        bg_removed_dir = base_dir / "background_removed"
        optimized_dir = base_dir / "optimized"
        generated_dir = base_dir / "generated_backgrounds"

        # Collect foreground images (background-removed and optimized)
        bg_removed_images = discover_images(bg_removed_dir) if bg_removed_dir.exists() else []
        optimized_images = discover_images(optimized_dir) if optimized_dir.exists() else []
        foreground_images = bg_removed_images + optimized_images
        foreground_options = [str(p) for p in foreground_images]

        generated_images = discover_images(generated_dir) if generated_dir.exists() else []
        generated_options = [str(p) for p in generated_images]

        if not foreground_images or not generated_images:
            st.info("No background-removed, optimized, or generated background images found. Run background removal, optimization, and generation first.")
        else:
            # Show original images
            st.subheader("Selected Images")
            col1, col2 = st.columns(2)
            with col1:
                foreground_path = st.selectbox(
                    "Select Foreground Image (Background-Removed or Optimized)",
                    foreground_options,
                    key="foreground_select",
                    format_func=lambda x: f"{Path(x).parent.name}/{Path(x).name}"
                )
                if foreground_path and Path(foreground_path).exists():
                    try:
                        st.image(foreground_path, caption="Foreground Image", width=200)
                    except Exception as e:
                        st.write(f"Error displaying {foreground_path}: {e}")
                else:
                    st.write("No foreground image selected or image not found")
            with col2:
                background_path = st.selectbox(
                    "Select Generated Background",
                    generated_options,
                    key="background_select",
                    format_func=lambda x: Path(x).name
                )
                if background_path and Path(background_path).exists():
                    try:
                        st.image(background_path, caption="Background Image", width=200)
                    except Exception as e:
                        st.write(f"Error displaying {background_path}: {e}")
                else:
                    st.write("No background image selected or image not found")

            # Adjustment controls
            st.subheader("Adjustment Controls")
            col3, col4 = st.columns(2)
            with col3:
                foreground_scale = st.slider(
                    "Foreground Size Scale",
                    min_value=0.5,
                    max_value=2.0,
                    value=st.session_state.merge_params['foreground_scale'],
                    step=0.1,
                    key="fg_scale",
                    help="Set to 1.0 for auto-scaling (50% of background width)"
                )
                background_scale = st.slider(
                    "Background Zoom",
                    min_value=0.8,
                    max_value=1.5,
                    value=st.session_state.merge_params['background_scale'],
                    step=0.1,
                    key="bg_scale"
                )
            with col4:
                position_x = st.slider(
                    "Horizontal Position (X)",
                    min_value=-200,
                    max_value=200,
                    value=st.session_state.merge_params['position_x'],
                    step=10,
                    key="pos_x",
                    help="Set to 0 for auto-centering"
                )
                position_y = st.slider(
                    "Vertical Position (Y)",
                    min_value=-200,
                    max_value=200,
                    value=st.session_state.merge_params['position_y'],
                    step=10,
                    key="pos_y",
                    help="Set to 0 for auto-centering"
                )

            # Update session state
            if (st.session_state.merge_params['foreground_scale'] != foreground_scale or
                st.session_state.merge_params['background_scale'] != background_scale or
                st.session_state.merge_params['position_x'] != position_x or
                st.session_state.merge_params['position_y'] != position_y):
                st.session_state.merge_params.update({
                    'foreground_scale': foreground_scale,
                    'background_scale': background_scale,
                    'position_x': position_x,
                    'position_y': position_y
                })
                # Trigger preview
                if foreground_path and background_path and Path(foreground_path).exists() and Path(background_path).exists():
                    payload = {
                        "foreground_path": foreground_path,
                        "background_path": background_path,
                        "foreground_scale": foreground_scale,
                        "background_scale": background_scale,
                        "position_x": position_x,
                        "position_y": position_y,
                        "output_dir": "images/merged",
                        "preview_mode": True
                    }
                    try:
                        with st.spinner("Generating preview..."):
                            resp = requests.post(f"{api_base}/api/generation/merge", json=payload, timeout=30)
                            if not resp.ok:
                                st.error(f"Preview failed: {resp.status_code} - {resp.text[:200]}")
                            else:
                                data = resp.json()
                                merged_image = data.get("merged_image")
                                if merged_image and Path(merged_image).exists():
                                    st.session_state.preview_image = merged_image
                                else:
                                    st.session_state.preview_image = None
                    except Exception as e:
                        st.error(f"Preview request failed: {e}")
                        st.session_state.preview_image = None

            # Display preview
            merge_result = st.empty()
            if st.session_state.preview_image and Path(st.session_state.preview_image).exists():
                merge_result.image(st.session_state.preview_image, caption="Preview (Adjust sliders to update)", width=400)
            else:
                merge_result.info("No preview available. Adjust settings and select valid images to generate a preview.")

            # Finalize and save
            save_btn = st.button("Finalize and Save", type="primary")
            if save_btn:
                if not foreground_path or not background_path or not Path(foreground_path).exists() or not Path(background_path).exists():
                    st.warning("Please select valid foreground and background images.")
                    st.stop()
                payload = {
                    "foreground_path": foreground_path,
                    "background_path": background_path,
                    "foreground_scale": foreground_scale,
                    "background_scale": background_scale,
                    "position_x": position_x,
                    "position_y": position_y,
                    "output_dir": "images/merged",
                    "preview_mode": False
                }
                try:
                    with st.spinner("Saving final image..."):
                        resp = requests.post(f"{api_base}/api/generation/merge", json=payload, timeout=30)
                        if not resp.ok:
                            merge_result.error(f"API error: {resp.status_code} - {resp.text[:200]}")
                            st.stop()
                        data = resp.json()
                        merged_image = data.get("merged_image")
                        if merged_image and Path(merged_image).exists():
                            st.image(merged_image, caption=f"Final Merged Image (Saved: {merged_image})", width=400)
                            with open(merged_image, "rb") as f:
                                st.download_button("Download Final Image", f, file_name=Path(merged_image).name)
                            merge_result.success("Image saved successfully!")
                            st.session_state.preview_image = None  # Clear preview after saving
                        else:
                            merge_result.info("No merged image returned or image not found")
                            st.session_state.preview_image = None
                except Exception as e:
                    merge_result.error(f"Save request failed: {e}")
                    st.session_state.preview_image = None

with tasks_tab:
    st.subheader("Task Report")
    st.info("View the history of all tasks (scraping, background removal, generation, optimization, pipeline).")

    # Add task type filter
    task_type_filter = st.selectbox("Filter by Task Type", ["All", "scraping", "background_removal", "generation", "optimization", "merge", "pipeline"], index=0)

    # Fetch tasks from database
    try:
        with st.spinner("Loading tasks from database..."):
            resp = requests.get(f"{api_base}/api/tasks/", timeout=10)
            if resp.ok:
                data = resp.json()
                tasks = data.get("tasks", [])
                if task_type_filter != "All":
                    tasks = [task for task in tasks if task["task_type"] == task_type_filter]
                total = len(tasks)

                if tasks:
                    st.success(f"Found {total} tasks")
                    
                    # Display tasks in a table
                    for i, task in enumerate(tasks):
                        task_type = task.get('task_type', 'Unknown')
                        with st.expander(f"Task {i+1}: {task.get('task_id', 'Unknown')[:8]}... - {task_type} - {task.get('status', 'Unknown')}"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.write(f"**Type:** {task_type}")
                                st.write(f"**Status:** {task.get('status', 'Unknown')}")
                                st.write(f"**Created:** {task.get('created_at', 'Unknown')}")
                                st.write(f"**Completed:** {task.get('completed_at', 'Not completed')}")

                            with col2:
                                if task_type in ["background_removal", "optimization", "generation", "merge"]:
                                    st.write(f"**Input Images:** {len(task.get('input_paths', []))}")
                                    st.write(f"**Output Images:** {len(task.get('output_paths', []))}")
                                elif task_type == "scraping":
                                    st.write(f"**URL:** {task.get('url', 'N/A')}")
                                    st.write(f"**Platform:** {task.get('platform', 'N/A')}")
                                elif task_type == "pipeline":
                                    st.write(f"**Platform:** {task.get('platform', 'N/A')}")
                                    st.write(f"**Product URL:** {task.get('product_url', 'N/A')}")

                            with col3:
                                metadata = task.get('metadata', {})
                                if task_type == "scraping":
                                    st.write(f"**Total Products:** {metadata.get('total_products', 0)}")
                                    st.write(f"**Total Images:** {metadata.get('total_images', 0)}")
                                    st.write(f"**Successful Downloads:** {metadata.get('successful_downloads', 0)}")
                                elif task_type in ["background_removal", "optimization"]:
                                    st.write(f"**Processed:** {metadata.get('processed', 0)}")
                                    st.write(f"**Successful:** {metadata.get('successful', 0)}")
                                    st.write(f"**Failed:** {metadata.get('failed', 0)}")
                                elif task_type == "generation":
                                    st.write(f"**Prompt:** {task.get('prompt', 'N/A')}")
                                    st.write(f"**Total Generated:** {metadata.get('total_generated', 0)}")
                                elif task_type == "merge":
                                    st.write(f"**Foreground Path:** {task.get('foreground_path', 'N/A')}")
                                    st.write(f"**Background Path:** {task.get('background_path', 'N/A')}")
                                elif task_type == "pipeline":
                                    st.write(f"**Max Products:** {task.get('max_products', 'N/A')}")
                                    st.write(f"**Generated Images:** {len(task.get('generated_images', []))}")

                            # Show output images if available
                            output_paths = task.get('output_paths', [])
                            if output_paths:
                                st.write("**Output Images:**")
                                cols = st.columns(4)
                                for j, path in enumerate(output_paths[:8]):  # Show max 8 images
                                    with cols[j % 4]:
                                        try:
                                            if Path(path).exists():
                                                st.image(path, caption=Path(path).name, width=150)
                                            else:
                                                st.write(f"Image not found: {Path(path).name}")
                                        except Exception as e:
                                            st.write(f"Error displaying {Path(path).name}: {e}")
                else:
                    st.info("No tasks found in database")
            else:
                st.error(f"Failed to fetch tasks: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        st.error(f"Error loading tasks: {e}")