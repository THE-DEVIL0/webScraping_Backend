from typing import Dict, List, Optional
from scrappers.amazon_image_scraper import AmazonImageScraper
from scrappers.ebay_image_scraper import EbayImageScraper
from scrappers.shopify_image_scraper import ShopifyImageScraper
from image_processing.background_generation import BackgroundGenerator
from PIL import Image
import hashlib
import logging
from pathlib import Path
import httpx
from io import BytesIO
import cv2
import numpy as np
from cv2.dnn_superres import DnnSuperResImpl_create
import time
import aiohttp
import tempfile
import asyncio
from fastapi import HTTPException
import os
import json
from pydantic import HttpUrl
from image_processing.background_remover import BackgroundRemover
from database import get_database
from models import BackgroundRemovalTask
import uuid
from cloudinary_config import upload_to_cloudinary  # Added for Cloudinary uploadfrom datetime import datetime
from datetime import datetime



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraper.log'
)
logger = logging.getLogger(__name__)

# Initialize BackgroundGenerator with OpenAI API key
from config import OPENAI_API_KEY
generator = BackgroundGenerator(api_key=OPENAI_API_KEY, model_name="dall-e-3")

async def _download_image_to_temp(url: HttpUrl) -> Optional[str]:
    """Download an image from a URL to a temporary file asynchronously."""
    try:
        url_str = str(url)  # 🔥 Convert HttpUrl to plain string

        async with aiohttp.ClientSession() as session:
            async with session.get(url_str, timeout=10) as resp:
                if resp.status != 200:
                    logger.error(f"Failed to download {url_str}: HTTP {resp.status}")
                    print(f"_download_image_to_temp: Failed for {url_str}, HTTP {resp.status}, returned None")
                    return None
                ext = url_str.split('.')[-1].split('?')[0].lower()
                if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                    ext = 'jpg'
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
                async for chunk in resp.content.iter_chunked(8192):
                    temp_file.write(chunk)
                temp_file.close()
                logger.info(f"Temporarily downloaded {url_str} to {temp_file.name}")
                print(f"_download_image_to_temp: Returned temp file: {temp_file.name}")
                return temp_file.name
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        print(f"_download_image_to_temp: Failed for {url}, returned None")
        return None

    
async def _fetch_image_to_bytes(url: HttpUrl) -> Optional[BytesIO]:
    """Fetch an image from a URL to a BytesIO object asynchronously."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(str(url), timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to fetch {url}: HTTP {response.status_code}")
                print(f"_fetch_image_to_bytes: Failed for {url}, HTTP {response.status_code}, returned None")
                return None
            image_bytes = BytesIO(response.content)
            print(f"_fetch_image_to_bytes: Fetched {url} to BytesIO")
            return image_bytes
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        print(f"_fetch_image_to_bytes: Failed for {url}, error: {e}, returned None")
        return None

async def remove_background(image_urls: List[HttpUrl], add_white_bg: bool = False) -> Dict:
    task_id = str(uuid.uuid4())
    logger.info(f"Starting background removal task {task_id} with URLs: {image_urls}")
    print(f"remove_background: Starting task {task_id} with image_urls: {image_urls}")

    # Create DB task
    task = BackgroundRemovalTask(
        task_id=task_id,
        input_urls=image_urls,
        status="processing",
        add_white_bg=add_white_bg,
        metadata={"total_input_images": len(image_urls)}
    )

    try:
        db = get_database()
        if db:
            task_dict = task.model_dump(by_alias=True)
            task_dict['input_urls'] = [str(url) for url in task_dict['input_urls']]
            await db.background_removal_tasks.insert_one(task_dict)
            logger.info(f"Background removal task {task_id} created in database")
            print(f"remove_background: Task {task_id} inserted into database")
    except Exception as e:
        logger.error(f"Failed to save task {task_id} to database: {e}")
        print(f"remove_background: Failed to insert task {task_id} to database: {e}")

    # Fetch images into PIL
    image_objects = []
    for url in image_urls:
        image_bytes = await _fetch_image_to_bytes(url)
        if image_bytes:
            try:
                image = Image.open(image_bytes).convert("RGB")
                image_objects.append((url, image))
                print(f"remove_background: Fetched and converted {url} to PIL Image")
            except Exception as e:
                logger.warning(f"Failed to open image {url}: {e}")
                print(f"remove_background: Failed to open image {url}: {e}, skipping")
        else:
            logger.warning(f"Failed to fetch {url}")
            print(f"remove_background: Failed to fetch {url}, skipping")

    # Process images
    remover = BackgroundRemover(max_workers=2)
    stats = remover.process_selected_images(
        image_objects=[(url, img) for url, img in image_objects],
        add_white_background=add_white_bg,
        progress_callback=None
    )
    print(f"remove_background: BackgroundRemover stats: {stats}")

    # Upload processed images
    cloud_urls: List[HttpUrl] = []
    processed_bytes = stats.get("results", [])  # List of (url, BytesIO)
    for original_url, processed_image_bytes in processed_bytes:
        if not isinstance(processed_image_bytes, BytesIO):
            logger.error(f"Expected BytesIO, got {type(processed_image_bytes)} for {original_url}")
            continue

        try:
            # Always upload via temp file
            cloud_url = await upload_to_cloudinary(processed_image_bytes, folder="background_removal")
            if cloud_url:
                cloud_urls.append(cloud_url)
                logger.info(f"Uploaded image from {original_url} to Cloudinary: {cloud_url}")
                print(f"remove_background: Uploaded image from {original_url} to Cloudinary: {cloud_url}")
            else:
                logger.error(f"Cloudinary returned no URL for {original_url}")
                print(f"remove_background: No Cloudinary URL for {original_url}")
                stats["failed"] = stats.get("failed", 0) + 1
                stats["successful"] = max(stats.get("successful", 0) - 1, 0)
        except Exception as e:
            logger.error(f"Failed to upload image {original_url} to Cloudinary: {e}")
            print(f"remove_background: Failed to upload image from {original_url} to Cloudinary: {e}")
            stats["failed"] = stats.get("failed", 0) + 1
            stats["successful"] = max(stats.get("successful", 0) - 1, 0)

    # Update DB with results
    try:
        if db:
            update_dict = {
                "status": "completed",
                "output_urls": [str(url) for url in cloud_urls],
                "completed_at": datetime.now().isoformat(),
                "metadata.successful": stats.get("successful", 0),
                "metadata.failed": stats.get("failed", 0),
                "metadata.processed": stats.get("processed", 0)
            }
            await db.background_removal_tasks.update_one(
                {"task_id": task_id},
                {"$set": update_dict}
            )
            logger.info(f"Background removal task {task_id} completed and saved to database")
            print(f"remove_background: Task {task_id} updated in database")
    except Exception as e:
        logger.error(f"Failed to update task {task_id} in database: {e}")
        print(f"remove_background: Failed to update task {task_id} in database: {e}")

    result = {
        "processed": stats.get("processed", 0),
        "successful": stats.get("successful", 0),
        "failed": stats.get("failed", 0),
        "results": cloud_urls,
        "task_id": task_id
    }
    print(f"remove_background: Returning result: {result}")
    return result



async def scrape_amazon(url: str, max_products: Optional[int] = None, headless: bool = True, driver_type: str = "edge") -> Dict:
    """Scrape image URLs from Amazon asynchronously."""
    try:
        def custom_log(msg: str, log_type: str):
            logger.log(logging.INFO if log_type in ["info", "success"] else logging.ERROR, f"[Amazon] {msg}")

        scraper = AmazonImageScraper(url=url, max_workers=8, log_callback=custom_log, headless=headless, driver_type=driver_type)
        start_time = time.time()
        image_urls = await scraper.scrape_async(max_products=max_products)
        elapsed_time = time.time() - start_time

        product_urls = await scraper._fetch_product_urls(max_products or 100)
        
        if not image_urls:
            custom_log("No images scraped, possible selector mismatch or rate-limiting", "error")
            raise HTTPException(status_code=500, detail="No images scraped, check URL or selectors")

        result = {
            "total_products": len(product_urls),
            "total_images": len(image_urls),
            "image_urls": image_urls,
            "unique_images": len(set(image_urls)),
            "errors": [],
            "elapsed_time": elapsed_time
        }
        print(f"scrape_amazon response:\n{json.dumps(result, indent=2)}")  # Added console log
        logger.info(f"Amazon scrape completed: {len(image_urls)} images from {url}")
        return result
    except Exception as e:
        logger.error(f"Amazon scraping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Amazon scraping failed: {str(e)}")

def scrape_ebay(url: str, max_products: Optional[int] = None) -> Dict:
    """Scrape image URLs from eBay."""
    try:
        def custom_log(msg: str, log_type: str):
            logger.log(logging.INFO if log_type in ["info", "success"] else logging.ERROR, f"[eBay] {msg}")

        scraper = EbayImageScraper(max_workers=4, log_callback=custom_log, high_res_only=True)
        start_time = time.time()
        image_urls = scraper.scrape(url, max_products=max_products)
        elapsed_time = time.time() - start_time

        product_urls, error = scraper._fetch_product_urls(url, max_products)
        
        result = {
            "total_products": len(product_urls),
            "total_images": len(image_urls),
            "image_urls": image_urls,
            "unique_images": len(set(image_urls)),
            "errors": [] if not error else [{"error": error}],
            "elapsed_time": elapsed_time
        }
        print(f"scrape_ebay response:\n{json.dumps(result, indent=2)}")  # Added console log
        logger.info(f"[eBay] Scrape completed: {len(image_urls)} images from {url}")
        return result
    except Exception as e:
        logger.error(f"[eBay] Scraping failed: {e}")
        result = {
            "total_products": 0,
            "total_images": 0,
            "image_urls": [],
            "unique_images": 0,
            "errors": [{"error": str(e)}],
            "elapsed_time": 0,
            "error": str(e)
        }
        print(f"scrape_ebay error response:\n{json.dumps(result, indent=2)}")  # Added console log
        return result

def scrape_shopify(url: str, max_products: Optional[int] = None) -> Dict:
    """Scrape image URLs from a Shopify store."""
    try:
        def custom_log(msg: str, log_type: str):
            logger.log(logging.INFO if log_type in ["info", "success"] else logging.ERROR, f"[Shopify] {msg}")

        scraper = ShopifyImageScraper(max_workers=8, log_callback=custom_log)
        start_time = time.time()
        image_urls = scraper.scrape(url, max_products=max_products)
        elapsed_time = time.time() - start_time

        product_urls, error = scraper._fetch_product_urls(url, max_products)
        
        result = {
            "total_products": len(product_urls),
            "total_images": len(image_urls),
            "image_urls": image_urls,
            "unique_images": len(set(image_urls)),
            "errors": [] if not error else [{"error": error}],
            "elapsed_time": elapsed_time
        }
        print(f"scrape_shopify response:\n{json.dumps(result, indent=2)}")  # Added console log
        logger.info(f"Shopify scrape completed: {len(image_urls)} images from {url}")
        return result
    except Exception as e:
        logger.error(f"Shopify scraping failed: {e}")
        result = {
            "total_products": 0,
            "total_images": 0,
            "image_urls": [],
            "unique_images": 0,
            "errors": [{"error": str(e)}],
            "elapsed_time": 0,
            "error": str(e)
        }
        print(f"scrape_shopify error response:\n{json.dumps(result, indent=2)}")  # Added console log
        return result



async def generate_background(
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_images: int = 1,
    size: str = "1024x1024",
    quality: str = "standard"
) -> List[HttpUrl]:
    try:
        image_bytes_list = generator.generate(
            prompt=prompt,
            num_images=num_images,
            negative_prompt=negative_prompt,
            seed=None,
            size=size,
            quality=quality
        )
        output_urls = []

        for i, image_bytes in enumerate(image_bytes_list, 1):
            try:
                # Load image from BytesIO
                image_bytes.seek(0)
                img_array = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning(f"Could not load generated image {i}")
                    continue

                # Sharpen image
                gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
                sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

                # Save to BytesIO
                output_buffer = BytesIO()
                _, encoded_img = cv2.imencode(".jpg", sharpened, [cv2.IMWRITE_JPEG_QUALITY, 95])
                output_buffer.write(encoded_img)
                output_buffer.seek(0)

                # Upload to Cloudinary
                cloud_url = await upload_to_cloudinary(output_buffer, folder="generated")
                if cloud_url:
                    output_urls.append(cloud_url)
                    logger.info(f"Sharpened and uploaded image {i} to Cloudinary: {cloud_url}")
                else:
                    logger.error(f"Failed to upload sharpened image {i} to Cloudinary")

            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                continue

        return output_urls
    except Exception as e:
        logger.error(f"Failed to generate backgrounds: {e}")
        raise
    
    
async def optimize_images(
    image_objects: List[tuple[HttpUrl, BytesIO]], 
    upscale: bool = True, 
    denoise: bool = True, 
    enhance_lighting: bool = True
) -> List[HttpUrl]:
    if not image_objects:
        logger.warning("No images provided for optimization.")
        return []

    optimized_urls = []
    output_dir = Path("images") / "optimized"  # ✅ Kept for logging, not used for storage

    # Load super-res model
    sr = None
    if upscale:
        model_path = Path("EDSR_x4.pb")
        if not model_path.exists():
            logger.error(
                f"Super-resolution model {model_path} not found. "
                "Download EDSR_x4.pb from https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb "
                "and place it in 'D:\\webScrip-main\\EDSR_x4.pb'."
            )
            upscale = False
        else:
            try:
                sr = DnnSuperResImpl_create()
                sr.readModel(str(model_path))
                sr.setModel("edsr", 4)
                logger.info("Super-resolution model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load super-resolution model: {e}. Skipping upscaling.")
                upscale = False
                sr = None

    failed_count = 0
    for url, image_bytes in image_objects:
        try:
            # Load image from BytesIO
            image_bytes.seek(0)
            img_array = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"Could not load image from {url}")
                failed_count += 1
                continue

            h, w = img.shape[:2]
            has_alpha = len(img.shape) == 3 and img.shape[2] == 4

            # Handle alpha
            if has_alpha:
                alpha = img[:, :, 3]
                img_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGBA2BGR)
            else:
                img_bgr = img.copy()

            # 1. Upscale
            if upscale and sr is not None:
                img_upscaled = sr.upsample(img_bgr)
                img_upscaled = cv2.bilateralFilter(img_upscaled, 3, 30, 30)
                laplacian_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * 0.7
                img_upscaled = cv2.filter2D(img_upscaled, -1, laplacian_kernel)
                img_bgr = img_upscaled
                logger.info(f"Upscaled image from {url} from {w}x{h} to {img_bgr.shape[1]}x{img_bgr.shape[0]}")

            # 2. Denoise
            if denoise:
                img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)

            # 3. Enhance lighting
            if enhance_lighting and not has_alpha:
                gamma = 1.1
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                img_bgr = cv2.LUT(img_bgr, table)

                kernel_size = 15
                dark_channel = np.min(img_bgr, axis=2)
                dark_channel = cv2.erode(dark_channel, np.ones((kernel_size, kernel_size), np.uint8))
                atmospheric_light = np.percentile(dark_channel, 99)
                transmission = 1.0 - np.minimum(0.95 * dark_channel / max(1, atmospheric_light), 0.95)
                transmission = cv2.resize(transmission, (img_bgr.shape[1], img_bgr.shape[0]))
                dehazed = np.zeros_like(img_bgr, dtype=np.float32)
                for c in range(3):
                    dehazed[:, :, c] = np.clip(
                        (img_bgr[:, :, c].astype(np.float32) - 0.95 * atmospheric_light) / np.maximum(transmission, 0.1) + 0.95 * atmospheric_light,
                        0, 255
                    )
                img_bgr = dehazed.astype(np.uint8)
            elif enhance_lighting and has_alpha:
                gamma = 1.1
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                img_bgr = cv2.LUT(img_bgr, table)

            # 4. Contrast/Sharpen
            if denoise or enhance_lighting:
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                img_bgr = cv2.merge((cl, a, b))
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_LAB2BGR)

                channels = list(cv2.split(img_bgr))
                means = [np.mean(ch) for ch in channels]
                avg_mean = np.mean(means)
                for i in range(len(channels)):
                    scale = avg_mean / max(means[i], 1)
                    channels[i] = cv2.multiply(channels[i], scale)
                img_bgr = cv2.merge(channels)

                gaussian = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=0.8)
                unsharp = cv2.addWeighted(img_bgr, 2.5, gaussian, -1.5, 0)
                sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * 0.4
                img_bgr = cv2.addWeighted(unsharp, 1.0, cv2.filter2D(unsharp, -1, sharpen_kernel), 0.6, 0)

            # Re-attach alpha
            if has_alpha:
                if upscale and sr is not None:
                    alpha = cv2.resize(alpha, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                img_final = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA)
                img_final[:, :, 3] = alpha
            else:
                img_final = img_bgr

            # Save to BytesIO
            output_buffer = BytesIO()
            if has_alpha:
                _, encoded_img = cv2.imencode(".png", img_final, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            else:
                _, encoded_img = cv2.imencode(".jpg", img_final, [cv2.IMWRITE_JPEG_QUALITY, 98])
            output_buffer.write(encoded_img)
            output_buffer.seek(0)

            # Upload to Cloudinary
            cloud_url = await upload_to_cloudinary(output_buffer, folder="optimized")
            if cloud_url:
                optimized_urls.append(cloud_url)
                logger.info(f"Optimized and uploaded image from {url} to {cloud_url}")
            else:
                logger.error(f"Failed to upload optimized image from {url} to Cloudinary")
                failed_count += 1

        except Exception as e:
            logger.error(f"Error optimizing image from {url}: {e}")
            failed_count += 1
            continue

    logger.info(f"Optimized {len(image_objects)} images: {len(optimized_urls)} success, {failed_count} failed.")
    return optimized_urls


async def merge_images(
    foreground_bytes: BytesIO,
    background_bytes: BytesIO,
    foreground_scale: float = 1.0,
    background_scale: float = 1.0,
    position_x: int = 0,
    position_y: int = 0,
    preview_mode: bool = False
) -> BytesIO:
    from PIL import Image
    import logging
    import base64

    logger = logging.getLogger(__name__)

    try:
        # Load images from BytesIO
        foreground = Image.open(foreground_bytes)
        background = Image.open(background_bytes)

        if foreground.mode != "RGBA":
            foreground = foreground.convert("RGBA")

        if foreground_scale == 1.0:
            target_width = int(background.width * 0.5)
            aspect_ratio = foreground.height / foreground.width
            new_size = (target_width, int(target_width * aspect_ratio))
            foreground = foreground.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Auto-scaled foreground to {new_size}")
        else:
            new_size = (int(foreground.width * foreground_scale), int(foreground.height * foreground_scale))
            foreground = foreground.resize(new_size, Image.Resampling.LANCZOS)

        if background_scale != 1.0:
            new_size = (int(background.width * background_scale), int(background.height * background_scale))
            background = background.resize(new_size, Image.Resampling.LANCZOS)

        if background.mode != "RGB":
            background = background.convert("RGB")

        # Create merged image
        merged = Image.new("RGBA", background.size, (0, 0, 0, 0))
        merged.paste(background, (0, 0))

        if position_x == 0 and position_y == 0:
            paste_x = (background.width - foreground.width) // 2
            paste_y = (background.height - foreground.height) // 2
        else:
            paste_x = max(0, min(position_x, background.width - foreground.width))
            paste_y = max(0, min(position_y, background.height - foreground.width))

        merged.paste(foreground, (paste_x, paste_y), foreground)

        # Save to BytesIO
        output_buffer = BytesIO()
        merged.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        logger.info("Merged image created in memory")

        return output_buffer

    except Exception as e:
        logger.error(f"Failed to merge images: {e}")
        raise