from fastapi import APIRouter, HTTPException
from app.schemas import GenerationRequest, GenerationResponse, MergeRequest, MergeResponse
from app.services import generate_background, merge_images
from database import get_db_collection
import logging, uuid
from datetime import datetime, timezone
from io import BytesIO
import aiohttp
import cloudinary.uploader
import base64

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/generation", tags=["Background Generation"])

async def fetch_image_to_bytes(url: str) -> BytesIO:
    """Fetch an image from a URL to a BytesIO object asynchronously."""
    try:
        url_str = str(url)  # Convert HttpUrl to string
        logger.info(f"Fetching image from URL: {url_str}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url_str, timeout=10) as resp:
                logger.info(f"Response status for {url_str}: {resp.status}")
                if resp.status != 200:
                    logger.error(f"Failed to fetch {url_str}: HTTP {resp.status}")
                    return None
                content = await resp.read()
                logger.info(f"Successfully fetched {len(content)} bytes from {url_str}")
                return BytesIO(content)
    except Exception as e:
        logger.error(f"Failed to fetch {url_str}: {str(e)}")
        return None

@router.get("/health")
async def generation_health():
    return {"status": "ok"}

@router.post("/generate", response_model=GenerationResponse)
async def generation_generate(req: GenerationRequest):
    task_id = str(uuid.uuid4())
    tasks_collection = get_db_collection("tasks")

    try:
        logger.info(f"Creating generation task {task_id} with prompt: {req.prompt}")
        await tasks_collection.insert_one({
            "task_id": task_id,
            "task_type": "generation",
            "status": "queued",
            "created_at": datetime.now(timezone.utc),
            "input_urls": [],
            "output_urls": [],
            "metadata": {"prompt": req.prompt}
        })

        output_urls = await generate_background(
            req.prompt,
            req.size,
            req.quality,
            req.num_images,
            req.negative_prompt,
            
           
        )

        logger.info(f"Generated {len(output_urls)} images for task {task_id}")
        await tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "completed",
                "output_urls": output_urls,
                "completed_at": datetime.now(timezone.utc),
                "metadata": {"total_generated": len(output_urls)}
            }}
        )

        return GenerationResponse(status="success", images=output_urls, total_generated=len(output_urls))

    except Exception as e:
        logger.error(f"Generation task {task_id} failed: {str(e)}")
        await tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {"status": "error", "metadata": {"error": str(e)}}}
        )
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/merge", response_model=MergeResponse)
async def merge_background(req: MergeRequest):
    task_id = str(uuid.uuid4())
    tasks_collection = get_db_collection("tasks")

    try:
        logger.info(f"Creating merge task {task_id} with input URLs: {str(req.foreground_url)}, {str(req.background_url)}")
        await tasks_collection.insert_one({
            "task_id": task_id,
            "task_type": "merge",
            "status": "queued",
            "created_at": datetime.now(timezone.utc),
            "input_urls": [str(req.foreground_url), str(req.background_url)],
            "output_urls": [],
            "metadata": {
                "preview_mode": req.preview_mode,
                "foreground_scale": req.foreground_scale,
                "background_scale": req.background_scale,
                "position_x": req.position_x,
                "position_y": req.position_y
            }
        })

        # Fetch images to BytesIO
        logger.info(f"Fetching foreground image: {str(req.foreground_url)}")
        foreground_bytes = await fetch_image_to_bytes(req.foreground_url)
        logger.info(f"Fetching background image: {str(req.background_url)}")
        background_bytes = await fetch_image_to_bytes(req.background_url)
        if not foreground_bytes or not background_bytes:
            logger.error("One or both images failed to fetch")
            raise HTTPException(status_code=400, detail="Failed to fetch one or both input images")

        # Merge images in memory
        logger.info("Merging images with parameters: "
                    f"foreground_scale={req.foreground_scale}, "
                    f"background_scale={req.background_scale}, "
                    f"position_x={req.position_x}, position_y={req.position_y}, "
                    f"preview_mode={req.preview_mode}")
        merged = await merge_images(
            foreground_bytes,
            background_bytes,
            req.foreground_scale,
            req.background_scale,
            req.position_x,
            req.position_y,
            req.preview_mode
        )

        output_urls = []
        if not req.preview_mode:
            logger.info("Uploading merged image to Cloudinary")
            try:
                upload = cloudinary.uploader.upload(
                    merged,
                    folder="merged",
                    resource_type="image",
                    overwrite=True
                )
                output_urls.append(upload["secure_url"])
                logger.info(f"Uploaded to Cloudinary: {upload['secure_url']}")
                merged.seek(0)
            except Exception as e:
                logger.error(f"Cloudinary upload failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Cloudinary upload failed: {str(e)}")
        else:
            logger.info("Converting merged image to base64 for preview")
            try:
                merged.seek(0)
                merged_base64 = base64.b64encode(merged.getvalue()).decode()
                merged_url = f"data:image/png;base64,{merged_base64}"
                logger.info("Base64 encoding successful")
            except Exception as e:
                logger.error(f"Base64 encoding failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Base64 encoding failed: {str(e)}")

        logger.info(f"Updating task {task_id} to completed")
        await tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "completed",
                "output_urls": output_urls,
                "completed_at": datetime.now(timezone.utc),
                "metadata": {
                    "preview_mode": req.preview_mode,
                    "output_size": merged.getbuffer().nbytes / 1024**2
                }
            }}
        )

        logger.info(f"Returning MergeResponse with merged_image: {'base64' if req.preview_mode else output_urls[0]}")
        return MergeResponse(status="success", merged_image=output_urls[0] if output_urls else merged_url)

    except HTTPException as e:
        logger.error(f"Merge task {task_id} failed: {e.status_code}: {e.detail}")
        await tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "error",
                "metadata": {"error": e.detail, "status_code": e.status_code}
            }}
        )
        raise  # Re-raise HTTPException to preserve status code
    except Exception as e:
        logger.error(f"Merge task {task_id} failed: {str(e)}")
        await tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "error",
                "metadata": {"error": str(e)}
            }}
        )
        raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")