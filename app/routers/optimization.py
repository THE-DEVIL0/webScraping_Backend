from fastapi import APIRouter, HTTPException
from app.schemas import OptimizationRequest, OptimizationResponse
from app.services import optimize_images, _fetch_image_to_bytes  # ✅ Added _fetch_image_to_bytes
from database import get_db_collection
import logging, uuid
from datetime import datetime, timezone
from cloudinary_config import upload_to_cloudinary
from io import BytesIO  # ✅ Added for BytesIO handling

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/optimization", tags=["Image Optimization"])

@router.get("/health")
async def optimization_health():
    return {"status": "ok"}

async def store_task(task_id, task_type, status, input_data=None, output_data=None, metadata=None):
    tasks_collection = get_db_collection("tasks")
    task = {
        "task_id": task_id,
        "task_type": task_type,
        "status": status,
        "created_at": datetime.now(timezone.utc),
        "input_urls": input_data.get("input_urls", []) if input_data else [],
        "output_urls": output_data.get("output_urls", []) if output_data else [],
        "metadata": metadata or {}
    }
    if output_data and "completed_at" in output_data:
        task["completed_at"] = datetime.now(timezone.utc)
    await tasks_collection.insert_one(task)
    return task_id

async def update_task(task_id, status, output_data=None, metadata=None):
    tasks_collection = get_db_collection("tasks")
    update_data = {
        "$set": {
            "status": status,
            "metadata": metadata or {},
            "output_urls": output_data.get("output_urls", []) if output_data else []
        }
    }
    if output_data and "completed_at" in output_data:
        update_data["$set"]["completed_at"] = datetime.now(timezone.utc)
    await tasks_collection.update_one({"task_id": task_id}, update_data)
    return task_id

@router.post("/optimize", response_model=OptimizationResponse)
async def optimization_optimize(req: OptimizationRequest):
    task_id = str(uuid.uuid4())
    tasks_collection = get_db_collection("tasks")

    try:
        # Store queued task with URLs
        await tasks_collection.insert_one({
            "task_id": task_id,
            "task_type": "optimization",
            "status": "queued",
            "created_at": datetime.now(timezone.utc),
            "input_urls": [str(url) for url in req.image_urls],  # ✅ Use URLs
            "output_urls": [],
            "metadata": {"upscale": req.upscale, "denoise": req.denoise, "enhance_lighting": req.enhance_lighting}
        })

        # Fetch images into BytesIO
        image_objects = []
        for url in req.image_urls:
            image_bytes = await _fetch_image_to_bytes(url)
            if image_bytes:
                image_objects.append((url, image_bytes))
                logger.info(f"Fetched image from {url}")
            else:
                logger.warning(f"Failed to fetch image from {url}")
                continue

        # Process images in memory
        optimized_urls = await optimize_images(
            image_objects,
            upscale=req.upscale,
            denoise=req.denoise,
            enhance_lighting=req.enhance_lighting
        )

        # Update task in DB
        await tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {
                "status": "completed",
                "output_urls": optimized_urls,
                "completed_at": datetime.now(timezone.utc),
                "metadata": {
                    "processed": len(req.image_urls),
                    "successful": len(optimized_urls),
                    "failed": len(req.image_urls) - len(optimized_urls)
                }
            }}
        )

        return OptimizationResponse(optimized=optimized_urls)

    except Exception as e:
        await tasks_collection.update_one(
            {"task_id": task_id},
            {"$set": {"status": "error", "metadata": {"error": str(e)}}}
        )
        logger.error(f"Image optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")