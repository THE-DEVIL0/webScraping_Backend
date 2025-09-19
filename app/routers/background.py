from fastapi import APIRouter, HTTPException
from app.schemas import BackgroundRemovalRequest, BackgroundRemovalResponse
from app.services import remove_background
from database import get_db_collection
import logging
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/background", tags=["Background Removal"])

@router.get("/health")
async def background_health():
    logger.info("Health check endpoint called for background removal")
    return {"status": "ok"}

# app/background.py
@router.post("/remove", response_model=BackgroundRemovalResponse)
async def remove_background_endpoint(req: BackgroundRemovalRequest):
    task_id = str(uuid.uuid4())
    logger.info(f"New background removal task started: {task_id}, URLs: {req.image_urls}")
    print(f"background.py: Received image_urls: {req.image_urls}")

    try:
        # Delegate to services.remove_background, passing task_id
        result = await remove_background(req.image_urls, add_white_bg=req.add_white_bg, task_id=task_id)  # ✅ Pass task_id
        logger.info(f"[{task_id}] Background removal result: {result}")
        print(f"background.py: Result from remove_background: {result}")

        return BackgroundRemovalResponse(
            processed=result.get("processed", 0),
            successful=result.get("successful", 0),
            failed=result.get("failed", 0),
            results=result.get("results", []),
            task_id=task_id
        )

    except Exception as e:
        logger.error(f"[{task_id}] Background removal failed: {e}")
        # Update task status to error
        try:
            tasks_collection = get_db_collection("background_removal_tasks")
            if tasks_collection is not None:  # ✅ Fix: Use explicit None check
                await tasks_collection.update_one(
                    {"task_id": task_id},
                    {
                        "$set": {
                            "status": "error",
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "metadata": {"error": str(e)}
                        }
                    }
                )
            else:
                logger.warning(f"Database connection is None for task {task_id} error update")
        except Exception as db_e:
            logger.error(f"[{task_id}] Failed to update task status to error: {db_e}")
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")
    
    

@router.get("/tasks")
async def get_background_tasks():
    """Get all background removal tasks from database"""
    logger.info("Fetching all background removal tasks")
    try:
        tasks_collection = get_db_collection("background_removal_tasks")
        tasks = await tasks_collection.find().to_list(length=50)
        for task in tasks:
            task["_id"] = str(task["_id"])
            if "created_at" in task:
                task["created_at"] = task["created_at"].isoformat()
            if "completed_at" in task:
                task["completed_at"] = task["completed_at"].isoformat()
        logger.info(f"Fetched {len(tasks)} background tasks")
        return {"tasks": tasks, "total": len(tasks)}
    except Exception as e:
        logger.error(f"Failed to retrieve background tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tasks: {str(e)}")

@router.get("/tasks/{task_id}")
async def get_background_task(task_id: str):
    """Get specific background removal task by ID"""
    logger.info(f"Fetching task {task_id} from database")
    try:
        tasks_collection = get_db_collection("background_removal_tasks")
        task = await tasks_collection.find_one({"task_id": task_id})
        if not task:
            logger.warning(f"Task {task_id} not found")
            raise HTTPException(status_code=404, detail="Task not found")
        task["_id"] = str(task["_id"])
        if "created_at" in task:
            task["created_at"] = task["created_at"].isoformat()
        if "completed_at" in task:
            task["completed_at"] = task["completed_at"].isoformat()
        logger.info(f"Fetched task {task_id} successfully")
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve background task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task: {str(e)}")