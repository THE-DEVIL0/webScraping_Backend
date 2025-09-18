from fastapi import APIRouter, HTTPException
from database import get_db_collection, get_all_tasks
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraper.log'
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.get("/", response_model=dict)
async def get_tasks():
    try:
        tasks = await get_all_tasks()  # Correctly await the function
        return {"tasks": tasks, "total": len(tasks)}
    except Exception as e:
        logger.error(f"Failed to fetch tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch tasks: {str(e)}")