from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
import os
from dotenv import load_dotenv
import logging
from pymongo.errors import ConnectionFailure
from typing import Optional

# Load environment variables
load_dotenv()

# Database configuration
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "webScraping")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraper.log',
    filemode='a',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

# Global database client (asynchronous Motor)
client: Optional[AsyncIOMotorClient] = None


async def connect_to_mongo():
    """Create async database connection."""
    global client
    try:
        client = AsyncIOMotorClient(MONGO_URI)   # ✅ FIXED: using MONGO_URI
        await client.admin.command('ping')
        logger.info(f"✅ Connected to MongoDB at {MONGO_URI}, db={DATABASE_NAME}")
    except ConnectionFailure as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise Exception(f"Failed to connect to MongoDB: {e}")


async def close_mongo_connection():
    """Close async database connection."""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")
        client = None


def get_db_collection(collection_name: str) -> AsyncIOMotorCollection:
    """Get async database collection (synchronous operation)."""
    if not client:
        raise Exception("MongoDB client not initialized. Call connect_to_mongo first.")
    return client[DATABASE_NAME][collection_name]


async def get_all_tasks():
    try:
        tasks_collection = get_db_collection("tasks")  # Synchronous call
        tasks = await tasks_collection.find().to_list(length=None)
        for task in tasks:
            task["_id"] = str(task["_id"])
            if "created_at" in task:
                task["created_at"] = task["created_at"].isoformat()
            if "completed_at" in task:
                task["completed_at"] = task["completed_at"].isoformat()
        return tasks
    except Exception as e:
        logger.error(f"Failed to fetch tasks: {e}")
        raise Exception(f"Failed to fetch tasks: {str(e)}")


def get_database():
    """Get database instance"""
    return client[DATABASE_NAME] if client else None
