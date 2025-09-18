from typing import Dict, Optional, Any
from datetime import datetime
import logging
from database import get_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="scraper.log",
    filemode="a",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


def make_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure MongoDB-safe data:
    - Convert datetime -> ISO string
    - Convert HttpUrl / Pydantic types -> str
    - Recursively handle dicts/lists
    """
    safe = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            safe[key] = value.isoformat()
        elif isinstance(value, (list, tuple)):
            safe[key] = [str(v) if not isinstance(v, (dict, list)) else make_serializable(v) for v in value]
        elif isinstance(value, dict):
            safe[key] = make_serializable(value)
        else:
            # Convert HttpUrl or any Pydantic/BaseModel-ish object to str
            safe[key] = str(value) if not isinstance(value, (int, float, bool, type(None))) else value
    return safe


class DatabaseService:
    def __init__(self):
        self.db = get_database()

    def create_scraping_session(self, session_data: Dict) -> str:
        """Create a new scraping session in MongoDB."""
        try:
            session_data["created_at"] = datetime.utcnow()
            safe_data = make_serializable(session_data)
            self.db.scraping_sessions.insert_one(safe_data)
            logger.info(f"Created scraping session: {session_data['session_id']}")
            return session_data["session_id"]
        except Exception as e:
            logger.error(f"Failed to create scraping session: {e}")
            raise

    def get_scraping_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve a scraping session by ID."""
        try:
            session = self.db.scraping_sessions.find_one({"session_id": session_id})
            if session:
                session["_id"] = str(session["_id"])
            return session
        except Exception as e:
            logger.error(f"Failed to retrieve session {session_id}: {e}")
            raise

    def update_scraping_session(self, session_id: str, update_data: Dict) -> bool:
        """Update a scraping session."""
        try:
            update_data["updated_at"] = datetime.utcnow()
            safe_data = make_serializable(update_data)
            result = self.db.scraping_sessions.update_one(
                {"session_id": session_id}, {"$set": safe_data}
            )
            logger.info(
                f"Updated session {session_id}: {result.modified_count} documents"
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            raise

    def create_product(self, product_data: Dict) -> str:
        """Create a new product in MongoDB."""
        try:
            product_data["created_at"] = datetime.utcnow()
            safe_data = make_serializable(product_data)
            self.db.products.insert_one(safe_data)
            logger.info(f"Created product: {product_data['product_id']}")
            return product_data["product_id"]
        except Exception as e:
            logger.error(f"Failed to create product: {e}")
            raise

    def get_product(self, product_id: str) -> Optional[Dict]:
        """Retrieve a product by ID."""
        try:
            product = self.db.products.find_one({"product_id": product_id})
            if product:
                product["_id"] = str(product["_id"])
            return product
        except Exception as e:
            logger.error(f"Failed to retrieve product {product_id}: {e}")
            raise

    def create_background_removal_task(self, task_data: Dict) -> str:
        """Create a new background removal task in MongoDB."""
        try:
            task_data["created_at"] = datetime.utcnow()
            safe_data = make_serializable(task_data)
            self.db.background_removal_tasks.insert_one(safe_data)
            logger.info(f"Created background removal task: {task_data['task_id']}")
            return task_data["task_id"]
        except Exception as e:
            logger.error(f"Failed to create background removal task: {e}")
            raise

    def get_background_removal_task(self, task_id: str) -> Optional[Dict]:
        """Retrieve a background removal task by ID."""
        try:
            task = self.db.background_removal_tasks.find_one({"task_id": task_id})
            if task:
                task["_id"] = str(task["_id"])
            return task
        except Exception as e:
            logger.error(f"Failed to retrieve background task {task_id}: {e}")
            raise
