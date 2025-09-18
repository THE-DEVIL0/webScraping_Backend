import motor.motor_asyncio
import asyncio
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    uri = os.getenv("MONGO_URI", "mongodb+srv://qasim:test123@cluster0.x60ytnz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)
    try:
        await client.admin.command('ping')
        logger.info("MongoDB connection successful")
    except Exception as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(test_connection())