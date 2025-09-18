from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import background, generation, optimization, scrapers, pipeline, tasks
from database import connect_to_mongo, close_mongo_connection
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import sys
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()  # log to console too
    ]
)
logger = logging.getLogger(__name__)


# --- Lifespan events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle."""
    try:
        await connect_to_mongo()
        logger.info("✅ MongoDB connection established")
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
        sys.exit(1)  # stop app if DB fails

    yield  # Application runs here

    try:
        await close_mongo_connection()
        logger.info("✅ MongoDB connection closed")
    except Exception as e:
        logger.error(f"❌ Failed to close MongoDB connection: {str(e)}")


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="AI Product Image Pipeline with MongoDB",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# --- Routers ---
try:
    app.include_router(scrapers.router)
    app.include_router(background.router)
    app.include_router(generation.router)
    app.include_router(optimization.router)
    app.include_router(pipeline.router, prefix="/api")
    app.include_router(tasks.router, prefix="/api")
    logger.info("✅ Routers included successfully")
except Exception as e:
    logger.error(f"❌ Failed to include routers: {str(e)}")
    sys.exit(1)


# --- Health check endpoint ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "mongodb"}
