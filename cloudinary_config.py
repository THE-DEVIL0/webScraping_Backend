import os
import logging
from typing import Optional
from pydantic import HttpUrl
import cloudinary
import cloudinary.uploader
from io import BytesIO
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting Cloudinary uploader module")

# Cloudinary configuration
cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")
logger.info("Fetched Cloudinary environment variables")

if not all([cloud_name, api_key, api_secret]):
    logger.critical("Cloudinary configuration incomplete")
    raise RuntimeError("Cloudinary configuration incomplete")

cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret,
    secure=True,
)
logger.info("Cloudinary configured successfully")

async def upload_to_cloudinary(image_bytes: BytesIO, folder: str = "processed", format: str = "png") -> Optional[HttpUrl]:
    """
    Upload a BytesIO image to Cloudinary directly, without temp files.
    
    Args:
        image_bytes: BytesIO object containing image data
        folder: Cloudinary folder for organization
        format: Image format (default: "png" to preserve transparency)
    
    Returns:
        HttpUrl: Secure Cloudinary URL if successful, None otherwise
    """
    try:
        logger.info("Starting upload_to_cloudinary function")

        # Log size and hash of input image
        image_bytes.seek(0)
        image_data = image_bytes.getvalue()
        image_size = len(image_data)
        image_hash = hashlib.md5(image_data).hexdigest()
        logger.info(f"Received image: size={image_size} bytes, md5={image_hash}")

        # Reset pointer for upload
        image_bytes.seek(0)
        logger.info("Reset BytesIO pointer to start")

        # Upload directly from BytesIO
        upload_result = cloudinary.uploader.upload(
            file=image_bytes,
            folder=folder,
            use_filename=True,
            unique_filename=True,
            overwrite=False,
            resource_type="image",
            format=format  # Explicitly set format (e.g., "png")
        )

        logger.info(f"Upload result received: {upload_result}")

        url = upload_result.get("secure_url")
        if url:
            logger.info(f"Uploaded image to Cloudinary successfully: {url}")
            return url
        else:
            logger.error("Cloudinary upload returned no URL")
            return None

    except Exception as e:
        logger.exception(f"Cloudinary upload failed: {e}")
        return None