from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import ScrapeRequest, ScrapeResponse
from app.services import scrape_amazon, scrape_ebay, scrape_shopify, _download_image_to_temp
from pydantic import BaseModel, HttpUrl
from typing import List, Dict
import logging
import zipfile
from io import BytesIO
from pathlib import Path
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraper.log'
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scrapers", tags=["Scrapers"])

class DownloadZipRequest(BaseModel):
    files: List[Dict[str, str]]  # Each dict has "url" (HttpUrl) and "filename" (str)
    platform: str

@router.post("/amazon", response_model=ScrapeResponse)
async def scrape_amazon_endpoint(req: ScrapeRequest):
    try:
        logger.info(f"Scraping amazon with URL {req.url}   max_products {req.max_products} ")
        stats = await scrape_amazon(str(req.url), max_products=req.max_products, headless=req.headless, driver_type=req.driver_type)
        return ScrapeResponse(**stats)
    except Exception as e:
        logger.error(f"Scraping amazon failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping amazon failed: {str(e)}")

@router.post("/ebay", response_model=ScrapeResponse)
def scrape_ebay_endpoint(req: ScrapeRequest):
    try:
        logger.info(f"Scraping ebay with URL {req.url}")
        stats = scrape_ebay(str(req.url), max_products=req.max_products)
        return ScrapeResponse(**stats)
    except Exception as e:
        logger.error(f"Scraping ebay failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping ebay failed: {str(e)}")

@router.post("/shopify", response_model=ScrapeResponse)
def scrape_shopify_endpoint(req: ScrapeRequest):
    try:
        logger.info(f"Scraping shopify with URL {req.url}")
        stats = scrape_shopify(str(req.url), max_products=req.max_products)
        return ScrapeResponse(**stats)
    except Exception as e:
        logger.error(f"Scraping shopify failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping shopify failed: {str(e)}")

@router.post("/download-zip")
async def download_zip(req: DownloadZipRequest) -> StreamingResponse:
    try:
        logger.info(f"Creating ZIP for {len(req.files)} images from platform {req.platform}")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file in req.files:
                url = file.get("url")
                filename = file.get("filename", url.split("/")[-1].split("?")[0] or "image.jpg")
                if not url:
                    logger.warning(f"Skipping invalid file entry: {file}")
                    continue
                try:
                    temp_path = await _download_image_to_temp(url)
                    if temp_path and Path(temp_path).exists():
                        zip_file.write(temp_path, f"{req.platform}/{filename}")
                        Path(temp_path).unlink()  # Delete temp file
                        logger.info(f"Added {filename} to ZIP")
                    else:
                        logger.warning(f"Failed to download {url}")
                except Exception as e:
                    logger.error(f"Failed to process {url}: {e}")
                    continue
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={req.platform}-images-{datetime.now().strftime('%Y-%m-%d')}.zip"}
        )
    except Exception as e:
        logger.error(f"Failed to create ZIP: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP: {str(e)}")