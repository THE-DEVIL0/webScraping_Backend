from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field

class ScrapeRequest(BaseModel):
    url: str
    max_products: Optional[int] = 30
    headless: Optional[bool] = True
    driver_type: Optional[str] = "edge"

class ScrapeResponse(BaseModel):
    total_products: int
    total_images: int
    image_urls: List[HttpUrl]
    unique_images: int
    errors: List[Dict[str, Any]]
    elapsed_time: float
    error: Optional[str] = None

class BackgroundRemovalRequest(BaseModel):
    image_urls: List[HttpUrl]  # Changed from image_paths to validate URLs
    add_white_bg: bool = False

class BackgroundRemovalResponse(BaseModel):
    processed: int
    successful: int
    failed: int
    results: List[HttpUrl] = []  # Changed to HttpUrl for Cloudinary URLs

class GenerationRequest(BaseModel):
    prompt: str
    num_images: int = 1
    negative_prompt: Optional[str] = None
    size: str = "1024x1024"
    quality: str = "standard"

class GenerationResponse(BaseModel):
    status: str
    images: List[HttpUrl]  # Changed to HttpUrl assuming Cloudinary URLs
    total_generated: int



class MergeRequest(BaseModel):
    foreground_url: HttpUrl
    background_url: HttpUrl
    foreground_scale: float = Field(default=1.0, ge=0.5, le=2.0)
    background_scale: float = Field(default=1.0, ge=0.8, le=1.5)
    position_x: int = Field(default=0, ge=-200, le=200)
    position_y: int = Field(default=0, ge=-200, le=200)
    preview_mode: bool = Field(default=False)

class MergeResponse(BaseModel):
    status: str
    merged_image:  str # Allow HttpUrl or str for base64
class OptimizationRequest(BaseModel):
    image_urls: List[HttpUrl]  # Changed from image_paths
    upscale: bool = True
    denoise: bool = True
    enhance_lighting: bool = True

class OptimizationResponse(BaseModel):
    optimized: List[HttpUrl]  # Changed to HttpUrl for Cloudinary URLs

class ProcessProductRequest(BaseModel):
    platform: str
    product_url: HttpUrl
    max_products: Optional[int] = 1
    bg_add_white: bool = False
    gen_prompt: Optional[str] = None
    upload: bool = False

class TaskStatus(BaseModel):
    task_id: str
    status: str
    detail: Optional[Dict[str, Any]] = None

class Task(BaseModel):
    task_id: str
    task_type: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    input_urls: List[HttpUrl] = []  # Changed from input_paths
    output_urls: List[HttpUrl] = []  # Changed from output_paths
    metadata: Dict[str, Any] = {}
    platform: Optional[str] = None
    url: Optional[str] = None
    prompt: Optional[str] = None
    foreground_url: Optional[HttpUrl] = None  # Changed from foreground_path
    background_url: Optional[HttpUrl] = None  # Changed from background_path