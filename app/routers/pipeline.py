import threading
import uuid
from typing import Dict

from fastapi import APIRouter

from app.schemas import ProcessProductRequest, TaskStatus
from app.services import scrape_amazon, scrape_ebay, scrape_shopify, remove_background, generate_background, optimize_images

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

_TASKS: Dict[str, TaskStatus] = {}


def _set_status(task_id: str, status: str, detail: Dict | None = None):
    _TASKS[task_id] = TaskStatus(task_id=task_id, status=status, detail=detail or {})


def _run_pipeline(task_id: str, req: ProcessProductRequest):
    try:
        _set_status(task_id, "scraping")
        if req.platform.lower() == "amazon":
            stats = scrape_amazon(str(req.product_url), max_products=req.max_products)
        elif req.platform.lower() == "ebay":
            stats = scrape_ebay(str(req.product_url), max_products=req.max_products)
        else:
            stats = scrape_shopify(str(req.product_url), max_products=req.max_products)

        _set_status(task_id, "background_removal", {"scrape": stats})
        # In a real pipeline, collect image paths from scrape result/output folder
        image_paths = []
        bg = remove_background(image_paths, add_white_bg=req.bg_add_white)

        gen_images = []
        if req.gen_prompt:
            _set_status(task_id, "generation", {"bg": bg})
            gen_images = generate_background(req.gen_prompt, None, 1)

        _set_status(task_id, "optimization", {"generated": gen_images})
        optimized = optimize_images(gen_images or image_paths)

        # TODO optional: upload, s3
        _set_status(task_id, "completed", {"optimized": optimized, "stats": stats})
    except Exception as e:
        _set_status(task_id, "error", {"message": str(e)})


@router.post("/process-product", response_model=TaskStatus)
def process_product(req: ProcessProductRequest):
    task_id = str(uuid.uuid4())
    _set_status(task_id, "queued")
    thread = threading.Thread(target=_run_pipeline, args=(task_id, req), daemon=True, name=f"pipeline-{task_id}")
    thread.start()
    return _TASKS[task_id]


@router.get("/tasks/{task_id}", response_model=TaskStatus)
def task_status(task_id: str):
    return _TASKS.get(task_id) or TaskStatus(task_id=task_id, status="unknown", detail={})

