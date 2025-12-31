# app/api.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import time

import io
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .inference import yolo, pil_to_base64_jpeg


router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Home", "active": "home"},
    )


@router.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "title": "Upload", "active": "upload"},
    )


@router.get("/live", response_class=HTMLResponse)
async def live(request: Request):
    return templates.TemplateResponse(
        "live.html",
        {"request": request, "title": "Live", "active": "live", "ts": int(time.time())},
    )


@router.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse(
        "settings.html",
        {"request": request, "title": "Settings", "active": "settings"},
    )

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()

    try:
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)

    detections, annotated = yolo.predict_pil(img, conf=0.25)

    return {
        "filename": file.filename,
        "detections": [d.__dict__ for d in detections],
        "image_base64": pil_to_base64_jpeg(annotated, quality=85),
    }
