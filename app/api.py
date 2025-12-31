# app/api.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import time

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
