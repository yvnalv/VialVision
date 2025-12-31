# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import api
from .camera_routes import router as camera_router
from .camera import camera

from .live_inference import live_detector

app = FastAPI(title="Raspberry Pi YOLO Demo")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api.router)
app.include_router(camera_router)


@app.on_event("startup")
def _startup():
    camera.start(width=640, height=480, fps=20)
    live_detector.start(fps=10)


@app.on_event("shutdown")
def _shutdown():
    live_detector.stop()
    camera.stop()
