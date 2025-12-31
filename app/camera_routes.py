# app/camera_routes.py
from __future__ import annotations

import time
from typing import Iterator

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

from .camera import camera

from .live_inference import live_detector
import numpy as np
from PIL import Image
import io


router = APIRouter(prefix="/camera", tags=["camera"])


def mjpeg_generator() -> Iterator[bytes]:
    boundary = b"--frame"
    while True:
        jpg = camera.get_jpeg(quality=80)
        if jpg is None:
            # If camera not ready yet, wait a bit
            time.sleep(0.1)
            continue

        yield boundary + b"\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8")
        yield jpg + b"\r\n"
        # Small sleep to reduce CPU; actual FPS controlled by camera thread
        time.sleep(0.02)

def rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int = 80) -> bytes:
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()

@router.get("/mjpeg")
def mjpeg():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/snapshot")
def snapshot():
    jpg = camera.get_jpeg(quality=90)
    if jpg is None:
        return Response(content=b"Camera not ready", status_code=503, media_type="text/plain")
    return Response(content=jpg, media_type="image/jpeg")

@router.get("/mjpeg_detect")
def mjpeg_detect():
    def gen():
        boundary = b"--frame"
        while True:
            frame = live_detector.last_annotated_rgb
            if frame is None:
                time.sleep(0.1)
                continue

            jpg = rgb_to_jpeg_bytes(frame, quality=80)
            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpg)}\r\n\r\n".encode()
            yield jpg + b"\r\n"
            time.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
