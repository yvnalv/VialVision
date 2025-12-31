# app/camera_routes.py
from __future__ import annotations

import time
from typing import Iterator

from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

from .camera import camera

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
