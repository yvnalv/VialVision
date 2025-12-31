# app/camera.py
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import cv2


class CameraManager:
    """
    Camera abstraction:
    - Prefer Picamera2 on Raspberry Pi (Camera Module 3)
    - Fallback to OpenCV VideoCapture (USB webcam)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame: Optional[np.ndarray] = None

        self._use_picamera2 = False
        self._picam2 = None
        self._cap = None

    def start(self, width: int = 640, height: int = 480, fps: int = 20) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        # Try Picamera2 (best for Pi Camera Module 3)
        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam2 = Picamera2()
            config = self._picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self._picam2.configure(config)
            self._picam2.start()
            self._use_picamera2 = True
        except Exception:
            # Fallback to OpenCV
            self._use_picamera2 = False
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)

        self._thread = threading.Thread(target=self._loop, args=(fps,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False

        if self._thread:
            self._thread.join(timeout=1.0)

        if self._use_picamera2 and self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            self._picam2 = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _loop(self, fps: int) -> None:
        delay = 1.0 / max(1, fps)
        while True:
            with self._lock:
                if not self._running:
                    break

            frame = self._read_frame()
            if frame is not None:
                # store BGR for encoding convenience
                self._last_frame = frame

            time.sleep(delay)

    def _read_frame(self) -> Optional[np.ndarray]:
        try:
            if self._use_picamera2 and self._picam2 is not None:
                rgb = self._picam2.capture_array()  # RGB
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return bgr
            if self._cap is not None:
                ok, frame = self._cap.read()
                return frame if ok else None
        except Exception:
            return None
        return None

    def get_jpeg(self, quality: int = 80) -> Optional[bytes]:
        frame = self._last_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buf.tobytes()


camera = CameraManager()
