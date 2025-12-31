# app/camera.py
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import cv2


class CameraManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._use_picamera2 = False
        self._picam2 = None
        self._cap = None

        # store always as RGB for PIL encoding
        self._last_frame_rgb: Optional[np.ndarray] = None

        # IMPORTANT: if web view is blueish, turn this ON
        # (it swaps R<->B, fixing "wood looks blue")
        self.swap_rb: bool = True

    def start(self, width: int = 640, height: int = 480, fps: int = 20) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam2 = Picamera2()

            # Request a stable 3-channel format from libcamera
            config = self._picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={
                    "AeEnable": True,
                    "AwbEnable": True,
                    "AwbMode": 0,   # Auto
                    "AfMode": 2,    # Continuous
                    "AfSpeed": 1,   # Fast
                }
            )
            self._picam2.configure(config)
            self._picam2.start()
            self._use_picamera2 = True
            print("[Camera] Picamera2 started (RGB888). swap_rb =", self.swap_rb)

        except Exception as e:
            print(f"[Camera] Picamera2 failed -> OpenCV fallback. Reason: {e}")
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

            frame_rgb = self._read_frame_rgb()
            if frame_rgb is not None:
                self._last_frame_rgb = frame_rgb

            time.sleep(delay)

    def _read_frame_rgb(self) -> Optional[np.ndarray]:
        """
        Returns RGB uint8 frame (H,W,3).
        """
        try:
            if self._use_picamera2 and self._picam2 is not None:
                frame = self._picam2.capture_array()

                # If 4 channels appear, drop alpha/X channel
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]

                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)

                # Some Pi builds deliver swapped channels even when asking RGB888.
                # swap_rb=True fixes "blueish" output (wood->blue).
                if self.swap_rb:
                    frame = frame[:, :, ::-1]  # swap R<->B

                return frame  # now RGB

            if self._cap is not None:
                ok, bgr = self._cap.read()
                if not ok:
                    return None
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return rgb

        except Exception:
            return None

        return None

    def get_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """
        Encode RGB frame using PIL to avoid OpenCV channel assumptions.
        """
        frame_rgb = self._last_frame_rgb
        if frame_rgb is None:
            return None

        try:
            from PIL import Image
            import io

            img = Image.fromarray(frame_rgb, mode="RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(quality))
            return buf.getvalue()
        except Exception:
            return None


camera = CameraManager()
