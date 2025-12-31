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
    - Prefer Picamera2 on Raspberry Pi (Camera Module 3 / libcamera)
    - Optional fallback to OpenCV VideoCapture (USB webcam)

    This version standardizes all frames to RGB and encodes JPEG via PIL
    to avoid channel-order surprises (brown -> blue issues).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_frame_rgb: Optional[np.ndarray] = None  # ALWAYS RGB

        self._use_picamera2 = False
        self._picam2 = None
        self._cap = None

    def start(self, width: int = 640, height: int = 480, fps: int = 20) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        # Prefer Picamera2 (best for Pi Camera Module 3)
        try:
            from picamera2 import Picamera2  # type: ignore

            self._picam2 = Picamera2()

            # Force a known-good format for web display pipeline: RGB888
            config = self._picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self._picam2.configure(config)

            # Match "rpicam-hello --autofocus-mode continuous" behavior
            # Enable continuous AF + keep auto exposure & auto white balance
            try:
                # Not all builds expose the same control names, so we guard each.
                self._picam2.set_controls(
                    {
                        "AfMode": 2,  # 2 = Continuous (libcamera enum)
                        # AE/AWB are typically automatic by default, but these controls help stability
                        # Keep these if your build supports them; otherwise they are ignored by exception.
                    }
                )
            except Exception:
                pass

            self._picam2.start()
            self._use_picamera2 = True
            print("[Camera] Using Picamera2 (RGB888, continuous AF)")
        except Exception as e:
            # Fallback to OpenCV (USB webcam)
            print(f"[Camera] Picamera2 failed -> fallback OpenCV. Reason: {e}")
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
        Returns an RGB uint8 numpy array (H,W,3).
        """
        try:
            if self._use_picamera2 and self._picam2 is not None:
                frame = self._picam2.capture_array()  # expected RGB888

                # Safety: if camera gives 4 channels (XRGB/RGBA), drop alpha/X
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]

                # Ensure uint8
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)

                return frame

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
        Encode last RGB frame into JPEG bytes using PIL.
        This avoids OpenCV BGR assumptions that can distort colors.
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
