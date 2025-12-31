# app/live_inference.py
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, List

import numpy as np
import cv2

from .camera import camera
from .inference import yolo, Detection


class LiveDetector:
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.last_annotated_rgb: Optional[np.ndarray] = None
        self.last_detections: List[dict] = []
        self.last_ts: float = 0.0

        # Tuning knobs (later can be controlled via Settings)
        self.infer_every_n_frames = 2  # run YOLO every N frames to keep Pi fast
        self.jpeg_quality = 80
        self.conf = 0.25

    def start(self, fps: int = 10) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(target=self._loop, args=(fps,), daemon=True)
        self._thread.start()
        print("[LiveDetector] Started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        print("[LiveDetector] Stopped")

    def _loop(self, fps: int) -> None:
        delay = 1.0 / max(1, fps)
        frame_count = 0

        while True:
            with self._lock:
                if not self._running:
                    break

            frame_rgb = camera._last_frame_rgb  # already RGB in our camera.py
            if frame_rgb is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count % self.infer_every_n_frames != 0:
                time.sleep(delay)
                continue

            # Convert numpy RGB -> PIL inside yolo.predict_pil
            try:
                from PIL import Image
                img = Image.fromarray(frame_rgb, mode="RGB")

                detections, annotated_pil = yolo.predict_pil(img, conf=self.conf)

                annotated_rgb = np.array(annotated_pil)  # RGB
                with self._lock:
                    self.last_annotated_rgb = annotated_rgb
                    self.last_detections = [d.__dict__ for d in detections]
                    self.last_ts = time.time()

            except Exception:
                # keep running even if one inference fails
                pass

            time.sleep(delay)


live_detector = LiveDetector()
