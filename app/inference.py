# app/inference.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import base64
import io

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2]


class YoloService:
    def __init__(self, weights_path: str = "models/yolov8n.pt"):
        # Load once
        self.model = YOLO(weights_path)

    def predict_pil(self, image: Image.Image, conf: float = 0.25) -> Tuple[List[Detection], Image.Image]:
        img = np.array(image.convert("RGB"))

        results = self.model.predict(img, conf=conf, verbose=False)
        r = results[0]
        names = r.names

        dets: List[Detection] = []
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                cls_id = int(b.cls[0].item())
                conf_val = float(b.conf[0].item())
                dets.append(
                    Detection(
                        class_id=cls_id,
                        class_name=str(names.get(cls_id, str(cls_id))),
                        confidence=conf_val,
                        box=[float(x) for x in xyxy],
                    )
                )

        # --- Color safety: some builds return plot() in BGR, some in RGB.
        # If your output looks blueish, set SWAP_RB_PLOT = False.
        SWAP_RB_PLOT = False  # <-- IMPORTANT: this fixes your blueish output

        annotated = r.plot()  # numpy image
        # Ensure 3 channels
        if annotated.ndim == 3 and annotated.shape[2] == 4:
            annotated = annotated[:, :, :3]

        # If plot() is BGR, swap to RGB; if plot() already RGB, do not swap.
        if SWAP_RB_PLOT:
            annotated = annotated[:, :, ::-1]

        annotated_pil = Image.fromarray(annotated.astype(np.uint8), mode="RGB")

        return dets, annotated_pil


def pil_to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Singleton instance (best practice)
yolo = YoloService("models/best.pt")
# yolo = YoloService("models/yolov8n.pt")
