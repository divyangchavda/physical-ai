"""YOLO-based object detector implementation.

Wraps Ultralytics YOLO models (e.g. yolov8n).
- Loads Ultralytics lazily to keep the pipeline independent.
- Runs entirely on CPU or CUDA depending on config.
- Translates YOLO outputs into the strict pipeline schemas.
- Exposes clean unload behavior to manage memory.
"""
from __future__ import annotations

import gc
import uuid

import numpy as np

from src.interfaces.detector import ObjectDetector
from src.logging_utils import get_logger
from src.schema.detection import BoundingBox, Detection

logger = get_logger(__name__)


class YOLODetector(ObjectDetector):
    """YOLO-based object detector.

    ``model_name`` is configurable; defaults to "yolov8n".
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence: float = 0.35,
        nms_iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self._model_name = model_name
        self._confidence = confidence
        self._nms_iou = nms_iou
        self._device = device
        self._model = None

    def load(self) -> None:
        """Lazy-load the YOLO model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics is not installed. "
                "Install it with: pip install ultralytics"
            ) from e

        logger.info("Loading YOLO model: %s.pt on device: %s", self._model_name, self._device)
        self._model = YOLO(f"{self._model_name}.pt")

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        """Run detection on a single BGR uint8 frame."""
        if self._model is None:
            raise RuntimeError("YOLODetector must be load()ed before detect()")

        # Run inference
        results = self._model(
            frame,
            verbose=False,
            device=self._device,
            conf=self._confidence,
            iou=self._nms_iou,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        names = result.names

        for box in boxes:
            # ultralytics returns tensors; convert to floats/ints
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names[class_id]

            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            det = Detection(
                detection_id=uuid.uuid4().hex[:12],
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                class_id=class_id,
                class_name=class_name,
                confidence=conf,
                bbox=bbox,
                source=self.model_name,
            )
            detections.append(det)

        return detections

    def unload(self) -> None:
        """Release model weights and free memory."""
        if self._model is not None:
            logger.info("Unloading YOLO model to free memory.")
            del self._model
            self._model = None
            
            # Force garbage collection
            gc.collect()

            # If CUDA was used, free the PyTorch cache
            if self._device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    @property
    def model_name(self) -> str:
        return f"yolo:{self._model_name}"
