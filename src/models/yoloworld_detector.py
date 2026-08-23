"""YOLO-World open-vocabulary object detector implementation.

Wraps Ultralytics YOLO-World models for custom vocabulary detection.
- Loads Ultralytics YOLOWorld lazily to keep the pipeline independent.
- Supports runtime text-based vocabulary configuration.
- Produces the EXACT same Detection schema as YOLOv8n.
- Runs entirely on CPU or CUDA depending on config.
"""
from __future__ import annotations

import gc
import sys
import uuid

import numpy as np

# Workaround: Ultralytics expects 'clip' module but we have 'open_clip'
# Import our compatibility shim before ultralytics loads
try:
    import clip  # noqa: F401
except ImportError:
    import clip_compat
    sys.modules['clip'] = clip_compat

from src.interfaces.detector import ObjectDetector
from src.logging_utils import get_logger
from src.schema.detection import BoundingBox, Detection

logger = get_logger(__name__)


class YOLOWorldDetector(ObjectDetector):
    """YOLO-World open-vocabulary object detector.

    Uses text prompts to define detection vocabulary at runtime.
    Compatible with YOLOv8n Detection schema output.
    """

    def __init__(
        self,
        model_name: str = "yolov8s-world",
        vocabulary: list[str] | None = None,
        confidence: float = 0.35,
        nms_iou: float = 0.45,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_name: YOLO-World model variant (e.g., "yolov8s-world", "yolov8m-world")
            vocabulary: List of text class names for open-vocabulary detection
            confidence: Minimum detection confidence threshold
            nms_iou: Non-maximum suppression IoU threshold
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self._model_name = model_name
        self._vocabulary = vocabulary or []
        self._confidence = confidence
        self._nms_iou = nms_iou
        self._device = device
        self._model = None

    def load(self) -> None:
        """Lazy-load the YOLO-World model and set vocabulary."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLOWorld
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics with YOLO-World support is not installed. "
                "Install with: pip install ultralytics>=8.4.0"
            ) from e

        if not self._vocabulary:
            raise RuntimeError(
                "YOLOWorldDetector requires a vocabulary. "
                "Set detector.vocabulary in config."
            )

        logger.info(
            "Loading YOLO-World model: %s.pt on device: %s with vocabulary: %s",
            self._model_name, self._device, self._vocabulary
        )
        
        # Load model
        self._model = YOLOWorld(f"{self._model_name}.pt")
        
        # Set custom vocabulary
        self._model.set_classes(self._vocabulary)
        
        logger.info(
            "YOLO-World loaded. Vocabulary classes: %d", len(self._vocabulary)
        )

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        """Run detection on a single BGR uint8 frame.
        
        Returns Detection objects matching YOLOv8n schema exactly.
        """
        if self._model is None:
            raise RuntimeError("YOLOWorldDetector must be load()ed before detect()")

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
            logger.info("Unloading YOLO-World model to free memory.")
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
        return f"yolo-world:{self._model_name}"
