"""ObjectDetector abstract base class.

The pipeline imports ONLY this ABC. Concrete implementations live in
``src/models/``. Swapping detectors never changes pipeline code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.schema.detection import Detection


class ObjectDetector(ABC):
    """Interface for all object detection backends.

    Contract:
      - ``load()`` must be called once before the first ``detect()``.
      - ``detect()`` returns an empty list (not None) when nothing is found.
      - ``detect()`` never fabricates detections.
      - ``unload()`` releases GPU/CPU memory (safe to call multiple times).
      - ``model_name`` is used for data provenance; set it on every Detection.
    """

    @abstractmethod
    def load(self) -> None:
        """Lazy-load model weights. Called once before the first ``detect()``."""

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        """Run detection on a single BGR uint8 frame.

        Args:
            frame: HxWxC uint8 BGR image (OpenCV convention).
            frame_index: 0-based index of this frame in the source video.
            timestamp_sec: wall-clock position of this frame in seconds.

        Returns:
            List of :class:`~src.schema.detection.Detection` objects.
            Empty list if nothing detected — never None.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free VRAM / RAM."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable identifier used in logs and data provenance."""
