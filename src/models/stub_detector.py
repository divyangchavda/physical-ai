"""Stub object detector — zero-dependency, used in tests and stub-mode runs.

Never fabricates detections. Always returns an empty list.
This is the correct stub behaviour: honest absence, not fake presence.
"""
from __future__ import annotations

import numpy as np

from src.interfaces.detector import ObjectDetector
from src.schema.detection import Detection


class StubDetector(ObjectDetector):
    """Zero-dependency detector stub.

    Used when ``stub_mode=True`` or in CI tests.
    Returns an empty detection list — never fabricates detections.
    """

    def load(self) -> None:
        pass  # nothing to load

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        return []  # honest: no detections, not fabricated ones

    def unload(self) -> None:
        pass  # nothing to release

    @property
    def model_name(self) -> str:
        return "stub"
