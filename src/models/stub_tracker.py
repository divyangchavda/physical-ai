"""Stub object tracker — zero-dependency, used in tests and stub-mode runs.

Never fabricates track points. Always returns an empty list.
"""
from __future__ import annotations

from src.interfaces.tracker import ObjectTracker
from src.schema.detection import Detection
from src.schema.track import Track


class StubTracker(ObjectTracker):
    """Zero-dependency tracker stub.

    Used when ``stub_mode=True`` or in CI tests.
    Returns an empty track list — never fabricates track points.
    """

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Track]:
        return []  # honest: no tracks fabricated

    def reset(self) -> None:
        pass

    @property
    def backend_name(self) -> str:
        return "stub"
