"""ObjectTracker abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema.detection import Detection
from src.schema.track import Track


class ObjectTracker(ABC):
    """Interface for all object tracking backends.

    Contract:
      - ``update()`` is called once per frame with that frame's detections.
      - ``update()`` returns the current set of ACTIVE tracks (may be empty).
      - ``reset()`` clears all state between separate video files.
      - Track IDs must be consistent across calls (same object → same ID).
      - Never fabricate track points.
    """

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Track]:
        """Update tracker with new detections; return currently active tracks.

        Args:
            detections: detections from the current frame.
            frame_index: 0-based frame index (for timestamp bookkeeping).

        Returns:
            List of currently active :class:`~src.schema.track.Track` objects.
            Empty list if no objects are being tracked.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all tracker state. Call between separate video files."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable identifier used in logs and data provenance."""
