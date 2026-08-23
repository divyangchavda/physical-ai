"""Track schema — output of the object tracking stage (s04)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.schema.detection import BoundingBox


class TrackPoint(BaseModel):
    """One observation of a tracked object at a specific frame."""

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    bbox: BoundingBox
    detection_confidence: float = Field(ge=0.0, le=1.0)
    tracking_confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class Track(BaseModel):
    """A single object tracked across multiple frames."""

    track_id: int = Field(ge=0)
    class_name: str
    class_id: int = Field(ge=0, default=0)
    points: list[TrackPoint] = Field(default_factory=list)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    source: str  # e.g. "bytetrack", "stub"
    is_estimated: bool = True
