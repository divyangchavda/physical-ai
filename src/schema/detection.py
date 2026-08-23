"""Detection schema — output of the object detection stage (s03).

BoundingBox is also used by track.py; import from here.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in image pixel coordinates (top-left origin)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _validate_order(self) -> BoundingBox:
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be > x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be > y1 ({self.y1})")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        """Center x coordinate."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Center y coordinate."""
        return (self.y1 + self.y2) / 2.0


class Detection(BaseModel):
    """A single object detection in one video frame."""

    detection_id: str
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    bbox: BoundingBox
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # e.g. "yolov8n", "stub" — always populated for provenance
    is_estimated: bool = True  # False only for ground-truth annotations


class DetectionFrame(BaseModel):
    """All detections for a single sampled frame."""

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    detections: list[Detection] = Field(default_factory=list)
    status: Literal["OK", "SKIPPED", "ERROR"] = "OK"
    message: str = ""
