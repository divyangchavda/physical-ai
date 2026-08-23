"""Trajectory schema — output of the trajectory extraction stage (s09).

CRITICAL CONSTRAINT:
    All trajectories produced by this pipeline are 2-D image-space trajectories.
    They MUST NOT be interpreted or presented as real 3-D physical trajectories.

    The ``coordinate_space`` field is typed as ``Literal["2D_IMAGE_PIXELS"]``
    and cannot be set to any other value. This constraint is enforced by Pydantic
    and cannot be bypassed.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrajectoryPoint2D(BaseModel):
    """One point in a 2-D image-space trajectory."""

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    x_px: float  # image pixel coordinate (horizontal, left = 0)
    y_px: float  # image pixel coordinate (vertical, top = 0)
    confidence: float = Field(ge=0.0, le=1.0)


class Trajectory2D(BaseModel):
    """2-D image-space trajectory of a tracked object.

    ``coordinate_space`` is locked to ``"2D_IMAGE_PIXELS"`` and cannot be
    changed. This structurally prevents the system from ever claiming a
    3-D physical trajectory.

    ``total_distance_px`` and ``mean_speed_px_per_sec`` are in pixels,
    not physical units. Do not convert to metres without a valid calibration.
    """

    trajectory_id: str
    track_id: int = Field(ge=0)
    # IMMUTABLE: prevents any code from claiming this is a 3-D trajectory
    coordinate_space: Literal["2D_IMAGE_PIXELS"] = "2D_IMAGE_PIXELS"
    points: list[TrajectoryPoint2D] = Field(default_factory=list)
    source: str
    is_estimated: bool = True
    # Summary statistics (pixels only — not physical units)
    total_distance_px: float | None = None
    mean_speed_px_per_sec: float | None = None
