"""Episode schema — the root output object assembled by s11_episode."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Basic metadata about the input video file."""

    file_path: str
    duration_sec: float = Field(ge=0.0)
    fps: float = Field(gt=0.0)
    frame_count: int = Field(ge=0, default=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    codec: str = ""
    file_size_bytes: int = Field(ge=0)
    # Non-empty when metadata could be read but reliability is uncertain
    # (e.g. frame_count from container header vs. scan-verified count).
    # Never fabricated — only set when there is genuine evidence of a discrepancy.
    metadata_warnings: list[str] = Field(default_factory=list)


class PipelineStageStatus(BaseModel):
    """Execution status of a single pipeline stage."""

    stage: str
    status: Literal["OK", "WARNING", "SKIPPED", "ERROR"]
    message: str = ""
    duration_sec: float | None = None


class PhysicalEpisode(BaseModel):
    """Root data structure describing one processed video episode.

    Summary counts from all pipeline stages.
    Detailed data lives in per-stage JSON output files.

    Counts are 0 for stages that were SKIPPED — never fabricated.
    """

    episode_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    pipeline_version: str = "0.1.0"
    video_metadata: VideoMetadata | None = None
    stage_statuses: list[PipelineStageStatus] = Field(default_factory=list)

    # Summary counts — always accurate; 0 means stage was skipped or found nothing
    n_frames_sampled: int = Field(ge=0, default=0)
    n_frames_with_detections: int = Field(ge=0, default=0)
    n_detections: int = Field(ge=0, default=0)
    n_tracks: int = Field(ge=0, default=0)
    n_candidate_segments: int = Field(ge=0, default=0)
    n_events: int = Field(ge=0, default=0)
    n_state_transitions: int = Field(ge=0, default=0)
    n_trajectories: int = Field(ge=0, default=0)

    notes: list[str] = Field(default_factory=list)


class InteractionEpisode(BaseModel):
    episode_id: str
    event_ids: list[str]
    start_sec: float
    end_sec: float
    timing_precision: Literal["EXACT", "SEGMENT", "MIXED"]
    actor_track_ids: list[str] = Field(default_factory=list)
    object_track_ids: list[str] = Field(default_factory=list)
    graph_edge_ids: list[str] = Field(default_factory=list)
    state_transition_ids: list[str] = Field(default_factory=list)
    quality_score_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    episode_quality_tier: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    source: str = ""
    is_estimated: bool = False
    assembly_reasons: list[str] = Field(default_factory=list)
