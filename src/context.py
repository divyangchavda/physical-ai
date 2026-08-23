"""Pipeline execution context — shared mutable state threaded through stages.

Each stage reads required inputs from this object and writes outputs back.
Stages must not modify fields they do not own.

Pixel data is NOT stored in the context. Stages that need pixel data
re-read the source video directly, keeping memory usage bounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import PipelineConfig
from src.schema.detection import DetectionFrame
from src.schema.episode import (
    InteractionEpisode,
    PhysicalEpisode,
    PipelineStageStatus,
    VideoMetadata,
)
from src.schema.evaluation import EvaluationReport
from src.schema.event import PhysicalEvent
from src.schema.interaction_graph import GraphEdge, GraphNode
from src.schema.quality import EventQualityScore
from src.schema.segment import CandidateSegment
from src.schema.state import ObjectState, StateTransition
from src.schema.track import Track
from src.schema.trajectory import Trajectory2D
from src.schema.vlm import RawVLMObservation


@dataclass
class SampledFrameInfo:
    """Metadata for one sampled video frame (no pixel data stored).

    Pixel data is never stored here. Later stages (e.g. s03_detect) use
    ``video_path`` + ``frame_index`` to seek-and-read the exact frame.
    """

    frame_index: int        # 0-based index in the source video stream
    timestamp_sec: float    # wall-clock position in seconds (frame_index / fps)
    video_path: Path = field(default_factory=Path)  # source video for this frame


@dataclass
class PipelineContext:
    """Shared state passed between all pipeline stages.

    Stages are responsible for populating their designated fields.
    ``record_stage()`` appends execution status for episode assembly.
    """

    config: PipelineConfig
    video_path: Path
    output_dir: Path

    # ── Stage outputs (populated sequentially as the pipeline runs) ─────────
    video_metadata: VideoMetadata | None = None
    sampled_frame_infos: list[SampledFrameInfo] = field(default_factory=list)
    detection_frames: list[DetectionFrame] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    candidate_segments: list[CandidateSegment] = field(default_factory=list)
    vlm_observations: list[RawVLMObservation] = field(default_factory=list)
    events: list[PhysicalEvent] = field(default_factory=list)
    object_states: list[ObjectState] = field(default_factory=list)
    state_transitions: list[StateTransition] = field(default_factory=list)

    # Stage 09: Interaction Graph
    graph_nodes: list[GraphNode] = field(default_factory=list)
    graph_edges: list[GraphEdge] = field(default_factory=list)

    trajectories: list[Trajectory2D] = field(default_factory=list)
    quality_scores: list[EventQualityScore] = field(default_factory=list)
    episodes: list[InteractionEpisode] = field(default_factory=list)
    episode: PhysicalEpisode | None = None
    evaluation: EvaluationReport | None = None

    # ── Stage execution record ───────────────────────────────────────────────
    stage_statuses: list[PipelineStageStatus] = field(default_factory=list)

    def record_stage(self, status: PipelineStageStatus) -> None:
        """Append a completed stage status to the execution record."""
        self.stage_statuses.append(status)
