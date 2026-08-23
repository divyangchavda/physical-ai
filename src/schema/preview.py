"""Preview schema — output of the preview stage (s14)."""
from __future__ import annotations

from pydantic import BaseModel


class PreviewCounts(BaseModel):
    total_events: int = 0
    total_episodes: int = 0
    total_state_transitions: int = 0
    total_graph_nodes: int = 0
    total_graph_edges: int = 0
    total_trajectories: int = 0

class QualityDistribution(BaseModel):
    high: int = 0
    medium: int = 0
    low: int = 0
    rejected: int = 0

class EvaluationSummary(BaseModel):
    dataset_health: str
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

class EpisodeSummary(BaseModel):
    episode_id: str
    start_sec: float
    end_sec: float
    timing_precision: str
    quality_tier: str | None
    event_ids: list[str]
    human_readable_events: list[str]

class PreviewReport(BaseModel):
    dataset_health: str
    counts: PreviewCounts
    quality_distribution: QualityDistribution
    evaluation_summary: EvaluationSummary
    timeline: list[EpisodeSummary]
