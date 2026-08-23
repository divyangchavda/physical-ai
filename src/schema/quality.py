"""Quality scoring schema — output of the quality scoring stage (s11)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ComponentScores(BaseModel):
    """Component scores used to calculate the composite quality score."""
    action_certainty: float = Field(ge=0.0, le=1.0)
    actor_resolution: float = Field(ge=0.0, le=1.0)
    object_resolution: float = Field(ge=0.0, le=1.0)
    state_evidence: float = Field(ge=0.0, le=1.0)
    timing_precision: float = Field(ge=0.0, le=1.0)
    trajectory_support: float = Field(ge=0.0, le=1.0)


class QualityProvenance(BaseModel):
    """IDs of the evidence used to compute the quality score."""
    graph_edge_id: str | None = None
    state_transition_ids: list[str] = Field(default_factory=list)
    trajectory_ids: list[str] = Field(default_factory=list)


class EventQualityScore(BaseModel):
    """Quality score and tier for a single PhysicalEvent."""
    event_id: str
    vlm_confidence: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    quality_tier: Literal["AUTO_ACCEPT", "HUMAN_REVIEW", "REJECT"]
    components: ComponentScores
    provenance: QualityProvenance
    reasons: list[str] = Field(default_factory=list)
