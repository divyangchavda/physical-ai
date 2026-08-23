"""Object state and state transition schemas — output of s08_states."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ObjectState(BaseModel):
    """The inferred state of a tracked object at a point in time."""

    track_id: int | None = Field(default=None, ge=0)
    semantic_label: str | None = None
    identity_resolution: Literal["RESOLVED", "UNRESOLVED", "AMBIGUOUS"] = "UNRESOLVED"
    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    state_label: str  # e.g. "IN_HAND", "ON_SURFACE", "OPEN", "CLOSED", "UNKNOWN"
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # e.g. "rule_based", "vlm"
    is_estimated: bool = True  # False only for ground-truth annotations


class StateTransition(BaseModel):
    """A change in a tracked object's inferred state."""

    transition_id: str
    track_id: int | None = Field(default=None, ge=0)
    semantic_label: str | None = None
    identity_resolution: Literal["RESOLVED", "UNRESOLVED", "AMBIGUOUS"] = "UNRESOLVED"
    from_state: str
    to_state: str
    trigger_event_id: str | None = None  # PhysicalEvent that caused this transition
    observation_id: str | None = None
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    timing_precision: Literal["EXACT", "SEGMENT"] = "SEGMENT"
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    is_estimated: bool = True
    evidence: dict = Field(default_factory=dict)
