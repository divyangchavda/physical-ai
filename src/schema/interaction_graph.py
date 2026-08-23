"""Interaction Graph schema — output of the interaction graph extraction stage (s09)."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.schema.event import ActionType


class NodeRole(str, Enum):
    PERSON = "PERSON"
    OBJECT = "OBJECT"


class GraphNode(BaseModel):
    """A physical entity in the scene."""

    node_id: str
    role: NodeRole
    track_id: int | None = Field(default=None, ge=0)
    semantic_label: str | None = None


class GraphEdge(BaseModel):
    """A directed interaction from a source node (actor) to a target node (object)."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    action: ActionType
    
    actor_resolution: Literal["RESOLVED", "UNRESOLVED", "AMBIGUOUS"] = "UNRESOLVED"
    object_resolution: Literal["RESOLVED", "UNRESOLVED", "AMBIGUOUS"] = "UNRESOLVED"
    
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    timing_precision: Literal["EXACT", "SEGMENT"] = "SEGMENT"
    
    event_id: str
    observation_id: str | None = None
    segment_id: str | None = None
    state_transition_ids: list[str] = Field(default_factory=list)
    
    confidence: float = Field(ge=0.0, le=1.0)
