"""Physical event schema — output of the physical event extraction stage (s07).

Design principles:
  - Raw predictions (action + confidence + source) are ALWAYS preserved.
  - The quality engine (s10_score) assigns review_status after the fact.
  - UNKNOWN must be used when evidence is genuinely insufficient —
    NOT as a confidence-threshold fallback.
  - is_estimated=True for all model predictions; False only for GT annotations.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Controlled vocabulary of physical manipulation actions (MVP v1).

    Rules:
      - Use UNKNOWN when the system cannot determine the action.
      - Do NOT assign another action when evidence is insufficient.
      - Do NOT force an action to avoid UNKNOWN.
    """

    GRASP = "GRASP"
    RELEASE = "RELEASE"
    PICK = "PICK"
    PLACE = "PLACE"
    MOVE = "MOVE"
    PUSH = "PUSH"
    PULL = "PULL"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    INSERT = "INSERT"
    REMOVE = "REMOVE"
    USE_TOOL = "USE_TOOL"
    TOUCH = "TOUCH"
    INSPECT = "INSPECT"
    UNKNOWN = "UNKNOWN"  # default; use when evidence is insufficient


class PhysicalEvent(BaseModel):
    """A detected physical interaction event.

    The quality engine (s10_score) assigns review_status based on thresholds.
    It does NOT modify or discard the raw action / confidence / source fields.
    Raw predictions are preserved for evaluation and failure analysis.
    """

    event_id: str
    segment_id: str | None = None  # originating CandidateSegment, if any
    observation_id: str | None = None  # originating RawVLMObservation, if any

    # ── Raw prediction fields (ALWAYS preserved) ───────────────────────────
    action: ActionType = ActionType.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source: str  # e.g. "vlm:stub", "rule_based", "manual_annotation"
    is_estimated: bool = True  # False only for ground-truth annotations

    # ── Participants ────────────────────────────────────────────────────────
    actor_track_id: int | None = None    # who performed the action
    object_track_id: int | None = None   # what was acted upon

    # ── Timing ─────────────────────────────────────────────────────────────
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)

    # ── Extra context ───────────────────────────────────────────────────────
    attributes: dict = Field(default_factory=dict)  # raw VLM output / extra fields

    # ── Quality engine decision (set by s10, never by the event extractor) ──
    review_status: Literal[
        "PENDING", "AUTO_ACCEPT", "HUMAN_REVIEW", "REJECT", "REPROCESS"
    ] = "PENDING"
