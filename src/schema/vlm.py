"""Raw VLM observation schema — output of the VLM semantic analysis stage (s06)."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class VLMSegmentStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RawVLMObservation(BaseModel):
    """Raw VLM interpretation of a candidate interaction segment.
    
    This schema represents the model's textual beliefs. It does NOT normalize
    the action into the ActionType vocabulary. Missing/unknown fields do NOT
    constitute a FAILED status if the model successfully processed the segment.
    """

    observation_id: str
    segment_id: str
    status: VLMSegmentStatus
    error_reason: str | None = None
    
    # ── Provenance ──────────────────────────────────────────────────────────
    backend: str = "UNKNOWN"
    model_name: str = "UNKNOWN"
    prompt_version: str = "v1"
    segment_start_sec: float = Field(ge=0.0)
    segment_end_sec: float = Field(ge=0.0)
    
    # ── Raw VLM Response Preservation ───────────────────────────────────────
    raw_response: str | None = None  # the literal string returned by the VLM API
    
    # ── Semantic Fields (Populated on SUCCESS) ──────────────────────────────
    actor: str | None = None
    active_hand: str | None = None
    objects: list[str] = Field(default_factory=list)
    raw_action: str | None = None
    
    # Absolute video timestamps (Stage 06 converts relative VLM offsets to absolute)
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    
    state_change: str | None = None
    visible_facts: str | None = None
    inference: str | None = None
    uncertainty: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    
    @model_validator(mode="before")
    @classmethod
    def check_required_keys_on_success(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("status") == VLMSegmentStatus.SUCCESS:
            required_keys = {
                "actor", "active_hand", "objects", "raw_action",
                "start_time_sec", "end_time_sec", "state_change",
                "visible_facts", "inference", "uncertainty", "confidence"
            }
            missing = required_keys - data.keys()
            if missing:
                raise ValueError(f"Missing required keys in VLM JSON: {missing}")
        return data
        
    @model_validator(mode="after")
    def validate_timestamps(self) -> Any:
        """Ensure absolute timestamps fall within the segment bounds if provided."""
        # Only validate timestamps if it's a SUCCESS and both are present
        if self.status != VLMSegmentStatus.SUCCESS:
            return self
            
        if self.start_time_sec is not None and self.end_time_sec is not None and self.start_time_sec > self.end_time_sec:
                raise ValueError(
                    f"start_time_sec ({self.start_time_sec}) > end_time_sec ({self.end_time_sec})"
                )
                
        # Validate against segment bounds
        if self.start_time_sec is not None and not (self.segment_start_sec <= self.start_time_sec <= self.segment_end_sec):
                raise ValueError(
                    f"start_time_sec ({self.start_time_sec}) is outside segment "
                    f"[{self.segment_start_sec}, {self.segment_end_sec}]"
                )
                
        if self.end_time_sec is not None and not (self.segment_start_sec <= self.end_time_sec <= self.segment_end_sec):
                raise ValueError(
                    f"end_time_sec ({self.end_time_sec}) is outside segment "
                    f"[{self.segment_start_sec}, {self.segment_end_sec}]"
                )
                
        return self
