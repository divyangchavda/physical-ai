"""Evaluation report schema — output of s12_evaluate."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StageEvaluation(BaseModel):
    """Evaluation result for a single pipeline stage."""

    stage: str
    status: Literal["OK", "SKIPPED", "ERROR", "WARNING"]
    message: str = ""
    metrics: dict = Field(default_factory=dict)


class IntegrityIssue(BaseModel):
    """Data integrity or referential consistency issue."""
    
    severity: Literal["INFO", "WARNING", "ERROR"]
    dimension: Literal[
        "REFERENTIAL_INTEGRITY",
        "TEMPORAL_CONSISTENCY",
        "STATE_CONSISTENCY",
        "GRAPH_CONSISTENCY",
        "EPISODE_CONSISTENCY",
        "QUALITY_CONSISTENCY"
    ]
    message: str
    reference_id: str


class EvaluationReport(BaseModel):
    """Pipeline-level quality and completeness evaluation."""

    episode_id: str
    overall_status: Literal["PASS", "FAIL", "PARTIAL", "SKIPPED"]
    stage_evaluations: list[StageEvaluation] = Field(default_factory=list)
    dataset_health: Literal["HEALTHY", "WARNING", "CRITICAL"] = "HEALTHY"
    integrity_issues: list[IntegrityIssue] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
