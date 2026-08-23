"""Stage 11 — Quality and confidence scoring.

Assigns review_status to each PhysicalEvent based on quality engine thresholds.
Raw predictions (action + confidence + source) are NEVER modified.

Quality engine decisions:
  confidence >= auto_accept_threshold  → AUTO_ACCEPT
  confidence >= human_review_threshold → HUMAN_REVIEW
  confidence <  human_review_threshold → REJECT

SKIPPED in stub mode or when no events are available.

Output: modifies ctx.events[*].review_status in place
        (events.json is re-written by this stage after scoring)
        writes quality_scores.json
"""
from __future__ import annotations

import json
import time

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.models.quality_scorer import QualityScorer
from src.schema.episode import PipelineStageStatus

logger = get_logger(__name__)
STAGE = "s11_score"


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if ctx.config.stub_mode:
        logger.info("[%s] stub_mode=True — SKIPPED", STAGE)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: quality scoring skipped",
            duration_sec=time.monotonic() - t0,
        )

    if not ctx.events:
        logger.info("[%s] No events to score — SKIPPED", STAGE)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="no events available to score",
            duration_sec=time.monotonic() - t0,
        )

    scores = QualityScorer.score_events(
        events=ctx.events,
        edges=ctx.graph_edges,
        nodes=ctx.graph_nodes,
        transitions=ctx.state_transitions,
        trajectories=ctx.trajectories,
        config=ctx.config.event,
    )
    
    ctx.quality_scores = scores
    
    score_by_event = {s.event_id: s for s in scores}
    for event in ctx.events:
        if event.event_id in score_by_event:
            event.review_status = score_by_event[event.event_id].quality_tier

    events_path = ctx.output_dir / "events.json"
    events_path.write_text(json.dumps([e.model_dump() for e in ctx.events], indent=2))
    
    scores_path = ctx.output_dir / "quality_scores.json"
    scores_path.write_text(json.dumps([s.model_dump() for s in scores], indent=2))

    return PipelineStageStatus(
        stage=STAGE, status="OK",
        message=f"Scored {len(scores)} events",
        duration_sec=time.monotonic() - t0,
    )
