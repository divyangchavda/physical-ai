"""Stage 07: Physical Event Extraction and Action Normalization."""

import json
import logging
import time

from src.context import PipelineContext, PipelineStageStatus
from src.models.action_normalizer import ActionNormalizer
from src.schema.vlm import VLMSegmentStatus

STAGE = "s07_event_extraction"
logger = logging.getLogger(__name__)


def _write_output(ctx: PipelineContext, status: str) -> None:
    """Serialize the extracted events to events.json."""
    out_file = ctx.config.output_dir / "events.json"
    
    output_data = {
        "metadata": {
            "video": str(ctx.video_path),
            "stage": STAGE,
            "status": status,
        },
        "events": [evt.model_dump(mode="json") for evt in ctx.events]
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def run(ctx: PipelineContext) -> PipelineStageStatus:
    """Run the physical event extraction stage.
    
    This stage deterministically normalizes raw VLM observations into
    canonical PhysicalEvent objects using a rule-based engine.
    """
    t0 = time.monotonic()
    logger.info("[%s] Starting physical event extraction", STAGE)
    
    if not ctx.config.event_extraction.enabled:
        logger.info("[%s] Stage disabled in config", STAGE)
        _write_output(ctx, status="SKIPPED")
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="Event extraction stage skipped",
            duration_sec=time.monotonic() - t0,
        )
        
    normalizer = ActionNormalizer()
    ctx.events = []
    
    consumed = 0
    successful_obs = 0
    failed_skipped_obs = 0
    unknown_events = 0
    multi_events = 0
    
    for obs in ctx.vlm_observations:
        consumed += 1
        if obs.status != VLMSegmentStatus.SUCCESS:
            failed_skipped_obs += 1
            continue
            
        successful_obs += 1
        new_events = normalizer.normalize(obs)
        
        if len(new_events) > 1:
            multi_events += 1
            
        for evt in new_events:
            if evt.action.value == "UNKNOWN":
                unknown_events += 1
            ctx.events.append(evt)
            
    _write_output(ctx, status="SUCCESS")
    
    # Generate action breakdown for logging
    action_counts = {}
    for evt in ctx.events:
        action_counts[evt.action.value] = action_counts.get(evt.action.value, 0) + 1
        
    duration = time.monotonic() - t0
    logger.info(
        "[%s] Consumed %d obs | %d SUCCESS, %d FAILED/SKIPPED",
        STAGE, consumed, successful_obs, failed_skipped_obs
    )
    logger.info(
        "[%s] Generated %d events (%d UNKNOWN, %d from multi-event obs)",
        STAGE, len(ctx.events), unknown_events, multi_events
    )
    logger.info("[%s] Action breakdown: %s", STAGE, action_counts)
    
    return PipelineStageStatus(
        stage=STAGE, status="OK",
        message=f"Generated {len(ctx.events)} events",
        duration_sec=duration
    )
