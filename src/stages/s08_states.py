"""Stage 08: Object State Transitions."""

import json
import logging
import time

from src.context import PipelineContext, PipelineStageStatus
from src.models.state_inferencer import StateInferencer

STAGE = "s08_states"
logger = logging.getLogger(__name__)


def _write_output(ctx: PipelineContext, status: str) -> None:
    """Serialize the extracted state transitions to states.json."""
    # ctx.output_dir is this run's directory; ctx.config.output_dir is its
    # parent. Writing to the parent put states.json outside the run and let
    # every subsequent run overwrite it.
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = ctx.output_dir / "states.json"

    output_data = {
        "metadata": {
            "video": str(ctx.video_path),
            "stage": STAGE,
            "status": status,
        },
        "transitions": [t.model_dump(mode="json") for t in getattr(ctx, "state_transitions", [])]
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def run(ctx: PipelineContext) -> PipelineStageStatus:
    """Run the object state inference stage.
    
    This stage deterministically infers temporal state transitions from
    verified PhysicalEvent objects and RawVLMObservation evidence.
    """
    t0 = time.monotonic()
    logger.info("[%s] Starting object state inference", STAGE)
    
    if getattr(ctx.config, "stub_mode", False):
        logger.info("[%s] Stub mode enabled (skipping)", STAGE)
        _write_output(ctx, status="SKIPPED")
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: state extraction skipped",
            duration_sec=time.monotonic() - t0,
        )

    if not getattr(ctx.config, "state_extraction", None) or not ctx.config.state_extraction.enabled:
        logger.info("[%s] Stage disabled in config", STAGE)
        _write_output(ctx, status="SKIPPED")
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="State extraction stage skipped",
            duration_sec=time.monotonic() - t0,
        )
    inferencer = StateInferencer()
    if not hasattr(ctx, "state_transitions"):
        ctx.state_transitions = []
    
    obs_map = {obs.observation_id: obs for obs in ctx.vlm_observations if obs.observation_id}
    
    # Group events by segment
    segment_events = {}
    for evt in ctx.events:
        sid = evt.segment_id or "global"
        segment_events.setdefault(sid, []).append(evt)

    consumed = len(ctx.events)
    
    for sid, events in segment_events.items():
        # Get tracks for this segment if possible
        seg_tracks = []
        if sid != "global":
            for cand in getattr(ctx, "candidate_segments", []):
                if cand.segment_id == sid:
                    for tid in cand.track_ids:
                        for t in ctx.tracks:
                            if t.track_id == tid:
                                seg_tracks.append(t)
                                break
                    break
        else:
            seg_tracks = ctx.tracks
            
        transitions = inferencer.infer_transitions(events, obs_map, seg_tracks)
        ctx.state_transitions.extend(transitions)
            
    _write_output(ctx, status="SUCCESS")
    
    duration = time.monotonic() - t0
    logger.info(
        "[%s] Consumed %d events | Generated %d state transitions",
        STAGE, consumed, len(ctx.state_transitions)
    )
    
    return PipelineStageStatus(
        stage=STAGE, status="OK",
        message=f"Generated {len(ctx.state_transitions)} state transitions",
        duration_sec=duration
    )
