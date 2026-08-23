"""Stage 05 — Candidate Interaction Segmentation.

Applies deterministic heuristics (proximity + movement) to tracking data to 
identify temporal windows where a physical interaction may be occurring.
Does not classify the semantic action (that happens in VLM stage).

Output file: output/candidate_segments.json
Output context: ctx.candidate_segments (list[CandidateSegment])
"""
from __future__ import annotations

import json
import time

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.models.heuristic_segmenter import generate_candidate_segments
from src.schema.episode import PipelineStageStatus

logger = get_logger(__name__)
STAGE = "s05_segment"


def _write_output(ctx: PipelineContext, status: str = "OK") -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "candidate_segments.json"
    data = [c.model_dump(mode="json") for c in ctx.candidate_segments]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if ctx.config.stub_mode:
        logger.info("[%s] stub_mode=True — SKIPPED (no candidate segments fabricated)", STAGE)
        ctx.candidate_segments = []
        _write_output(ctx, status="SKIPPED")
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: segmentation stage skipped",
            duration_sec=time.monotonic() - t0,
        )

    if not hasattr(ctx, "tracks") or ctx.tracks is None:
        msg = "No tracks — s04_track must run first"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    # Ensure video metadata exists for normalized dimensions/duration
    w, h = 1920, 1080
    dur = 600.0
    if hasattr(ctx, "video_metadata") and ctx.video_metadata:
        w = ctx.video_metadata.width or w
        h = ctx.video_metadata.height or h
        dur = ctx.video_metadata.duration_sec or dur

    # Run heuristics
    try:
        segments = generate_candidate_segments(
            tracks=ctx.tracks,
            config=ctx.config.segment,
            frame_width=w,
            frame_height=h,
            video_duration_sec=dur,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[%s] Heuristic segmentation failed: %s", STAGE, exc)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=str(exc))
        
    t_end = time.monotonic()
    duration = t_end - t0

    ctx.candidate_segments = segments
    _write_output(ctx)

    logger.info(
        "[%s] Processed %d tracks | generated %d candidate segments in %.3fs",
        STAGE, len(ctx.tracks), len(segments), duration
    )
    
    msg = f"Generated {len(segments)} candidate segments from {len(ctx.tracks)} tracks."
    
    return PipelineStageStatus(
        stage=STAGE, status="OK", message=msg, duration_sec=duration
    )
