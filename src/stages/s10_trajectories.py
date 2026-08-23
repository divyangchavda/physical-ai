"""Stage 10 — 2-D trajectory extraction.

Converts Track point sequences into Trajectory2D objects.

IMPORTANT: All trajectories are 2-D image-space trajectories.
They MUST NOT be interpreted as real 3-D physical trajectories.
The coordinate_space field is locked to '2D_IMAGE_PIXELS' by the schema.

SKIPPED in stub mode or when no tracks are available.

Output file: output/trajectories.json
Output context: ctx.trajectories (list[Trajectory2D])
"""
from __future__ import annotations

import json
import time

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.models.trajectory_extractor import TrajectoryExtractor
from src.schema.episode import PipelineStageStatus

logger = get_logger(__name__)
STAGE = "s10_trajectories"


def _write_output(ctx: PipelineContext) -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "trajectories.json"
    data = [t.model_dump(mode="json") for t in ctx.trajectories]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if ctx.config.stub_mode:
        logger.info("[%s] stub_mode=True — SKIPPED (no trajectories fabricated)", STAGE)
        ctx.trajectories = []
        _write_output(ctx)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: trajectory extraction skipped",
            duration_sec=time.monotonic() - t0,
        )

    if not ctx.tracks:
        logger.info("[%s] No tracks available — SKIPPED", STAGE)
        ctx.trajectories = []
        _write_output(ctx)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="no tracks available (s04_track must produce tracks first)",
            duration_sec=time.monotonic() - t0,
        )

    extractor = TrajectoryExtractor()
    trajectories = extractor.extract(ctx.tracks)
    ctx.trajectories = trajectories
    _write_output(ctx)

    logger.info("[%s] Extracted %d trajectories from %d tracks", STAGE, len(trajectories), len(ctx.tracks))
    return PipelineStageStatus(
        stage=STAGE, status="OK",
        message=f"Extracted {len(trajectories)} trajectories",
        duration_sec=time.monotonic() - t0,
    )
