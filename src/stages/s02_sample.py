"""Stage 02 — Frame sampling plan computation.

Determines WHICH frames to sample from the source video.
Pure arithmetic + deduplication — does NOT read pixel data.
Always runs.

Design:
  - Samples are computed in floating-point index space to minimize
    accumulation of rounding error across 600+ samples.
  - Each sample index is clamped to [0, frame_count - 1] before use.
  - Duplicate indices (possible only when target_fps > video_fps) are
    removed so each physical frame appears at most once.
  - Timestamps are derived from the frame index and the source FPS —
    NOT from wall-clock time — so they are reproducible.
  - The resulting plan is serialized to output/sampling_plan.json for
    debugging and later YOLO stage consumption.
  - segment_fps is preserved in config for future VLM segment resampling;
    it is NOT implemented or used here.

Memory contract:
  - No pixel data is allocated.
  - ctx.sampled_frame_infos holds only (frame_index, timestamp_sec, video_path)
    per sample — ~100 bytes each → ~60 KB for 600 frames. Negligible.

Output file: output/sampling_plan.json
Output context: ctx.sampled_frame_infos (list[SampledFrameInfo])
"""
from __future__ import annotations

import json
import time

from src.context import PipelineContext, SampledFrameInfo
from src.logging_utils import get_logger
from src.schema.episode import PipelineStageStatus

logger = get_logger(__name__)
STAGE = "s02_sample"


def _compute_sample_indices(
    video_fps: float,
    frame_count: int,
    target_fps: float,
    max_frames: int,
    every_n_frames: int | None = None,
) -> list[int]:
    """Compute a deduplicated, clamped list of 0-based frame indices to sample.

    Args:
        video_fps:    FPS of the source video.
        frame_count:  Total number of frames in the source video.
        target_fps:   Desired sampling rate (fps). Clamped to video_fps.
        max_frames:   Hard cap on the number of frames returned.
        every_n_frames: If set, sample every N frames instead of using target_fps.

    Returns:
        Sorted, unique list of frame indices, length <= max_frames.
    """
    if frame_count <= 0:
        return []

    # Mode 1: Sample every N frames (sparse detection mode)
    if every_n_frames is not None and every_n_frames > 0:
        indices = list(range(0, frame_count, every_n_frames))
        # Apply max_frames cap
        if len(indices) > max_frames:
            indices = indices[:max_frames]
        return indices

    # Mode 2: Sample by target FPS (original behavior)
    # Clamp target to source FPS — cannot sample faster than source
    effective_fps = min(target_fps, video_fps)
    # Number of source frames to advance per sample
    step = video_fps / effective_fps  # always >= 1.0 after clamping

    indices: list[int] = []
    seen: set[int] = set()
    position = 0.0

    while position < frame_count and len(indices) < max_frames:
        idx = round(position)          # round rather than truncate to minimise drift
        idx = max(0, min(idx, frame_count - 1))   # clamp to valid range
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
        position += step

    return indices


def _write_sampling_plan(ctx: PipelineContext) -> None:
    """Write the sampling plan to output/sampling_plan.json."""
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "sampling_plan.json"
    plan = {
        "video_path": str(ctx.video_path),
        "video_fps": ctx.video_metadata.fps if ctx.video_metadata else None,
        "video_frame_count": ctx.video_metadata.frame_count if ctx.video_metadata else None,
        "target_fps": ctx.config.frame_sampling.fps,
        "segment_fps_reserved": ctx.config.frame_sampling.segment_fps,
        "max_frames_cap": ctx.config.frame_sampling.max_frames,
        "n_frames_sampled": len(ctx.sampled_frame_infos),
        "frames": [
            {
                "frame_index": info.frame_index,
                "timestamp_sec": round(info.timestamp_sec, 6),
            }
            for info in ctx.sampled_frame_infos
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)


def run(ctx: PipelineContext) -> PipelineStageStatus:
    """Compute the frame sampling plan and populate ``ctx.sampled_frame_infos``."""
    t0 = time.monotonic()

    if ctx.video_metadata is None:
        msg = "video_metadata is None — s01_ingest must succeed first"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    video_fps = ctx.video_metadata.fps
    frame_count = ctx.video_metadata.frame_count
    target_fps = ctx.config.frame_sampling.fps
    max_frames = ctx.config.frame_sampling.max_frames
    every_n_frames = ctx.config.frame_sampling.every_n_frames

    # ── Handle edge cases ────────────────────────────────────────────────────
    if frame_count <= 0:
        # Frame count unknown (container metadata unreliable).
        # Fall back to duration × fps estimate.
        estimated = int(ctx.video_metadata.duration_sec * video_fps)
        if estimated <= 0:
            msg = (
                f"Cannot determine frame count: "
                f"frame_count={frame_count}, "
                f"duration={ctx.video_metadata.duration_sec:.2f}s"
            )
            logger.warning("[%s] %s — producing empty sample list", STAGE, msg)
            ctx.sampled_frame_infos = []
            _write_sampling_plan(ctx)
            return PipelineStageStatus(
                stage=STAGE,
                status="WARNING",
                message=msg,
                duration_sec=time.monotonic() - t0,
            )
        logger.warning(
            "[%s] frame_count=0 from container; using duration-based estimate: %d frames",
            STAGE, estimated,
        )
        frame_count = estimated

    # ── Compute sample indices ───────────────────────────────────────────────
    indices = _compute_sample_indices(
        video_fps=video_fps,
        frame_count=frame_count,
        target_fps=target_fps,
        max_frames=max_frames,
        every_n_frames=every_n_frames,
    )

    # ── Build SampledFrameInfo list ──────────────────────────────────────────
    # Timestamps are derived from frame_index / video_fps.
    # This is the standard definition for constant-fps video: the frame
    # at index N begins at N / fps seconds from the stream start.
    ctx.sampled_frame_infos = [
        SampledFrameInfo(
            frame_index=idx,
            timestamp_sec=idx / video_fps,
            video_path=ctx.video_path,
        )
        for idx in indices
    ]

    # ── Write plan to disk ───────────────────────────────────────────────────
    _write_sampling_plan(ctx)

    # ── Effective sampling rate (actual) ─────────────────────────────────────
    if every_n_frames is not None:
        effective_sampling = f"every {every_n_frames} frames"
    else:
        effective_fps = min(target_fps, video_fps)
        effective_sampling = f"effective {effective_fps:.4f} fps"
    
    n = len(ctx.sampled_frame_infos)
    first_ts = ctx.sampled_frame_infos[0].timestamp_sec if n > 0 else 0.0
    last_ts = ctx.sampled_frame_infos[-1].timestamp_sec if n > 0 else 0.0

    logger.info(
        "[%s] %d frames selected (%s, video=%.4f fps, cap=%d) | first=%.3fs last=%.3fs",
        STAGE,
        n,
        effective_sampling,
        video_fps,
        max_frames,
        first_ts,
        last_ts,
    )

    if n == max_frames:
        logger.warning(
            "[%s] Sample count hit the max_frames cap (%d). "
            "Consider reducing target fps or increasing max_frames for long videos.",
            STAGE, max_frames,
        )

    return PipelineStageStatus(
        stage=STAGE,
        status="OK",
        duration_sec=time.monotonic() - t0,
    )
