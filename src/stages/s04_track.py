"""Stage 04 — Object tracking.

Runs the configured ObjectTracker over the detection frames.
SKIPPED in stub mode — produces an empty-but-valid tracks.json.

Output file: output/tracks.json
Output context: ctx.tracks (list[Track])
"""
from __future__ import annotations

import json
import time

from src.context import PipelineContext
from src.interfaces.tracker import ObjectTracker
from src.logging_utils import get_logger
from src.models.stub_tracker import StubTracker
from src.models.track_stitcher import stitch_tracks
from src.schema.detection import Detection
from src.schema.episode import PipelineStageStatus
from src.schema.track import Track

logger = get_logger(__name__)
STAGE = "s04_track"


def _detection_stride(detection_frames: list[int]) -> int:
    """Infer the gap between detection attempts from the sampled frame indices.

    The tracker needs this because its miss counter ticks on every video frame,
    not just the ones where detection ran. Median rather than min so a single
    dropped/ERROR frame doubling one gap does not skew the result.
    """
    if len(detection_frames) < 2:
        return 1
    ordered = sorted(detection_frames)
    gaps = sorted(b - a for a, b in zip(ordered, ordered[1:]))
    return max(1, gaps[len(gaps) // 2])


def _build_tracker(ctx: PipelineContext, detection_stride: int = 1) -> ObjectTracker:
    """Return the appropriate tracker for the current config."""
    if ctx.config.stub_mode:
        return StubTracker()

    try:
        if ctx.config.tracker.backend == "bytetrack":
            from src.models.bytetrack_tracker import ByteTrackTracker
            return ByteTrackTracker()
        elif ctx.config.tracker.backend == "kalman_sparse":
            from src.models.kalman_sparse_tracker import KalmanSparseTracker
            return KalmanSparseTracker(
                iou_threshold=ctx.config.tracker.iou_threshold,
                max_age=ctx.config.tracker.max_age,
                min_hits=ctx.config.tracker.min_hits,
                frame_width=ctx.video_metadata.width if ctx.video_metadata else None,
                frame_height=ctx.video_metadata.height if ctx.video_metadata else None,
                detection_stride=detection_stride,
                max_missed_detections=ctx.config.tracker.max_missed_detections,
            )
        else:
            logger.warning(
                "[%s] Unknown tracker backend '%s'. Using StubTracker.",
                STAGE, ctx.config.tracker.backend
            )
            return StubTracker()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[%s] Cannot instantiate tracker (%s); falling back to StubTracker",
            STAGE, exc,
        )
        return StubTracker()


def _write_output(ctx: PipelineContext, status: str = "OK") -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "tracks.json"
    data = [t.model_dump(mode="json") for t in ctx.tracks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _class_counts(tracks: list[Track]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tracks:
        counts[t.class_name] = counts.get(t.class_name, 0) + 1
    return dict(sorted(counts.items()))


def _write_raw_tracks(ctx: PipelineContext) -> None:
    """Snapshot the tracker's own output before stitching rewrites it.

    Without this, ``tracks.json`` holds the merged entities and the fragments
    are unrecoverable, so every threshold change costs a full GPU run to
    re-detect. With it, the stitcher can be re-tuned offline against a saved
    file. It is also the only honest record of what the tracker actually
    produced.
    """
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    with open(ctx.output_dir / "tracks_raw.json", "w", encoding="utf-8") as f:
        json.dump([t.model_dump(mode="json") for t in ctx.tracks], f, indent=2)


def _stitch(ctx: PipelineContext, detection_stride: int) -> None:
    """Merge fragmented tracks into one track per physical object.

    Writes ``track_merges.json`` alongside ``tracks.json``: every surviving id
    with the original ids folded into it, plus its frame span. A merge must be
    auditable, and the spans are what tell us *where* the entity breaks landed —
    on a real cut, or in the middle of a continuous motion.

    The overlap and duplicate thresholds are *derived* from the tracker's own
    settings rather than picked. A guessed overlap budget of 2 frames was the
    reason the first version of this stage merged 7 fragments out of 40: the
    tracker's deletion lag makes sibling fragments overlap by up to
    ``max(max_age, stride * (max_missed_detections + 1))`` frames by
    construction, so 2 could not fire. Both effective values are written into
    the report so the number in use is never implicit.
    """
    cfg = ctx.config.tracker
    if not getattr(cfg, "stitch_enabled", False) or not ctx.tracks:
        return

    # Same expression as kalman_sparse_tracker.py:386, which is what actually
    # bounds how long a dying track keeps emitting points beside its successor.
    tail_frames = max(cfg.max_age, detection_stride * (cfg.max_missed_detections + 1))
    overlap_budget = (
        cfg.stitch_max_overlap_frames
        if cfg.stitch_max_overlap_frames is not None
        else tail_frames
    )
    duplicate_min_iou = (
        cfg.stitch_duplicate_min_iou
        if cfg.stitch_duplicate_min_iou is not None
        else cfg.iou_threshold
    )

    before = _class_counts(ctx.tracks)
    n_before = len(ctx.tracks)

    entities, absorbed = stitch_tracks(
        ctx.tracks,
        frame_width=ctx.video_metadata.width if ctx.video_metadata else 0,
        frame_height=ctx.video_metadata.height if ctx.video_metadata else 0,
        max_gap_frames=cfg.stitch_max_gap_frames,
        max_overlap_frames=overlap_budget,
        iou_threshold=cfg.stitch_iou_threshold,
        max_center_dist_norm=cfg.stitch_max_center_dist_norm,
        duplicate_min_iou=duplicate_min_iou,
    )
    ctx.tracks = entities
    after = _class_counts(ctx.tracks)

    report = {
        "enabled": True,
        "max_gap_frames": cfg.stitch_max_gap_frames,
        "max_overlap_frames": overlap_budget,
        "max_overlap_frames_source": (
            "config" if cfg.stitch_max_overlap_frames is not None
            else f"derived: max(max_age={cfg.max_age}, "
                 f"stride={detection_stride} * (max_missed_detections="
                 f"{cfg.max_missed_detections} + 1)) = {tail_frames}"
        ),
        "iou_threshold": cfg.stitch_iou_threshold,
        "max_center_dist_norm": cfg.stitch_max_center_dist_norm,
        "duplicate_min_iou": duplicate_min_iou,
        "duplicate_min_iou_source": (
            "config" if cfg.stitch_duplicate_min_iou is not None
            else f"derived: tracker.iou_threshold = {cfg.iou_threshold}"
        ),
        "tracks_before": n_before,
        "tracks_after": len(entities),
        "by_class_before": before,
        "by_class_after": after,
        "entities": [
            {
                "entity_id": t.track_id,
                "class_name": t.class_name,
                "start_frame": t.start_frame,
                "end_frame": t.end_frame,
                "n_points": len(t.points),
                "absorbed_track_ids": absorbed.get(t.track_id, [t.track_id]),
            }
            for t in entities
        ],
    }
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    with open(ctx.output_dir / "track_merges.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "[%s] Stitched %d fragments into %d entities "
        "(overlap budget %d frames, duplicate IoU %.2f) | by class before=%s after=%s",
        STAGE, n_before, len(entities), overlap_budget, duplicate_min_iou,
        before, after,
    )


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if ctx.config.stub_mode:
        logger.info("[%s] stub_mode=True — SKIPPED (no tracks fabricated)", STAGE)
        ctx.tracks = []
        _write_output(ctx, status="SKIPPED")
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: tracking stage skipped",
            duration_sec=time.monotonic() - t0,
        )

    if not hasattr(ctx, "detection_frames") or ctx.detection_frames is None:
        msg = "No detection frames — s03_detect must run first"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    if not hasattr(ctx, "video_metadata") or ctx.video_metadata is None:
        msg = "No video metadata — s01_ingest must run first"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    # Build a map of frame_index → DetectionFrame for quick lookup
    detection_map: dict[int, list[Detection]] = {}
    for df in ctx.detection_frames:
        if df.status == "OK":
            detection_map[df.frame_index] = df.detections

    # The tracker needs the detection stride to size its miss budget.
    stride = _detection_stride(list(detection_map.keys()))

    tracker = _build_tracker(ctx, detection_stride=stride)
    try:
        # Some trackers might need explicit setup/load, though ObjectTracker doesn't formally mandate load()
        # ByteTrackTracker does lazy setup in update().
        tracker.reset()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] Tracker initialization failed (%s) — using StubTracker", STAGE, exc)
        tracker = StubTracker()

    # Get total frame count from video metadata
    total_frames = ctx.video_metadata.frame_count
    
    # CRITICAL: Process EVERY frame in the video, not just sampled frames
    # On frames with detections: pass actual detections to tracker
    # On frames without detections: pass empty list to tracker (allows interpolation)
    tracking_time_sec = 0.0
    frames_processed = 0
    frames_with_detections = 0
    frames_without_detections = 0
    all_tracks_by_id: dict[int, Track] = {}
    max_simultaneous = 0

    logger.info(
        "[%s] Processing ALL %d frames (detection frames: %d, interpolation "
        "frames: %d, detection stride: %d)",
        STAGE, total_frames, len(detection_map),
        total_frames - len(detection_map), stride,
    )

    for frame_idx in range(total_frames):
        # Check if this frame has detections
        if frame_idx in detection_map:
            detections = detection_map[frame_idx]
            frames_with_detections += 1
        else:
            # No detections for this frame — pass empty list to tracker
            # This allows ByteTrack to interpolate/predict track positions
            detections = []
            frames_without_detections += 1
        
        t_trk0 = time.monotonic()
        active_tracks = tracker.update(detections, frame_idx)
        t_trk1 = time.monotonic()
        
        tracking_time_sec += (t_trk1 - t_trk0)
        frames_processed += 1
        
        # Track statistics
        max_simultaneous = max(max_simultaneous, len(active_tracks))
        
        # Collect final track objects
        for t in active_tracks:
            all_tracks_by_id[t.track_id] = t

    # Finalize tracks
    ctx.tracks = list(all_tracks_by_id.values())
    _write_raw_tracks(ctx)
    _stitch(ctx, detection_stride=stride)
    _write_output(ctx)

    tracker.reset()

    total_unique_tracks = len(ctx.tracks)
    total_detections_received = sum(len(detections) for detections in detection_map.values())
    avg_trk = tracking_time_sec / frames_processed if frames_processed > 0 else 0.0
    
    logger.info(
        "[%s] %d total frames processed (%d with detections, %d interpolated) | "
        "detections received: %d | unique tracks: %d | max simultaneous: %d | "
        "tracking: %.2fs total (%.3fs avg/frame)",
        STAGE, frames_processed, frames_with_detections, frames_without_detections,
        total_detections_received, total_unique_tracks, max_simultaneous, 
        tracking_time_sec, avg_trk
    )
    
    msg = (
        f"Processed {frames_processed} frames ({frames_with_detections} with detections, "
        f"{frames_without_detections} interpolated). Created {total_unique_tracks} tracks. "
        f"Max simultaneous: {max_simultaneous}. Tracking time: {tracking_time_sec:.2f}s total, "
        f"{avg_trk:.3f}s avg/frame."
    )
    
    return PipelineStageStatus(
        stage=STAGE, status="OK", message=msg, duration_sec=time.monotonic() - t0
    )
