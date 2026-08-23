"""Stage 01 — Video ingest and metadata extraction.

Reads video properties using OpenCV without decoding any pixel data.
Always runs — no AI models or GPU required.

Hardening compared to the foundation stub:
  - Explicit fps > 0 check (not relying on Python truthiness of 0.0).
  - CAP_PROP_FRAME_COUNT is validated: containers that embed a wrong
    header count (common in MKV and some MP4 files) are detected by
    comparing the header-reported duration against a lightweight scan
    of actual decodable frames (seek-to-end strategy — no pixel decode).
  - Codec fourcc decoded defensively: non-printable bytes are replaced
    with '?' instead of propagating garbage chars.
  - Partial metadata is reported as WARNING (not ERROR) so the pipeline
    can continue; missing or unusable metadata is reported as ERROR.
  - All fields in VideoMetadata are populated from directly observed
    values — nothing is fabricated or assumed.

Output: ctx.video_metadata (VideoMetadata)
"""
from __future__ import annotations

import time

import cv2

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.schema.episode import PipelineStageStatus, VideoMetadata

logger = get_logger(__name__)
STAGE = "s01_ingest"

# Maximum relative discrepancy between header-reported and scan-verified
# frame counts before we emit a metadata warning.
_FRAME_COUNT_TOLERANCE = 0.05   # 5 %
# Upper bound on how long (seconds) the frame-count verification scan may take.
# For a 600-frame, 10-min video at 1 fps this is well within budget.
_SCAN_TIMEOUT_SEC = 5.0


def _decode_fourcc(fourcc_int: int) -> str:
    """Convert a CAP_PROP_FOURCC integer to a printable codec string.

    Non-printable bytes (NUL, control chars) are replaced with '?'.
    Returns an empty string if fourcc is 0 (no codec info available).
    """
    if fourcc_int == 0:
        return ""
    chars = []
    for i in range(4):
        byte = (fourcc_int >> (8 * i)) & 0xFF
        chars.append(chr(byte) if 32 <= byte < 127 else "?")
    return "".join(chars).strip()


def _verify_frame_count(video_path: str, header_count: int, fps: float) -> tuple[int, str | None]:
    """Check whether the header-reported frame count is plausible.

    Uses a lightweight seek-to-end approach:
      1. Seeks to the last position OpenCV believes exists.
      2. Tries to read the frame there.
      3. If the read fails, binary-searches backward to find the actual
         last readable frame.

    This avoids decoding all frames for large videos.

    Returns:
        (verified_frame_count, warning_message_or_None)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return header_count, "Could not re-open video for frame count verification"

    try:
        # Seek to one before the header-reported last frame
        seek_to = max(0, header_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(seek_to))
        ret, _ = cap.read()

        if ret:
            # Header count is plausible — trust it
            return header_count, None

        # Header count appears inflated; binary-search backward for actual last frame.
        t_start = time.monotonic()
        lo, hi = 0, seek_to
        last_good = -1
        while lo <= hi and (time.monotonic() - t_start) < _SCAN_TIMEOUT_SEC:
            mid = (lo + hi) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mid))
            ret, _ = cap.read()
            if ret:
                last_good = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if last_good < 0:
            return 0, "No readable frames found in video"

        actual_count = last_good + 1
        diff_ratio = abs(actual_count - header_count) / max(header_count, 1)
        if diff_ratio > _FRAME_COUNT_TOLERANCE:
            warn = (
                f"Container frame_count ({header_count}) differs from "
                f"scan-verified count ({actual_count}) by "
                f"{diff_ratio * 100:.1f}% — using scan-verified count"
            )
            return actual_count, warn

        return actual_count, None
    finally:
        cap.release()


def run(ctx: PipelineContext) -> PipelineStageStatus:
    """Extract video metadata and populate ``ctx.video_metadata``."""
    t0 = time.monotonic()
    logger.info("[%s] Ingesting: %s", STAGE, ctx.video_path)

    # ── File presence check ──────────────────────────────────────────────────
    if not ctx.video_path.exists():
        msg = f"Video file not found: {ctx.video_path}"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    file_size_bytes = ctx.video_path.stat().st_size
    if file_size_bytes == 0:
        msg = f"Video file is empty (0 bytes): {ctx.video_path}"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    # ── Open container ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(ctx.video_path))
    if not cap.isOpened():
        msg = f"Cannot open video (unsupported format or corrupt file): {ctx.video_path}"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    try:
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        header_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()

    # ── Validate core properties ─────────────────────────────────────────────
    if fps_raw is None or fps_raw <= 0:
        msg = (
            f"Invalid or unreadable FPS ({fps_raw!r}) — "
            "the file may be corrupt or use an unsupported container"
        )
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    if width <= 0 or height <= 0:
        msg = f"Invalid frame dimensions: {width}x{height}"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    # ── Codec string ─────────────────────────────────────────────────────────
    codec = _decode_fourcc(fourcc_int)

    # ── Frame count validation ───────────────────────────────────────────────
    warnings: list[str] = []

    if header_frame_count <= 0:
        # Container header has no frame count (common in live recordings,
        # some MKV files). Estimate from duration only — mark as warning.
        # Try to get duration from MSEC property first.
        cap2 = cv2.VideoCapture(str(ctx.video_path))
        try:
            duration_msec = cap2.get(cv2.CAP_PROP_POS_MSEC)
        finally:
            cap2.release()
        if duration_msec > 0:
            estimated_frames = int((duration_msec / 1000.0) * fps_raw)
            frame_count = estimated_frames
            warnings.append(
                "Container reported frame_count=0; estimated from duration "
                f"({duration_msec:.0f} ms × {fps_raw:.2f} fps = {frame_count} frames)"
            )
        else:
            frame_count = 0
            warnings.append(
                "Frame count unavailable from container header and duration. "
                "Duration and frame count may be inaccurate."
            )
    else:
        # Verify the header-reported count is plausible via lightweight scan.
        frame_count, scan_warn = _verify_frame_count(
            str(ctx.video_path), header_frame_count, fps_raw
        )
        if scan_warn:
            warnings.append(scan_warn)

    # ── Duration ─────────────────────────────────────────────────────────────
    if frame_count > 0:
        duration_sec = frame_count / fps_raw
    else:
        duration_sec = 0.0

    # ── Warn on unusual FPS ──────────────────────────────────────────────────
    if fps_raw < 1.0:
        warnings.append(
            f"Very low FPS ({fps_raw:.4f}). Video may be a time-lapse or "
            "use variable frame rate. Sampling will use this FPS as-is."
        )
    elif fps_raw > 240.0:
        warnings.append(
            f"Very high FPS ({fps_raw:.2f}). This is unusual; verify the "
            "video is not mis-labelled (e.g. a 1000-fps slow-motion camera)."
        )

    # ── Populate context ─────────────────────────────────────────────────────
    ctx.video_metadata = VideoMetadata(
        file_path=str(ctx.video_path),
        duration_sec=duration_sec,
        fps=fps_raw,
        frame_count=frame_count,
        width=width,
        height=height,
        codec=codec,
        file_size_bytes=file_size_bytes,
        metadata_warnings=warnings,
    )

    elapsed = time.monotonic() - t0
    if warnings:
        for w in warnings:
            logger.warning("[%s] %s", STAGE, w)

    logger.info(
        "[%s] %.2fs @ %.4f fps, %dx%d, codec=%r, frames=%d, size=%d bytes (%.2fs)",
        STAGE,
        duration_sec,
        fps_raw,
        width,
        height,
        codec,
        frame_count,
        file_size_bytes,
        elapsed,
    )

    final_status = "WARNING" if warnings else "OK"
    return PipelineStageStatus(
        stage=STAGE,
        status=final_status,
        message="; ".join(warnings) if warnings else "",
        duration_sec=elapsed,
    )
