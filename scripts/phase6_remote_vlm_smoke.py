"""Opt-in smoke test for real Gemini VLM inference.

This script is NOT part of the normal pytest suite.

It requires:
  - GEMINI_API_KEY environment variable set to a valid key.
  - A real video file passed as the first argument (or placed at input/test.mp4).

Run manually:
  python scripts/phase6_remote_vlm_smoke.py [path/to/video.mp4]

The test:
  1. Checks GEMINI_API_KEY is set.
  2. Opens the video and picks a segment near the start.
  3. Instantiates GeminiVLM and calls analyze_segment().
  4. Parses the response as JSON.
  5. Validates the expected semantic fields.
  6. Verifies timestamp bounds.
  7. Prints latency and a redacted result summary.
  8. Feeds the observation through Stage 07 to verify the downstream contract.

NEVER prints the API key.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("phase6_remote_vlm_smoke")

    # ── 1. Check API key ────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.warning("SKIPPED: GEMINI_API_KEY is not set.")
        log.warning(
            "Obtain a free key at https://aistudio.google.com/ and set it "
            "in your shell: set GEMINI_API_KEY=<your-key>"
        )
        return 0

    log.info("GEMINI_API_KEY is set (not printed).")

    # ── 2. Find video ───────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        video_path = Path("input") / "test.mp4"

    if not video_path.exists():
        log.error(
            "Video file not found: %s\n"
            "Pass a path as the first argument or place a video at input/test.mp4",
            video_path,
        )
        return 1

    log.info("Video: %s", video_path)

    # ── 3. Choose a short test segment ─────────────────────────────────────
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_sec = total_frames / fps
    start_sec = min(2.0, duration_sec * 0.1)
    end_sec = min(start_sec + 4.0, duration_sec - 0.5)

    if end_sec <= start_sec:
        log.error("Video too short to extract a test segment.")
        return 1

    log.info("Segment: [%.2f, %.2f]s (%.2fs duration)", start_sec, end_sec, end_sec - start_sec)

    # ── 4. Build GeminiVLM and run analyze_segment ─────────────────────────
    from src.models.gemini_vlm import GeminiVLM
    from src.stages.s06_vlm import PROMPT

    log.info("Initialising GeminiVLM (model: gemini-2.5-flash)...")
    vlm = GeminiVLM(model_name="gemini-2.5-flash", timeout_sec=90.0)

    log.info("Running analyze_segment()...")
    t0 = time.monotonic()
    try:
        raw_response = vlm.analyze_segment(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec,
            prompt=PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("analyze_segment() raised: %s", exc)
        return 1

    latency = time.monotonic() - t0
    log.info("Latency: %.2fs", latency)

    # ── 5. Validate JSON ────────────────────────────────────────────────────
    from src.stages.s06_vlm import extract_json
    try:
        json_str = extract_json(raw_response)
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.error("Invalid JSON: %s", exc)
        log.error("Raw response: %.200s", raw_response)
        return 1

    log.info("JSON valid: YES")

    # ── 6. Validate semantic fields ─────────────────────────────────────────
    REQUIRED = {"actor", "active_hand", "objects", "raw_action", "confidence"}
    missing = REQUIRED - data.keys()
    if missing:
        log.warning("Missing fields: %s", missing)
    else:
        log.info("All required semantic fields present.")

    # Redacted summary — never prints raw response
    log.info("actor        : %s", data.get("actor", "<missing>"))
    log.info("active_hand  : %s", data.get("active_hand", "<missing>"))
    log.info("objects      : %s", data.get("objects", "<missing>"))
    log.info("raw_action   : %s", data.get("raw_action", "<missing>"))
    log.info("state_change : %s", data.get("state_change", "<missing>"))
    log.info("confidence   : %s", data.get("confidence", "<missing>"))
    start_t = data.get("start_time_sec")
    end_t = data.get("end_time_sec")
    log.info("start_time_sec: %s", start_t)
    log.info("end_time_sec  : %s", end_t)

    # ── 7. Validate timestamps ──────────────────────────────────────────────
    if start_t is not None:
        abs_start = start_sec + float(start_t)
        if not (start_sec <= abs_start <= end_sec):
            log.warning("start_time_sec converts to %.2f — outside [%.2f, %.2f]", abs_start, start_sec, end_sec)
        else:
            log.info("start_time_sec is within segment bounds (after offset conversion).")

    # ── 8. Build RawVLMObservation (Stage 06 contract) ─────────────────────
    from src.schema.vlm import RawVLMObservation, VLMSegmentStatus

    if data.get("start_time_sec") is not None:
        data["start_time_sec"] = start_sec + float(data["start_time_sec"])
    if data.get("end_time_sec") is not None:
        data["end_time_sec"] = start_sec + float(data["end_time_sec"])

    try:
        obs = RawVLMObservation(
            observation_id="obs_smoke_001",
            segment_id="seg_smoke_001",
            status=VLMSegmentStatus.SUCCESS,
            backend="REMOTE_MODEL",
            model_name=vlm.model_name,
            prompt_version="v1",
            segment_start_sec=start_sec,
            segment_end_sec=end_sec,
            raw_response="<redacted>",
            **data,
        )
        log.info("RawVLMObservation: VALID")
    except Exception as exc:  # noqa: BLE001
        log.error("RawVLMObservation validation FAILED: %s", exc)
        return 1

    # ── 9. Feed through Stage 07 ────────────────────────────────────────────
    from src.models.action_normalizer import ActionNormalizer
    normalizer = ActionNormalizer()
    events = normalizer.normalize(obs)
    log.info("Stage 07 → %d PhysicalEvent(s):", len(events))
    for ev in events:
        log.info("  event_id=%s  action=%s  confidence=%.2f", ev.event_id, ev.action, ev.confidence)

    # ── Final report ────────────────────────────────────────────────────────
    log.info("-" * 60)
    log.info("SMOKE TEST: PASS")
    log.info("  Segment duration : %.2fs", end_sec - start_sec)
    log.info("  Model            : %s", vlm.model_name)
    log.info("  Latency          : %.2fs", latency)
    log.info("  JSON valid       : YES")
    log.info("  RawVLMObservation: VALID")
    log.info("  Stage 07 events  : %d", len(events))
    log.info("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
