"""Real Model Smoke Test for Phase 3.

Executes: Ingest -> Sample -> YOLO Detect -> ByteTrack

Uses a real MP4 (input/video.mp4) if available, otherwise falls back
to a synthetic video to validate the infrastructure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.stages import s01_ingest, s02_sample, s03_detect, s04_track
from tests.test_phase1_ingest_sample import make_pipeline_context, make_video


def run_smoke_test():
    workspace = Path.cwd()
    real_video = workspace / "input" / "video.mp4"
    out_dir = workspace / "output_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    if real_video.exists():
        print(f"Using REAL video: {real_video}")
        video_path = real_video
        fps = 1.0  # sample 1 frame per second
    else:
        video_path = workspace / "input" / "smoke_test.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Real video not found. Creating 5-second synthetic video at {video_path}")
        make_video(video_path, duration_sec=5.0, fps=10.0)
        fps = 2.0  # 10 frames total to test tracking

    ctx = make_pipeline_context(video_path, out_dir, fps=fps)
    
    # Enable REAL mode
    ctx.config.stub_mode = False
    ctx.config.detector.model = "yolov8n"
    ctx.config.detector.device = "auto"
    ctx.config.detector.confidence = 0.35
    ctx.config.tracker.backend = "bytetrack"

    print("\n--- Stage 01: Ingest ---")
    ctx.record_stage(s01_ingest.run(ctx))

    print("\n--- Stage 02: Sample ---")
    ctx.record_stage(s02_sample.run(ctx))

    print("\n--- Stage 03: Detect (YOLO) ---")
    try:
        ctx.record_stage(s03_detect.run(ctx))
    except Exception as e:  # noqa: BLE001
        print(f"CRITICAL FAILURE IN S03: {e}")
        sys.exit(1)

    print("\n--- Stage 04: Track (ByteTrack) ---")
    try:
        st04 = s04_track.run(ctx)
        ctx.record_stage(st04)
    except Exception as e:  # noqa: BLE001
        print(f"CRITICAL FAILURE IN S04: {e}")
        sys.exit(1)

    print("\n--- RESULTS ---")
    print(f"Status: {st04.status}")
    print(f"Message: {st04.message}")

    if st04.status == "OK":
        track_file = out_dir / "tracks.json"
        if track_file.exists():
            data = json.loads(track_file.read_text())
            print(f"Total unique tracks found: {len(data)}")
            
            for t in data:
                pts = t.get("points", [])
                print(f"  Track ID {t.get('track_id')} ({t.get('class_name')}): {len(pts)} points")
                
            print(f"Output saved to: {track_file}")
            
            if not real_video.exists():
                print("\nNOTE: Used synthetic black video. Zero detections/tracks is EXPECTED behavior.")
                print("Infrastructure is verified, but tracking accuracy requires real video.")
        else:
            print("ERROR: tracks.json was not created.")
            sys.exit(1)
    else:
        print("ERROR: Pipeline stage failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
