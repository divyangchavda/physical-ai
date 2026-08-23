"""Real Model Smoke Test for Phase 2.

This script executes the actual YOLO model against a short synthetic video
to verify end-to-end processing without relying on mocks.

It MUST be run separately from the unit test suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.stages import s01_ingest, s02_sample, s03_detect
from tests.test_phase1_ingest_sample import make_pipeline_context, make_video


def run_smoke_test():
    workspace = Path.cwd()
    video_path = workspace / "input" / "smoke_test.mp4"
    out_dir = workspace / "output_smoke"
    
    video_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating 3-second synthetic video at {video_path}")
    make_video(video_path, duration_sec=3.0, fps=10.0)

    # 1 fps for 3 seconds -> ~3 frames
    ctx = make_pipeline_context(video_path, out_dir, fps=1.0)
    
    # Must use non-stub mode to test the real model
    ctx.config.stub_mode = False
    
    # Ensure default YOLO model
    ctx.config.detector.model = "yolov8n"
    ctx.config.detector.device = "auto"
    ctx.config.detector.confidence = 0.35

    print("\n--- Running Stage 01: Ingest ---")
    st01 = s01_ingest.run(ctx)
    ctx.record_stage(st01)

    print("\n--- Running Stage 02: Sample ---")
    st02 = s02_sample.run(ctx)
    ctx.record_stage(st02)
    
    print("\n--- Running Stage 03: Detect (REAL YOLO) ---")
    try:
        st03 = s03_detect.run(ctx)
        ctx.record_stage(st03)
    except Exception as e:  # noqa: BLE001
        print(f"\nCRITICAL FAILURE IN S03: {e}")
        sys.exit(1)

    print("\n--- RESULTS ---")
    print(f"Status: {st03.status}")
    print(f"Message: {st03.message}")
    
    if st03.status == "OK":
        det_file = out_dir / "detections.json"
        if det_file.exists():
            data = json.loads(det_file.read_text())
            n_det = sum(len(f.get("detections", [])) for f in data)
            print(f"Total detections found: {n_det}")
            print(f"Output saved to: {det_file}")
        else:
            print("ERROR: detections.json was not created.")
            sys.exit(1)
    else:
        print("ERROR: Pipeline stage failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
