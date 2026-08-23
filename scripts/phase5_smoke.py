"""Integration Smoke Test for Phase 5 VLM Semantic Analysis.

Executes: Ingest -> Sample -> Detect -> Track -> Segment -> VLM
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages import (
    s01_ingest,
    s02_sample,
    s03_detect,
    s04_track,
    s05_segment,
    s06_vlm,
)
from tests.test_phase1_ingest_sample import make_pipeline_context, make_video


def run_smoke_test():
    workspace = Path.cwd()
    real_video = workspace / "input" / "video.mp4"
    out_dir = workspace / "output_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    if real_video.exists():
        print(f"Using REAL video: {real_video}")
        video_path = real_video
        fps = 1.0
    else:
        video_path = workspace / "input" / "smoke_test.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Real video not found. Creating 5-second synthetic video at {video_path}")
        make_video(video_path, duration_sec=5.0, fps=10.0)
        fps = 2.0

    ctx = make_pipeline_context(video_path, out_dir, fps=fps)
    
    # Enable REAL mode for previous stages
    ctx.config.stub_mode = False
    ctx.config.detector.model = "yolov8n"
    ctx.config.detector.device = "auto"
    ctx.config.tracker.backend = "bytetrack"
    
    # Configure Phase 4
    ctx.config.segment.person_classes = ["person"]
    ctx.config.segment.proximity.iou_threshold = 0.05
    
    # Configure Phase 5
    ctx.config.vlm.enabled = True
    ctx.config.vlm.backend = "LOCAL_MODEL"
    ctx.config.vlm.model_name = "stub_local"

    print("\n--- Running pipeline through Stage 05 ---")
    ctx.record_stage(s01_ingest.run(ctx))
    ctx.record_stage(s02_sample.run(ctx))
    try:
        ctx.record_stage(s03_detect.run(ctx))
        ctx.record_stage(s04_track.run(ctx))
        ctx.record_stage(s05_segment.run(ctx))
    except Exception as e:  # noqa: BLE001
        print(f"CRITICAL FAILURE IN PIPELINE: {e}")
        sys.exit(1)
        
    print("\n--- Injecting mock candidate segment for VLM test ---")
    # If using synthetic video with 0 detections, we manually inject a candidate
    # so we can test the VLM actually running.
    if not ctx.candidate_segments:
        from src.schema.segment import CandidateSegment
        ctx.candidate_segments = [
            CandidateSegment(
                segment_id="cand_mock",
                start_frame=0,
                end_frame=10,
                start_sec=0.0,
                end_sec=5.0,
                trigger_reason="mock_injection"
            )
        ]
        print("Injected 1 mock candidate segment for testing.")

    print("\n--- Stage 06: VLM Semantic Analysis ---")
    try:
        st06 = s06_vlm.run(ctx)
        ctx.record_stage(st06)
    except Exception as e:  # noqa: BLE001
        print(f"CRITICAL FAILURE IN S06: {e}")
        sys.exit(1)

    print("\n--- RESULTS ---")
    print(f"Status: {st06.status}")
    print(f"Message: {st06.message}")

    if st06.status == "OK":
        obs_file = out_dir / "vlm_observations.json"
        if obs_file.exists():
            data = json.loads(obs_file.read_text())
            print(f"Total VLM observations generated: {len(data)}")
            
            for o in data:
                print(f"  Obs {o['observation_id']} | Status: {o['status']} | Action: {o.get('raw_action', 'None')} | Conf: {o.get('confidence', 'None')}")
                
            print(f"\nOutput saved to: {obs_file}")
            print("\nNOTE: VLM Infrastructure verified (Using LOCAL_MODEL stub).")
        else:
            print("ERROR: vlm_observations.json was not created.")
            sys.exit(1)
    else:
        print("ERROR: Pipeline stage failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
