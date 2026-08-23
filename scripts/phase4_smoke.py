"""Integration Smoke Test for Phase 4 Candidate Interaction Segmentation.

Executes: Ingest -> Sample -> Detect -> Track -> Segment
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stages import s01_ingest, s02_sample, s03_detect, s04_track, s05_segment
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
    
    # Enable REAL mode
    ctx.config.stub_mode = False
    ctx.config.detector.model = "yolov8n"
    ctx.config.detector.device = "auto"
    ctx.config.tracker.backend = "bytetrack"
    
    # Configure segmentation heuristics (Phase 4)
    ctx.config.segment.person_classes = ["person"]
    ctx.config.segment.proximity.iou_threshold = 0.05
    ctx.config.segment.proximity.gap_threshold_normalized = 0.2
    ctx.config.segment.movement.threshold = 0.05
    ctx.config.segment.movement.window_frames = 5
    ctx.config.segment.temporal_padding_sec = 2.0
    ctx.config.segment.merge_gap_sec = 1.0

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
        
    print("\n--- Stage 05: Segment ---")
    try:
        st05 = s05_segment.run(ctx)
        ctx.record_stage(st05)
    except Exception as e:  # noqa: BLE001
        print(f"CRITICAL FAILURE IN S05: {e}")
        sys.exit(1)

    print("\n--- RESULTS ---")
    print(f"Status: {st05.status}")
    print(f"Message: {st05.message}")

    if st05.status == "OK":
        segment_file = out_dir / "candidate_segments.json"
        if segment_file.exists():
            data = json.loads(segment_file.read_text())
            print(f"Total candidate segments generated: {len(data)}")
            
            for s in data:
                print(f"  Segment {s['segment_id']} | Tracks: {s['track_ids']} | {s['start_sec']}s -> {s['end_sec']}s | Reason: {s['trigger_reason']}")
                
            print(f"\nOutput saved to: {segment_file}")
            
            if not real_video.exists():
                print("\nNOTE: Used synthetic black video. Zero interactions is EXPECTED behavior.")
                print("Infrastructure is verified (SYNTHETIC VIDEO). Semantic accuracy requires a REAL physical-world video.")
            else:
                print("\nNOTE: Verified on REAL VIDEO.")
        else:
            print("ERROR: candidate_segments.json was not created.")
            sys.exit(1)
    else:
        print("ERROR: Pipeline stage failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
