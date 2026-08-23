"""Integration test: GroundingDINO + Kalman + VLM pipeline.

This test verifies the full pipeline with a small subset of frames:
1. GroundingDINO runs only on sampled frames (every 10 frames)
2. Kalman tracker produces tracks for EVERY frame
3. Tracks feed into s05_segment
4. Candidate segments feed into s06_vlm (using stub VLM)

Test uses approximately 100-150 frames from tt6.mp4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import PipelineConfig
from src.context import PipelineContext
from src.stages import s01_ingest, s02_sample, s03_detect, s04_track, s05_segment, s06_vlm


def main():
    print("\n" + "=" * 80)
    print("DINO + KALMAN + VLM INTEGRATION TEST")
    print("=" * 80 + "\n")
    
    # Configuration
    video_path = Path("tt6.mp4")
    output_dir = Path("output_integration_test")
    config_path = Path("config/tt6_groundingdino_kalman.yaml")
    
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}")
        return 1
    
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    # Load config
    from src.config import load_config
    config = load_config(yaml_path=config_path)
    
    # OVERRIDE: Limit to first 150 frames for quick test
    config.max_frames = 150
    
    # OVERRIDE: Enable stub VLM (we're testing infrastructure, not real VLM)
    config.vlm.enabled = True
    config.vlm.backend = "LOCAL_MODEL"
    config.vlm.model_name = "stub_local"
    
    print(f"Video: {video_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Max frames: {config.max_frames}")
    print(f"Detector: {config.detector.backend}")
    print(f"Detection interval: every {config.sample.every_n_frames} frames")
    print(f"Tracker: {config.tracker.backend}")
    print(f"VLM: {config.vlm.backend} ({config.vlm.model_name})")
    print()
    
    # Create context
    ctx = PipelineContext(
        config=config,
        video_path=video_path,
        output_dir=output_dir,
    )
    
    # Stage 01: Ingest
    print("─" * 80)
    print("Stage 01: Ingest")
    print("─" * 80)
    st01 = s01_ingest.run(ctx)
    ctx.record_stage(st01)
    
    if st01.status != "OK":
        print(f"ERROR: {st01.message}")
        return 1
    
    print(f"✓ Ingested video: {ctx.video_metadata.frame_count} frames, "
          f"{ctx.video_metadata.fps:.2f} FPS, "
          f"{ctx.video_metadata.width}x{ctx.video_metadata.height}")
    print(f"  Limited to: {config.max_frames} frames")
    print()
    
    # Stage 02: Sample
    print("─" * 80)
    print("Stage 02: Sample")
    print("─" * 80)
    st02 = s02_sample.run(ctx)
    ctx.record_stage(st02)
    
    if st02.status != "OK":
        print(f"ERROR: {st02.message}")
        return 1
    
    print(f"✓ Sampling: every {config.sample.every_n_frames} frames")
    print(f"  Total frames: {ctx.video_metadata.frame_count}")
    print(f"  Sampled frames: {len(ctx.sampling_plan.frames)}")
    print()
    
    # Stage 03: Detect
    print("─" * 80)
    print("Stage 03: Detect (GroundingDINO - sparse)")
    print("─" * 80)
    st03 = s03_detect.run(ctx)
    ctx.record_stage(st03)
    
    if st03.status != "OK":
        print(f"ERROR: {st03.message}")
        return 1
    
    detection_frames = len(ctx.detection_frames)
    total_detections = sum(len(df.detections) for df in ctx.detection_frames if df.status == "OK")
    
    print(f"✓ Detection complete")
    print(f"  Detection frames: {detection_frames}")
    print(f"  Total detections: {total_detections}")
    print(f"  Detector: {config.detector.backend}")
    print()
    
    # Stage 04: Track
    print("─" * 80)
    print("Stage 04: Track (Kalman - dense)")
    print("─" * 80)
    st04 = s04_track.run(ctx)
    ctx.record_stage(st04)
    
    if st04.status != "OK":
        print(f"ERROR: {st04.message}")
        return 1
    
    # Analyze tracking results
    total_tracks = len(ctx.tracks)
    total_points = sum(len(t.points) for t in ctx.tracks)
    detected_points = sum(1 for t in ctx.tracks for p in t.points if p.detection_confidence > 0)
    estimated_points = total_points - detected_points
    
    # Check frame coverage
    covered_frames = set()
    for track in ctx.tracks:
        for point in track.points:
            covered_frames.add(point.frame_index)
    
    frame_coverage = len(covered_frames)
    expected_frames = min(config.max_frames, ctx.video_metadata.frame_count)
    
    print(f"✓ Tracking complete")
    print(f"  Total tracks: {total_tracks}")
    print(f"  Total track points: {total_points}")
    print(f"  Detected points: {detected_points}")
    print(f"  Estimated points: {estimated_points} ({100 * estimated_points / total_points:.1f}%)")
    print(f"  Frame coverage: {frame_coverage}/{expected_frames} frames")
    print()
    
    # Stage 05: Segment
    print("─" * 80)
    print("Stage 05: Segment")
    print("─" * 80)
    st05 = s05_segment.run(ctx)
    ctx.record_stage(st05)
    
    if st05.status != "OK":
        print(f"ERROR: {st05.message}")
        return 1
    
    print(f"✓ Segmentation complete")
    print(f"  Candidate segments: {len(ctx.candidate_segments)}")
    if ctx.candidate_segments:
        print(f"  Segments:")
        for seg in ctx.candidate_segments[:5]:  # Show first 5
            print(f"    - {seg.segment_id}: frames {seg.start_frame}-{seg.end_frame}, "
                  f"tracks {seg.track_ids}, trigger: {seg.trigger_reason}")
        if len(ctx.candidate_segments) > 5:
            print(f"    ... and {len(ctx.candidate_segments) - 5} more")
    print()
    
    # Stage 06: VLM
    print("─" * 80)
    print("Stage 06: VLM (stub)")
    print("─" * 80)
    st06 = s06_vlm.run(ctx)
    ctx.record_stage(st06)
    
    if st06.status != "OK" and st06.status != "SKIPPED":
        print(f"ERROR: {st06.message}")
        return 1
    
    success_count = sum(1 for o in ctx.vlm_observations if o.status == "SUCCESS")
    
    print(f"✓ VLM analysis complete")
    print(f"  Total observations: {len(ctx.vlm_observations)}")
    print(f"  Success: {success_count}")
    print(f"  Backend: {config.vlm.backend}")
    print()
    
    # Summary
    print("=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print()
    print(f"Frames processed:           {expected_frames}")
    print(f"DINO detection frames:      {detection_frames}")
    print(f"Total detections:           {total_detections}")
    print(f"Total tracks:               {total_tracks}")
    print(f"Tracked frames:             {frame_coverage}")
    print(f"Estimated points:           {estimated_points}")
    print(f"Detected points:            {detected_points}")
    print(f"Candidate segments:         {len(ctx.candidate_segments)}")
    print(f"VLM observations:           {len(ctx.vlm_observations)}")
    print(f"VLM success count:          {success_count}")
    print()
    
    # Validation
    errors = []
    
    if detection_frames == 0:
        errors.append("No detection frames (GroundingDINO failed)")
    
    if total_tracks == 0:
        errors.append("No tracks created (Kalman failed)")
    
    if frame_coverage < expected_frames * 0.8:  # Allow 20% tolerance
        errors.append(f"Insufficient frame coverage: {frame_coverage}/{expected_frames}")
    
    if estimated_points == 0 and detected_points > 0:
        errors.append("No Kalman predictions (tracker not interpolating)")
    
    if len(ctx.candidate_segments) == 0:
        print("WARNING: No candidate segments generated (segmentation heuristics may need tuning)")
    
    if errors:
        print("ERRORS:")
        for error in errors:
            print(f"  ✗ {error}")
        print()
        return 1
    
    print("✓ All validations passed")
    print(f"\nOutput saved to: {output_dir}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
