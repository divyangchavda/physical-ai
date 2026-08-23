"""Analysis script for sparse detection + dense tracking experiment.

Analyzes the output of GroundingDINO sparse detection with ByteTrack 
interpolation to evaluate tracking coverage and quality.

Usage:
    python analyze_sparse_tracking.py output_tt6_dino_sparse/
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_json(path: Path) -> dict | list:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_sparse_tracking(output_dir: Path) -> None:
    """Analyze sparse detection + dense tracking results."""
    
    print("=" * 80)
    print("SPARSE DETECTION + DENSE TRACKING ANALYSIS")
    print("=" * 80)
    print()
    
    # Load files
    try:
        detections_path = output_dir / "detections.json"
        tracks_path = output_dir / "tracks.json"
        sampling_plan_path = output_dir / "sampling_plan.json"
        
        if not detections_path.exists():
            print(f"ERROR: {detections_path} not found")
            return
        if not tracks_path.exists():
            print(f"ERROR: {tracks_path} not found")
            return
        if not sampling_plan_path.exists():
            print(f"ERROR: {sampling_plan_path} not found")
            return
        
        detections_data = load_json(detections_path)
        tracks_data = load_json(tracks_path)
        sampling_plan = load_json(sampling_plan_path)
        
    except Exception as e:
        print(f"ERROR loading files: {e}")
        return
    
    # ── Video & Sampling Info ────────────────────────────────────────────────
    print("VIDEO & SAMPLING INFORMATION")
    print("-" * 80)
    
    video_fps = sampling_plan.get('video_fps', 'unknown')
    video_frame_count = sampling_plan.get('video_frame_count', 'unknown')
    n_frames_sampled = sampling_plan.get('n_frames_sampled', len(detections_data))
    
    print(f"Video FPS:              {video_fps}")
    print(f"Total video frames:     {video_frame_count}")
    print(f"Detection frames:       {n_frames_sampled}")
    
    if isinstance(video_frame_count, int) and isinstance(n_frames_sampled, int):
        detection_coverage = (n_frames_sampled / video_frame_count) * 100
        interpolation_frames = video_frame_count - n_frames_sampled
        interpolation_ratio = (interpolation_frames / video_frame_count) * 100
        print(f"Detection coverage:     {detection_coverage:.1f}%")
        print(f"Interpolation frames:   {interpolation_frames} ({interpolation_ratio:.1f}%)")
    print()
    
    # ── Detection Statistics ─────────────────────────────────────────────────
    print("DETECTION STATISTICS")
    print("-" * 80)
    
    total_detections = 0
    detections_by_class = defaultdict(int)
    detection_frames_with_objects = 0
    
    for df in detections_data:
        detections = df.get('detections', [])
        if detections:
            detection_frames_with_objects += 1
            total_detections += len(detections)
            for det in detections:
                class_name = det.get('class_name', 'unknown')
                detections_by_class[class_name] += 1
    
    print(f"Total detections:       {total_detections}")
    print(f"Frames with detections: {detection_frames_with_objects}/{n_frames_sampled}")
    if n_frames_sampled > 0:
        avg_det_per_frame = total_detections / n_frames_sampled
        print(f"Avg detections/frame:   {avg_det_per_frame:.2f}")
    
    if detections_by_class:
        print()
        print("Detections by class:")
        for class_name in sorted(detections_by_class.keys()):
            count = detections_by_class[class_name]
            print(f"  {class_name:20s} {count:5d}")
    print()
    
    # ── Track Statistics ─────────────────────────────────────────────────────
    print("TRACK STATISTICS")
    print("-" * 80)
    
    num_tracks = len(tracks_data)
    tracks_by_class = defaultdict(int)
    track_points_by_class = defaultdict(int)
    track_durations = []
    track_gaps = []
    
    frames_with_tracks = set()
    
    for track in tracks_data:
        class_name = track.get('class_name', 'unknown')
        tracks_by_class[class_name] += 1
        
        points = track.get('points', [])
        track_points_by_class[class_name] += len(points)
        
        if points:
            # Track duration in frames
            start_frame = track.get('start_frame', 0)
            end_frame = track.get('end_frame', 0)
            duration = end_frame - start_frame + 1
            track_durations.append(duration)
            
            # Check for gaps in track points
            point_frames = sorted([p['frame_index'] for p in points])
            for i in range(len(point_frames) - 1):
                gap = point_frames[i + 1] - point_frames[i] - 1
                if gap > 0:
                    track_gaps.append(gap)
            
            # Collect all frames with track points
            frames_with_tracks.update(point_frames)
    
    print(f"Unique tracks:          {num_tracks}")
    
    if tracks_by_class:
        print()
        print("Tracks by class:")
        for class_name in sorted(tracks_by_class.keys()):
            count = tracks_by_class[class_name]
            points = track_points_by_class[class_name]
            avg_points = points / count if count > 0 else 0
            print(f"  {class_name:20s} {count:5d} tracks, {points:6d} points ({avg_points:.1f} avg)")
    
    print()
    if track_durations:
        print(f"Track duration stats:")
        print(f"  Min:     {min(track_durations):5d} frames")
        print(f"  Max:     {max(track_durations):5d} frames")
        print(f"  Average: {sum(track_durations) / len(track_durations):5.1f} frames")
        print(f"  Median:  {sorted(track_durations)[len(track_durations) // 2]:5d} frames")
    
    print()
    tracked_frame_count = len(frames_with_tracks)
    print(f"Tracked frames:         {tracked_frame_count}")
    if isinstance(video_frame_count, int):
        track_coverage = (tracked_frame_count / video_frame_count) * 100
        print(f"Track coverage:         {track_coverage:.1f}%")
    
    # ── Gap Analysis ─────────────────────────────────────────────────────────
    print()
    print("GAP ANALYSIS (frames without track points within tracks)")
    print("-" * 80)
    
    if track_gaps:
        total_gaps = len(track_gaps)
        total_gap_frames = sum(track_gaps)
        print(f"Number of gaps:         {total_gaps}")
        print(f"Total gap frames:       {total_gap_frames}")
        print(f"Min gap:                {min(track_gaps)} frames")
        print(f"Max gap:                {max(track_gaps)} frames")
        print(f"Average gap:            {sum(track_gaps) / len(track_gaps):.1f} frames")
        
        # Gap size distribution
        gap_buckets = defaultdict(int)
        for gap in track_gaps:
            if gap <= 5:
                gap_buckets['1-5'] += 1
            elif gap <= 10:
                gap_buckets['6-10'] += 1
            elif gap <= 20:
                gap_buckets['11-20'] += 1
            else:
                gap_buckets['>20'] += 1
        
        print()
        print("Gap size distribution:")
        for bucket in ['1-5', '6-10', '11-20', '>20']:
            count = gap_buckets.get(bucket, 0)
            pct = (count / total_gaps * 100) if total_gaps > 0 else 0
            print(f"  {bucket:6s} frames: {count:5d} ({pct:5.1f}%)")
    else:
        print("No gaps found (continuous tracking)")
    
    # ── Fragmentation Analysis ───────────────────────────────────────────────
    print()
    print("FRAGMENTATION ANALYSIS")
    print("-" * 80)
    
    if num_tracks > 0 and total_detections > 0:
        fragmentation = num_tracks / total_detections
        print(f"Fragmentation ratio:    {fragmentation:.3f} (tracks per detection)")
        print(f"  < 0.5 = good (objects tracked across multiple detections)")
        print(f"  > 1.0 = poor (tracks fragmenting more than detections)")
    
    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if isinstance(video_frame_count, int):
        print(f"✓ Video: {video_frame_count} frames @ {video_fps} FPS")
        print(f"✓ Detection: {n_frames_sampled} frames ({detection_coverage:.1f}% coverage)")
        print(f"✓ Tracking: {tracked_frame_count} frames ({track_coverage:.1f}% coverage)")
        print(f"✓ Interpolation: {interpolation_ratio:.1f}% of frames")
    
    print(f"✓ Detections: {total_detections} total")
    print(f"✓ Tracks: {num_tracks} unique tracks")
    
    if track_gaps:
        print(f"⚠ Gaps: {len(track_gaps)} gaps in tracks ({sum(track_gaps)} frames total)")
    else:
        print(f"✓ No gaps: Continuous tracking")
    
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze sparse detection + dense tracking experiment results"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory containing detections.json and tracks.json"
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"ERROR: Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    analyze_sparse_tracking(args.output_dir)


if __name__ == "__main__":
    main()
