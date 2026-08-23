"""Analysis script for Kalman sparse tracking experiment.

Analyzes GroundingDINO sparse detection + Kalman prediction tracking results.

Usage:
    python analyze_kalman_tracking.py output_tt6_dino_kalman/
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


def analyze_kalman_tracking(output_dir: Path) -> None:
    """Analyze Kalman sparse tracking results."""
    
    print("=" * 80)
    print("KALMAN SPARSE TRACKING ANALYSIS")
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
    
    # ── VIDEO INFO ────────────────────────────────────────────────────────────
    print("VIDEO INFORMATION")
    print("-" * 80)
    
    video_fps = sampling_plan.get('video_fps', 'unknown')
    video_frame_count = sampling_plan.get('video_frame_count', 'unknown')
    
    print(f"Total frames:           {video_frame_count}")
    print(f"FPS:                    {video_fps}")
    if isinstance(video_frame_count, int) and isinstance(video_fps, (int, float)):
        duration = video_frame_count / video_fps
        print(f"Duration:               {duration:.2f} seconds")
    print()
    
    # ── DETECTION STATISTICS ──────────────────────────────────────────────────
    print("DETECTION STATISTICS")
    print("-" * 80)
    
    detection_frames = len([df for df in detections_data if df.get('detections')])
    total_detections = sum(len(df.get('detections', [])) for df in detections_data)
    
    detections_by_class = defaultdict(int)
    for df in detections_data:
        for det in df.get('detections', []):
            class_name = det.get('class_name', 'unknown')
            detections_by_class[class_name] += 1
    
    print(f"Detection frames:       {detection_frames}")
    if isinstance(video_frame_count, int):
        det_coverage = (detection_frames / video_frame_count) * 100
        print(f"Detection coverage:     {det_coverage:.1f}%")
    
    print(f"Total detections:       {total_detections}")
    
    if detections_by_class:
        print()
        print("Detections by class:")
        for class_name in sorted(detections_by_class.keys()):
            count = detections_by_class[class_name]
            print(f"  {class_name:20s} {count:5d}")
    print()
    
    # ── TRACKING STATISTICS ───────────────────────────────────────────────────
    print("TRACKING STATISTICS")
    print("-" * 80)
    
    num_tracks = len(tracks_data)
    
    tracks_by_class = defaultdict(int)
    points_by_class = defaultdict(int)
    duration_by_class = defaultdict(list)
    
    all_tracked_frames = set()
    track_durations = []
    
    detected_points = 0
    estimated_points = 0
    
    for track in tracks_data:
        class_name = track.get('class_name', 'unknown')
        tracks_by_class[class_name] += 1
        
        points = track.get('points', [])
        points_by_class[class_name] += len(points)
        
        # Track duration
        start_frame = track.get('start_frame', 0)
        end_frame = track.get('end_frame', 0)
        duration = end_frame - start_frame + 1
        track_durations.append(duration)
        duration_by_class[class_name].append(duration)
        
        # Count detected vs estimated points
        for point in points:
            frame_idx = point.get('frame_index')
            all_tracked_frames.add(frame_idx)
            
            # A point is detected if detection_confidence > 0
            if point.get('detection_confidence', 0.0) > 0.0:
                detected_points += 1
            else:
                estimated_points += 1
    
    print(f"Unique tracks:          {num_tracks}")
    
    tracked_frame_count = len(all_tracked_frames)
    print(f"Tracked frames:         {tracked_frame_count}")
    if isinstance(video_frame_count, int):
        track_coverage = (tracked_frame_count / video_frame_count) * 100
        print(f"Track coverage:         {track_coverage:.1f}%")
    
    if track_durations:
        print()
        print(f"Track duration statistics:")
        print(f"  Min:     {min(track_durations):5d} frames")
        print(f"  Max:     {max(track_durations):5d} frames")
        print(f"  Average: {sum(track_durations) / len(track_durations):5.1f} frames")
        print(f"  Median:  {sorted(track_durations)[len(track_durations) // 2]:5d} frames")
    print()
    
    # ── PER-CLASS STATISTICS ──────────────────────────────────────────────────
    print("PER-CLASS STATISTICS")
    print("-" * 80)
    
    if tracks_by_class:
        print("Track count:")
        for class_name in sorted(tracks_by_class.keys()):
            count = tracks_by_class[class_name]
            print(f"  {class_name:20s} {count:5d} tracks")
        
        print()
        print("Total points:")
        for class_name in sorted(points_by_class.keys()):
            count = points_by_class[class_name]
            print(f"  {class_name:20s} {count:6d} points")
        
        print()
        print("Average duration:")
        for class_name in sorted(duration_by_class.keys()):
            durations = duration_by_class[class_name]
            avg_dur = sum(durations) / len(durations) if durations else 0
            print(f"  {class_name:20s} {avg_dur:6.1f} frames")
    print()
    
    # ── ESTIMATION STATISTICS ─────────────────────────────────────────────────
    print("ESTIMATION STATISTICS")
    print("-" * 80)
    
    total_points = detected_points + estimated_points
    
    print(f"Actual detection points: {detected_points}")
    print(f"Estimated points:        {estimated_points}")
    print(f"Total points:            {total_points}")
    
    if total_points > 0:
        est_pct = (estimated_points / total_points) * 100
        print(f"Percentage estimated:    {est_pct:.1f}%")
    print()
    
    # ── GAP ANALYSIS ──────────────────────────────────────────────────────────
    print("GAP ANALYSIS")
    print("-" * 80)
    
    all_gaps = []
    
    for track in tracks_data:
        points = track.get('points', [])
        if len(points) < 2:
            continue
        
        # Check for gaps
        point_frames = sorted([p['frame_index'] for p in points])
        for i in range(len(point_frames) - 1):
            gap = point_frames[i + 1] - point_frames[i] - 1
            if gap > 0:
                all_gaps.append(gap)
    
    if all_gaps:
        print(f"Number of gaps:          {len(all_gaps)}")
        print(f"Maximum gap:             {max(all_gaps)} frames")
        print(f"Average gap:             {sum(all_gaps) / len(all_gaps):.1f} frames")
        
        # Gap distribution
        gap_buckets = defaultdict(int)
        for gap in all_gaps:
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
            if len(all_gaps) > 0:
                pct = (count / len(all_gaps) * 100)
                print(f"  {bucket:6s} frames: {count:5d} ({pct:5.1f}%)")
    else:
        print("No gaps found (continuous tracking)")
    print()
    
    # ── FRAGMENTATION ANALYSIS ────────────────────────────────────────────────
    print("FRAGMENTATION ANALYSIS")
    print("-" * 80)
    
    if num_tracks > 0 and total_detections > 0:
        fragmentation = num_tracks / total_detections
        print(f"Fragmentation ratio:     {fragmentation:.3f} (tracks per detection)")
        print(f"  < 0.5 = good (multiple detections per track)")
        print(f"  > 1.0 = poor (tracks fragmenting)")
        
        # Per-class fragmentation
        print()
        print("Per-class fragmentation:")
        for class_name in sorted(tracks_by_class.keys()):
            n_tracks = tracks_by_class[class_name]
            n_dets = detections_by_class.get(class_name, 0)
            if n_dets > 0:
                frag = n_tracks / n_dets
                print(f"  {class_name:20s} {frag:.3f} ({n_tracks} tracks / {n_dets} detections)")
    print()
    
    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if isinstance(video_frame_count, int):
        print(f"✓ Video: {video_frame_count} frames @ {video_fps} FPS")
        print(f"✓ Detection: {detection_frames} frames ({det_coverage:.1f}% coverage)")
        print(f"✓ Tracking: {tracked_frame_count} frames ({track_coverage:.1f}% coverage)")
    
    print(f"✓ Detections: {total_detections} total")
    print(f"✓ Tracks: {num_tracks} unique tracks")
    print(f"✓ Points: {total_points} total ({detected_points} detected, {estimated_points} estimated)")
    
    if track_durations:
        avg_dur = sum(track_durations) / len(track_durations)
        print(f"✓ Average track duration: {avg_dur:.1f} frames")
    
    if all_gaps:
        print(f"⚠ Gaps: {len(all_gaps)} gaps found")
    else:
        print(f"✓ No gaps: Continuous tracking")
    
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Kalman sparse tracking experiment results"
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
    
    analyze_kalman_tracking(args.output_dir)


if __name__ == "__main__":
    main()
