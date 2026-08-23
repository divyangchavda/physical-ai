"""Visualization tool for tracking results validation.

Generates an annotated video showing:
- Track bounding boxes
- Track IDs and class names
- Detection vs estimated points (different colors)
- Confidence/status information

Usage:
    python visualize_tracking.py <video_path> <output_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# Color schemes (BGR format for OpenCV)
COLOR_DETECTED = (0, 255, 0)      # Green for detected points
COLOR_ESTIMATED = (0, 165, 255)   # Orange for estimated points
COLOR_TEXT_BG = (0, 0, 0)         # Black background for text
COLOR_TEXT = (255, 255, 255)      # White text

# Class-specific colors for track boxes
CLASS_COLORS = {
    'person': (255, 0, 0),           # Blue
    'cardboard box': (0, 255, 255),  # Yellow
    'push chopper': (255, 0, 255),   # Magenta
    'dining table': (0, 128, 255),   # Orange
}


def load_json(path: Path) -> dict | list:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def draw_bbox(frame, bbox, color, thickness=2):
    """Draw bounding box on frame."""
    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_text_with_background(frame, text, position, font_scale=0.6, thickness=2):
    """Draw text with black background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # Draw background rectangle
    cv2.rectangle(frame, (x, y - text_height - baseline), 
                  (x + text_width, y + baseline), COLOR_TEXT_BG, -1)
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, COLOR_TEXT, thickness)


def analyze_tracks(tracks_data, video_frame_count):
    """Analyze track data and generate statistics."""
    
    stats = {
        'total_tracks': len(tracks_data),
        'tracks_by_class': defaultdict(int),
        'durations': [],
        'long_tracks': [],  # >100 frames
        'full_video_tracks': [],  # Span entire video
        'detected_points': 0,
        'estimated_points': 0,
        'suspicious_boxes': [],
        'fragmentation_by_class': defaultdict(lambda: {'tracks': 0, 'total_points': 0}),
    }
    
    for track in tracks_data:
        track_id = track['track_id']
        class_name = track['class_name']
        points = track['points']
        
        start_frame = track['start_frame']
        end_frame = track['end_frame']
        duration = end_frame - start_frame + 1
        
        stats['tracks_by_class'][class_name] += 1
        stats['durations'].append(duration)
        stats['fragmentation_by_class'][class_name]['tracks'] += 1
        stats['fragmentation_by_class'][class_name]['total_points'] += len(points)
        
        # Long tracks
        if duration > 100:
            stats['long_tracks'].append({
                'track_id': track_id,
                'class': class_name,
                'duration': duration,
                'start': start_frame,
                'end': end_frame,
            })
        
        # Full video tracks
        if duration == video_frame_count:
            stats['full_video_tracks'].append({
                'track_id': track_id,
                'class': class_name,
                'duration': duration,
            })
        
        # Count detected vs estimated
        for point in points:
            det_conf = point.get('detection_confidence', 0.0)
            if det_conf > 0.0:
                stats['detected_points'] += 1
            else:
                stats['estimated_points'] += 1
            
            # Check for suspicious bounding boxes
            bbox = point['bbox']
            if bbox['x1'] < 0 or bbox['y1'] < 0:
                stats['suspicious_boxes'].append({
                    'track_id': track_id,
                    'frame': point['frame_index'],
                    'reason': 'negative coordinates',
                    'bbox': bbox,
                })
            if bbox['x2'] <= bbox['x1'] or bbox['y2'] <= bbox['y1']:
                stats['suspicious_boxes'].append({
                    'track_id': track_id,
                    'frame': point['frame_index'],
                    'reason': 'invalid dimensions',
                    'bbox': bbox,
                })
    
    return stats


def print_analysis_report(stats):
    """Print analysis report."""
    print()
    print("=" * 80)
    print("TRACKING VALIDATION REPORT")
    print("=" * 80)
    print()
    
    print("TRACK STATISTICS")
    print("-" * 80)
    print(f"Total tracks:            {stats['total_tracks']}")
    
    if stats['durations']:
        durations = stats['durations']
        print(f"Average track duration:  {sum(durations) / len(durations):.1f} frames")
        print(f"Median track duration:   {sorted(durations)[len(durations) // 2]} frames")
        print(f"Max track duration:      {max(durations)} frames")
        print(f"Min track duration:      {min(durations)} frames")
    
    print()
    print("TRACKS BY CLASS")
    print("-" * 80)
    for class_name in sorted(stats['tracks_by_class'].keys()):
        count = stats['tracks_by_class'][class_name]
        print(f"  {class_name:20s} {count:5d} tracks")
    
    print()
    print("LONG TRACKS (>100 frames)")
    print("-" * 80)
    if stats['long_tracks']:
        print(f"Found {len(stats['long_tracks'])} tracks spanning >100 frames:")
        for track_info in stats['long_tracks'][:10]:  # Show first 10
            print(f"  Track {track_info['track_id']:3d} ({track_info['class']:20s}): "
                  f"{track_info['duration']:4d} frames (frame {track_info['start']}-{track_info['end']})")
        if len(stats['long_tracks']) > 10:
            print(f"  ... and {len(stats['long_tracks']) - 10} more")
    else:
        print("  No tracks >100 frames")
    
    print()
    print("FULL VIDEO TRACKS")
    print("-" * 80)
    if stats['full_video_tracks']:
        print(f"Found {len(stats['full_video_tracks'])} track(s) spanning entire video:")
        for track_info in stats['full_video_tracks']:
            print(f"  Track {track_info['track_id']:3d} ({track_info['class']:20s}): "
                  f"{track_info['duration']} frames")
    else:
        print("  No tracks span entire video")
    
    print()
    print("DETECTED vs ESTIMATED POINTS")
    print("-" * 80)
    detected = stats['detected_points']
    estimated = stats['estimated_points']
    total = detected + estimated
    
    print(f"Detected points:         {detected:6d} ({detected/total*100:.1f}%)")
    print(f"Estimated points:        {estimated:6d} ({estimated/total*100:.1f}%)")
    print(f"Total points:            {total:6d}")
    
    print()
    print("FRAGMENTATION BY CLASS")
    print("-" * 80)
    for class_name in sorted(stats['fragmentation_by_class'].keys()):
        data = stats['fragmentation_by_class'][class_name]
        tracks = data['tracks']
        points = data['total_points']
        avg_points = points / tracks if tracks > 0 else 0
        print(f"  {class_name:20s} {tracks:3d} tracks, {points:6d} points ({avg_points:.1f} pts/track)")
    
    print()
    print("SUSPICIOUS BOUNDING BOXES")
    print("-" * 80)
    if stats['suspicious_boxes']:
        print(f"Found {len(stats['suspicious_boxes'])} suspicious bounding boxes:")
        for issue in stats['suspicious_boxes'][:5]:  # Show first 5
            print(f"  Track {issue['track_id']} frame {issue['frame']}: {issue['reason']}")
            print(f"    bbox: {issue['bbox']}")
        if len(stats['suspicious_boxes']) > 5:
            print(f"  ... and {len(stats['suspicious_boxes']) - 5} more")
    else:
        print("  No suspicious bounding boxes detected")
    
    print()
    print("=" * 80)
    print()


def create_visualization(video_path, output_dir, output_name="dino_kalman_tracking_visualization.mp4"):
    """Create visualization video."""
    
    print(f"Loading data from {output_dir}...")
    
    # Load data
    tracks_path = output_dir / "tracks.json"
    detections_path = output_dir / "detections.json"
    
    if not tracks_path.exists():
        print(f"ERROR: {tracks_path} not found")
        return None
    
    tracks_data = load_json(tracks_path)
    print(f"Loaded {len(tracks_data)} tracks")
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps:.2f} FPS, {frame_count} frames")
    
    # Analyze tracks
    print("Analyzing tracks...")
    stats = analyze_tracks(tracks_data, frame_count)
    print_analysis_report(stats)
    
    # Build frame index for tracks
    print("Building frame index...")
    frame_tracks = defaultdict(list)
    
    for track in tracks_data:
        for point in track['points']:
            frame_idx = point['frame_index']
            frame_tracks[frame_idx].append({
                'track_id': track['track_id'],
                'class_name': track['class_name'],
                'bbox': point['bbox'],
                'detection_confidence': point.get('detection_confidence', 0.0),
                'tracking_confidence': point.get('tracking_confidence', 1.0),
            })
    
    # Setup output video
    output_path = output_dir / output_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Creating visualization video: {output_path}")
    print(f"Processing {frame_count} frames...")
    
    frame_idx = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get tracks for this frame
        tracks_in_frame = frame_tracks.get(frame_idx, [])
        
        # Draw tracks
        for track_info in tracks_in_frame:
            bbox = track_info['bbox']
            track_id = track_info['track_id']
            class_name = track_info['class_name']
            det_conf = track_info['detection_confidence']
            
            # Determine color based on detection vs estimation
            if det_conf > 0.0:
                color = COLOR_DETECTED
                status = "DETECTED"
                thickness = 3
            else:
                color = COLOR_ESTIMATED
                status = "ESTIMATED"
                thickness = 2
            
            # Draw bounding box
            draw_bbox(frame, bbox, color, thickness)
            
            # Draw label
            label = f"T{track_id} {class_name} {status}"
            label_pos = (int(bbox['x1']), int(bbox['y1']) - 5)
            draw_text_with_background(frame, label, label_pos, font_scale=0.5, thickness=1)
        
        # Draw frame info
        info_text = f"Frame: {frame_idx}/{frame_count-1} | Tracks: {len(tracks_in_frame)}"
        draw_text_with_background(frame, info_text, (10, 30), font_scale=0.7, thickness=2)
        
        # Draw legend
        legend_y = 70
        draw_text_with_background(frame, "DETECTED", (10, legend_y), font_scale=0.6, thickness=2)
        cv2.rectangle(frame, (150, legend_y - 15), (180, legend_y - 5), COLOR_DETECTED, -1)
        
        legend_y += 30
        draw_text_with_background(frame, "ESTIMATED", (10, legend_y), font_scale=0.6, thickness=2)
        cv2.rectangle(frame, (150, legend_y - 15), (180, legend_y - 5), COLOR_ESTIMATED, -1)
        
        # Write frame
        out.write(frame)
        
        frame_idx += 1
        processed_frames += 1
        
        if processed_frames % 100 == 0:
            progress = (processed_frames / frame_count) * 100
            print(f"  Progress: {processed_frames}/{frame_count} ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n✓ Visualization complete: {output_path}")
    print(f"  Frames processed: {processed_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create tracking visualization video"
    )
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to input video file"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory containing tracks.json"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="dino_kalman_tracking_visualization.mp4",
        help="Output video filename"
    )
    
    args = parser.parse_args()
    
    if not args.video_path.exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)
    
    if not args.output_dir.exists():
        print(f"ERROR: Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    output_path = create_visualization(args.video_path, args.output_dir, args.output_name)
    
    if output_path:
        print()
        print("=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)
        print(f"Output: {output_path}")
        print(f"View with: ffplay {output_path}")
        print("=" * 80)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
