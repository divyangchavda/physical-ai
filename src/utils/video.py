"""Video utilities for extracting frames and processing media."""
from __future__ import annotations

from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    fps: float
) -> list[tuple[float, bytes]]:
    """Extract JPEG frames from a video segment.
    
    Args:
        video_path: Absolute path to the source video.
        start_sec: Segment start time in seconds.
        end_sec: Segment end time in seconds.
        fps: Target frames per second to extract.
        
    Returns:
        List of tuples: (relative_timestamp_sec, jpeg_bytes)
        The timestamp is relative to start_sec (e.g. 0.0, 0.2, 0.4...).
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    if start_sec >= end_sec:
        raise ValueError(f"start_sec ({start_sec}) must be less than end_sec ({end_sec})")
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0  # Fallback
        
    frames_extracted = []
    
    try:
        # Seek to start
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        
        current_time_sec = start_sec
        step_sec = 1.0 / fps
        
        while current_time_sec <= end_sec:
            # We must manually seek to the exact timestamp because reading sequentially 
            # might not match the target FPS step if video_fps != target fps
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time_sec * 1000.0)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Encode as JPEG
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                relative_ts = current_time_sec - start_sec
                frames_extracted.append((relative_ts, buffer.tobytes()))
                
            current_time_sec += step_sec
            
    finally:
        cap.release()
        
    return frames_extracted
