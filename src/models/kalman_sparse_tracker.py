"""Kalman Sparse Tracker — designed for sparse detection + dense prediction.

Status: EXPERIMENTAL

This tracker is designed specifically for scenarios where detections arrive
sparsely (e.g., every 10 frames) but tracking predictions are needed on every frame.

Uses a constant-velocity Kalman filter to predict object positions between
sparse detection frames.

Key features:
- Class-aware IoU-based data association
- Kalman filter for motion prediction
- Handles sparse detection gracefully (does not age out tracks immediately)
- Produces estimated track points on frames without detections
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag

from src.interfaces.tracker import ObjectTracker
from src.logging_utils import get_logger
from src.schema.detection import BoundingBox, Detection
from src.schema.track import Track, TrackPoint

logger = get_logger(__name__)


# Supported class names (filter out malformed labels)
SUPPORTED_CLASSES = {
    "person",
    "cardboard box",
    "push chopper",
    "dining table",
}


def filter_detection(detection: Detection) -> Detection | None:
    """Filter out detections with unsupported class names.
    
    Returns:
        Detection if class is supported, None otherwise
    """
    class_name = detection.class_name.strip().lower()
    if class_name in SUPPORTED_CLASSES:
        return detection
    return None


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ix1 = max(box1.x1, box2.x1)
    iy1 = max(box1.y1, box2.y1)
    ix2 = min(box1.x2, box2.x2)
    iy2 = min(box1.y2, box2.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = box1.area + box2.area - inter
    if union <= 0:
        return 0.0
    return inter / union


class KalmanBoxTracker:
    """Kalman filter for tracking a single bounding box with constant velocity.
    
    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    - cx, cy: center x, y
    - w, h: width, height
    - vx, vy: velocity in x, y
    - vw, vh: velocity in width, height
    """
    
    def __init__(self, bbox: BoundingBox, frame_width: int | None = None, frame_height: int | None = None):
        """Initialize Kalman filter with initial bounding box."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        # State dimension: 8 (cx, cy, w, h, vx, vy, vw, vh)
        # Observation dimension: 4 (cx, cy, w, h)
        
        # Initialize state
        cx = bbox.cx
        cy = bbox.cy
        w = bbox.width
        h = bbox.height
        
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # State covariance matrix
        self.P = np.eye(8, dtype=np.float32)
        self.P[4:, 4:] *= 1000.0  # High uncertainty in velocities initially
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # w += vw
        self.F[3, 7] = 1.0  # h += vh
        
        # Observation matrix (observe position and size, not velocity)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = 1.0  # observe cx
        self.H[1, 1] = 1.0  # observe cy
        self.H[2, 2] = 1.0  # observe w
        self.H[3, 3] = 1.0  # observe h
        
        # Process noise covariance
        q_pos = 1.0    # position noise
        q_size = 10.0  # size noise
        q_vel = 100.0  # velocity noise
        
        self.Q = block_diag(
            q_pos * np.eye(2),   # cx, cy noise
            q_size * np.eye(2),  # w, h noise
            q_vel * np.eye(4)    # velocity noise
        ).astype(np.float32)
        
        # Measurement noise covariance
        r = 10.0
        self.R = r * np.eye(4, dtype=np.float32)
    
    def predict(self) -> BoundingBox:
        """Predict next state and return predicted bounding box."""
        # Predict state: x = F * x
        self.x = self.F @ self.x
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Convert state to bounding box
        return self._state_to_bbox()
    
    def update(self, bbox: BoundingBox) -> None:
        """Update state with new observation (detection)."""
        # Measurement
        z = np.array([bbox.cx, bbox.cy, bbox.width, bbox.height], dtype=np.float32)
        
        # Innovation: y = z - H * x
        y = z - (self.H @ self.x)
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H^T * S^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
    
    def _state_to_bbox(self) -> BoundingBox:
        """Convert Kalman state to bounding box with boundary clamping."""
        cx, cy, w, h = self.x[0], self.x[1], self.x[2], self.x[3]
        
        # Ensure positive width and height
        w = max(1.0, w)
        h = max(1.0, h)
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Clamp to frame boundaries if dimensions are available
        if self.frame_width is not None and self.frame_height is not None:
            # First clamp the coordinates
            x1_clamped = max(0.0, min(x1, float(self.frame_width)))
            y1_clamped = max(0.0, min(y1, float(self.frame_height)))
            x2_clamped = max(0.0, min(x2, float(self.frame_width)))
            y2_clamped = max(0.0, min(y2, float(self.frame_height)))
            
            # Ensure x2 > x1 and y2 > y1 after clamping
            # If box collapses to a line/point, adjust to maintain minimum size
            if x2_clamped <= x1_clamped:
                # If both hit right edge, shift x1 left
                if x1_clamped >= float(self.frame_width) - 1.0:
                    x1_clamped = max(0.0, float(self.frame_width) - 1.0)
                    x2_clamped = float(self.frame_width)
                # If both hit left edge, shift x2 right
                elif x2_clamped <= 1.0:
                    x1_clamped = 0.0
                    x2_clamped = min(1.0, float(self.frame_width))
                else:
                    # Maintain minimum width of 1 pixel
                    x2_clamped = x1_clamped + 1.0
            
            if y2_clamped <= y1_clamped:
                # If both hit bottom edge, shift y1 up
                if y1_clamped >= float(self.frame_height) - 1.0:
                    y1_clamped = max(0.0, float(self.frame_height) - 1.0)
                    y2_clamped = float(self.frame_height)
                # If both hit top edge, shift y2 down
                elif y2_clamped <= 1.0:
                    y1_clamped = 0.0
                    y2_clamped = min(1.0, float(self.frame_height))
                else:
                    # Maintain minimum height of 1 pixel
                    y2_clamped = y1_clamped + 1.0
            
            x1, y1, x2, y2 = x1_clamped, y1_clamped, x2_clamped, y2_clamped
        
        return BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
    
    def get_state(self) -> BoundingBox:
        """Get current state as bounding box without prediction."""
        return self._state_to_bbox()


class KalmanTrack:
    """A single tracked object with Kalman filter."""
    
    def __init__(
        self,
        track_id: int,
        detection: Detection,
        frame_index: int,
        timestamp_sec: float,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ):
        self.track_id = track_id
        self.class_name = detection.class_name
        self.class_id = detection.class_id
        
        # Kalman filter
        self.kf = KalmanBoxTracker(detection.bbox, frame_width, frame_height)
        
        # Track state
        self.last_detection_frame = frame_index
        self.last_update_frame = frame_index
        self.age = 0  # frames since creation
        self.hits = 1  # number of detection matches
        self.consecutive_misses = 0  # frames without detection
        
        # Track history
        self.points: list[TrackPoint] = []
        self.start_frame = frame_index
        self.end_frame = frame_index
        self.start_sec = timestamp_sec
        self.end_sec = timestamp_sec
        
        # Add initial point
        self.points.append(TrackPoint(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            bbox=detection.bbox,
            detection_confidence=detection.confidence,
            tracking_confidence=detection.confidence,
        ))
    
    def predict(self, frame_index: int, timestamp_sec: float) -> TrackPoint:
        """Predict track position for current frame."""
        predicted_bbox = self.kf.predict()
        
        self.age += 1
        self.consecutive_misses += 1
        self.last_update_frame = frame_index
        self.end_frame = frame_index
        self.end_sec = timestamp_sec
        
        # Create predicted track point
        point = TrackPoint(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            bbox=predicted_bbox,
            detection_confidence=0.0,  # No detection
            tracking_confidence=max(0.0, 1.0 - self.consecutive_misses * 0.05),  # Decay confidence
        )
        
        self.points.append(point)
        return point
    
    def update(self, detection: Detection, frame_index: int, timestamp_sec: float) -> TrackPoint:
        """Update track with new detection."""
        self.kf.update(detection.bbox)
        
        self.age += 1
        self.hits += 1
        self.consecutive_misses = 0
        self.last_detection_frame = frame_index
        self.last_update_frame = frame_index
        self.end_frame = frame_index
        self.end_sec = timestamp_sec
        
        # Create detection-based track point
        point = TrackPoint(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            bbox=detection.bbox,
            detection_confidence=detection.confidence,
            tracking_confidence=1.0,
        )
        
        self.points.append(point)
        return point
    
    def to_track(self) -> Track:
        """Convert to Track schema."""
        return Track(
            track_id=self.track_id,
            class_name=self.class_name,
            class_id=self.class_id,
            points=self.points,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            source="kalman_sparse",
            is_estimated=True,
        )


class KalmanSparseTracker(ObjectTracker):
    """Kalman-based tracker designed for sparse detections with dense predictions."""
    
    def __init__(
        self,
        iou_threshold: float = 0.20,
        max_age: int = 15,
        min_hits: int = 1,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.tracks: dict[int, KalmanTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        logger.info(
            "KalmanSparseTracker initialized: iou_threshold=%.2f, max_age=%d, min_hits=%d, frame_dims=%sx%s",
            iou_threshold, max_age, min_hits, frame_width, frame_height
        )
    
    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Track]:
        """Update tracker with detections (or empty list) for current frame."""
        self.frame_count += 1
        
        # Filter detections to only supported classes
        filtered_detections = [d for d in detections if filter_detection(d) is not None]
        
        if len(detections) != len(filtered_detections):
            logger.debug(
                "Filtered %d/%d detections (unsupported classes)",
                len(detections) - len(filtered_detections), len(detections)
            )
        
        # Calculate timestamp
        # Note: We don't have FPS here, so we use frame_index as proxy
        # In real pipeline, timestamp_sec should come from video metadata
        timestamp_sec = float(frame_index) / 30.0  # Assume 30 FPS
        
        # Step 1: Predict all existing tracks
        predicted_tracks = {}
        for track_id, track in self.tracks.items():
            track.predict(frame_index, timestamp_sec)
            predicted_tracks[track_id] = track
        
        # Step 2: Associate detections with tracks (if any detections)
        if filtered_detections:
            matched_tracks, unmatched_detections = self._associate(
                predicted_tracks, filtered_detections
            )
            
            # Step 3: Update matched tracks
            for track_id, detection in matched_tracks.items():
                track = self.tracks[track_id]
                # Remove the predicted point we just added
                track.points.pop()
                # Add the updated point instead
                track.update(detection, frame_index, timestamp_sec)
            
            # Step 4: Create new tracks for unmatched detections
            for detection in unmatched_detections:
                new_track = KalmanTrack(
                    track_id=self.next_track_id,
                    detection=detection,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
        
        # Step 5: Delete old tracks
        tracks_to_delete = []
        for track_id, track in self.tracks.items():
            if track.consecutive_misses > self.max_age:
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
        
        # Step 6: Return tracks that meet min_hits threshold
        active_tracks = [
            track.to_track()
            for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]
        
        return active_tracks
    
    def _associate(
        self,
        tracks: dict[int, KalmanTrack],
        detections: list[Detection],
    ) -> tuple[dict[int, Detection], list[Detection]]:
        """Associate detections with tracks using class-aware IoU matching.
        
        Returns:
            (matched_tracks, unmatched_detections)
            matched_tracks: dict[track_id] = detection
            unmatched_detections: list of unmatched Detection objects
        """
        if not tracks or not detections:
            return {}, detections
        
        # Build IoU matrix (only for same class)
        track_ids = list(tracks.keys())
        iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        
        for t_idx, track_id in enumerate(track_ids):
            track = tracks[track_id]
            track_bbox = track.kf.get_state()
            
            for d_idx, detection in enumerate(detections):
                # Only compute IoU for same class
                if track.class_name.lower() == detection.class_name.lower():
                    iou = compute_iou(track_bbox, detection.bbox)
                    iou_matrix[t_idx, d_idx] = iou
        
        # Greedy matching: highest IoU first
        matched_tracks: dict[int, Detection] = {}
        matched_detection_indices: set[int] = set()
        
        while True:
            # Find maximum IoU
            max_iou = float(np.max(iou_matrix))
            if max_iou < self.iou_threshold:
                break
            
            # Find indices of maximum IoU
            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            t_idx, d_idx = int(t_idx), int(d_idx)
            
            # Match
            track_id = track_ids[t_idx]
            detection = detections[d_idx]
            matched_tracks[track_id] = detection
            matched_detection_indices.add(d_idx)
            
            # Zero out matched row and column
            iou_matrix[t_idx, :] = 0.0
            iou_matrix[:, d_idx] = 0.0
        
        # Unmatched detections
        unmatched_detections = [
            d for i, d in enumerate(detections)
            if i not in matched_detection_indices
        ]
        
        return matched_tracks, unmatched_detections
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
    
    @property
    def backend_name(self) -> str:
        return "kalman_sparse"
