"""ByteTrack tracker implementation.

Status: IMPLEMENTED

Integrates ultralytics' BYTETracker while maintaining strict isolation.
Uses deterministic IoU matching to associate tracked states back to original
detections, ensuring zero fabricated data and rejecting ambiguous associations.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from src.interfaces.tracker import ObjectTracker
from src.logging_utils import get_logger
from src.schema.detection import BoundingBox, Detection
from src.schema.track import Track, TrackPoint

logger = get_logger(__name__)


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


def associate_tracks_to_detections(
    tracked_bboxes: list[BoundingBox],
    detections: list[Detection],
    iou_threshold: float = 0.50,
) -> dict[int, Detection]:
    """Map tracked bounding boxes back to original detections using deterministic IoU.

    Avoids double assignments and explicitly rejects ambiguous overlaps.
    """
    if not tracked_bboxes or not detections:
        return {}

    # Compute pairwise IoU
    # Matrix of shape (num_tracks, num_detections)
    iou_matrix = np.zeros((len(tracked_bboxes), len(detections)))
    for t_idx, t_box in enumerate(tracked_bboxes):
        for d_idx, d_box in enumerate(detections):
            iou_matrix[t_idx, d_idx] = compute_iou(t_box, d_box.bbox)

    matches: dict[int, Detection] = {}
    used_detections: set[int] = set()

    # Greedy match highest IoU first
    while True:
        max_iou = float(np.max(iou_matrix))
        if max_iou < iou_threshold:
            break

        # Find indices of max_iou
        t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

        # Check for ambiguity: Is there another detection with very similar IoU for this track?
        # Or another track with very similar IoU for this detection?
        # We consider IoU difference < 0.05 as ambiguous.
        track_ious = iou_matrix[t_idx, :]
        det_ious = iou_matrix[:, d_idx]
        
        # Zero out the current max to find the second best
        track_ious_copy = track_ious.copy()
        track_ious_copy[d_idx] = 0.0
        det_ious_copy = det_ious.copy()
        det_ious_copy[t_idx] = 0.0
        
        ambiguous = False
        if np.max(track_ious_copy) >= max_iou - 0.05:
            ambiguous = True
        if np.max(det_ious_copy) >= max_iou - 0.05:
            ambiguous = True

        if ambiguous:
            # Reject match due to ambiguity for this pair
            iou_matrix[t_idx, :] = 0.0
            iou_matrix[:, d_idx] = 0.0
            continue

        # Valid unambiguous match
        matches[int(t_idx)] = detections[int(d_idx)]
        used_detections.add(int(d_idx))

        # Clear rows and columns for matched elements
        iou_matrix[t_idx, :] = 0.0
        iou_matrix[:, d_idx] = 0.0

    return matches


class MockResultsBoxes:
    """Wrapper to feed pure numpy array detections into ultralytics BYTETracker."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __getitem__(self, idx: Any) -> MockResultsBoxes:
        return MockResultsBoxes(self.xywh[idx], self.conf[idx], self.cls[idx])

    def __len__(self) -> int:
        return len(self.conf)


class ByteTrackTracker(ObjectTracker):
    """ByteTrack-based object tracker."""

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        gmc_method: str = "sparseOptFlow",
        fuse_score: bool = True,
    ) -> None:
        self._tracker = None
        self._args = argparse.Namespace(
            tracker_type="bytetrack",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            gmc_method=gmc_method,
            fuse_score=fuse_score,
        )
        # Active tracks maintained by our orchestrator
        self._active_tracks: dict[int, Track] = {}

    def _setup(self) -> None:
        """Lazily initialize the ultralytics BYTETracker."""
        if self._tracker is not None:
            return

        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics is not installed. Install it with: pip install ultralytics"
            ) from e

        self._tracker = BYTETracker(self._args)

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> list[Track]:
        """Update tracker with new detections and return active tracks."""
        self._setup()

        # Build mock results for BYTETracker
        n = len(detections)
        if n == 0:
            xywh = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros((0,), dtype=np.float32)
            cls = np.zeros((0,), dtype=np.float32)
        else:
            xywh_list = []
            conf_list = []
            cls_list = []
            for d in detections:
                w = d.bbox.width
                h = d.bbox.height
                cx = d.bbox.cx
                cy = d.bbox.cy
                xywh_list.append([cx, cy, w, h])
                conf_list.append(d.confidence)
                cls_list.append(d.class_id)
            xywh = np.array(xywh_list, dtype=np.float32)
            conf = np.array(conf_list, dtype=np.float32)
            cls = np.array(cls_list, dtype=np.float32)

        results = MockResultsBoxes(xywh, conf, cls)

        # ByteTracker needs an image if gmc is enabled, but we just pass a dummy one
        # as we are operating on pre-computed boxes (assuming no camera motion compensation).
        # We also pass a dummy image to avoid None checks inside tracker.
        dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # run tracker
        assert self._tracker is not None
        tracked_stracks = self._tracker.update(results, dummy_img)
        
        if len(tracked_stracks) == 0:
            # If tracker loses all tracks, clear our active memory for next time
            # Wait, our `self._active_tracks` tracks the *current* frame's output.
            # We must only return tracks that are active THIS frame.
            self._active_tracks.clear()
            return []

        # Parse tracked objects
        tracked_bboxes = []
        track_ids = []
        track_confs = []
        for row in tracked_stracks:
            # row is [x1, y1, x2, y2, track_id, conf, cls, idx]
            x1, y1, x2, y2, t_id, score = row[0:6]
            t_box = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            tracked_bboxes.append(t_box)
            track_ids.append(int(t_id))
            track_confs.append(float(score))

        # Associate back to original detections
        matches = associate_tracks_to_detections(tracked_bboxes, detections, iou_threshold=0.5)

        current_active_tracks: dict[int, Track] = {}

        for i, t_box in enumerate(tracked_bboxes):
            t_id = track_ids[i]
            matched_det = matches.get(i)

            if matched_det is None:
                # Track is active but couldn't reliably map to a detection in this frame.
                # In strict schemas, we do not fabricate track points. 
                # We skip updating this track for this frame.
                continue

            # We have a valid match
            tp = TrackPoint(
                frame_index=frame_index,
                timestamp_sec=matched_det.timestamp_sec,
                bbox=matched_det.bbox,
                detection_confidence=matched_det.confidence,
                tracking_confidence=track_confs[i],
            )

            if t_id in self._active_tracks:
                # Existing track
                track = self._active_tracks[t_id]
                track.points.append(tp)
                track.end_frame = frame_index
                track.end_sec = matched_det.timestamp_sec
                # Ensure class stays consistent with its starting identity
                # (We do not overwrite class_name on every frame)
            else:
                # New track
                track = Track(
                    track_id=t_id,
                    class_name=matched_det.class_name,
                    class_id=matched_det.class_id,
                    points=[tp],
                    start_frame=frame_index,
                    end_frame=frame_index,
                    start_sec=matched_det.timestamp_sec,
                    end_sec=matched_det.timestamp_sec,
                    source=self.backend_name,
                    is_estimated=True,
                )
            current_active_tracks[t_id] = track

        self._active_tracks = current_active_tracks
        return list(self._active_tracks.values())

    def reset(self) -> None:
        """Reset internal tracker state."""
        self._tracker = None
        self._active_tracks.clear()

    @property
    def backend_name(self) -> str:
        return "bytetrack"
