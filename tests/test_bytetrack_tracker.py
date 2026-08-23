"""Unit tests for ByteTrackTracker.

Validates the tracker orchestration, identity persistence, and
the deterministic bounding-box IoU association logic without needing
real YOLO weights or real videos.
"""
from __future__ import annotations

import pytest

from src.models.bytetrack_tracker import (
    ByteTrackTracker,
    associate_tracks_to_detections,
    compute_iou,
)
from src.schema.detection import BoundingBox, Detection

# ---------------------------------------------------------------------------
# IoU Matching Utility Tests
# ---------------------------------------------------------------------------

def test_compute_iou():
    box1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    box2 = BoundingBox(x1=5, y1=5, x2=15, y2=15)
    # Area1 = 100, Area2 = 100, Inter = 25, Union = 175, IoU = 25/175 = 1/7 = 0.142857
    iou = compute_iou(box1, box2)
    assert pytest.approx(iou, 0.01) == 0.1428

def test_associate_tracks_unambiguous():
    t_boxes = [
        BoundingBox(x1=0, y1=0, x2=10, y2=10),
        BoundingBox(x1=20, y1=20, x2=30, y2=30),
    ]
    dets = [
        Detection(detection_id="d0", frame_index=0, timestamp_sec=0.0, bbox=BoundingBox(x1=21, y1=21, x2=31, y2=31), class_id=0, class_name="A", confidence=0.9, source="test"),
        Detection(detection_id="d1", frame_index=0, timestamp_sec=0.0, bbox=BoundingBox(x1=1, y1=1, x2=11, y2=11), class_id=0, class_name="A", confidence=0.9, source="test"),
    ]
    
    matches = associate_tracks_to_detections(t_boxes, dets, iou_threshold=0.5)
    
    # Track 0 -> Det 1
    # Track 1 -> Det 0
    assert matches[0] == dets[1]
    assert matches[1] == dets[0]

def test_associate_tracks_ambiguous_rejected():
    # Two detections overlap heavily with one track box
    t_boxes = [
        BoundingBox(x1=0, y1=0, x2=10, y2=10),
    ]
    dets = [
        Detection(detection_id="d0", frame_index=0, timestamp_sec=0.0, bbox=BoundingBox(x1=0.1, y1=0.1, x2=10.1, y2=10.1), class_id=0, class_name="A", confidence=0.9, source="test"),
        Detection(detection_id="d1", frame_index=0, timestamp_sec=0.0, bbox=BoundingBox(x1=0.2, y1=0.2, x2=10.2, y2=10.2), class_id=0, class_name="A", confidence=0.9, source="test"),
    ]
    
    matches = associate_tracks_to_detections(t_boxes, dets, iou_threshold=0.5)
    # Due to ambiguity (both IoUs > threshold and very close to each other), the match should be rejected.
    assert len(matches) == 0

# ---------------------------------------------------------------------------
# Tracker Implementation Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker():
    t = ByteTrackTracker(
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.4,
        track_buffer=30,
        match_thresh=0.8,
    )
    yield t
    t.reset()


def make_det(id_str, x, y, class_id=0, class_name="person", conf=0.9):
    return Detection(
        detection_id=id_str,
        frame_index=0,
        timestamp_sec=0.0,
        bbox=BoundingBox(x1=x, y1=y, x2=x+50, y2=y+50),
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        source="test",
    )

def test_tracker_empty_input(tracker):
    active = tracker.update([], frame_index=0)
    assert len(active) == 0

def test_single_object_persistence(tracker):
    # Frame 1
    d1 = make_det("d1", 100, 100)
    tracks1 = tracker.update([d1], frame_index=0)
    assert len(tracks1) == 1
    t_id = tracks1[0].track_id

    # Frame 2 (moves slightly)
    d2 = make_det("d2", 102, 102)
    tracks2 = tracker.update([d2], frame_index=1)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == t_id
    
    # 2 points collected
    assert len(tracks2[0].points) == 2

def test_multiple_objects_same_class(tracker):
    d1_f1 = make_det("objA_1", 100, 100)
    d2_f1 = make_det("objB_1", 300, 300)
    
    t_f1 = tracker.update([d1_f1, d2_f1], frame_index=0)
    assert len(t_f1) == 2
    idA, idB = sorted([t.track_id for t in t_f1])
    
    d1_f2 = make_det("objA_2", 102, 102)
    d2_f2 = make_det("objB_2", 298, 298)
    
    t_f2 = tracker.update([d1_f2, d2_f2], frame_index=1)
    assert len(t_f2) == 2
    assert sorted([t.track_id for t in t_f2]) == [idA, idB]

def test_missing_detection_preserves_id(tracker):
    # Frame 1
    d1 = make_det("obj1", 100, 100)
    t_f1 = tracker.update([d1], frame_index=0)
    track_id = t_f1[0].track_id
    
    # Frame 2: MISSING
    t_f2 = tracker.update([], frame_index=1)
    assert len(t_f2) == 0  # no ACTIVE tracks returned for this frame
    
    # Frame 3: REAPPEARS
    d3 = make_det("obj1_again", 105, 105)
    t_f3 = tracker.update([d3], frame_index=2)
    assert len(t_f3) == 1
    assert t_f3[0].track_id == track_id  # ID preserved

def test_new_object(tracker):
    t_f1 = tracker.update([make_det("a", 100, 100)], frame_index=0)
    id1 = t_f1[0].track_id
    
    # b1 is introduced but unconfirmed (ByteTrack requires it to be matched again)
    t_f2 = tracker.update([make_det("a2", 102, 102), make_det("b1", 500, 500)], frame_index=1)
    assert len(t_f2) == 1
    
    # b1 is matched again, becomes activated
    t_f3 = tracker.update([make_det("a3", 104, 104), make_det("b2", 502, 502)], frame_index=2)
    assert len(t_f3) == 2
    ids = [t.track_id for t in t_f3]
    assert id1 in ids

def test_reset_clears_state(tracker):
    t_f1 = tracker.update([make_det("a", 100, 100)], frame_index=0)
    assert len(t_f1) == 1
    
    tracker.reset()
    
    # Now passing the exact same box should yield a DIFFERENT or new track state
    # (ByteTracker uses global counters internally sometimes, but our state is fresh)
    # The crucial part is that `self._active_tracks` is cleared.
    t_f2 = tracker.update([make_det("a", 100, 100)], frame_index=0)
    assert len(t_f2) == 1
    assert len(t_f2[0].points) == 1  # fresh track
