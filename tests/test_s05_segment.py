"""Unit tests for Candidate Interaction Segment Detection (Stage 05)."""
from __future__ import annotations

import pytest

from src.config import SegmentConfig, SegmentMovementConfig, SegmentProximityConfig
from src.models.heuristic_segmenter import (
    check_movement,
    check_proximity,
    generate_candidate_segments,
)
from src.schema.detection import BoundingBox
from src.schema.track import Track, TrackPoint


@pytest.fixture
def segment_config():
    return SegmentConfig(
        person_classes=["person"],
        proximity=SegmentProximityConfig(iou_threshold=0.05, gap_threshold_normalized=0.2),
        movement=SegmentMovementConfig(threshold=0.05, window_frames=5),
        temporal_padding_sec=2.0,
        merge_gap_sec=1.0,
    )


# --- 1. Proximity Tests (Overlap, Gap, Distant) ---

def test_proximity_overlapping():
    b1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    b2 = BoundingBox(x1=50, y1=50, x2=150, y2=150)
    ok, info = check_proximity(b1, b2, 0.05, 0.2, 1920, 1080)
    assert ok
    assert info["proximity_type"] == "overlap"


def test_proximity_nearby_gap():
    # Gap is 50px horizontally (100 to 150).
    # 50 / 1920 = 0.026 < 0.2, should be TRUE
    b1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    b2 = BoundingBox(x1=150, y1=0, x2=250, y2=100)
    ok, info = check_proximity(b1, b2, 0.05, 0.2, 1920, 1080)
    assert ok
    assert info["proximity_type"] == "gap"
    assert info["gap_normalized"] > 0


def test_proximity_distant_gap():
    # Gap is 1000px horizontally. 1000/1920 = 0.52 > 0.2, should be FALSE
    b1 = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    b2 = BoundingBox(x1=1100, y1=0, x2=1200, y2=100)
    ok, info = check_proximity(b1, b2, 0.05, 0.2, 1920, 1080)
    assert not ok
    assert not info


# --- 2. Movement Tests ---

def _make_track(id: int, points: list[tuple[int, float, float]], cls="person"):
    track_points = []
    for f, cx, cy in points:
        # Construct box centered at cx, cy with size 10x10
        b = BoundingBox(x1=cx-5, y1=cy-5, x2=cx+5, y2=cy+5)
        track_points.append(TrackPoint(frame_index=f, timestamp_sec=f*0.1, bbox=b, detection_confidence=0.9, tracking_confidence=0.9))
    
    return Track(
        track_id=id, class_name=cls, class_id=0, points=track_points,
        start_frame=points[0][0] if points else 0,
        end_frame=points[-1][0] if points else 0,
        start_sec=points[0][1] if points else 0.0,
        end_sec=points[-1][1] if points else 0.0,
        source="test", is_estimated=True
    )


def test_movement_stationary():
    # 0px movement
    t = _make_track(1, [(0, 100, 100), (1, 100, 100), (2, 100, 100)])
    moved, _disp = check_movement(t, current_frame=2, window_frames=5, threshold_norm=0.05, frame_height=1000)
    assert not moved


def test_movement_moving():
    # Moves from 100 to 200 vertically in 2 frames (disp=100)
    # 100/1000 = 0.1 > 0.05, should be TRUE
    t = _make_track(1, [(0, 100, 100), (1, 100, 150), (2, 100, 200)])
    moved, _disp = check_movement(t, current_frame=2, window_frames=5, threshold_norm=0.05, frame_height=1000)
    assert moved
    assert _disp == 0.1


def test_movement_zero_duration():
    t = _make_track(1, [(0, 100, 100)])
    moved, _disp = check_movement(t, 0, 5, 0.05, 1000)
    assert not moved


def test_movement_missing_points():
    t = _make_track(1, [(0, 100, 100), (10, 100, 200)])
    # At frame 10, window=5 looks back to frame 5. The earliest point in window is frame 10 itself!
    # So displacement over window is 0.
    moved, _disp = check_movement(t, 10, 5, 0.05, 1000)
    assert not moved


# --- 3. Segmentation Heuristic Tests ---

def test_person_alone(segment_config):
    t1 = _make_track(1, [(0, 100, 100), (1, 100, 200)], "person")
    segs = generate_candidate_segments([t1], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 0


def test_object_alone(segment_config):
    t1 = _make_track(1, [(0, 100, 100), (1, 100, 200)], "bottle")
    segs = generate_candidate_segments([t1], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 0


def test_person_far_from_object(segment_config):
    p = _make_track(1, [(0, 0, 0), (1, 100, 100)], "person")
    o = _make_track(2, [(0, 1000, 1000), (1, 1100, 1100)], "bottle")
    segs = generate_candidate_segments([p, o], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 0


def test_stationary_person_object(segment_config):
    p = _make_track(1, [(0, 100, 100), (1, 100, 100)], "person")
    o = _make_track(2, [(0, 120, 120), (1, 120, 120)], "bottle")
    segs = generate_candidate_segments([p, o], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 0


def test_moving_person_near_object(segment_config):
    # Person moves vertically by 100px (100/1080 = 0.09 > 0.05)
    # Object is stationary nearby
    p = _make_track(1, [(0, 100, 100), (1, 100, 150), (2, 100, 200)], "person")
    o = _make_track(2, [(0, 120, 120), (1, 120, 120), (2, 120, 120)], "bottle")
    segs = generate_candidate_segments([p, o], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 1
    assert 1 in segs[0].track_ids and 2 in segs[0].track_ids
    assert "proximity_" in segs[0].trigger_reason


def test_moving_object_near_person(segment_config):
    p = _make_track(1, [(0, 100, 100), (1, 100, 100), (2, 100, 100)], "person")
    o = _make_track(2, [(0, 120, 120), (1, 120, 150), (2, 120, 200)], "bottle")
    segs = generate_candidate_segments([p, o], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 1


def test_both_moving(segment_config):
    p = _make_track(1, [(0, 100, 100), (1, 100, 150), (2, 100, 200)], "person")
    o = _make_track(2, [(0, 120, 120), (1, 120, 150), (2, 120, 200)], "bottle")
    segs = generate_candidate_segments([p, o], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 1


def test_multiple_people_and_objects(segment_config):
    p1 = _make_track(1, [(0, 100, 100), (1, 100, 200)], "person")
    p2 = _make_track(2, [(0, 500, 500), (1, 500, 600)], "person")
    o1 = _make_track(3, [(0, 120, 120), (1, 120, 120)], "bottle")
    o2 = _make_track(4, [(0, 520, 520), (1, 520, 520)], "cup")
    
    segs = generate_candidate_segments([p1, p2, o1, o2], segment_config, 1920, 1080, 10.0)
    # p1 moves near o1, p2 moves near o2 at the SAME time.
    # We should get a segment spanning frame 1 with ALL involved tracks (1, 3, 2, 4) merged or overlapping.
    assert len(segs) == 1
    assert set(segs[0].track_ids) == {1, 2, 3, 4}


def test_temporal_padding_and_merging(segment_config):
    # Hit at frame 1 (t=0.1) -> padding [0, 2.1]
    # Hit at frame 10 (t=1.0) -> padding [0, 3.0] (merged with first)
    # Hit at frame 50 (t=5.0) -> padding [3.0, 7.0] 
    # Gap between 3.0 and 3.0 is 0 <= merge_gap(1.0) -> merged into [0, 7.0]
    p = _make_track(1, [
        (0, 100, 100), (1, 100, 200),     # hit at f=1 (t=0.1)
        (9, 100, 200), (10, 100, 300),    # hit at f=10 (t=1.0)
        (49, 100, 300), (50, 100, 400)    # hit at f=50 (t=5.0)
    ], "person")
    o = _make_track(2, [
        (0, 100, 100), (1, 100, 200), 
        (9, 100, 200), (10, 100, 300), 
        (49, 100, 300), (50, 100, 400)
    ], "bottle")
    
    segs = generate_candidate_segments([p, o], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 1
    assert segs[0].start_sec == 0.0 # Clamped to 0
    assert segs[0].end_sec == 7.0


def test_merge_gap_boundary(segment_config):
    # Hit at t=1.0 -> padding [-1.0, 3.0] clamped to [0, 3.0]
    # Hit at t=7.0 -> padding [5.0, 9.0]
    # Gap = 5.0 - 3.0 = 2.0 > merge_gap(1.0). Should NOT merge.
    p = _make_track(1, [(8,100,100), (10,100,200), (68,100,200), (70,100,300)], "person")
    o = _make_track(2, [(8,100,100), (10,100,200), (68,100,200), (70,100,300)], "bottle")
    # p moves at f=10(t=1.0) and f=70(t=7.0)
    
    segs = generate_candidate_segments([p, o], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 2
    assert segs[0].start_sec == 0.0
    assert segs[0].end_sec == 3.0
    assert segs[1].start_sec == 5.0
    assert segs[1].end_sec == 9.0


def test_invalid_timestamps(segment_config):
    # Tracks without points
    t = Track(track_id=1, class_name="person", class_id=0, points=[], start_frame=0, end_frame=0, start_sec=0, end_sec=0, source="test", is_estimated=False)
    segs = generate_candidate_segments([t], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 0


def test_same_class_objects(segment_config):
    # Two bottles moving near each other should NOT trigger if no person is involved
    o1 = _make_track(1, [(0, 100, 100), (1, 100, 200)], "bottle")
    o2 = _make_track(2, [(0, 120, 120), (1, 120, 120)], "bottle")
    segs = generate_candidate_segments([o1, o2], segment_config, 1920, 1080, 10.0)
    assert len(segs) == 0
