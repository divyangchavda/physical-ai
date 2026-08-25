"""Tests for track points the Kalman filter invents after losing an object.

Every box asserted here is transcribed from ``tracks.json`` of the tt7 run on
2026-08-25 (200 frames, 1920x1080, GroundingDINO stride 3). Three of the seven
candidate windows contained a track point that no detector could produce:

    track  2  push chopper   frame  57   [300, 0, 882, 1]
    track 12  push chopper   frame 116   [1119, 1079, 1120, 1080]
    track  6  cardboard box  frame 199   [1919, 0, 1920, 1]

Each is one pixel thick and pinned to a frame edge, and each passed
``BoundingBox._validate_order`` because x2>x1 and y2>y1 by exactly 1. They come
from ``_state_to_bbox``, which pads a collapsed box up to a pixel so the schema
accepts it, and they reach every consumer of ``Track.points`` as fact:

  - the frame-116 box gave an area ratio of 82208x across one window;
  - the frame-57 box is the reason a containment test could not see the
    video's one real INSERT. Ground truth puts it at 1.0-2.0s; frame 57 is
    1.90s, and the chopper being inserted is track 2.

There are two independent causes, not one. Off-frame boxes clamp both of an
axis's coordinates onto the same edge; separately, a predicted extent under one
pixel gets floored by ``max(1.0, w)``. The frame-116 box needs both: its x span
of 1119->1120 sits in the middle of a 1920-wide frame, so nothing clamped it —
the size state had collapsed.
"""
from __future__ import annotations

import pytest

from src.models.kalman_sparse_tracker import (
    KalmanBoxTracker,
    KalmanSparseTracker,
    KalmanTrack,
)
from src.schema.detection import BoundingBox, Detection

FRAME_W, FRAME_H = 1920, 1080


def _kf(cx: float, cy: float, w: float, h: float) -> KalmanBoxTracker:
    """A tracker whose state is set directly, bypassing predict()'s dynamics.

    The state is the input to _state_to_bbox, so setting it is how a specific
    observed output box gets reproduced exactly rather than approximately.
    """
    kf = KalmanBoxTracker(
        BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0), FRAME_W, FRAME_H
    )
    kf.x[0], kf.x[1], kf.x[2], kf.x[3] = cx, cy, w, h
    return kf


def _det(bbox: BoundingBox, frame: int, name: str = "push chopper") -> Detection:
    return Detection(
        detection_id=f"d{frame}",
        frame_index=frame,
        timestamp_sec=frame / 30.0,
        bbox=bbox,
        class_id=0,
        class_name=name,
        confidence=0.9,
        source="test",
    )


def _box(b: BoundingBox) -> list[int]:
    return [round(b.x1), round(b.y1), round(b.x2), round(b.y2)]


# ── The three real boxes, reproduced coordinate for coordinate ───────────────

def test_track_6_frame_199_box_is_flagged():
    """[1919, 0, 1920, 1] — wholly past the right edge and above the top."""
    box, fabricated = _kf(cx=2000.0, cy=-50.0, w=100.0, h=100.0)._state_to_bbox()
    assert _box(box) == [1919, 0, 1920, 1]
    assert fabricated is True


def test_track_2_frame_57_box_is_flagged():
    """[300, 0, 882, 1] — a real 582px width, but wholly above the top edge.

    The x span is untouched, so anything measuring "is this box degenerate" by
    area alone has to pick a threshold. Reading the collapsed axis needs none.
    """
    box, fabricated = _kf(cx=591.0, cy=-50.0, w=582.0, h=50.0)._state_to_bbox()
    assert _box(box) == [300, 0, 882, 1]
    assert fabricated is True


def test_track_12_frame_116_box_is_flagged():
    """[1119, 1079, 1120, 1080] — needs the size-collapse cause, not clamping.

    x runs 1119->1120 inside a 1920-wide frame, so no clamp fired on that axis;
    the width was floored from 0.5 to 1.0. Catching only off-frame boxes would
    let this one through.
    """
    box, fabricated = _kf(cx=1119.5, cy=1200.0, w=0.5, h=0.5)._state_to_bbox()
    assert _box(box) == [1119, 1079, 1120, 1080]
    assert fabricated is True


def test_a_sub_pixel_box_in_open_frame_is_flagged_with_no_clamping_at_all():
    """The size cause is independent of the frame edges."""
    box, fabricated = _kf(cx=960.0, cy=540.0, w=0.5, h=0.5)._state_to_bbox()
    # The floor produces exactly one pixel, centred: 959.5 -> 960.5.
    assert (box.x1, box.y1, box.x2, box.y2) == (959.5, 539.5, 960.5, 540.5)
    assert box.width == 1.0 and box.height == 1.0
    assert fabricated is True


# ── The boxes that must keep working ─────────────────────────────────────────

def test_a_box_in_open_frame_is_not_flagged():
    box, fabricated = _kf(cx=960.0, cy=540.0, w=200.0, h=100.0)._state_to_bbox()
    assert _box(box) == [860, 490, 1060, 590]
    assert fabricated is False


def test_a_box_half_off_the_edge_is_kept():
    """Partly outside is a real observation, clipped. Only wholly outside is not.

    Guarding this because an object at the frame edge is the normal case in tt7
    — the person's track spans [2, 421, 1918, 1077] and similar for all 200
    frames — and dropping those points would gut tracking.
    """
    box, fabricated = _kf(cx=1900.0, cy=540.0, w=200.0, h=100.0)._state_to_bbox()
    assert _box(box) == [1800, 490, 1920, 590]
    assert fabricated is False


def test_a_box_exactly_one_pixel_wide_is_allowed():
    """w == 1.0 does not trip the floor, so it is not evidence of collapse."""
    _, fabricated = _kf(cx=960.0, cy=540.0, w=1.0, h=1.0)._state_to_bbox()
    assert fabricated is False


def test_no_frame_dimensions_means_no_clamping_and_no_off_frame_flag():
    """frame_width/height are optional; the size cause must still work."""
    kf = KalmanBoxTracker(BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0))
    kf.x[0], kf.x[1], kf.x[2], kf.x[3] = 5000.0, 5000.0, 100.0, 100.0
    box, fabricated = kf._state_to_bbox()
    assert _box(box) == [4950, 4950, 5050, 5050]  # not clamped
    assert fabricated is False

    kf.x[2], kf.x[3] = 0.5, 0.5
    _, fabricated = kf._state_to_bbox()
    assert fabricated is True


def test_get_state_still_returns_a_box_for_iou():
    """Association reads get_state(); a collapsed box must not break it.

    It scores ~0 IoU against any detection and so matches nothing, which is the
    right answer for an object that has left frame and is what happened before
    the flag existed.
    """
    kf = _kf(cx=2000.0, cy=-50.0, w=100.0, h=100.0)
    box = kf.get_state()
    assert isinstance(box, BoundingBox)
    assert _box(box) == [1919, 0, 1920, 1]


# ── The point is withheld, and only the point ────────────────────────────────

def _track_at(cx: float, cy: float, w: float, h: float) -> KalmanTrack:
    track = KalmanTrack(
        track_id=1,
        detection=_det(
            BoundingBox(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2),
            frame=0,
        ),
        frame_index=0,
        timestamp_sec=0.0,
        frame_width=FRAME_W,
        frame_height=FRAME_H,
    )
    return track


def test_a_fabricated_prediction_records_no_point():
    track = _track_at(960.0, 540.0, 200.0, 100.0)
    track.kf.x[0], track.kf.x[1] = 3000.0, -500.0  # far off frame
    point = track.predict(frame_index=1, timestamp_sec=1 / 30.0)
    assert point is None
    assert len(track.points) == 1  # only the frame-0 detection


def test_a_real_prediction_still_records_a_point():
    track = _track_at(960.0, 540.0, 200.0, 100.0)
    point = track.predict(frame_index=1, timestamp_sec=1 / 30.0)
    assert point is not None
    assert len(track.points) == 2
    assert track.points[-1].detection_confidence == 0.0  # still a prediction
    assert track.end_frame == 1


def test_ageing_is_unchanged_when_the_point_is_withheld():
    """Withholding the observation must not extend the track's lifetime.

    consecutive_misses is what KalmanSparseTracker.update tests against
    max_unmatched_frames, so it has to tick on every frame either way.
    """
    track = _track_at(960.0, 540.0, 200.0, 100.0)
    track.kf.x[0], track.kf.x[1] = 3000.0, -500.0
    for frame in (1, 2, 3):
        assert track.predict(frame, frame / 30.0) is None
    assert track.consecutive_misses == 3
    assert track.age == 3


def test_the_span_stops_at_the_last_recorded_point():
    """end_frame must equal points[-1].frame_index.

    track_stitcher.py sets end_frame = merged[-1].frame_index after every
    merge, so a tracker that advanced the span onto frames with no point would
    disagree with the stitcher about where a track ends.
    """
    track = _track_at(960.0, 540.0, 200.0, 100.0)
    track.predict(1, 1 / 30.0)
    track.kf.x[0], track.kf.x[1] = 3000.0, -500.0
    track.predict(2, 2 / 30.0)
    track.predict(3, 3 / 30.0)
    assert track.end_frame == track.points[-1].frame_index == 1
    assert track.to_track().end_frame == 1


# ── Through the whole tracker ────────────────────────────────────────────────

def test_a_withheld_prediction_does_not_cost_the_previous_real_point():
    """The pop() in step 3 replaces a provisional prediction, so it must not run
    when there was none — it would delete the last real observation instead.

    Sequence: detect, let the state fly off frame so the prediction is withheld,
    then detect again in a spot the state can still match.
    """
    tracker = KalmanSparseTracker(
        frame_width=FRAME_W, frame_height=FRAME_H, detection_stride=1, min_hits=1
    )
    first = BoundingBox(x1=860.0, y1=490.0, x2=1060.0, y2=590.0)
    tracker.update([_det(first, 0)], frame_index=0)
    assert len(tracker.tracks) == 1
    track = next(iter(tracker.tracks.values()))
    points_before = len(track.points)

    track.kf.x[0], track.kf.x[1] = 3000.0, -500.0  # next predict is fabricated
    tracker.update([_det(first, 1)], frame_index=1)

    # The new detection is unmatched (the off-frame state scores no IoU), so it
    # opens a second track — but the first track keeps every point it had.
    assert len(track.points) == points_before
    assert track.points[0].detection_confidence > 0.0


def test_matched_tracks_still_replace_their_prediction_with_the_detection():
    """The normal path: one point per frame, detection-backed, not two."""
    tracker = KalmanSparseTracker(
        frame_width=FRAME_W, frame_height=FRAME_H, detection_stride=1, min_hits=1
    )
    box = BoundingBox(x1=860.0, y1=490.0, x2=1060.0, y2=590.0)
    for frame in range(4):
        tracker.update([_det(box, frame)], frame_index=frame)

    track = next(iter(tracker.tracks.values()))
    assert len(track.points) == 4
    assert [p.frame_index for p in track.points] == [0, 1, 2, 3]
    assert all(p.detection_confidence > 0.0 for p in track.points)


def test_an_off_frame_track_still_dies_on_the_same_frame_as_before():
    """Deletion is driven by consecutive_misses, which this change leaves alone.

    max_unmatched_frames = max(max_age, stride * (max_missed + 1))
                         = max(15, 1 * (2 + 1)) = 15 with these settings, and
    deletion fires on consecutive_misses > 15. The track is created on frame 0
    with 0 misses, so frame 15 is its last and frame 16 removes it. Verified
    identical on the pre-change tracker: see the module docstring of
    tools/compare_tracker_deletion.py.
    """
    tracker = KalmanSparseTracker(
        frame_width=FRAME_W, frame_height=FRAME_H, detection_stride=1, min_hits=1
    )
    assert tracker.max_unmatched_frames == 15
    box = BoundingBox(x1=860.0, y1=490.0, x2=1060.0, y2=590.0)
    tracker.update([_det(box, 0)], frame_index=0)
    track = next(iter(tracker.tracks.values()))
    track.kf.x[0], track.kf.x[1] = 3000.0, -500.0

    for frame in range(1, 16):
        tracker.update([], frame_index=frame)
        assert tracker.tracks, f"deleted too early, at frame {frame}"
    tracker.update([], frame_index=16)
    assert not tracker.tracks


def test_no_track_point_is_ever_a_one_pixel_box():
    """The property the three real boxes violated, asserted over a full run.

    A detection walks off the right edge of the frame at 200px/frame and is
    then never detected again — the same shape as track 6, which held the
    cardboard box for 194 frames and ended at [1919, 0, 1920, 1].
    """
    tracker = KalmanSparseTracker(
        frame_width=FRAME_W, frame_height=FRAME_H, detection_stride=1, min_hits=1
    )
    for frame in range(6):
        x = 900.0 + frame * 200.0
        tracker.update(
            [_det(BoundingBox(x1=x, y1=490.0, x2=x + 200.0, y2=590.0), frame)],
            frame_index=frame,
        )
    for frame in range(6, 20):
        tracker.update([], frame_index=frame)
        for track in tracker.tracks.values():
            for point in track.points:
                assert point.bbox.width > 1.0, _box(point.bbox)
                assert point.bbox.height > 1.0, _box(point.bbox)
