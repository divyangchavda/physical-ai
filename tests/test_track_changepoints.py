"""Tests for geometric change-point detection.

The module under test exists so events can be timed by track geometry instead of
inheriting a candidate segment's bounds. Its whole claim is that no number in it
is a judgement — every threshold falls out of a config value — so the tests here
check the derivations as well as the behaviour, and pin the twelve change points
tt7 actually produces so a future edit is a local, testable change rather than a
GPU run.

Two bugs found while writing these are pinned below:

* ``displacement_ratio`` originally normalised a 2-D distance by the box
  diagonal, which is 1.41x the width of a square box, so the effective threshold
  became 0.54 box-widths against a derivation that says 0.379.
* ``_flips`` discarded runs shorter than min_hits without merging the survivors,
  so a single-frame jump between two long still stretches emitted a still ->
  still "change".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.track_changepoints import (
    APPEAR,
    DISAPPEAR,
    ENCLOSE_END,
    ENCLOSE_START,
    MOVE_START,
    MOVE_STOP,
    ObservedTrack,
    containment_changes,
    displacement_ratio,
    find_change_points,
    from_fixture,
    from_track_dicts,
    is_inside,
    life_changes,
    motion_changes,
    still_displacement_ratio,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "tt7_real_detections.json"
)

# config/kaggle_tt7_decoy_b.yaml, verbatim. Written out rather than loaded so a
# config edit shows up as a failing test rather than a silently different test.
NMS_IOU = 0.45
STRIDE = 3
MIN_HITS = 3
FPS = 30.0


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = (x2 - x1) * (y2 - y1) if x2 > x1 and y2 > y1 else 0.0
    area = lambda z: (z[2] - z[0]) * (z[3] - z[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)


# ────────────────────────────────────────────────── the derivation, not the code
def test_the_still_ratio_is_exactly_the_offset_that_reaches_nms_iou():
    """The claim the module rests on, checked against a real IoU computation.

    If two equal boxes offset by ``still_displacement_ratio(t)`` widths do not
    have IoU == t, then the algebra in the module docstring is wrong and every
    motion verdict is measured against the wrong number.
    """
    for target in (0.45, 0.20, 0.50, 0.75):
        ratio = still_displacement_ratio(target)
        w, h = 100.0, 50.0
        a = (0.0, 0.0, w, h)
        b = (ratio * w, 0.0, ratio * w + w, h)
        assert _iou(a, b) == pytest.approx(target, abs=1e-12)


def test_the_still_ratio_rejects_an_impossible_nms_iou():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            still_displacement_ratio(bad)


def test_the_configured_nms_iou_gives_the_documented_numbers():
    """0.379 box-widths per stride step, 0.126 per frame at stride 3."""
    ratio = still_displacement_ratio(NMS_IOU)
    assert ratio == pytest.approx(0.3793, abs=5e-5)
    assert ratio / STRIDE == pytest.approx(0.1264, abs=5e-5)


def test_displacement_is_measured_per_axis_not_by_the_diagonal():
    """The regression: a diagonal normaliser silently moves the threshold 1.41x.

    A square box shifted by exactly its own width along x is 1.0 in box-widths.
    Normalised by the diagonal it would read 0.707, which passes a 0.379 test as
    "not moved" for a full-width jump.
    """
    a = (0.0, 0.0, 100.0, 100.0)
    assert displacement_ratio(a, (100.0, 0.0, 200.0, 100.0)) == pytest.approx(1.0)
    # y uses the height, so a box twice as wide as it is tall is not penalised
    # for vertical motion.
    wide = (0.0, 0.0, 200.0, 100.0)
    assert displacement_ratio(wide, (0.0, 100.0, 200.0, 200.0)) == pytest.approx(1.0)
    # The larger axis decides.
    assert displacement_ratio(a, (10.0, 60.0, 110.0, 160.0)) == pytest.approx(0.6)


def test_a_degenerate_box_does_not_divide_by_zero():
    flat = (5.0, 5.0, 5.0, 5.0)
    assert displacement_ratio(flat, (9.0, 9.0, 9.0, 9.0)) == 0.0


# ───────────────────────────────────────────────────────────────── containment
def test_exact_containment_needs_no_threshold():
    outer = (0.0, 0.0, 100.0, 100.0)
    inner = (40.0, 40.0, 60.0, 60.0)
    assert is_inside(inner, outer, NMS_IOU)
    assert not is_inside(outer, inner, NMS_IOU)


def test_a_box_relabelled_as_its_own_contents_is_not_containment():
    """The guard. tt7's baseline vocabulary produced four of these.

    A "chopper" detection that is really the carton sits at essentially the same
    coordinates, so it passes an exact-containment test for no physical reason.
    IoU at or above nms_iou means NMS considered them one detection.
    """
    outer = (0.0, 0.0, 100.0, 100.0)
    same = (1.0, 1.0, 99.0, 99.0)
    assert _iou(same, outer) >= NMS_IOU
    assert not is_inside(same, outer, NMS_IOU)


def test_a_partial_overlap_is_not_containment():
    assert not is_inside((50.0, 50.0, 150.0, 150.0), (0.0, 0.0, 100.0, 100.0), NMS_IOU)


def test_a_separate_box_is_not_containment():
    assert not is_inside((200.0, 200.0, 210.0, 210.0), (0.0, 0.0, 100.0, 100.0), NMS_IOU)


def test_a_zero_area_box_is_never_inside_anything():
    assert not is_inside((5.0, 5.0, 5.0, 5.0), (0.0, 0.0, 100.0, 100.0), NMS_IOU)


def test_containment_reports_the_frame_the_state_changed():
    outer = ObservedTrack(1, "box", {f: (0.0, 0.0, 300.0, 300.0) for f in range(0, 30, 3)})
    # Outside for four observations, then inside for six.
    boxes = {}
    for f in range(0, 30, 3):
        boxes[f] = (500.0, 500.0, 520.0, 520.0) if f < 12 else (100.0, 100.0, 120.0, 120.0)
    inner = ObservedTrack(2, "chopper", boxes)
    points = containment_changes(inner, outer, nms_iou=NMS_IOU, min_hits=MIN_HITS, fps=FPS)
    assert [(p.frame, p.kind) for p in points] == [(12, ENCLOSE_START)]
    assert points[0].other_track_id == 1


def test_containment_needs_shared_observed_frames():
    """Comparing an observed box against an interpolated one is the discredited
    measurement this module was built to make impossible."""
    a = ObservedTrack(1, "box", {0: (0.0, 0.0, 300.0, 300.0)})
    b = ObservedTrack(2, "chopper", {3: (10.0, 10.0, 20.0, 20.0)})
    assert containment_changes(a, b, nms_iou=NMS_IOU, min_hits=MIN_HITS, fps=FPS) == []


# ──────────────────────────────────────────────────────────────────── lifetime
def test_appear_and_disappear_sit_on_observed_frames():
    track = ObservedTrack(3, "chopper", {0: (0.0, 0.0, 10.0, 10.0),
                                         3: (0.0, 0.0, 10.0, 10.0),
                                         42: (0.0, 0.0, 10.0, 10.0)})
    points = life_changes(track, fps=FPS)
    assert [(p.frame, p.kind) for p in points] == [(0, APPEAR), (42, DISAPPEAR)]
    assert points[1].sec == pytest.approx(1.4)


def test_an_empty_track_reports_nothing():
    assert life_changes(ObservedTrack(9, "ghost", {}), fps=FPS) == []


# ────────────────────────────────────────────────────────────────────── motion
def _straight_line(step: float, n: int = 8) -> ObservedTrack:
    """A 100x100 box moving *step* pixels in x every stride frames."""
    return ObservedTrack(
        1, "thing",
        {i * STRIDE: (i * step, 0.0, i * step + 100.0, 100.0) for i in range(n)},
    )


def test_a_still_track_reports_no_motion_change():
    assert motion_changes(
        _straight_line(0.0), nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS
    ) == []


def test_a_track_that_starts_moving_reports_move_start():
    budget = still_displacement_ratio(NMS_IOU)
    fast = 100.0 * budget * 1.5  # comfortably over, in box-widths
    boxes = {}
    x = 0.0
    for i in range(8):
        boxes[i * STRIDE] = (x, 0.0, x + 100.0, 100.0)
        if i >= 3:
            x += fast
    points = motion_changes(
        ObservedTrack(1, "thing", boxes),
        nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS,
    )
    assert [p.kind for p in points] == [MOVE_START]
    assert points[0].frame == 4 * STRIDE


def test_a_track_that_stops_reports_move_stop():
    budget = still_displacement_ratio(NMS_IOU)
    fast = 100.0 * budget * 1.5
    boxes, x = {}, 0.0
    for i in range(8):
        boxes[i * STRIDE] = (x, 0.0, x + 100.0, 100.0)
        if i < 4:
            x += fast
    points = motion_changes(
        ObservedTrack(1, "thing", boxes),
        nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS,
    )
    assert [p.kind for p in points] == [MOVE_STOP]


def test_a_single_frame_jump_is_not_a_state_change():
    """The debounce regression.

    Discarding runs shorter than min_hits can leave two survivors in the SAME
    state — still, then still — and the first version emitted a change point for
    that, reporting a state change in exactly the case the debounce exists to
    suppress.
    """
    budget = still_displacement_ratio(NMS_IOU)
    jump = 100.0 * budget * 2.0
    boxes = {}
    for i in range(8):
        x = 0.0 if i < 4 else jump
        boxes[i * STRIDE] = (x, 0.0, x + 100.0, 100.0)
    assert motion_changes(
        ObservedTrack(1, "thing", boxes),
        nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS,
    ) == []


def test_a_track_moving_from_its_first_frame_reports_no_move_start():
    """APPEAR already says the object arrived; MOVE_START would double-count."""
    budget = still_displacement_ratio(NMS_IOU)
    points = motion_changes(
        _straight_line(100.0 * budget * 1.5),
        nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS,
    )
    assert points == []


def test_an_unequal_gap_is_scored_per_frame_not_per_pair():
    """A missed detection doubles the gap; the same speed must read the same.

    At stride 3 only 67 of tt7's 200 frames reach the detector and the observed
    gaps are not all 3, so a per-pair budget would call a normal-speed object
    fast whenever a detection was dropped.
    """
    budget = still_displacement_ratio(NMS_IOU)
    step = 100.0 * budget * 0.5  # half the budget per stride step
    # Same speed throughout, but one gap is 6 frames instead of 3.
    boxes = {0: (0.0, 0.0, 100.0, 100.0),
             3: (step, 0.0, step + 100.0, 100.0),
             9: (step * 3, 0.0, step * 3 + 100.0, 100.0),
             12: (step * 4, 0.0, step * 4 + 100.0, 100.0)}
    assert motion_changes(
        ObservedTrack(1, "thing", boxes),
        nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS,
    ) == []


def test_a_track_with_one_observation_reports_no_motion():
    assert motion_changes(
        ObservedTrack(1, "thing", {0: (0.0, 0.0, 10.0, 10.0)}),
        nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS,
    ) == []


# ──────────────────────────────────────────────────────────────────── adapters
def test_the_track_dict_adapter_drops_interpolated_points():
    """Kalman extrapolations carry detection_confidence 0 and must not be read.

    Two thirds of Track.points is extrapolation at stride 3.
    """
    tracks = [{
        "track_id": 4, "class_name": "box",
        "points": [
            {"frame_index": 0, "detection_confidence": 0.5,
             "bbox": {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}},
            {"frame_index": 1, "detection_confidence": 0.0,
             "bbox": {"x1": 9.0, "y1": 9.0, "x2": 9.0, "y2": 9.0}},
            {"frame_index": 2, "detection_confidence": None,
             "bbox": {"x1": 9.0, "y1": 9.0, "x2": 9.0, "y2": 9.0}},
        ],
    }]
    got = from_track_dicts(tracks)
    assert len(got) == 1
    assert got[0].boxes == {0: (1.0, 2.0, 3.0, 4.0)}


def test_the_fixture_adapter_reads_the_flat_rows():
    got = from_fixture([{"track_id": 7, "class_name": "thing",
                         "real": [[3, 1.0, 2.0, 3.0, 4.0]]}])
    assert got[0].boxes == {3: (1.0, 2.0, 3.0, 4.0)}
    assert got[0].frames == [3]


# ───────────────────────────────────────────────── what tt7 actually produces
def test_background_classes_are_excluded_by_the_caller_not_by_this_module():
    """A dining table spans the clip and everything is inside it — true, useless.

    The decision lives in segment.background_classes, so the module must honour
    the argument and must not hardcode a class name.
    """
    tracks = from_fixture(json.loads(FIXTURE.read_text(encoding="utf-8"))["tracks"])
    kwargs = dict(nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS)
    with_table = find_change_points(tracks, **kwargs)
    without = find_change_points(
        tracks, exclude_classes=frozenset({"Dining Table"}), **kwargs
    )
    assert len(without) < len(with_table)
    assert not any(
        "dining table" in (p.class_name.lower(), (p.other_class_name or "").lower())
        for p in without
    )


def test_the_tt7_change_points_are_the_twelve_measured():
    """Pin the whole set, so any edit to this module is a local testable change.

    From tests/fixtures/tt7_real_detections.json at commit 071dd95 with
    config/kaggle_tt7_decoy_b.yaml. Note what is NOT here: no MOVE_START and no
    MOVE_STOP. Nothing in tt7 moves fast enough for long enough to trip the
    NMS-derived budget, which is measured in the module docstring. If a future
    change makes motion points appear on tt7 without a config change, the budget
    or the debounce has been quietly loosened.
    """
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    points = find_change_points(
        from_fixture(fixture["tracks"]),
        nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS,
        exclude_classes=frozenset({"dining table"}),
    )
    assert [(p.frame, p.kind, p.class_name, p.other_track_id) for p in points] == [
        (0, APPEAR, "person", None),
        (0, APPEAR, "push chopper", None),
        (6, APPEAR, "cardboard box", None),
        (21, ENCLOSE_START, "cardboard box", 2),
        (24, ENCLOSE_START, "push chopper", 2),
        (36, ENCLOSE_END, "push chopper", 2),
        (39, ENCLOSE_END, "cardboard box", 2),
        (42, DISAPPEAR, "push chopper", None),
        (99, APPEAR, "push chopper", None),
        (144, DISAPPEAR, "push chopper", None),
        # Both end on the clip's last sampled frame; the sort key breaks the tie
        # on track id, and person is track 2 against the carton's 6.
        (198, DISAPPEAR, "person", None),
        (198, DISAPPEAR, "cardboard box", None),
    ]


def test_tt7_produces_no_motion_change_points():
    """Stated separately because it is the module's main negative result.

    Reported in the module docstring with the measured rates: the fastest
    per-axis rate is 0.176/frame against a budget of 0.126, and it lasts one
    observed pair where min_hits demands three.
    """
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    points = find_change_points(
        from_fixture(fixture["tracks"]),
        nms_iou=NMS_IOU, stride=STRIDE, min_hits=MIN_HITS, fps=FPS,
    )
    assert not [p for p in points if p.kind in (MOVE_START, MOVE_STOP)]
