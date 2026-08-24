"""Tests for post-tracking entity resolution.

The behaviour under test came out of real Kaggle output: tt6 produced 21 chopper
ids, 13 box ids and 5 person ids for a video containing one chopper, one box and
one person. The tracker has no re-identification path, so every detection gap
mints a new id and nothing can ever undo it.

The hardest requirement is the one that stops this becoming a "collapse
everything by class" hack: tt6 is one ~8.3s clip copied four times, so an object
genuinely teleports at each cut and the correct answer is about four entities per
class. A stitcher that returned one would be wrong.

The first version of this module merged 7 fragments out of 40 on the real run,
because its overlap budget was guessed at 2 frames. It should have been read off
the tracker's own config: a dying track keeps emitting predicted points beside
its successor for up to ``max(max_age, stride * (max_missed + 1))`` frames, and
tt6's person fragments overlapped by exactly 15 every time. Two of the tests
below encode that tail, and two more guard the opposite error — genuinely
concurrent objects must survive.
"""
from __future__ import annotations

from src.models.track_stitcher import stitch_tracks
from src.schema.detection import BoundingBox
from src.schema.track import Track, TrackPoint

W, H = 1280, 720


def _track(track_id: int, class_name: str, frames: range, x: float, y: float,
           size: float = 60.0, conf: float = 0.5) -> Track:
    points = [
        TrackPoint(
            frame_index=f,
            timestamp_sec=f / 30.0,
            bbox=BoundingBox(x1=x, y1=y, x2=x + size, y2=y + size),
            detection_confidence=conf,
        )
        for f in frames
    ]
    return Track(
        track_id=track_id, class_name=class_name, points=points,
        start_frame=points[0].frame_index, end_frame=points[-1].frame_index,
        start_sec=points[0].timestamp_sec, end_sec=points[-1].timestamp_sec,
        source="kalman_sparse",
    )


def test_empty_input():
    assert stitch_tracks([], W, H) == ([], {})


def test_a_gap_in_the_same_place_is_stitched():
    """The exact tt6 failure: one object, one dropout, two ids."""
    a = _track(2, "push chopper", range(0, 30, 3), x=400, y=300)
    b = _track(9, "push chopper", range(45, 75, 3), x=410, y=305)

    entities, absorbed = stitch_tracks([a, b], W, H)

    assert len(entities) == 1
    e = entities[0]
    assert e.track_id == 2, "the surviving id must be the lowest original"
    assert absorbed[2] == [2, 9]
    assert e.start_frame == 0 and e.end_frame == 72
    assert len(e.points) == 20
    # Frame order must be restored, since s05 and the direction check both walk
    # points expecting time order.
    assert [p.frame_index for p in e.points] == sorted(p.frame_index for p in e.points)


def test_a_far_jump_is_not_stitched():
    """A hard cut teleports the object. Two entities is the correct answer."""
    a = _track(2, "push chopper", range(0, 30, 3), x=100, y=100)
    b = _track(9, "push chopper", range(36, 66, 3), x=1100, y=600)

    entities, _ = stitch_tracks([a, b], W, H)
    assert len(entities) == 2


def test_a_long_time_gap_is_not_stitched():
    """Same position, but far too long a silence to claim continuity."""
    a = _track(2, "cardboard box", range(0, 30, 3), x=400, y=300)
    b = _track(9, "cardboard box", range(400, 430, 3), x=400, y=300)

    entities, _ = stitch_tracks([a, b], W, H)
    assert len(entities) == 2


def test_the_deletion_lag_tail_is_stitched():
    """The dominant real failure mode, and the one a guessed threshold missed.

    A dying track keeps appending Kalman-predicted points until it is deleted,
    while the unmatched detection at that same frame has already started its
    successor. So siblings overlap by up to the tail bound with no gap at all.
    On tt6 that overlap was exactly 15 frames for all five person fragments,
    and the shipped budget of 2 rejected every one of them.

    Predicted points carry detection_confidence=0.0, so the merge keeps the real
    observation at every shared frame and the ghost is dropped.
    """
    a = _track(3, "person", range(0, 220, 3), x=400, y=300, conf=0.0)
    b = _track(22, "person", range(204, 418, 3), x=402, y=302, conf=0.8)

    entities, absorbed = stitch_tracks([a, b], W, H, max_overlap_frames=18)

    assert len(entities) == 1
    e = entities[0]
    assert absorbed[3] == [3, 22]
    assert e.start_frame == 0 and e.end_frame == 417
    shared = [p for p in e.points if 204 <= p.frame_index <= 219]
    assert shared, "the overlap region must survive the merge"
    assert all(p.detection_confidence == 0.8 for p in shared), (
        "at a shared frame the real detection must beat the predicted ghost"
    )


def test_a_concurrent_duplicate_on_the_same_pixels_is_stitched():
    """Overlap past the tail bound, but the two boxes are the same box.

    tt6 chopper 13 (frames 99-162) sat entirely inside chopper 12 (69-186) —
    63 frames of overlap, far too long to be a deletion tail. Two objects cannot
    occupy the same space, so holding the same pixels for every shared frame is
    what identifies a duplicate.
    """
    a = _track(12, "push chopper", range(69, 187, 3), x=400, y=300)
    b = _track(13, "push chopper", range(99, 163, 3), x=404, y=303)

    entities, absorbed = stitch_tracks([a, b], W, H, max_overlap_frames=18)

    assert len(entities) == 1
    assert absorbed[12] == [12, 13]


def test_concurrent_tracks_far_apart_stay_separate():
    """The guard against collapsing everything by class.

    Same class, fully overlapping in time, but in different places. This is two
    objects and must stay two, which is what stops the duplicate rule above
    from becoming "merge whatever shares a label".
    """
    a = _track(1, "person", range(0, 60, 3), x=120, y=300)
    b = _track(2, "person", range(0, 60, 3), x=1000, y=300)

    entities, _ = stitch_tracks([a, b], W, H, max_overlap_frames=18)
    assert len(entities) == 2


def test_two_objects_crossing_briefly_stay_separate():
    """Passing through each other is not being each other.

    The duplicate rule averages IoU over every shared frame, so a moment of
    high overlap in the middle of a crossing cannot carry the decision.
    """
    a = _track(1, "person", range(0, 60, 3), x=300, y=300)
    b_points = []
    for f in range(0, 60, 3):
        # Walks in from the right, coincides with `a` near frame 30, walks on.
        x = 900 - f * 20
        b_points.append(
            TrackPoint(
                frame_index=f,
                timestamp_sec=f / 30.0,
                bbox=BoundingBox(x1=x, y1=300, x2=x + 60, y2=360),
                detection_confidence=0.5,
            )
        )
    b = Track(
        track_id=2, class_name="person", points=b_points,
        start_frame=0, end_frame=57, start_sec=0.0, end_sec=57 / 30.0,
        source="kalman_sparse",
    )

    entities, _ = stitch_tracks([a, b], W, H, max_overlap_frames=18)
    assert len(entities) == 2


def test_classes_are_never_mixed():
    a = _track(1, "push chopper", range(0, 30, 3), x=400, y=300)
    b = _track(2, "cardboard box", range(45, 75, 3), x=400, y=300)

    entities, _ = stitch_tracks([a, b], W, H)
    assert len(entities) == 2
    assert {e.class_name for e in entities} == {"push chopper", "cardboard box"}


def test_a_chain_of_fragments_collapses_to_one():
    """Six fragments is what tt6 produced for the chopper inside one copy."""
    frags = [
        _track(tid, "push chopper", range(start, start + 15, 3), x=400 + i * 8, y=300)
        for i, (tid, start) in enumerate([(2, 0), (9, 30), (12, 60), (13, 90), (15, 120), (19, 150)])
    ]
    entities, absorbed = stitch_tracks(frags, W, H)

    assert len(entities) == 1
    assert absorbed[2] == [2, 9, 12, 13, 15, 19]


def test_four_copies_produce_four_entities():
    """The acceptance criterion for tt6, and the guard against over-merging.

    Four copies of one clip. Inside a copy the object moves steadily across the
    frame and the tracker drops it repeatedly, so each seam is a short hop. At
    each cut it snaps back to its start position, a long way from where the copy
    left it.

    Note the frame gap at the cut (42) is *smaller* than the gap budget (45), so
    time cannot separate a cut from a dropout here — the geometry has to, which
    is exactly what this asserts. A stitcher returning 1 ignored the cuts; one
    returning 20 did not stitch at all.
    """
    frags: list[Track] = []
    tid = 1
    for copy in range(4):
        base = copy * 249
        # 700 -> 140 in five steps of 140px (0.095 of the diagonal): each seam
        # is close enough to join. IoU is 0 at every seam, so the centre-distance
        # bound is what carries these.
        for i, x in enumerate((700, 560, 420, 280, 140)):
            start = base + i * 45
            frags.append(_track(tid, "push chopper", range(start, start + 30, 3), x=x, y=300))
            tid += 1

    entities, _ = stitch_tracks(frags, W, H)
    assert len(entities) == 4, [
        (e.track_id, e.start_frame, e.end_frame) for e in entities
    ]
    # Every break must sit on a copy boundary, never inside a copy.
    assert sorted(e.start_frame for e in entities) == [0, 249, 498, 747]


def test_a_long_move_across_a_dropout_is_not_stitched():
    """A limitation worth recording, not a bug to hide.

    The seam test is geometric, so if an object moves a long way *while*
    undetected, there is no evidence left that the two fragments are the same
    object, and the stitcher declines. Fragmentation therefore survives exactly
    where motion is fastest — which is where events happen. Closing this needs
    velocity extrapolation or appearance, neither of which is in the tracker.
    """
    a = _track(1, "cardboard box", range(0, 30, 3), x=700, y=300)
    b = _track(2, "cardboard box", range(45, 75, 3), x=150, y=300)

    entities, _ = stitch_tracks([a, b], W, H)
    assert len(entities) == 2


def test_overlapping_seam_keeps_the_confident_point():
    """Fragments can share a frame. Both are real observations of one object."""
    a = _track(1, "person", range(0, 31, 3), x=400, y=300, conf=0.4)
    b = _track(2, "person", range(30, 61, 3), x=402, y=302, conf=0.9)

    entities, absorbed = stitch_tracks([a, b], W, H)

    assert len(entities) == 1
    e = entities[0]
    frames = [p.frame_index for p in e.points]
    assert len(frames) == len(set(frames)), "one point per frame"
    seam = next(p for p in e.points if p.frame_index == 30)
    assert seam.detection_confidence == 0.9


def test_disabled_by_threshold_is_a_no_op():
    """iou 0 and distance 0 can match nothing, so every fragment survives."""
    a = _track(2, "push chopper", range(0, 30, 3), x=400, y=300)
    b = _track(9, "push chopper", range(45, 75, 3), x=400, y=300)

    entities, _ = stitch_tracks(
        [a, b], W, H, iou_threshold=1.01, max_center_dist_norm=-1.0
    )
    assert len(entities) == 2


def test_output_is_deterministic_regardless_of_input_order():
    a = _track(2, "push chopper", range(0, 30, 3), x=400, y=300)
    b = _track(9, "push chopper", range(45, 75, 3), x=410, y=305)
    c = _track(5, "person", range(0, 75, 3), x=100, y=100)

    forward, map_f = stitch_tracks([a, b, c], W, H)
    backward, map_b = stitch_tracks([c, b, a], W, H)

    assert [e.track_id for e in forward] == [e.track_id for e in backward]
    assert map_f == map_b


def test_original_tracks_are_not_mutated():
    """s04 writes tracks.json from the returned list; the input must survive."""
    a = _track(2, "push chopper", range(0, 30, 3), x=400, y=300)
    b = _track(9, "push chopper", range(45, 75, 3), x=410, y=305)
    before = len(a.points)

    stitch_tracks([a, b], W, H)

    assert len(a.points) == before
    assert a.end_frame == 27
