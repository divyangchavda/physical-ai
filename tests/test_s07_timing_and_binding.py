"""Tests for s07's two run-driven fixes: surface tracks and clause windows.

Both come from the same Kaggle run of tt7 under config/kaggle_tt7_decoy_b.yaml —
one candidate segment covering the whole 6.67s clip, one Gemini answer, two
events. That run said a hand inserted a *picture* of a push chopper into the box,
and gave both of its events the same 6.67s span.

The numbers here are measured from tests/fixtures/tt7_real_detections.json, which
is the detection output of that exact run at commit 071dd95, rather than asserted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.context import PipelineContext
from src.models.track_changepoints import from_fixture, is_inside
from src.schema.detection import BoundingBox
from src.schema.episode import VideoMetadata
from src.schema.segment import CandidateSegment
from src.schema.track import Track, TrackPoint
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus
from src.stages.s07_events import (
    _clause_windows,
    _extract_events_from_vlm_observations,
    _resolve_object_track,
    _surface_track_ids,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "tt7_real_detections.json"
)

# config/kaggle_tt7_decoy_b.yaml, verbatim — same convention as
# test_track_changepoints.py: written out so a config edit fails a test rather
# than silently changing one.
NMS_IOU = 0.45
MIN_HITS = 3
FPS = 30.0


@pytest.fixture(scope="module")
def tt7_tracks():
    return from_fixture(json.loads(FIXTURE.read_text(encoding="utf-8"))["tracks"])


def _track_from_observed(observed) -> Track:
    """Wrap an ObservedTrack as a pipeline Track, all points detection-backed."""
    frames = observed.frames
    return Track(
        track_id=observed.track_id,
        class_name=observed.class_name,
        class_id=0,
        points=[
            TrackPoint(
                frame_index=f,
                timestamp_sec=f / FPS,
                bbox=BoundingBox(
                    x1=observed.boxes[f][0], y1=observed.boxes[f][1],
                    x2=observed.boxes[f][2], y2=observed.boxes[f][3],
                ),
                detection_confidence=0.9,
                tracking_confidence=0.9,
            )
            for f in frames
        ],
        start_frame=frames[0],
        end_frame=frames[-1],
        start_sec=frames[0] / FPS,
        end_sec=frames[-1] / FPS,
        source="fixture",
        is_estimated=False,
    )


# ────────────────────────────────── the measurement the surface rule rests on
def test_only_the_printed_chopper_is_contained_on_every_shared_frame(tt7_tracks):
    """The whole basis for _surface_track_ids, as the numbers rather than a claim.

    If any other pair ever reaches 100%, the "every shared frame" rule is no
    longer discriminating on this clip and the reasoning needs revisiting.
    """
    fractions = {}
    for inner in tt7_tracks:
        for outer in tt7_tracks:
            if inner.track_id == outer.track_id:
                continue
            shared = sorted(set(inner.boxes) & set(outer.boxes))
            if not shared:
                continue
            n = sum(
                1 for f in shared
                if is_inside(inner.boxes[f], outer.boxes[f], NMS_IOU)
            )
            fractions[(inner.track_id, outer.track_id)] = (n, len(shared))

    # The artwork: push chopper#13 inside cardboard box#6, every frame of its life.
    assert fractions[(13, 6)] == (16, 16)
    # The real chopper, against the same box: never.
    assert fractions[(3, 6)] == (0, 12)
    # And nothing else is total.
    total = [k for k, (n, m) in fractions.items() if n == m and m >= MIN_HITS]
    assert total == [(13, 6)]


def test_the_surface_track_is_the_printed_chopper(tt7_tracks):
    tracks = [_track_from_observed(o) for o in tt7_tracks]
    assert _surface_track_ids(tracks, nms_iou=NMS_IOU, min_hits=MIN_HITS) == {13}


def test_a_short_coincidence_cannot_demote_a_track(tt7_tracks):
    """min_hits is the tracker's own standard for how many observations count.

    Raised past the artwork's 16 shared frames, it stops being called a surface —
    which is the guard working, not a bug: with too few shared frames there is no
    evidence either way.
    """
    tracks = [_track_from_observed(o) for o in tt7_tracks]
    assert _surface_track_ids(tracks, nms_iou=NMS_IOU, min_hits=17) == set()


# ─────────────────────────────────────────────── the binding the run got wrong
def test_the_real_chopper_wins_the_label_tie(tt7_tracks):
    """The run's actual failure: two tracks named "push chopper", artwork longer.

    Track 3 is the real chopper (14 observed points, frames 0-42) and track 13 is
    the picture printed on the carton (16 points, frames 99-144). Ties were broken
    by length, so the picture won and the INSERT event pointed at it.
    """
    tracks = [_track_from_observed(o) for o in tt7_tracks]
    by_id = {t.track_id: t for t in tracks}
    assert len(by_id[13].points) > len(by_id[3].points)  # why it used to win

    track_id, label = _resolve_object_track(
        ["push chopper"], tracks,
        person_classes={"person"}, background_classes={"dining table"},
        raw_action="placing the push chopper into the cardboard box",
        surface_track_ids=frozenset({13}),
    )
    assert (track_id, label) == (3, "push chopper")


def test_without_the_surface_set_the_run_s_wrong_answer_is_reproduced(tt7_tracks):
    """States what the bug was, so the fix is not taken on faith."""
    tracks = [_track_from_observed(o) for o in tt7_tracks]
    track_id, _ = _resolve_object_track(
        ["push chopper"], tracks,
        person_classes={"person"}, background_classes={"dining table"},
        raw_action="placing the push chopper into the cardboard box",
    )
    assert track_id == 13


def test_a_surface_track_is_demoted_not_excluded(tt7_tracks):
    """When the picture is the only match, saying so beats saying nothing."""
    tracks = [_track_from_observed(o) for o in tt7_tracks if o.track_id != 3]
    track_id, label = _resolve_object_track(
        ["push chopper"], tracks,
        person_classes={"person"}, background_classes={"dining table"},
        raw_action="placing the push chopper into the cardboard box",
        surface_track_ids=frozenset({13}),
    )
    assert (track_id, label) == (13, "push chopper")


def test_the_label_match_still_outranks_the_surface_penalty(tt7_tracks):
    """A better label match must win even if it is a marking.

    Otherwise the demotion would start overriding what the VLM actually named,
    which is a bigger error than the one it fixes.
    """
    tracks = [_track_from_observed(o) for o in tt7_tracks]
    track_id, label = _resolve_object_track(
        ["cardboard box"], tracks,
        person_classes={"person"}, background_classes={"dining table"},
        raw_action="closing the cardboard box",
        surface_track_ids=frozenset({6}),  # pretend the box is a marking
    )
    assert (track_id, label) == (6, "cardboard box")


# ──────────────────────────────────────────────────── clause windows (item 2)
def test_two_clauses_split_at_the_single_interior_change_point():
    windows = _clause_windows(2, [0.0, 2.5, 6.0], 0.0, 6.0)
    assert windows == [(0.0, 2.5), (2.5, 6.0)]


def test_cuts_are_spread_over_the_change_points_not_over_time():
    """The rule, stated as arithmetic: cut i of n is change point int(i*k/n).

    Six change points, three clauses -> indices 2 and 4. Note the result is NOT
    the two most evenly spaced times, which is the whole distinction from the
    uniform-grid control in tools/tt7_changepoints.py.
    """
    changes = [0.1, 0.2, 0.3, 4.0, 5.0, 5.9]
    windows = _clause_windows(3, changes, 0.0, 6.0)
    assert windows == [(0.0, 0.3), (0.3, 5.0), (5.0, 6.0)]


def test_too_few_change_points_refuses_rather_than_inventing_a_cut():
    assert _clause_windows(3, [2.0], 0.0, 6.0) is None
    assert _clause_windows(2, [], 0.0, 6.0) is None


def test_change_points_on_the_segment_edges_are_not_cuts():
    """A cut at the segment bound would produce a zero-width window."""
    assert _clause_windows(2, [0.0, 6.0], 0.0, 6.0) is None


def test_a_single_clause_never_gets_a_window():
    """One clause keeps whatever the VLM said; geometry is not consulted."""
    assert _clause_windows(1, [1.0, 2.0, 3.0], 0.0, 6.0) is None


def test_duplicate_change_point_times_refuse_rather_than_collapse():
    """Two tracks changing on the same frame must not yield start == end."""
    assert _clause_windows(3, [2.0, 2.0], 0.0, 6.0) is None


def test_the_windows_tile_the_segment_exactly():
    windows = _clause_windows(4, [1.0, 2.0, 3.0, 4.0, 5.0], 0.0, 6.0)
    assert windows[0][0] == 0.0
    assert windows[-1][1] == 6.0
    for (_, end), (start, _) in zip(windows, windows[1:]):
        assert end == start


# ───────────────────────────────────────── the two fixes through the stage
def _tt7_ctx(tt7_tracks, raw_action: str, objects: list[str]) -> PipelineContext:
    """The run's own segment, tracks and metadata, with one observation."""
    config = PipelineConfig(stub_mode=False)
    config.detector.nms_iou = NMS_IOU
    config.tracker.min_hits = MIN_HITS
    config.frame_sampling.every_n_frames = 3
    config.segment.person_classes = ["person"]
    config.segment.background_classes = ["dining table"]

    ctx = PipelineContext(
        config=config,
        video_path=Path("tt7.mp4"),
        output_dir=Path("output"),
    )
    ctx.video_metadata = VideoMetadata(
        file_path="tt7.mp4", duration_sec=200 / FPS, fps=FPS, frame_count=200,
        width=1920, height=1080, file_size_bytes=1,
    )
    ctx.tracks = [_track_from_observed(o) for o in tt7_tracks]
    ctx.candidate_segments = [CandidateSegment(
        segment_id="cand_0000_ece278",
        track_ids=[t.track_id for t in ctx.tracks],
        start_frame=3, end_frame=199,
        start_sec=0.0, end_sec=200 / FPS,
        trigger_reason="proximity_overlap+movement",
    )]
    ctx.vlm_observations = [RawVLMObservation(
        observation_id="obs_001",
        segment_id="cand_0000_ece278",
        status=VLMSegmentStatus.SUCCESS,
        backend="GEMINI",
        model_name="gemini-3.1-flash-lite",
        segment_start_sec=0.0,
        segment_end_sec=200 / FPS,
        actor="person",
        active_hand="BOTH",
        objects=objects,
        raw_action=raw_action,
        start_time_sec=None,
        end_time_sec=None,
        state_change="push chopper is inside the box, box is closed",
        visible_facts="hands move the chopper into the carton, then fold the top",
        inference="packing the chopper",
        uncertainty="the lid is partly occluded",
        confidence=1.0,
    )]
    return ctx


# gemini-3.1-flash-lite's verbatim answer for that run's one segment.
_RUN_2_ANSWER = "placing the push chopper into the cardboard box and closing the lid"


def test_the_run_s_two_events_no_longer_share_one_span(tt7_tracks):
    """The headline: two events, two distinct windows, both from real changes."""
    ctx = _tt7_ctx(tt7_tracks, _RUN_2_ANSWER, ["push chopper", "cardboard box"])
    events = _extract_events_from_vlm_observations(ctx)

    assert len(events) == 2
    assert (events[0].start_sec, events[0].end_sec) != (
        events[1].start_sec, events[1].end_sec
    )
    # Contiguous and tiling the segment, so nothing is dropped or double-counted.
    assert events[0].start_sec == 0.0
    assert events[0].end_sec == events[1].start_sec
    assert events[1].end_sec == pytest.approx(200 / FPS)
    for e in events:
        assert e.attributes["timing_source"] == "CHANGE_POINT"


def test_the_derived_window_does_not_claim_exact_precision(tt7_tracks):
    """A geometry window is not the VLM localising the action.

    timing_precision also has to stay in {EXACT, SEGMENT}: src/schema/episode.py
    and src/schema/interaction_graph.py declare it a Literal, so a third value
    would fail validation in s09 and s12.
    """
    ctx = _tt7_ctx(tt7_tracks, _RUN_2_ANSWER, ["push chopper", "cardboard box"])
    events = _extract_events_from_vlm_observations(ctx)
    for e in events:
        assert e.attributes["timing_precision"] == "SEGMENT"


def test_the_insert_binds_to_the_real_chopper_through_the_stage(tt7_tracks):
    """The run bound object_track_id 13 — the picture. It must be 3."""
    ctx = _tt7_ctx(tt7_tracks, _RUN_2_ANSWER, ["push chopper", "cardboard box"])
    events = _extract_events_from_vlm_observations(ctx)
    insert = [e for e in events if e.action.value == "INSERT"]
    assert len(insert) == 1
    assert insert[0].object_track_id == 3


def test_the_close_still_resolves_nothing_for_the_lid(tt7_tracks):
    """"closing the lid" names a part in no vocabulary. Unresolved is honest, and
    this fix must not start borrowing the other clause's object to fill it."""
    ctx = _tt7_ctx(tt7_tracks, _RUN_2_ANSWER, ["push chopper", "cardboard box"])
    events = _extract_events_from_vlm_observations(ctx)
    close = [e for e in events if e.action.value == "CLOSE"]
    assert len(close) == 1
    assert close[0].object_track_id is None
    assert close[0].attributes["object_label"] is None


def test_without_video_metadata_the_segment_fallback_is_unchanged(tt7_tracks):
    """No fps means no change points, and the old behaviour must survive intact."""
    ctx = _tt7_ctx(tt7_tracks, _RUN_2_ANSWER, ["push chopper", "cardboard box"])
    ctx.video_metadata = None
    events = _extract_events_from_vlm_observations(ctx)
    assert len(events) == 2
    for e in events:
        assert (e.start_sec, e.end_sec) == (0.0, pytest.approx(200 / FPS))
        assert e.attributes["timing_source"] == "SEGMENT"
