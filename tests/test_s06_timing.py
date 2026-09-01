"""Tests for s06's handling of VLM offsets that do not fit their clip.

Every number here is transcribed from a real Kaggle run: tt7 at
``segment.max_segment_duration_sec=1.0``, which split the 6.67s video into seven
0.95s windows. Five of the seven came back from gemini-3.1-flash-lite with
``end_time_sec: 2.0`` — an offset more than twice the length of the clip it
describes — because the prompt asks for "float offset from segment start"
without ever saying how long the segment is.

Those five were recorded FAILED and their ``raw_action`` discarded, so a run
that analysed seven segments emitted two events. The action text was intact in
every one; only a metadata field was wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.context import PipelineContext
from src.schema.segment import CandidateSegment
from src.schema.vlm import VLMSegmentStatus
from src.stages import s06_vlm
from src.stages.s06_vlm import _absolute_timing


# The five verbatim (segment_start, segment_end) windows that failed, paired
# with the relative end offset Gemini returned for each. Kept as evidence.
_REAL_FAILURES = [
    (0.0, 0.9523809523809524, 2.0),
    (1.9047619047619049, 2.857142857142857, 2.0),
    (3.8095238095238098, 4.761904761904762, 2.0),
    (4.761904761904762, 5.714285714285714, 2.0),
    (5.714285714285714, 6.666666666666667, 2.0),
]


def test_the_five_real_failures_keep_the_observation():
    """None of these may cost more than the timing."""
    for seg_start, seg_end, rel_end in _REAL_FAILURES:
        record = {"start_time_sec": 0.0, "end_time_sec": rel_end}
        start, end, warning = _absolute_timing(record, seg_start, seg_end)
        assert start is None and end is None
        assert warning is not None
        # The warning has to carry the numbers, or the next person debugging
        # this reads "timing dropped" and learns nothing. At full precision:
        # rounded to 2 decimals the tt7 whole-clip rejection printed
        # "end_time_sec=6.67 outside segment [0.00, 6.67]", which reads as a
        # contradiction because the 0.0033s that caused it was rounded away.
        assert "end_time_sec" in warning
        assert f"{seg_start:.6f}" in warning


def test_offsets_inside_the_segment_are_untouched():
    """The common case must convert relative to absolute exactly as before."""
    record = {"start_time_sec": 0.5, "end_time_sec": 1.25}
    start, end, warning = _absolute_timing(record, 42.0, 47.0)
    assert (start, end) == (42.5, 43.25)
    assert warning is None


def test_a_start_offset_past_the_segment_is_dropped_too():
    record = {"start_time_sec": 9.0, "end_time_sec": 0.5}
    start, end, warning = _absolute_timing(record, 0.0, 1.0)
    assert start is None and end is None
    assert "start_time_sec" in warning


def test_both_bounds_go_when_only_one_is_bad():
    """An end past the clip discredits the start that came with it.

    Clamping end to the segment edge would invent a boundary the model never
    reported; keeping the start alone would present half a fabricated interval
    as localisation. s07 reports SEGMENT precision for None, which is true.
    """
    record = {"start_time_sec": 0.1, "end_time_sec": 2.0}
    start, end, _ = _absolute_timing(record, 0.0, 0.95)
    assert start is None
    assert end is None


def test_an_end_before_its_start_is_dropped():
    record = {"start_time_sec": 0.8, "end_time_sec": 0.2}
    start, end, warning = _absolute_timing(record, 0.0, 1.0)
    assert start is None and end is None
    assert ">" in warning


def test_null_offsets_are_not_a_warning():
    """The model declining to time the action is allowed by the prompt."""
    record = {"start_time_sec": None, "end_time_sec": None}
    assert _absolute_timing(record, 0.0, 1.0) == (None, None, None)


def test_a_non_numeric_offset_costs_only_the_timing():
    """"UNKNOWN" in a float field used to raise and fail the segment."""
    record = {"start_time_sec": "UNKNOWN", "end_time_sec": "UNKNOWN"}
    start, end, warning = _absolute_timing(record, 0.0, 1.0)
    assert start is None and end is None
    assert "UNKNOWN" in warning


def test_one_missing_bound_still_converts_the_other():
    record = {"start_time_sec": 0.5, "end_time_sec": None}
    start, end, warning = _absolute_timing(record, 10.0, 12.0)
    assert (start, end) == (10.5, None)
    assert warning is None


# ── The sub-frame overshoot ──────────────────────────────────────────────────
#
# tt7's whole-clip segment is [0.0, 6.666666666666667] — 200 frames at 30fps.
# _render_prompt tells the model the clip is "6.67" seconds long, so the model
# answered 6.67 and the answer was rejected for being 0.0033s past the end. It
# was the correct answer, thrown away by the rounding in the question.

_TT7_FPS = 30.0
_TT7_SEG_END = 200 / _TT7_FPS  # 6.666666666666667, s01's own arithmetic


def test_the_real_tt7_whole_clip_answer_is_no_longer_rejected():
    """The verbatim run-2 case: end_time_sec 6.67 on a 6.666666666666667s clip."""
    record = {"start_time_sec": 0.0, "end_time_sec": 6.67}
    start, end, warning = _absolute_timing(record, 0.0, _TT7_SEG_END, _TT7_FPS)
    assert start == 0.0
    assert end == _TT7_SEG_END          # snapped to the edge, not to 6.67
    assert warning is not None          # and the snap is reported
    assert "within one frame" in warning


def test_the_overshoot_is_smaller_than_one_frame():
    """The premise of the snap, stated as arithmetic rather than asserted."""
    assert 6.67 - _TT7_SEG_END < 1.0 / _TT7_FPS
    assert 6.67 - _TT7_SEG_END == pytest.approx(0.00333, abs=1e-5)


def test_an_overshoot_of_more_than_one_frame_is_still_dropped():
    """The snap must not become a clamp. One frame at 30fps is 0.0333s."""
    record = {"start_time_sec": 0.0, "end_time_sec": _TT7_SEG_END + 0.05}
    start, end, warning = _absolute_timing(record, 0.0, _TT7_SEG_END, _TT7_FPS)
    assert start is None and end is None
    assert "outside segment" in warning


def test_the_2_0_offset_on_a_short_clip_is_still_dropped_with_fps():
    """The five real failures must not be rescued by the snap."""
    for seg_start, seg_end, rel_end in _REAL_FAILURES:
        record = {"start_time_sec": 0.0, "end_time_sec": rel_end}
        start, end, warning = _absolute_timing(record, seg_start, seg_end, 30.0)
        assert start is None and end is None
        assert "outside segment" in warning


def test_no_fps_means_no_snapping():
    """Absent video metadata must not produce a guessed frame rate."""
    record = {"start_time_sec": 0.0, "end_time_sec": 6.67}
    start, end, warning = _absolute_timing(record, 0.0, _TT7_SEG_END)
    assert start is None and end is None
    assert "outside segment" in warning


def test_a_bound_inside_the_segment_is_never_moved():
    """Snapping only ever pulls a value IN; it must not round a real offset."""
    inside = _TT7_SEG_END - 0.001  # within one frame of the edge, but inside
    record = {"start_time_sec": 0.001, "end_time_sec": inside}
    start, end, warning = _absolute_timing(record, 0.0, _TT7_SEG_END, _TT7_FPS)
    assert start == 0.001
    assert end == inside
    assert warning is None


def test_a_sub_frame_undershoot_at_the_start_snaps_too():
    """The rule is about the segment's edges, not about the end bound."""
    record = {"start_time_sec": -0.01, "end_time_sec": 1.0}
    start, end, warning = _absolute_timing(record, 0.0, 2.0, _TT7_FPS)
    assert start == 0.0
    assert end == 1.0
    assert "start_time_sec" in warning


def test_snapping_cannot_invert_the_interval():
    """A snap that produced start > end would have to drop both bounds."""
    record = {"start_time_sec": 1.0, "end_time_sec": -0.01}
    start, end, warning = _absolute_timing(record, 0.0, 2.0, _TT7_FPS)
    assert start is None and end is None
    assert ">" in warning


# ── The same failure through the whole stage ─────────────────────────────────
#
# The helper above is where the rule lives, but the rule only pays off if the
# observation survives s06 and reaches s07. This is the run that produced two
# events from seven segments, reduced to one segment.

# gemini-3.1-flash-lite's verbatim answer for tt7 segment [5.71, 6.67], the last
# of the five that were discarded.
_REAL_RESPONSE = json.dumps({
    "actor": "person wearing a blue t-shirt",
    "active_hand": "BOTH",
    "objects": ["cardboard box"],
    "raw_action": "pushed the box across the table",
    "start_time_sec": 0.0,
    "end_time_sec": 2.0,
    "state_change": "box has moved",
    "visible_facts": "both hands contact the box and it slides",
    "inference": "repositioning the box",
    "uncertainty": "none",
    "confidence": 0.95,
})


class _FakeVLM:
    backend = "GEMINI"
    model_name = "gemini-3.1-flash-lite"

    def analyze_segment(self, video_path, start_sec, end_sec, prompt):
        return _REAL_RESPONSE


@pytest.fixture
def short_segment_ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(s06_vlm, "LocalVLM", lambda **kw: _FakeVLM())
    config = PipelineConfig(stub_mode=False)
    config.vlm.enabled = True
    config.vlm.backend = "LOCAL_MODEL"
    ctx = PipelineContext(
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path / "output",
        config=config,
    )
    ctx.candidate_segments = [CandidateSegment(
        segment_id="cand_0006_a8108d",
        start_frame=172,
        end_frame=199,
        start_sec=5.714285714285714,
        end_sec=6.666666666666667,
        trigger_reason="proximity_overlap+movement",
    )]
    return ctx


def test_an_impossible_offset_no_longer_fails_the_segment(short_segment_ctx):
    status = s06_vlm.run(short_segment_ctx)
    assert status.status == "OK"

    obs = short_segment_ctx.vlm_observations
    assert len(obs) == 1
    # This assertion is the whole point: it was FAILED, with error_reason
    # "end_time_sec (7.714285714285714) is outside segment [5.71, 6.67]".
    assert obs[0].status == VLMSegmentStatus.SUCCESS
    assert obs[0].raw_action == "pushed the box across the table"
    assert obs[0].objects == ["cardboard box"]
    assert obs[0].error_reason is None
    # The unusable offsets are gone rather than clamped, so nothing downstream
    # can mistake them for localisation.
    assert obs[0].start_time_sec is None
    assert obs[0].end_time_sec is None
