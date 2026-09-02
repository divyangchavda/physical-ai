"""Tests for the run report: does it name the right cause for each failure?

The report's whole value is that its causes are read off recorded fields instead
of guessed from what the numbers look like. So what is pinned here is exactly
that: each cause is produced only by the evidence that actually implies it, and a
run too old to carry the evidence says so rather than inventing a cause.

Nothing here runs the pipeline or the VLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from ssv2_report import (  # noqa: E402
    caption_of,
    cause,
    failure_reasons,
    reason_signature,
    render,
)


def _record(**overrides) -> dict:
    """A MISS with one good caption — the base every case below deviates from."""
    record = {
        "clip_id": "100001",
        "truth_action": "PUSH",
        "template": "Pushing [something] from left to right",
        "label": "pushing pencil from left to right",
        "verdict": "MISS",
        "got": "PICK",
        "all_verbs": ["PICK"],
        "n_events": 1,
        "direction": "N/A",
        "exit_code": 0,
        "seconds": 34.0,
        "raw_action": "picked up the mug",
        "captions": [{"raw_action": "picked up the mug", "objects": ["mug"],
                      "status": "SUCCESS", "error_reason": None}],
        "events": [{"action": "PICK", "confidence": 0.7, "start_sec": 0.0}],
    }
    record.update(overrides)
    return record


# ───────────────────────────────────────────────────── one cause per kind of evidence
def test_a_right_answer_is_not_a_failure_at_all():
    assert cause(_record(verdict="TOP1", got="PUSH")) == "CORRECT"


def test_no_caption_at_all_means_s05_never_produced_a_segment():
    """Three clips of the 60-clip run: GroundingDINO found no hand, so s05 made
    no segment and s06 was never called. An empty list is the evidence."""
    assert cause(_record(verdict="NO_EVENTS", got=None, captions=[])) == "NO_SEGMENT"


def test_a_caption_record_with_no_text_means_the_vlm_call_failed():
    """Five clips of the 60-clip run: the segment was sent and 503'd."""
    record = _record(
        verdict="NO_EVENTS", got=None, raw_action=None,
        captions=[{"raw_action": None, "status": "FAILED",
                   "error_reason": "503 UNAVAILABLE"}],
    )
    assert cause(record) == "VLM_FAILED"


def test_one_failed_call_beside_one_good_one_is_not_a_failed_clip():
    """A clip whose second segment 503'd still has an answer from the first."""
    record = _record(captions=[
        {"raw_action": None, "status": "FAILED", "error_reason": "503 UNAVAILABLE"},
        {"raw_action": "picked up the mug", "status": "SUCCESS"},
    ])
    assert cause(record) == "WRONG_VERB"


def test_a_caption_that_maps_to_nothing_is_a_vocabulary_gap():
    """"lifting the notebook off the newspaper" mapped to UNKNOWN: the VLM named
    an action and our table has no stem for it."""
    record = _record(
        got="UNKNOWN", raw_action="lifting the notebook off the newspaper",
        captions=[{"raw_action": "lifting the notebook off the newspaper",
                   "status": "SUCCESS"}],
    )
    assert cause(record) == "UNMAPPED"


def test_no_event_despite_a_usable_caption_is_also_a_vocabulary_gap():
    """The caption came back fine and s07 still emitted nothing."""
    record = _record(verdict="NO_EVENTS", got=None,
                     captions=[{"raw_action": "doing something", "status": "SUCCESS"}])
    assert cause(record) == "UNMAPPED"


def test_the_right_verb_ranked_second_is_a_confidence_problem():
    """Separated from the rest because no verb-table change can fix it and no
    prompt change is needed for it."""
    record = _record(verdict="ANY_ONLY", got="TOUCH", all_verbs=["TOUCH", "PUSH"])
    assert cause(record) == "RANKING"


def test_a_clean_map_to_the_wrong_verb_is_left_to_the_reader():
    """The tool will not claim to know whether the VLM misread the video or
    whether we simply disagree with SSv2 about the word."""
    assert cause(_record()) == "WRONG_VERB"


def test_a_run_predating_the_caption_field_admits_it_cannot_tell():
    """The 200-clip run has no captions key. Calling that NO_SEGMENT would report
    five 503s and three missing actors as a segmentation problem."""
    record = _record()
    del record["captions"]
    assert cause(record) == "UNKNOWN_CAUSE"


# ────────────────────────────────────────────────────── the API's reason, not ours
def test_the_recorded_reason_is_preferred_over_re_reading_the_run_dir(tmp_path):
    record = _record(captions=[{"raw_action": None,
                                "error_reason": "503 UNAVAILABLE. high demand"}])
    assert failure_reasons(record, tmp_path) == ["503 UNAVAILABLE. high demand"]


def test_an_older_run_falls_back_to_the_run_directory(tmp_path):
    """Recorded reasons only exist for runs after this field was added; the run
    dirs of the 60-clip run still had them."""
    clip_dir = tmp_path / "100001"
    clip_dir.mkdir()
    (clip_dir / "vlm_observations.json").write_text(json.dumps(
        [{"status": "FAILED", "error_reason": "Gemini inference failed: 503"}]
    ), encoding="utf-8")
    record = _record(captions=[{"raw_action": None, "status": "FAILED"}])
    assert failure_reasons(record, tmp_path) == ["Gemini inference failed: 503"]


def test_a_missing_run_directory_reports_no_reason_rather_than_a_plausible_one(tmp_path):
    record = _record(captions=[{"raw_action": None, "status": "FAILED"}])
    assert failure_reasons(record, tmp_path) == []


def test_an_unreadable_observations_file_is_not_fatal(tmp_path):
    clip_dir = tmp_path / "100001"
    clip_dir.mkdir()
    (clip_dir / "vlm_observations.json").write_text("{ truncated", encoding="utf-8")
    record = _record(captions=[{"raw_action": None, "status": "FAILED"}])
    assert failure_reasons(record, tmp_path) == []


def test_ten_identical_failures_group_into_one_line():
    long_error = ("API/Execution failed: Gemini inference failed: 503 UNAVAILABLE. "
                  "{'error': {'code': 503, 'message': 'This model is currently "
                  "experiencing high demand.'}}")
    assert reason_signature(long_error) == "503 UNAVAILABLE"


def test_an_error_with_no_status_code_is_still_reported():
    assert reason_signature("could not parse the response") == (
        "could not parse the response"
    )


# ───────────────────────────────────────────────────────── which caption is shown
def test_the_winning_events_caption_is_the_one_the_verb_came_from():
    record = _record(raw_action="picked up the mug", captions=[
        {"raw_action": "touching the mug"},
        {"raw_action": "picked up the mug"},
    ])
    assert caption_of(record) == "picked up the mug"


def test_a_clip_with_no_winner_still_shows_the_text_it_did_get():
    record = _record(raw_action=None,
                     captions=[{"raw_action": "hand hovers over the mug"}])
    assert caption_of(record) == "hand hovers over the mug"


def test_a_clip_with_no_text_anywhere_shows_nothing_invented():
    assert caption_of(_record(raw_action=None, captions=[])) == ""


# ─────────────────────────────────────────────────────────── the report as a whole
def test_the_report_covers_every_clip_and_every_section():
    records = [
        _record(verdict="TOP1", got="PUSH"),
        _record(clip_id="100002", verdict="NO_EVENTS", got=None, raw_action=None,
                captions=[{"raw_action": None, "status": "FAILED",
                           "error_reason": "503 UNAVAILABLE"}]),
        _record(clip_id="100003", truth_action="GRASP", verdict="NO_EVENTS",
                got=None, raw_action=None, captions=[]),
    ]
    text = render({"_config": "c.yaml", "_bundle": "b"}, records, None)
    for clip_id in ("100001", "100002", "100003"):
        assert clip_id in text
    for section in ("FULL REPORT", "WHY THE OTHER CLIPS", "VLM RELIABILITY",
                    "HEADROOM", "APPENDIX"):
        assert section in text
    assert "503 UNAVAILABLE" in text


def test_the_headroom_is_labelled_as_a_bound_and_not_as_a_prediction():
    """The failure mode this guards against is a 45% run being described as an
    88% run because the causes were added up."""
    text = render({}, [_record()], None)
    assert "UPPER BOUND, NOT A FORECAST" in text


def test_a_single_correct_clip_does_not_divide_by_zero():
    text = render({}, [_record(verdict="TOP1", got="PUSH")], None)
    assert "1/1 = 100.0%" in text


def test_a_cause_we_could_not_identify_claims_no_headroom():
    """There is no such thing as fixing UNKNOWN_CAUSE, so it gets no line — a
    gain printed for it would inflate the bound with clips nobody can act on."""
    old = _record()
    del old["captions"]
    text = render({}, [old], None)
    assert "UNKNOWN_CAUSE" in text
    assert "if UNKNOWN_CAUSE were fixed" not in text
