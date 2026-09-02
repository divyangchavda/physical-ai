"""Tests for the pure parts of the SSv2 evaluation harness.

Nothing here runs the pipeline. What is tested is the four decisions the harness
makes on its own, each of which could silently distort a reported accuracy:

  * which vocabulary the detector is handed per clip,
  * which of several emitted events counts as "the answer",
  * which clips a direction figure may include,
  * that direction truth comes from SSv2's wording and not from our own verb map.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from ssv2_eval import (  # noqa: E402
    BASE_CLASSES,
    build_prompt,
    caption_corpus,
    event_summary,
    interleave_by_verb,
    merge_results,
    primary_event,
    score_direction,
    template_direction,
)


# ───────────────────────────────────────────────── run order, so a partial run reads
def _clips(spec: list[tuple[str, int]]) -> list[dict]:
    """``[("CLOSE", 3), ("OPEN", 2)]`` -> a verb-grouped clip list like the bundle's."""
    return [
        {"clip_id": f"{verb}{i}", "action": verb}
        for verb, count in spec for i in range(count)
    ]


def test_a_short_run_covers_every_verb_instead_of_only_the_first():
    """The flaw the first live batch exposed: five clips, all of them CLOSE.

    truth.json is grouped by verb, so ``--limit 5`` measured one verb and said
    nothing about the other ten.
    """
    grouped = _clips([("CLOSE", 3), ("OPEN", 3), ("PUSH", 3)])
    assert {c["action"] for c in grouped[:3]} == {"CLOSE"}
    assert {c["action"] for c in interleave_by_verb(grouped)[:3]} == (
        {"CLOSE", "OPEN", "PUSH"}
    )


def test_no_clip_is_lost_or_duplicated_by_reordering():
    grouped = _clips([("CLOSE", 4), ("OPEN", 2), ("PUSH", 7)])
    reordered = interleave_by_verb(grouped)
    assert sorted(c["clip_id"] for c in reordered) == sorted(
        c["clip_id"] for c in grouped
    )
    assert len(reordered) == len(grouped)


def test_a_verb_that_runs_out_does_not_stall_the_others():
    """Verb counts differ (CLOSE had 22 available against PLACE's 338), so the
    shorter verbs simply drop out of later rounds."""
    reordered = interleave_by_verb(_clips([("CLOSE", 1), ("PUSH", 3)]))
    assert [c["clip_id"] for c in reordered] == ["CLOSE0", "PUSH0", "PUSH1", "PUSH2"]


def test_each_verb_keeps_its_own_order_so_the_draw_stays_deterministic():
    reordered = interleave_by_verb(_clips([("CLOSE", 3), ("PUSH", 3)]))
    assert [c["clip_id"] for c in reordered if c["action"] == "CLOSE"] == (
        ["CLOSE0", "CLOSE1", "CLOSE2"]
    )


def test_reordering_an_empty_list_is_not_an_error():
    assert interleave_by_verb([]) == []


# ──────────────────────────────────────────── resuming must not shrink the result
def _result(clip_id: str, verdict: str = "TOP1") -> dict:
    return {"clip_id": clip_id, "truth_action": "CLOSE", "verdict": verdict,
            "got": "CLOSE", "exit_code": 0, "seconds": 1.0, "direction": "N/A"}


def test_resuming_keeps_the_clips_a_previous_run_already_paid_for(tmp_path):
    """A 195-clip run that dies at clip 180 must not come back as a 15-clip
    measurement."""
    path = tmp_path / "results.json"
    path.write_text('{"results": [{"clip_id": "a", "verdict": "TOP1"}]}',
                    encoding="utf-8")
    merged = merge_results(path, [_result("b")])
    assert sorted(r["clip_id"] for r in merged) == ["a", "b"]


def test_a_rerun_of_one_clip_replaces_its_old_record(tmp_path):
    path = tmp_path / "results.json"
    path.write_text('{"results": [{"clip_id": "a", "verdict": "MISS"}]}',
                    encoding="utf-8")
    merged = merge_results(path, [_result("a", verdict="TOP1")])
    assert len(merged) == 1
    assert merged[0]["verdict"] == "TOP1"


def test_no_prior_file_is_the_normal_first_run(tmp_path):
    merged = merge_results(tmp_path / "nothing.json", [_result("a")])
    assert [r["clip_id"] for r in merged] == ["a"]


def test_an_unreadable_prior_file_does_not_discard_this_runs_work(tmp_path):
    """An hour of GPU must not be lost to a truncated file from an earlier crash."""
    path = tmp_path / "results.json"
    path.write_text("{ truncated", encoding="utf-8")
    merged = merge_results(path, [_result("a")])
    assert [r["clip_id"] for r in merged] == ["a"]


# ───────────────────────────────────────────────────────── the per-clip prompt
def test_the_clips_own_object_names_are_offered_to_the_detector():
    prompt = build_prompt(["gate"])
    assert prompt == "person . hand . table . gate ."


def test_a_multi_word_placeholder_survives_intact():
    """SSv2 fills its brackets with phrases, and GroundingDINO grounds phrases."""
    assert build_prompt(["a box of eggs"]).endswith("a box of eggs .")


def test_the_actor_classes_are_always_present():
    """Without an actor class s05 produces no segments and the clip scores
    nothing, so these are not optional per clip."""
    for phrase in BASE_CLASSES:
        assert f"{phrase} ." in build_prompt(["gate"]) or f"{phrase} . " in build_prompt(["gate"])


def test_a_placeholder_repeating_a_base_class_is_not_repeated():
    """A duplicated span competes with itself for the same box."""
    assert build_prompt(["table"]) == "person . hand . table ."
    assert build_prompt(["Table", "TABLE"]) == "person . hand . table ."


def test_placeholders_are_lowercased_and_whitespace_collapsed():
    assert build_prompt(["  Wooden   Spoon "]) == "person . hand . table . wooden spoon ."


def test_an_empty_placeholder_list_still_gives_a_usable_prompt():
    assert build_prompt([]) == "person . hand . table ."


# ─────────────────────────────────────────────────────── which event is "the" answer
def _ev(action: str, confidence: float, start: float) -> dict:
    return {"action": action, "confidence": confidence, "start_sec": start}


def test_the_highest_confidence_event_is_the_answer():
    events = [_ev("PLACE", 0.4, 0.0), _ev("INSERT", 0.9, 1.0)]
    assert primary_event(events)["action"] == "INSERT"


def test_a_confidence_tie_is_broken_by_start_time_not_list_order():
    """Otherwise the answer depends on the order s07 happened to append in."""
    a = [_ev("PLACE", 0.5, 2.0), _ev("INSERT", 0.5, 0.5)]
    b = [_ev("INSERT", 0.5, 0.5), _ev("PLACE", 0.5, 2.0)]
    assert primary_event(a)["action"] == "INSERT"
    assert primary_event(b)["action"] == "INSERT"


def test_no_events_has_no_answer():
    assert primary_event([]) is None


def test_a_missing_confidence_is_treated_as_zero_not_as_an_error():
    """An event without a confidence must not win by accident, and must not crash
    the whole 200-clip run either."""
    events = [{"action": "PUSH", "start_sec": 0.0}, _ev("PULL", 0.1, 1.0)]
    assert primary_event(events)["action"] == "PULL"


# ──────────────────────────────────────────────── direction, read off SSv2's words
@pytest.mark.parametrize("template,expected", [
    ("Putting [something] into [something]", "INTO"),
    ("Taking [something] out of [something]", "OUT_OF"),
    ("Putting [something] onto [something]", "ONTO"),
    ("Pushing [something] off of [something]", "OFF"),
])
def test_a_stated_preposition_is_read_from_the_class_name(template, expected):
    assert template_direction(template) == expected


@pytest.mark.parametrize("template", [
    "Opening [something]",
    "Closing [something]",
    "Picking [something] up",
    "Holding [something]",
    "Touching (without moving) part of [something]",
])
def test_a_class_stating_no_preposition_yields_no_direction_truth(template):
    """Our own table calls OPEN an OUT_OF. Asserting that here would score us
    against ourselves, so these clips are excluded from the direction figure."""
    assert template_direction(template) is None


def test_out_of_is_matched_before_into_when_both_could_appear():
    """"out of" must not be shadowed by a substring rule; order is deliberate."""
    assert template_direction("Taking [something] out of [something]") == "OUT_OF"


def test_a_preposition_inside_a_word_does_not_count():
    """"into" appears inside no SSv2 word today, so this pins the boundary rule
    rather than a current failure."""
    assert template_direction("Reintotal something") is None


# ─────────────────────────────────────────────────────────── direction verdicts
def test_the_right_verb_in_the_right_direction_is_same():
    assert score_direction("INSERT", "Putting [something] into [something]") == "SAME"
    assert score_direction("REMOVE", "Taking [something] out of [something]") == "SAME"


def test_the_inversion_this_project_keeps_reproducing_reads_as_reversed():
    """OPEN is OUT_OF in the scorer's table; on a "putting into" clip that is the
    exact inversion tt6 and tt7 both showed."""
    assert score_direction("OPEN", "Putting [something] into [something]") == "REVERSED"
    assert score_direction("INSERT", "Taking [something] out of [something]") == "REVERSED"


def test_a_verb_with_no_direction_in_the_table_is_not_a_direction_error():
    """GRASP, MOVE, PUSH, PULL and TOUCH have no entry, so nothing is claimed."""
    assert score_direction("GRASP", "Putting [something] into [something]") == "N/A"
    assert score_direction("MOVE", "Putting [something] onto [something]") == "N/A"


def test_a_class_with_no_preposition_is_not_a_direction_error_either():
    assert score_direction("OPEN", "Opening [something]") == "N/A"


def test_onto_against_off_is_reversed_not_other():
    assert score_direction("PICK", "Putting [something] onto [something]") == "REVERSED"
    assert score_direction("PLACE", "Pushing [something] off of [something]") == "REVERSED"


def test_a_crossed_axis_is_other_not_reversed():
    """ONTO and INTO are different axes; calling that REVERSED would overstate
    how close the answer was."""
    assert score_direction("PLACE", "Putting [something] into [something]") == "OTHER"


# ───────────────────────── the replay corpus, so a lost run dir costs nothing
def test_every_caption_is_recorded_not_only_the_winning_one(tmp_path):
    """The 113-minute run whose run dirs a kernel restart destroyed left only the
    winning event's caption. A clip describing two actions was then unscoreable."""
    (tmp_path / "vlm_observations.json").write_text(json.dumps([
        {"raw_action": "opening the box", "objects": ["box"], "status": "SUCCESS",
         "segment_start_sec": 0.0, "segment_end_sec": 2.0},
        {"raw_action": "closing the lid", "objects": ["lid"], "status": "SUCCESS",
         "segment_start_sec": 2.0, "segment_end_sec": 4.0},
    ]), encoding="utf-8")
    corpus = caption_corpus(tmp_path)
    assert [c["raw_action"] for c in corpus] == ["opening the box", "closing the lid"]


def test_the_objects_are_recorded_so_object_blanking_can_be_replayed(tmp_path):
    """Without them "red folder" reads as a fold and outranks the real verb."""
    (tmp_path / "vlm_observations.json").write_text(json.dumps(
        [{"raw_action": "placing it on a red folder", "objects": ["red folder"]}]
    ), encoding="utf-8")
    assert caption_corpus(tmp_path)[0]["objects"] == ["red folder"]


def test_a_missing_observations_file_is_empty_not_fatal(tmp_path):
    """The corpus helps the next run; losing it must not kill the one in progress."""
    assert caption_corpus(tmp_path) == []


def test_why_a_call_returned_nothing_is_recorded_beside_the_fact_that_it_did(tmp_path):
    """Five clips of the 60-clip run failed on a Gemini 503, which is the API
    being busy. A parse failure or a safety block would mean something entirely
    different about our prompt, and the two are indistinguishable afterwards
    unless the reason is kept."""
    (tmp_path / "vlm_observations.json").write_text(json.dumps([{
        "raw_action": None, "status": "FAILED",
        "error_reason": "Gemini inference failed: 503 UNAVAILABLE.",
    }]), encoding="utf-8")
    entry = caption_corpus(tmp_path)[0]
    assert entry["raw_action"] is None
    assert "503" in entry["error_reason"]


def test_an_unreadable_observations_file_is_empty_not_fatal(tmp_path):
    (tmp_path / "vlm_observations.json").write_text("{ truncated", encoding="utf-8")
    assert caption_corpus(tmp_path) == []


def test_an_observations_file_wrapped_in_a_dict_is_read_too(tmp_path):
    """s06 writes a bare list, but the replay path accepts either shape."""
    (tmp_path / "vlm_observations.json").write_text(
        json.dumps({"observations": [{"raw_action": "opening the box"}]}),
        encoding="utf-8",
    )
    assert caption_corpus(tmp_path)[0]["raw_action"] == "opening the box"


def test_the_confidence_that_decides_top1_is_recorded(tmp_path):
    """primary_event ranks on confidence, so a two-event clip cannot be scored
    offline without it."""
    summary = event_summary([_ev("TOUCH", 0.4, 0.0), _ev("PICK", 0.9, 1.0)])
    assert summary == [
        {"action": "TOUCH", "confidence": 0.4, "start_sec": 0.0},
        {"action": "PICK", "confidence": 0.9, "start_sec": 1.0},
    ]
    assert primary_event(summary)["action"] == "PICK"
