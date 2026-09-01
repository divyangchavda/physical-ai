"""Tests for the pure parts of the SSv2 evaluation harness.

Nothing here runs the pipeline. What is tested is the four decisions the harness
makes on its own, each of which could silently distort a reported accuracy:

  * which vocabulary the detector is handed per clip,
  * which of several emitted events counts as "the answer",
  * which clips a direction figure may include,
  * that direction truth comes from SSv2's wording and not from our own verb map.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from ssv2_eval import (  # noqa: E402
    BASE_CLASSES,
    build_prompt,
    primary_event,
    score_direction,
    template_direction,
)


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
