"""Tests for the state-contradiction correction in s07.

Every string here comes from tests/fixtures/tt7_b_seg1_observations.json, which
is the verbatim VLM output of the Kaggle run of tt7 at
``segment.max_segment_duration_sec=1.0`` on 2026-09-01. That run scored 2 EXACT,
2 REVERSED and 3 OTHER, and the two REVERSED were both OPEN where the hand
labels say CLOSE — the fourth reproduction of that inversion.

The point of testing against the fixture rather than invented sentences is that
the first fix considered here did not survive contact with it: adding
``state_before``/``state_after`` to the prompt would have found nothing to
contradict, because all three OPEN answers describe the transition progressively
("flaps are being unfolded", "flap is being lifted") and never name the closed
state at all.

**The bindings in this file are synthetic.** That run's ``events.json`` was lost
to a kernel restart, so every event here is built with ``object_track_id=6``,
which is an assumption and not a measurement — and it is exactly the assumption
that let this file pass while the rule corrected nothing on the next live run.
Object attribution is therefore tested in ``test_s07_state_subject.py`` against
``tt7_c_seg1_observations.json``, which records the real bindings. What this file
tests is the state machine: which verb flips, when, and how runs are grouped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schema.event import ActionType, PhysicalEvent
from src.stages.s07_events import (
    _map_raw_action_to_type,
    _resolve_state_contradictions,
    _state_evidence,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "tt7_b_seg1_observations.json"
)

# The one entity every observation in that run is about. entity 6 in the run's
# own track-stitching table is the cardboard box (frames 6-199), and all three
# OPEN events were scored against cardboard-box labels (FOLD, CLOSE, GRASP), so
# they resolved the same object.
BOX = 6


def _seed(event: PhysicalEvent) -> tuple[str, str] | None:
    """The state and phrase from ``_state_evidence``, dropping its offset/text.

    Those two exist only so ``_state_subject`` can decide which object is meant,
    which is a separate question from whether a state was stated at all.
    """
    found = _state_evidence(event)
    return None if found is None else (found[0], found[1])



@pytest.fixture(scope="module")
def run_observations() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["observations"]


def _event(
    action: ActionType,
    start: float,
    *,
    obj: int | None = BOX,
    visible_facts: str = "",
    state_change: str = "",
) -> PhysicalEvent:
    return PhysicalEvent(
        event_id=f"evt_{start:.4f}",
        segment_id="seg",
        action=action,
        object_track_id=obj,
        start_sec=start,
        end_sec=start + 0.95,
        source="vlm:gemini",
        attributes={
            "visible_facts": visible_facts,
            "state_change": state_change,
        },
    )


def _events_from_fixture(observations: list[dict]) -> list[PhysicalEvent]:
    """Rebuild the run's events, taking each verb from the real stem table.

    The verb is not asserted here — ``_map_raw_action_to_type`` derives it from
    the verbatim ``raw_action``, so if the stem table changes these tests change
    with it rather than silently testing a stale mapping.
    """
    return [
        _event(
            _map_raw_action_to_type(o["raw_action"].lower()),
            o["segment_start_sec"],
            visible_facts=o["visible_facts"],
            state_change=o["state_change"],
        )
        for o in observations
    ]


# ─────────────────────────────────────────────── what the run actually produced
def test_the_run_really_did_emit_three_consecutive_opens(run_observations):
    """The failure, taken from the stem table rather than described."""
    verbs = [
        _map_raw_action_to_type(o["raw_action"].lower())
        for o in run_observations
    ]
    assert verbs[2:5] == [ActionType.OPEN] * 3


def test_no_single_answer_contradicts_itself(run_observations):
    """Why the prompt fix was dropped, as a test.

    If any one of the three OPEN answers named the closed state, a within-answer
    check would have worked and this whole cross-observation mechanism would be
    unnecessary. None of them does.
    """
    for o in run_observations[2:5]:
        text = (o["visible_facts"] + " " + o["state_change"]).lower()
        assert "closed" not in text
        assert "shut" not in text


def test_the_only_state_seed_in_the_run_is_the_first_observation(run_observations):
    """One explicit state word in seven observations, and it is at 0.0s."""
    seeds = {
        o["segment_start_sec"]: _seed(
            _event(ActionType.UNKNOWN, o["segment_start_sec"],
                   visible_facts=o["visible_facts"],
                   state_change=o["state_change"])
        )
        for o in run_observations
    }
    assert seeds[0.0] == ("OPEN", "open")
    assert seeds[0.9523809523809524] == ("OPEN", "open")  # "into an open box"
    for sec, seed in seeds.items():
        if sec > 1.0:
            assert seed is None, (sec, seed)


# ────────────────────────────────────────────────────────── the correction
def test_the_three_opens_become_closes(run_observations):
    """The headline: the run's own words rule out three consecutive opens."""
    events = _resolve_state_contradictions(_events_from_fixture(run_observations))
    assert [e.action for e in events[2:5]] == [ActionType.CLOSE] * 3


def test_the_two_reversed_events_are_the_ones_that_flip(run_observations):
    """Named against the labels: FOLD 2.0-3.0, CLOSE 3.0-4.0.

    The events at 2.86 and 3.81 were both scored REVERSED (OPEN against a
    labelled CLOSE) and are the two the fix is claimed to convert. The 1.90 event
    covers FOLD, which the stem table maps to UNKNOWN deliberately, so it is not
    expected to become correct — only consistent.
    """
    events = _resolve_state_contradictions(_events_from_fixture(run_observations))
    by_start = {round(e.start_sec, 2): e for e in events}
    assert by_start[2.86].action is ActionType.CLOSE
    assert by_start[3.81].action is ActionType.CLOSE


def test_correcting_one_at_a_time_would_have_alternated(run_observations):
    """The reason runs are corrected together, as the failure it avoids.

    Flipping the tracked state after each event leaves the second OPEN facing a
    CLOSED box, where opening is perfectly satisfiable — so it would survive and
    the corrections would alternate CLOSE, OPEN, CLOSE instead of converging.
    Simulated here by feeding the events in one at a time, each call starting
    from a fresh state built out of the previous result.
    """
    events = _events_from_fixture(run_observations)
    verbs = []
    state = "OPEN"  # what observation 0 establishes
    for e in events[2:5]:
        if e.action is ActionType.OPEN and state == "OPEN":
            verbs.append(ActionType.CLOSE)
            state = "CLOSED"
        else:
            verbs.append(e.action)
            state = "OPEN"
    assert verbs == [ActionType.CLOSE, ActionType.OPEN, ActionType.CLOSE]
    # And the real implementation does not do that.
    assert [e.action for e in _resolve_state_contradictions(events)[2:5]] == (
        [ActionType.CLOSE] * 3
    )


def test_every_correction_is_recorded(run_observations):
    """A rewritten verb that leaves no trace in events.json is not auditable.

    The recorded seed is 0.95s, not 0.00s: two separate observations state the
    box is open ("the top flap of an open cardboard box" and "moves it downward
    into an open cardboard box") and the most recent statement is the one carried.
    Two independent mentions is a stronger seed than one, not a weaker one.
    """
    events = _resolve_state_contradictions(_events_from_fixture(run_observations))
    for e in events[2:5]:
        assert e.attributes["verb_source"] == "STATE_UNSATISFIABLE"
        assert e.attributes["verb_before_correction"] == "OPEN"
        assert "already OPEN" in e.attributes["state_evidence"]
        assert "0.95s" in e.attributes["state_evidence"]


def test_the_untouched_events_carry_no_verb_source(run_observations):
    events = _resolve_state_contradictions(_events_from_fixture(run_observations))
    for e in events[:2] + events[5:]:
        assert "verb_source" not in e.attributes


# ──────────────────────────────────────────────────────────── the guard rails
def test_an_object_with_no_stated_state_is_never_corrected():
    """No seed means no deduction. Silence must not be read as CLOSED."""
    events = [
        _event(ActionType.OPEN, 1.0, visible_facts="hands are on the flaps"),
        _event(ActionType.OPEN, 2.0, visible_facts="hands are on the flaps"),
    ]
    assert [e.action for e in _resolve_state_contradictions(events)] == (
        [ActionType.OPEN] * 2
    )


def test_a_satisfiable_open_is_left_alone():
    """The whole point is impossibility, not a preference for CLOSE."""
    events = [
        _event(ActionType.GRASP, 0.0, visible_facts="the box is closed"),
        _event(ActionType.OPEN, 1.0),
    ]
    assert _resolve_state_contradictions(events)[1].action is ActionType.OPEN


def test_the_rule_is_symmetric_for_close():
    events = [
        _event(ActionType.GRASP, 0.0, visible_facts="the box is closed"),
        _event(ActionType.CLOSE, 1.0),
    ]
    result = _resolve_state_contradictions(events)
    assert result[1].action is ActionType.OPEN
    assert result[1].attributes["verb_before_correction"] == "CLOSE"


def test_an_unresolved_object_is_never_corrected():
    """"closing the lid" resolves no track, so there is no state to consult."""
    events = [
        _event(ActionType.GRASP, 0.0, visible_facts="an open cardboard box"),
        _event(ActionType.OPEN, 1.0, obj=None),
    ]
    assert _resolve_state_contradictions(events)[1].action is ActionType.OPEN


def test_state_is_tracked_per_object_not_globally():
    """An open box says nothing about whether a different container is open."""
    events = [
        _event(ActionType.GRASP, 0.0, visible_facts="an open cardboard box"),
        _event(ActionType.OPEN, 1.0, obj=99),
    ]
    assert _resolve_state_contradictions(events)[1].action is ActionType.OPEN


def test_a_progressive_verb_never_seeds_a_state():
    """"opening" is an action in flight, not a state that has been reached.

    If it seeded OPEN then every one of these observations would vouch for
    itself, and the first OPEN in any run would correct itself with no outside
    evidence at all.
    """
    assert _seed(
        _event(ActionType.OPEN, 0.0, visible_facts="hands are opening the flaps")
    ) is None
    assert _seed(
        _event(ActionType.OPEN, 0.0, state_change="the box is being opened")
    ) == ("OPEN", "opened")  # a completed past participle DOES state a state


def test_text_naming_both_states_seeds_nothing():
    """Two states in one sentence must not be resolved by match order."""
    assert _seed(
        _event(ActionType.GRASP, 0.0,
               visible_facts="a closed carton beside an open cardboard box")
    ) is None


def test_a_progressive_verb_does_not_cancel_a_stated_state():
    """"closing the open box" states that the box IS open and that closing is
    in flight. Only the state word counts, which is the correct reading and the
    reason the both-states guard above needs two real state words to fire."""
    assert _seed(
        _event(ActionType.GRASP, 0.0, visible_facts="closing the open box")
    ) == ("OPEN", "open")


def test_the_verb_under_test_cannot_seed_its_own_state():
    """raw_action is excluded, so a CLOSE cannot veto itself."""
    event = _event(ActionType.CLOSE, 1.0)
    event.attributes["raw_action"] = "closed the cardboard box"
    assert _seed(event) is None
    assert _resolve_state_contradictions([event])[0].action is ActionType.CLOSE


def test_inference_is_not_evidence():
    """The prompt defines inference as reasoning beyond what is visible."""
    event = _event(ActionType.GRASP, 0.0)
    event.attributes["inference"] = "the box is probably closed"
    assert _seed(event) is None


def test_events_out_of_time_order_are_still_read_in_time_order():
    """s06 emits in segment order today; the rule must not depend on that."""
    events = [
        _event(ActionType.OPEN, 2.0),
        _event(ActionType.GRASP, 0.0, visible_facts="an open cardboard box"),
    ]
    result = _resolve_state_contradictions(events)
    assert result[0].action is ActionType.CLOSE


def test_a_run_broken_by_another_object_is_two_runs():
    """Two OPENs either side of a different box's OPEN are still one run each."""
    events = [
        _event(ActionType.GRASP, 0.0, visible_facts="an open cardboard box"),
        _event(ActionType.OPEN, 1.0),
        _event(ActionType.OPEN, 2.0, obj=99),
        _event(ActionType.OPEN, 3.0),
    ]
    result = _resolve_state_contradictions(events)
    assert result[1].action is ActionType.CLOSE
    assert result[2].action is ActionType.OPEN   # no state known for 99
    # The box is CLOSED after the run at 1.0, so opening it at 3.0 is possible.
    assert result[3].action is ActionType.OPEN


def test_no_events_is_not_an_error():
    assert _resolve_state_contradictions([]) == []
