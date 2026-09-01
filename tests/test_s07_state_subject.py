"""Tests for which object a state word is attributed to (s07).

Everything here comes from tests/fixtures/tt7_c_seg1_observations.json, the
Kaggle run of tt7 at ``segment.max_segment_duration_sec=1.0`` on 2026-09-01 at
commit c400ee8 — the run where the state-contradiction rule was live and
corrected nothing.

That fixture records ``object_track_id`` per observation, read off the run's own
events.json, and this file may not substitute its own. The bug it exists to pin
is exactly an assumed binding: tests/test_s07_state_contradiction.py builds every
event with ``object_track_id=6``, so a rule that seeded state from the event's own
object passed 21 tests and then fired on nothing, because in the real run the
sentence carrying the state word had bound track 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schema.event import ActionType, PhysicalEvent
from src.stages.s07_events import (
    _mention_index,
    _resolve_state_contradictions,
    _state_evidence,
    _state_subject,
)

FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "tt7_c_seg1_observations.json"
)

# The run's stitched entities, from its printed track table.
CHOPPER = 3
BOX = 6
CARTON_ARTWORK = 13


@pytest.fixture(scope="module")
def run_observations() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["observations"]


def _events(observations: list[dict]) -> list[PhysicalEvent]:
    """The run's seven events, every field taken from the fixture.

    The verb and the binding are both *recorded*, not derived: this file is about
    attribution, and deriving either one here would reintroduce the assumption
    that caused the failure.
    """
    return [
        PhysicalEvent(
            event_id=f"evt_{o['segment_start_sec']:.4f}",
            segment_id="seg",
            action=ActionType(o["action"]),
            object_track_id=o["object_track_id"],
            start_sec=o["segment_start_sec"],
            end_sec=o["segment_end_sec"],
            source="vlm:gemini",
            attributes={
                "raw_action": o["raw_action"],
                "objects": o["objects"],
                "object_label": o["object_label"],
                "state_change": o["state_change"],
                "visible_facts": o["visible_facts"],
            },
        )
        for o in observations
    ]


def _label_map(observations: list[dict]) -> dict[str, int]:
    """What the events themselves resolved, the way the stage builds it."""
    out: dict[str, int] = {}
    for o in observations:
        if o["object_label"] and o["object_track_id"] is not None:
            out.setdefault(o["object_label"].strip().lower(), o["object_track_id"])
    return out


# ────────────────────────────────────────────────── the failure, as measurement
def test_the_runs_only_state_word_sits_in_an_event_bound_to_the_chopper(
    run_observations,
):
    """The diagnosis. One state word in seven observations, in the wrong event."""
    seeded = [
        (o["segment_start_sec"], o["object_track_id"], _state_evidence(e))
        for o, e in zip(run_observations, _events(run_observations))
        if _state_evidence(e) is not None
    ]
    assert len(seeded) == 1
    start, track_id, found = seeded[0]
    assert start == 0.0
    assert track_id == CHOPPER          # <- not the box the word describes
    assert found[0] == "OPEN"
    assert found[1] == "open"


def test_the_three_opens_are_all_on_the_box(run_observations):
    """The other half of the run: the verbs needing correction did bind track 6.

    So the mechanism was never the problem — only which object held the state.
    """
    events = _events(run_observations)
    opens = [e for e in events if e.action is ActionType.OPEN]
    assert len(opens) == 3
    assert {e.object_track_id for e in opens} == {BOX}


def test_the_chopper_is_not_named_in_the_sentence_at_all(run_observations):
    """Why nearest-mention can pick the box: the chopper has no mention to pick.

    The VLM listed "push chopper" in ``objects`` but wrote "a small kitchen
    appliance" in the text, so ``_mention_index`` finds neither the phrase nor its
    head noun "chopper".
    """
    text = run_observations[0]["visible_facts"]
    assert "push chopper" in run_observations[0]["objects"]
    assert _mention_index("push chopper", text) is None
    assert _mention_index("cardboard box", text) is not None


# ────────────────────────────────────────────────────────────── the attribution
def test_the_state_word_is_attributed_to_the_box_not_the_chopper(run_observations):
    """The fix, on the exact sentence that defeated the previous version."""
    obs = run_observations[0]
    found = _state_evidence(_events(run_observations)[0])
    assert found is not None
    _, _, offset, text = found
    label_map = _label_map(run_observations)
    subject = _state_subject(
        text, offset, [*obs["objects"], *label_map], label_map
    )
    assert subject == BOX


def test_the_artwork_label_loses_on_distance_not_on_being_excluded(
    run_observations,
):
    """"Push Chopper box" does have a mention — its head noun "box" — and is a
    real resolved track, so it is a genuine competitor. It loses because
    "cardboard box" starts nearer the word "open" in that sentence, which is the
    whole rule and is checked here as the two offsets rather than asserted.
    """
    text = run_observations[0]["visible_facts"]
    found = _state_evidence(_events(run_observations)[0])
    offset = found[2]
    near = _mention_index("cardboard box", text)
    far = _mention_index("Push Chopper box", text)
    assert far is not None
    assert abs(near - offset) < abs(far - offset)
    assert _label_map(run_observations)["push chopper box"] == CARTON_ARTWORK


def test_the_three_opens_become_closes_on_the_real_bindings(run_observations):
    """The headline, and the assertion the previous version could not make."""
    events = _resolve_state_contradictions(_events(run_observations))
    opens = [e for e in events if e.start_sec >= 1.9 and e.start_sec < 4.0]
    assert len(opens) == 3
    assert [e.action for e in opens] == [ActionType.CLOSE] * 3
    for e in opens:
        assert e.attributes["verb_source"] == "STATE_UNSATISFIABLE"
        assert e.attributes["verb_before_correction"] == "OPEN"
        assert f"{BOX} already OPEN" in e.attributes["state_evidence"]


def test_seeding_the_events_own_object_would_correct_nothing(run_observations):
    """The bug this file exists for, simulated so the fix is not taken on faith.

    Attributing the state to the event's own ``object_track_id`` — the previous
    behaviour — puts OPEN on track 3 and leaves track 6 unseeded, so all three
    OPEN events survive. That is precisely what the live run printed.
    """
    events = _events(run_observations)
    state: dict[int, str] = {}
    verbs = []
    for e in events:
        if e.action is ActionType.OPEN and e.object_track_id is not None:
            verbs.append(
                ActionType.CLOSE
                if state.get(e.object_track_id) == "OPEN"
                else ActionType.OPEN
            )
            continue
        found = _state_evidence(e)
        if found and e.object_track_id is not None:
            state[e.object_track_id] = found[0]
    assert state == {CHOPPER: "OPEN"}
    assert verbs == [ActionType.OPEN] * 3


def test_the_two_reversed_events_are_the_ones_that_flip(run_observations):
    """Against the score report: 2.86 and 3.81 were REVERSED, covering CLOSE."""
    events = _resolve_state_contradictions(_events(run_observations))
    by_start = {round(e.start_sec, 2): e for e in events}
    assert by_start[2.86].action is ActionType.CLOSE
    assert by_start[3.81].action is ActionType.CLOSE


def test_the_events_that_were_already_right_are_untouched(run_observations):
    """GRASP at 0.0 and INSERT at 0.95 are the run's two EXACT verdicts."""
    events = _resolve_state_contradictions(_events(run_observations))
    assert events[0].action is ActionType.GRASP
    assert events[1].action is ActionType.INSERT
    for e in (events[0], events[1], events[5], events[6]):
        assert "verb_source" not in e.attributes


# ──────────────────────────────────────────────────────────────── the rule itself
def test_nearest_mention_wins_between_two_named_objects():
    label_map = {"cardboard box": BOX, "push chopper": CHOPPER}
    text = "the push chopper sits beside an open cardboard box"
    offset = text.index("open")
    assert _state_subject(text, offset, [*label_map], label_map) == BOX

    text = "the open push chopper sits beside a cardboard box"
    offset = text.index("open")
    assert _state_subject(text, offset, [*label_map], label_map) == CHOPPER


def test_a_label_the_text_never_names_cannot_be_chosen():
    label_map = {"dining table": 1}
    text = "an open cardboard box"
    assert _state_subject(text, text.index("open"), [*label_map], label_map) is None


def test_label_matching_reuses_the_usual_matcher():
    """"box" must reach a "cardboard box" track, as it does when binding objects."""
    label_map = {"cardboard box": BOX}
    text = "the box is closed"
    assert _state_subject(text, text.index("closed"), ["box"], label_map) == BOX


def test_the_result_does_not_depend_on_dict_order():
    """Two labels mentioned the same distance away must resolve the same either
    way round, so a reordered ``objects`` list cannot change the answer."""
    text = "an open box box"
    a = _state_subject(text, text.index("open"), ["box", "cardboard box"],
                       {"box": 7, "cardboard box": BOX})
    b = _state_subject(text, text.index("open"), ["cardboard box", "box"],
                       {"cardboard box": BOX, "box": 7})
    assert a == b


def test_opening_as_a_noun_is_still_not_a_state_word(run_observations):
    """Observation 1 says "moves it into the opening of a cardboard box".

    "opening" names an aperture here, not a state that was reached, and the
    boundary in the OPEN pattern already excludes it. If that ever changes this
    run would seed OPEN from the INSERT event too, which happens to give the same
    answer on this clip and would be luck rather than reasoning.
    """
    text = run_observations[1]["visible_facts"]
    assert "opening" in text
    assert _state_evidence(_events(run_observations)[1]) is None


def test_the_fallback_keeps_the_events_own_object_when_nothing_is_named():
    """No resolvable mention must not mean no seed — the old path still applies.

    Without this, a text like "the box is closed" on an event whose label never
    resolved would silently stop seeding, which would be a regression rather than
    a fix.
    """
    def _ev(action, start, obj, vf):
        return PhysicalEvent(
            event_id=f"e{start}", segment_id="s", action=action,
            object_track_id=obj, start_sec=start, end_sec=start + 1.0,
            source="vlm:gemini",
            attributes={"visible_facts": vf, "state_change": "",
                        "objects": [], "object_label": None},
        )

    events = [
        _ev(ActionType.GRASP, 0.0, BOX, "the container is closed"),
        _ev(ActionType.CLOSE, 1.0, BOX, "hands on the flaps"),
    ]
    result = _resolve_state_contradictions(events)
    assert result[1].action is ActionType.OPEN
    assert result[1].attributes["verb_before_correction"] == "CLOSE"
