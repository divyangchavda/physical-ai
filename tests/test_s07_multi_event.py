"""Tests for ActionNormalizer's decomposition of one observation into N events.

ActionNormalizer is NOT the pipeline's event stage — s07_events is. It reached a
thin s07_event_extraction stage that has since been deleted, because that stage
emitted events with no actor_track_id, object_track_id or review_status and so
silently broke graph_builder, state_inferencer, episode_assembler and s11_score.
Measured on the five real Gemini observations in tests/fixtures, this
normalizer's verb rules are also the weaker of the two: 4/5 against 5/5, reading
PUSH out of the object name "push chopper" on the compound sentence.

Kept and tested because its evidence-corroboration rules — refuse an action the
visible facts do not support — are the part s07_events does not have, and are
what "pretending to" and "failing to" in Something-Something V2 will need.
"""

from src.models.action_normalizer import ActionNormalizer
from src.schema.event import ActionType
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus

import pytest


@pytest.fixture
def normalizer():
    return ActionNormalizer()


def _obs(raw_action, facts="", status=VLMSegmentStatus.SUCCESS) -> RawVLMObservation:
    return RawVLMObservation(
        observation_id="obs_1",
        segment_id="seg_1",
        status=status,
        backend="test",
        model_name="test",
        prompt_version="v1",
        segment_start_sec=1.0,
        segment_end_sec=5.0,
        raw_response="",
        raw_action=raw_action,
        visible_facts=facts,
        state_change="",
        inference="",
        uncertainty="",
        start_time_sec=2.0,
        end_time_sec=4.0,
        actor="person",
        active_hand="right",
        objects=["object"],
        confidence=0.9
    )


def test_1_clearly_separable(normalizer):
    obs = _obs("picked up the phone and placed it on the table", facts="lifted phone then put it down")
    events = normalizer.normalize(obs)
    assert len(events) == 2
    assert events[0].action == ActionType.PICK
    assert events[1].action == ActionType.PLACE


def test_2_an_unreadable_clause_does_not_delete_a_readable_one(normalizer):
    """"picked up the phone" -> PICK; "moved it" -> UNKNOWN, since MOVE demands
    explicit evidence the object moved and the facts do not give it.

    This used to assert 1 UNKNOWN: any UNKNOWN clause discarded the whole
    sentence. That rule cost more than it saved -- on the one real observation
    the project has it turned a good INSERT into nothing -- so a clause that
    fails to parse is now dropped and the rest is kept. PICK was derived from
    "lifted phone" and is not made wrong by MOVE being unclear.
    """
    obs = _obs("picked up the phone and moved it", facts="lifted phone but no distinct placement")
    events = normalizer.normalize(obs)
    assert len(events) == 1
    assert events[0].action == ActionType.PICK
    # One surviving clause keeps the VLM's own timestamps. Those bounds are only
    # forfeited when several actions share them and none can claim them.
    assert events[0].attributes["timing_precision"] == "EXACT"


def test_3_three_actions(normalizer):
    """Three clauses, two of which parse.

    Previously this asserted 1 UNKNOWN because the splitter accepted exactly two
    pieces. It now splits on every separator and returns what it could read:
    PICK ("lifted object") and PLACE ("set it down"). The middle clause,
    "carried it", is dropped -- MOVE needs the object's movement stated, and
    "moved it" in the facts is not attributed to an object.
    """
    obs = _obs("picked up the phone, carried it, and placed it on the table", facts="lifted object, moved it, set it down")
    events = normalizer.normalize(obs)
    assert [e.action for e in events] == [ActionType.PICK, ActionType.PLACE]


def test_4_conflicting_evidence_in_part(normalizer):
    """Evidence contradicting one clause must not discredit the other.

    "kept holding it" refutes PLACE, and PLACE is correctly dropped. It says
    nothing about the lift, which "lifted cup" states outright, so PICK stands.
    The old assertion of 1 UNKNOWN threw away the half the evidence supported.
    """
    obs = _obs("picked up the cup and placed it", facts="lifted cup but kept holding it")
    events = normalizer.normalize(obs)
    assert len(events) == 1
    assert events[0].action == ActionType.PICK


def test_5_temporal_limitation_segment_bounds(normalizer):
    obs = _obs("picked up cup and placed it", facts="lifted cup then set it down")
    events = normalizer.normalize(obs)
    assert len(events) == 2
    
    # Since decomposition strips precise timestamps by design for multi-events
    assert events[0].start_sec == 1.0  # segment bounds
    assert events[0].end_sec == 5.0
    assert events[0].attributes["timing_precision"] == "SEGMENT"
    
    assert events[1].start_sec == 1.0
    assert events[1].end_sec == 5.0
    assert events[1].attributes["timing_precision"] == "SEGMENT"


def test_6_failed_observation(normalizer):
    """A non-SUCCESS observation is the caller's to filter, not the normalizer's.

    This and test_7 used to run the deleted s07_event_extraction stage to check
    that FAILED and SKIPPED observations yield no events. The pipeline's own
    stage is covered by
    tests/test_s07_events.py::test_extract_events_skips_failed_observations; what
    is left to pin here is that a null raw_action does not raise.
    """
    obs = _obs(None, status=VLMSegmentStatus.FAILED)
    assert obs.status != VLMSegmentStatus.SUCCESS
    events = normalizer.normalize(obs)
    assert [e.action for e in events] == [ActionType.UNKNOWN]


def test_7_skipped_observation(normalizer):
    obs = _obs(None, status=VLMSegmentStatus.SKIPPED)
    assert obs.status != VLMSegmentStatus.SUCCESS
    events = normalizer.normalize(obs)
    assert [e.action for e in events] == [ActionType.UNKNOWN]


def test_8_success_unknown_action(normalizer):
    obs = _obs("UNKNOWN")
    events = normalizer.normalize(obs)
    assert len(events) == 1
    assert events[0].action == ActionType.UNKNOWN
