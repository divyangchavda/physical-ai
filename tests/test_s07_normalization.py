"""Tests for Stage 07 Action Normalizer."""

import pytest

from src.models.action_normalizer import ActionNormalizer
from src.schema.event import ActionType
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus


@pytest.fixture
def normalizer():
    return ActionNormalizer()


def _obs(raw_action, facts="", state="", inf="", unc="", start=1.0, end=2.0) -> RawVLMObservation:
    return RawVLMObservation(
        observation_id="obs_1",
        segment_id="seg_1",
        status=VLMSegmentStatus.SUCCESS,
        backend="test",
        model_name="test",
        prompt_version="v1",
        segment_start_sec=start,
        segment_end_sec=end,
        raw_response="",
        raw_action=raw_action,
        visible_facts=facts,
        state_change=state,
        inference=inf,
        uncertainty=unc,
        start_time_sec=start,
        end_time_sec=end,
        actor="person",
        active_hand="right",
        objects=["cup"],
        confidence=0.9
    )


def test_grasp_basic(normalizer):
    obs = _obs("grasped cup")
    events = normalizer.normalize(obs)
    assert len(events) == 1
    assert events[0].action == ActionType.GRASP


def test_pick_basic_with_evidence(normalizer):
    obs = _obs("picked up cup", facts="the cup moves upward")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.PICK


def test_pick_ambiguous_no_lifting_evidence(normalizer):
    obs = _obs("picked up cup", facts="the person holds it")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.UNKNOWN


def test_grasp_upgraded_to_pick(normalizer):
    obs = _obs("grabbed the cup", facts="lifted it off the table")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.PICK


def test_false_keyword_not_pick(normalizer):
    obs = _obs("lifted hand toward cup", facts="did not touch the cup")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.UNKNOWN


def test_pick_contradicted_by_evidence(normalizer):
    obs = _obs("picked up the cup", facts="the cup is stationary")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.UNKNOWN


def test_false_move(normalizer):
    obs = _obs("moved hand across table")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.UNKNOWN


def test_true_move(normalizer):
    obs = _obs("moved the cup", facts="object was moved")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.MOVE


def test_state_only_is_unknown(normalizer):
    obs = _obs("UNKNOWN", state="door is open")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.UNKNOWN


def test_transition_is_open(normalizer):
    obs = _obs("opened the door")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.OPEN


def test_multi_action_decomposition(normalizer):
    # If it clearly describes two actions that map securely
    obs = _obs("picked up cup and placed it", facts="lifted cup then set it down")
    events = normalizer.normalize(obs)
    assert len(events) == 2
    assert events[0].action == ActionType.PICK
    assert events[1].action == ActionType.PLACE


def test_multi_action_ambiguous_fallback(normalizer):
    # If one of the parts fails to map securely, the whole observation falls back to UNKNOWN
    obs = _obs("picked up cup and wiggled it", facts="lifted cup and shook it")
    events = normalizer.normalize(obs)
    assert len(events) == 1
    assert events[0].action == ActionType.UNKNOWN


def test_null_raw_action(normalizer):
    obs = _obs(None)
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.UNKNOWN


def test_unknown_raw_action(normalizer):
    obs = _obs("UNKNOWN")
    events = normalizer.normalize(obs)
    assert events[0].action == ActionType.UNKNOWN


def test_timestamp_preservation(normalizer):
    obs = _obs("grasped", start=2.5, end=3.5)
    events = normalizer.normalize(obs)
    assert events[0].start_sec == 2.5
    assert events[0].end_sec == 3.5
    assert events[0].attributes["timing_precision"] == "EXACT"


def test_timestamp_segment_fallback(normalizer):
    obs = _obs("grasped")
    obs.start_time_sec = None
    obs.end_time_sec = None
    events = normalizer.normalize(obs)
    assert events[0].start_sec == 1.0  # segment_start_sec default in _obs
    assert events[0].end_sec == 2.0
    assert events[0].attributes["timing_precision"] == "SEGMENT"


def test_confidence_and_provenance_preservation(normalizer):
    obs = _obs("grasped")
    events = normalizer.normalize(obs)
    assert events[0].confidence == 0.9
    assert events[0].source == "vlm_normalized:test"
    assert events[0].observation_id == "obs_1"
