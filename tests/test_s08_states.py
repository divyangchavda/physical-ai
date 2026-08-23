"""Tests for Stage 08 State Transitions."""

import pytest

from src.models.state_inferencer import StateInferencer
from src.schema.event import ActionType, PhysicalEvent
from src.schema.track import Track
from src.schema.vlm import RawVLMObservation


@pytest.fixture
def inferencer():
    return StateInferencer()

def _obs(obs_id="obs_1", facts="", raw_action="action", objects=None):
    return RawVLMObservation(
        observation_id=obs_id,
        segment_id="seg_1",
        status="SUCCESS",
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
        objects=objects or ["cup"],
        confidence=0.9
    )

def _evt(evt_id="evt_1", obs_id="obs_1", action=ActionType.UNKNOWN, conf=0.8, start=2.0, end=4.0, timing="EXACT"):
    return PhysicalEvent(
        event_id=evt_id,
        segment_id="seg_1",
        observation_id=obs_id,
        action=action,
        confidence=conf,
        source="rule_based",
        start_sec=start,
        end_sec=end,
        attributes={"timing_precision": timing}
    )

def _track(tid=1, name="cup"):
    return Track(
        track_id=tid,
        class_id=0,
        class_name=name,
        confidence=0.9,
        points=[],
        start_frame=0,
        end_frame=100,
        start_sec=0.0,
        end_sec=10.0,
        source="test"
    )

def test_closed_to_open_with_evidence(inferencer):
    evt = _evt(action=ActionType.OPEN)
    obs = _obs(facts="the door was closed and opened")
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert len(trans) == 1
    assert trans[0].from_state == "CLOSED"
    assert trans[0].to_state == "OPEN"

def test_unknown_to_open_no_evidence(inferencer):
    evt = _evt(action=ActionType.OPEN)
    obs = _obs(facts="the door is opened") # No "was closed"
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].from_state == "UNKNOWN"
    assert trans[0].to_state == "OPEN"

def test_unknown_to_closed_no_evidence(inferencer):
    evt = _evt(action=ActionType.CLOSE)
    obs = _obs(facts="door shuts")
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].from_state == "UNKNOWN"
    assert trans[0].to_state == "CLOSED"

def test_unknown_to_in_hand_no_evidence(inferencer):
    evt = _evt(action=ActionType.PICK)
    obs = _obs(facts="picked it up")
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].from_state == "UNKNOWN"
    assert trans[0].to_state == "IN_HAND"
    
def test_on_surface_to_in_hand_with_evidence(inferencer):
    evt = _evt(action=ActionType.PICK)
    obs = _obs(facts="picked up from table")
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].from_state == "ON_SURFACE"
    assert trans[0].to_state == "IN_HAND"

def test_in_hand_to_unknown_place_no_evidence(inferencer):
    evt = _evt(action=ActionType.PLACE)
    obs = _obs(facts="placed it")
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].from_state == "UNKNOWN"
    assert trans[0].to_state == "ON_SURFACE"

def test_multi_event_intermediate_states(inferencer):
    # PICK then PLACE in same segment
    e1 = _evt(evt_id="e1", action=ActionType.PICK, start=1.0, end=2.0)
    e2 = _evt(evt_id="e2", action=ActionType.PLACE, start=3.0, end=4.0)
    obs = _obs(facts="picked it up and put it down")
    trans = inferencer.infer_transitions([e1, e2], {obs.observation_id: obs}, [])
    
    assert len(trans) == 2
    # e1: no explicit evidence of ON_SURFACE, so UNKNOWN -> IN_HAND
    assert trans[0].from_state == "UNKNOWN"
    assert trans[0].to_state == "IN_HAND"
    # e2: memory maintains IN_HAND across the segment
    assert trans[1].from_state == "IN_HAND"
    assert trans[1].to_state == "ON_SURFACE"

def test_unresolved_identity(inferencer):
    evt = _evt(action=ActionType.PICK)
    obs = _obs(objects=["cup"])
    # Zero cups in tracks
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].track_id is None
    assert trans[0].semantic_label == "cup"
    assert trans[0].identity_resolution == "UNRESOLVED"
    assert trans[0].confidence == evt.confidence  # exact confidence preserved

def test_ambiguous_identity(inferencer):
    evt = _evt(action=ActionType.PICK)
    obs = _obs(objects=["cup"])
    tracks = [_track(1, "cup"), _track(2, "cup")]
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, tracks)
    
    assert trans[0].track_id is None
    assert trans[0].semantic_label == "cup"
    assert trans[0].identity_resolution == "AMBIGUOUS"

def test_resolved_identity(inferencer):
    evt = _evt(action=ActionType.PICK)
    obs = _obs(objects=["cup"])
    tracks = [_track(17, "cup")]
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, tracks)
    
    assert trans[0].track_id == 17
    assert trans[0].semantic_label == "cup"
    assert trans[0].identity_resolution == "RESOLVED"

def test_timing_precision_preservation(inferencer):
    evt = _evt(action=ActionType.PICK, start=1.0, end=5.0, timing="SEGMENT")
    obs = _obs()
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].start_sec == 1.0
    assert trans[0].end_sec == 5.0
    assert trans[0].timing_precision == "SEGMENT"

def test_conflicting_evidence_pick_stationary(inferencer):
    evt = _evt(action=ActionType.PICK)
    obs = _obs(facts="object remained stationary")
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].from_state == "UNKNOWN"
    assert trans[0].to_state == "UNKNOWN"
    assert "stationary" in trans[0].evidence["visible_facts"]

def test_weak_tracking_movement_does_not_conflict(inferencer):
    # Only explicit contradictory text invalidates PICK
    evt = _evt(action=ActionType.PICK)
    obs = _obs(facts="lifted the object") # no explicit contradiction
    trans = inferencer.infer_transitions([evt], {obs.observation_id: obs}, [])
    
    assert trans[0].to_state == "IN_HAND"
