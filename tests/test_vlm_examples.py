"""Test the exact examples requested by the user to verify schema semantics."""
from src.schema.vlm import RawVLMObservation


def test_user_example_1():
    data = {
        "observation_id": "obs_ex1",
        "segment_id": "seg_01",
        "segment_start_sec": 42.0,
        "segment_end_sec": 47.0,
        "status": "SUCCESS",
        "actor": "person_01",
        "active_hand": "UNKNOWN",
        "objects": [],
        "raw_action": "UNKNOWN",
        "start_time_sec": None,
        "end_time_sec": None,
        "state_change": None,
        "visible_facts": "The person is near an object.",
        "inference": None,
        "uncertainty": "The actual manipulation is not visible.",
        "confidence": 0.25
    }
    obs = RawVLMObservation.model_validate(data)
    assert obs.status == "SUCCESS"
    assert obs.actor == "person_01"
    assert obs.active_hand == "UNKNOWN"
    assert obs.start_time_sec is None
    assert obs.inference is None


def test_user_example_2():
    data = {
        "observation_id": "obs_ex2",
        "segment_id": "seg_02",
        "segment_start_sec": 42.0,
        "segment_end_sec": 47.0,
        "status": "SUCCESS",
        "actor": "person_01",
        "active_hand": "UNKNOWN",
        "objects": ["cup"],
        "raw_action": "picked up the cup",
        "start_time_sec": 43.2, # Note: user example had 1.2 which is relative, s06 converts to absolute
        "end_time_sec": 44.4,   # User example had 2.4 (relative)
        "state_change": None,
        "visible_facts": "The person's hand contacts the cup and the cup moves upward.",
        "inference": "The person appears to pick up the cup.",
        "uncertainty": "Hand orientation is partially occluded.",
        "confidence": 0.72
    }
    obs = RawVLMObservation.model_validate(data)
    assert obs.status == "SUCCESS"
    assert obs.objects == ["cup"]
    assert obs.start_time_sec == 43.2
