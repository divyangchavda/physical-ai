"""Unit tests for Stage 07 (Physical Event Extraction)."""
from __future__ import annotations

from pathlib import Path

from src.schema.event import PhysicalEvent, ActionType
from src.schema.segment import CandidateSegment
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus
from src.stages.s07_events import _map_raw_action_to_type, _extract_events_from_vlm_observations
from src.context import PipelineContext
from src.config import PipelineConfig


def test_map_raw_action_to_type():
    """Test raw action string mapping to ActionType enum."""
    assert _map_raw_action_to_type("picking up a cup") == ActionType.PICK
    assert _map_raw_action_to_type("placing the object") == ActionType.PLACE
    assert _map_raw_action_to_type("grasping the handle") == ActionType.GRASP
    assert _map_raw_action_to_type("releasing the item") == ActionType.RELEASE
    assert _map_raw_action_to_type("pushing the door") == ActionType.PUSH
    assert _map_raw_action_to_type("pulling the lever") == ActionType.PULL
    assert _map_raw_action_to_type("opening the box") == ActionType.OPEN
    assert _map_raw_action_to_type("closing the lid") == ActionType.CLOSE
    assert _map_raw_action_to_type("inserting the key") == ActionType.INSERT
    assert _map_raw_action_to_type("removing the cap") == ActionType.REMOVE
    assert _map_raw_action_to_type("using a screwdriver") == ActionType.USE_TOOL
    assert _map_raw_action_to_type("touching the surface") == ActionType.TOUCH
    assert _map_raw_action_to_type("inspecting the product") == ActionType.INSPECT
    # Complex actions should map to UNKNOWN
    assert _map_raw_action_to_type("folding and assembling a box") == ActionType.UNKNOWN
    assert _map_raw_action_to_type("") == ActionType.UNKNOWN
    assert _map_raw_action_to_type("doing something unclear") == ActionType.UNKNOWN


def test_extract_events_from_vlm_observations():
    """Test conversion of VLM observations to PhysicalEvents."""
    # Setup minimal context
    config = PipelineConfig(stub_mode=False)
    ctx = PipelineContext(
        config=config,
        video_path=Path("test.mp4"),
        output_dir=Path("output")
    )
    
    # Create candidate segment
    ctx.candidate_segments = [
        CandidateSegment(
            segment_id="seg_001",
            track_ids=[1, 2],
            start_frame=100,
            end_frame=200,
            start_sec=10.0,
            end_sec=20.0,
            trigger_reason="test",
            confidence=0.8,
            source="test",
            status="PENDING"
        )
    ]
    
    # Create VLM observation with SUCCESS status
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs_001",
            segment_id="seg_001",
            status=VLMSegmentStatus.SUCCESS,
            backend="GEMINI",
            model_name="gemini-test",
            segment_start_sec=10.0,
            segment_end_sec=20.0,
            actor="person",
            active_hand="BOTH",
            objects=["box"],
            raw_action="picking up the box",
            start_time_sec=11.0,
            end_time_sec=15.0,
            state_change="box moved",
            visible_facts="hands on box",
            inference="pickup action",
            uncertainty="none",
            confidence=0.9
        )
    ]
    
    # Extract events
    events = _extract_events_from_vlm_observations(ctx)
    
    # Assertions
    assert len(events) == 1
    event = events[0]
    
    assert event.segment_id == "seg_001"
    assert event.observation_id == "obs_001"
    assert event.action == ActionType.PICK
    assert event.confidence == 0.9
    assert event.source == "vlm:gemini"
    assert event.is_estimated is True
    assert event.actor_track_id == 1  # First track from segment
    assert event.object_track_id is None
    assert event.start_sec == 11.0
    assert event.end_sec == 15.0
    assert event.review_status == "PENDING"
    
    # Check attributes
    assert event.attributes["raw_action"] == "picking up the box"
    assert event.attributes["actor"] == "person"
    assert event.attributes["active_hand"] == "BOTH"
    assert event.attributes["objects"] == ["box"]
    assert event.attributes["state_change"] == "box moved"


def test_extract_events_skips_failed_observations():
    """Test that FAILED observations are skipped."""
    config = PipelineConfig(stub_mode=False)
    ctx = PipelineContext(
        config=config,
        video_path=Path("test.mp4"),
        output_dir=Path("output")
    )
    
    ctx.candidate_segments = []
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs_failed",
            segment_id="seg_001",
            status=VLMSegmentStatus.FAILED,
            error_reason="API error",
            backend="GEMINI",
            segment_start_sec=10.0,
            segment_end_sec=20.0
        )
    ]
    
    events = _extract_events_from_vlm_observations(ctx)
    assert len(events) == 0


def test_extract_events_complex_action():
    """Test that complex actions like 'folding' map to UNKNOWN."""
    config = PipelineConfig(stub_mode=False)
    ctx = PipelineContext(
        config=config,
        video_path=Path("test.mp4"),
        output_dir=Path("output")
    )
    
    ctx.candidate_segments = [
        CandidateSegment(
            segment_id="seg_002",
            track_ids=[1],
            start_frame=100,
            end_frame=200,
            start_sec=13.0,
            end_sec=29.0,
            trigger_reason="test",
            confidence=0.8,
            source="test",
            status="PENDING"
        )
    ]
    
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs_002",
            segment_id="seg_002",
            status=VLMSegmentStatus.SUCCESS,
            backend="GEMINI",
            model_name="gemini-test",
            segment_start_sec=13.0,
            segment_end_sec=29.0,
            actor="person in plaid shirt",
            active_hand="BOTH",
            objects=["cardboard box"],
            raw_action="folding and assembling the sides of a cardboard box",
            start_time_sec=13.0,
            end_time_sec=29.0,
            state_change="flat cardboard sheet is transformed into a box",
            visible_facts="hands pressing and folding edges",
            inference="assembling packaging box",
            uncertainty="none",
            confidence=1.0
        )
    ]
    
    events = _extract_events_from_vlm_observations(ctx)
    
    assert len(events) == 1
    event = events[0]
    assert event.action == ActionType.UNKNOWN  # Complex action
    assert event.confidence == 1.0
    assert event.attributes["raw_action"] == "folding and assembling the sides of a cardboard box"



if __name__ == "__main__":
    print("Running S07 Event Extraction Tests...\n")
    
    # Test 1
    print("Test 1: Map raw action to ActionType")
    try:
        test_map_raw_action_to_type()
        print("✓ PASSED\n")
    except AssertionError as e:
        import traceback
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        print()
    except Exception as e:
        import traceback
        print(f"✗ ERROR: {e}")
        traceback.print_exc()
        print()
    
    # Test 2
    print("Test 2: Extract events from VLM observations")
    try:
        test_extract_events_from_vlm_observations()
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
    
    # Test 3
    print("Test 3: Skip FAILED observations")
    try:
        test_extract_events_skips_failed_observations()
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
    
    # Test 4
    print("Test 4: Complex action maps to UNKNOWN")
    try:
        test_extract_events_complex_action()
        print("✓ PASSED\n")
    except Exception as e:
        print(f"✗ FAILED: {e}\n")
    
    print("All tests completed!")
