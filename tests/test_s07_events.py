"""Unit tests for Stage 07 (Physical Event Extraction)."""
from __future__ import annotations

from pathlib import Path

from src.schema.detection import BoundingBox
from src.schema.event import PhysicalEvent, ActionType
from src.schema.segment import CandidateSegment
from src.schema.track import Track, TrackPoint
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus
from src.stages.s07_events import (
    _map_raw_action_to_type,
    _match_action,
    _order_objects_by_action,
    _extract_events_from_vlm_observations,
)
from src.context import PipelineContext
from src.config import PipelineConfig


def _make_track(track_id: int, class_name: str, n_points: int = 3) -> Track:
    """Minimal Track with *n_points* points — point count breaks resolver ties."""
    points = [
        TrackPoint(
            frame_index=i,
            timestamp_sec=i * 0.1,
            bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
            detection_confidence=0.9,
            tracking_confidence=0.9,
        )
        for i in range(n_points)
    ]
    return Track(
        track_id=track_id,
        class_name=class_name,
        class_id=0,
        points=points,
        start_frame=0,
        end_frame=n_points - 1,
        start_sec=0.0,
        end_sec=(n_points - 1) * 0.1,
        source="test",
        is_estimated=True,
    )


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


def test_a_preposition_refines_a_generic_verb():
    """"Placing X inside Y" is an INSERT. The verb alone cannot tell you.

    These four strings are the real Gemini output for tt6, in
    tests/fixtures/tt6_vlm_observations.json. Scored against the hand labels,
    Gemini's direction was 2/4 while the pipeline emitted 0/4 — the stem table
    matched "plac" and discarded the "inside"/"into" the model had already got
    right, shipping PLACE (direction ONTO) for a containment action.
    """
    objects = ["push chopper", "cardboard box"]
    assert _map_raw_action_to_type(
        "placing the push chopper inside the cardboard box", objects
    ) == ActionType.INSERT
    assert _map_raw_action_to_type(
        "placing the push chopper back into the box", objects
    ) == ActionType.INSERT
    # No containment preposition: PLACE is the honest answer and must survive.
    assert _map_raw_action_to_type(
        "placing the cardboard box on the dining table",
        ["cardboard box", "dining table"],
    ) == ActionType.PLACE
    assert _map_raw_action_to_type("placing the object down") == ActionType.PLACE
    assert _map_raw_action_to_type(
        "picking the push chopper out of the cardboard box", objects
    ) == ActionType.REMOVE


def test_a_preposition_cannot_cross_a_clause_break():
    """The preposition must belong to the clause of the verb that matched.

    "removes the chopper from the box and then places the box back down" already
    maps to REMOVE by stem precedence. The promotion logic must read only the
    matched verb's own clause, or a later verb's preposition could rewrite an
    earlier action.
    """
    objects = ["push chopper", "cardboard box"]
    assert _map_raw_action_to_type(
        "the person removes the push chopper from the cardboard box "
        "and then places the box back down",
        objects,
    ) == ActionType.REMOVE
    # "placing" matches first here; the "out of" sits past the clause break and
    # describes a different action, so it must not promote this to REMOVE.
    assert _map_raw_action_to_type(
        "placing the box down, then lifting the chopper out of the container",
        ["box", "chopper", "container"],
    ) == ActionType.PLACE


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
    
    # Identities are resolved by class against ctx.tracks, so the tracks must
    # exist. Track 2 is the person and track 1 the object on purpose: a
    # position-based resolver would pick track 1 as the actor and get it wrong.
    ctx.tracks = [
        _make_track(1, "box", n_points=3),
        _make_track(2, "person", n_points=5),
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
    assert event.actor_track_id == 2   # the "person" track, not track_ids[0]
    assert event.object_track_id == 1  # "box" matched to the box track
    assert event.start_sec == 11.0
    assert event.end_sec == 15.0
    assert event.review_status == "PENDING"

    # Check attributes
    assert event.attributes["raw_action"] == "picking up the box"
    assert event.attributes["actor"] == "person"
    assert event.attributes["active_hand"] == "BOTH"
    assert event.attributes["objects"] == ["box"]
    assert event.attributes["object_label"] == "box"
    assert event.attributes["state_change"] == "box moved"
    # 11.0-15.0 sits strictly inside the 10.0-20.0 segment: real localisation.
    assert event.attributes["timing_precision"] == "EXACT"


def test_timing_precision_segment_when_vlm_echoes_clip_bounds():
    """VLM returning the clip bounds is describing the clip, not timing it."""
    config = PipelineConfig(stub_mode=False)
    ctx = PipelineContext(
        config=config,
        video_path=Path("test.mp4"),
        output_dir=Path("output"),
    )
    ctx.candidate_segments = [
        CandidateSegment(
            segment_id="seg_001",
            track_ids=[1],
            start_frame=0,
            end_frame=100,
            start_sec=0.0,
            end_sec=7.0,
            trigger_reason="test",
            confidence=0.8,
            source="test",
            status="PENDING",
        )
    ]
    ctx.tracks = [_make_track(1, "person")]
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs_001",
            segment_id="seg_001",
            status=VLMSegmentStatus.SUCCESS,
            backend="GEMINI",
            model_name="gemini-test",
            segment_start_sec=0.0,
            segment_end_sec=7.0,
            actor="person",
            active_hand="RIGHT",
            objects=["cardboard box"],
            raw_action="placing the box",
            start_time_sec=0.0,   # == segment bounds
            end_time_sec=7.0,
            state_change="box on table",
            visible_facts="hand on box",
            inference="placement",
            uncertainty="none",
            confidence=0.9,
        )
    ]

    events = _extract_events_from_vlm_observations(ctx)
    assert len(events) == 1
    assert events[0].attributes["timing_precision"] == "SEGMENT"


def test_actor_unresolved_when_segment_has_no_person():
    """No person track in the segment -> None, not a guess at another class."""
    config = PipelineConfig(stub_mode=False)
    ctx = PipelineContext(
        config=config,
        video_path=Path("test.mp4"),
        output_dir=Path("output"),
    )
    ctx.candidate_segments = [
        CandidateSegment(
            segment_id="seg_001",
            track_ids=[7, 8],
            start_frame=0,
            end_frame=100,
            start_sec=0.0,
            end_sec=5.0,
            trigger_reason="test",
            confidence=0.8,
            source="test",
            status="PENDING",
        )
    ]
    # A background class and an object — neither may stand in for the actor.
    ctx.tracks = [
        _make_track(7, "dining table"),
        _make_track(8, "cardboard box"),
    ]
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs_001",
            segment_id="seg_001",
            status=VLMSegmentStatus.SUCCESS,
            backend="GEMINI",
            model_name="gemini-test",
            segment_start_sec=0.0,
            segment_end_sec=5.0,
            actor="person",
            active_hand="RIGHT",
            objects=["box", "dining table"],
            raw_action="pushing the box",
            start_time_sec=1.0,
            end_time_sec=4.0,
            state_change="box slid across the table",
            visible_facts="hand against the box",
            inference="push",
            uncertainty="none",
            confidence=0.9,
        )
    ]

    events = _extract_events_from_vlm_observations(ctx)
    assert len(events) == 1
    assert events[0].actor_track_id is None
    # "box" is a substring of "cardboard box", and the table is excluded as scene.
    assert events[0].object_track_id == 8
    assert events[0].attributes["object_label"] == "box"


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



def test_compound_action_object_comes_from_the_matched_clause():
    """Event 2 of run_20260824_055411 took the object from the wrong clause."""
    objects = ["cardboard box", "push chopper"]
    action = ("opening and closing the cardboard box, then removing the "
              "push chopper from the box")
    # "removing" is the matched verb, so the chopper is the object even though
    # the box is mentioned first — ranking by absolute position picked the box.
    act, verb_index = _match_action(action, objects)
    assert act == ActionType.REMOVE
    assert _order_objects_by_action(action, objects, verb_index)[0] == "push chopper"

    # Blanking the object names must not shift the verb offset.
    assert action[verb_index:].startswith("removing")


def test_verb_offset_survives_multiple_clauses():
    """A second clause after the verb must not steal the object role."""
    objects = ["cardboard box", "push chopper"]
    action = ("the person removes the push chopper from the cardboard box "
              "and then places the box back down")
    act, verb_index = _match_action(action, objects)
    assert act == ActionType.REMOVE
    assert _order_objects_by_action(action, objects, verb_index)[0] == "push chopper"


def test_verb_is_not_read_out_of_an_object_name():
    """Event 4 of run_20260824_053624: "unboxing a push chopper" scored PUSH."""
    # The object's name carries a verb, so matching against the whole string
    # made every action on a push chopper look like a PUSH.
    objects = ["cardboard box", "push chopper"]
    assert _map_raw_action_to_type("unboxing a push chopper", objects) == ActionType.REMOVE
    # Was asserted as PLACE when this test was written. The hand labels in
    # tests/fixtures/tt6_ground_truth.json call it INSERT, and they win: "into"
    # is a containment relation the verb alone does not carry. What this test
    # actually guards is that "push" is not read out of "push chopper", which
    # holds either way.
    assert _map_raw_action_to_type(
        "placing the push chopper into the cardboard box", objects
    ) == ActionType.INSERT
    # A real push of that object still reads as PUSH.
    assert _map_raw_action_to_type(
        "pushing the push chopper across the table", objects
    ) == ActionType.PUSH
    # Compound verb keeps its stronger reading rather than becoming REMOVE.
    assert _map_raw_action_to_type(
        "opening and unpacking a cardboard box", objects
    ) == ActionType.OPEN


def test_head_noun_match_respects_word_boundaries():
    """"box" must not match inside "unboxing" and hand the box the object role."""
    assert _order_objects_by_action(
        "unboxing a push chopper", ["cardboard box", "push chopper"]
    )[0] == "push chopper"
    # The same head-noun fallback still works when it is a real word.
    assert _order_objects_by_action(
        "unboxing the cardboard box", ["push chopper", "cardboard box"]
    )[0] == "cardboard box"


def test_order_objects_by_action():
    """The manipulated object beats the container, whatever the list order."""
    # Both cases are verbatim from run_20260824_051032, where trusting the
    # VLM's list order named the container as the manipulated object.
    assert _order_objects_by_action(
        "placing the push chopper back into the box", ["box", "push chopper"]
    )[0] == "push chopper"
    assert _order_objects_by_action(
        "removes the push chopper from the cardboard box",
        ["cardboard box", "push chopper"],
    )[0] == "push chopper"
    # Head-noun fallback: the action says "box", the list says "cardboard box".
    assert _order_objects_by_action(
        "picking up the box", ["dining table", "cardboard box"]
    )[0] == "cardboard box"
    # Destination named first is still a destination.
    assert _order_objects_by_action(
        "moving the chopper to the table", ["table", "chopper"]
    )[0] == "chopper"
    # No action text, or none of the objects mentioned -> original order.
    assert _order_objects_by_action("", ["box", "chopper"]) == ["box", "chopper"]
    assert _order_objects_by_action("doing something", ["box", "chopper"]) == [
        "box", "chopper"
    ]


def test_object_is_the_thing_moved_not_the_container():
    """Event 3 of run_20260824_051032: chopper into box resolved to the box."""
    config = PipelineConfig(stub_mode=False)
    ctx = PipelineContext(
        config=config,
        video_path=Path("test.mp4"),
        output_dir=Path("output"),
    )
    ctx.candidate_segments = [
        CandidateSegment(
            segment_id="seg_003",
            track_ids=[40, 44],
            start_frame=0,
            end_frame=100,
            start_sec=16.6,
            end_sec=24.9,
            trigger_reason="test",
            confidence=0.8,
            source="test",
            status="PENDING",
        )
    ]
    # The box track is longer-lived than the chopper, so a tie-break on point
    # count alone would also pick the box. Only the action text disambiguates.
    ctx.tracks = [
        _make_track(40, "cardboard box", n_points=30),
        _make_track(44, "push chopper", n_points=5),
    ]
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs_003",
            segment_id="seg_003",
            status=VLMSegmentStatus.SUCCESS,
            backend="GEMINI",
            model_name="gemini-test",
            segment_start_sec=16.6,
            segment_end_sec=24.9,
            actor="person",
            active_hand="RIGHT",
            objects=["box", "push chopper"],
            raw_action="placing the push chopper back into the box",
            start_time_sec=16.6,
            end_time_sec=24.9,
            state_change="chopper inside the box",
            visible_facts="hand holding the chopper over the box",
            inference="placement into container",
            uncertainty="none",
            confidence=0.9,
        )
    ]

    events = _extract_events_from_vlm_observations(ctx)
    assert len(events) == 1
    assert events[0].object_track_id == 44
    assert events[0].attributes["object_label"] == "push chopper"


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
