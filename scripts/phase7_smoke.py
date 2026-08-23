"""Smoke test for Phase 7 (Object State Transitions)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.context import PipelineContext
from src.schema.event import ActionType, PhysicalEvent
from src.schema.track import Track
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus
from src.stages import s08_states

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_smoke_test():
    # 1. Setup mock context
    config = load_config(set_overrides=[
        "state_extraction.enabled=true"
    ])
    config.output_dir = Path("output")
    config.output_dir.mkdir(exist_ok=True)
    
    ctx = PipelineContext(config=config, video_path=Path("mock_video.mp4"), output_dir=config.output_dir)
    
    # Mock VLM observations
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs_1",
            segment_id="seg_1",
            status=VLMSegmentStatus.SUCCESS,
            segment_start_sec=1.0,
            segment_end_sec=5.0,
            actor="person_1",
            active_hand="RIGHT",
            objects=["cup"],
            raw_action="picked up the cup from the table",
            start_time_sec=2.0,
            end_time_sec=3.5,
            state_change=None,
            visible_facts="picked up from table",
            inference=None,
            uncertainty=None,
            confidence=0.9
        ),
        RawVLMObservation(
            observation_id="obs_2",
            segment_id="seg_1",
            status=VLMSegmentStatus.SUCCESS,
            segment_start_sec=1.0,
            segment_end_sec=5.0,
            actor="person_1",
            active_hand="RIGHT",
            objects=["cup"],
            raw_action="placed it",
            start_time_sec=4.0,
            end_time_sec=4.5,
            state_change=None,
            visible_facts="placed it",
            inference=None,
            uncertainty=None,
            confidence=0.8
        )
    ]

    ctx.events = [
        PhysicalEvent(
            event_id="evt_1",
            segment_id="seg_1",
            observation_id="obs_1",
            action=ActionType.PICK,
            confidence=0.9,
            source="rule_based",
            start_sec=2.0,
            end_sec=3.5,
            attributes={"timing_precision": "EXACT"}
        ),
        PhysicalEvent(
            event_id="evt_2",
            segment_id="seg_1",
            observation_id="obs_2",
            action=ActionType.PLACE,
            confidence=0.8,
            source="rule_based",
            start_sec=4.0,
            end_sec=4.5,
            attributes={"timing_precision": "EXACT"}
        )
    ]
    
    ctx.tracks = [
        Track(
            track_id=17,
            class_id=1,
            class_name="cup",
            confidence=0.9,
            points=[],
            start_frame=0,
            end_frame=100,
            start_sec=0.0,
            end_sec=10.0,
            source="test"
        )
    ]
    
    from src.schema.segment import CandidateSegment
    ctx.candidate_segments = [
        CandidateSegment(
            segment_id="seg_1",
            start_frame=0,
            end_frame=100,
            start_sec=1.0,
            end_sec=5.0,
            track_ids=[17],
            score=0.9
        )
    ]

    # 2. Run state extraction
    logger.info("Running s08_states...")
    status = s08_states.run(ctx)
    assert status.status == "OK"
    
    # 3. Assertions
    transitions = ctx.state_transitions
    assert len(transitions) == 2
    
    # Event 1 -> PICK
    assert transitions[0].trigger_event_id == "evt_1"
    assert transitions[0].from_state == "ON_SURFACE" # because explicit evidence
    assert transitions[0].to_state == "IN_HAND"
    assert transitions[0].identity_resolution == "RESOLVED"
    assert transitions[0].track_id == 17
    assert transitions[0].confidence == 0.9
    
    # Event 2 -> PLACE
    assert transitions[1].trigger_event_id == "evt_2"
    assert transitions[1].from_state == "IN_HAND" # carried over from before state tracking
    assert transitions[1].to_state == "ON_SURFACE"
    assert transitions[1].confidence == 0.8
    
    logger.info("Phase 7 Smoke Test Passed!")

if __name__ == "__main__":
    run_smoke_test()
