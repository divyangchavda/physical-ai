"""Smoke test for Phase 6 (Event Extraction / Normalization)."""

import logging
from pathlib import Path

from src.config import load_config
from src.context import PipelineContext
from src.schema.event import ActionType
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus
from src.stages import s07_event_extraction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_smoke_test():
    # 1. Setup mock context
    config = load_config(set_overrides=[
        "event_extraction.enabled=true"
    ])
    config.output_dir = Path("output")
    config.output_dir.mkdir(exist_ok=True)
    
    ctx = PipelineContext(config=config, video_path=Path("mock_video.mp4"), output_dir=config.output_dir)
    
    # Mock VLM observations
    ctx.vlm_observations = [
        # Observation 1: Standard pick (has lifting evidence)
        RawVLMObservation(
            observation_id="obs_1",
            segment_id="seg_1",
            status=VLMSegmentStatus.SUCCESS,
            segment_start_sec=1.0,
            segment_end_sec=5.0,
            actor="person_1",
            active_hand="RIGHT",
            objects=["cup"],
            raw_action="picked up the cup",
            start_time_sec=2.0,
            end_time_sec=3.5,
            state_change=None,
            visible_facts="lifted upward off table",
            inference=None,
            uncertainty=None,
            confidence=0.9
        ),
        # Observation 2: Ambiguous pick (no lifting evidence) -> UNKNOWN
        RawVLMObservation(
            observation_id="obs_2",
            segment_id="seg_2",
            status=VLMSegmentStatus.SUCCESS,
            segment_start_sec=10.0,
            segment_end_sec=15.0,
            actor="person_1",
            active_hand="RIGHT",
            objects=["box"],
            raw_action="picked up box",
            start_time_sec=11.0,
            end_time_sec=12.0,
            state_change=None,
            visible_facts="holding box",
            inference=None,
            uncertainty=None,
            confidence=0.7
        ),
        # Observation 3: Multi-event
        RawVLMObservation(
            observation_id="obs_3",
            segment_id="seg_3",
            status=VLMSegmentStatus.SUCCESS,
            segment_start_sec=20.0,
            segment_end_sec=25.0,
            actor="person_1",
            active_hand="LEFT",
            objects=["phone"],
            raw_action="picked up phone and placed it",
            start_time_sec=21.0,
            end_time_sec=24.0,
            state_change=None,
            visible_facts="lifted up phone then set down",
            inference=None,
            uncertainty=None,
            confidence=0.8
        ),
        # Observation 4: FAILED
        RawVLMObservation(
            observation_id="obs_4",
            segment_id="seg_4",
            status=VLMSegmentStatus.FAILED,
            error_reason="JSON decoding failed",
            segment_start_sec=30.0,
            segment_end_sec=35.0,
            actor=None,
            active_hand=None,
            objects=[],
            raw_action=None,
            start_time_sec=None,
            end_time_sec=None,
            state_change=None,
            visible_facts=None,
            inference=None,
            uncertainty=None
        ),
    ]

    # 2. Run event extraction
    logger.info("Running s07_event_extraction...")
    status = s07_event_extraction.run(ctx)
    assert status.status == "OK"
    
    # 3. Assertions
    events = ctx.events
    assert len(events) == 4  # obs_1 yields 1, obs_2 yields 1, obs_3 yields 2, obs_4 yields 0
    
    # Event 1 -> PICK
    assert events[0].observation_id == "obs_1"
    assert events[0].action == ActionType.PICK
    assert events[0].start_sec == 2.0
    
    # Event 2 -> UNKNOWN (ambiguous pick)
    assert events[1].observation_id == "obs_2"
    assert events[1].action == ActionType.UNKNOWN
    
    # Event 3 -> PICK
    assert events[2].observation_id == "obs_3"
    assert events[2].action == ActionType.PICK
    
    # Event 4 -> PLACE
    assert events[3].observation_id == "obs_3"
    assert events[3].action == ActionType.PLACE
    
    logger.info("Phase 6 Smoke Test Passed!")

if __name__ == "__main__":
    run_smoke_test()
