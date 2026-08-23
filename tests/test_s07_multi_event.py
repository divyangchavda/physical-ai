"""Tests for Stage 07 Multi-Event Normalization logic."""

from pathlib import Path

import pytest

from src.config import EventExtractionConfig, PipelineConfig
from src.context import PipelineContext
from src.models.action_normalizer import ActionNormalizer
from src.schema.event import ActionType
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus
from src.stages.s07_event_extraction import run as run_s07


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


def test_2_ambiguous_cannot_reliably_decompose(normalizer):
    obs = _obs("picked up the phone and moved it", facts="lifted phone but no distinct placement")
    events = normalizer.normalize(obs)
    # The normalizer attempts to parse "picked up the phone" -> PICK, "moved it" -> MOVE.
    # However, "moved it" doesn't have "object was moved" evidence in facts, so it might return UNKNOWN for split 2.
    # If one part is UNKNOWN, the entire decomposition fails and falls back to 1 UNKNOWN event.
    assert len(events) == 1
    assert events[0].action == ActionType.UNKNOWN


def test_3_three_actions(normalizer):
    obs = _obs("picked up the phone, carried it, and placed it on the table", facts="lifted object, moved it, set it down")
    events = normalizer.normalize(obs)
    # The regex split might yield 2 or 3 pieces. Currently it only accepts exactly 2 splits.
    # So it should conservatively fall back to 1 UNKNOWN event.
    assert len(events) == 1
    assert events[0].action == ActionType.UNKNOWN


def test_4_conflicting_evidence_in_part(normalizer):
    obs = _obs("picked up the cup and placed it", facts="lifted cup but kept holding it")
    events = normalizer.normalize(obs)
    # "placed it" but evidence says "kept holding it" (no placement).
    # This contradicts PLACE -> UNKNOWN for the second part -> whole thing collapses to 1 UNKNOWN event.
    assert len(events) == 1
    assert events[0].action == ActionType.UNKNOWN


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


def test_6_failed_observation():
    obs = _obs(None, status=VLMSegmentStatus.FAILED)
    ctx = PipelineContext(config=PipelineConfig(output_dir=Path("output"), event_extraction=EventExtractionConfig(enabled=True)), video_path=Path("vid.mp4"), output_dir=Path("output"))
    ctx.vlm_observations = [obs]
    run_s07(ctx)
    assert len(ctx.events) == 0


def test_7_skipped_observation():
    obs = _obs(None, status=VLMSegmentStatus.SKIPPED)
    ctx = PipelineContext(config=PipelineConfig(output_dir=Path("output"), event_extraction=EventExtractionConfig(enabled=True)), video_path=Path("vid.mp4"), output_dir=Path("output"))
    ctx.vlm_observations = [obs]
    run_s07(ctx)
    assert len(ctx.events) == 0


def test_8_success_unknown_action(normalizer):
    obs = _obs("UNKNOWN")
    events = normalizer.normalize(obs)
    assert len(events) == 1
    assert events[0].action == ActionType.UNKNOWN
