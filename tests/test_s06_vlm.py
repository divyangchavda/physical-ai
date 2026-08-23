"""Unit tests for Stage 06 VLM Semantic Analysis."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.context import PipelineContext
from src.schema.segment import CandidateSegment
from src.schema.vlm import VLMSegmentStatus
from src.stages import s06_vlm


@pytest.fixture
def mock_ctx(tmp_path):
    config = PipelineConfig(stub_mode=False)
    config.vlm.enabled = True
    config.vlm.backend = "LOCAL_MODEL"
    config.vlm.model_name = "stub_local"
    
    ctx = PipelineContext(
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path / "output",
        config=config
    )
    # Give it a candidate segment
    seg = CandidateSegment(
        segment_id="cand_001",
        start_frame=0,
        end_frame=10,
        start_sec=42.0,
        end_sec=47.0,
        trigger_reason="test"
    )
    ctx.candidate_segments = [seg]
    return ctx


def test_vlm_disabled_skipped(mock_ctx):
    mock_ctx.config.vlm.enabled = False
    status = s06_vlm.run(mock_ctx)
    assert status.status == "SKIPPED"
    assert len(mock_ctx.vlm_observations) == 1
    assert mock_ctx.vlm_observations[0].status == VLMSegmentStatus.SKIPPED


def test_vlm_success_provenance_and_preservation(mock_ctx):
    status = s06_vlm.run(mock_ctx)
    assert status.status == "OK"
    assert len(mock_ctx.vlm_observations) == 1
    
    obs = mock_ctx.vlm_observations[0]
    assert obs.status == VLMSegmentStatus.SUCCESS
    
    # Check provenance
    assert obs.backend == "LOCAL_MODEL"
    assert obs.model_name == "stub_local"
    assert obs.prompt_version == "v1"
    assert obs.segment_start_sec == 42.0
    assert obs.segment_end_sec == 47.0
    
    # Check raw response preservation
    assert obs.raw_response is not None
    assert "picked up the cup" in obs.raw_response
    
    # Check absolute timestamp mapping (mock returns 0.5 and 1.5, start is 42.0)
    assert obs.start_time_sec == 42.5
    assert obs.end_time_sec == 43.5


def test_vlm_valid_unknown_is_success(mock_ctx):
    mock_ctx._test_prompt_flags = "return_unknown"
    status = s06_vlm.run(mock_ctx)
    assert status.status == "OK"
    assert len(mock_ctx.vlm_observations) == 1
    
    obs = mock_ctx.vlm_observations[0]
    assert obs.status == VLMSegmentStatus.SUCCESS
    assert obs.raw_action == "UNKNOWN"
    assert obs.active_hand == "UNKNOWN"
    assert obs.start_time_sec is None


def test_vlm_malformed_json_fails(mock_ctx):
    mock_ctx._test_prompt_flags = "return_malformed"
    status = s06_vlm.run(mock_ctx)
    # The stage itself succeeds, but the observation is FAILED
    assert status.status == "OK"
    assert len(mock_ctx.vlm_observations) == 1
    
    obs = mock_ctx.vlm_observations[0]
    assert obs.status == VLMSegmentStatus.FAILED
    assert "JSON parsing failed" in obs.error_reason
    assert obs.raw_response == "```json\n{malformed_json\n```"


def test_vlm_missing_fields_fails(mock_ctx):
    mock_ctx._test_prompt_flags = "return_missing"
    s06_vlm.run(mock_ctx)
    obs = mock_ctx.vlm_observations[0]
    assert obs.status == VLMSegmentStatus.FAILED
    assert "Validation failed" in obs.error_reason


def test_vlm_timestamps_out_of_bounds_fails(mock_ctx):
    mock_ctx._test_prompt_flags = "return_out_of_bounds"
    s06_vlm.run(mock_ctx)
    obs = mock_ctx.vlm_observations[0]
    assert obs.status == VLMSegmentStatus.FAILED
    assert "Validation failed" in obs.error_reason
    assert "outside segment" in obs.error_reason


def test_remote_vlm_api_failure(mock_ctx):
    mock_ctx.config.vlm.backend = "REMOTE_MODEL"
    mock_ctx.config.vlm.model_name = "error_remote"
    
    s06_vlm.run(mock_ctx)
    obs = mock_ctx.vlm_observations[0]
    assert obs.status == VLMSegmentStatus.FAILED
    assert "Connection timed out" in obs.error_reason
    assert obs.backend == "REMOTE_MODEL"


def test_multiple_segments(mock_ctx):
    # Add a second segment
    seg2 = CandidateSegment(
        segment_id="cand_002", start_frame=100, end_frame=110, start_sec=100.0, end_sec=105.0, trigger_reason="test"
    )
    mock_ctx.candidate_segments.append(seg2)
    
    s06_vlm.run(mock_ctx)
    assert len(mock_ctx.vlm_observations) == 2
    assert mock_ctx.vlm_observations[0].status == VLMSegmentStatus.SUCCESS
    assert mock_ctx.vlm_observations[1].status == VLMSegmentStatus.SUCCESS
