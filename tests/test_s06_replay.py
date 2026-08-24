"""Tests for s06's replay path.

Gemini is not reproducible: two runs of tt6 eighteen minutes apart, at
temperature 0, returned materially different action text. Replay freezes the
VLM output so a code change is the only variable between two runs. These tests
prove the two properties that make it trustworthy — no model is called, and a
segment that cannot be matched is a loud error rather than a silent omission.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.context import PipelineContext
from src.schema.segment import CandidateSegment
from src.schema.vlm import VLMSegmentStatus
from src.stages import s06_vlm


def _observation(segment_id: str, start: float, end: float, action: str) -> dict:
    """A recorded observation as s06 writes it, verbatim schema."""
    return {
        "observation_id": f"obs_{segment_id}",
        "segment_id": segment_id,
        "status": "SUCCESS",
        "backend": "GEMINI",
        "model_name": "gemini-3.1-flash-lite",
        "prompt_version": "v1",
        "segment_start_sec": start,
        "segment_end_sec": end,
        "actor": "person wearing a blue SpongeBob t-shirt",
        "active_hand": "RIGHT",
        "objects": ["cardboard box", "push chopper"],
        "raw_action": action,
        "start_time_sec": start,
        "end_time_sec": end - 1.3,
        "state_change": "chopper out of the box",
        "visible_facts": "hand on the chopper",
        "inference": "removal",
        "uncertainty": "none",
        "confidence": 1.0,
    }


@pytest.fixture
def replay_ctx(tmp_path):
    """A run whose segment ids differ from the recorded ones, as they always do.

    heuristic_segmenter mints ids as f"cand_{i:04d}_{uuid4().hex[:6]}", so the
    recorded file can never share an id with the run replaying it. Bounds are
    the stable key.
    """
    recorded = [
        _observation("cand_0000_abb6fd", 0.0, 8.3, "placing the push chopper into the cardboard box"),
        # 24.900000000000002 is what float accumulation actually produces, and
        # is why the join key is rounded.
        _observation("cand_0001_71608a", 24.900000000000002, 33.2, "unboxing a push chopper"),
    ]
    obs_file = tmp_path / "vlm_observations.json"
    obs_file.write_text(json.dumps(recorded), encoding="utf-8")

    config = PipelineConfig(stub_mode=False)
    config.vlm.enabled = True
    config.vlm.backend = "GEMINI"
    config.vlm.replay_from = str(obs_file)

    ctx = PipelineContext(
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path / "output",
        config=config,
    )
    ctx.candidate_segments = [
        CandidateSegment(
            segment_id="cand_0000_7ad163", start_frame=0, end_frame=249,
            start_sec=0.0, end_sec=8.3, trigger_reason="test",
        ),
        CandidateSegment(
            segment_id="cand_0003_82e8cf", start_frame=747, end_frame=996,
            start_sec=24.9, end_sec=33.2, trigger_reason="test",
        ),
    ]
    return ctx, obs_file


def test_replay_calls_no_model(replay_ctx, monkeypatch):
    """The whole point: replaying must not construct a backend or hit the API."""
    def explode(*args, **kwargs):
        raise AssertionError("replay constructed a VLM backend")

    monkeypatch.setattr(s06_vlm, "GeminiVLM", explode)
    monkeypatch.setattr(s06_vlm, "LocalVLM", explode)
    monkeypatch.setattr(s06_vlm, "RemoteVLM", explode)

    ctx, _ = replay_ctx
    status = s06_vlm.run(ctx)

    assert status.status == "OK"
    assert len(ctx.vlm_observations) == 2
    assert all(o.status == VLMSegmentStatus.SUCCESS for o in ctx.vlm_observations)


def test_replay_rebinds_segment_ids_to_this_run(replay_ctx):
    """s07 looks observations up by obs.segment_id, so the id must be rewritten."""
    ctx, _ = replay_ctx
    s06_vlm.run(ctx)

    ids = [o.segment_id for o in ctx.vlm_observations]
    assert ids == ["cand_0000_7ad163", "cand_0003_82e8cf"]
    # The recorded ids must not survive, or s07 would resolve no tracks.
    assert "cand_0000_abb6fd" not in ids

    # The text is what carries over — that is what makes the run reproducible.
    assert ctx.vlm_observations[0].raw_action == (
        "placing the push chopper into the cardboard box"
    )
    assert ctx.vlm_observations[1].raw_action == "unboxing a push chopper"
    # Provenance stays honest: the text really did come from Gemini.
    assert ctx.vlm_observations[0].backend == "GEMINI"


def test_replay_matches_bounds_despite_float_drift(replay_ctx):
    """Recorded 24.900000000000002 must match this run's 24.9."""
    ctx, _ = replay_ctx
    s06_vlm.run(ctx)
    assert ctx.vlm_observations[1].segment_start_sec == 24.9


def test_replay_errors_when_a_segment_has_no_match(replay_ctx):
    """An unmatched segment must fail loudly, not quietly yield fewer events."""
    ctx, _ = replay_ctx
    ctx.candidate_segments.append(
        CandidateSegment(
            segment_id="cand_0009_ffffff", start_frame=0, end_frame=10,
            start_sec=99.0, end_sec=105.0, trigger_reason="test",
        )
    )
    status = s06_vlm.run(ctx)

    assert status.status == "ERROR"
    assert "no observation for 1 segment" in status.message
    assert "99.00" in status.message
    # Nothing partial is left behind for s07 to score.
    assert ctx.vlm_observations == []


def test_replay_errors_on_missing_file(replay_ctx, tmp_path):
    ctx, _ = replay_ctx
    ctx.config.vlm.replay_from = str(tmp_path / "nope.json")
    status = s06_vlm.run(ctx)
    assert status.status == "ERROR"
    assert "not found" in status.message


def test_replay_writes_observations_to_this_run(replay_ctx):
    """The replayed run must leave its own vlm_observations.json behind."""
    ctx, _ = replay_ctx
    s06_vlm.run(ctx)
    written = json.loads(
        (ctx.output_dir / "vlm_observations.json").read_text(encoding="utf-8")
    )
    assert [r["segment_id"] for r in written] == [
        "cand_0000_7ad163", "cand_0003_82e8cf"
    ]
