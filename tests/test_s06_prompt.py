"""Tests for the clip duration the s06 prompt states.

The prompt asked for a "float offset from segment start" and never said what the
segment's length was. Measured across twelve tt7 segments at
max_segment_duration_sec=1.0, gemini-3.1-flash-lite answered with a relative
end_time_sec of exactly 2.0 every single time, on clips 0.95 seconds long:

    [0.00,0.95] end 2.00    [2.86,3.81] end 4.86
    [0.95,1.90] end 2.95    [3.81,4.76] end 5.81
    [1.90,2.86] end 3.90    [4.76,5.71] end 6.76
                            [5.71,6.67] end 7.71

Twelve for twelve at a constant is not a model failing to see the action, it is
a field with no stated bound. Five of those were discarded outright and the other
seven fell back to SEGMENT precision, so every event spanned its whole window and
covered two to four labelled actions.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.context import PipelineContext
from src.schema.segment import CandidateSegment
from src.stages import s06_vlm
from src.stages.s06_vlm import (
    _DURATION_TOKEN,
    PROMPT,
    PROMPT_VERSION,
    _render_prompt,
)


def test_the_rendered_prompt_states_the_duration():
    rendered = _render_prompt(0.9523809523809524)
    assert "0.95" in rendered
    # A leftover sentinel would ship the literal "{{CLIP_DURATION}}" to the model.
    assert _DURATION_TOKEN not in rendered


def test_every_occurrence_is_substituted():
    """The bound appears in the schema and again in the rules; both must fill."""
    assert PROMPT.count(_DURATION_TOKEN) >= 3
    assert _render_prompt(2.5).count("2.50") >= 3


def test_the_json_schema_block_survives_rendering():
    """The prompt's braces are literal JSON, not format fields."""
    rendered = _render_prompt(1.0)
    assert '"raw_action"' in rendered
    assert '"start_time_sec"' in rendered
    # The schema's own opening brace has to still be there.
    assert "Schema:\n{" in rendered


def test_each_segment_is_told_its_own_length(monkeypatch, tmp_path):
    """Two segments of different lengths must not get the same prompt."""
    seen: list[str] = []

    class _RecordingVLM:
        backend = "GEMINI"
        model_name = "recorder"

        def analyze_segment(self, video_path, start_sec, end_sec, prompt):
            seen.append(prompt)
            raise RuntimeError("stop here — the prompt is what is under test")

    monkeypatch.setattr(s06_vlm, "LocalVLM", lambda **kw: _RecordingVLM())
    config = PipelineConfig(stub_mode=False)
    config.vlm.enabled = True
    config.vlm.backend = "LOCAL_MODEL"
    config.vlm.max_retries = 0
    ctx = PipelineContext(
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path / "output",
        config=config,
    )
    ctx.candidate_segments = [
        CandidateSegment(segment_id="a", start_frame=0, end_frame=1,
                         start_sec=0.0, end_sec=0.9523809523809524),
        CandidateSegment(segment_id="b", start_frame=2, end_frame=3,
                         start_sec=10.0, end_sec=14.0),
    ]

    s06_vlm.run(ctx)

    assert len(seen) == 2
    assert "0.95" in seen[0]
    assert "4.00" in seen[1]
    # The second segment's length, not the first's, and not an absolute timestamp.
    assert "0.95" not in seen[1]
    assert "14.00" not in seen[1]


def test_observations_record_the_prompt_version(monkeypatch, tmp_path):
    """Runs from two different prompts have to be distinguishable afterwards."""
    assert PROMPT_VERSION != "v1"
    config = PipelineConfig(stub_mode=False)
    config.vlm.enabled = False
    ctx = PipelineContext(
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path / "output",
        config=config,
    )
    ctx.candidate_segments = [
        CandidateSegment(segment_id="a", start_frame=0, end_frame=1,
                         start_sec=0.0, end_sec=1.0),
    ]
    s06_vlm.run(ctx)
    assert ctx.vlm_observations[0].prompt_version == PROMPT_VERSION
