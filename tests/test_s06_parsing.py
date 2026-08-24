"""Tests for s06's JSON parsing and provenance.

Both behaviours here were bugs found in real Kaggle output, not hypotheticals:

  - Gemini answered segment 2 of run_20260824_065415 with a JSON *array*. It
    survived only because the array held one element and the old scraper
    happened to pull that element out. A two-element array produced
    ``JSONDecodeError: Extra data`` and failed the whole segment.
  - ``GeminiVLM.backend`` returned ``"REMOTE_MODEL"``, so every event this
    pipeline shipped recorded ``source="vlm:remote_model"`` with no trace of
    which provider produced it.
"""
from __future__ import annotations

import json

from src.stages.s06_vlm import extract_json


# The verbatim array response Gemini returned for segment 2. Kept as evidence.
_REAL_ARRAY_RESPONSE = """[
  {
    "actor": "person wearing a blue SpongeBob t-shirt",
    "raw_action": "opening and unpacking a cardboard box",
    "confidence": 1.0
  }
]"""


def test_extract_json_object_unchanged():
    """The common case must behave exactly as before."""
    assert json.loads(extract_json('{"a": 1}')) == {"a": 1}
    assert json.loads(extract_json('```json\n{"a": 1}\n```')) == {"a": 1}
    # Preamble text around a bare object is still scraped off.
    assert json.loads(extract_json('Here you go:\n{"a": 1}\nhope that helps')) == {"a": 1}


def test_extract_json_keeps_a_real_gemini_array():
    parsed = json.loads(extract_json(_REAL_ARRAY_RESPONSE))
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert parsed[0]["raw_action"] == "opening and unpacking a cardboard box"


def test_extract_json_survives_a_two_element_array():
    """The case that used to raise JSONDecodeError and fail the segment."""
    text = '[{"raw_action": "opening the box"}, {"raw_action": "removing the chopper"}]'
    parsed = json.loads(extract_json(text))
    assert [r["raw_action"] for r in parsed] == [
        "opening the box", "removing the chopper"
    ]


def test_extract_json_array_in_a_fenced_block():
    text = '```json\n[{"a": 1}, {"a": 2}]\n```'
    assert json.loads(extract_json(text)) == [{"a": 1}, {"a": 2}]


def test_extract_json_prefers_the_bracket_that_opens_first():
    """An object holding a list must not be read as an array."""
    text = '{"objects": ["box", "chopper"], "raw_action": "placing"}'
    parsed = json.loads(extract_json(text))
    assert isinstance(parsed, dict)
    assert parsed["objects"] == ["box", "chopper"]
