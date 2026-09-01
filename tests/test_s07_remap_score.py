"""Tests for the offline re-scorer's own decisions.

The tool reports a delta against 200 human labels, so the two things that could
quietly distort that number are tested here: which clips it excludes, and what it
considers the answer for a caption describing more than one action.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

from s07_remap_score import (  # noqa: E402
    is_multi_verb,
    load,
    remap,
    remap_all,
)


def _rec(raw: str | None, **kw) -> dict:
    base = {"clip_id": "1", "truth_action": "PLACE", "got": "PLACE",
            "raw_action": raw}
    base.update(kw)
    return base


# ───────────────────────────────────────────────── one caption, one verb
def test_a_single_action_caption_has_one_verb():
    assert remap_all(_rec("plugging a charger into a wall outlet")) == ["INSERT"]
    assert is_multi_verb(_rec("plugging a charger into a wall outlet")) is False


def test_remap_agrees_with_remap_all_on_a_single_action_caption():
    """Where they disagree the tool's headline would be measuring something other
    than what the pipeline emits, so this is the assumption the delta rests on."""
    for caption in ("tapping the bicycle basket", "uncapping a marker",
                    "peeling a sticker off a newspaper", "closing a drawer"):
        assert remap_all(_rec(caption)) == [remap(_rec(caption))]


# ─────────────────────────────── more than one action means no reproducible top-1
def test_a_two_action_caption_is_excluded_rather_than_guessed():
    """s07 emits one event per clause and the evaluation picks by confidence,
    which needs the track and segment data this tool does not have."""
    record = _rec("the person touches and then picks up the blush compact")
    assert remap_all(record) == ["TOUCH", "PICK"]
    assert is_multi_verb(record) is True


def test_the_whole_sentence_match_is_not_the_live_answer_for_such_a_caption():
    """Precisely why it is excluded: the sentence match gives PICK, the live run
    recorded TOUCH, and neither is wrong — they are different events."""
    record = _rec("the person touches and then picks up the blush compact",
                  got="TOUCH")
    assert remap(record) == "PICK"
    assert record["got"] == "TOUCH"
    assert is_multi_verb(record)


# ────────────────────────────────────────────────────────── missing captions
def test_a_clip_with_no_caption_has_no_verb_and_no_verbs():
    assert remap(_rec(None)) is None
    assert remap_all(_rec(None)) == []
    assert is_multi_verb(_rec(None)) is False


def test_an_empty_caption_is_treated_as_no_caption():
    assert remap(_rec("")) is None
    assert remap_all(_rec("")) == []


# ────────────────────────────────────────────────────────────── loading
def test_clips_without_a_caption_are_dropped_at_load(tmp_path):
    """The 8 clips that produced no event at all have nothing to replay, and
    counting them would understate the mapping's accuracy."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": [
        _rec("closing a drawer"), _rec(None), _rec(""),
    ]}), encoding="utf-8")
    assert len(load(path)) == 1


def test_a_bare_list_of_results_loads_too(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps([_rec("closing a drawer")]), encoding="utf-8")
    assert len(load(path)) == 1


def test_an_empty_results_file_is_an_error_not_a_zero_percent_score(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"results": []}), encoding="utf-8")
    with pytest.raises(SystemExit):
        load(path)
