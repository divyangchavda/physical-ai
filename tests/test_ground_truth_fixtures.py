"""Guard the hand labels against silent drift.

These are the only ground truth the project has, and two files now hold them:
``tt6_ground_truth.json`` for the four-copy video and
``tt6_single_ground_truth.json`` for the source clip on its own. The second was
derived from the first, so nothing but a test stops them diverging — and a
divergence would be invisible, showing up only as a score that moved for no
reason anybody could name.

Deliberately asserts the *labels*, not the pipeline. If a future change makes a
score worse, the fix belongs in the pipeline; editing these files to recover a
number would destroy the only independent measurement there is.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
MULTI = FIXTURES / "tt6_ground_truth.json"
SINGLE = FIXTURES / "tt6_single_ground_truth.json"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_the_two_label_files_hold_identical_actions():
    """tt6.mp4 is this clip four times, so the labels must be the same labels."""
    assert _load(MULTI)["actions"] == _load(SINGLE)["actions"]


def test_the_single_clip_file_declares_no_repeats():
    """repeats=1 makes score_run use the times as written, with no expansion."""
    single = _load(SINGLE)
    assert single["repeats"] == 1
    assert "source_clip_duration_sec" not in single, (
        "a clip that is not a loop has no loop period; leaving one would invite "
        "a tool to fold frames that were never repeated"
    )
    assert single["total_duration_sec"] == _load(MULTI)["source_clip_duration_sec"]


def test_the_multi_copy_file_still_describes_a_loop():
    multi = _load(MULTI)
    assert multi["repeats"] == 4
    assert (
        multi["source_clip_duration_sec"] * multi["repeats"]
        == multi["total_duration_sec"]
    )


def test_labels_stay_inside_the_source_clip():
    """A label past the clip end would silently expand onto the next copy."""
    single = _load(SINGLE)
    for action in single["actions"]:
        assert 0.0 <= action["start_sec"] < action["end_sec"]
        assert action["end_sec"] <= single["total_duration_sec"]
