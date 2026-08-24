"""Guard the hand labels against silent drift.

These are the only ground truth the project has, and two files now hold them:
``tt6_ground_truth.json`` for the five-copy video and
``tt7_ground_truth.json`` for tt7.mp4, which IS that source clip. The second was
derived from the first, so nothing but a test stops them diverging — and a
divergence would be invisible, showing up only as a score that moved for no
reason anybody could name.

The period these files declare was once wrong: four copies of 8.3s, got by
dividing 33.2s by a guessed count. tools/align_clips.py matched tt7.mp4 against
tt6.mp4 frame by frame and found five copies with a 199-frame stride, so the
numbers here are now pixel-derived. These tests pin the arithmetic that connects
them so the guess cannot creep back.

Deliberately asserts the *labels*, not the pipeline. If a future change makes a
score worse, the fix belongs in the pipeline; editing these files to recover a
number would destroy the only independent measurement there is.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
MULTI = FIXTURES / "tt6_ground_truth.json"
SINGLE = FIXTURES / "tt7_ground_truth.json"

# The clip is 30fps, so a single frame is this many seconds. Several assertions
# below must allow a frame of slack, because tt6 drops one frame at each concat
# seam and the two files therefore round to values a frame apart.
FRAME_SEC = 1.0 / 30.0


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_the_two_label_files_hold_identical_actions():
    """tt6.mp4 is this clip five times, so the labels must be the same labels."""
    assert _load(MULTI)["actions"] == _load(SINGLE)["actions"]


def test_the_single_clip_file_declares_no_repeats():
    """repeats=1 makes score_run use the times as written, with no expansion."""
    single = _load(SINGLE)
    assert single["repeats"] == 1
    assert "source_clip_duration_sec" not in single, (
        "a clip that is not a loop has no loop period; leaving one would invite "
        "a tool to fold frames that were never repeated"
    )
    # The single clip is a full 200 frames; the multi file's period is the copy
    # STRIDE, 199 frames, because tt6 drops one frame at each seam. They describe
    # the same clip and must agree to within that one dropped frame — but not be
    # forced equal, which would erase the seam that align_clips.py measured.
    multi_period = _load(MULTI)["source_clip_duration_sec"]
    assert abs(single["total_duration_sec"] - multi_period) <= 1.5 * FRAME_SEC, (
        f"single clip {single['total_duration_sec']}s vs multi stride "
        f"{multi_period}s differ by more than one frame; they should be the "
        f"same clip give or take the dropped seam frame"
    )


def test_the_multi_copy_file_still_describes_a_loop():
    """Five copies, stride x repeats within a frame of the whole-file duration.

    Not exact: score_run.py lays each copy at i * stride, so the stride is the
    199-frame gap between copy starts (6.6333s), not the 200-frame clip. Five
    strides cover 995 frames; the file is 996, because only the four *internal*
    seams drop a frame and the last copy keeps its full length. So the product
    lands one frame short of total_duration_sec, and that is correct.
    """
    multi = _load(MULTI)
    assert multi["repeats"] == 5
    covered = multi["source_clip_duration_sec"] * multi["repeats"]
    assert abs(covered - multi["total_duration_sec"]) <= 1.5 * FRAME_SEC, (
        f"{multi['repeats']} x {multi['source_clip_duration_sec']}s = {covered}s "
        f"is more than one frame off total {multi['total_duration_sec']}s"
    )


def test_no_label_runs_past_the_clip_it_belongs_to():
    """A label past its clip end silently bleeds onto the next copy, or off the
    end of the file, and scores against footage it never described.

    For the single clip the bound is the whole duration; for the looped file it
    is the copy stride, because a label reaching past one stride lands inside the
    next copy when score_run.py expands it.
    """
    single = _load(SINGLE)
    for action in single["actions"]:
        assert 0.0 <= action["start_sec"] < action["end_sec"]
        assert action["end_sec"] <= single["total_duration_sec"] + 1e-9, (
            f"action {action['order']} ends at {action['end_sec']}s, past the "
            f"{single['total_duration_sec']}s clip end"
        )

    multi = _load(MULTI)
    stride = multi["source_clip_duration_sec"]
    for action in multi["actions"]:
        assert action["end_sec"] <= stride + 1e-9, (
            f"action {action['order']} ends at {action['end_sec']}s, past the "
            f"{stride}s copy stride; it would bleed into the next copy"
        )
