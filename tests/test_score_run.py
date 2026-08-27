"""Tests for tools/score_run.py, the scorer that produces the headline numbers.

The scorer is the only thing that says whether the pipeline is getting better, so
a bug in it is worse than a bug in a stage: it moves every number at once and
there is nothing else to check it against.

Loaded by path because tools/ is not a package. Same reason tools/ scripts insert
the repo root on sys.path to import src.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRUTH = ROOT / "tests" / "fixtures" / "tt7_ground_truth.json"


def _load_score_run():
    spec = importlib.util.spec_from_file_location(
        "score_run", ROOT / "tools" / "score_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score_run = _load_score_run()


@pytest.fixture(scope="module")
def tt7_actions() -> list[dict]:
    """The seven hand-labelled tt7 actions, expanded the way the scorer does."""
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    return score_run.expand_truth(truth)


def _event(action: str, start: float, end: float, label: str | None) -> dict:
    return {
        "action": action, "start_sec": start, "end_sec": end,
        "attributes": {"object_label": label},
    }


# ─────────────────────────────────────────── the object/target narrowing bug
def test_an_insert_naming_its_destination_still_scores_exact(tt7_actions):
    """The bug this test was written for.

    tt7's INSERT is object 'push chopper', target 'cardboard box'. s07 resolves
    the container as object_label often enough that its own
    _order_objects_by_action exists to fight it — the VLM returned
    ["box", "push chopper"] for "placing the push chopper back into the box".

    Testing only the truth's 'object' field excluded INSERT from the pool while
    the six other cardboard-box actions matched, so an exactly correct INSERT was
    scored against a pool that could not contain INSERT, and came out OTHER.
    """
    ev = _event("INSERT", 1.0, 2.0, "cardboard box")
    result = score_run.assess(ev, tt7_actions, tolerance=0.0)
    assert result["verdict"] == "EXACT"
    assert "INSERT" in [a["action"] for a in result["covered"]]


def test_naming_the_object_still_narrows(tt7_actions):
    """The widening must not become 'match everything'.

    'push chopper' is the object of PICK and INSERT and the target of neither
    other action, so the pool must be those two and not all seven.
    """
    ev = _event("PICK", 0.0, 6.63, "push chopper")
    result = score_run.assess(ev, tt7_actions, tolerance=0.0)
    assert result["narrowed_by_object"]
    assert sorted(a["action"] for a in result["covered"]) == ["INSERT", "PICK"]


def test_a_label_matching_nothing_falls_back_to_everything_covered(tt7_actions):
    ev = _event("PLACE", 0.0, 0.4, "bicycle")
    result = score_run.assess(ev, tt7_actions, tolerance=0.0)
    assert not result["narrowed_by_object"]
    assert [a["action"] for a in result["covered"]] == ["PLACE"]


def test_the_dining_table_target_is_reachable(tt7_actions):
    """PICK's target is the dining table and no action's object is.

    Before the fix an event that resolved the table matched nothing and silently
    fell back to every covered action; now it narrows to the two actions that
    actually involve the table.
    """
    ev = _event("PICK", 0.0, 6.63, "dining table")
    result = score_run.assess(ev, tt7_actions, tolerance=0.0)
    assert result["narrowed_by_object"]
    assert sorted(a["action"] for a in result["covered"]) == ["PICK", "PLACE"]


def test_an_empty_target_is_not_matched_by_an_empty_label(tt7_actions):
    """FOLD, CLOSE and GRASP all have target ''. An unresolved label must not
    match them by both sides being blank."""
    ev = _event("CLOSE", 3.0, 4.0, None)
    result = score_run.assess(ev, tt7_actions, tolerance=0.0)
    assert not result["narrowed_by_object"]


# ───────────────────────────────────────────────────────── verdict categories
def test_a_time_reversed_verb_is_reversed_not_other(tt7_actions):
    """The distinction the scorer exists to draw: on tt6 the VLM picked the exact
    time-reverse verb twice, which plain accuracy would hide."""
    ev = _event("REMOVE", 1.0, 2.0, "push chopper")
    assert score_run.assess(ev, tt7_actions, tolerance=0.0)["verdict"] == "REVERSED"


def test_an_action_outside_the_vocabulary_is_reported_not_rewritten(tt7_actions):
    """FOLD is not in src/schema/event.py ActionType and the label file says so.

    The pipeline cannot emit FOLD — s07's stem table maps "fold" to UNKNOWN
    deliberately, because folding a flap is several primitives at once — so the
    case that actually occurs is UNKNOWN against a labelled FOLD. Reporting that
    as NOT_IN_VOCABULARY keeps the gap visible instead of charging the pipeline
    for a word it has no way to say.
    """
    ev = _event("UNKNOWN", 2.1, 2.9, "cardboard box")
    result = score_run.assess(ev, tt7_actions, tolerance=0.0)
    assert [a["action"] for a in result["covered"]] == ["FOLD"]
    assert result["verdict"] == "NOT_IN_VOCABULARY"


def test_an_event_overlapping_nothing_is_unmatched(tt7_actions):
    ev = _event("PLACE", 50.0, 60.0, "cardboard box")
    result = score_run.assess(ev, tt7_actions, tolerance=0.0)
    assert result["verdict"] == "UNMATCHED"
    assert result["covered"] == []


# ────────────────────────────────────────────────────────────────── direction
def test_direction_comes_from_the_verb_alone(tt7_actions):
    """INSERT implies INTO with no geometry involved, which is why the timing
    work does not try to derive direction."""
    ev = _event("INSERT", 1.0, 2.0, "push chopper")
    assert score_run.assess(ev, tt7_actions, tolerance=0.0)["direction"] == "SAME"


def test_a_reversed_verb_reports_a_reversed_direction(tt7_actions):
    ev = _event("REMOVE", 1.0, 2.0, "push chopper")
    assert score_run.assess(ev, tt7_actions, tolerance=0.0)["direction"] == "REVERSED"


def test_a_labelled_direction_of_none_reports_na(tt7_actions):
    """CLOSE is labelled NONE. The truth stating no direction must not be scored
    against ACTION_DIRECTION's CLOSE -> INTO."""
    ev = _event("CLOSE", 3.1, 3.9, "cardboard box")
    assert score_run.assess(ev, tt7_actions, tolerance=0.0)["direction"] == "N/A"


def test_neither_truth_file_uses_a_direction_outside_the_label_vocabulary():
    """Pins the check behind the ACTION_DIRECTION comment.

    OUT_OF appears on the emitted side only. If a label file ever starts using it,
    the reasoning recorded in score_run.py needs revisiting.
    """
    allowed = {"ONTO", "OFF", "INTO", "NONE", ""}
    for name in ("tt6_ground_truth.json", "tt7_ground_truth.json"):
        truth = json.loads((ROOT / "tests" / "fixtures" / name).read_text(encoding="utf-8"))
        for a in truth["actions"]:
            assert (a.get("direction") or "").upper() in allowed, (name, a)


# ────────────────────────────────────────────────────────────── the tt7 state
def test_a_whole_clip_event_covers_every_tt7_action(tt7_actions):
    """Why timing is the blocker, as a test rather than a claim.

    s05 emits one candidate segment for tt7 and s07 gives every event that
    segment's bounds, so one event covers all seven labelled actions and every
    verdict is an upper bound over the whole clip. No accuracy number means
    anything until this stops being true.
    """
    ev = _event("INSERT", 0.0, 6.6667, None)
    result = score_run.assess(ev, tt7_actions, tolerance=1.0)
    assert len(result["covered"]) == 7
