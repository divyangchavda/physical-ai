"""Tests for the SSv2 class-name -> ActionType allow-list.

The keys in ``src/eval/ssv2_action_map.py`` were typed by hand from the 174 class
names, and a typo there does not raise — it silently removes a verb from the
evaluation set, which later reads as "that verb never appears in SSv2". So the
first test here checks every key against the dataset's own class list, committed
verbatim as ``tests/fixtures/ssv2_class_names.json`` (metadata only, no clips).

The rest pin the exclusion rules, because the risk with an allow-list is not that
it is wrong today but that someone widens it later to raise a score. Each excluded
example below is named with the rule it falls under.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.ssv2_action_map import (
    SSV2_TEMPLATE_TO_ACTION,
    map_template,
    normalize_template,
)
from src.schema.event import ActionType

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ssv2_class_names.json"


@pytest.fixture(scope="module")
def class_names() -> dict[str, int]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["classes"]


def test_the_fixture_is_the_whole_class_list(class_names):
    """174 is the published class count; a partial fixture would weaken test 2."""
    assert len(class_names) == 174


def test_every_allow_list_key_is_a_real_class_name(class_names):
    """The typo guard. Same check ``tools/ssv2_bundle.py`` runs before copying."""
    known = {normalize_template(name) for name in class_names}
    unknown = sorted(
        name for name in SSV2_TEMPLATE_TO_ACTION
        if normalize_template(name) not in known
    )
    assert unknown == []


def test_the_allow_list_is_a_minority_of_the_dataset(class_names):
    """Recorded as a measurement, not a target: our 15 verbs cover 43 of 174.

    If this number moves, the coverage claim in the module docstring and in any
    reported accuracy figure moves with it.
    """
    assert len(SSV2_TEMPLATE_TO_ACTION) == 43


def test_no_class_maps_to_unknown():
    """UNKNOWN is what the pipeline outputs when it cannot decide.

    Using it as ground truth would score "I don't know" as correct, so an SSv2
    class we have no verb for is excluded instead of being mapped here.
    """
    assert ActionType.UNKNOWN not in set(SSV2_TEMPLATE_TO_ACTION.values())


def test_every_value_is_in_the_pipelines_own_vocabulary():
    for action in SSV2_TEMPLATE_TO_ACTION.values():
        assert action in set(ActionType)


def test_no_two_keys_normalize_to_the_same_class():
    """A duplicate would make one entry unreachable and the count above a lie."""
    normalized = [normalize_template(k) for k in SSV2_TEMPLATE_TO_ACTION]
    assert len(set(normalized)) == len(normalized)


# ──────────────────────────────────────────────────────── the two written forms
def test_the_bracketed_form_a_clip_carries_resolves():
    """``labels.json`` says "Putting something into something"; a clip's
    ``template`` field says "Putting [something] into [something]". Same class."""
    assert map_template("Putting [something] into [something]") is ActionType.INSERT
    assert map_template("Putting something into something") is ActionType.INSERT


def test_normalize_collapses_the_whitespace_brackets_leave_behind():
    assert normalize_template("Moving [something] up") == "Moving something up"
    assert normalize_template("  Opening   [something] ") == "Opening something"


def test_a_multi_word_placeholder_still_resolves():
    """SSv2 fills brackets with phrases, e.g. "[something in it]"."""
    assert normalize_template("Tipping [something] with [something in it] over") == (
        "Tipping something with something in it over"
    )


# ──────────────────────────────────────────────────────────── the four exclusions
@pytest.mark.parametrize("template", [
    "Folding something",            # rule 1: no FOLD in ActionType -- open decision
    "Pouring something into something",
    "Squeezing something",
    "Dropping something into something",
    "Lifting something up completely without letting it drop down",
    "Turning the camera left while filming something",
])
def test_a_class_naming_no_verb_of_ours_is_excluded(template):
    assert map_template(template) is None


@pytest.mark.parametrize("template", [
    "Pulling something out of something",   # PULL and REMOVE at once
    "Pushing something with something",     # PUSH and USE_TOOL at once
    "Scooping something up with something",
])
def test_a_class_naming_two_of_our_verbs_is_excluded(template):
    """No precedence rule was invented to break these ties."""
    assert map_template(template) is None


@pytest.mark.parametrize("template", [
    "Putting something on the edge of something so it is not supported and falls down",
    "Putting something onto something else that cannot support it so it falls down",
    "Pushing something so that it falls off the table",
    "Plugging something into something but pulling it right out as you remove your hand",
])
def test_a_class_bundling_a_second_event_is_excluded(template):
    """Our events are single actions, so a put-then-fall clip has no one answer."""
    assert map_template(template) is None


@pytest.mark.parametrize("template", [
    "Pretending to put something into something",
    "Pretending to open something without actually opening it",
    "Pretending to close something without actually closing it",
    "Failing to put something into something because something does not fit",
    "Trying but failing to attach something to something because it doesn't stick",
    "Something falling like a rock",
])
def test_a_class_with_no_action_performed_is_excluded(template):
    assert map_template(template) is None


def test_pretending_to_open_is_excluded_while_opening_is_kept():
    """The pair that matters most: OPEN/CLOSE is the inversion we just fixed, and
    a pretend-open clip would punish a correct reading of the video."""
    assert map_template("Opening something") is ActionType.OPEN
    assert map_template("Pretending to open something without actually opening it") is None


def test_an_unknown_string_is_excluded_not_guessed():
    assert map_template("Reticulating splines") is None
    assert map_template("") is None
