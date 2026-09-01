"""Verb-stem mapping in s07, pinned against the captions a real run produced.

Every caption quoted here is a verbatim ``raw_action`` from the 200-clip SSv2
evaluation at commit a7f1e06, and every expected verb is that clip's human SSv2
label. Nothing here is invented, and nothing here was chosen after the fact to
make a number look better: the measurement is +17 clips with 0 regressions,
reproducible with

    python tools/s07_remap_score.py --results <that run's results.json>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.schema.event import ActionType  # noqa: E402
from src.stages.s07_events import (  # noqa: E402
    _ACTION_STEMS,
    _map_raw_action_to_type,
    _stem_pattern,
)


def verb(caption: str, objects: list[str] | None = None) -> str:
    return _map_raw_action_to_type(caption, objects).value


# ───────────────────────────── a noun must not outrank the verb beside it
@pytest.mark.parametrize("caption,expected", [
    # The one that was measurably wrong: "fold" is the highest-precedence stem in
    # the table, so the object's name decided the whole clip.
    ("placing a digital thermometer on top of a red folder", "PLACE"),
    ("moving the scissors and the smartphone closer together", "MOVE"),
    ("picked up the bottle opener", "PICK"),
    ("moving the red pot holder", "MOVE"),
])
def test_a_longer_word_starting_with_a_stem_is_not_that_verb(caption, expected):
    """folder/closer/opener/holder each collided with a stem in the live run."""
    assert verb(caption) == expected


def test_object_blanking_is_not_what_protects_the_verb():
    """These captions resolved correctly live only because the object list
    happened to contain the colliding noun. The verb must survive without it."""
    assert verb("placing a thermometer on a red folder", objects=None) == "PLACE"


@pytest.mark.parametrize("caption,expected", [
    ("placing a lid on a cup", "PLACE"),
    ("placed the pen on the table", "PLACE"),
    ("the person places their hand on the table", "PLACE"),
    ("closing a drawer", "CLOSE"),
    ("the person closed the box", "CLOSE"),
    ("holding plastic bottle", "GRASP"),
    ("the person holds a black boot", "GRASP"),
    ("opening a door", "OPEN"),
    ("the person opens the wallet", "OPEN"),
    ("the person pushes the pen with their finger", "PUSH"),
    ("gripping the handle", "GRASP"),
    ("the person uses their right hand", "USE_TOOL"),
    ("the person touches the doll's hand", "TOUCH"),
    ("removing the lid from a container", "REMOVE"),
    ("person is removing the lid", "REMOVE"),
    ("releasing the button", "RELEASE"),
    ("examining the label", "INSPECT"),
])
def test_every_inflection_the_run_actually_produced_still_matches(caption, expected):
    """Bounding the stem must not cost a form the VLM really writes."""
    assert verb(caption) == expected


def test_the_stem_pattern_accepts_the_bare_stem_as_a_word():
    assert _stem_pattern("push").search("push the box")
    assert not _stem_pattern("push").search("pushcart the box")


# ───────────────────────────────────────── stems added from the run's captions
@pytest.mark.parametrize("caption,expected", [
    # INSERT: eleven clips, all of them UNKNOWN before.
    ("plugging a charger into a wall plug", "INSERT"),
    ("plugged a power adapter into a wall outlet", "INSERT"),
    ("plugging an audio cable into the input jack", "INSERT"),
    ("stuffing a hand towel into a glass mug", "INSERT"),
    ("threading the belt strap through the buckle", "INSERT"),
    # TOUCH: four clips.
    ("tapping the bicycle basket", "TOUCH"),
    ("the person taps the top of the aerosol spray can", "TOUCH"),
    ("pressing down on the pump of a soap dispenser", "TOUCH"),
    ("pressing down on the lid of a plastic container", "TOUCH"),
    # REMOVE: two clips.
    ("peeling a sticker off a newspaper", "REMOVE"),
    ("reached into a pencil case and retrieved a red pen", "REMOVE"),
    # one each.
    ("uncapping a marker", "OPEN"),
    ("flicked the ring with a finger", "PUSH"),
    ("sliding a fork across a wooden table", "MOVE"),
])
def test_a_caption_the_run_scored_unknown_now_maps_to_its_label(caption, expected):
    assert verb(caption) == expected


def test_drop_reaches_insert_through_the_existing_preposition_promotion():
    """"drop" is PLACE in the table; the containment comes from "into", which is
    machinery that already existed and is not special-cased for this verb."""
    assert verb("dropped a piece of paper into a trash can") == "INSERT"
    assert verb("dropped the book on the table") == "PLACE"


# ────────────────────────────────── precedence, where a new stem could steal
def test_plug_as_an_object_does_not_beat_a_real_verb():
    """"picking plug(blue) up" scored PICK in the run and must keep doing so.

    Morphology cannot tell the noun "plug" from the verb, so the group holding it
    takes inflected forms only and sits last in the table.
    """
    assert verb("picking plug(blue) up") == "PICK"
    assert verb("placing the plug on the table") == "PLACE"
    assert verb("moving the thread across the table") == "MOVE"


def test_a_bare_object_noun_produces_no_verb_at_all():
    """"put the stuff down" first mapped to INSERT off the noun "stuff". It is
    UNKNOWN rather than PLACE because the stem is the contiguous phrase "put
    down" and this caption splits it — a separate gap, with no caption in the run
    to justify closing it yet."""
    assert verb("put the stuff down") == "UNKNOWN"
    assert verb("the stuff is on the table") == "UNKNOWN"
    assert verb("a length of thread") == "UNKNOWN"


def test_press_does_not_outrank_push():
    """Contact and displacement in one caption is a displacement."""
    assert verb("pushing and pressing the mouse from left to right") == "PUSH"


def test_the_noun_tape_is_not_the_verb_tap():
    """Why the stems are "tapp"/"taps" and not "tap"."""
    assert verb("the tape dispenser is on the table") == "UNKNOWN"
    assert verb("taping the box shut") == "UNKNOWN"
    assert verb("tapping the box") == "TOUCH"


def test_slid_covers_the_forms_the_run_wrote():
    for caption in ("sliding a bottle across a table",
                    "the person slides the hard drive across the table",
                    "the box slid across the surface"):
        assert verb(caption) == "MOVE"


def test_sliding_is_move_and_not_a_guessed_direction():
    """Six PUSH/PULL-labelled clips are described as "sliding across". MOVE is
    what the caption says; picking PUSH or PULL to collect them would be fitting
    the table to the answer sheet, so those clips stay wrong."""
    assert verb("sliding the shampoo bottle across the surface") == "MOVE"
    assert verb("sliding a matchbox across a surface") == "MOVE"


# ───────────────────────────────── words deliberately left out of the table
@pytest.mark.parametrize("caption,reason", [
    ("pointing at the title of the book", "pointing at a thing is not touching it"),
    ("rotating the lid on the pitcher", "OPEN in one clip, CLOSE in another"),
    ("The orange is resting above the CD", "a state, not an action"),
    ("unfolding a newspaper", "several primitives at once, as with fold"),
])
def test_a_word_that_names_no_single_primitive_stays_unknown(caption, reason):
    """Each of these would have gained a clip. UNKNOWN is the honest answer, and
    scoring "I don't know" as correct is what we are trying not to do."""
    assert verb(caption) == "UNKNOWN", reason


def test_lift_is_absent_because_no_evidence_supports_it():
    """The two clips it would match are labelled GRASP and MOVE, so PICK would
    still be wrong on both."""
    assert verb("lifting the notebook to reveal the newspaper underneath") == "UNKNOWN"


# ────────────────────────────────────────────────── table-wide invariants
def test_no_stem_is_listed_twice_in_the_table():
    """A stem in two groups would make precedence depend on list order for a
    reason no comment explains."""
    seen: dict[str, ActionType] = {}
    for stems, action in _ACTION_STEMS:
        for stem in stems:
            assert stem not in seen, f"{stem!r} appears twice"
            seen[stem] = action


def test_the_object_noun_group_is_last():
    """These can only fire when no real verb matched."""
    assert _ACTION_STEMS[-1][0] == ("plugging", "plugged", "stuffing", "stuffed",
                                    "threading", "threaded")
    assert _ACTION_STEMS[-1][1] is ActionType.INSERT


def test_fold_still_refuses_to_name_a_primitive():
    """The rule the folder collision was hiding behind is still in force."""
    assert verb("folding a piece of paper") == "UNKNOWN"
    assert verb("assembling the shelf") == "UNKNOWN"
