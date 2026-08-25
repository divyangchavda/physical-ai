"""Pin the verb forms the normalizer must recognise.

This file exists because of a silent failure. The rules in action_normalizer.py
were written as bare lemmas — ``\\b(put|set|place|drop)`` — and a lemma ending in
a silent ``e`` loses it before ``-ing``: "placing" does not contain "place",
"closing" does not contain "close". Gemini writes gerunds, so those rules could
never fire on real VLM output, while "lifting" and "pushing" matched fine and
made the gap look like it was not there.

The measured consequence: "placing the push chopper into the cardboard box and
closing the lid" produced one UNKNOWN event. PLACE missed "placing", so the
clause fell through to PUSH on the word "push" inside the *object's name*, and
CLOSE missed "closing".

So this tests the morphology directly, on a word list, rather than testing it
through the pipeline where a miss shows up only as a slightly worse score. A new
verb is added as a lemma; these tests are what prove the lemma expanded.
"""
from __future__ import annotations

import pytest

from src.models.action_normalizer import _inflect, _verbs

# (lemma, forms that MUST match). Gerunds are listed for every verb because the
# VLM prompt asks for a description of what is happening, which is the tense it
# answers in.
MUST_MATCH = [
    ("place", ["place", "places", "placed", "placing"]),
    ("close", ["close", "closes", "closed", "closing"]),
    ("move", ["move", "moves", "moved", "moving"]),
    ("slide", ["slide", "slides", "sliding"]),
    ("raise", ["raise", "raises", "raised", "raising"]),
    ("release", ["release", "releases", "released", "releasing"]),
    ("push", ["push", "pushes", "pushed", "pushing"]),
    ("pull", ["pull", "pulls", "pulled", "pulling"]),
    ("lift", ["lift", "lifts", "lifted", "lifting"]),
    ("pick", ["pick", "picks", "picked", "picking"]),
    ("drop", ["drop", "drops", "dropped", "dropping"]),
    ("plug", ["plug", "plugs", "plugged", "plugging"]),
    ("stuff", ["stuff", "stuffs", "stuffed", "stuffing"]),
    ("grab", ["grab", "grabs", "grabbed", "grabbing"]),
    ("put", ["put", "puts", "putting"]),
    ("shut", ["shut", "shuts", "shutting"]),
    ("carry", ["carry", "carries", "carried", "carrying"]),
    ("open", ["open", "opens", "opened", "opening"]),
    ("insert", ["insert", "inserts", "inserted", "inserting"]),
    ("take", ["take", "takes", "taking"]),
]


@pytest.mark.parametrize("lemma,forms", MUST_MATCH)
def test_every_inflected_form_matches(lemma: str, forms: list[str]):
    pattern = _verbs(lemma)
    for form in forms:
        assert pattern.search(form), (
            f"{lemma!r} expands to {_inflect(lemma)!r}, which does not match "
            f"{form!r} — the VLM writes this form and the rule would miss it"
        )


@pytest.mark.parametrize("lemma,forms", MUST_MATCH)
def test_forms_match_inside_a_sentence(lemma: str, forms: list[str]):
    """Matching a bare word is not enough; the rules run against prose."""
    pattern = _verbs(lemma)
    for form in forms:
        assert pattern.search(f"the person is {form} the object on the table")


def test_the_gerund_that_started_this():
    """The exact two clauses from the one real observation the project has."""
    assert _verbs("place").search("placing the push chopper into the cardboard box")
    assert _verbs("close").search("closing the lid")


def test_a_lemma_does_not_match_an_unrelated_longer_word():
    """Over-broad prefixes would fire on nouns that merely share a stem.

    ``\\bplace`` matched "placement", so the phrase "no placement" — used as
    *negative* evidence that nothing was placed — read as a PLACE verb.
    """
    assert not _verbs("place").search("no placement was observed")
    assert not _verbs("close").search("the closet door")
    assert not _verbs("open").search("the opener is on the shelf")


def test_multiword_lemmas_are_matched_verbatim():
    pattern = _verbs("let go", "lets go", "letting go")
    for form in ("let go", "lets go", "letting go"):
        assert pattern.search(f"the person {form} of the handle")


def test_irregular_past_tenses_are_a_known_gap():
    """Recorded, not fixed. _inflect applies spelling rules; it cannot know that
    the past of "slide" is "slid" or of "take" is "took".

    Left alone deliberately. The VLM is asked what is happening and answers in
    the present participle, so gerunds are what actually arrive; a stem table for
    irregular pasts would be code carrying no measured weight. If a real
    observation ever turns up "slid" or "took", add the form as its own lemma and
    delete this test.
    """
    assert not _verbs("slide").search("slid the box across")
    assert not _verbs("take").search("took the cup")
