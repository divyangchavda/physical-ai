"""Tests for grounded-span -> class resolution in the GroundingDINO-HF detector.

GroundingDINO does phrase grounding, so it returns the text span it matched
rather than a class index, and the span is a detokenised run of wordpieces. It
can therefore be a fragment, a merge of two prompt terms, or — the case that
broke tt7 — a label the tokenizer split mid-word, returned with the wordpiece
continuation marker still attached.

The tt7 decoy-B run (prompt below) produced these tracks:

    13  ##on label        f102..114   5 detections
    14  chopper picture   f153..168   3 detections
    16  ##on label        f174..186   3 detections

All 11 boxes are the chopper printed on the carton — precisely what the two
decoy labels were added to absorb. They survived because substring matching
resolved neither span: "##on label" contains no prompt label and is contained
in none, so it fell to the unmatched bucket, and the decoy filter tests
class_id against the decoy ids, which the unmatched id can never be. Nothing
downstream drops unmatched spans, so each became a class of its own and reached
the tracker.

No model is loaded here — resolution is pure string work on the parsed prompt.
"""
from __future__ import annotations

from src.models.groundingdino_hf_detector import GroundingDINOHFDetector

# The tt7 vocabulary B, verbatim from config/kaggle_tt7_decoy_b.yaml.
PROMPT_B = ("person . cardboard box . push chopper . dining table . "
            "picture of a push chopper . printed carton label .")
DECOYS_B = ["picture of a push chopper", "printed carton label"]

# The tt6/baseline vocabulary, which must keep resolving exactly as it did.
PROMPT_BASE = "person . cardboard box . push chopper . dining table ."


def _det(prompt: str = PROMPT_B, **kwargs) -> GroundingDINOHFDetector:
    return GroundingDINOHFDetector(text_prompt=prompt, device="cpu", **kwargs)


# ───────────────────────────────────────────────── the two spans tt7 produced
def test_a_wordpiece_span_resolves_to_the_label_it_came_from():
    """"##on label" is "printed carton label" with its first words split off."""
    det = _det()
    class_id, class_name = det._resolve_class("##on label")
    assert class_name == "printed carton label"
    assert class_id == det._label_to_id["printed carton label"]
    assert class_id != det._unmatched_id


def test_a_wordpiece_span_of_a_decoy_is_now_droppable():
    """The point of the fix: the box reaches the decoy filter as a decoy id.

    This is what the 8 "##on label" detections in the tt7 run needed and did
    not get.
    """
    det = _det(decoy_classes=DECOYS_B)
    class_id, _ = det._resolve_class("##on label")
    assert class_id in det._decoy_ids


def test_a_reordered_span_prefers_the_label_sharing_more_words():
    """"chopper picture" shares 2 words with the decoy and 1 with the real class."""
    det = _det(decoy_classes=DECOYS_B)
    class_id, class_name = det._resolve_class("chopper picture")
    assert class_name == "picture of a push chopper"
    assert class_id in det._decoy_ids


def test_a_bare_head_noun_still_prefers_the_shorter_label():
    """The counterweight to the test above, and why ranking needs two keys.

    "chopper" shares one word with both "push chopper" and "picture of a push
    chopper". Shared count alone ties, and the old longest-label rule would
    hand a genuine chopper detection to the decoy and delete it. The fraction
    of the label's own words that matched — 1/2 against 1/5 — breaks it toward
    the real class.
    """
    det = _det(decoy_classes=DECOYS_B)
    class_id, class_name = det._resolve_class("chopper")
    assert class_name == "push chopper"
    assert class_id not in det._decoy_ids


# ─────────────────────────────────────────── behaviour that must not regress
def test_an_exact_label_wins_outright():
    det = _det()
    for name in ("person", "push chopper", "picture of a push chopper"):
        class_id, class_name = det._resolve_class(name)
        assert (class_name, class_id) == (name, det._label_to_id[name])


def test_a_merged_span_resolves_to_the_label_it_shares_most_with():
    """The documented merge case: "cardboard box chopper" -> "cardboard box"."""
    det = _det()
    _, class_name = det._resolve_class("cardboard box chopper")
    assert class_name == "cardboard box"


def test_a_span_truncated_mid_word_still_falls_back_to_substring():
    """No whole word survives in "choppe", so word matching cannot fire.

    Substring matching is kept behind word matching for exactly this case;
    dropping it would send these spans to the unmatched bucket.
    """
    det = _det(PROMPT_BASE)
    _, class_name = det._resolve_class("choppe")
    assert class_name == "push chopper"


def test_the_baseline_vocabulary_resolves_unchanged():
    """The tt6 prompt is single- and two-word labels with no shared words.

    Word matching and the old substring rule agree on every fragment it can
    produce, so this fix cannot move the tt6 numbers.
    """
    det = _det(PROMPT_BASE)
    for span, expected in (
        ("chopper", "push chopper"),
        ("box", "cardboard box"),
        ("cardboard", "cardboard box"),
        ("table", "dining table"),
        ("dining", "dining table"),
        ("person", "person"),
        ("push chopper", "push chopper"),
    ):
        _, class_name = det._resolve_class(span)
        assert class_name == expected, span


def test_a_genuinely_unknown_span_is_still_unmatched():
    """Sharing no word with any label must not be coerced into a class."""
    det = _det()
    class_id, class_name = det._resolve_class("bicycle")
    assert class_id == det._unmatched_id
    assert class_name == "bicycle"
    assert det._unmatched_spans == {"bicycle": 1}


def test_an_empty_span_is_still_unmatched():
    det = _det()
    assert det._resolve_class("  . ") == (det._unmatched_id, "unmatched")
