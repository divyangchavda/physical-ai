"""Tests for decoy classes in the GroundingDINO-HF detector.

The behaviour under test came from a measured failure. tt6 shows a push chopper
being packed into its own retail carton, and the carton has the product printed
on it. GroundingDINO does phrase grounding — every box goes to its best-matching
span among the labels it was *given* — so the printed chopper had nowhere to land
except "push chopper". 52% of chopper detections came after the labels say the
box was closed, and crops of those frames show the carton's artwork.

Confidence cannot separate them: the printed panel scored 0.62-0.67 while genuine
detections scored 0.41-0.55. Raising box_threshold or text_threshold removes real
detections first, which is why the fix is a competing label rather than a number.

No model is loaded here. Decoy resolution happens in __init__ and the prompt is
built there too, so the parts worth pinning are reachable without a GPU.
"""
from __future__ import annotations

import pytest

from src.models.groundingdino_hf_detector import GroundingDINOHFDetector

PROMPT = ("person . cardboard box . push chopper . dining table . "
          "printed label on box . product photo .")


def _detector(**kwargs) -> GroundingDINOHFDetector:
    return GroundingDINOHFDetector(text_prompt=PROMPT, device="cpu", **kwargs)


def test_no_decoys_by_default():
    """The existing behaviour must be untouched when the setting is unused."""
    det = _detector()
    assert det._decoy_ids == set()


def test_a_decoy_must_be_in_the_prompt():
    """The whole mechanism depends on the model being offered the span.

    A decoy absent from the prompt silently never fires, and the symptom would
    be indistinguishable from "the idea did not work" — so it is an error, not a
    no-op.
    """
    with pytest.raises(ValueError, match="not in the text prompt"):
        _detector(decoy_classes=["box artwork"])


def test_decoys_resolve_to_prompt_label_ids():
    det = _detector(decoy_classes=["printed label on box", "product photo"])
    assert det._decoy_ids == {
        det._label_to_id["printed label on box"],
        det._label_to_id["product photo"],
    }


def test_decoys_are_matched_case_insensitively():
    det = _detector(decoy_classes=["Printed Label On Box"])
    assert det._decoy_ids == {det._label_to_id["printed label on box"]}


def test_a_decoy_stays_in_the_prompt_sent_to_the_model():
    """Dropping must happen after grounding, never by removing the label.

    If the decoy were stripped from the prompt the printed artwork would go
    straight back to "push chopper" — the confusion needs somewhere to land.
    """
    det = _detector(decoy_classes=["printed label on box", "product photo"])
    assert "printed label on box" in det.text_prompt
    assert "product photo" in det.text_prompt


def test_decoys_do_not_shift_the_ids_of_real_classes():
    """Class ids come from prompt order, so adding decoys last is deliberate.

    A decoy inserted before a real class would renumber it, and class_id is
    written into every Detection. Pinning this keeps the two prompts comparable
    across runs, which is what makes a before/after score difference readable.
    """
    without = GroundingDINOHFDetector(
        text_prompt="person . cardboard box . push chopper . dining table .",
        device="cpu",
    )
    with_decoys = _detector(decoy_classes=["printed label on box"])
    for name in ("person", "cardboard box", "push chopper", "dining table"):
        assert without._label_to_id[name] == with_decoys._label_to_id[name]


def test_the_unmatched_bucket_still_sits_past_every_label():
    """class_id must stay >= 0 and never collide with a real class."""
    det = _detector(decoy_classes=["product photo"])
    assert det._unmatched_id == len(det.labels)
    assert det._unmatched_id not in det._decoy_ids
