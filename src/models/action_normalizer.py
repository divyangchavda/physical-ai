"""Deterministic normalization of VLM observations into PhysicalEvents.

NOT the pipeline's event stage. ``src/stages/s07_events.py`` is, and it should
stay that way until a measurement says otherwise:

- It resolves ``actor_track_id``, ``object_track_id`` and ``object_label``, which
  graph_builder, state_inferencer and episode_assembler all read, and sets
  ``review_status`` for s11_score. This module sets none of them.
- On the five real Gemini observations in ``tests/fixtures`` its verb rules score
  4/5 against s07_events' 5/5. On "the person removes the push chopper from the
  cardboard box and then places the box back down" it returns PUSH + PLACE,
  reading PUSH out of the object's name and losing the REMOVE entirely, because
  it has no equivalent of ``_strip_object_phrases``.

What it has that s07_events does not is evidence corroboration: it refuses an
action the visible facts do not support. That is why it is kept.

Why the verb patterns are generated rather than written out
-----------------------------------------------------------
This module used to match bare lemmas: ``\\b(put|set|place|drop)``. Gemini writes
gerunds, and a lemma ending in a silent ``e`` loses it before ``-ing`` — so
"placing" does not contain "place", "closing" does not contain "close", and
"moving" does not contain "move". The rules were therefore blind to the exact
tense the VLM naturally produces, silently and only for e-stem verbs ("lifting"
matched fine, which is why the failure was not obvious). s07_events never had
this bug: its table matches truncated stems ("plac", "clos", "mov").

Measured on the one real observation the project has, "placing the push chopper
into the cardboard box and closing the lid" decomposed to PUSH + UNKNOWN: the
PLACE rule missed "placing", so the sentence fell through to PUSH on the word
"push" *in the object's name*, and "closing" missed CLOSE entirely. Because any
UNKNOWN clause discarded the whole observation, the result was a single UNKNOWN
event.

So verb forms come from :func:`_inflect`, and ``tests/test_action_inflection.py``
pins a word list of the forms that must match. Adding a verb means adding a
lemma, not hand-writing four spellings and forgetting one.
"""

import re
import uuid

from src.schema.event import ActionType, PhysicalEvent
from src.schema.vlm import RawVLMObservation


def _inflect(lemma: str) -> str:
    """Regex fragment matching *lemma* in the tenses a VLM actually writes.

    Covers the base form, third person, past and gerund. Multi-word lemmas
    ("let go") are matched verbatim, since they inflect on the first word and
    are few enough to list explicitly.
    """
    if " " in lemma:
        return re.escape(lemma)
    if lemma.endswith("e"):
        # place -> place places placed placing. The silent e is dropped before
        # -ing, which is the whole reason this function exists.
        return rf"{lemma[:-1]}(?:e|es|ed|ing)"
    if lemma.endswith("y") and len(lemma) > 2 and lemma[-2] not in "aeiou":
        # carry -> carry carries carried carrying
        return rf"{lemma[:-1]}(?:y|ies|ied|ying)"
    # Consonant doubling (drop -> dropped) is not predictable from spelling
    # alone, so both spellings are offered. Generating a form English never
    # uses is harmless — it simply never matches — while missing a real form is
    # the defect this replaces.
    return rf"{lemma}(?:s|es|ed|ing|{lemma[-1]}ed|{lemma[-1]}ing)?"


def _verbs(*lemmas: str) -> re.Pattern[str]:
    """Compile an alternation over the inflected forms of *lemmas*."""
    return re.compile(r"\b(?:" + "|".join(_inflect(w) for w in lemmas) + r")\b")


# Verb groups, as lemmas. _inflect expands each one.
_PICK = _verbs("pick", "lift", "raise")
_GRASP = _verbs("grasp", "grab", "hold", "take hold", "took hold")
_PLACE = _verbs("put", "set", "place", "drop", "lay")
_OPEN = _verbs("open")
_CLOSE = _verbs("close", "shut", "seal")
_MOVE = _verbs("move", "slide", "carry", "shift")
_PUSH = _verbs("push", "shove")
_PULL = _verbs("pull", "drag")
_TOUCH = _verbs("touch", "tap")
_INSPECT = _verbs("look", "inspect", "examine")
_RELEASE = _verbs("release", "let go", "lets go", "letting go", "let it go")
# Deliberately narrow. There is no ground truth for tool use yet, so this fires
# only on explicit tool verbs rather than guessing from context.
_USE_TOOL = _verbs("use", "cut", "chop", "slice", "stir", "hammer", "screw",
                   "wipe", "scrub")

# INSERT and REMOVE are a placement or retrieval verb *plus* a preposition. The
# preposition carries the direction, which is the information the pipeline
# exists to produce, so it is matched separately and checked first.
_INSERT_V = _verbs("put", "place", "insert", "drop", "stuff", "plug", "slide",
                   "push", "load", "pack", "lower")
_REMOVE_V = _verbs("take", "pull", "remove", "lift", "pick", "dig", "extract",
                   "unload", "fish")
_INTO = re.compile(r"\b(into|inside|in to)\b")
_OUT_OF = re.compile(r"\b(out of|out from|from inside|from within)\b")

# Clauses the VLM marks as not actually happening. Something-Something V2 has
# these as their own classes ("Pretending to put something into something",
# "Failing to put something into something because something does not fit"), so
# treating them as real INSERTs would score as confidently wrong.
_NEGATED = re.compile(r"\b(did not|does not fit|didn't|failed to|fails to|"
                      r"pretend|pretends|pretending|without)\b")

# Split on ", and ", ", then ", " and ", " then ", or a bare comma.
_SPLIT = re.compile(r",\s+(?:and\s+|then\s+)?|\s+and\s+|\s+then\s+")


class ActionNormalizer:
    """Deterministically normalizes raw VLM observations into PhysicalEvent objects."""

    def normalize(self, obs: RawVLMObservation) -> list[PhysicalEvent]:
        """Convert a single RawVLMObservation into one or more PhysicalEvents."""

        # 1. Handle missing / explicitly unknown raw action
        if not obs.raw_action or obs.raw_action.strip().upper() == "UNKNOWN":
            return [self._build_event(obs, ActionType.UNKNOWN)]

        raw = obs.raw_action.lower()
        facts = (obs.visible_facts or "").lower()
        state = (obs.state_change or "").lower()
        inf = (obs.inference or "").lower()
        unc = (obs.uncertainty or "").lower()

        # Combine all evidence for general conflict checking
        all_evidence = f"{facts} {inf} {unc}"

        # 2. Decompose a sentence that describes more than one action.
        if re.search(r"\b(and|then)\b", raw) or "," in raw:
            splits = [s.strip() for s in _SPLIT.split(raw) if s.strip()]

            if len(splits) > 1:
                actions = [
                    self._evaluate_single_action(s, facts, state, inf, unc, all_evidence)
                    for s in splits
                ]
                # Keep the clauses that parsed and discard the ones that did not.
                # The previous code required *every* clause to parse and returned
                # a single UNKNOWN otherwise, so one unreadable clause deleted the
                # readable ones -- a confidently derived PICK was thrown away on
                # the strength of an unrelated "wiggled it". Dropping a clause
                # loses recall; letting it veto the sentence loses more.
                known: list[ActionType] = []
                for act in actions:
                    if act is ActionType.UNKNOWN:
                        continue
                    # Collapse consecutive repeats ("picked up and lifted").
                    if not known or known[-1] != act:
                        known.append(act)

                if known:
                    # Only a genuine decomposition forfeits the VLM's own
                    # timestamps: those describe the whole sentence, so they
                    # cannot be attributed to one clause of several. A single
                    # surviving clause is treated like the single-action path.
                    multi = len(known) > 1
                    return [
                        self._build_event(obs, act, segment_timing=multi)
                        for act in known
                    ]

            # No clause parsed. The sentence may still read as one action when
            # taken whole ("moved it up and to the left"), so fall through to the
            # single-action path rather than returning UNKNOWN unexamined.

        # 3. Single event evaluation
        action_type = self._evaluate_single_action(raw, facts, state, inf, unc, all_evidence)
        return [self._build_event(obs, action_type)]

    def _evaluate_single_action(
        self, raw: str, facts: str, state: str, inf: str, unc: str, all_evidence: str
    ) -> ActionType:
        """Evaluate a single action string against the evidence."""

        # Check for uncertainty override
        if "cannot see" in unc or "occluded" in unc or "unclear" in unc:
            return ActionType.UNKNOWN

        # INSERT / REMOVE are tested first. They are ordinary placement and
        # retrieval verbs distinguished only by a preposition, so PLACE or PICK
        # below would claim them and the direction would be lost -- "placing the
        # chopper into the box" is an INSERT, not a PLACE that happens to
        # mention a box.
        if _INTO.search(raw) and _INSERT_V.search(raw):
            if _NEGATED.search(raw) or _NEGATED.search(all_evidence):
                return ActionType.UNKNOWN
            return ActionType.INSERT

        if _OUT_OF.search(raw) and _REMOVE_V.search(raw):
            if _NEGATED.search(raw) or _NEGATED.search(all_evidence):
                return ActionType.UNKNOWN
            return ActionType.REMOVE

        if _RELEASE.search(raw):
            return ActionType.RELEASE

        # PICK & GRASP logic
        if _PICK.search(raw):
            # PICK requires explicit upward movement evidence
            if re.search(r'\b(lift|up|upward|rise|raise)', all_evidence):
                # Check for contradiction
                if re.search(r'\b(did not move|no movement|stationary)\b', all_evidence):
                    return ActionType.UNKNOWN
                return ActionType.PICK
            else:
                # "picked up cup" + no lifting evidence -> UNKNOWN
                return ActionType.UNKNOWN

        if _GRASP.search(raw):
            # Contradiction check
            if re.search(r'\b(did not grasp|no hold|dropped)', all_evidence):
                return ActionType.UNKNOWN
            # Upgrade check: if evidence explicitly shows lifting, upgrade to PICK
            if re.search(r'\b(lift|up|upward|rise|raise)', facts):
                return ActionType.PICK
            return ActionType.GRASP

        # PLACE
        if _PLACE.search(raw):
            if re.search(r'\b(did not put|kept holding|no placement)', all_evidence):
                return ActionType.UNKNOWN
            return ActionType.PLACE

        # OPEN & CLOSE
        if _OPEN.search(raw):
            if "already open" in state or "already open" in facts:
                return ActionType.UNKNOWN
            return ActionType.OPEN

        if _CLOSE.search(raw):
            if "already closed" in state or "already closed" in facts:
                return ActionType.UNKNOWN
            return ActionType.CLOSE

        # State-only fallback check: "door is open" without action transition
        if " is open" in raw or " is closed" in raw:
            return ActionType.UNKNOWN

        # MOVE
        if _MOVE.search(raw):
            # Must confirm the object moved, not just the hand
            if "object was moved" in all_evidence or "moved the object" in all_evidence or "object moves" in all_evidence:
                return ActionType.MOVE
            # False keyword: "moved hand" -> NOT MOVE (unless object movement supported)
            if "moved hand" in raw and not re.search(r'\b(object|it)\b', all_evidence):
                return ActionType.UNKNOWN
            return ActionType.UNKNOWN # require strict evidence for MOVE

        # PUSH / PULL
        if _PUSH.search(raw):
            return ActionType.PUSH
        if _PULL.search(raw):
            return ActionType.PULL

        if _USE_TOOL.search(raw):
            return ActionType.USE_TOOL

        # TOUCH / INSPECT
        if _TOUCH.search(raw):
            return ActionType.TOUCH
        if _INSPECT.search(raw):
            if "inspect" in all_evidence or "examine" in all_evidence or "look closely" in all_evidence:
                return ActionType.INSPECT
            return ActionType.UNKNOWN

        # Default fallback
        return ActionType.UNKNOWN

    def _build_event(self, obs: RawVLMObservation, action_type: ActionType, segment_timing: bool = False) -> PhysicalEvent:
        """Construct a PhysicalEvent from a RawVLMObservation and derived ActionType."""
        # Determine timestamps
        timing_precision = "EXACT"
        start_sec = obs.start_time_sec
        end_sec = obs.end_time_sec

        if start_sec is None or end_sec is None or segment_timing:
            start_sec = obs.segment_start_sec
            end_sec = obs.segment_end_sec
            timing_precision = "SEGMENT"

        # Extract attributes for provenance
        attributes = {
            "vlm_raw_action": obs.raw_action,
            "vlm_visible_facts": obs.visible_facts,
            "vlm_uncertainty": obs.uncertainty,
            "timing_precision": timing_precision
        }

        return PhysicalEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            segment_id=obs.segment_id,
            observation_id=obs.observation_id,
            action=action_type,
            confidence=obs.confidence if obs.confidence is not None else 0.0,
            source=f"vlm_normalized:{obs.model_name}",
            is_estimated=True,
            start_sec=start_sec,
            end_sec=end_sec,
            attributes=attributes
        )
