"""Stage 07 — Physical event extraction.

Converts VLM output and rule-based signals into typed PhysicalEvent objects.
SKIPPED in stub mode or when no candidate segments are available.

Rules:
  - UNKNOWN action when evidence is genuinely insufficient.
  - Raw action + confidence + source always preserved.
  - review_status left as PENDING (set later by s10_score).

Output file: output/events.json
Output context: ctx.events (list[PhysicalEvent])
"""
from __future__ import annotations

import json
import re
import time
import uuid

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.models.track_changepoints import (
    from_pipeline_tracks,
    find_change_points,
    is_inside,
)
from src.schema.episode import PipelineStageStatus
from src.schema.event import PhysicalEvent, ActionType
from src.schema.vlm import VLMSegmentStatus

logger = get_logger(__name__)
STAGE = "s07_events"


def _write_output(ctx: PipelineContext) -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "events.json"
    data = [e.model_dump(mode="json") for e in ctx.events]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Verb stems in precedence order, matched at a leading word boundary so a stem
# cannot fire from inside an unrelated word.
_ACTION_STEMS: tuple[tuple[tuple[str, ...], ActionType], ...] = (
    # Folding/assembling is several primitives at once, not one action type.
    (("fold", "assembl"), ActionType.UNKNOWN),
    (("releas", "let go"), ActionType.RELEASE),
    (("insert",), ActionType.INSERT),
    (("remov",), ActionType.REMOVE),
    (("pick",), ActionType.PICK),
    (("plac", "put down"), ActionType.PLACE),
    (("grasp", "grip", "hold"), ActionType.GRASP),
    (("push",), ActionType.PUSH),
    (("pull",), ActionType.PULL),
    (("open",), ActionType.OPEN),
    (("clos",), ActionType.CLOSE),
    # Placed after open/close deliberately: "opening and unpacking a box" is an
    # OPEN, while "unboxing a push chopper" has no other verb and is a REMOVE.
    (("unbox", "unpack"), ActionType.REMOVE),
    (("using", "use"), ActionType.USE_TOOL),
    (("touch",), ActionType.TOUCH),
    (("inspect", "examin"), ActionType.INSPECT),
    (("mov",), ActionType.MOVE),
)

# A preposition changes which primitive a generic verb is. "placing the chopper
# inside the box" is an INSERT, not a PLACE; the verb alone cannot tell you.
#
# This is not a refinement for its own sake. On tt6 Gemini wrote "inside" and
# "into" on two of four segments — the correct containment relation — and the
# stem table above threw the preposition away and emitted PLACE, whose implied
# direction is ONTO. Measured against hand labels, Gemini's direction was 2/4
# and the pipeline's own output was 0/4: half the shipped direction error was
# introduced here, after the model had already got it right.
#
# Keyed by the action the stems produced, so a promotion can only ever refine a
# verb that already matched, never invent one.
_PREPOSITION_PROMOTIONS: dict[ActionType, tuple[tuple[tuple[str, ...], ActionType], ...]] = {
    ActionType.PLACE: (
        (("into", "inside", "in to", "within"), ActionType.INSERT),
        (("out of", "outside"), ActionType.REMOVE),
    ),
    ActionType.PICK: (
        (("out of", "from inside"), ActionType.REMOVE),
    ),
    ActionType.MOVE: (
        (("into", "inside"), ActionType.INSERT),
        (("out of",), ActionType.REMOVE),
    ),
}

# Words that end the matched verb's clause. A preposition after one of these
# belongs to a different action: in "removes the chopper from the box and then
# places the box down", the "from" must not reach the "places".
_CLAUSE_BREAKS = (" and ", " then ", " after ", " before ", " while ", ",", ";")


def _clause_after(text: str, offset: int) -> str:
    """The matched verb's own clause: from the verb to the next clause break."""
    tail = text[offset:]
    cut = len(tail)
    for token in _CLAUSE_BREAKS:
        found = tail.find(token)
        if found != -1:
            cut = min(cut, found)
    return tail[:cut]


# Built from _CLAUSE_BREAKS so the splitter and the preposition scoping above
# cannot disagree about where one clause ends and the next begins.
_CLAUSE_SPLIT = re.compile("|".join(re.escape(t) for t in _CLAUSE_BREAKS))


def _clause_spans(text: str) -> list[tuple[int, str]]:
    """``(offset, clause)`` for each clause of *text*.

    Offsets are into *text*, because the action an object belongs to is decided
    by distance from the verb — see :func:`_order_objects_by_action` — and a
    clause-relative offset would rank every clause's objects against the first.
    """
    spans: list[tuple[int, str]] = []
    pos = 0
    for match in [*_CLAUSE_SPLIT.finditer(text), None]:
        end = match.start() if match is not None else len(text)
        piece = text[pos:end]
        if piece.strip():
            spans.append((pos + len(piece) - len(piece.lstrip()), piece.strip()))
        pos = match.end() if match is not None else len(text)
    return spans


def _action_clauses(
    action_lower: str, objects: list[str] | None
) -> list[tuple[ActionType, int | None]]:
    """Every distinct action the text describes, as ``(action, verb offset)``.

    One observation is one VLM call over one clip, and the VLM answers about the
    whole clip. "placing the push chopper into the cardboard box and closing the
    lid" is two actions; matching the sentence as a unit returns the first stem
    that hits and the CLOSE was never emitted at all. On the frozen tt7
    observation that is the difference between one event and two, against seven
    hand labels.

    Two rules keep the split from inventing actions:

    - A clause that is nothing but its own verb is not an independent action.
      "opening and unpacking a cardboard box" coordinates two verbs over one
      shared object; splitting it emitted an OPEN *and* a REMOVE for what the
      VLM described as one thing. A verb with no complement of its own belongs
      to the following clause, so it is dropped and the sentence is matched
      whole. (Limited to a one-word clause: "opening it" would still split.)
    - If fewer than two independent actions survive, the whole sentence is
      matched as before. So a single-action observation, or one whose clauses
      are all unreadable, behaves exactly as it did.
    """
    spans = _clause_spans(action_lower)
    found: list[tuple[ActionType, int | None]] = []
    for offset, clause in spans:
        action, rel = _match_action(clause, objects)
        if action is ActionType.UNKNOWN or rel is None:
            continue
        # Bare coordinate verb: the clause is the verb and nothing else.
        if len(clause.split()) == 1:
            continue
        absolute = offset + rel
        # Collapse a verb repeated across adjacent clauses ("lifting and raising").
        if found and found[-1][0] is action:
            continue
        found.append((action, absolute))

    if len(found) > 1:
        return found
    return [_match_action(action_lower, objects)]


def _promote_by_preposition(
    action: ActionType, text: str, offset: int | None
) -> ActionType:
    """Refine a matched verb using the preposition in its own clause."""
    rules = _PREPOSITION_PROMOTIONS.get(action)
    if not rules or offset is None:
        return action
    clause = _clause_after(text, offset)
    for prepositions, promoted in rules:
        if any(re.search(r"\b" + re.escape(p), clause) for p in prepositions):
            return promoted
    return action


def _strip_object_phrases(action_lower: str, objects: list[str] | None) -> str:
    """Blank out the object names in the action text before matching verbs.

    "push chopper" is an object whose name contains a verb, so matching verbs
    against the whole string read "unboxing a push chopper" as a PUSH. Longest
    phrases first so "cardboard box" goes before a bare "box". Each phrase is
    replaced by spaces of equal length, which keeps every offset in the result
    lined up with the original text.
    """
    if not objects:
        return action_lower
    phrases = {o.strip().lower() for o in objects if o and o.strip()}
    for phrase in sorted(phrases, key=len, reverse=True):
        action_lower = re.sub(
            r"\b" + re.escape(phrase) + r"\b",
            lambda m: " " * len(m.group(0)),
            action_lower,
        )
    return action_lower


def _match_action(
    action_lower: str, objects: list[str] | None
) -> tuple[ActionType, int | None]:
    """Return the action type and the offset of the verb that produced it.

    The offset matters because the VLM returns compound actions — "opening and
    closing the cardboard box, then removing the push chopper" — where the
    object belongs to the clause of the matched verb, not to the sentence.
    Offsets are valid against the original text: :func:`_strip_object_phrases`
    preserves length.
    """
    text = _strip_object_phrases(action_lower, objects)
    for stems, action in _ACTION_STEMS:
        starts = [
            m.start() for m in
            (re.search(r"\b" + re.escape(s), text) for s in stems)
            if m is not None
        ]
        if starts:
            offset = min(starts)
            return _promote_by_preposition(action, text, offset), offset
    return ActionType.UNKNOWN, None


def _map_raw_action_to_type(
    raw_action: str, objects: list[str] | None = None
) -> ActionType:
    """Map VLM raw_action string to canonical ActionType enum.

    Rules:
    - Return UNKNOWN when evidence is insufficient or ambiguous
    - Do not force a match; prefer UNKNOWN over incorrect mapping
    - Never read a verb out of an object's name; *objects* is blanked first
    """
    if not raw_action:
        return ActionType.UNKNOWN
    return _match_action(raw_action.lower(), objects)[0]




# Both ends must be strictly inside the segment by this margin for the VLM's
# reported offsets to count as localisation rather than an echo of the clip.
_TIMING_EPSILON_SEC = 0.05


def _timing_precision(obs) -> str:
    """Classify whether the VLM actually localised the action within the clip.

    A VLM handed an 8-second clip frequently returns offsets equal to the clip
    bounds — it is describing the whole thing, not timing the action. Reporting
    that as EXACT puts fabricated precision into the dataset, so only offsets
    strictly inside the segment on *both* ends earn EXACT.
    """
    if obs.start_time_sec is None or obs.end_time_sec is None:
        return "SEGMENT"
    inside_start = obs.start_time_sec > obs.segment_start_sec + _TIMING_EPSILON_SEC
    inside_end = obs.end_time_sec < obs.segment_end_sec - _TIMING_EPSILON_SEC
    return "EXACT" if inside_start and inside_end else "SEGMENT"


def _resolve_actor_track(
    segment_tracks: list, person_classes: set[str]
) -> int | None:
    """Pick the person track that acts in this segment, or None.

    Longest-lived track wins: under fragmentation one person becomes several
    tracks, and the one with the most points is the best evidence of who was
    present for the action. Returns None rather than guessing when the segment
    contains no person track at all.
    """
    candidates = [
        t for t in segment_tracks if t.class_name.lower() in person_classes
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda t: len(t.points)).track_id


def _match_label(vlm_label: str, class_name: str) -> int:
    """Score a VLM object phrase against a track class name. 0 = no match.

    Bidirectional containment: the VLM says "box" for a "cardboard box" track,
    and "the cardboard box" for a "box" one. Exact match scores highest so a
    literal hit always beats a substring hit.
    """
    a = vlm_label.strip().lower()
    b = class_name.strip().lower()
    if not a or not b:
        return 0
    if a == b:
        return 3
    if b in a:
        return 2
    if a in b:
        return 1
    return 0


def _mention_index(label: str, action_lower: str) -> int | None:
    """Character offset where *label* is mentioned in the action text, or None.

    Falls back to the head noun ("cardboard box" -> "box") because the VLM
    names the same thing one way in ``objects`` and another in ``raw_action``.
    Matched at word boundaries: a bare ``in`` test found "box" inside
    "unboxing", which named the box as the object of "unboxing a push chopper".
    """
    phrase = label.strip().lower()
    if not phrase:
        return None
    for needle in (phrase, phrase.split()[-1]):
        match = re.search(r"\b" + re.escape(needle) + r"\b", action_lower)
        if match:
            return match.start()
    return None


# Prepositions that introduce a destination or container rather than the thing
# in the hand: "into the box", "from the shelf", "onto the table".
_LOCATIVE_PREPOSITIONS = (
    "into", "in", "onto", "on", "from", "out of", "off", "inside", "under",
    "over", "at", "against", "toward", "towards", "near", "beside", "to",
)
_DETERMINERS = (" the", " a", " an", " its", " his", " her", " their")


def _is_locative_mention(action_lower: str, idx: int) -> bool:
    """True when the phrase at *idx* is the object of a locative preposition."""
    prefix = action_lower[:idx].rstrip()
    for det in _DETERMINERS:
        if prefix.endswith(det):
            prefix = prefix[: -len(det)].rstrip()
            break
    return any(
        prefix == prep or prefix.endswith(" " + prep)
        for prep in _LOCATIVE_PREPOSITIONS
    )


def _order_objects_by_action(
    raw_action: str | None,
    vlm_objects: list[str],
    verb_index: int | None = None,
) -> list[str]:
    """Reorder *vlm_objects* so the thing the verb acts on comes first.

    The VLM's ``objects`` order is not reliable. For "placing the push chopper
    back into the box" it returned ``["box", "push chopper"]``, so trusting
    position named the container as the manipulated object. Three signals decide
    instead, in order:

    1. Anything led by a locative preposition is a destination, not the object,
       however early it appears ("from the cardboard box").
    2. Objects named after *verb_index* beat objects named before it. The VLM
       returns compound actions, and in "opening and closing the cardboard box,
       then removing the push chopper" the chopper belongs to the matched verb
       while the box belongs to a clause that was not selected.
    3. Nearest to the verb wins, since English puts the direct object first.

    Objects the action never mentions keep their original order at the back; if
    it mentions none, the original order stands.
    """
    action_lower = (raw_action or "").lower()
    if not action_lower:
        return list(vlm_objects)

    mentioned: list[tuple[tuple[bool, int, int, int], str]] = []
    unmentioned: list[str] = []
    for pos, label in enumerate(vlm_objects):
        idx = _mention_index(label, action_lower)
        if idx is None:
            unmentioned.append(label)
            continue
        if verb_index is None:
            before_verb, distance = False, idx
        else:
            before_verb = idx < verb_index
            distance = abs(idx - verb_index)
        key = (_is_locative_mention(action_lower, idx), before_verb, distance, pos)
        mentioned.append((key, label))

    if not mentioned:
        return list(vlm_objects)
    mentioned.sort(key=lambda m: m[0])
    return [label for _, label in mentioned] + unmentioned


def _names_an_object(clause: str, vlm_objects: list[str] | None) -> bool:
    """True when *clause* mentions any of the observation's objects."""
    if not vlm_objects:
        return False
    return any(
        _mention_index(label, clause) is not None for label in vlm_objects
    )


def _surface_track_ids(
    tracks: list, *, nms_iou: float, min_hits: int
) -> set[int]:
    """Tracks that are a marking ON another object, not an object of their own.

    A picture of a thing printed on a carton is detected as that thing. tt7's
    prompt names two decoy classes to absorb exactly this, and the decoys did
    absorb 202 boxes — but three of the printed chopper's detections still won
    the real "push chopper" label, and the tracker built them into a second
    entity. The run then bound the INSERT event to it: _resolve_object_track
    breaks a label tie by track length, and the artwork (frames 99-162) outlived
    the real chopper (frames 0-47). The event said a hand inserted the picture.

    The discriminator is containment, which needs no threshold of mine and is
    exact: a marking on a surface is wholly inside that surface's box on *every*
    frame the two are seen together, because it is physically part of it. Measured
    on tests/fixtures/tt7_real_detections.json:

        push chopper#13 in cardboard box#6 : 16/16 shared frames  <- the artwork
        push chopper#3  in cardboard box#6 :  0/12 shared frames  <- the real one
        cardboard box#6 in person#2        :  6/65 shared frames
        push chopper#3  in person#2        :  4/14 shared frames

    Only the artwork is contained on all of its shared frames; nothing else comes
    close, so the rule is "every shared frame" rather than a fraction picked to
    fit. ``is_inside`` carries its own nms_iou guard against a box relabelled as
    its own contents, and ``min_hits`` — the tracker's own standard for how many
    observations make something real — sets the minimum number of shared frames,
    so a one-frame coincidence cannot demote a track.

    Observed points only: ``from_pipeline_tracks`` drops the Kalman
    extrapolations that make up two thirds of Track.points at stride 3.
    """
    observed = from_pipeline_tracks(tracks)
    surface: set[int] = set()
    for inner in observed:
        for outer in observed:
            if inner.track_id == outer.track_id:
                continue
            shared = sorted(set(inner.boxes) & set(outer.boxes))
            if len(shared) < max(1, min_hits):
                continue
            if all(
                is_inside(inner.boxes[f], outer.boxes[f], nms_iou) for f in shared
            ):
                surface.add(inner.track_id)
                break
    return surface


def _resolve_object_track(
    vlm_objects: list[str] | None,
    segment_tracks: list,
    person_classes: set[str],
    background_classes: set[str],
    raw_action: str | None = None,
    verb_index: int | None = None,
    surface_track_ids: frozenset[int] = frozenset(),
) -> tuple[int | None, str | None]:
    """Resolve the manipulated object to a track id, or (None, label).

    Candidate objects are ordered by what the matched verb acts on rather than
    by the VLM's list order — see :func:`_order_objects_by_action`. Scene
    classes are excluded: a "dining table" is never what the hand is acting on.
    Returns the label even when no track matches, so downstream stages can still
    say *what* went unresolved.

    *surface_track_ids* (see :func:`_surface_track_ids`) lose a label tie to any
    track that is a real object. They are demoted rather than excluded: when the
    only track matching the label is a printed picture of it, saying so is more
    use than saying nothing, and the caller can still see which track was bound.
    """
    if not vlm_objects:
        return None, None

    ordered = _order_objects_by_action(raw_action, vlm_objects, verb_index)

    excluded = person_classes | background_classes
    candidates = [
        t for t in segment_tracks if t.class_name.lower() not in excluded
    ]

    for label in ordered:
        scored = [
            (
                _match_label(label, t.class_name),
                t.track_id not in surface_track_ids,
                len(t.points),
                t.track_id,
            )
            for t in candidates
        ]
        scored = [s for s in scored if s[0] > 0]
        if scored:
            # Best label match first, then a real object over a marking on one,
            # then the longest-lived of those tracks.
            best = max(scored, key=lambda s: (s[0], s[1], s[2]))
            return best[3], label

    return None, ordered[0]


def _clause_windows(
    n_clauses: int,
    change_secs: list[float],
    seg_start: float,
    seg_end: float,
) -> list[tuple[float, float]] | None:
    """Cut ``[seg_start, seg_end]`` into *n_clauses* windows at scene changes.

    Returns ``None`` when the change points cannot supply enough cuts, which
    leaves the caller on its existing whole-segment fallback. Nothing is invented:
    every boundary is a frame at which the tracks measurably changed.

    The cuts are spread evenly over the *ordered list of change points*, not over
    time — cut ``i`` of ``n`` is change point ``int(i * k / n)`` of ``k``. That is
    a rule rather than a fit: no clause is matched to the change point that would
    score best. Which matters, because on tt7 it has been measured that geometry
    does NOT beat the naive alternative. tools/tt7_changepoints.py scores this
    clip's 12 change points against the 7 hand labels at 7/8 in-order, and a
    CONTROL of 12 evenly spaced times — which never saw the video — scores 8/8.

    So the claim this makes is narrow and is the whole point: clauses of one
    observation get *distinct* spans where today they all inherit the same
    segment, which is what tools/score_run.py needs before any attribution is
    possible at all. It is not a claim that these boundaries are the right ones.
    """
    if n_clauses <= 1:
        return None
    interior = sorted({s for s in change_secs if seg_start < s < seg_end})
    k = len(interior)
    if k < n_clauses - 1:
        return None

    cuts = [interior[int(i * k / n_clauses)] for i in range(1, n_clauses)]
    # int(i*k/n) is strictly increasing for k >= n-1, but a repeated change-point
    # time would still collapse two windows to zero width. Refuse rather than
    # emit an event with start == end.
    if any(b <= a for a, b in zip(cuts, cuts[1:])):
        return None

    bounds = [seg_start, *cuts, seg_end]
    return list(zip(bounds, bounds[1:]))


# Verbs that set a binary object state, with the state each one requires the
# object to be in beforehand and the state it leaves behind. Only these two
# qualify: every other ActionType either has no binary state (MOVE, TOUCH) or
# does not become impossible by being repeated (you can PICK an object twice).
_STATE_VERBS: dict[ActionType, tuple[str, str]] = {
    ActionType.OPEN: ("CLOSED", "OPEN"),
    ActionType.CLOSE: ("OPEN", "CLOSED"),
}
_COMPLEMENT: dict[ActionType, ActionType] = {
    ActionType.OPEN: ActionType.CLOSE,
    ActionType.CLOSE: ActionType.OPEN,
}

# Explicit state evidence. Deliberately wider than the patterns in
# src/models/state_inferencer.py, which are `is open|opened` and
# `is closed|closed|shuts` — those miss the only seed tt7 actually provides.
# Measured, not preferred: the run's first observation says "holding the top flap
# of an open cardboard box", where "open" is a bare adjective that `is open`
# cannot match. Adding the bare form is what the text forces.
#
# "open" does not match "opening": \b requires a boundary after the n, and
# "opening" continues with an i. That is essential — a progressive verb describes
# an action in flight, not a state that has been reached, and letting "opening"
# seed OPEN would make every one of these observations vouch for itself.
_OPEN_EVIDENCE = re.compile(r"\b(?:is open|was open|opened|open)\b")
_CLOSED_EVIDENCE = re.compile(r"\b(?:is closed|was closed|closed|shut|shuts)\b")


def _state_evidence(event: PhysicalEvent) -> tuple[str, str, int, str] | None:
    """Read an explicit open/closed state off one event's observed text.

    Returns ``(state, the phrase it came from, that phrase's character offset,
    the text it was found in)``, or ``None`` when the text says nothing about the
    state. The offset and text are returned because the state word does not
    necessarily describe the object the *action* is on — see ``_state_subject``,
    which needs both to decide which object is meant.

    Only ``visible_facts`` and ``state_change`` are consulted:

    * ``raw_action`` is excluded because it holds the verb under test. "closing
      the box" would seed CLOSED and then veto its own CLOSE.
    * ``inference`` is excluded because the prompt defines it as the model's
      reasoning *beyond* what is visible, so it is the one field explicitly not
      evidence.

    A text mentioning both states is ambiguous and seeds nothing, rather than
    letting match order decide.
    """
    for field in ("visible_facts", "state_change"):
        text = (event.attributes.get(field) or "").lower()
        if not text:
            continue
        opened = _OPEN_EVIDENCE.search(text)
        closed = _CLOSED_EVIDENCE.search(text)
        if opened and closed:
            continue
        if opened:
            return "OPEN", opened.group(0), opened.start(), text
        if closed:
            return "CLOSED", closed.group(0), closed.start(), text
    return None


def _best_label_track(label: str, label_to_track: dict[str, int]) -> int | None:
    """The track some event already resolved for *label*, by the usual matcher.

    ``_match_label`` is reused rather than a second rule, so "box" reaches a
    "cardboard box" track exactly as it does when binding an event's object.
    """
    best_score, best_track = 0, None
    for resolved, track_id in label_to_track.items():
        score = _match_label(label, resolved)
        if score > best_score:
            best_score, best_track = score, track_id
    return best_track


def _state_subject(
    text: str,
    offset: int,
    candidate_labels: list[str],
    label_to_track: dict[str, int],
) -> int | None:
    """The track a state word at *offset* describes: the nearest object named.

    The reason this exists, from the C_seg1 run of tt7::

        "holding a small kitchen appliance in their left hand
         and an OPEN cardboard box in their right hand"   -> GRASP, object=push chopper

    The run's only state word is in that sentence, but the event it belongs to
    resolved the *chopper*, so seeding the event's own object recorded the box's
    openness against the chopper and the three OPEN events on the box found
    nothing to contradict. The state word describes whatever noun it sits beside,
    which is not in general the noun the verb acts on.

    Nearest-by-character-offset is a stated design choice, not a measured
    constant: there is no threshold here, only which mention is closer in the
    text that arrived. It is wrong under distant modification ("the box that the
    chopper came out of is open" picks the chopper). ``_mention_index`` supplies
    the offsets, so a label the text never names cannot be chosen at all — on the
    sentence above "push chopper" falls back to its head noun "chopper", which
    that text does not contain, leaving "cardboard box" as the only candidate.

    Ties break toward the longer label, then the lower track id, so the result
    never depends on dict ordering.
    """
    best: tuple[int, int, int] | None = None
    seen_labels: set[str] = set()
    for label in candidate_labels:
        key = label.strip().lower()
        if not key or key in seen_labels:
            continue
        seen_labels.add(key)
        track_id = label_to_track.get(key)
        if track_id is None:
            track_id = _best_label_track(label, label_to_track)
        if track_id is None:
            continue
        idx = _mention_index(label, text)
        if idx is None:
            continue
        candidate = (abs(idx - offset), -len(key), track_id)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best else None


def _resolve_state_contradictions(events: list[PhysicalEvent]) -> list[PhysicalEvent]:
    """Correct state verbs that the object's own tracked state makes impossible.

    The failure this exists for, from the B_seg1 run of tt7 (committed verbatim
    as tests/fixtures/tt7_b_seg1_observations.json)::

        0.00  "...holding the top flap of an OPEN cardboard box"
        1.90  "opening the flaps of a cardboard box"
        2.86  "opening the top flaps of a cardboard box"
        3.81  "opening the top flap of a cardboard box"

    Ground truth is FOLD then CLOSE. A box the model itself called open at 0.0s
    cannot be opened three more times; for a binary state, OPEN applied to an
    already-OPEN object is unsatisfiable, and the only transition still available
    to those flaps is closing. That is a deduction from the model's own words,
    not a threshold chosen here, and the seed state is required to be an explicit
    word in the text — an object with no stated state is never touched.

    The seed is attributed to the object the state word *describes*, via
    ``_state_subject``, not to the object the event's verb acts on. The first live
    run of this rule corrected nothing precisely because those differ: the only
    state word in the clip sat in a GRASP event that had bound the push chopper,
    so the box's openness was recorded against the chopper.

    Runs of the same verb on the same object are corrected **together**. Treating
    them one at a time flips the state after each event, so the second OPEN would
    find a CLOSED box, read as satisfiable, and survive — the corrections would
    alternate instead of converging. Consecutive identical state verbs on one
    object are one continuing transition, which is also the only physical reading:
    a binary state cannot be set to the same value three times by three actions.

    Every rewrite records ``verb_source``, the phrase that seeded the state and
    when it was said, so no correction is invisible in events.json.
    """
    ordered = sorted(range(len(events)), key=lambda i: events[i].start_sec)
    # Every label some event in this episode managed to bind, so a state word can
    # name an object the event it appears in is not about. Built once from the
    # events themselves — no second resolution pass, and no access to ctx needed.
    label_to_track: dict[str, int] = {}
    for event in events:
        label = event.attributes.get("object_label")
        if label and event.object_track_id is not None:
            label_to_track.setdefault(str(label).strip().lower(), event.object_track_id)

    # track_id -> (state, the phrase that said so, when it was said)
    state: dict[int, tuple[str, str, float]] = {}
    corrected = 0

    pos = 0
    while pos < len(ordered):
        idx = ordered[pos]
        event = events[idx]
        verb, obj = event.action, event.object_track_id

        if verb not in _STATE_VERBS or obj is None:
            # Not a state verb, so it cannot contradict anything — but its text
            # may still be the only place any object's state is ever stated, and
            # not necessarily this event's object.
            seen = _state_evidence(event)
            if seen:
                seen_state, phrase, offset, text = seen
                subject = _state_subject(
                    text,
                    offset,
                    [*(event.attributes.get("objects") or []), *label_to_track],
                    label_to_track,
                )
                # Falling back to the event's own object keeps the old behaviour
                # whenever the text names nothing better; the subject lookup only
                # ever overrides it when the text actually points elsewhere.
                if subject is None:
                    subject = obj
                if subject is not None:
                    state[subject] = (seen_state, phrase, event.start_sec)
            pos += 1
            continue

        # The maximal run of this same verb on this same object.
        run = [idx]
        while pos + len(run) < len(ordered):
            nxt = events[ordered[pos + len(run)]]
            if nxt.action is not verb or nxt.object_track_id != obj:
                break
            run.append(ordered[pos + len(run)])

        known = state.get(obj)
        _, yields = _STATE_VERBS[verb]
        if known is not None and known[0] == yields:
            replacement = _COMPLEMENT[verb]
            for i in run:
                events[i].action = replacement
                events[i].attributes["verb_source"] = "STATE_UNSATISFIABLE"
                events[i].attributes["verb_before_correction"] = verb.value
                events[i].attributes["state_evidence"] = (
                    f"{obj} already {known[0]} from \"{known[1]}\" "
                    f"at {known[2]:.2f}s"
                )
            corrected += len(run)
            logger.info(
                "[%s] track %d was already %s (\"%s\" at %.2fs); %d consecutive "
                "%s event(s) are unsatisfiable -> %s",
                STAGE, obj, known[0], known[1], known[2], len(run),
                verb.value, replacement.value,
            )
            state[obj] = (_STATE_VERBS[replacement][1], "corrected verb",
                          events[run[-1]].start_sec)
        else:
            state[obj] = (yields, f"{verb.value} event", events[run[-1]].start_sec)

        pos += len(run)

    if corrected:
        logger.info(
            "[%s] %d of %d event(s) had a state verb the object's own tracked "
            "state made impossible", STAGE, corrected, len(events),
        )
    return events


def _extract_events_from_vlm_observations(ctx: PipelineContext) -> list[PhysicalEvent]:
    """Convert VLM observations to PhysicalEvent objects."""
    events = []

    # Build segment to track mapping
    segment_track_map = {}
    for seg in ctx.candidate_segments:
        segment_track_map[seg.segment_id] = seg.track_ids

    track_by_id = {t.track_id: t for t in ctx.tracks}
    person_classes = {c.lower() for c in ctx.config.segment.person_classes}
    background_classes = {c.lower() for c in ctx.config.segment.background_classes}

    # Both derived once per run from the config's own numbers. See
    # _surface_track_ids and src/models/track_changepoints for where each
    # threshold comes from; none of them is a judgement made here.
    nms_iou = ctx.config.detector.nms_iou
    min_hits = ctx.config.tracker.min_hits
    stride = ctx.config.frame_sampling.every_n_frames
    fps = ctx.video_metadata.fps if ctx.video_metadata else 0.0

    surface_ids = frozenset(
        _surface_track_ids(ctx.tracks, nms_iou=nms_iou, min_hits=min_hits)
    )
    if surface_ids:
        logger.info(
            "[%s] %d track(s) are a marking on another object, demoted when "
            "binding objects: %s",
            STAGE, len(surface_ids),
            {tid: track_by_id[tid].class_name
             for tid in sorted(surface_ids) if tid in track_by_id},
        )

    # Frames at which the tracks measurably changed, used to give the clauses of
    # one observation distinct spans. Computed once; filtered per segment below.
    change_points = (
        find_change_points(
            from_pipeline_tracks(ctx.tracks),
            nms_iou=nms_iou,
            stride=stride,
            min_hits=min_hits,
            fps=fps,
            exclude_classes=frozenset(background_classes),
        )
        if fps > 0.0 else []
    )

    for obs in ctx.vlm_observations:
        if obs.status != VLMSegmentStatus.SUCCESS:
            logger.debug("[%s] Skipping observation %s with status %s",
                        STAGE, obs.observation_id, obs.status)
            continue

        # Resolve identities from the segment's tracks by class, never by list
        # position — track_ids is ordered by track id, so track_ids[0] is
        # whatever was detected first, not the actor.
        segment_tracks = [
            track_by_id[tid]
            for tid in segment_track_map.get(obs.segment_id, [])
            if tid in track_by_id
        ]
        actor_track_id = _resolve_actor_track(segment_tracks, person_classes)

        # Map raw_action to ActionType. The object names are blanked out first —
        # "push chopper" carries a verb in its name — and the verb's offset then
        # decides which clause of a compound action owns the object.
        clauses = _action_clauses((obs.raw_action or "").lower(), obs.objects)

        # The VLM's own offsets describe the sentence. When the sentence held
        # several actions they cannot be attributed to one of them from the text,
        # so the spans come from geometry: the frames at which this segment's own
        # tracks changed, cut into one window per clause. When the change points
        # cannot supply enough cuts, the segment bounds remain the honest answer.
        multi = len(clauses) > 1
        windows: list[tuple[float, float]] | None = None
        if multi:
            segment_track_ids = {t.track_id for t in segment_tracks}
            change_secs = [
                p.sec for p in change_points if p.track_id in segment_track_ids
            ]
            windows = _clause_windows(
                len(clauses), change_secs,
                obs.segment_start_sec, obs.segment_end_sec,
            )
            start_sec, end_sec = obs.segment_start_sec, obs.segment_end_sec
            # timing_precision stays in {EXACT, SEGMENT}: src/schema/episode.py
            # and interaction_graph.py declare it a Literal and state_inferencer
            # coerces anything else, so a third value would break those stages.
            # A derived window is not the VLM localising the action, so SEGMENT is
            # correct either way; timing_source records where the span came from.
            timing_precision = "SEGMENT"
            timing_source = "CHANGE_POINT" if windows else "SEGMENT"
            if windows:
                logger.info(
                    "[%s] segment %s: %d clauses cut at change points -> %s",
                    STAGE, obs.segment_id, len(clauses),
                    [f"[{a:.2f}, {b:.2f}]" for a, b in windows],
                )
            else:
                logger.info(
                    "[%s] segment %s: %d clauses but only %d interior change "
                    "point(s) — all clauses keep the segment span",
                    STAGE, obs.segment_id, len(clauses),
                    len({s for s in change_secs
                         if obs.segment_start_sec < s < obs.segment_end_sec}),
                )
        else:
            start_sec = (
                obs.start_time_sec if obs.start_time_sec is not None
                else obs.segment_start_sec
            )
            end_sec = (
                obs.end_time_sec if obs.end_time_sec is not None
                else obs.segment_end_sec
            )
            timing_precision = _timing_precision(obs)
            timing_source = "VLM" if timing_precision == "EXACT" else "SEGMENT"

        for clause_index, (action_type, verb_index) in enumerate(clauses):
            if windows is not None:
                start_sec, end_sec = windows[clause_index]
            clause_text = (
                _clause_after((obs.raw_action or "").lower(), verb_index)
                if verb_index is not None else None
            )

            # Resolved per clause, not per observation. "removes the push
            # chopper from the cardboard box and then places the box back down"
            # acts on two different objects, and resolving against the whole
            # sentence gave both events the chopper: _mention_index finds a
            # label's FIRST occurrence, so the "box" of the second clause was
            # scored at the position of the first, and the chopper's locative
            # penalty from "from the cardboard box" was applied to a clause that
            # has no locative at all.
            #
            # Only done when the observation really produced several events. For
            # a single event the clause IS the sentence, and scoping a
            # coordinate verb like "opening" to its own one-word clause would
            # lose the object it shares with the rest of the sentence.
            if multi and clause_text is not None:
                if _names_an_object(clause_text, obs.objects):
                    object_track_id, object_label = _resolve_object_track(
                        obs.objects, segment_tracks, person_classes,
                        background_classes, raw_action=clause_text, verb_index=0,
                        surface_track_ids=surface_ids,
                    )
                else:
                    # "closing the lid" names a part the VLM never listed as an
                    # object. Borrowing another clause's object would attribute
                    # the close to the chopper, which is a fact about the video
                    # that is not true; unresolved is the honest answer.
                    object_track_id, object_label = None, None
            else:
                object_track_id, object_label = _resolve_object_track(
                    obs.objects, segment_tracks, person_classes,
                    background_classes,
                    raw_action=obs.raw_action, verb_index=verb_index,
                    surface_track_ids=surface_ids,
                )

            # Create PhysicalEvent
            event = PhysicalEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                segment_id=obs.segment_id,
                observation_id=obs.observation_id,
                action=action_type,
                confidence=obs.confidence if obs.confidence is not None else 0.0,
                source=f"vlm:{obs.backend.lower()}",
                is_estimated=True,
                actor_track_id=actor_track_id,
                object_track_id=object_track_id,
                start_sec=start_sec,
                end_sec=end_sec,
                attributes={
                    "raw_action": obs.raw_action,
                    "actor": obs.actor,
                    "active_hand": obs.active_hand,
                    "objects": obs.objects,
                    "object_label": object_label,
                    "state_change": obs.state_change,
                    "visible_facts": obs.visible_facts,
                    "inference": obs.inference,
                    "uncertainty": obs.uncertainty,
                    "model_name": obs.model_name,
                    "timing_precision": timing_precision,
                    # Where start_sec/end_sec actually came from. Kept separate
                    # from timing_precision because that field is a Literal in
                    # three schemas; this one is free text in attributes, so a
                    # geometry-derived window is distinguishable from a segment
                    # fallback without breaking a downstream validator.
                    "timing_source": timing_source,
                    # Which clause of raw_action produced this event. With one
                    # observation now yielding several, there is otherwise no
                    # way to tell them apart in events.json.
                    "action_clause": (
                        _clause_after((obs.raw_action or "").lower(), verb_index)
                        if verb_index is not None else None
                    ),
                },
                review_status="PENDING"
            )
            events.append(event)

            logger.debug("[%s] Created event %s: action=%s, confidence=%.2f, time=[%.1f, %.1f]s",
                        STAGE, event.event_id, event.action, event.confidence,
                        event.start_sec, event.end_sec)

    # Last, because it reads the object each event resolved and needs them all.
    return _resolve_state_contradictions(events)


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if ctx.config.stub_mode:
        logger.info("[%s] stub_mode=True — SKIPPED (no events fabricated)", STAGE)
        ctx.events = []
        _write_output(ctx)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: event extraction skipped",
            duration_sec=time.monotonic() - t0,
        )

    if not ctx.candidate_segments:
        logger.info("[%s] No candidate segments — SKIPPED", STAGE)
        ctx.events = []
        _write_output(ctx)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="no candidate segments available",
            duration_sec=time.monotonic() - t0,
        )

    # Extract events from VLM observations
    ctx.events = _extract_events_from_vlm_observations(ctx)
    
    _write_output(ctx)
    
    duration = time.monotonic() - t0
    logger.info("[%s] Extracted %d events from %d VLM observations in %.3fs",
               STAGE, len(ctx.events), len(ctx.vlm_observations), duration)
    
    return PipelineStageStatus(
        stage=STAGE, status="OK",
        message=f"Extracted {len(ctx.events)} physical events",
        duration_sec=duration,
    )
