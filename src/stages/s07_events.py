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
            return action, min(starts)
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


def _resolve_object_track(
    vlm_objects: list[str] | None,
    segment_tracks: list,
    person_classes: set[str],
    background_classes: set[str],
    raw_action: str | None = None,
    verb_index: int | None = None,
) -> tuple[int | None, str | None]:
    """Resolve the manipulated object to a track id, or (None, label).

    Candidate objects are ordered by what the matched verb acts on rather than
    by the VLM's list order — see :func:`_order_objects_by_action`. Scene
    classes are excluded: a "dining table" is never what the hand is acting on.
    Returns the label even when no track matches, so downstream stages can still
    say *what* went unresolved.
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
            (_match_label(label, t.class_name), len(t.points), t.track_id)
            for t in candidates
        ]
        scored = [s for s in scored if s[0] > 0]
        if scored:
            # Best label match first, then the longest-lived of those tracks.
            best = max(scored, key=lambda s: (s[0], s[1]))
            return best[2], label

    return None, ordered[0]



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
        action_type, verb_index = _match_action(
            (obs.raw_action or "").lower(), obs.objects
        )
        object_track_id, object_label = _resolve_object_track(
            obs.objects, segment_tracks, person_classes, background_classes,
            raw_action=obs.raw_action, verb_index=verb_index,
        )

        timing_precision = _timing_precision(obs)

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
            start_sec=obs.start_time_sec if obs.start_time_sec is not None else obs.segment_start_sec,
            end_sec=obs.end_time_sec if obs.end_time_sec is not None else obs.segment_end_sec,
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
            },
            review_status="PENDING"
        )
        events.append(event)

        logger.debug("[%s] Created event %s: action=%s, confidence=%.2f, time=[%.1f, %.1f]s",
                    STAGE, event.event_id, event.action, event.confidence,
                    event.start_sec, event.end_sec)

    return events


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
