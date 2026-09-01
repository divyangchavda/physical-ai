"""Something-Something V2 class names mapped onto our ``ActionType`` vocabulary.

SSv2 has 174 classes; ``ActionType`` has 15 verbs. This module is an **allow-list**
of the 43 SSv2 classes whose own wording names one of our verbs. Everything not
listed here is excluded from evaluation — deliberately, and not scored as
``UNKNOWN``, because a class we have no verb for is a gap in the vocabulary rather
than a prediction failure, and mixing the two would make the accuracy number
unreadable.

The allow-list was written from the 174 class names alone, **before any clip was
run**, so no class is in or out because of how the pipeline scores on it.

Inclusion rule: the class name states one of our verbs, applied to an object, with
no second event bundled in. Four exclusion rules follow from it:

1. **No verb of ours.** ``Pouring``, ``Tearing``, ``Squeezing``, ``Twisting``,
   ``Spinning``, ``Folding``, ``Bending``, ``Scooping``, ``Wiping``, ``Spreading``,
   ``Sprinkling``, ``Stacking``, ``Piling``, ``Throwing``, ``Poking``, ``Dropping``,
   ``Lifting``, ``Covering``, ``Uncovering``, ``Digging``, ``Attaching``, ``Burying``,
   ``Tipping``, ``Tilting``, ``Rolling``, ``Laying``, ``Hitting``, ``Stuffing`` of a
   substance, and the camera-motion classes. ``Folding`` is the notable one: it is
   why tt7's 1.90s event stays OTHER, and adding ``FOLD`` to ``ActionType`` is an
   open decision, not something to settle inside an evaluation harness.

2. **Names two of our verbs at once.** ``Pulling something out of something`` is
   both ``PULL`` and ``REMOVE``; ``Pushing something with something`` is both
   ``PUSH`` and ``USE_TOOL``. A single truth value cannot be chosen without
   inventing a precedence rule, so these are out.

3. **Bundles a second event.** ``Putting something on the edge of something so it
   is not supported and falls down`` is a put *and* a fall. Our events are single
   actions, so the clip has no single correct answer.

4. **No actual action.** Every ``Pretending to ...``, ``Trying but failing ...``,
   ``Failing to ...``, and the physics-only classes (``Something falling like a
   rock``) — there is nothing for a manipulation verb to be right about.

Direction is not mapped here. ``tools/score_run.py`` derives direction from the
verb via its own ``ACTION_DIRECTION`` table, which covers six verbs
(INSERT/CLOSE → INTO, REMOVE/OPEN → OUT_OF, PLACE → ONTO, PICK → OFF); for the
other nine, direction is ``N/A`` on both sides and nothing is claimed.

Key format: SSv2 ships the same string two ways — ``labels.json`` writes
``"Putting something into something"`` while each clip's ``template`` field writes
``"Putting [something] into [something]"``. The keys here are the bracket-free
form, and ``normalize_template`` strips brackets so either reaches the same entry.
"""
from __future__ import annotations

import re

from src.schema.event import ActionType

_BRACKETS = re.compile(r"[\[\]]")


def normalize_template(template: str) -> str:
    """The bracket-free, whitespace-collapsed form of an SSv2 class name.

    ``"Putting [something] into [something]"`` and
    ``"Putting something into something"`` are the same class written two ways in
    the dataset's own files, so both must reach the same key.
    """
    return " ".join(_BRACKETS.sub("", template).split())


# The allow-list. 43 of 174 classes, grouped by the verb they name.
SSV2_TEMPLATE_TO_ACTION: dict[str, ActionType] = {
    # ── GRASP: the class names holding, and the relation is only where it is held
    "Holding something": ActionType.GRASP,
    "Holding something behind something": ActionType.GRASP,
    "Holding something in front of something": ActionType.GRASP,
    "Holding something next to something": ActionType.GRASP,
    "Holding something over something": ActionType.GRASP,
    # ── PICK
    "Picking something up": ActionType.PICK,
    # ── PLACE: put onto a surface or into a spatial relation, never a container
    "Putting something on a surface": ActionType.PLACE,
    "Putting something onto something": ActionType.PLACE,
    "Putting something on a flat surface without letting it roll": ActionType.PLACE,
    "Putting something upright on the table": ActionType.PLACE,
    "Putting something and something on the table": ActionType.PLACE,
    "Putting something, something and something on the table": ActionType.PLACE,
    "Putting something behind something": ActionType.PLACE,
    "Putting something in front of something": ActionType.PLACE,
    "Putting something next to something": ActionType.PLACE,
    "Putting something underneath something": ActionType.PLACE,
    "Putting number of something onto something": ActionType.PLACE,
    "Putting something similar to other things that are already on the table":
        ActionType.PLACE,
    # ── MOVE: translation with no contact verb named and no second event
    "Moving something up": ActionType.MOVE,
    "Moving something down": ActionType.MOVE,
    "Moving something away from something": ActionType.MOVE,
    "Moving something closer to something": ActionType.MOVE,
    "Moving something across a surface without it falling down": ActionType.MOVE,
    "Moving something and something away from each other": ActionType.MOVE,
    "Moving something and something closer to each other": ActionType.MOVE,
    # ── PUSH
    "Pushing something from left to right": ActionType.PUSH,
    "Pushing something from right to left": ActionType.PUSH,
    "Pushing something so that it slightly moves": ActionType.PUSH,
    "Pushing something so it spins": ActionType.PUSH,
    "Pushing something off of something": ActionType.PUSH,
    "Pushing something onto something": ActionType.PUSH,
    # ── PULL
    "Pulling something from left to right": ActionType.PULL,
    "Pulling something from right to left": ActionType.PULL,
    "Pulling something from behind of something": ActionType.PULL,
    "Pulling something onto something": ActionType.PULL,
    # ── OPEN / CLOSE: the inversion this project has reproduced four times
    "Opening something": ActionType.OPEN,
    "Closing something": ActionType.CLOSE,
    # ── INSERT: into a container, which is what INTO means to the scorer
    "Putting something into something": ActionType.INSERT,
    "Stuffing something into something": ActionType.INSERT,
    "Plugging something into something": ActionType.INSERT,
    # ── REMOVE
    "Taking something out of something": ActionType.REMOVE,
    "Removing something, revealing something behind": ActionType.REMOVE,
    # ── TOUCH: contact explicitly without motion, which is the whole definition
    "Touching (without moving) part of something": ActionType.TOUCH,
}

# Keyed on the normalized form so a clip's bracketed ``template`` resolves too.
_BY_NORMALIZED: dict[str, ActionType] = {
    normalize_template(name): action
    for name, action in SSV2_TEMPLATE_TO_ACTION.items()
}


def map_template(template: str) -> ActionType | None:
    """The ``ActionType`` an SSv2 class name states, or ``None`` if it is excluded.

    ``None`` means "not in the allow-list", which is not the same as ``UNKNOWN``:
    an excluded class is not scored at all. See this module's docstring for why.
    """
    return _BY_NORMALIZED.get(normalize_template(template))
