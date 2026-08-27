"""Score a pipeline run against a hand-written ground-truth label file.

Read-only. It opens a run directory and a label file and prints numbers; it never
writes to the run and never touches the pipeline, so it cannot change the output
it is measuring.

    python tools/score_run.py --run <run_dir> --truth tests/fixtures/tt6_ground_truth.json

Two deliberate choices about how the numbers are computed:

*Antonyms are their own category.* On tt6 the VLM described the same 8.3s clip
four times and picked the exact time-reverse verb on two of them (OPEN for CLOSE,
REMOVE for INSERT). Plain string accuracy would score those the same as an
unrelated wrong guess, hiding the single most important fact about the output.

*Attribution is reported, not guessed.* Events currently span a whole segment, so
one event overlaps every ground-truth action in its copy. Picking the
largest-overlap action would silently mean "whichever labelled action is
longest", and picking the best-matching action would flatter the score. So an
event covering more than one action is marked AMBIGUOUS, and its verdict is an
explicit **upper bound**: EXACT means the emitted action matched *at least one*
of the actions it covers. Treat every accuracy figure here as a ceiling.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Time-reverse pairs. An emitted action landing on the truth's antonym is a
# direction failure, not a random miss.
ANTONYMS: dict[str, str] = {
    "INSERT": "REMOVE", "REMOVE": "INSERT",
    "OPEN": "CLOSE", "CLOSE": "OPEN",
    "PICK": "PLACE", "PLACE": "PICK",
    "PUSH": "PULL", "PULL": "PUSH",
    "GRASP": "RELEASE", "RELEASE": "GRASP",
}

# The controlled vocabulary in src/schema/event.py, as a literal so the scorer
# runs without importing the pipeline.
VOCABULARY = {
    "GRASP", "RELEASE", "PICK", "PLACE", "MOVE", "PUSH", "PULL", "OPEN",
    "CLOSE", "INSERT", "REMOVE", "USE_TOOL", "TOUCH", "INSPECT", "UNKNOWN",
}

# Direction implied by each action, for scoring against the label file's own
# 'direction' field. Written out rather than inferred so the mapping is arguable
# in the open.
#
# Note what this means for the pipeline: direction is a pure function of the verb.
# There is nothing for geometry to contribute here — get the verb right and the
# direction follows. Geometry's job is TIMING.
#
# The vocabularies on the two sides differ and that was checked rather than
# assumed: both tt6_ground_truth.json and tt7_ground_truth.json use only
# ONTO / OFF / INTO / NONE, so the OUT_OF below can only ever be compared against
# INTO — where OPPOSITE_DIRECTION resolves it to REVERSED, which is correct — or
# against ONTO/OFF, where OTHER is also correct: OUT_OF is de-containment and OFF
# is de-support, and the label files distinguish INTO from ONTO precisely so that
# INSERT can be distinguished from PLACE. No change needed.
ACTION_DIRECTION: dict[str, str] = {
    "INSERT": "INTO", "CLOSE": "INTO",
    "REMOVE": "OUT_OF", "OPEN": "OUT_OF",
    "PLACE": "ONTO", "PICK": "OFF",
}
OPPOSITE_DIRECTION = {
    "INTO": "OUT_OF", "OUT_OF": "INTO", "ONTO": "OFF", "OFF": "ONTO",
}


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_truth(truth: dict) -> list[dict]:
    """Repeat the source-clip labels once per copy, offset in time.

    tt6 is one clip copied four times, so the labels describe the clip once and
    the whole-video truth is that list shifted by copy * clip_duration. A video
    that is not a loop sets repeats=1 and nothing here changes.
    """
    repeats = int(truth.get("repeats", 1))
    clip = float(truth.get("source_clip_duration_sec", 0.0))
    out = []
    for copy in range(repeats):
        for a in truth["actions"]:
            if not a.get("action"):
                continue  # unfilled template row
            out.append({
                **a,
                "copy": copy,
                "start_sec": float(a["start_sec"]) + copy * clip,
                "end_sec": float(a["end_sec"]) + copy * clip,
            })
    return out


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _labels_match(vlm_label: str | None, truth_object: str) -> bool:
    """Bidirectional containment, the same rule s07 uses to bind a track."""
    if not vlm_label or not truth_object:
        return False
    a, b = vlm_label.strip().lower(), truth_object.strip().lower()
    return a == b or a in b or b in a


def _names_either_role(vlm_label: str | None, action: dict) -> bool:
    """True when the event's object names either role of a labelled action.

    Testing only ``object`` and never ``target`` threw away correct answers. tt7's
    INSERT is ``object: push chopper, target: cardboard box``; an event that
    resolved "cardboard box" was therefore excluded from the INSERT while the six
    other cardboard-box actions matched, so the pool it was scored against could
    not contain the very action it got right, and an exactly correct INSERT was
    reported OTHER.

    That is not hypothetical for this pipeline. s07's _order_objects_by_action
    exists because the VLM returns the container first — for "placing the push
    chopper back into the box" it returned ``["box", "push chopper"]`` — so the
    destination reaching object_label is the specific failure s07 already fights.
    The scorer must not turn a near miss there into a wrong verb here.

    This widens the pool rather than narrowing it, which does make the upper bound
    looser. That is visible: the per-event line prints AMBIGUOUS xN, and N grows.
    A looser bound that can be read off the output is preferable to a tighter one
    that is wrong in a direction the output does not show.
    """
    return (
        _labels_match(vlm_label, action.get("object", ""))
        or _labels_match(vlm_label, action.get("target", ""))
    )


def assess(event: dict, actions: list[dict], tolerance: float) -> dict:
    """Describe what an event covers and how well it named it.

    Never picks one action out of several. When the event's span covers many
    labelled actions, that is the finding, and the verdict is a best case over
    everything covered.
    """
    attrs = event.get("attributes") or {}
    label = attrs.get("object_label")
    got = event.get("action", "UNKNOWN")
    e1 = float(event.get("start_sec") or 0.0)
    e2 = float(event.get("end_sec") or 0.0)

    covered = [
        a for a in actions
        if _overlap(e1 - tolerance, e2 + tolerance, a["start_sec"], a["end_sec"]) > 0
    ]
    # Narrow by object when the event resolved one and it matches something.
    # Either role counts — see _names_either_role for why testing only 'object'
    # dropped correct answers.
    on_object = [a for a in covered if _names_either_role(label, a)]
    pool = on_object or covered

    if not pool:
        return {"verdict": "UNMATCHED", "direction": "N/A", "covered": [],
                "narrowed_by_object": False, "got": got}

    truths = [a["action"] for a in pool]
    if got in truths:
        verdict = "EXACT"
    elif any(ANTONYMS.get(t) == got for t in truths):
        verdict = "REVERSED"
    elif all(t not in VOCABULARY for t in truths):
        verdict = "NOT_IN_VOCABULARY"
    else:
        verdict = "OTHER"

    # Direction: compare against the labelled direction of whichever covered
    # action shares the emitted verb, else the first covered action that states
    # a direction at all.
    want = next(
        (a["direction"] for a in pool if a["action"] == got and a.get("direction")),
        next((a["direction"] for a in pool
              if (a.get("direction") or "").upper() not in ("", "NONE")), ""),
    )
    mine = ACTION_DIRECTION.get(got)
    want = (want or "").strip().upper()
    if not mine or want in ("", "NONE"):
        direction = "N/A"
    elif mine == want:
        direction = "SAME"
    elif OPPOSITE_DIRECTION.get(want) == mine:
        direction = "REVERSED"
    else:
        direction = "OTHER"

    return {"verdict": verdict, "direction": direction, "covered": pool,
            "narrowed_by_object": bool(on_object), "got": got}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing events.json")
    ap.add_argument("--truth", required=True, help="hand-written label file")
    args = ap.parse_args()

    run = Path(args.run)
    truth = _load(Path(args.truth))
    events = _load(run / "events.json")
    tolerance = float(truth.get("timing_tolerance_sec", 1.0))
    actions = expand_truth(truth)
    n_labelled = len([a for a in truth["actions"] if a.get("action")])

    print(f"=== {run.name} vs {Path(args.truth).name} ===")
    print(f"ground-truth actions : {len(actions)}"
          f"  ({n_labelled} labelled x {truth.get('repeats', 1)} copies)")
    print(f"events emitted       : {len(events)}")
    if actions:
        print(f"recall ceiling       : {len(events)}/{len(actions)} = "
              f"{100.0 * len(events) / len(actions):.0f}%"
              f"   (one event can name at most one action)")
    print("NOTE: every accuracy below is an UPPER BOUND -- an event spanning a "
          "whole\n      segment is credited if it matched ANY action it covers.")

    verdicts: dict[str, int] = {}
    directions: dict[str, int] = {}
    covered_keys: set[tuple[int, int]] = set()
    n_ambiguous = 0

    print("\n--- events ---")
    for ev in events:
        r = assess(ev, actions, tolerance)
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        directions[r["direction"]] = directions.get(r["direction"], 0) + 1
        span = f"[{ev.get('start_sec')}, {ev.get('end_sec')}]"
        if not r["covered"]:
            print(f"  {r['got']:<9} {span:<18} -> nothing labelled overlaps this span")
            continue
        for a in r["covered"]:
            covered_keys.add((a["copy"], a["order"]))
        attribution = "PRECISE" if len(r["covered"]) == 1 else f"AMBIGUOUS x{len(r['covered'])}"
        names = ",".join(a["action"] for a in r["covered"])
        print(f"  {r['got']:<9} {span:<18} -> {r['verdict']:<18} "
              f"direction={r['direction']:<9} {attribution:<14} covers[{names}]")
        if len(r["covered"]) > 1:
            n_ambiguous += 1

    print("\n--- action verdicts (upper bound) ---")
    for k in ("EXACT", "REVERSED", "OTHER", "NOT_IN_VOCABULARY", "UNMATCHED"):
        if k in verdicts:
            print(f"  {k:<18} {verdicts[k]}")
    print("--- direction verdicts (where both sides state one) ---")
    for k in ("SAME", "REVERSED", "OTHER", "N/A"):
        if k in directions:
            print(f"  {k:<18} {directions[k]}")
    print("--- attribution ---")
    print(f"  events covering >1 labelled action : {n_ambiguous}/{len(events)}")
    print(f"  labelled actions inside some event span : {len(covered_keys)}/{len(actions)}")
    never: dict[str, int] = {}
    for a in actions:
        if (a["copy"], a["order"]) not in covered_keys:
            never[a["action"]] = never.get(a["action"], 0) + 1
    if never:
        print(f"  never inside any event span            : {dict(sorted(never.items()))}")

    merges = run / "track_merges.json"
    if merges.is_file():
        m = _load(merges)
        print("\n--- track stitching ---")
        print(f"  fragments {m['tracks_before']} -> entities {m['tracks_after']}")
        print(f"  by class before : {m['by_class_before']}")
        print(f"  by class after  : {m['by_class_after']}")
        # Print where each threshold came from. A stitcher that under-merges
        # because a threshold was guessed looks identical to one that under-merges
        # because the video is hard, unless the number in use is on screen.
        if "max_overlap_frames_source" in m:
            print(f"  overlap budget  : {m['max_overlap_frames']} frames "
                  f"({m['max_overlap_frames_source']})")
        if "duplicate_min_iou_source" in m:
            print(f"  duplicate IoU   : {m['duplicate_min_iou']} "
                  f"({m['duplicate_min_iou_source']})")
        reps = int(truth.get("repeats", 1))
        if reps > 1:
            # tt6's copies are hard cuts; an entity SHOULD break there and
            # nowhere else. Printing where the breaks landed IS the test.
            print(f"  expected ~{reps} entities per class, breaking on copy boundaries")
        for e in m["entities"]:
            print(f"    entity {e['entity_id']:<4} {e['class_name']:<15} "
                  f"frames {e['start_frame']}-{e['end_frame']} "
                  f"pts={e['n_points']:<4} absorbed={e['absorbed_track_ids']}")

    graph = run / "interaction_graph.json"
    if graph.is_file():
        g = _load(graph)
        nodes = g.get("nodes", g if isinstance(g, list) else [])
        by_role: dict[str, int] = {}
        for n in nodes:
            by_role[n.get("role", "?")] = by_role.get(n.get("role", "?"), 0) + 1
        print("\n--- graph ---")
        print(f"  nodes by role           : {dict(sorted(by_role.items()))}")
        print(f"  people visible in truth : {truth.get('people_visible')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
