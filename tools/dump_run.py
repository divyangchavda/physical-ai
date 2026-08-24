"""Print the semantic content of a pipeline run directory.

The pipeline log says how many events came out; it does not say whether the
actor is the person or the trolley. This prints the fields that decide whether
a run is trustworthy — resolved track ids, identity resolution status, timing
precision — resolved against tracks.json so ids read as class names.

Kaggle's /kaggle/working is wiped on session restart, so run this in the same
cell as the pipeline rather than as a follow-up.

Usage:
    python tools/dump_run.py <run_dir>
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def _load(run_dir: Path, name: str):
    """Return parsed JSON, or None when the stage never wrote the file."""
    path = run_dir / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"  !! {name} is not valid JSON: {exc}")
        return None


def _rows(data) -> list:
    """Stage outputs are sometimes a bare list, sometimes wrapped in a dict."""
    if data is None:
        return []
    if isinstance(data, list):
        return data
    for key in ("scores", "nodes", "events", "items"):
        if isinstance(data.get(key), list):
            return data[key]
    return []


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    run_dir = Path(argv[1])
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}")
        return 1

    print(f"=== {run_dir}")

    tracks = _rows(_load(run_dir, "tracks.json"))
    cls = {t["track_id"]: t["class_name"] for t in tracks}
    counts = collections.Counter(cls.values())
    print(f"\nTRACKS {len(tracks)} {dict(counts)}")
    # Fragmentation is the ratio that matters: ids per real-world object.
    for name, n in counts.most_common():
        ids = sorted(tid for tid, c in cls.items() if c == name)
        print(f"  {name}: {n} ids {ids[:20]}{' ...' if len(ids) > 20 else ''}")

    segments = _rows(_load(run_dir, "candidate_segments.json"))
    print(f"\nSEGMENTS {len(segments)}")
    for s in segments:
        print(f"  {s['segment_id']} [{s['start_sec']:.1f},{s['end_sec']:.1f}]s "
              f"tracks={s['track_ids']} trigger={s.get('trigger_reason')!r}")

    obs = _rows(_load(run_dir, "vlm_observations.json"))
    print(f"\nVLM {len(obs)}")
    for o in obs:
        print(f"  {o['segment_id']} {o['status']} actor={o.get('actor')!r} "
              f"objects={o.get('objects')} action={o.get('raw_action')!r}")
        print(f"      seg=[{o.get('segment_start_sec')},{o.get('segment_end_sec')}] "
              f"reported=[{o.get('start_time_sec')},{o.get('end_time_sec')}]")

    events = _rows(_load(run_dir, "events.json"))
    print(f"\nEVENTS {len(events)}")
    for e in events:
        a, o = e.get("actor_track_id"), e.get("object_track_id")
        attrs = e.get("attributes") or {}
        print(f"  {e.get('action'):8} actor={a}({cls.get(a)}) object={o}({cls.get(o)}) "
              f"label={attrs.get('object_label')!r} tp={attrs.get('timing_precision')} "
              f"t=[{e.get('start_sec')},{e.get('end_sec')}] conf={e.get('confidence')}")

    states = _rows(_load(run_dir, "states.json"))
    print(f"\nSTATES {len(states)}")
    for s in states:
        tid = s.get("track_id")
        print(f"  track={tid}({cls.get(tid)}) label={s.get('semantic_label')!r} "
              f"res={s.get('identity_resolution')} "
              f"{s.get('from_state')}->{s.get('to_state')} tp={s.get('timing_precision')}")

    graph = _load(run_dir, "interaction_graph.json") or {}
    nodes, edges = _rows(graph.get("nodes")), _rows(graph.get("edges"))
    print(f"\nGRAPH {len(nodes)} nodes / {len(edges)} edges")
    for n in nodes:
        tid = n.get("track_id")
        print(f"  {n.get('node_id')} role={n.get('role')} track={tid}({cls.get(tid)}) "
              f"label={n.get('label') or n.get('semantic_label')!r}")
    for e in edges:
        print(f"  {e.get('source')} -{e.get('relation') or e.get('action')}-> "
              f"{e.get('target')} res={e.get('actor_resolution')}/"
              f"{e.get('object_resolution')}")

    scores = _rows(_load(run_dir, "quality_scores.json"))
    print(f"\nSCORES {len(scores)}")
    for q in scores:
        keys = ("composite_score", "tier", "actor_resolution", "object_resolution",
                "trajectory_support", "timing_precision")
        print(f"  {q.get('event_id')} " + " ".join(
            f"{k}={q.get(k)}" for k in keys if k in q))

    ev = _load(run_dir, "evaluation.json") or {}
    print(f"\nEVAL overall={ev.get('overall_status')} health={ev.get('health')} "
          f"issues={len(_rows(ev.get('issues')))} warnings={len(_rows(ev.get('warnings')))}")
    for issue in _rows(ev.get("issues")) + _rows(ev.get("warnings")):
        print(f"  {issue}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
