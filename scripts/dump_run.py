#!/usr/bin/env python
"""Dump every output file of a pipeline run: title, size, and JSON contents.

Purpose: after a Kaggle run, produce ONE readable block covering the whole
result set so it can be pasted back and analysed — which video, which config,
which stages ran, and what every stage actually produced.

Large files (tracks.json ~1.7 MB, trajectories.json ~1 MB) are summarised
instead of dumped whole: record count, key histograms, and the first N records.

Usage:
    python scripts/dump_run.py                          # newest run under runs/
    python scripts/dump_run.py runs/tt6_dino            # newest run_* inside
    python scripts/dump_run.py runs/tt6_dino/run_2026...  # exact run dir
    python scripts/dump_run.py <dir> --full             # dump everything, no summarising
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SEP = "=" * 78

# Files summarised rather than dumped in full (they are megabytes of points).
BIG_FILES = {"tracks.json", "trajectories.json", "detections.json",
             "sampling_plan.json"}
# Preferred reading order: pipeline order, so the dump reads like the run.
STAGE_ORDER = [
    "preview.json", "sampling_plan.json", "detections.json", "tracks.json",
    "candidate_segments.json", "vlm_observations.json", "events.json",
    "states.json", "interaction_graph.json", "trajectories.json",
    "episode.json", "episodes.json", "evaluation.json",
]
MAX_RECORDS = 3


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def find_run_dir(arg: str | None) -> Path:
    """Resolve the run directory: explicit, or newest run_* found."""
    base = Path(arg) if arg else Path("runs")
    if not base.exists():
        sys.exit(f"ERROR: {base} does not exist. Pass the run directory explicitly.")

    # Already a run dir?
    if any(base.glob("*.json")):
        return base

    candidates = sorted(base.glob("**/run_*"), key=lambda p: p.name, reverse=True)
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        sys.exit(f"ERROR: no run_* directory with JSON found under {base}")
    return candidates[0]


def summarise_records(name: str, records: list[Any]) -> None:
    """Print counts and key histograms for a large list-of-dicts payload."""
    print(f"  RECORD COUNT: {len(records)}")
    if not records or not isinstance(records[0], dict):
        return

    keys = sorted(records[0].keys())
    print(f"  KEYS: {keys}")

    # detections.json wraps per-frame results: [{frame_index, detections: [...]}, ...].
    # Flatten one level so the class histogram actually fires — that histogram is
    # what exposes label bugs (e.g. a fallback class_id silently labelling
    # everything as the first class in the prompt).
    nested_key = next((k for k in ("detections", "tracks", "trajectories")
                       if isinstance(records[0].get(k), list)), None)
    if nested_key:
        inner = [d for r in records if isinstance(r, dict)
                 for d in r.get(nested_key, []) if isinstance(d, dict)]
        print(f"  NESTED '{nested_key}': {len(inner)} total across {len(records)} frames"
              f"  ({len(inner) / max(1, len(records)):.2f} per frame)")
        if inner:
            for field in ("class_name", "class_id", "source"):
                values = [d.get(field) for d in inner if field in d]
                if values:
                    print(f"    {field}: {dict(Counter(map(str, values)).most_common(20))}")
            # Cross-tab name against id: any name mapping to a shared id is the bug.
            pairs = Counter((str(d.get("class_name")), str(d.get("class_id")))
                            for d in inner if "class_name" in d and "class_id" in d)
            if pairs:
                print("    class_name -> class_id (count):")
                for (nm, cid), cnt in pairs.most_common(20):
                    print(f"      {nm!r:<32} -> {cid:<4} {cnt}")

    # Histogram the fields that actually tell us whether the stage worked.
    for field in ("class_name", "class_id", "source", "track_id", "status",
                  "review_status", "action_type", "coordinate_space"):
        values = [r.get(field) for r in records if isinstance(r, dict) and field in r]
        if not values:
            continue
        counts = Counter(map(str, values))
        if field in ("track_id",):
            print(f"  {field}: {len(counts)} distinct")
            top = counts.most_common(8)
            print(f"    most points: {top}")
        else:
            print(f"  {field}: {dict(counts.most_common(12))}")

    # null-rate on the fields that were the known failure mode
    for field in ("object_track_id", "actor_track_id"):
        present = [r for r in records if isinstance(r, dict) and field in r]
        if present:
            nulls = sum(1 for r in present if r.get(field) is None)
            print(f"  {field}: {nulls}/{len(present)} null "
                  f"({100 * nulls / len(present):.0f}% unresolved)")

    # Nested point lists (tracks/trajectories) — report length distribution.
    for field in ("points", "track_points", "trajectory_points"):
        lengths = [len(r[field]) for r in records
                   if isinstance(r, dict) and isinstance(r.get(field), list)]
        if lengths:
            print(f"  {field} per record: total={sum(lengths)} "
                  f"min={min(lengths)} max={max(lengths)} "
                  f"mean={sum(lengths) / len(lengths):.1f}")

    print(f"  FIRST {min(MAX_RECORDS, len(records))} RECORD(S):")
    for rec in records[:MAX_RECORDS]:
        text = json.dumps(rec, indent=4, default=str)
        # Truncate any single huge record so one track can't flood the log.
        if len(text) > 2500:
            text = text[:2500] + "\n    ... (record truncated)"
        print("\n".join("    " + ln for ln in text.splitlines()))


def dump_file(path: Path, full: bool) -> None:
    print(f"\n{SEP}")
    print(f"FILE: {path.name}    ({human(path.stat().st_size)})")
    print(SEP)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not parse as JSON: {exc}")
        print("  raw head:")
        print(path.read_text(encoding="utf-8", errors="replace")[:800])
        return

    is_big = path.name in BIG_FILES and not full

    # Payload is sometimes a bare list, sometimes {"key": [...]}.
    if isinstance(data, list):
        if is_big:
            summarise_records(path.name, data)
        else:
            print(json.dumps(data, indent=2, default=str))
        return

    if isinstance(data, dict):
        list_keys = [k for k, v in data.items() if isinstance(v, list) and len(v) > 20]
        if is_big and list_keys:
            scalars = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
            if scalars:
                print("  METADATA:")
                print(json.dumps(scalars, indent=4, default=str))
            for k in list_keys:
                print(f"\n  --- {k} ---")
                summarise_records(k, data[k])
            other = {k: v for k, v in data.items()
                     if k not in list_keys and isinstance(v, (list, dict))}
            if other:
                print("\n  OTHER KEYS:")
                print(json.dumps(other, indent=4, default=str)[:4000])
        else:
            print(json.dumps(data, indent=2, default=str))
        return

    print(json.dumps(data, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Run directory (default: newest under runs/)")
    parser.add_argument("--full", action="store_true",
                        help="Dump large files in full instead of summarising")
    args = parser.parse_args()

    run_dir = find_run_dir(args.run_dir)
    json_files = sorted(run_dir.glob("*.json"))
    other_files = sorted(p for p in run_dir.iterdir()
                         if p.is_file() and p.suffix != ".json")

    print(SEP)
    print("PHYSICAL DATA COMPILER — RUN OUTPUT DUMP")
    print(SEP)
    print(f"  run directory : {run_dir.resolve()}")
    print(f"  json files    : {len(json_files)}")
    print(f"  other files   : {[p.name for p in other_files]}")

    # Which video / config was this? preview.json + episode.json carry provenance.
    for probe in ("preview.json", "episode.json", "sampling_plan.json"):
        candidate = run_dir / probe
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("video_path", "source_video", "video", "video_id",
                            "fps", "total_frames", "duration_sec", "width", "height"):
                    if key in data:
                        print(f"  {key:<14}: {data[key]}")
        except Exception:  # noqa: BLE001
            pass

    # Order known stage outputs first, then anything unexpected.
    ordered = [run_dir / n for n in STAGE_ORDER if (run_dir / n).exists()]
    ordered += [p for p in json_files if p not in ordered]

    for path in ordered:
        dump_file(path, args.full)

    # Empty outputs are the fastest signal that a stage was SKIPPED or failed.
    print(f"\n{SEP}")
    print("EMPTY / NEAR-EMPTY OUTPUTS  (stage skipped, or produced nothing)")
    print(SEP)
    for path in ordered:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        count = len(data) if isinstance(data, (list, dict)) else 1
        if count == 0:
            print(f"  EMPTY   {path.name}")
        elif isinstance(data, list) and count < 2:
            print(f"  THIN    {path.name}  ({count} record)")
    print(f"\n{SEP}\nDUMP COMPLETE\n{SEP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
