"""Re-score s07's verb mapping over a finished SSv2 run, with no GPU and no API.

Why this exists. The 200-clip run in ``tools/ssv2_eval.py`` costs 113 minutes of
GPU and 200 Gemini calls. Most of what we then change is ``_ACTION_STEMS`` and the
preposition promotions in ``s07_events`` — pure functions of the caption text the
run already recorded. This replays the mapping over those recorded captions, so a
verb change is measured against 200 human labels in about a second.

What it measures: which ``ActionType`` the current code assigns to each recorded
``raw_action``, against the SSv2 label. Both the gain and, more importantly, the
**regressions** — a stem that fixes eight clips and breaks nine is a loss, and a
per-verb total alone will not show it.

What it does NOT measure, and must not be reported as if it did:
  * **the pipeline.** Only ``_map_raw_action_to_type`` runs. Detection, tracking,
    segmentation, attribution, event confidence and timing are all untouched, so
    a gain here is a gain in the mapping and nothing more.
  * **the objects.** results.json records the caption but not the object list, so
    the mapping runs with ``objects=None`` and ``_strip_object_phrases`` does not
    blank anything. That makes this strictly harsher than the live pipeline: a
    verb hidden in an object's name is visible here and was hidden there. Any
    regression it reports is therefore worth reading; a gain may be understated.
  * **the 8 clips with no event at all**, which have no caption to replay.

Reproduction is checked rather than assumed: ``--verify`` reports how many clips
the offline mapping reproduces the recorded verb for, and the divergences are
excluded from the headline. On the first run of the 200-clip corpus that was
187/192, the five exceptions all being clips where the live object list blanked a
word that is visible here.

Usage:

    python tools/s07_remap_score.py --results path/to/results.json
    python tools/s07_remap_score.py --results path/to/results.json --verify
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.stages.s07_events import _action_clauses, _map_raw_action_to_type  # noqa: E402


def record_objects(record: dict) -> list[str] | None:
    """The object names the VLM listed alongside this clip's winning caption.

    Runs before the ``captions`` field existed do not have this, and for those
    the mapping runs with ``objects=None`` — strictly harsher than the live
    pipeline, since ``_strip_object_phrases`` then blanks nothing and an object
    whose name contains a verb ("red folder", "push chopper") can outrank the
    real verb. Newer runs record it, so those replay exactly.
    """
    for entry in record.get("captions") or []:
        if entry.get("raw_action") == record.get("raw_action"):
            objects = entry.get("objects")
            return objects if objects else None
    return None


def remap(record: dict) -> str | None:
    """The verb the current code assigns to this clip's recorded caption."""
    raw = record.get("raw_action")
    if not raw:
        return None
    return _map_raw_action_to_type(raw, record_objects(record)).value


def remap_all(record: dict) -> list[str]:
    """Every verb the current code would emit for this caption.

    s07 emits one event per independent clause, so "the person touches and then
    picks up the blush compact" is a TOUCH *and* a PICK. Which of the two the
    evaluation calls the answer depends on event confidence, which is computed
    from track and segment data this tool does not have.
    """
    raw = record.get("raw_action")
    if not raw:
        return []
    return [a.value for a, _ in _action_clauses(raw.lower(), record_objects(record))]


def is_multi_verb(record: dict) -> bool:
    """True when the caption yields more than one event and top-1 is unresolvable.

    These clips are excluded from the top-1 delta rather than guessed at: the
    live run picked among them by confidence, and reproducing that offline would
    require the run directories that the Kaggle kernel restart destroyed.
    """
    return len(remap_all(record)) > 1


def load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not results:
        raise SystemExit(f"no results in {path}")
    return [r for r in results if r.get("raw_action")]


def verify(records: list[dict]) -> list[dict]:
    """Print how far the offline mapping agrees with what the pipeline recorded.

    Run this on unmodified code to establish the baseline. On changed code every
    clip the change touched shows up here too, which is expected and is why the
    delta in :func:`report` is measured against the run's recorded verb rather
    than against this.
    """
    diverged = [r for r in records if remap(r) != r.get("got")]
    agree = len(records) - len(diverged)
    print(f"offline mapping reproduces the recorded verb: "
          f"{agree}/{len(records)}")
    for r in diverged:
        print(f"  recorded {str(r.get('got')):<8} offline "
              f"{str(remap(r)):<8} | {r['raw_action'][:64]}")
    return diverged


def report(records: list[dict], baseline_key: str = "got") -> None:
    """Per-verb table of recorded vs re-mapped, then the gains and regressions."""
    by_verb: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_verb[r["truth_action"]].append(r)

    print("\n" + "=" * 62)
    print("s07 VERB MAPPING, RECORDED RUN vs CURRENT CODE")
    print("=" * 62)
    print(f"{'truth':<9}{'n':>4}{'was':>6}{'now':>6}{'delta':>7}")
    print("-" * 62)
    was_total = now_total = 0
    for verb in sorted(by_verb):
        rows = by_verb[verb]
        was = sum(r.get(baseline_key) == verb for r in rows)
        now = sum(remap(r) == verb for r in rows)
        was_total += was
        now_total += now
        flag = "" if now == was else ("  +" if now > was else "  -")
        print(f"{verb:<9}{len(rows):>4}{was:>6}{now:>6}{now - was:>+7}{flag}")
    print("-" * 62)
    n = len(records)
    print(f"{'TOTAL':<9}{n:>4}{was_total:>6}{now_total:>6}"
          f"{now_total - was_total:>+7}")
    print(f"\nmapping accuracy on clips with a caption: "
          f"{was_total}/{n} = {100.0 * was_total / n:.1f}%  ->  "
          f"{now_total}/{n} = {100.0 * now_total / n:.1f}%")

    fixed = [r for r in records
             if r.get(baseline_key) != r["truth_action"] == remap(r)]
    broken = [r for r in records
              if remap(r) != r["truth_action"] == r.get(baseline_key)]
    changed = [r for r in records
               if remap(r) != r.get(baseline_key) and r not in fixed
               and r not in broken]

    print(f"\n--- FIXED: {len(fixed)} ---")
    for r in fixed:
        print(f"  {r['truth_action']:<7} was {str(r.get(baseline_key)):<8} "
              f"| {r['raw_action'][:58]}")

    print(f"\n--- BROKEN: {len(broken)} ---")
    for r in broken:
        print(f"  {r['truth_action']:<7} now {str(remap(r)):<8} "
              f"| {r['raw_action'][:58]}")
    if not broken:
        print("  none")

    print(f"\n--- changed but wrong either way: {len(changed)} ---")
    for r in changed:
        print(f"  {r['truth_action']:<7} {str(r.get(baseline_key)):<8} -> "
              f"{str(remap(r)):<8} | {r['raw_action'][:44]}")

    still = Counter(
        (r["truth_action"], remap(r)) for r in records
        if remap(r) != r["truth_action"]
    )
    print("\n--- still wrong, most common pairs ---")
    for (want, got), count in still.most_common(10):
        print(f"  truth {want:<8} -> {str(got):<8} x{count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path,
                        help="results.json written by tools/ssv2_eval.py")
    parser.add_argument("--verify", action="store_true",
                        help="report how far the offline mapping reproduces the "
                             "verbs the pipeline actually recorded")
    args = parser.parse_args()

    records = load(args.results)
    print(f"results : {args.results}")
    print(f"clips with a caption to replay: {len(records)}")

    multi = [r for r in records if is_multi_verb(r)]
    if multi:
        print(f"excluded: {len(multi)} caption(s) describing more than one "
              f"action, where top-1 depends on event confidence this tool "
              f"cannot recompute")
        for r in multi:
            print(f"  {r['truth_action']:<7} recorded {str(r.get('got')):<8} "
                  f"-> {remap_all(r)} | {r['raw_action'][:44]}")
        records = [r for r in records if not is_multi_verb(r)]

    if args.verify:
        verify(records)

    report(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
