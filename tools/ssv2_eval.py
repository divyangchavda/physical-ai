"""Score the pipeline's verbs against SSv2 ground truth, over many clips.

The measurement this project has never had. Everything to date is one 6.7s video
with seven labels I wrote, where a uniform grid of times scores 8/8 against our
geometry's 7/8 — a ruler that cannot tell an improvement from luck. Here each clip
carries one human-verified action label covering the whole clip, so a verb is
right or it is not, and 200 clips give a per-verb accuracy instead of an anecdote.

What is measured: the verb, and direction where SSv2's own class name states a
preposition.

What is NOT measured, and must not be reported as if it were:
  * **object detection** — ``detector.text_prompt`` is built from each clip's own
    ``placeholders``, so the object names are handed to the detector. This is
    intentional (it isolates the verb) and it means the numbers here say nothing
    about finding objects in an unseen video.
  * **timing** — SSv2 has no timestamps.
  * **SSv2 as a whole** — 43 of its 174 classes name one of our 15 verbs; see
    ``src/eval/ssv2_action_map.py``. A figure here is accuracy on the part of SSv2
    our vocabulary can be judged on, not on SSv2.

Each clip runs as a separate subprocess. That costs a model reload per clip (s03
builds the detector per run, with no cache), which is accepted rather than fixed:
adding a cache to a production stage to speed up an evaluation would put an
untested change in the path every real run takes. A crash or CUDA OOM on one clip
therefore also cannot take the other 199 with it.

Two verdicts are reported per clip and both matter:
  * **top-1** — the highest-confidence event's verb equals the label's verb. This
    is the honest headline, because a consumer of the data gets one answer.
  * **any** — some emitted event has the right verb. The gap between them is a
    ranking problem, not a vocabulary problem, and it is worth seeing separately.

Usage:

    python tools/ssv2_eval.py --bundle /kaggle/input/ssv2-eval-200 \\
        --out /kaggle/working/runs/ssv2 --limit 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from score_run import ACTION_DIRECTION, OPPOSITE_DIRECTION, stub_observations  # noqa: E402

# Classes always offered to the detector, on top of the clip's own placeholders.
# "person" and "hand" are actor classes for s05 (SSv2 is shot close-in, so many
# clips show only hands); "table" is the surface that PLACE labels leave unnamed.
# Kept here rather than in the YAML because the per-clip prompt is built here.
BASE_CLASSES = ("person", "hand", "table")

# Prepositions SSv2 class names use, mapped to the scorer's own direction
# vocabulary. Read off the class name, NOT off our verb map, so direction is an
# independent check rather than a restatement of the verb.
_TEMPLATE_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("out of", "OUT_OF"),
    ("off of", "OFF"),
    ("into", "INTO"),
    ("onto", "ONTO"),
)


def build_prompt(placeholders: list[str]) -> str:
    """GroundingDINO's dot-separated vocabulary for one clip.

    Placeholders are the label's own object names ("gate", "a box of eggs"), so
    the detector is told what to look for. Duplicates and empties are dropped, and
    a placeholder that repeats a base class is not repeated.
    """
    seen: list[str] = []
    for phrase in [*BASE_CLASSES, *placeholders]:
        cleaned = " ".join(str(phrase).lower().split())
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return " . ".join(seen) + " ."


def template_direction(template: str) -> str | None:
    """The direction stated by an SSv2 class name, or ``None`` if it states none.

    "Taking [something] out of [something]" states OUT_OF. "Opening [something]"
    states nothing — our own table calls OPEN an OUT_OF, but that is our
    convention and asserting it here would be scoring ourselves against ourselves.
    So such clips are counted as "no direction stated" and excluded from the
    direction figure.
    """
    text = template.replace("[", "").replace("]", "").lower()
    for phrase, direction in _TEMPLATE_DIRECTIONS:
        if f" {phrase} " in f" {text} ":
            return direction
    return None


def primary_event(events: list[dict]) -> dict | None:
    """The one answer a consumer would take: highest confidence, then earliest.

    Ties are broken by start time so the choice is deterministic and does not
    depend on the order s07 happened to append in.
    """
    if not events:
        return None
    return min(
        events,
        key=lambda e: (-float(e.get("confidence") or 0.0),
                       float(e.get("start_sec") or 0.0)),
    )


def score_direction(got_verb: str, template: str) -> str:
    """SAME / REVERSED / OTHER / N/A for one clip, using score_run's table."""
    want = template_direction(template)
    mine = ACTION_DIRECTION.get(got_verb)
    if want is None or mine is None:
        return "N/A"
    if mine == want:
        return "SAME"
    if OPPOSITE_DIRECTION.get(want) == mine:
        return "REVERSED"
    return "OTHER"


def interleave_by_verb(clips: list[dict]) -> list[dict]:
    """Reorder so every prefix is as verb-balanced as the whole list is.

    ``tools/ssv2_bundle.py`` writes truth.json grouped by verb, which made the
    first five-clip batch five CLOSE clips and told us nothing about the other
    ten verbs. Order is a property of the run, not of the bundle, so it is fixed
    here rather than by re-uploading the dataset.

    Round-robin over the verbs, each verb keeping its own file order, so the
    result is deterministic and interrupting the run after any number of clips
    still leaves a readable per-verb table.
    """
    by_verb: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        by_verb[clip["action"]].append(clip)
    verbs = sorted(by_verb)
    out: list[dict] = []
    for row in range(max(len(v) for v in by_verb.values()) if by_verb else 0):
        for verb in verbs:
            if row < len(by_verb[verb]):
                out.append(by_verb[verb][row])
    return out


def run_one(clip: dict, bundle: Path, config: Path, out: Path,
            python: str) -> dict:
    """Run the pipeline on one clip and read its verdict off events.json."""
    clip_id = clip["clip_id"]
    video = bundle / "clips" / f"{clip_id}.webm"
    run_dir = out / "runs" / clip_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(clip["placeholders"])

    cmd = [
        python, "-m", "src.pipeline", str(video),
        "--config", str(config),
        "--output-dir", str(run_dir),
        "--set", f"detector.text_prompt={prompt}",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        cmd, cwd=str(REPO), capture_output=True, text=True, errors="replace"
    )
    elapsed = time.monotonic() - started
    (run_dir / "stdout.txt").write_text(
        completed.stdout + "\n--- stderr ---\n" + completed.stderr,
        encoding="utf-8",
    )

    record = {
        "clip_id": clip_id,
        "truth_action": clip["action"],
        "template": clip["template"],
        "label": clip["label"],
        "prompt": prompt,
        "exit_code": completed.returncode,
        "seconds": round(elapsed, 1),
    }

    events_path = run_dir / "events.json"
    if not events_path.exists():
        record.update(verdict="NO_OUTPUT", got=None, n_events=0, direction="N/A")
        return record

    events = json.loads(events_path.read_text(encoding="utf-8"))
    # The lesson from the run that scored EXACT on fabricated observations: a
    # stub answer must abort, not be annotated.
    if stub_observations(run_dir):
        raise SystemExit(
            f"clip {clip_id}: the VLM fell through to the stub adapter. Nothing "
            f"below would describe the pipeline. Check {run_dir / 'config.json'} "
            f"and that GEMINI_API_KEY is in the environment."
        )

    verbs = [str(e.get("action")) for e in events]
    chosen = primary_event(events)
    got = str(chosen.get("action")) if chosen else None
    if got is None:
        verdict = "NO_EVENTS"
    elif got == clip["action"]:
        verdict = "TOP1"
    elif clip["action"] in verbs:
        verdict = "ANY_ONLY"
    else:
        verdict = "MISS"

    record.update(
        verdict=verdict,
        got=got,
        all_verbs=verbs,
        n_events=len(events),
        direction=score_direction(got, clip["template"]) if got else "N/A",
        raw_action=(chosen.get("attributes") or {}).get("raw_action") if chosen else None,
        verb_source=(chosen.get("attributes") or {}).get("verb_source") if chosen else None,
    )
    return record


def merge_results(path: Path, records: list[dict]) -> list[dict]:
    """This invocation's records folded into whatever a prior one already wrote.

    Without this, resuming with ``--skip-done`` would report only the clips run
    since the interruption and overwrite the rest — a 195-clip run that died at
    clip 180 would come back as a 15-clip measurement. Keyed on ``clip_id``, and
    a fresh record replaces an older one for the same clip so a re-run of a
    single clip is how you correct it.

    A results.json that cannot be parsed is ignored rather than fatal: the run
    that just cost an hour of GPU must not be lost to a truncated file from a
    previous crash.
    """
    existing: dict[str, dict] = {}
    if path.is_file():
        try:
            prior = json.loads(path.read_text(encoding="utf-8")).get("results", [])
            existing = {r["clip_id"]: r for r in prior}
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
            print(f"warning: ignoring unreadable {path}: {exc}")
    for record in records:
        existing[record["clip_id"]] = record
    return list(existing.values())


def report(records: list[dict]) -> None:
    """Print the per-verb table, the confusion pairs, and the two headlines."""
    by_verb: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_verb[r["truth_action"]].append(r)

    print("\n" + "=" * 66)
    print("SSv2 VERB ACCURACY")
    print("=" * 66)
    print(f"{'truth':<9}{'n':>4}{'top1':>7}{'any':>7}{'miss':>6}"
          f"{'noev':>6}   most common wrong answer")
    print("-" * 66)
    for verb in sorted(by_verb):
        rows = by_verb[verb]
        top1 = sum(r["verdict"] == "TOP1" for r in rows)
        any_hit = top1 + sum(r["verdict"] == "ANY_ONLY" for r in rows)
        miss = sum(r["verdict"] == "MISS" for r in rows)
        noev = sum(r["verdict"] in ("NO_EVENTS", "NO_OUTPUT") for r in rows)
        wrong = Counter(
            r["got"] for r in rows if r["verdict"] == "MISS" and r["got"]
        )
        common = f"{wrong.most_common(1)[0][0]} x{wrong.most_common(1)[0][1]}" \
            if wrong else "-"
        print(f"{verb:<9}{len(rows):>4}{top1:>7}{any_hit:>7}{miss:>6}"
              f"{noev:>6}   {common}")
    print("-" * 66)

    n = len(records)
    top1 = sum(r["verdict"] == "TOP1" for r in records)
    any_hit = top1 + sum(r["verdict"] == "ANY_ONLY" for r in records)
    noev = sum(r["verdict"] in ("NO_EVENTS", "NO_OUTPUT") for r in records)
    failed = sum(r["exit_code"] != 0 for r in records)
    print(f"top-1 verb accuracy : {top1}/{n} = {100.0 * top1 / n:.1f}%")
    print(f"any-event recall    : {any_hit}/{n} = {100.0 * any_hit / n:.1f}%")
    print(f"clips with no event : {noev}/{n}")
    print(f"pipeline exit != 0  : {failed}/{n}")

    # Balanced draw means the unweighted mean over verbs is close to the overall
    # figure; printed anyway because they diverge as soon as a verb runs short.
    per_verb_rates = [
        sum(r["verdict"] == "TOP1" for r in rows) / len(rows)
        for rows in by_verb.values()
    ]
    print(f"mean per-verb top-1 : "
          f"{100.0 * sum(per_verb_rates) / len(per_verb_rates):.1f}%")

    directions = Counter(
        r["direction"] for r in records if r["verdict"] in ("TOP1", "ANY_ONLY", "MISS")
    )
    stated = sum(v for k, v in directions.items() if k != "N/A")
    print(f"\ndirection (only the {stated} clips whose class name states one): "
          + ", ".join(f"{k}={v}" for k, v in sorted(directions.items())))

    print("\n--- wrong answers, grouped ---")
    pairs = Counter(
        (r["truth_action"], r["got"]) for r in records if r["verdict"] == "MISS"
    )
    for (want, got), count in pairs.most_common(15):
        print(f"  truth {want:<8} -> got {str(got):<8} x{count}")

    total_sec = sum(r["seconds"] for r in records)
    print(f"\nwall clock: {total_sec / 60:.1f} min for {n} clips "
          f"({total_sec / max(n, 1):.1f}s each)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path,
                        help="directory holding truth.json and clips/")
    parser.add_argument("--config", type=Path,
                        default=REPO / "config" / "ssv2_eval.yaml")
    parser.add_argument("--out", required=True, type=Path,
                        help="where per-clip run dirs and results.json go")
    parser.add_argument("--start", type=int, default=0,
                        help="index into truth.json's clip list (default 0)")
    parser.add_argument("--limit", type=int, default=0,
                        help="how many clips to run, 0 = all remaining")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--order", choices=("interleave", "file"),
                        default="interleave",
                        help="interleave (default) round-robins the verbs so a "
                             "partial run is still verb-balanced; file keeps "
                             "truth.json's own verb-grouped order")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip clips that already have an events.json under "
                             "--out, so an interrupted run can be resumed "
                             "without paying for the finished clips again")
    args = parser.parse_args()

    manifest = json.loads(
        (args.bundle / "truth.json").read_text(encoding="utf-8")
    )
    clips = manifest["clips"]
    if args.order == "interleave":
        clips = interleave_by_verb(clips)
    end = len(clips) if args.limit <= 0 else min(len(clips), args.start + args.limit)
    selected = clips[args.start:end]
    if args.skip_done:
        before = len(selected)
        selected = [
            c for c in selected
            if not (args.out / "runs" / c["clip_id"] / "events.json").exists()
        ]
        print(f"--skip-done: {before - len(selected)} clip(s) already have "
              f"events.json and are skipped")
    if not selected:
        raise SystemExit(f"no clips to run in range [{args.start}, {end}) "
                         f"of {len(clips)}")

    print(f"bundle : {args.bundle}")
    print(f"config : {args.config}")
    print(f"order  : {args.order}")
    print(f"clips  : {len(selected)} of {len(clips)} "
          f"(index {args.start}..{end - 1})")
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    results_path = args.out / "results.json"
    for i, clip in enumerate(selected, start=1):
        record = run_one(clip, args.bundle, args.config, args.out, args.python)
        records.append(record)
        mark = {"TOP1": "OK  ", "ANY_ONLY": "any ", "MISS": "WRONG",
                "NO_EVENTS": "none", "NO_OUTPUT": "FAIL"}[record["verdict"]]
        print(f"[{i:>3}/{len(selected)}] {mark} {clip['clip_id']:>7}  "
              f"want {clip['action']:<8} got {str(record['got']):<8} "
              f"{record['seconds']:>5.1f}s  {clip['label'][:44]}",
              flush=True)
        # Written after every clip rather than at the end. At ~38s per clip the
        # full bundle is nearly two hours, and an interruption at clip 180 must
        # not cost the measurement — the per-clip events.json would survive but
        # the verdicts would not.
        results_path.write_text(
            json.dumps({"_config": str(args.config),
                        "_bundle": str(args.bundle),
                        "_counts": manifest.get("_counts"),
                        "results": merge_results(results_path, [record])},
                       indent=2),
            encoding="utf-8",
        )

    report(merge_results(results_path, records))
    print(f"\nresults -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
