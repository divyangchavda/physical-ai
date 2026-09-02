"""The full picture of one SSv2 evaluation run: not just the score, but the cause.

``tools/ssv2_eval.py`` answers "how many did we get right". That is the headline
and it is not enough to decide what to change next. The 60-clip run scored 27/60,
and the 33 that failed split into causes with wildly different fixes: 5 clips lost
to a Gemini 503, 3 to a detector that found no hand, and the rest to captions that
described a different action than the label. A per-verb table shows none of that —
it shows PULL at 0/5 and leaves you guessing whether the fix is a stem, a prompt,
or a retry.

So this reads a finished ``results.json`` and sorts every clip by **why**:

  NO_SEGMENT   s05 emitted no candidate segment, so the VLM was never asked. Its
               ``captions`` list is empty. Almost always no actor was detected.
  VLM_FAILED   the VLM was asked and returned nothing usable — every recorded
               caption has a status but no text. The API's own reason is quoted.
  UNMAPPED     a caption exists and s07 mapped it to UNKNOWN or to nothing: our
               verb table has no stem for the word the VLM chose.
  RANKING      the right verb IS among the emitted events, just not the highest
               confidence one. A confidence problem, not a vocabulary one.
  WRONG_VERB   a caption exists, mapped cleanly to a real verb, and that verb is
               not the label's. The caption is printed beside it, because whether
               the VLM misread the video or SSv2 and we simply disagree about the
               word is a judgement for the reader, not for this tool.

Every cause here is read off recorded fields. None of them is inferred from what
the numbers "look like" — the 5 VLM failures are reported as 503s because the
observation records say 503, and a run whose ``results.json`` predates the
``captions`` field reports UNKNOWN_CAUSE rather than guessing.

The headroom section is arithmetic, not a forecast: it says what top-1 would be if
a whole cause class vanished *and* every one of those clips then answered
correctly. That is an upper bound. Treating it as a prediction is how a 45% run
gets described as an 88% run.

Usage:

    python tools/ssv2_report.py --results /kaggle/working/runs/ssv2_b2/results.json
    python tools/ssv2_report.py --results .../results.json --save report.txt
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from s07_remap_score import remap_all  # noqa: E402
from ssv2_eval import report as verb_table  # noqa: E402

# Ordered worst-plumbing-first, because that is the order they are worth fixing
# in: a clip that never reached the VLM cannot be helped by any verb work.
CAUSES = ("NO_SEGMENT", "VLM_FAILED", "UNMAPPED", "RANKING", "WRONG_VERB",
          "UNKNOWN_CAUSE")

# "503 UNAVAILABLE" out of a much longer error string, so ten failures with the
# same cause group into one line instead of ten near-identical ones.
_STATUS = re.compile(r"\b(\d{3})\s+([A-Z][A-Z_]+)")


def cause(record: dict) -> str:
    """Why this clip did not produce the labelled verb, from recorded fields only.

    ``captions`` absent (rather than empty) means the run predates that field, and
    the cause genuinely cannot be recovered — say so instead of calling it
    NO_SEGMENT, which is what an empty list means.
    """
    if record.get("verdict") == "TOP1":
        return "CORRECT"
    if "captions" not in record:
        return "UNKNOWN_CAUSE"
    captions = record.get("captions") or []
    if not captions:
        return "NO_SEGMENT"
    if not any(c.get("raw_action") for c in captions):
        return "VLM_FAILED"
    if record.get("verdict") == "ANY_ONLY":
        return "RANKING"
    if record.get("got") in (None, "UNKNOWN"):
        return "UNMAPPED"
    return "WRONG_VERB"


def failure_reasons(record: dict, runs_dir: Path | None) -> list[str]:
    """The API's own words for why a call returned nothing.

    Read from ``results.json`` when the run recorded them, and otherwise from the
    per-clip ``vlm_observations.json`` if those directories still exist. Older
    runs have neither, and then the failure is reported without a reason rather
    than with a plausible one.
    """
    recorded = [c["error_reason"] for c in (record.get("captions") or [])
                if c.get("error_reason")]
    if recorded or runs_dir is None:
        return recorded
    path = runs_dir / str(record.get("clip_id")) / "vlm_observations.json"
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(raw, dict):
        raw = raw.get("observations", [])
    if not isinstance(raw, list):
        return []
    return [obs["error_reason"] for obs in raw
            if isinstance(obs, dict) and obs.get("error_reason")]


def reason_signature(text: str) -> str:
    """A short, groupable form of one error string."""
    match = _STATUS.search(text)
    return f"{match.group(1)} {match.group(2)}" if match else text[:56]


def caption_of(record: dict) -> str:
    """The caption the answer came from, or the first one recorded.

    ``raw_action`` is the winning event's caption and is what the verb was read
    from, so it is preferred. A clip with no event has no winner, and there the
    first recorded caption is the only text there is.
    """
    if record.get("raw_action"):
        return str(record["raw_action"])
    for entry in record.get("captions") or []:
        if entry.get("raw_action"):
            return str(entry["raw_action"])
    return ""


def _print_run_header(records: list[dict], payload: dict) -> None:
    print("=" * 78)
    print("SSv2 EVALUATION - FULL REPORT")
    print("=" * 78)
    print(f"clips     : {len(records)}")
    print(f"config    : {payload.get('_config')}")
    print(f"bundle    : {payload.get('_bundle')}")
    total = sum(float(r.get("seconds") or 0.0) for r in records)
    print(f"wall clock: {total / 60:.1f} min total, "
          f"{total / max(len(records), 1):.1f}s per clip")
    slowest = sorted(records, key=lambda r: -float(r.get("seconds") or 0.0))[:5]
    print("slowest   : " + ", ".join(
        f"{r['clip_id']}={r.get('seconds')}s" for r in slowest))


def _print_causes(by_cause: dict[str, list[dict]], n: int, top1: int) -> None:
    print("\n" + "=" * 78)
    print("WHY THE OTHER CLIPS DID NOT PRODUCE THE LABELLED VERB")
    print("=" * 78)
    print(f"{'cause':<14}{'clips':>6}{'share':>8}")
    print("-" * 78)
    for name in CAUSES:
        rows = by_cause.get(name, [])
        if rows:
            print(f"{name:<14}{len(rows):>6}{100.0 * len(rows) / n:>7.1f}%")
    print("-" * 78)
    print(f"{'CORRECT':<14}{top1:>6}{100.0 * top1 / n:>7.1f}%")


def _print_cause_detail(by_cause: dict[str, list[dict]],
                        runs_dir: Path | None) -> None:
    for name in CAUSES:
        rows = by_cause.get(name, [])
        if not rows:
            continue
        print(f"\n--- {name}: {len(rows)} clip(s) ---")
        for r in rows:
            head = (f"  {str(r['clip_id']):>7} want {str(r['truth_action']):<7} "
                    f"got {str(r.get('got')):<8}")
            if name == "VLM_FAILED":
                reasons = failure_reasons(r, runs_dir)
                shown = "; ".join(sorted({reason_signature(x) for x in reasons}))
                print(f"{head} | {shown or 'no reason recorded'}")
            elif name == "NO_SEGMENT":
                print(f"{head} | {r.get('label', '')}")
            elif name == "RANKING":
                print(f"{head} | ranked {r.get('all_verbs')} "
                      f"| {caption_of(r)[:52]}")
            else:
                print(f"{head} | {caption_of(r)[:60]}")


def _print_vlm_reliability(records: list[dict], runs_dir: Path | None) -> None:
    """How often the API answered at all — separately from whether it was right.

    Worth its own section because it is the one number here that says nothing
    about our code. A run where 8% of calls 503 is not a run whose accuracy can
    be compared against one where 0% did.
    """
    asked = failed = 0
    reasons: Counter[str] = Counter()
    for r in records:
        captions = r.get("captions")
        if captions is None:
            continue
        for entry in captions:
            asked += 1
            if not entry.get("raw_action"):
                failed += 1
        for text in failure_reasons(r, runs_dir):
            reasons[reason_signature(text)] += 1
    if not asked:
        return
    print("\n" + "=" * 78)
    print("VLM RELIABILITY (nothing here is about our code)")
    print("=" * 78)
    print(f"segments sent to the VLM : {asked}")
    print(f"returned nothing usable  : {failed} "
          f"({100.0 * failed / asked:.1f}%)")
    for signature, count in reasons.most_common():
        print(f"  x{count:<3} {signature}")


def _print_headroom(by_cause: dict[str, list[dict]], n: int, top1: int) -> None:
    print("\n" + "=" * 78)
    print("HEADROOM - AN UPPER BOUND, NOT A FORECAST")
    print("=" * 78)
    print("Each line assumes that cause disappears AND every one of those clips")
    print("then answers correctly. Neither assumption holds in full.")
    print(f"\n{'as measured':<34}{top1:>4}/{n} = {100.0 * top1 / n:>5.1f}%")
    for name in CAUSES:
        rows = by_cause.get(name, [])
        # UNKNOWN_CAUSE has no line: there is no such thing as fixing a cause we
        # could not identify, and printing a gain for it would inflate the bound.
        if rows and name != "UNKNOWN_CAUSE":
            gained = top1 + len(rows)
            print(f"{'+ if ' + name + ' were fixed':<34}{gained:>4}/{n} = "
                  f"{100.0 * gained / n:>5.1f}%")


def _print_appendix(records: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("APPENDIX - EVERY CLIP")
    print("=" * 78)
    print(f"{'clip':>7} {'want':<7} {'got':<8} {'cause':<13} caption / label")
    print("-" * 78)
    for r in sorted(records, key=lambda r: (r["truth_action"], str(r["clip_id"]))):
        text = caption_of(r) or f"[{r.get('label', '')}]"
        print(f"{str(r['clip_id']):>7} {str(r['truth_action']):<7} "
              f"{str(r.get('got')):<8} {cause(r):<13} {text[:38]}")


def _print_multi_verb(records: list[dict]) -> None:
    """Captions naming more than one action, which is where top-1 is arguable.

    s07 emits one event per clause, so "opens the pouch and pulls out a cable" is
    an OPEN and a PULL. Both are true; only one can be the answer. These are
    listed because a change to event confidence would move them and a change to
    the verb table would not.
    """
    multi = [(r, remap_all(r)) for r in records if r.get("raw_action")]
    multi = [(r, verbs) for r, verbs in multi if len(verbs) > 1]
    if not multi:
        return
    print("\n" + "=" * 78)
    print(f"CAPTIONS NAMING MORE THAN ONE ACTION: {len(multi)}")
    print("=" * 78)
    for r, verbs in multi:
        hit = "ok " if r["truth_action"] in verbs else "   "
        print(f"  {hit}{str(r['clip_id']):>7} want {str(r['truth_action']):<7} "
              f"chose {str(r.get('got')):<8} of {verbs} | {caption_of(r)[:40]}")


def render(payload: dict, records: list[dict], runs_dir: Path | None) -> str:
    """The whole report as text, so it can be printed and saved from one path."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_run_header(records, payload)
        verb_table(records)

        by_cause: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            by_cause[cause(record)].append(record)
        top1 = len(by_cause.pop("CORRECT", []))
        n = len(records)

        _print_causes(by_cause, n, top1)
        _print_cause_detail(by_cause, runs_dir)
        _print_vlm_reliability(records, runs_dir)
        _print_multi_verb(records)
        _print_headroom(by_cause, n, top1)
        _print_appendix(records)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path,
                        help="results.json written by tools/ssv2_eval.py")
    parser.add_argument("--runs", type=Path, default=None,
                        help="the run directory tree, read only for error "
                             "reasons that the results file does not carry "
                             "(defaults to <results parent>/runs if present)")
    parser.add_argument("--save", type=Path, default=None,
                        help="also write the report to this file")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    records = payload.get("results") if isinstance(payload, dict) else payload
    if not records:
        raise SystemExit(f"no results in {args.results}")

    runs_dir = args.runs
    if runs_dir is None:
        default = args.results.parent / "runs"
        runs_dir = default if default.is_dir() else None

    text = render(payload if isinstance(payload, dict) else {}, records, runs_dir)
    print(text, end="")
    if args.save:
        args.save.write_text(text, encoding="utf-8")
        print(f"saved -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
