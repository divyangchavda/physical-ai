"""Check geometric change points against the hand labels, with no GPU and no Gemini.

This is the gate for timing events by geometry. If the change points found in
``tests/fixtures/tt7_real_detections.json`` do not land near the boundaries of
the seven hand-labelled actions, then the geometry does not carry the timing and
wiring it into s07 would be building on nothing — which is worth knowing before
spending a GPU run rather than after.

Run from the repo root:

    python tools/tt7_changepoints.py

Every threshold is read out of the config file named by ``--config`` and printed,
so the numbers on screen can be checked against the numbers in use. Nothing here
is tt7-specific except the fixture and the label file; point ``--fixture`` and
``--truth`` at another pair and it runs unchanged.

What it prints:

1. the derived thresholds and where each came from
2. every change point, in frame order
3. for each labelled action boundary, the nearest change point and the gap,
   against the label file's own ``timing_tolerance_sec``
4. how many boundaries a change point reached, which is the number that decides
   whether step 4 of the plan is worth doing

What it said on 2026-08-27, fixture at 071dd95
----------------------------------------------
Twelve change points, and **the naive score is 8/8 but it is worthless**: every
instant of the 6.67s clip is within 1.0s of some change point, so any label times
whatever would have scored 100%. That is why the coverage line and the two harder
tests below it exist.

On the honest test — one distinct change point per boundary, in increasing order —
it is 7/8, or 5/6 once the clip's own start and end are set aside. **But twelve
evenly spaced times score 8/8.** A uniform grid that has never seen the video does
better than the geometry. So the conclusion is not "the geometry carries the
timing"; it is:

* the change points are real. 0 of 12 land more than 1.0s from a labelled
  boundary, so the geometry is not inventing cuts.
* tt7 cannot prove they beat an arbitrary split. A 1.0s tolerance is 15% of a
  6.67s clip; at that ratio timing is close to unfalsifiable.

Both facts matter for what to do next. Wiring change points into s07 is still
right — the pipeline currently gives every event the same whole-clip span, and
ordered distinct windows is what turns AMBIGUOUS into PRECISE and makes any
accuracy number mean something. What must not happen is reporting the resulting
tt7 score as evidence that geometric timing works. On this clip a uniform 7-way
split would score the same, and only a longer video or a tighter tolerance can
tell them apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# Running this as a script puts tools/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.track_changepoints import (  # noqa: E402
    find_change_points,
    from_fixture,
    still_displacement_ratio,
)

ROOT = Path(__file__).resolve().parent.parent


def _fps(fixture: dict, override: float | None) -> tuple[float, str]:
    """Frames per second, preferring a measured value over a default.

    tt7_dino_spans.json records the fps cv2 read off the video, so the rate does
    not have to be hardcoded here.
    """
    if override:
        return override, "--fps"
    spans = ROOT / "tests" / "fixtures" / "tt7_dino_spans.json"
    if fixture.get("video") and spans.is_file():
        rec = json.loads(spans.read_text(encoding="utf-8"))
        if rec.get("video") == fixture.get("video") and rec.get("fps"):
            return float(rec["fps"]), f"measured from {fixture['video']} in {spans.name}"
    raise SystemExit(
        "cannot determine fps: no matching span fixture, pass --fps explicitly"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", default="tests/fixtures/tt7_real_detections.json")
    ap.add_argument("--truth", default="tests/fixtures/tt7_ground_truth.json")
    ap.add_argument("--config", default="config/kaggle_tt7_decoy_b.yaml")
    ap.add_argument("--fps", type=float, default=None)
    args = ap.parse_args()

    fixture = json.loads((ROOT / args.fixture).read_text(encoding="utf-8"))
    truth = json.loads((ROOT / args.truth).read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))

    nms_iou = float(cfg["detector"]["nms_iou"])
    stride = int(cfg["frame_sampling"]["every_n_frames"])
    min_hits = int(cfg["tracker"]["min_hits"])
    background = frozenset(cfg.get("segment", {}).get("background_classes", []))
    fps, fps_source = _fps(fixture, args.fps)
    tolerance = float(truth.get("timing_tolerance_sec", 1.0))

    ratio = still_displacement_ratio(nms_iou)
    print(f"=== {args.fixture} vs {Path(args.truth).name} ===")
    print(f"fixture commit {fixture.get('commit')} config {fixture.get('config')}")
    print("\n--- thresholds, and where each came from ---")
    print(f"  nms_iou            {nms_iou:<8} detector.nms_iou in {Path(args.config).name}")
    print(f"  still ratio        {ratio:.4f}   (1-nms_iou)/(1+nms_iou), same-place offset")
    print(f"  stride             {stride:<8} frame_sampling.every_n_frames")
    print(f"  motion budget      {ratio / stride:.4f}   still ratio / stride, per frame")
    print(f"  min_hits           {min_hits:<8} tracker.min_hits, observations a state must hold")
    print(f"  background classes {sorted(background)}  segment.background_classes")
    print(f"  fps                {fps:<8} {fps_source}")
    print(f"  tolerance          {tolerance:<8} timing_tolerance_sec in {Path(args.truth).name}")

    tracks = from_fixture(fixture["tracks"])
    print(f"\n--- {len(tracks)} tracks, observed points only ---")
    for t in tracks:
        frames = t.frames
        span = f"f{frames[0]}..{frames[-1]}" if frames else "-"
        skip = " (background, excluded)" if t.class_name.lower() in {
            c.lower() for c in background
        } else ""
        print(f"  {t.track_id:>3} {t.class_name:<16} {len(frames):>3} pts  {span}{skip}")

    points = find_change_points(
        tracks, nms_iou=nms_iou, stride=stride, min_hits=min_hits, fps=fps,
        exclude_classes=background,
    )
    print(f"\n--- {len(points)} change points ---")
    for p in points:
        print(f"  {p}")

    # Boundaries: the start of every labelled action, plus the end of the last.
    # A boundary is where one action gives way to the next, which is what an
    # event's start_sec has to land on.
    actions = [a for a in truth["actions"] if a.get("action")]
    boundaries: list[tuple[float, str]] = [
        (float(a["start_sec"]), f"{a['order']} {a['action']} {a['object']} starts")
        for a in actions
    ]
    if actions:
        last = actions[-1]
        boundaries.append((float(last["end_sec"]), f"{last['order']} {last['action']} ends"))

    print(f"\n--- {len(boundaries)} labelled boundaries vs nearest change point ---")
    hits = 0
    for sec, name in boundaries:
        if not points:
            print(f"  {sec:>5.2f}s {name:<34} no change points at all")
            continue
        nearest = min(points, key=lambda p: abs(p.sec - sec))
        gap = abs(nearest.sec - sec)
        ok = gap <= tolerance
        hits += ok
        print(f"  {sec:>5.2f}s {name:<34} {'HIT ' if ok else 'MISS'} "
              f"gap={gap:>4.2f}s  <- {nearest.kind} {nearest.class_name}"
              f"#{nearest.track_id} at {nearest.sec:.2f}s")

    print(f"\n  boundaries reached : {hits}/{len(boundaries)} within {tolerance}s")

    # ── how much of that count is real, and how much is the tolerance
    #
    # Nearest-point matching flatters itself when the points are dense relative to
    # the tolerance: if every instant in the clip is within tolerance of SOME
    # change point, then every boundary hits no matter where the labels fall and
    # the count above measures nothing. Report the covered fraction so the count
    # can be read with that in mind.
    duration = float(truth.get("total_duration_sec") or 0.0)
    if points and duration > 0.0:
        covered = _covered_fraction(sorted(p.sec for p in points), duration, tolerance)
        print(f"  fraction of the clip within {tolerance}s of a change point : "
              f"{100.0 * covered:.0f}%")
        if covered >= 1.0:
            print("  => the hit count above is VACUOUS: any label times at all would "
                  "score 100%.\n     Read the assignment below instead.")

    # ── the honest test: one change point per boundary, in order
    #
    # This is what s07 would actually need — the clip cut into ordered windows,
    # one per action. Nearest-point matching can hand the same point to two
    # boundaries and can run backwards; neither is usable as timing. Assigning
    # strictly increasing points to boundaries in order, minimising the total
    # gap, is the same question asked in a way the answer can be used.
    if points and boundaries:
        assignment = _monotone_assign([s for s, _ in boundaries], [p.sec for p in points])
        print(f"\n--- one distinct change point per boundary, in order ---")
        if assignment is None:
            print(f"  IMPOSSIBLE: {len(boundaries)} boundaries need {len(boundaries)} "
                  f"points in increasing order, only {len(points)} exist")
        else:
            ordered_hits = 0
            interior_hits = interior_total = 0
            last = len(boundaries) - 1
            for i, ((sec, name), j) in enumerate(zip(boundaries, assignment)):
                p = points[j]
                gap = abs(p.sec - sec)
                ok = gap <= tolerance
                ordered_hits += ok
                # The first and last boundary ARE the clip's own start and end —
                # the list is built that way. Any track appearing or disappearing
                # reaches them, and every clip has one of each, so crediting the
                # geometry with them measures the video's edges rather than its
                # content. Tallied separately for that reason.
                edge = i == 0 or i == last
                if not edge:
                    interior_total += 1
                    interior_hits += ok
                print(f"  {sec:>5.2f}s {name:<34} {'HIT ' if ok else 'MISS'} "
                      f"gap={gap:>4.2f}s  <- {p.kind} {p.class_name}#{p.track_id} "
                      f"at {p.sec:.2f}s{'   (clip edge)' if edge else ''}")
            print(f"\n  boundaries with a distinct in-order point : "
                  f"{ordered_hits}/{len(boundaries)} within {tolerance}s")
            print(f"  of those, away from the clip edges        : "
                  f"{interior_hits}/{interior_total}")

            # ── the control
            #
            # The number above is only evidence if a set of points that knows
            # nothing about the video would do worse. Same count, spread evenly
            # over the clip: deterministic, and it needs no video at all. If the
            # grid scores as well, then this clip and this tolerance cannot tell
            # geometric cuts from arbitrary ones, and the score must not be
            # reported as proof that the geometry is what did it.
            if duration > 0.0:
                n = len(points)
                grid = [duration * i / (n - 1) for i in range(n)] if n > 1 else [0.0]
                grid_assign = _monotone_assign([s for s, _ in boundaries], grid)
                grid_hits = sum(
                    abs(grid[j] - sec) <= tolerance
                    for (sec, _), j in zip(boundaries, grid_assign or [])
                ) if grid_assign else 0
                print(f"  CONTROL, {n} evenly spaced times instead   : "
                      f"{grid_hits}/{len(boundaries)}"
                      f"{'  <- the grid does as well or better; this clip cannot' if grid_hits >= ordered_hits else ''}")
                if grid_hits >= ordered_hits:
                    print(f"     distinguish geometric cuts from arbitrary ones at "
                          f"{tolerance}s tolerance on a {duration:.2f}s clip.")

    # The reverse direction: a change point that matches no boundary is a cut the
    # geometry would make that the labels do not, which is how over-segmentation
    # would show up.
    if points and boundaries:
        spurious = [
            p for p in points
            if min(abs(p.sec - s) for s, _ in boundaries) > tolerance
        ]
        print(f"\n  change points matching no boundary : {len(spurious)}/{len(points)}")
        for p in spurious:
            print(f"    {p}")

    return 0


def _covered_fraction(secs: list[float], duration: float, tolerance: float) -> float:
    """Fraction of ``[0, duration]`` lying within *tolerance* of some point in *secs*."""
    spans: list[tuple[float, float]] = []
    for s in secs:
        lo, hi = max(0.0, s - tolerance), min(duration, s + tolerance)
        if hi <= lo:
            continue
        if spans and lo <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))
    return sum(hi - lo for lo, hi in spans) / duration if duration > 0 else 0.0


def _monotone_assign(
    boundaries: list[float], points: list[float]
) -> list[int] | None:
    """Strictly increasing point index per boundary, minimising total gap.

    Exact, by dynamic programming over (boundary, point) — both lists are short.
    Returns None when there are fewer points than boundaries.
    """
    m, n = len(boundaries), len(points)
    if m > n:
        return None
    inf = float("inf")
    # cost[i][j] = best total gap assigning boundaries[i:] to points from j on.
    cost = [[inf] * (n + 1) for _ in range(m + 1)]
    pick = [[-1] * (n + 1) for _ in range(m + 1)]
    for j in range(n + 1):
        cost[m][j] = 0.0
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            take = abs(points[j] - boundaries[i]) + cost[i + 1][j + 1]
            skip = cost[i][j + 1]
            if take <= skip:
                cost[i][j], pick[i][j] = take, j
            else:
                cost[i][j], pick[i][j] = skip, pick[i][j + 1]
    if cost[0][0] == inf:
        return None
    out, i, j = [], 0, 0
    while i < m:
        chosen = pick[i][j]
        out.append(chosen)
        i, j = i + 1, chosen + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
