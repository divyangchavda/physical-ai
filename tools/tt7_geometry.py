"""Read tt7 geometry from observed detections only, with no GPU and no Gemini.

``tests/fixtures/tt7_real_detections.json`` holds every track point of the tt7
run at commit 7dde80d whose ``detection_confidence`` was above zero — 243 points
out of 795. The other 552 are Kalman extrapolations: stride 3 means only 67 of
200 frames were ever shown to GroundingDINO, so two thirds of ``Track.points``
is interpolation. An earlier pass read track geometry at candidate-window
boundary frames, which are mostly interpolated, and produced offsets of six box
widths and size ratios of 1.8. None of that was in the video.

Run from the repo root:

    python tools/tt7_geometry.py

What the observed detections say, on 2026-08-26:

  track  2  push chopper  15 real pts, f6..42   iou vs box6 = 0.000 on all 13
            shared frames. A genuinely separate object that never once overlaps
            the cardboard box during its whole observed life.

  track  9  push chopper   3 real pts, f45..57
            f45 iou=0.746 area=1.336 | f54 iou=0.648 area=1.045
  track 12  push chopper  26 real pts, f69..168
            f90 iou=0.992 area=1.008 | f93 iou=0.990 area=1.010

            Four detections that are the cardboard box itself, relabelled: same
            pixels to within 1-2%, and larger than the box rather than inside
            it. GroundingDINO is matching the words "Push Chopper" printed on
            the carton. Their centres sit inside the box by construction, so any
            containment test scores them True for no physical reason.

  track 12  f99..168  intersection with box6 equals its own area exactly, to
            the float, on all 22 frames. That is the algebraic signature of
            A within B. f69 and f75 are not contained (31161 of 119719 px).

So the clip contains exactly one containment onset, between f75 (2.50s) and
f99 (3.30s). tt7 ground truth puts its only INSERT at 1.0-2.0s with a 1.0s
tolerance, and the chopper's own track (2) stops being detected at f42 = 1.40s.
Either the label timing or the detector's notion of "chopper" is wrong here;
these numbers cannot say which, because both stories predict the same boxes.

The duplicate and the containment cases separate on area ratio with room to
spare: genuine containment peaks at 0.255 (f138), the duplicates start at
1.008 (f90). A factor of four, no threshold judgment needed.
"""
from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tt7_real_detections.json"
FPS = 30.0


def _area(b: list[float]) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def _intersection(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return (x2 - x1) * (y2 - y1) if x2 > x1 and y2 > y1 else 0.0


def _iou(a: list[float], b: list[float]) -> float:
    i = _intersection(a, b)
    return i / (_area(a) + _area(b) - i) if i else 0.0


def load() -> tuple[dict, dict[int, dict[int, list[float]]], dict[int, str]]:
    """Return (fixture, {track_id: {frame: bbox}}, {track_id: class_name})."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    boxes = {
        t["track_id"]: {r[0]: r[1:] for r in t["real"]} for t in data["tracks"]
    }
    names = {t["track_id"]: t["class_name"] for t in data["tracks"]}
    return data, boxes, names


def main() -> int:
    data, boxes, names = load()
    print(f"fixture from commit {data['commit']} on {data['video']}")
    for tid, name in names.items():
        frames = sorted(boxes[tid])
        print(f"  {tid:>3} {name:<14} {len(frames):>3} real  f{frames[0]}..{frames[-1]}")

    # Every push-chopper detection measured against the cardboard box on the
    # same frame. Same frame only: comparing across frames would reintroduce
    # the interpolation this fixture exists to exclude.
    box_id = next(t for t, n in names.items() if n == "cardboard box")
    box = boxes[box_id]
    print(f"\npush chopper vs box{box_id}, observed frames only")
    for tid, name in names.items():
        if name != "push chopper":
            continue
        shared = sorted(set(boxes[tid]) & set(box))
        print(f"\n  chopper{tid}: {len(shared)} shared frames")
        for frame in shared:
            own = boxes[tid][frame]
            inter = _intersection(own, box[frame])
            contained = abs(inter - _area(own)) < 1e-6
            ratio = _area(own) / _area(box[frame])
            verdict = (
                "SAME PIXELS" if ratio > 0.9 and _iou(own, box[frame]) > 0.5
                else "contained" if contained
                else "separate"
            )
            print(
                f"    f{frame:>3} {frame / FPS:>5.2f}s iou={_iou(own, box[frame]):.3f} "
                f"area={ratio:.3f} {verdict}"
            )

    # The onset: the first observed frame on which a chopper is wholly inside
    # the box and is not simply the box relabelled.
    onset = None
    prev = None
    for frame in sorted({f for t, n in names.items() if n == "push chopper" for f in boxes[t]}):
        for tid, name in names.items():
            if name != "push chopper" or frame not in boxes[tid] or frame not in box:
                continue
            own = boxes[tid][frame]
            ratio = _area(own) / _area(box[frame])
            if ratio > 0.9:
                continue  # the box relabelled, not an object inside it
            inside = abs(_intersection(own, box[frame]) - _area(own)) < 1e-6
            if inside and onset is None and prev is False:
                onset = (frame, tid)
            prev = inside
    if onset:
        frame, tid = onset
        print(
            f"\ncontainment onset: chopper{tid} first wholly inside box{box_id} "
            f"at f{frame} = {frame / FPS:.2f}s"
        )
    for seg in data["segments"]:
        print(
            f"  window {seg['segment_id'][:9]} f{seg['start_frame']}-{seg['end_frame']} "
            f"[{seg['start_sec']:.2f},{seg['end_sec']:.2f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
