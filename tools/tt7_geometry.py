"""Read tt7 geometry from observed detections only, with no GPU and no Gemini.

``tests/fixtures/tt7_real_detections.json`` holds every track point of a tt7 run
whose ``detection_confidence`` was above zero — 229 points out of 706. The rest
are Kalman extrapolations: stride 3 means only 67 of 200 frames were ever shown
to GroundingDINO, so two thirds of ``Track.points`` is interpolation. An earlier
pass read track geometry at candidate-window boundary frames, which are mostly
interpolated, and produced offsets of six box widths and size ratios of 1.8.
None of that was in the video.

Run from the repo root:

    python tools/tt7_geometry.py

What the observed detections say, on 2026-08-26, for the run at 071dd95 with
config/kaggle_tt7_decoy_b.yaml:

  track  3  push chopper  14 real pts, f0..42
            iou vs box6 = 0.000 on all 12 shared frames. A genuinely separate
            object that never once overlaps the cardboard box in its observed
            life, and its last observation is f42 = 1.40s. The user's labels put
            the chopper inside the carton before 2.0s, so this track ending is
            the INSERT completing — the object stops existing on screen at the
            moment the action finishes, which is why no containment test can
            ever detect this INSERT. The signal is disappearance.

            Its area ratio against the carton falls 0.926 (f6) -> 0.685 (f9) ->
            0.282 (f15) and then holds near 0.31-0.35. The chopper is not
            shrinking; the carton is still entering frame and growing. The user
            reported the chopper is "rock steady from 0.20 to 0.50s while the
            carton is still entering", which is the same fact from the other
            side.

  track 13  push chopper  16 real pts, f99..144
            Wholly inside box6 on all 16 frames — intersection equals its own
            area exactly, to the float — at 0.178-0.255 of the carton's area.
            f99..144 is 3.30-4.80s, and the user independently reported the
            printed artwork is visible on the carton face from 3.3s to 4.8s.
            This is the print, and it is a clean sub-region of the carton.

An earlier fixture, from the baseline vocabulary, also contained four detections
that WERE the carton relabelled: f45 area ratio 1.336, f54 1.045, f90 1.008,
f93 1.010 — boxes as large as or larger than the box they were supposedly
inside, which any containment test scores True for no physical reason. Under
config/kaggle_tt7_decoy_b.yaml there are none: every remaining chopper
detection is either genuinely separate (track 3) or a genuine sub-region
(track 13). That, rather than the total print count, is what the decoy
vocabulary bought.

One caution this file cannot fix: the run produced a single candidate segment,
f3..199 = the whole 6.67s clip, where the baseline produced seven. Fewer
phantom chopper tracks means fewer person-object proximity transitions for
s05 to cut on. The seven windows were partly an artifact of the phantoms, but
one window means the VLM is asked to describe all seven labelled actions at
once and every extracted event inherits the same timestamps. Timing has to
come from geometry, not from the window.
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
