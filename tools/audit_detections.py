"""Ask whether detections exist when the labelled world says they cannot.

Read-only, no GPU: reads ``tracks_raw.json`` from a finished run.

    python tools/audit_detections.py --run <run_dir> \
        --truth tests/fixtures/tt6_ground_truth.json --class-name "push chopper"

Tracking metrics cannot see this failure. A detector that reports an object
continuously scores well on every continuity measure precisely *because* it
never stops — and if the object is actually hidden, the track is a phantom and
every event built on it is fabricated.

On tt6 the chopper had unbroken coverage across all 996 frames, while the hand
labels say it is sealed inside a closed box from 3.0s of each 8.3s copy. Those
two statements cannot both be true.

The audit folds every frame onto its position within one copy (via *repeats* and
*source_clip_duration_sec* in the label file) so all copies stack, then reports
per time-bucket: how many real detections, at what confidence, and how much of
the object's box sits inside a container-class box at that same frame. High
containment means the detector is outlining the container and naming it the
contained object. A video that is not a loop sets repeats=1 and the buckets are
just absolute time.

Containment alone cannot say *why*. Retail cartons carry a photograph of the
product, so an open-vocabulary detector asked for "push chopper" will happily
outline the picture — and a picture is glued to the cardboard. The print test
therefore measures the detection's pose *relative to the container*: fixed pose
across hundreds of frames is print, moving pose is an object with its own
motion. That distinction changes the fix. A phantom needs suppression; a print
needs the detector taught to want the physical object.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema.detection import BoundingBox  # noqa: E402
from src.schema.track import Track  # noqa: E402


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _containment(inner: BoundingBox, outer: BoundingBox) -> float:
    """Fraction of *inner* that lies inside *outer*."""
    ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / inner.area if inner.area > 0 else 0.0


def _relative_pose(inner: BoundingBox, outer: BoundingBox) -> tuple[float, float, float]:
    """Where *inner* sits inside *outer*, in units of *outer*'s own size.

    This is the discriminator between a real object inside a container and a
    picture of that object printed on the container. Print is fixed to the
    cardboard: as the carton is carried, folded and set down, the print moves
    with it, so its pose *relative to the carton* barely changes. A real object
    being handled inside or near a container does not hold still that way.
    """
    ow = max(outer.x2 - outer.x1, 1e-6)
    oh = max(outer.y2 - outer.y1, 1e-6)
    cx = ((inner.x1 + inner.x2) / 2.0 - outer.x1) / ow
    cy = ((inner.y1 + inner.y2) / 2.0 - outer.y1) / oh
    return cx, cy, inner.area / max(outer.area, 1e-6)


def _spread(values: list[float]) -> float:
    """Population standard deviation, 0.0 for fewer than two samples."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--truth", default=None, help="label file, for the loop period")
    ap.add_argument("--class-name", required=True, help="class to audit")
    ap.add_argument("--container-class", default=None,
                    help="class the object may be hidden inside")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--bucket-sec", type=float, default=1.0)
    ap.add_argument("--print-containment", type=float, default=0.9,
                    help="containment above which a detection is treated as "
                         "sitting on/in the container, for the print test")
    args = ap.parse_args()

    run = Path(args.run)
    raw = run / "tracks_raw.json"
    if not raw.is_file():
        print(f"no tracks_raw.json in {run}")
        return 1

    tracks = [Track.model_validate(t) for t in _load(raw)]

    period_frames = None
    if args.truth:
        truth = _load(Path(args.truth))
        if int(truth.get("repeats", 1)) > 1:
            period_frames = int(round(
                float(truth["source_clip_duration_sec"]) * args.fps
            ))

    subject = [t for t in tracks if t.class_name == args.class_name]
    if not subject:
        print(f"no tracks of class {args.class_name!r}; "
              f"present: {sorted({t.class_name for t in tracks})}")
        return 1

    containers: dict[int, list[BoundingBox]] = {}
    if args.container_class:
        for t in tracks:
            if t.class_name != args.container_class:
                continue
            for p in t.points:
                if p.detection_confidence > 0.0:
                    containers.setdefault(p.frame_index, []).append(p.bbox)

    # Only real detections. Predicted points are the tracker's opinion, not the
    # detector's, and counting them would answer a different question.
    buckets: dict[int, list[tuple[float, float]]] = {}
    contained_poses: list[tuple[float, float, float]] = []
    n_real = 0
    for t in subject:
        for p in t.points:
            if p.detection_confidence <= 0.0:
                continue
            n_real += 1
            frame = p.frame_index % period_frames if period_frames else p.frame_index
            key = int((frame / args.fps) / args.bucket_sec)
            best, best_box = 0.0, None
            for c in containers.get(p.frame_index, []):
                score = _containment(p.bbox, c)
                if score > best:
                    best, best_box = score, c
            if best_box is not None and best >= args.print_containment:
                contained_poses.append(_relative_pose(p.bbox, best_box))
            buckets.setdefault(key, []).append((p.detection_confidence, best))

    print(f"=== {args.class_name} — {len(subject)} raw tracks, "
          f"{n_real} real detections ===")
    if period_frames:
        print(f"all copies folded onto one {period_frames}-frame "
              f"({period_frames / args.fps:.1f}s) clip")
    header = f"  {'t in clip':<12} {'dets':>5} {'mean conf':>10}"
    if args.container_class:
        header += f" {'in ' + args.container_class:>22}"
    print(header)
    for key in sorted(buckets):
        rows = buckets[key]
        conf = sum(c for c, _ in rows) / len(rows)
        line = (f"  {key * args.bucket_sec:>4.1f}-"
                f"{(key + 1) * args.bucket_sec:<7.1f} {len(rows):>5} {conf:>10.3f}")
        if args.container_class:
            cont = sum(x for _, x in rows) / len(rows)
            line += f" {cont:>21.0%}"
        print(line)

    if args.container_class and contained_poses:
        cxs = [p[0] for p in contained_poses]
        cys = [p[1] for p in contained_poses]
        areas = [p[2] for p in contained_poses]
        sx, sy, sa = _spread(cxs), _spread(cys), _spread(areas)
        print(f"\n--- print test: {len(contained_poses)} detections at "
              f">={args.print_containment:.0%} containment ---")
        print(f"  pose inside the {args.container_class}, "
              f"in units of its own width/height:")
        print(f"    centre x  mean {sum(cxs) / len(cxs):.3f}  spread {sx:.3f}")
        print(f"    centre y  mean {sum(cys) / len(cys):.3f}  spread {sy:.3f}")
        print(f"    area frac mean {sum(areas) / len(areas):.3f}  spread {sa:.3f}")
        worst = max(sx, sy)
        if worst <= 0.05:
            verdict = ("PRINT. The box is pinned to one spot on the container "
                       "across every frame — that is a picture on the carton, "
                       "not an object being handled.")
        elif worst <= 0.12:
            verdict = ("LIKELY PRINT. The pose barely moves relative to the "
                       "container; a handled object would not hold this still.")
        else:
            verdict = ("NOT PRINT. The pose moves relative to the container, so "
                       "these are detections of something that moves "
                       "independently — real object, or a wandering box.")
        print(f"  verdict: {verdict}")

    print("\nCompare against the label file: any bucket where the object is "
          "hidden\nbut detections continue is a false-positive rate, and every "
          "event built\non those frames is fabricated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
