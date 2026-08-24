"""Cut the pixels out of the video and put them in a grid, so a human can look.

Read-only, no GPU:

    python tools/crop_detections.py --run <run_dir> --video tt6.mp4 \
        --class-name "push chopper" --truth tests/fixtures/tt6_ground_truth.json \
        --clip-start 4.0 --clip-end 8.3 --out hidden.png

Why this exists. The statistical audit in ``audit_detections.py`` can prove that
detections continue while the labels say the object is hidden, but it cannot say
*what* the detector is looking at. Every candidate answer — a picture printed on
the carton, a reflection, a fold of cardboard shaped like a handle, the real
object still visible through a gap — produces the same detection count.

Pose statistics do not settle it either: a carton being folded changes its own
box, so even a print fixed to the cardboard moves in container-relative
coordinates. Rather than infer, extract the crops and let the eye decide. One
glance at a contact sheet answers a question no threshold can.

Run it twice — once over the interval where the object is genuinely visible,
once over the interval where the labels say it cannot be — and compare. If both
sheets show the same thing in the same place, the detector never tracked the
object at all.

*--clip-start* and *--clip-end* are seconds within ONE copy of the clip when the
label file declares repeats>1, matching how audit_detections.py folds buckets;
otherwise they are absolute seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schema.track import Track  # noqa: E402


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _wanted_frames(
    tracks: list[Track],
    class_name: str,
    period_frames: int | None,
    fps: float,
    clip_start: float,
    clip_end: float,
) -> dict[int, list[tuple[tuple[int, int, int, int], float, int]]]:
    """Map absolute frame index -> [(box, confidence, track_id)] to crop.

    Only real detections: a predicted point is where the tracker *guessed* the
    object was, so cropping one would show whatever happened to be at that
    coordinate and prove nothing about the detector.
    """
    wanted: dict[int, list[tuple[tuple[int, int, int, int], float, int]]] = {}
    for t in tracks:
        if t.class_name != class_name:
            continue
        for p in t.points:
            if p.detection_confidence <= 0.0:
                continue
            frame = p.frame_index % period_frames if period_frames else p.frame_index
            sec = frame / fps
            if not (clip_start <= sec < clip_end):
                continue
            box = (
                int(round(p.bbox.x1)), int(round(p.bbox.y1)),
                int(round(p.bbox.x2)), int(round(p.bbox.y2)),
            )
            wanted.setdefault(p.frame_index, []).append(
                (box, p.detection_confidence, t.track_id)
            )
    return wanted


def _thin(wanted: dict, max_tiles: int) -> dict:
    """Keep at most *max_tiles* crops, spread evenly over the frames selected."""
    total = sum(len(v) for v in wanted.values())
    if total <= max_tiles:
        return wanted
    frames = sorted(wanted)
    step = len(frames) / max_tiles
    keep = {frames[int(i * step)] for i in range(max_tiles)}
    return {f: wanted[f][:1] for f in sorted(keep)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--truth", default=None, help="label file, for the loop period")
    ap.add_argument("--clip-start", type=float, default=0.0)
    ap.add_argument("--clip-end", type=float, default=1e9)
    ap.add_argument("--out", required=True, help="PNG contact sheet to write")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-tiles", type=int, default=40)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile", type=int, default=140)
    args = ap.parse_args()

    raw = Path(args.run) / "tracks_raw.json"
    if not raw.is_file():
        print(f"no tracks_raw.json in {args.run}")
        return 1
    tracks = [Track.model_validate(t) for t in _load(raw)]

    period_frames = None
    if args.truth:
        truth = _load(Path(args.truth))
        if int(truth.get("repeats", 1)) > 1:
            period_frames = int(round(
                float(truth["source_clip_duration_sec"]) * args.fps
            ))

    wanted = _wanted_frames(
        tracks, args.class_name, period_frames, args.fps,
        args.clip_start, args.clip_end,
    )
    if not wanted:
        print(f"no real {args.class_name!r} detections in "
              f"[{args.clip_start}, {args.clip_end}) of the clip")
        return 1
    n_before = sum(len(v) for v in wanted.values())
    wanted = _thin(wanted, args.max_tiles)

    # One sequential pass. Seeking per frame is both slower and unreliable on
    # compressed video, where POS_FRAMES lands on the nearest keyframe.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open video: {args.video}")
        return 1
    tiles: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for box, conf, tid in wanted.get(idx, []):
            h, w = frame.shape[:2]
            x1, y1 = max(0, box[0]), max(0, box[1])
            x2, y2 = min(w, box[2]), min(h, box[3])
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            crop = cv2.resize(frame[y1:y2, x1:x2], (args.tile, args.tile))
            sec = ((idx % period_frames) if period_frames else idx) / args.fps
            cv2.putText(crop, f"{sec:.1f}s id{tid} {conf:.2f}", (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 2)
            cv2.putText(crop, f"{sec:.1f}s id{tid} {conf:.2f}", (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
            tiles.append(crop)
        idx += 1
    cap.release()

    if not tiles:
        print("no crops produced — do the frame indices exceed the video length?")
        return 1

    cols = min(args.cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * args.tile, cols * args.tile, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * args.tile:(r + 1) * args.tile,
              c * args.tile:(c + 1) * args.tile] = tile
    out = Path(args.out)
    cv2.imwrite(str(out), sheet)
    print(f"{args.class_name}: {n_before} real detections in "
          f"[{args.clip_start}, {args.clip_end})s of the clip -> "
          f"{len(tiles)} crops -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
