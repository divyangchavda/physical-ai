"""Print the geometry of every stitch decision, so tuning reads numbers.

Read-only. Reads ``tracks_raw.json`` (the tracker's own output, written before
stitching) and, if present, ``track_merges.json`` to say what actually happened.

    python tools/diagnose_stitch.py --run <run_dir>

Why this exists: the first two versions of the stitcher under-merged, and both
times the reason was invisible from the entity list alone. The counts tell you
*that* a merge was refused, never *why*. Guessing why cost a GPU run each time.
Every column below is the input to one of the stitcher's tests, so a refused
merge can be attributed to a specific bound rather than to a hypothesis.

Columns, per candidate pair (same class, ordered by start_frame):

    blind      frames between the last real detection and the next one — how long
               the detector was actually blind, which is what max_gap_frames
               should be judged against
    span       frag.start_frame - ent.end_frame; negative means the spans overlap.
               Overlap up to the tracker's deletion lag is an artifact, not
               evidence of two objects
    realIoU    IoU between those two real detections  -> vs iou_threshold
    realDist   centre distance / frame diagonal       -> vs max_center_dist_norm
    ghostIoU   the same IoU using the raw endpoints, ghost tail included. When
               realIoU is high and ghostIoU is 0, a Kalman extrapolation was
               being used as identity evidence
    shN/shIoU  frames where BOTH tracks have a real detection, and mean IoU there
               -> vs duplicate_min_iou
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.models.track_stitcher import (
    _center_dist_norm,
    _first_real,
    _iou,
    _last_real,
    _shared_real_iou,
)
from src.schema.track import Track


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory with tracks_raw.json")
    ap.add_argument("--class-name", default=None, help="restrict to one class")
    args = ap.parse_args()

    run = Path(args.run)
    raw_path = run / "tracks_raw.json"
    if not raw_path.is_file():
        print(f"no tracks_raw.json in {run} — needs a run at 1291510 or later")
        return 1

    tracks = [Track.model_validate(t) for t in _load(raw_path)]
    diagonal = 0.0
    meta = run / "episode.json"
    if meta.is_file():
        ep = _load(meta)
        vid = ep.get("video_metadata") or ep.get("video") or {}
        w, h = float(vid.get("width") or 0), float(vid.get("height") or 0)
        diagonal = (w * w + h * h) ** 0.5
    if diagonal <= 0:
        # Fall back to the largest box corner seen, which bounds the frame.
        w = max((p.bbox.x2 for t in tracks for p in t.points), default=0.0)
        h = max((p.bbox.y2 for t in tracks for p in t.points), default=0.0)
        diagonal = (w * w + h * h) ** 0.5
        print(f"(frame size not in episode.json; diagonal estimated at {diagonal:.0f}px)")

    # What the run actually decided, so the table can be checked against reality.
    merged_into: dict[int, int] = {}
    merges = run / "track_merges.json"
    if merges.is_file():
        m = _load(merges)
        print(f"run merged {m['tracks_before']} fragments -> {m['tracks_after']} entities")
        print(f"  overlap budget {m.get('max_overlap_frames')} | "
              f"iou {m.get('iou_threshold')} | dist {m.get('max_center_dist_norm')} | "
              f"duplicate iou {m.get('duplicate_min_iou')}")
        for e in m["entities"]:
            for tid in e["absorbed_track_ids"]:
                merged_into[tid] = e["entity_id"]

    by_class: dict[str, list[Track]] = {}
    for t in tracks:
        by_class.setdefault(t.class_name, []).append(t)

    for class_name in sorted(by_class):
        if args.class_name and class_name != args.class_name:
            continue
        frags = sorted(by_class[class_name], key=lambda t: (t.start_frame, t.track_id))
        print(f"\n=== {class_name} ({len(frags)} fragments) ===")
        print(f"  {'pair':<12} {'frames':<22} {'blind':>6} {'span':>6} "
              f"{'realIoU':>8} {'realDist':>9} {'ghostIoU':>9} {'shN':>4} "
              f"{'shIoU':>7}  outcome")
        for prev, cur in zip(frags, frags[1:]):
            a_obs, b_obs = _last_real(prev), _first_real(cur)
            if a_obs is None or b_obs is None:
                continue
            n_shared, mean_iou = _shared_real_iou(prev, cur)
            same = merged_into.get(prev.track_id) == merged_into.get(cur.track_id)
            outcome = (
                f"MERGED -> {merged_into.get(cur.track_id)}" if same
                else "separate"
            )
            print(
                f"  {prev.track_id:>4}->{cur.track_id:<6} "
                f"{f'{prev.start_frame}-{prev.end_frame} / {cur.start_frame}-{cur.end_frame}':<22} "
                f"{b_obs.frame_index - a_obs.frame_index:>6} "
                f"{cur.start_frame - prev.end_frame:>6} "
                f"{_iou(a_obs.bbox, b_obs.bbox):>8.3f} "
                f"{_center_dist_norm(a_obs.bbox, b_obs.bbox, diagonal):>9.3f} "
                f"{_iou(prev.points[-1].bbox, cur.points[0].bbox):>9.3f} "
                f"{n_shared:>4} {mean_iou:>7.3f}  {outcome}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
