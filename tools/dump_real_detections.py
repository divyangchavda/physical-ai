"""Rebuild the observed-detections fixture from a completed run.

``tests/fixtures/tt7_real_detections.json`` was assembled by an ad-hoc script,
which meant regenerating it after a config change cost a GPU run plus improvised
code. This makes it repeatable.

The fixture holds only the track points whose ``detection_confidence`` is above
zero. At stride 3 only 67 of tt7's 200 frames are ever shown to the detector, so
two thirds of ``Track.points`` is Kalman extrapolation. An earlier geometry pass
read boxes at candidate-window boundary frames, which are mostly interpolated,
and reported offsets of six box widths and size ratios of 1.8 — none of which
was in the video. Excluding them at the fixture boundary means no downstream
tool can make that mistake again.

Run on Kaggle against a finished run directory:

    python tools/dump_real_detections.py --run /kaggle/working/dbfix2

Prints the per-track observed/total split and a gzip+base64 blob to commit as
the fixture.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import subprocess
import sys
from pathlib import Path

# Running this as a script puts tools/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _commit(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="a pipeline --output-dir")
    ap.add_argument("--config", default="", help="config name, recorded only")
    ap.add_argument("--video", default="", help="video name, recorded only")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run = Path(args.run)
    tracks = json.loads((run / "tracks.json").read_text(encoding="utf-8"))

    seg_path = run / "candidate_segments.json"
    segments = (
        json.loads(seg_path.read_text(encoding="utf-8")) if seg_path.exists() else []
    )

    out_tracks = []
    for t in tracks:
        real = [
            [p["frame_index"],
             p["bbox"]["x1"], p["bbox"]["y1"], p["bbox"]["x2"], p["bbox"]["y2"]]
            for p in t["points"]
            if (p.get("detection_confidence") or 0.0) > 0.0
        ]
        out_tracks.append({
            "track_id": t["track_id"],
            "class_name": t["class_name"],
            "n_points": len(t["points"]),
            "real": sorted(real),
        })

    record = {
        "commit": _commit(Path(__file__).resolve().parent.parent),
        "video": args.video or "unknown",
        "config": args.config or "unknown",
        "run": run.name,
        "segments": [
            {
                "segment_id": s.get("segment_id", ""),
                "start_frame": s.get("start_frame"),
                "end_frame": s.get("end_frame"),
                "start_sec": s.get("start_sec"),
                "end_sec": s.get("end_sec"),
            }
            for s in segments
        ],
        "tracks": out_tracks,
    }

    print(f"{'track':>6} {'class':<26} {'observed':>8} {'total':>6}  span")
    for t in out_tracks:
        frames = [r[0] for r in t["real"]]
        span = f"f{frames[0]}..{frames[-1]}" if frames else "-"
        print(f"{t['track_id']:>6} {t['class_name']:<26} {len(t['real']):>8} "
              f"{t['n_points']:>6}  {span}")
    print(f"\n{len(segments)} candidate segment(s)")

    payload = json.dumps(record)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    blob = base64.b64encode(gzip.compress(payload.encode("utf-8"))).decode("ascii")
    print(f"\nFIXTURE ({len(blob)} chars)\n{blob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
