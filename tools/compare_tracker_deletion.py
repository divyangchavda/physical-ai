"""Compare this tracker against the pre-change one on identical input.

The change in HEAD withholds a track point when the Kalman box is a
fabrication. The claim it has to earn is that *nothing else* moves: same track
ids, same deletion frames, same detection-backed points. This loads the
committed version of kalman_sparse_tracker.py side by side with the working-tree
version and diffs their behaviour frame by frame.

Run from the repo root:

    python tools/compare_tracker_deletion.py

Result on 2026-08-25, working tree against 63a7b37:

    scenario 'learns velocity, then coasts off the right edge'
      deletion frame    : old=21     new=21     SAME
      track ids created : old=[1, 2] new=[1, 2] SAME
      detection points  : old=6      new=6      SAME
      predicted points  : old=30     new=16     DIFFERENT
      1px boxes recorded: old=14     new=0      DIFFERENT

    scenario 'never leaves the frame'
      deletion frame    : old=21     new=21     SAME
      track ids created : old=[1]    new=[1]    SAME
      detection points  : old=6      new=6      SAME
      predicted points  : old=15     new=15     SAME
      1px boxes recorded: old=0      new=0      SAME

The old tracker recorded fourteen one-pixel boxes as observations; the new one
records none. Deletion frame, track identity and every detection-backed point
are unchanged, and the 14 withheld points are exactly the difference in
predicted points (30 - 16 = 14). The second scenario is the control: with no
box ever fabricated the two trackers agree on every field.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.schema.detection import BoundingBox, Detection  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080
OLD_REV = "HEAD"


def _load_old_module():
    """Import the committed kalman_sparse_tracker.py under a separate name."""
    source = subprocess.run(
        ["git", "show", f"{OLD_REV}:src/models/kalman_sparse_tracker.py"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / "old_kalman.py"
    tmp.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("old_kalman", tmp)
    module = importlib.util.module_from_spec(spec)
    sys.modules["old_kalman"] = module
    spec.loader.exec_module(module)
    return module


def _det(x: float, frame: int) -> Detection:
    return Detection(
        detection_id=f"d{frame}",
        frame_index=frame,
        timestamp_sec=frame / 30.0,
        bbox=BoundingBox(x1=x, y1=490.0, x2=x + 200.0, y2=590.0),
        class_id=0,
        class_name="push chopper",
        confidence=0.9,
        source="compare",
    )


def _run(tracker_cls, xs: list[float], total_frames: int) -> dict:
    """Feed detections for len(xs) frames, then nothing. Report what happened."""
    tracker = tracker_cls(
        frame_width=FRAME_W, frame_height=FRAME_H,
        detection_stride=1, min_hits=1,
    )
    seen_ids: list[int] = []
    snapshots: dict[int, list] = {}
    deletion_frame = None
    for frame in range(total_frames):
        dets = [_det(xs[frame], frame)] if frame < len(xs) else []
        tracker.update(dets, frame_index=frame)
        for tid, track in tracker.tracks.items():
            if tid not in seen_ids:
                seen_ids.append(tid)
            snapshots[tid] = list(track.points)
        if deletion_frame is None and seen_ids and not tracker.tracks:
            deletion_frame = frame

    points = [p for pts in snapshots.values() for p in pts]
    return {
        "deletion_frame": deletion_frame,
        "ids": seen_ids,
        "detection_points": sum(1 for p in points if p.detection_confidence > 0.0),
        "predicted_points": sum(1 for p in points if p.detection_confidence == 0.0),
        "one_px": sum(
            1 for p in points if p.bbox.width <= 1.0 or p.bbox.height <= 1.0
        ),
    }


SCENARIOS = {
    # Walks right at 100px/frame from x=1400. The step is deliberately under
    # half the 200px box width so IoU stays above the 0.20 match threshold and
    # the filter actually learns a velocity — at 200px/frame every detection
    # opens a new track instead and no prediction is ever extrapolated. Once
    # detections stop the learned velocity carries the box past x=1920, which
    # is the shape of tt7 track 6 (cardboard box, ended [1919, 0, 1920, 1]).
    "learns velocity, then coasts off the right edge": [
        1400.0 + i * 100.0 for i in range(6)
    ],
    # Control: stationary in open frame, so no box is ever fabricated and the
    # two trackers must agree on every single point.
    "never leaves the frame": [900.0] * 6,
}


def main() -> int:
    old = _load_old_module()
    from src.models.kalman_sparse_tracker import KalmanSparseTracker as New

    print(f"working tree vs {OLD_REV}\n")
    ok = True
    for name, xs in SCENARIOS.items():
        a = _run(old.KalmanSparseTracker, xs, total_frames=30)
        b = _run(New, xs, total_frames=30)
        print(f"scenario {name!r}")
        for key, label in (
            ("deletion_frame", "deletion frame    "),
            ("ids", "track ids created "),
            ("detection_points", "detection points  "),
            ("predicted_points", "predicted points  "),
            ("one_px", "1px boxes recorded"),
        ):
            same = "SAME" if a[key] == b[key] else "DIFFERENT"
            print(f"  {label}: old={a[key]!s:<6} new={b[key]!s:<6} {same}")
        # Deletion timing, track identity and real observations must not move.
        for key in ("deletion_frame", "ids", "detection_points"):
            if a[key] != b[key]:
                print(f"  !! {key} changed - this is a regression")
                ok = False
        if b["one_px"]:
            print("  !! new tracker still records 1px boxes")
            ok = False
        print()
    print("OK - only fabricated points differ" if ok else "REGRESSION")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
