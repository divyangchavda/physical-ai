"""Find where one video sits inside another, by matching pixels.

Read-only, no GPU:

    python tools/align_clips.py --short tt7.mp4 --long tt6.mp4

Why this exists. tt6.mp4 is 996 frames and tt7.mp4 is 200; if tt7 were one of
tt6's four copies it would be 249. So the two files are not the same cut, and a
label file written against tt6 cannot be assumed to describe tt7 — the missing
1.63s might be at the front, at the back, or split.

That distinction decides whether the labels transfer as written. Trimmed at the
back, every label before the new end is still correct and only the last one is
clipped. Trimmed at the front, every timestamp is shifted and every score
computed against them is wrong by that shift, silently.

Nothing but the pixels can settle it, so this matches them directly: each frame
is reduced to a small greyscale thumbnail, and the short clip's sequence is slid
along the long one to find the offset where they agree. Distinctive frames are
not required — the whole sequence has to line up at once, so a coincidental
match on one frame cannot carry the result.

Reports the best few offsets with their disagreement scores. A true alignment
shows as one offset far below the rest; if the top scores are all similar, the
clips do not contain each other and nothing here should be trusted.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

# Frames are compared as 32x32 greyscale. Large enough that two different
# moments in the same scene disagree, small enough that codec noise and
# re-encoding between the two files do not.
THUMB = 32


def _thumbnails(path: str) -> np.ndarray:
    """Decode every frame to a flat 32x32 grey vector. Shape (n_frames, 1024)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    out: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.append(
            cv2.resize(grey, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
            .astype(np.float32)
            .ravel()
        )
    cap.release()
    if not out:
        raise SystemExit(f"no frames decoded from {path}")
    return np.stack(out)


def _offset_scores(short: np.ndarray, long: np.ndarray) -> np.ndarray:
    """Mean absolute pixel difference for every position of *short* in *long*.

    Index i is the score for short[0] landing on long[i]. Lower is better; 0.0
    would be a bit-identical cut.
    """
    span = long.shape[0] - short.shape[0]
    if span < 0:
        raise SystemExit("--short has more frames than --long")
    return np.array(
        [np.abs(long[i:i + short.shape[0]] - short).mean() for i in range(span + 1)]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True, help="the clip to locate")
    ap.add_argument("--long", required=True, help="the clip to search in")
    ap.add_argument("--fps", type=float, default=30.0, help="for reporting seconds")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    short = _thumbnails(args.short)
    long = _thumbnails(args.long)
    print(f"short: {short.shape[0]} frames   long: {long.shape[0]} frames")

    scores = _offset_scores(short, long)
    order = np.argsort(scores)

    # Neighbouring offsets of a good match are themselves good, so reporting the
    # raw top-N would list the same alignment five times. Keep only offsets that
    # are a clip-length apart, which is what makes a repeated clip show up as
    # several distinct hits.
    picks: list[int] = []
    for idx in order:
        if all(abs(int(idx) - p) >= short.shape[0] // 2 for p in picks):
            picks.append(int(idx))
        if len(picks) >= args.top:
            break

    print("\nbest alignments (mean abs grey difference, 0 = identical):")
    for rank, off in enumerate(picks, 1):
        print(
            f"  {rank}. frame {off:5d}  ({off / args.fps:6.2f}s)  score {scores[off]:7.3f}"
        )

    best = picks[0]
    runner = picks[1] if len(picks) > 1 else None
    print(f"\nworst offset scores {scores.max():.3f}, best {scores[best]:.3f}")
    if runner is not None and scores[runner] < scores[best] * 2:
        print(
            "AMBIGUOUS: the runner-up is within 2x of the best, so the short clip "
            "is not clearly a cut of the long one. Do not shift any labels on this."
        )
        return 0

    lead = best / args.fps
    tail = (long.shape[0] - short.shape[0] - best) / args.fps
    print(
        f"CUT: short starts at frame {best} of long ({lead:.2f}s in), "
        f"leaving {tail:.2f}s of long after it ends."
    )
    if best == 0:
        print("Front-aligned: label times transfer unchanged; only the tail is missing.")
    else:
        print(f"Front-trimmed: every label time must move EARLIER by {lead:.2f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
