"""Record every raw text span GroundingDINO returns for a video.

Why this exists
---------------
Two span-resolution rules have now been designed against spans I assumed the
model returns rather than spans it does. The first missed that the tokenizer
splits long labels ("printed carton label" came back as "##on label"); the
second missed that a decoy containing every word of a real label swallows it
("picture of a push chopper" beat "push chopper" on shared-word count), which
cost 10 of 13 genuine chopper detections on tt7.

Neither failure was visible in detections.json, because that file records the
class the span was *resolved to* — never the span itself. This tool records the
spans, so the rule can be designed and tested offline against real strings.

``decoy_classes`` is deliberately not passed: the point is the full span
population, including the boxes a decoy would drop.

Run on Kaggle, where the GPU and the video are:

    python tools/dump_dino_spans.py \
        --video /kaggle/input/.../tt7.mp4 \
        --config config/kaggle_tt7_decoy_b.yaml \
        --out /kaggle/working/tt7_spans.json

It prints a span histogram, a real/print split at the frame given by
``--split-frame``, and a gzip+base64 blob of the whole record so the result can
be committed as a fixture and replayed without a GPU.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import yaml

from src.models.groundingdino_hf_detector import (
    DEFAULT_MODEL_ID,
    GroundingDINOHFDetector,
)


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="dino_spans.json")
    ap.add_argument(
        "--split-frame", type=int, default=None,
        help="Report spans at or before this frame separately from spans after "
             "it. For tt7 use 42: the chopper's own track is last observed "
             "there, so every later 'chopper' span is printed artwork.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    d = cfg.get("detector", {})
    stride = int(cfg.get("frame_sampling", {}).get("every_n_frames", 1))

    model_id = d.get("model") or DEFAULT_MODEL_ID
    if model_id.startswith("yolo"):
        model_id = DEFAULT_MODEL_ID

    det = GroundingDINOHFDetector(
        text_prompt=d["text_prompt"],
        box_threshold=float(d.get("confidence", 0.30)),
        text_threshold=float(d.get("text_threshold", 0.25)),
        device=_device(),
        model_id=model_id,
        nms_iou=d.get("nms_iou", 0.45),
        drop_unlabeled=bool(d.get("drop_unlabeled", True)),
        decoy_classes=None,  # record everything, drop nothing
    )

    # _resolve_class receives exactly the text detect() built from the span,
    # which is what any resolution rule has to work with. Wrapping it here is
    # therefore a faithful record and needs no change to the detector.
    rows: list[list] = []
    frame_now = {"idx": -1}
    original = det._resolve_class

    def spy(span: str):
        resolved = original(span)
        rows.append([frame_now["idx"], span, resolved[1]])
        return resolved

    det._resolve_class = spy  # type: ignore[method-assign]
    det.load()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for idx in range(0, total, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame_now["idx"] = idx
            det.detect(frame, idx, idx / fps)
    finally:
        cap.release()
        det.unload()

    record = {
        "commit": _commit(),
        "video": Path(args.video).name,
        "config": Path(args.config).name,
        "text_prompt": det.text_prompt,
        "labels": det.labels,
        "every_n_frames": stride,
        "frame_count": total,
        "fps": fps,
        "rows": rows,  # [frame_index, raw_span, resolved_class_name]
    }
    Path(args.out).write_text(json.dumps(record), encoding="utf-8")

    print(f"\nprompt: {det.text_prompt}")
    print(f"{len(rows)} spans over {total // stride + 1} sampled frames\n")
    print("raw span -> resolved, by count:")
    hist = Counter((r[1], r[2]) for r in rows)
    for (span, resolved), n in hist.most_common():
        flag = "" if span == resolved else "   <- rewritten"
        print(f"  {n:>4}  {span!r:<34} -> {resolved!r}{flag}")

    if args.split_frame is not None:
        cut = args.split_frame
        print(f"\nspans by side of frame {cut}:")
        for span in sorted({r[1] for r in rows}):
            before = sum(1 for r in rows if r[1] == span and r[0] <= cut)
            after = sum(1 for r in rows if r[1] == span and r[0] > cut)
            print(f"  {span!r:<34} f<={cut}: {before:>3}   f>{cut}: {after:>3}")

    blob = base64.b64encode(
        gzip.compress(json.dumps(record).encode("utf-8"))
    ).decode("ascii")
    print(f"\nFIXTURE ({len(blob)} chars)\n{blob}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
