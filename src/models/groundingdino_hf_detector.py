"""GroundingDINO detector via HuggingFace transformers.

Status: IMPLEMENTED

Why this exists alongside groundingdino_detector.py
---------------------------------------------------
The original IDEA-Research repo needs its MultiScaleDeformableAttention CUDA
op compiled with `pip install -e .` against a matching CUDA toolkit. That build
fails on Kaggle. The transformers port is pure PyTorch: no compile step, and
the weights download from the Hub instead of needing a 660 MB manual upload.

Same model family (`grounding-dino-tiny` is the Swin-T backbone, i.e. the same
weights as groundingdino_swint_ogc.pth), so detection quality is comparable.

It also fixes a real defect in the repo-based path. GroundingDINO does phrase
grounding, not closed-set classification: it returns whatever text span it
matched, which may be a fragment ("chopper") or a merge of two prompt terms
("cardboard box chopper"). The old code did:

    class_id = self._class_name_to_id.get(class_name.lower(), 0)

so every unrecognised span silently became class 0 — "person" in the tt6
prompt. That mislabelled 111 of 575 detections and made the segmenter fire
person-object proximity across the whole video. Here, spans are matched to the
prompt vocabulary explicitly and anything left over goes to a dedicated
`unmatched` bucket, never to class 0.
"""
from __future__ import annotations

import uuid

import numpy as np
import torch
from PIL import Image

from src.interfaces.detector import ObjectDetector
from src.logging_utils import get_logger
from src.schema.detection import BoundingBox, Detection

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


class GroundingDINOHFDetector(ObjectDetector):
    """Open-vocabulary detector using transformers' GroundingDINO port."""

    def __init__(
        self,
        text_prompt: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: str = "cuda",
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        """
        Args:
            text_prompt: Prompt in GroundingDINO form, e.g.
                "person . cardboard box . push chopper ."
            box_threshold: Minimum box confidence.
            text_threshold: Minimum text-match confidence.
            device: "cuda" or "cpu" (already resolved by s03_detect).
            model_id: Hub id. `grounding-dino-tiny` (Swin-T) or
                `grounding-dino-base` (Swin-B, larger and slower).
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.model_id = model_id

        self.model = None
        self.processor = None

        # Canonical vocabulary parsed from the prompt, in prompt order.
        self.labels: list[str] = [
            part.strip().lower()
            for part in text_prompt.split(".")
            if part.strip()
        ]
        if not self.labels:
            raise ValueError(f"No labels could be parsed from prompt: {text_prompt!r}")

        self._label_to_id: dict[str, int] = {
            name: idx for idx, name in enumerate(self.labels)
        }
        # Sentinel bucket for spans that match nothing. Detection.class_id is
        # constrained to >= 0, so this cannot be -1.
        self._unmatched_id: int = len(self.labels)

        # Normalised prompt: lowercase, "a. b. c." — the form the processor expects.
        self.text_prompt = ". ".join(self.labels) + "."

        self._unmatched_spans: dict[str, int] = {}

        logger.info(
            "GroundingDINO-HF: %d labels %s (unmatched bucket -> class_id %d)",
            len(self.labels), self.labels, self._unmatched_id,
        )

    # ────────────────────────────────────────────────────────────── lifecycle
    def load(self) -> None:
        if self.model is not None:
            logger.warning("GroundingDINO-HF already loaded")
            return

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for the groundingdino_hf backend. "
                "Install with: pip install transformers"
            ) from exc

        logger.info("Loading %s (downloads on first use)", self.model_id)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id
            ).to(self.device)
            self.model.eval()
        except Exception as exc:
            logger.exception("Failed to load %s", self.model_id)
            raise RuntimeError(f"Failed to load {self.model_id}: {exc}") from exc

        logger.info(
            "GroundingDINO-HF ready (device=%s, box_threshold=%.2f, "
            "text_threshold=%.2f, prompt=%r)",
            self.device, self.box_threshold, self.text_threshold, self.text_prompt,
        )

    def unload(self) -> None:
        if self.model is None:
            return
        del self.model
        self.model = None
        self.processor = None
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._unmatched_spans:
            logger.warning(
                "GroundingDINO-HF: %d span(s) matched no prompt label and were "
                "assigned class_id %d: %s",
                sum(self._unmatched_spans.values()), self._unmatched_id,
                dict(sorted(self._unmatched_spans.items(),
                            key=lambda kv: -kv[1])[:10]),
            )
        logger.info("GroundingDINO-HF unloaded")

    @property
    def model_name(self) -> str:
        return f"groundingdino_hf:{self.model_id.split('/')[-1]}"

    # ─────────────────────────────────────────────────────── span -> class id
    def _resolve_class(self, span: str) -> tuple[int, str]:
        """Map a grounded text span onto the prompt vocabulary.

        Returns (class_id, class_name). Falls back to the unmatched bucket
        rather than to class 0.
        """
        text = span.strip().lower().strip(".").strip()
        if not text:
            return self._unmatched_id, "unmatched"

        # 1. Exact vocabulary hit.
        if text in self._label_to_id:
            return self._label_to_id[text], text

        # 2. Longest prompt label contained in the span, or vice versa. This
        #    catches fragments ("chopper" -> "push chopper") and merges
        #    ("cardboard box chopper" -> "cardboard box").
        best: str | None = None
        for label in self.labels:
            if label in text or text in label:
                if best is None or len(label) > len(best):
                    best = label
        if best is not None:
            return self._label_to_id[best], best

        # 3. Genuinely unknown — record it, do not guess.
        self._unmatched_spans[text] = self._unmatched_spans.get(text, 0) + 1
        return self._unmatched_id, text

    def _post_process(self, outputs, input_ids, height: int, width: int) -> dict:
        """Call post_process_grounded_object_detection across transformers versions."""
        fn = self.processor.post_process_grounded_object_detection
        common = {
            "input_ids": input_ids,
            "target_sizes": [(height, width)],
            "text_threshold": self.text_threshold,
        }
        try:
            return fn(outputs, threshold=self.box_threshold, **common)[0]
        except TypeError:
            # Older transformers named it box_threshold.
            return fn(outputs, box_threshold=self.box_threshold, **common)[0]

    # ──────────────────────────────────────────────────────────────── detect
    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        """Detect objects in one BGR frame. Never raises — returns [] on error."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            height, width = frame.shape[:2]
            image = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB

            inputs = self.processor(
                images=image, text=self.text_prompt, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            result = self._post_process(outputs, inputs.input_ids, height, width)

            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            # transformers >= 4.51 returns "text_labels"; older returns "labels".
            spans = result.get("text_labels") or result.get("labels") or []

            detections: list[Detection] = []
            for box, score, span in zip(boxes, scores, spans):
                # Already absolute xyxy because target_sizes was supplied.
                x1, y1, x2, y2 = (float(v) for v in box)

                x1 = max(0.0, min(x1, width - 1.0))
                y1 = max(0.0, min(y1, height - 1.0))
                x2 = max(0.0, min(x2, float(width)))
                y2 = max(0.0, min(y2, float(height)))
                if x2 <= x1 or y2 <= y1:
                    continue

                class_id, class_name = self._resolve_class(str(span))

                detections.append(
                    Detection(
                        detection_id=f"dinohf_{frame_index}_{uuid.uuid4().hex[:8]}",
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=min(1.0, max(0.0, float(score))),
                        source=self.model_name,
                        is_estimated=True,
                    )
                )
            return detections

        except Exception:
            logger.exception(
                "GroundingDINO-HF detection failed on frame %d", frame_index
            )
            return []
