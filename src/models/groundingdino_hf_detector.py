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

import re
import uuid

import numpy as np
import torch
from PIL import Image

from src.interfaces.detector import ObjectDetector
from src.logging_utils import get_logger
from src.schema.detection import BoundingBox, Detection

logger = get_logger(__name__)

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    """Whole-word tokens, with BERT wordpiece continuation markers removed.

    The processor returns each grounded span by detokenising the wordpieces it
    matched, so a label the tokenizer split mid-word comes back carrying the
    continuation marker: on tt7 the prompt label "printed carton label" was
    returned as the span "##on label". Stripping "##" turns that back into
    ordinary words that can be compared with the vocabulary.
    """
    return _WORD_RE.findall(text.replace("##", ""))


class GroundingDINOHFDetector(ObjectDetector):
    """Open-vocabulary detector using transformers' GroundingDINO port."""

    def __init__(
        self,
        text_prompt: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: str = "cuda",
        model_id: str = DEFAULT_MODEL_ID,
        nms_iou: float | None = 0.45,
        drop_unlabeled: bool = True,
        decoy_classes: list[str] | None = None,
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
            nms_iou: Per-class NMS IoU threshold. GroundingDINO decodes 900
                queries independently and the HF post-processor applies no
                NMS, so several queries routinely return near-identical boxes
                for one object. Set None to disable.
            drop_unlabeled: Discard boxes whose grounded text span is empty.
                These pass box_threshold but no token clears text_threshold,
                so they carry no class information and are useless downstream.
            decoy_classes: Labels present in *text_prompt* purely to attract a
                known confusion, discarded after matching. The detector must be
                offered a span before a box can be assigned to it, so a decoy
                only works if it is also in the prompt — hence two settings for
                one idea. Matched case-insensitively against the prompt labels.
        """
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.model_id = model_id
        self.nms_iou = nms_iou
        self.drop_unlabeled = drop_unlabeled

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
        # Word sets for span matching, in prompt order. Precomputed because
        # _resolve_class runs once per box per frame.
        self._label_words: list[set[str]] = [set(_words(n)) for n in self.labels]
        # Sentinel bucket for spans that match nothing. Detection.class_id is
        # constrained to >= 0, so this cannot be -1.
        self._unmatched_id: int = len(self.labels)

        # Normalised prompt: lowercase, "a. b. c." — the form the processor expects.
        self.text_prompt = ". ".join(self.labels) + "."

        self._unmatched_spans: dict[str, int] = {}
        self._n_dropped_unlabeled = 0
        self._n_dropped_decoy = 0
        self._n_suppressed_nms = 0
        self._n_kept = 0

        # Resolved against the parsed vocabulary rather than kept as free text,
        # so a decoy naming a class that is not in the prompt fails loudly here
        # instead of silently never firing.
        self._decoy_ids: set[int] = set()
        for name in decoy_classes or []:
            key = name.strip().lower()
            if key not in self._label_to_id:
                raise ValueError(
                    f"decoy class {name!r} is not in the text prompt. A decoy "
                    f"only absorbs a confusion if the detector is offered it as "
                    f"a span; add it to detector.text_prompt. "
                    f"Prompt labels: {self.labels}"
                )
            self._decoy_ids.add(self._label_to_id[key])

        logger.info(
            "GroundingDINO-HF: %d labels %s (unmatched bucket -> class_id %d, "
            "nms_iou=%s, drop_unlabeled=%s, decoys=%s)",
            len(self.labels), self.labels, self._unmatched_id,
            self.nms_iou, self.drop_unlabeled,
            sorted(self.labels[i] for i in self._decoy_ids),
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
        total = (self._n_kept + self._n_dropped_unlabeled
                 + self._n_dropped_decoy + self._n_suppressed_nms)
        logger.info(
            "GroundingDINO-HF filtering: %d raw -> %d kept "
            "(%d dropped unlabeled, %d dropped as decoy, %d suppressed by NMS)",
            total, self._n_kept, self._n_dropped_unlabeled,
            self._n_dropped_decoy, self._n_suppressed_nms,
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

        # 2. Shared whole words. Substring matching alone was enough while every
        #    label was one or two words, but it silently fails on longer ones:
        #    the tokenizer splits "printed carton label" and the post-processor
        #    returned the span "##on label", which contains no label and is
        #    contained in none, so it fell through to the unmatched bucket in
        #    step 3. The unmatched bucket is never filtered, so on tt7 those
        #    boxes survived as classes of their own — 8 detections tracked as
        #    "##on label" and 3 as "chopper picture", all of them the chopper
        #    printed on the carton, i.e. exactly what the decoys existed to
        #    absorb. Word matching resolves both to their decoy and drops them.
        #
        #    Ranked by shared words first, then by the fraction of the label's
        #    own words that matched. Both keys are load-bearing, measured
        #    against the two spans tt7 actually produced:
        #      "chopper picture" -> 2 shared with "picture of a push chopper"
        #                           vs 1 with "push chopper"      -> the decoy
        #      "chopper"         -> 1 shared with each, but 1/2 of "push
        #                           chopper" against 1/5           -> the real
        #    Shared count alone would send a bare "chopper" to the long decoy;
        #    fraction alone would send "chopper picture" to the real class.
        #    Prompt order breaks any remaining tie, so this is deterministic.
        span_words = set(_words(text))
        best: str | None = None
        best_key: tuple[int, float] = (0, 0.0)
        for label, label_words in zip(self.labels, self._label_words):
            shared = len(span_words & label_words)
            if not shared:
                continue
            key = (shared, shared / len(label_words))
            if best is None or key > best_key:
                best, best_key = label, key
        if best is not None:
            return self._label_to_id[best], best

        # 3. Substring fallback, for spans the tokenizer truncated mid-word
        #    ("choppe") where there is no whole word left to share. Longest
        #    label wins, as before.
        best = None
        for label in self.labels:
            if label in text or text in label:
                if best is None or len(label) > len(best):
                    best = label
        if best is not None:
            return self._label_to_id[best], best

        # 4. Genuinely unknown — record it, do not guess.
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

    # ─────────────────────────────────────────────────────────────────── NMS
    def _apply_nms(
        self, kept: list[tuple[float, float, float, float, float, int, str]]
    ) -> list[tuple[float, float, float, float, float, int, str]]:
        """Per-class NMS. Falls back to keeping everything if torchvision is absent.

        Class-aware rather than class-agnostic: a person standing at a table
        legitimately overlaps the table box, and suppressing one because of the
        other would delete a real object.
        """
        try:
            from torchvision.ops import batched_nms
        except ImportError:
            logger.warning("torchvision unavailable — skipping NMS")
            self.nms_iou = None  # do not retry on every frame
            return kept

        boxes = torch.tensor([k[:4] for k in kept], dtype=torch.float32)
        scores = torch.tensor([k[4] for k in kept], dtype=torch.float32)
        class_ids = torch.tensor([k[5] for k in kept], dtype=torch.int64)

        keep_idx = batched_nms(boxes, scores, class_ids, self.nms_iou).tolist()
        self._n_suppressed_nms += len(kept) - len(keep_idx)
        # batched_nms returns descending-score order; restore prompt/spatial
        # order so detections.json stays stable across runs.
        return [kept[i] for i in sorted(keep_idx)]

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

            # ── 1. clamp, drop degenerate + unlabeled boxes ──────────────────
            kept: list[tuple[float, float, float, float, float, int, str]] = []
            for box, score, span in zip(boxes, scores, spans):
                # Already absolute xyxy because target_sizes was supplied.
                x1, y1, x2, y2 = (float(v) for v in box)

                x1 = max(0.0, min(x1, width - 1.0))
                y1 = max(0.0, min(y1, height - 1.0))
                x2 = max(0.0, min(x2, float(width)))
                y2 = max(0.0, min(y2, float(height)))
                if x2 <= x1 or y2 <= y1:
                    continue

                text = str(span).strip().strip(".").strip()
                if not text and self.drop_unlabeled:
                    # Cleared box_threshold but no token cleared text_threshold,
                    # so there is no class to attach. Useless downstream.
                    self._n_dropped_unlabeled += 1
                    continue

                class_id, class_name = self._resolve_class(text)
                if class_id in self._decoy_ids:
                    # The decoy did its job by winning this box away from a real
                    # class. Dropped before NMS: per-class NMS means a decoy box
                    # could never suppress a real one anyway, so the outcome is
                    # identical and this does less work.
                    self._n_dropped_decoy += 1
                    continue
                kept.append((x1, y1, x2, y2, float(score), class_id, class_name))

            # ── 2. per-class NMS ────────────────────────────────────────────
            if self.nms_iou is not None and len(kept) > 1:
                kept = self._apply_nms(kept)

            # ── 3. build Detection objects ──────────────────────────────────
            detections: list[Detection] = []
            for x1, y1, x2, y2, score, class_id, class_name in kept:
                detections.append(
                    Detection(
                        detection_id=f"dinohf_{frame_index}_{uuid.uuid4().hex[:8]}",
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=min(1.0, max(0.0, score)),
                        source=self.model_name,
                        is_estimated=True,
                    )
                )
            self._n_kept += len(detections)
            return detections

        except Exception:
            logger.exception(
                "GroundingDINO-HF detection failed on frame %d", frame_index
            )
            return []
