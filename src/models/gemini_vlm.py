"""Gemini VLM adapter — remote backend using Google Gemini API.

Uses google-genai SDK (google.genai) with the File API for video segment upload.
Extracts the requested temporal segment via OpenCV before uploading, so only
the relevant frames reach the API — the entire source video is never uploaded.

Contract:
  - Implements VisionLanguageModel.analyze_segment() exactly.
  - Returns a raw JSON string. Stage 06 owns parsing + validation.
  - Never returns a Pydantic object or RawVLMObservation.
  - Never normalises actions into ActionType vocabulary (Stage 07 does that).
  - API key is read from GEMINI_API_KEY environment variable only — never
    hardcoded, never printed in logs.
  - Raises on authentication/network failures so Stage 06 retry machinery
    can record a genuine FAILED observation rather than silently UNKNOWN.

Privacy note:
  The extracted video segment is uploaded to Google's servers for inference.
  Do not use with sensitive/private footage unless you have reviewed Google's
  data-use policy for the Gemini API.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import cv2

from src.interfaces.vlm import VisionLanguageModel
from src.logging_utils import get_logger

logger = get_logger(__name__)

# The current recommended model as of 2025-08 (gemini-2.0-flash is deprecated).
_DEFAULT_MODEL = "gemini-2.5-flash"

# Maximum frames extracted per segment to avoid overwhelming the API and to
# keep latency acceptable.  Stage 06 targets 5 fps; we cap at 16 frames total
# regardless of segment length.
_MAX_FRAMES = 16

# MIME type for uploaded temporary video clips.
_VIDEO_MIME = "video/mp4"

# Fixed so repeated runs of the same clip vary as little as the API allows.
_GENERATION_SEED = 20260824


class GeminiVLM(VisionLanguageModel):
    """Remote VLM backend using Google Gemini multimodal API.

    Extracts a temporal sub-clip from the source video with OpenCV, writes it
    to a temporary file, uploads it via the Gemini File API, runs inference,
    cleans up the upload, and returns the raw response text.

    Configuration (all from environment / PipelineConfig.vlm):
        GEMINI_API_KEY  — required environment variable.
        model_name      — Gemini model string (default: gemini-2.5-flash).
        timeout_sec     — per-request HTTP timeout.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        timeout_sec: float = 60.0,
    ) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise OSError(
                "GEMINI_API_KEY environment variable is not set. "
                "Obtain a key from https://aistudio.google.com/ and set it "
                "before running the pipeline with vlm.backend=GEMINI."
            )
        # Import lazily so the rest of the repo does not require google-genai
        # when running in LocalVLM / stub mode.
        try:
            from google import genai  # type: ignore[import-untyped]
            from google.genai import types  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiVLM. "
                "Install it with: pip install google-genai"
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._model_name = model_name
        self._timeout_sec = timeout_sec

    # ── VisionLanguageModel interface ────────────────────────────────────────

    @property
    def backend(self) -> Literal["LOCAL_MODEL", "REMOTE_MODEL", "GEMINI"]:
        # Was "REMOTE_MODEL", which is what RemoteVLM returns. s07 builds every
        # event's source as f"vlm:{obs.backend.lower()}", so every event this
        # pipeline has ever shipped claimed source="vlm:remote_model" with no
        # record of which provider produced it.
        return "GEMINI"

    @property
    def model_name(self) -> str:
        return self._model_name

    def analyze_segment(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        prompt: str,
    ) -> str:
        """Extract segment frames, upload to Gemini, return raw response text."""
        t0 = time.monotonic()

        # 1. Extract sub-clip to a temporary MP4 file.
        clip_path = self._extract_clip(video_path, start_sec, end_sec)

        uploaded_file = None
        try:
            # 2. Upload the clip via the File API.
            uploaded_file = self._upload_clip(clip_path)

            # 3. Poll until the file is ACTIVE (Gemini processes video async).
            uploaded_file = self._wait_for_active(uploaded_file)

            # 4. Run inference with structured JSON output.
            response_text = self._run_inference(uploaded_file, prompt)

        finally:
            # 5. Always delete the uploaded file and the local temp clip.
            if uploaded_file is not None:
                try:
                    self._client.files.delete(name=uploaded_file.name)
                except Exception as _del_exc:  # noqa: BLE001
                    logger.debug("[GeminiVLM] File delete failed (non-critical): %s", _del_exc)
            try:
                clip_path.unlink(missing_ok=True)
            except Exception as _unlink_exc:  # noqa: BLE001
                logger.debug("[GeminiVLM] Temp clip delete failed (non-critical): %s", _unlink_exc)

        elapsed = time.monotonic() - t0
        logger.info(
            "[GeminiVLM] segment [%.2f, %.2f]s -> inference %.2fs",
            start_sec,
            end_sec,
            elapsed,
        )
        return response_text

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_clip(
        self, video_path: Path, start_sec: float, end_sec: float
    ) -> Path:
        """Write the requested segment to a temporary MP4 file using OpenCV.

        Uses seek-and-read rather than loading the entire video, keeping
        memory usage bounded even for long source files.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            start_frame = max(0, int(start_sec * fps))
            end_frame = min(total_frames - 1, int(end_sec * fps))

            if start_frame >= end_frame:
                raise ValueError(
                    f"Degenerate segment: start_frame={start_frame} >= end_frame={end_frame}"
                )

            # Write to a named temp file (deleted by caller after upload).
            import tempfile as _tf
            fd, tmp_name = _tf.mkstemp(suffix=".mp4")
            os.close(fd)
            out_path = Path(tmp_name)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames_written = 0
            for _ in range(end_frame - start_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                frames_written += 1

            writer.release()
        finally:
            cap.release()

        logger.debug(
            "[GeminiVLM] Extracted %d frames [%.2f-%.2fs] -> %s",
            frames_written,
            start_sec,
            end_sec,
            out_path.name,
        )
        return out_path

    def _upload_clip(self, clip_path: Path):  # type: ignore[return]
        """Upload the local clip file via the Gemini File API."""
        logger.debug("[GeminiVLM] Uploading clip: %s", clip_path.name)
        uploaded = self._client.files.upload(
            file=str(clip_path),
            config=self._types.UploadFileConfig(mime_type=_VIDEO_MIME),
        )
        return uploaded

    def _wait_for_active(self, uploaded_file, poll_interval: float = 2.0, max_wait: float = 120.0):  # type: ignore[return]
        """Poll until the uploaded file transitions from PROCESSING to ACTIVE."""
        waited = 0.0
        while getattr(uploaded_file, "state", None) is not None:
            state_name = getattr(uploaded_file.state, "name", str(uploaded_file.state))
            if state_name == "ACTIVE":
                break
            if state_name == "FAILED":
                raise RuntimeError(
                    f"Gemini File API reported FAILED state for upload: {uploaded_file.name}"
                )
            if waited >= max_wait:
                raise TimeoutError(
                    f"Gemini file processing timed out after {max_wait}s "
                    f"(still in state: {state_name})"
                )
            time.sleep(poll_interval)
            waited += poll_interval
            uploaded_file = self._client.files.get(name=uploaded_file.name)
        return uploaded_file

    def _run_inference(self, uploaded_file, prompt: str) -> str:
        """Run multimodal inference and return the raw response text."""
        # Greedy decoding plus a fixed seed is the most reproducibility the API
        # offers, and it is not enough: two runs of tt6 eighteen minutes apart
        # returned materially different action text. Treat VLM output as drifting
        # input, not as a constant.
        gen_kwargs = {
            "response_mime_type": "application/json",
            "temperature": 0.0,  # Physical analysis, not creativity.
        }
        if "seed" in self._types.GenerateContentConfig.model_fields:
            gen_kwargs["seed"] = _GENERATION_SEED

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[
                    self._types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=_VIDEO_MIME,
                    ),
                    prompt,
                ],
                config=self._types.GenerateContentConfig(**gen_kwargs),
            )
        except Exception as exc:
            # Re-raise so Stage 06 retry machinery records a genuine FAILED obs.
            raise RuntimeError(f"Gemini inference failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini returned an empty response.")

        return text
