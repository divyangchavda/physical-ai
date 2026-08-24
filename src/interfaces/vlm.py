"""VisionLanguageModel abstract base class.

The pipeline is identical regardless of which backend is active. No provider
SDK is imported at this level.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal


class VisionLanguageModel(ABC):
    """Interface for all VLM backends.

    Backend identifiers:
      - ``"LOCAL_MODEL"`` — runs entirely on local hardware; no network calls.
      - ``"REMOTE_MODEL"`` — calls a generic external inference endpoint.
      - ``"GEMINI"`` — calls the Google Gemini API. Remote, but named
        separately: this string is the provenance recorded on every
        observation and, via s07, the ``source`` on every shipped event.
        Returning ``"REMOTE_MODEL"`` here made Gemini output indistinguishable
        from any other remote endpoint in the delivered data.

    Model selection is intentionally NOT fixed at this level.

    Contract:
      - ``analyze_segment()`` returns a raw dict (may be empty on failure).
      - The calling stage (s06_vlm) validates and converts the dict to schema.
      - Never fabricate physical information.
      - Be honest about uncertainty — return {} rather than guessing.
    """

    @property
    @abstractmethod
    def backend(self) -> Literal["LOCAL_MODEL", "REMOTE_MODEL", "GEMINI"]:
        """Which backend this implementation belongs to."""

    @abstractmethod
    def analyze_segment(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        prompt: str,
    ) -> str:
        """Analyze a video segment and return raw string output.

        Args:
            video_path: absolute path to the source video file.
            start_sec: segment start position in seconds.
            end_sec: segment end position in seconds.
            prompt: task-specific prompt (constructed by the calling stage).

        Returns:
            Raw string response from the model.
            The calling stage handles JSON parsing and validation.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier for logs and provenance."""
