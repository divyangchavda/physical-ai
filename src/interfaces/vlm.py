"""VisionLanguageModel abstract base class.

The pipeline is identical regardless of which backend (LOCAL_MODEL or
REMOTE_MODEL) is active. No provider SDK is imported at this level.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal


class VisionLanguageModel(ABC):
    """Interface for all VLM backends.

    Backend categories:
      - ``"LOCAL_MODEL"`` — runs entirely on local hardware; no network calls.
      - ``"REMOTE_MODEL"`` — calls an external inference endpoint.

    Model selection and provider are intentionally NOT fixed at this level.
    They are configured when the VLM stage is implemented.

    Contract:
      - ``analyze_segment()`` returns a raw dict (may be empty on failure).
      - The calling stage (s06_vlm) validates and converts the dict to schema.
      - Never fabricate physical information.
      - Be honest about uncertainty — return {} rather than guessing.
    """

    @property
    @abstractmethod
    def backend(self) -> Literal["LOCAL_MODEL", "REMOTE_MODEL"]:
        """Which backend category this implementation belongs to."""

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
