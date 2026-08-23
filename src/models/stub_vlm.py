"""Stub VLM backends — zero-dependency, used in tests and stub-mode runs.

Both stubs return empty dicts. Neither fabricates physical information.
These are the correct stub behaviours for disabled/skipped VLM analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from src.interfaces.vlm import VisionLanguageModel


class StubLocalVLM(VisionLanguageModel):
    """Stub local VLM backend.

    Used when ``vlm.enabled=False`` or ``stub_mode=True``.
    Returns an empty dict — never fabricates physical information.
    """

    @property
    def backend(self) -> Literal["LOCAL_MODEL", "REMOTE_MODEL"]:
        return "LOCAL_MODEL"

    def analyze_segment(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        prompt: str,
    ) -> dict:
        return {}  # honest: no analysis performed, no data fabricated

    @property
    def model_name(self) -> str:
        return "stub_local"


class StubRemoteVLM(VisionLanguageModel):
    """Stub remote VLM backend.

    Used when ``vlm.enabled=False`` or ``stub_mode=True``.
    Returns an empty dict — never fabricates physical information.
    """

    @property
    def backend(self) -> Literal["LOCAL_MODEL", "REMOTE_MODEL"]:
        return "REMOTE_MODEL"

    def analyze_segment(
        self,
        video_path: Path,
        start_sec: float,
        end_sec: float,
        prompt: str,
    ) -> dict:
        return {}  # honest: no analysis performed, no data fabricated

    @property
    def model_name(self) -> str:
        return "stub_remote"
