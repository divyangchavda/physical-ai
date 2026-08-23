"""Remote VLM implementation via HTTP."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from src.interfaces.vlm import VisionLanguageModel


class RemoteVLM(VisionLanguageModel):
    """Provider-neutral Remote VLM backend.
    
    Demonstrates the architecture for calling an external API.
    For MVP Phase 5, this acts as a stub representing a generic HTTP client.
    """

    def __init__(self, model_name: str, api_base_url: str | None = None, timeout_sec: float = 30.0):
        self._model_name = model_name
        self.api_base_url = api_base_url
        self.timeout_sec = timeout_sec

    @property
    def backend(self) -> Literal["LOCAL_MODEL", "REMOTE_MODEL"]:
        return "REMOTE_MODEL"

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
        """Mock remote call."""
        
        if self._model_name == "error_remote":
            raise RuntimeError("Connection timed out")
            
        if self._model_name == "malformed_remote":
            return "not json at all"
            
        mock_resp = {
            "actor": "person",
            "active_hand": "UNKNOWN",
            "objects": [],
            "raw_action": "UNKNOWN",
            "start_time_sec": None,
            "end_time_sec": None,
            "state_change": None,
            "visible_facts": None,
            "inference": None,
            "evidence": "nothing clear",
            "uncertainty": "occluded",
            "confidence": 0.5
        }
        return json.dumps(mock_resp)
