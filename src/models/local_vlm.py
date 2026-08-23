"""Local VLM implementation stub."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from src.interfaces.vlm import VisionLanguageModel


class LocalVLM(VisionLanguageModel):
    """Stub implementation for a Local VLM backend (e.g. Qwen2-VL).
    
    This acts as a placeholder adapter. It returns a mock successful response
    or can be configured to fail/timeout for testing purposes.
    """

    def __init__(self, model_name: str = "stub_local"):
        self._model_name = model_name

    @property
    def backend(self) -> Literal["LOCAL_MODEL", "REMOTE_MODEL"]:
        return "LOCAL_MODEL"

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
        """Returns a predefined mock JSON response string for local testing."""
        # For a real implementation, we would call extract_frames here and pass
        # the frames to the local model instance.
        
        if "return_malformed" in prompt:
            return "```json\n{malformed_json\n```"
            
        if "return_missing" in prompt:
            return '{"actor": "person"}'
            
        if "return_unknown" in prompt:
            return json.dumps({
                "actor": "UNKNOWN",
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
            })
            
        if "return_out_of_bounds" in prompt:
            return json.dumps({
                "actor": "person",
                "active_hand": "UNKNOWN",
                "objects": [],
                "raw_action": "UNKNOWN",
                "start_time_sec": 999.0, # Will be added to start_sec, out of bounds
                "end_time_sec": 1000.0,
                "state_change": None,
                "visible_facts": None,
                "inference": None,
                "evidence": "nothing clear",
                "uncertainty": "occluded",
                "confidence": 0.5
            })
        
        # We mock a successful JSON response matching the prompt schema.
        mock_resp = {
            "actor": "person in blue shirt",
            "active_hand": "RIGHT",
            "objects": ["white cup"],
            "raw_action": "picked up the cup",
            "start_time_sec": 0.5,
            "end_time_sec": 1.5,
            "state_change": "cup is now held in hand",
            "visible_facts": "person reached toward cup",
            "inference": "intention to drink",
            "evidence": "person's right hand closed around the cup and lifted it",
            "uncertainty": "cannot see if cup is empty",
            "confidence": 0.85
        }
        
        return json.dumps(mock_resp)
