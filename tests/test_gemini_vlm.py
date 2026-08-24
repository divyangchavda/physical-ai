"""Unit tests for GeminiVLM adapter.

All tests are fully deterministic — no real API calls are made.
The google.genai SDK is patched via unittest.mock.

Running these tests never:
  - requires GEMINI_API_KEY
  - makes a network connection
  - uploads a file
  - calls a real model
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gemini_vlm(model_name="gemini-2.5-flash", timeout_sec=30.0):
    """Return a GeminiVLM with the Gemini SDK fully mocked."""
    mock_client = MagicMock()
    mock_types = MagicMock()

    with (
        patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-abc"}),
        patch("src.models.gemini_vlm.cv2"),  # avoid needing a real video
        patch("src.models.gemini_vlm.GeminiVLM.__init__", autospec=True) as mock_init,
    ):
        from src.models.gemini_vlm import GeminiVLM

        def side_effect(self, model_name=model_name, timeout_sec=timeout_sec):
            self._client = mock_client
            self._types = mock_types
            self._model_name = model_name
            self._timeout_sec = timeout_sec

        mock_init.side_effect = side_effect
        vlm = GeminiVLM(model_name=model_name, timeout_sec=timeout_sec)

    return vlm, mock_client, mock_types


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestGeminiVLMConstruction:
    def test_raises_without_api_key(self):
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with patch.dict(os.environ, env, clear=True), pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
            import src.models.gemini_vlm as mod
            mod.GeminiVLM()

    def test_raises_if_google_genai_missing(self):
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "key"}),
            patch("builtins.__import__", side_effect=ImportError("no module")),
            pytest.raises((ImportError, Exception)),
        ):
            from importlib import reload

            import src.models.gemini_vlm as mod
            reload(mod)
            mod.GeminiVLM()

    def test_backend_identifies_the_provider(self):
        """Was "REMOTE_MODEL", which RemoteVLM also returns.

        s07 builds every event's source as f"vlm:{obs.backend.lower()}", so
        that made Gemini output indistinguishable from any other remote
        endpoint in the delivered data.
        """
        vlm, _, _ = _make_gemini_vlm()
        assert vlm.backend == "GEMINI"

        from src.models.remote_vlm import RemoteVLM

        assert RemoteVLM(model_name="x").backend == "REMOTE_MODEL"

    def test_model_name_stored(self):
        vlm, _, _ = _make_gemini_vlm(model_name="gemini-2.5-flash")
        assert vlm.model_name == "gemini-2.5-flash"

    def test_custom_model_name(self):
        vlm, _, _ = _make_gemini_vlm(model_name="gemini-1.5-flash")
        assert vlm.model_name == "gemini-1.5-flash"


# ---------------------------------------------------------------------------
# analyze_segment — success path
# ---------------------------------------------------------------------------

MOCK_PAYLOAD = {
    "actor": "person in blue shirt",
    "active_hand": "RIGHT",
    "objects": ["white cup"],
    "raw_action": "picked up the cup",
    "state_change": "cup lifted from table",
    "visible_facts": "hand reached down and gripped cup",
    "inference": "deliberate retrieval action",
    "uncertainty": "cannot see if cup is full",
    "confidence": 0.9,
    "start_time_sec": 0.5,
    "end_time_sec": 1.5,
}


def _mock_successful_vlm(vlm, mock_client, mock_types):
    """Configure mock_client so analyze_segment returns a valid JSON string."""
    # Fake uploaded file that is immediately ACTIVE
    mock_file = SimpleNamespace(
        name="files/fake-id",
        uri="https://fake-uri/files/fake-id",
        state=SimpleNamespace(name="ACTIVE"),
    )
    mock_client.files.upload.return_value = mock_file
    mock_client.files.get.return_value = mock_file

    # Fake response
    mock_response = SimpleNamespace(text=json.dumps(MOCK_PAYLOAD))
    mock_client.models.generate_content.return_value = mock_response

    # Mock Part.from_uri
    mock_types.Part.from_uri.return_value = MagicMock()
    mock_types.UploadFileConfig.return_value = MagicMock()
    mock_types.GenerateContentConfig.return_value = MagicMock()


def _make_dummy_video(tmp_path: Path) -> Path:
    """Create a tiny readable dummy video file (just a file — cv2 is mocked)."""
    p = tmp_path / "dummy.mp4"
    p.write_bytes(b"\x00" * 100)
    return p


class TestAnalyzeSegmentSuccess:
    def test_returns_string(self, tmp_path):
        vlm, mock_client, mock_types = _make_gemini_vlm()
        _mock_successful_vlm(vlm, mock_client, mock_types)
        video = _make_dummy_video(tmp_path)

        with patch.object(vlm, "_extract_clip", return_value=tmp_path / "clip.mp4"):
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 10)
            result = vlm.analyze_segment(video, 1.0, 4.0, "test prompt")

        assert isinstance(result, str)

    def test_returns_valid_json(self, tmp_path):
        vlm, mock_client, mock_types = _make_gemini_vlm()
        _mock_successful_vlm(vlm, mock_client, mock_types)
        video = _make_dummy_video(tmp_path)

        with patch.object(vlm, "_extract_clip", return_value=tmp_path / "clip.mp4"):
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 10)
            result = vlm.analyze_segment(video, 1.0, 4.0, "test prompt")

        data = json.loads(result)
        assert data["raw_action"] == "picked up the cup"
        assert data["confidence"] == 0.9

    def test_uploads_then_deletes_file(self, tmp_path):
        vlm, mock_client, mock_types = _make_gemini_vlm()
        _mock_successful_vlm(vlm, mock_client, mock_types)
        video = _make_dummy_video(tmp_path)

        with patch.object(vlm, "_extract_clip", return_value=tmp_path / "clip.mp4"):
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 10)
            vlm.analyze_segment(video, 1.0, 4.0, "prompt")

        mock_client.files.upload.assert_called_once()
        mock_client.files.delete.assert_called_once_with(name="files/fake-id")

    def test_correct_segment_bounds_passed(self, tmp_path):
        """The start_sec and end_sec must reach _extract_clip unmodified."""
        vlm, mock_client, mock_types = _make_gemini_vlm()
        _mock_successful_vlm(vlm, mock_client, mock_types)
        video = _make_dummy_video(tmp_path)

        captured = {}

        def fake_extract(vp, start, end):
            captured["start"] = start
            captured["end"] = end
            p = tmp_path / "clip.mp4"
            p.write_bytes(b"\x00" * 10)
            return p

        with patch.object(vlm, "_extract_clip", side_effect=fake_extract):
            vlm.analyze_segment(video, 12.5, 16.0, "prompt")

        assert captured["start"] == 12.5
        assert captured["end"] == 16.0

    def test_empty_response_raises(self, tmp_path):
        vlm, mock_client, mock_types = _make_gemini_vlm()
        _mock_successful_vlm(vlm, mock_client, mock_types)
        # Override response to empty string
        mock_client.models.generate_content.return_value = SimpleNamespace(text="")
        video = _make_dummy_video(tmp_path)

        with (
            patch.object(vlm, "_extract_clip", return_value=tmp_path / "clip.mp4"),
            pytest.raises((ValueError, RuntimeError)),
        ):
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 10)
            vlm.analyze_segment(video, 1.0, 4.0, "prompt")


# ---------------------------------------------------------------------------
# analyze_segment — failure paths
# ---------------------------------------------------------------------------

class TestAnalyzeSegmentFailures:
    def test_api_error_raises(self, tmp_path):
        """RuntimeError from the API must propagate so Stage 06 records FAILED."""
        vlm, mock_client, mock_types = _make_gemini_vlm()
        mock_file = SimpleNamespace(
            name="files/fake",
            uri="https://fake",
            state=SimpleNamespace(name="ACTIVE"),
        )
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        mock_client.models.generate_content.side_effect = RuntimeError("503 overloaded")
        mock_types.Part.from_uri.return_value = MagicMock()
        mock_types.UploadFileConfig.return_value = MagicMock()
        mock_types.GenerateContentConfig.return_value = MagicMock()
        video = _make_dummy_video(tmp_path)

        with (
            patch.object(vlm, "_extract_clip", return_value=tmp_path / "clip.mp4"),
            pytest.raises(RuntimeError, match="503 overloaded"),
        ):
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 10)
            vlm.analyze_segment(video, 1.0, 4.0, "prompt")

    def test_upload_error_raises(self, tmp_path):
        vlm, mock_client, mock_types = _make_gemini_vlm()
        mock_client.files.upload.side_effect = ConnectionError("network failure")
        mock_types.UploadFileConfig.return_value = MagicMock()
        video = _make_dummy_video(tmp_path)

        with (
            patch.object(vlm, "_extract_clip", return_value=tmp_path / "clip.mp4"),
            pytest.raises(ConnectionError),
        ):
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 10)
            vlm.analyze_segment(video, 1.0, 4.0, "prompt")

    def test_file_deleted_even_on_inference_failure(self, tmp_path):
        """Ensure File API cleanup happens even when inference raises."""
        vlm, mock_client, mock_types = _make_gemini_vlm()
        mock_file = SimpleNamespace(
            name="files/cleanup-me",
            uri="https://fake/cleanup-me",
            state=SimpleNamespace(name="ACTIVE"),
        )
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file
        mock_client.models.generate_content.side_effect = RuntimeError("boom")
        mock_types.Part.from_uri.return_value = MagicMock()
        mock_types.UploadFileConfig.return_value = MagicMock()
        mock_types.GenerateContentConfig.return_value = MagicMock()
        video = _make_dummy_video(tmp_path)

        with (
            patch.object(vlm, "_extract_clip", return_value=tmp_path / "clip.mp4"),
            pytest.raises(RuntimeError),
        ):
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 10)
            vlm.analyze_segment(video, 1.0, 4.0, "prompt")

        mock_client.files.delete.assert_called_once_with(name="files/cleanup-me")

    def test_processing_timeout_raises(self, tmp_path):
        """File stuck in PROCESSING state should raise TimeoutError."""
        vlm, mock_client, mock_types = _make_gemini_vlm()
        mock_file = SimpleNamespace(
            name="files/stuck",
            uri="https://fake/stuck",
            state=SimpleNamespace(name="PROCESSING"),
        )
        mock_client.files.upload.return_value = mock_file
        mock_client.files.get.return_value = mock_file  # never transitions
        mock_types.UploadFileConfig.return_value = MagicMock()

        with (
            patch.object(vlm, "_extract_clip", return_value=tmp_path / "clip.mp4"),
            patch("src.models.gemini_vlm.time.sleep"),  # don't actually sleep
        ):
            (tmp_path / "clip.mp4").write_bytes(b"\x00" * 10)
            with pytest.raises(TimeoutError):
                vlm._wait_for_active(mock_file, poll_interval=0.01, max_wait=0.02)


# ---------------------------------------------------------------------------
# Stage 06 integration — GeminiVLM wired via config
# ---------------------------------------------------------------------------

class TestS06GeminiIntegration:
    """Verify Stage 06 correctly instantiates and calls GeminiVLM."""

    def _make_ctx(self, tmp_path):
        from src.config import PipelineConfig
        from src.context import PipelineContext
        from src.schema.segment import CandidateSegment

        config = PipelineConfig(stub_mode=False)
        config.vlm.enabled = True
        config.vlm.backend = "GEMINI"
        config.vlm.model_name = "gemini-2.5-flash"
        ctx = PipelineContext(
            video_path=tmp_path / "dummy.mp4",
            output_dir=tmp_path / "output",
            config=config,
        )
        (tmp_path / "dummy.mp4").write_bytes(b"\x00" * 100)
        ctx.candidate_segments = [
            CandidateSegment(
                segment_id="seg_001",
                start_frame=0,
                end_frame=30,
                start_sec=0.0,
                end_sec=3.0,
                trigger_reason="test",
            )
        ]
        return ctx

    def test_gemini_backend_selected(self, tmp_path):
        from src.stages import s06_vlm

        ctx = self._make_ctx(tmp_path)

        mock_vlm = MagicMock()
        mock_vlm.backend = "REMOTE_MODEL"
        mock_vlm.model_name = "gemini-2.5-flash"
        mock_vlm.analyze_segment.return_value = json.dumps({
            "actor": "person",
            "active_hand": "RIGHT",
            "objects": ["box"],
            "raw_action": "opened the box",
            "state_change": "box is now open",
            "visible_facts": "hands pushed lid up",
            "inference": "deliberate action",
            "uncertainty": "none",
            "confidence": 0.85,
            "start_time_sec": 0.5,
            "end_time_sec": 2.5,
            "evidence": "visible hand motion",
        })

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("src.stages.s06_vlm.GeminiVLM", return_value=mock_vlm),
        ):
            status = s06_vlm.run(ctx)

        assert status.status == "OK"
        assert len(ctx.vlm_observations) == 1
        obs = ctx.vlm_observations[0]
        assert obs.status.value == "SUCCESS"
        assert obs.backend == "REMOTE_MODEL"
        assert obs.raw_action == "opened the box"
        assert obs.confidence == 0.85

    def test_gemini_api_failure_records_failed_obs(self, tmp_path):
        from src.stages import s06_vlm

        ctx = self._make_ctx(tmp_path)

        mock_vlm = MagicMock()
        mock_vlm.backend = "REMOTE_MODEL"
        mock_vlm.model_name = "gemini-2.5-flash"
        mock_vlm.analyze_segment.side_effect = RuntimeError("rate limit exceeded")

        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("src.stages.s06_vlm.GeminiVLM", return_value=mock_vlm),
        ):
            status = s06_vlm.run(ctx)

        assert status.status == "OK"  # Stage itself completes
        obs = ctx.vlm_observations[0]
        assert obs.status.value == "FAILED"
        assert "rate limit exceeded" in obs.error_reason
