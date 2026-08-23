"""Tests for the configuration system (src/config.py)."""
from __future__ import annotations

import pytest

from src.config import (
    PipelineConfig,
    _parse_set_override,
    load_config,
)


class TestDefaultConfig:
    def test_default_config_loads(self):
        cfg = load_config()
        assert isinstance(cfg, PipelineConfig)

    def test_frame_sampling_fps_is_1(self):
        """Spec default: 1 frame/sec ≈ 600 frames for a 10-minute video."""
        cfg = load_config()
        assert cfg.frame_sampling.fps == 1.0

    def test_frame_sampling_max_frames(self):
        cfg = load_config()
        assert cfg.frame_sampling.max_frames == 1200

    def test_segment_fps_is_configurable(self):
        """Candidate segments can be re-sampled at higher FPS."""
        cfg = load_config()
        assert cfg.frame_sampling.segment_fps > cfg.frame_sampling.fps

    def test_vlm_disabled_by_default(self):
        cfg = load_config()
        assert cfg.vlm.enabled is False

    def test_stub_mode_off_by_default(self):
        cfg = load_config()
        assert cfg.stub_mode is False

    def test_pose_disabled_by_default(self):
        cfg = load_config()
        assert cfg.pose.enabled is False

    def test_vlm_backend_no_provider_committed(self):
        """VLM backend must be LOCAL_MODEL or REMOTE_MODEL — no provider committed."""
        cfg = load_config()
        assert cfg.vlm.backend in {"LOCAL_MODEL", "REMOTE_MODEL"}

    def test_detector_model_is_configurable(self):
        """Detector model is a string, not hardcoded to a specific version."""
        cfg = load_config()
        assert isinstance(cfg.detector.model, str)
        assert len(cfg.detector.model) > 0


class TestYamlOverride:
    def test_yaml_overrides_fps(self, tmp_path):
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text("frame_sampling:\n  fps: 2.0\n")
        cfg = load_config(yaml_path=yaml_file)
        assert cfg.frame_sampling.fps == 2.0

    def test_yaml_enables_vlm(self, tmp_path):
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text("vlm:\n  enabled: true\n")
        cfg = load_config(yaml_path=yaml_file)
        assert cfg.vlm.enabled is True

    def test_yaml_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(yaml_path=tmp_path / "nonexistent.yaml")

    def test_yaml_overrides_output_dir(self, tmp_path):
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(f"output_dir: '{tmp_path / 'myout'}'\n")
        cfg = load_config(yaml_path=yaml_file)
        assert cfg.output_dir == tmp_path / "myout"


class TestSetOverrides:
    def test_set_override_fps(self):
        cfg = load_config(set_overrides=["frame_sampling.fps=3.0"])
        assert cfg.frame_sampling.fps == 3.0

    def test_set_override_bool_true(self):
        cfg = load_config(set_overrides=["stub_mode=true"])
        assert cfg.stub_mode is True

    def test_set_override_bool_false(self):
        cfg = load_config(set_overrides=["stub_mode=false"])
        assert cfg.stub_mode is False

    def test_set_override_int(self):
        cfg = load_config(set_overrides=["frame_sampling.max_frames=500"])
        assert cfg.frame_sampling.max_frames == 500

    def test_set_override_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            _parse_set_override("no_equals_sign")

    def test_multiple_set_overrides(self):
        cfg = load_config(set_overrides=[
            "frame_sampling.fps=5.0",
            "stub_mode=true",
            "detector.confidence=0.5",
        ])
        assert cfg.frame_sampling.fps == 5.0
        assert cfg.stub_mode is True
        assert cfg.detector.confidence == 0.5


class TestParseSetOverride:
    def test_parses_bool_true(self):
        _, val = _parse_set_override("stub_mode=true")
        assert val is True

    def test_parses_bool_false(self):
        _, val = _parse_set_override("stub_mode=False")
        assert val is False

    def test_parses_int(self):
        _, val = _parse_set_override("frame_sampling.max_frames=800")
        assert val == 800
        assert isinstance(val, int)

    def test_parses_float(self):
        _, val = _parse_set_override("frame_sampling.fps=1.5")
        assert val == 1.5
        assert isinstance(val, float)

    def test_parses_str(self):
        _, val = _parse_set_override("detector.model=yolov8s")
        assert val == "yolov8s"
        assert isinstance(val, str)

    def test_returns_dotted_key(self):
        key, _ = _parse_set_override("frame_sampling.fps=1.0")
        assert key == "frame_sampling.fps"
