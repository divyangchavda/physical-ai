"""Pipeline configuration system.

Configuration layers (last wins):
  1. config/default.yaml  — shipped defaults
  2. user YAML (--config) — user overrides
  3. CLI --set             — per-run overrides

All values are validated by Pydantic at load time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------


class FrameSamplingConfig(BaseModel):
    fps: float = 1.0           # spec default: 1 fps ≈ 600 frames/10-min video
    max_frames: int = 1200     # hard cap — prevents OOM on long recordings
    segment_fps: float = 5.0   # fps for re-sampling candidate segments for VLM
    every_n_frames: int | None = None  # if set, sample every N frames instead of by FPS


class DetectorConfig(BaseModel):
    backend: str = "yolov8"    # "yolov8" | "yolo-world" | "groundingdino" | "groundingdino_hf"
    model: str = "yolov8n"     # configurable; ObjectDetector ABC is model-agnostic
    confidence: float = 0.35
    nms_iou: float = 0.45
    device: str = "auto"       # "auto" → resolved to cuda/cpu at runtime
    vocabulary: list[str] | None = None  # YOLO-World custom vocabulary
    text_prompt: str | None = None  # GroundingDINO text prompt (e.g., "person . box . table .")
    text_threshold: float = 0.25  # GroundingDINO text matching threshold
    # GroundingDINO weights/config location. None → ~/GroundingDINO/ defaults.
    # Set these to read weights from a read-only mount (e.g. Kaggle /kaggle/input/...).
    model_checkpoint: str | None = None
    config_file: str | None = None


class TrackerConfig(BaseModel):
    backend: str = "bytetrack"  # "bytetrack" | "kalman_sparse"
    iou_threshold: float = 0.20  # For kalman_sparse
    max_age: int = 15  # For kalman_sparse
    min_hits: int = 1  # For kalman_sparse


class VLMConfig(BaseModel):
    enabled: bool = False
    backend: str = "LOCAL_MODEL"  # "LOCAL_MODEL" | "REMOTE_MODEL" | "GEMINI"
    model_name: str = "stub"
    api_base_url: str | None = None
    timeout_sec: float = 60.0
    max_retries: int = 2


class EventExtractionConfig(BaseModel):
    enabled: bool = True


class StateExtractionConfig(BaseModel):
    enabled: bool = True


class GraphExtractionConfig(BaseModel):
    enabled: bool = True


class PoseConfig(BaseModel):
    enabled: bool = False  # optional; disabled by default


class EventConfig(BaseModel):
    # Quality engine thresholds — used by s10_score only.
    # Raw predictions (action + confidence + source) are ALWAYS preserved.
    auto_accept_threshold: float = 0.70
    human_review_threshold: float = 0.40


class EpisodeConfig(BaseModel):
    enabled: bool = True
    max_event_gap_sec: float = 2.0
    require_shared_entity: bool = True


class SegmentProximityConfig(BaseModel):
    iou_threshold: float = 0.05
    gap_threshold_normalized: float = 0.2


class SegmentMovementConfig(BaseModel):
    threshold: float = 0.05
    window_frames: int = 5


class SegmentConfig(BaseModel):
    person_classes: list[str] = ["person"]
    proximity: SegmentProximityConfig = Field(default_factory=SegmentProximityConfig)
    movement: SegmentMovementConfig = Field(default_factory=SegmentMovementConfig)
    temporal_padding_sec: float = 2.0
    merge_gap_sec: float = 1.0


# ---------------------------------------------------------------------------
# Root configuration model
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Root configuration object for a single pipeline run."""
    output_dir: Path = Path("output")
    device: str = "auto"
    stub_mode: bool = False

    frame_sampling: FrameSamplingConfig = Field(default_factory=FrameSamplingConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    event_extraction: EventExtractionConfig = Field(default_factory=EventExtractionConfig)
    state_extraction: StateExtractionConfig = Field(default_factory=StateExtractionConfig)
    graph_extraction: GraphExtractionConfig = Field(default_factory=GraphExtractionConfig)
    event: EventConfig = Field(default_factory=EventConfig)
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *updates* into *base* (mutates and returns base)."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _parse_set_override(override: str) -> tuple[str, Any]:
    """Parse ``key.sub=value`` into ``(dotted_key, typed_value)``.

    Attempts bool → int → float casts before falling back to str.
    """
    if "=" not in override:
        raise ValueError(
            f"Invalid --set override (expected key=value): {override!r}"
        )
    key, _, raw_value = override.partition("=")
    key = key.strip()
    raw_value = raw_value.strip()

    # Bool
    if raw_value.lower() in {"true", "false"}:
        return key, raw_value.lower() == "true"
    # Int
    try:
        return key, int(raw_value)
    except ValueError:
        pass
    # Float
    try:
        return key, float(raw_value)
    except ValueError:
        pass
    # Fallback to str
    return key, raw_value


def _apply_dotted_overrides(
    data: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    """Apply ``key.sub=value`` strings to *data* (mutates in place)."""
    for override in overrides:
        key, value = _parse_set_override(override)
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return data


def load_config(
    yaml_path: Path | None = None,
    set_overrides: list[str] | None = None,
) -> PipelineConfig:
    """Load and validate the pipeline configuration.

    Args:
        yaml_path: optional user YAML file (overrides defaults).
        set_overrides: list of ``key.sub=value`` strings (overrides YAML).

    Returns:
        Validated :class:`PipelineConfig`.
    """
    # Layer 1: shipped defaults
    default_yaml = Path(__file__).parent.parent / "config" / "default.yaml"
    data: dict[str, Any] = {}
    if default_yaml.exists():
        with open(default_yaml, encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            _deep_update(data, loaded)

    # Layer 2: user YAML
    if yaml_path is not None:
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, encoding="utf-8") as f:
            user_data = yaml.safe_load(f) or {}
            _deep_update(data, user_data)

    # Layer 3: CLI --set overrides
    if set_overrides:
        _apply_dotted_overrides(data, set_overrides)

    return PipelineConfig.model_validate(data)
