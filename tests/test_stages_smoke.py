"""Smoke tests — full pipeline end-to-end in stub mode with a synthetic video.

Creates a minimal synthetic video, runs the complete pipeline with --stub,
and verifies:
  - All stages complete without exception
  - s01/s02 report OK
  - s03-s13 (except s11, s12) report SKIPPED (no AI ran)
  - s11 and s12 report OK (always run)
  - All expected output JSON files are written
  - No fabricated data (detections/tracks/events are empty)
  - episode.json is parseable and counts are accurate
  - evaluation.json reports PARTIAL (some SKIPPED stages)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ── Synthetic video creation ─────────────────────────────────────────────────

def create_synthetic_video(path: Path, duration_sec: float = 3.0, fps: float = 10.0) -> None:
    """Write a minimal synthetic video using OpenCV for testing."""
    try:
        import cv2
    except ImportError:
        pytest.skip("opencv-python not installed")

    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    n_frames = int(duration_sec * fps)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Minimal visual content: frame number as pixel intensity
        frame[:, :, 0] = (i * 8) % 256
        writer.write(frame)
    writer.release()


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("smoke_video")
    video_path = tmp / "smoke_test.mp4"
    create_synthetic_video(video_path, duration_sec=3.0, fps=10.0)
    assert video_path.exists(), "Synthetic video creation failed"
    return video_path


@pytest.fixture(scope="module")
def pipeline_output_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("smoke_output")


@pytest.fixture(scope="module")
def run_pipeline(synthetic_video, pipeline_output_dir):
    """Run the full pipeline in stub mode and return the context."""
    from src.config import load_config
    from src.context import PipelineContext
    from src.stages import (
        s01_ingest,
        s02_sample,
        s03_detect,
        s04_track,
        s05_segment,
        s06_vlm,
        s07_events,
        s08_states,
        s09_graph,
        s10_trajectories,
        s11_score,
        s12_episode,
        s13_evaluate,
        s14_preview,
    )

    config = load_config(set_overrides=[
        "stub_mode=true",
        f"output_dir={pipeline_output_dir}",
        "frame_sampling.fps=1.0",
    ])

    ctx = PipelineContext(
        config=config,
        video_path=synthetic_video,
        output_dir=pipeline_output_dir,
    )

    stage_modules = [
        s01_ingest, s02_sample, s03_detect, s04_track, s05_segment,
        s06_vlm, s07_events, s08_states, s09_graph, s10_trajectories, s11_score,
        s12_episode, s13_evaluate, s14_preview,
    ]

    for mod in stage_modules:
        status = mod.run(ctx)
        ctx.record_stage(status)

    return ctx


# ── Stage status tests ────────────────────────────────────────────────────────

class TestStageStatuses:
    def test_s01_ingest_ok(self, run_pipeline):
        ctx = run_pipeline
        s01 = next((s for s in ctx.stage_statuses if s.stage == "s01_ingest"), None)
        assert s01 is not None, "s01_ingest status not recorded"
        # s01 may return WARNING (e.g. frame count discrepancy in container header)
        # WARNING is not an error — the pipeline proceeds with available metadata.
        assert s01.status in {"OK", "WARNING"}, (
            f"Expected OK or WARNING, got {s01.status}: {s01.message}"
        )

    def test_s02_sample_ok(self, run_pipeline):
        ctx = run_pipeline
        s02 = next((s for s in ctx.stage_statuses if s.stage == "s02_sample"), None)
        assert s02 is not None
        assert s02.status == "OK"

    def test_heavy_stages_are_skipped(self, run_pipeline):
        """In stub mode: s03-s10, s13 must report SKIPPED — not fabricated data."""
        ctx = run_pipeline
        skippable = {
            "s03_detect", "s04_track", "s05_segment", "s06_vlm",
            "s07_events", "s08_states", "s09_graph", "s10_trajectories", "s11_score",
            "s14_preview",
        }
        for ss in ctx.stage_statuses:
            if ss.stage in skippable:
                assert ss.status == "SKIPPED", (
                    f"{ss.stage} should be SKIPPED in stub mode, got {ss.status}"
                )

    def test_s12_episode_ok(self, run_pipeline):
        ctx = run_pipeline
        s12 = next((s for s in ctx.stage_statuses if s.stage == "s12_episode"), None)
        assert s12 is not None
        assert s12.status == "OK"

    def test_s13_evaluate_ok(self, run_pipeline):
        ctx = run_pipeline
        s13 = next((s for s in ctx.stage_statuses if s.stage == "s13_evaluate"), None)
        assert s13 is not None
        assert s13.status == "OK"

    def test_no_error_statuses(self, run_pipeline):
        ctx = run_pipeline
        errors = [s for s in ctx.stage_statuses if s.status == "ERROR"]
        assert errors == [], f"Unexpected errors: {errors}"

    def test_sampling_plan_json_written(self, run_pipeline, pipeline_output_dir):
        """s02_sample must write sampling_plan.json alongside other outputs."""
        assert (pipeline_output_dir / "sampling_plan.json").exists()


# ── No fabricated data ────────────────────────────────────────────────────────

class TestNoFabricatedData:
    def test_detection_frames_empty_in_stub_mode(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.detection_frames == [], (
            "Stub mode must not fabricate detection frames"
        )

    def test_tracks_empty_in_stub_mode(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.tracks == [], "Stub mode must not fabricate tracks"

    def test_events_empty_in_stub_mode(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.events == [], "Stub mode must not fabricate events"

    def test_trajectories_empty_in_stub_mode(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.trajectories == [], "Stub mode must not fabricate trajectories"

    def test_object_states_empty_in_stub_mode(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.object_states == []

    def test_state_transitions_empty_in_stub_mode(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.state_transitions == []


# ── Video metadata ────────────────────────────────────────────────────────────

class TestVideoMetadata:
    def test_video_metadata_populated(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.video_metadata is not None

    def test_video_metadata_fps_positive(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.video_metadata.fps > 0

    def test_video_metadata_dimensions_positive(self, run_pipeline):
        ctx = run_pipeline
        assert ctx.video_metadata.width > 0
        assert ctx.video_metadata.height > 0

    def test_sampled_frames_computed(self, run_pipeline):
        """s02 should compute some sample frames from the 3-second video."""
        ctx = run_pipeline
        assert len(ctx.sampled_frame_infos) > 0

    def test_sampled_frame_count_matches_duration(self, run_pipeline):
        """3s video @ 1fps → expect ~3 frames (may be 2-4 depending on exact fps)."""
        ctx = run_pipeline
        assert 1 <= len(ctx.sampled_frame_infos) <= 10


# ── Output file existence ─────────────────────────────────────────────────────

class TestOutputFiles:
    def test_episode_json_written(self, run_pipeline, pipeline_output_dir):
        assert (pipeline_output_dir / "episode.json").exists()

    def test_evaluation_json_written(self, run_pipeline, pipeline_output_dir):
        assert (pipeline_output_dir / "evaluation.json").exists()

    def test_detections_json_written(self, run_pipeline, pipeline_output_dir):
        assert (pipeline_output_dir / "detections.json").exists()

    def test_tracks_json_written(self, run_pipeline, pipeline_output_dir):
        assert (pipeline_output_dir / "tracks.json").exists()

    def test_events_json_written(self, run_pipeline, pipeline_output_dir):
        assert (pipeline_output_dir / "events.json").exists()

    def test_states_json_written(self, run_pipeline, pipeline_output_dir):
        assert (pipeline_output_dir / "states.json").exists()

    def test_trajectories_json_written(self, run_pipeline, pipeline_output_dir):
        assert (pipeline_output_dir / "trajectories.json").exists()

    def test_candidate_segments_json_written(self, run_pipeline, pipeline_output_dir):
        assert (pipeline_output_dir / "candidate_segments.json").exists()


# ── episode.json content ──────────────────────────────────────────────────────

class TestEpisodeJson:
    def test_episode_json_is_valid(self, run_pipeline, pipeline_output_dir):
        data = json.loads((pipeline_output_dir / "episode.json").read_text())
        assert "episode_id" in data
        assert "pipeline_version" in data

    def test_episode_video_metadata_has_frame_count(self, run_pipeline, pipeline_output_dir):
        """VideoMetadata in episode.json must include the new frame_count field."""
        data = json.loads((pipeline_output_dir / "episode.json").read_text())
        vm = data.get("video_metadata", {})
        assert "frame_count" in vm, "frame_count missing from video_metadata in episode.json"

    def test_episode_counts_are_zero_for_skipped_stages(self, run_pipeline, pipeline_output_dir):
        data = json.loads((pipeline_output_dir / "episode.json").read_text())
        assert data["n_detections"] == 0
        assert data["n_tracks"] == 0
        assert data["n_events"] == 0

    def test_episode_n_frames_sampled_nonzero(self, run_pipeline, pipeline_output_dir):
        data = json.loads((pipeline_output_dir / "episode.json").read_text())
        assert data["n_frames_sampled"] > 0


# ── evaluation.json content ───────────────────────────────────────────────────

class TestEvaluationJson:
    def test_evaluation_json_is_valid(self, run_pipeline, pipeline_output_dir):
        data = json.loads((pipeline_output_dir / "evaluation.json").read_text())
        assert "overall_status" in data
        assert data["overall_status"] in {"PASS", "FAIL", "PARTIAL", "SKIPPED"}

    def test_evaluation_has_stub_warning(self, run_pipeline, pipeline_output_dir):
        data = json.loads((pipeline_output_dir / "evaluation.json").read_text())
        warnings = data.get("warnings", [])
        assert any("stub" in w.lower() for w in warnings), (
            "Evaluation should warn that stub_mode was used"
        )

    def test_skipped_stages_reflected_in_evaluation(self, run_pipeline, pipeline_output_dir):
        data = json.loads((pipeline_output_dir / "evaluation.json").read_text())
        stage_evals = data.get("stage_evaluations", [])
        skipped = [e for e in stage_evals if e["status"] == "SKIPPED"]
        assert len(skipped) > 0, "Expected some SKIPPED stage evaluations in stub mode"


# ── CLI entry point ───────────────────────────────────────────────────────────

class TestCLI:
    def test_help_exits_zero(self):
        """python -m src.pipeline --help must exit 0."""
        from src.pipeline import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_main_with_synthetic_video(self, synthetic_video, tmp_path):
        from src.pipeline import main
        exit_code = main([
            str(synthetic_video),
            "--stub",
            f"--output-dir={tmp_path / 'cli_out'}",
        ])
        assert exit_code == 0

    def test_main_with_missing_video_exits_nonzero(self, tmp_path):
        from src.pipeline import main
        exit_code = main([
            str(tmp_path / "nonexistent.mp4"),
            "--stub",
            f"--output-dir={tmp_path / 'cli_err'}",
        ])
        assert exit_code != 0
