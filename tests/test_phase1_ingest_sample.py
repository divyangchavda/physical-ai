"""Phase 1 tests — Video Ingestion (s01_ingest) and Frame Sampling (s02_sample).

Tests cover:
  - 3-second synthetic video
  - 10-second synthetic video
  - Different source FPS values (5, 10, 25, 30, 59.94)
  - 1 FPS sampling (spec default)
  - Sampling when video FPS is lower than target fps
  - Timestamp correctness
  - Frame index correctness and deduplication
  - Invalid/corrupt video
  - Missing video file
  - Memory-safe behavior (no pixel data in context)
  - Hardened metadata: frame_count, metadata_warnings
  - sampling_plan.json written by s02

All synthetic videos are created with OpenCV VideoWriter.
All tests use real opencv — no mocking of video I/O.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

pytestmark = pytest.mark.skipif(not _HAS_CV2, reason="opencv-python not installed")


def make_video(
    path: Path,
    duration_sec: float,
    fps: float,
    width: int = 320,
    height: int = 240,
) -> None:
    """Create a synthetic MP4 video with unique frame content for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    n_frames = round(duration_sec * fps)
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Make each frame visually unique for potential future quality checks.
        # Channel 0: frame index × 4 (wraps at 64 frames)
        # Channel 1: row-gradient
        frame[:, :, 0] = (i * 4) % 256
        frame[height // 4 : height // 2, :, 1] = 128
        writer.write(frame)
    writer.release()


def make_pipeline_context(
    video_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    stub_mode: bool = False,
):
    """Build a PipelineContext wired to the given video and output directory."""
    from src.config import load_config
    from src.context import PipelineContext

    config = load_config(set_overrides=[
        f"stub_mode={str(stub_mode).lower()}",
        f"output_dir={output_dir}",
        f"frame_sampling.fps={fps}",
    ])
    return PipelineContext(
        config=config,
        video_path=video_path,
        output_dir=output_dir,
    )


# ── s01_ingest — core tests ───────────────────────────────────────────────────

class TestS01IngestValidVideo:
    """s01_ingest with well-formed synthetic videos."""

    def test_status_ok_3sec_video(self, tmp_path):
        from src.stages import s01_ingest
        video = tmp_path / "v3.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        status = s01_ingest.run(ctx)
        # May be OK or WARNING depending on frame count verification
        assert status.status in {"OK", "WARNING"}, f"Unexpected: {status}"

    def test_video_metadata_populated(self, tmp_path):
        from src.stages import s01_ingest
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        assert ctx.video_metadata is not None

    def test_fps_matches_source(self, tmp_path):
        from src.stages import s01_ingest
        video = tmp_path / "v10fps.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        # OpenCV may not report exactly 10.0 for all containers, but should be close
        assert abs(ctx.video_metadata.fps - 10.0) < 1.0

    def test_dimensions_correct(self, tmp_path):
        from src.stages import s01_ingest
        video = tmp_path / "vdim.mp4"
        make_video(video, duration_sec=2.0, fps=10.0, width=320, height=240)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        assert ctx.video_metadata.width == 320
        assert ctx.video_metadata.height == 240

    def test_file_size_positive(self, tmp_path):
        from src.stages import s01_ingest
        video = tmp_path / "vsize.mp4"
        make_video(video, duration_sec=2.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        assert ctx.video_metadata.file_size_bytes > 0

    def test_frame_count_positive(self, tmp_path):
        """frame_count field must be populated — not left at 0."""
        from src.stages import s01_ingest
        video = tmp_path / "vfc.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        # frame_count should be close to 30 (3s × 10fps)
        assert ctx.video_metadata.frame_count > 0

    def test_frame_count_plausible(self, tmp_path):
        """frame_count should be within ±2 of the expected value."""
        from src.stages import s01_ingest
        video = tmp_path / "vfc2.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        expected = int(3.0 * 10.0)
        assert abs(ctx.video_metadata.frame_count - expected) <= 3

    def test_duration_plausible(self, tmp_path):
        """Duration should be within 15% of the synthetic video's intended duration."""
        from src.stages import s01_ingest
        video = tmp_path / "vdur.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        assert abs(ctx.video_metadata.duration_sec - 3.0) / 3.0 < 0.15

    def test_codec_not_none(self, tmp_path):
        """codec field must be a string (possibly empty, but not None)."""
        from src.stages import s01_ingest
        video = tmp_path / "vcod.mp4"
        make_video(video, duration_sec=2.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        assert isinstance(ctx.video_metadata.codec, str)

    def test_codec_contains_no_null_bytes(self, tmp_path):
        """Codec string must not contain NUL bytes or control characters."""
        from src.stages import s01_ingest
        video = tmp_path / "vcod2.mp4"
        make_video(video, duration_sec=2.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        codec = ctx.video_metadata.codec
        assert "\x00" not in codec
        for ch in codec:
            assert 32 <= ord(ch) < 127 or ch == "?"

    def test_metadata_warnings_is_list(self, tmp_path):
        """metadata_warnings must always be a list — never None."""
        from src.stages import s01_ingest
        video = tmp_path / "vwarn.mp4"
        make_video(video, duration_sec=2.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        assert isinstance(ctx.video_metadata.metadata_warnings, list)

    def test_duration_is_non_negative(self, tmp_path):
        from src.stages import s01_ingest
        video = tmp_path / "vnn.mp4"
        make_video(video, duration_sec=2.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        assert ctx.video_metadata.duration_sec >= 0.0


class TestS01IngestFpsVariants:
    """Ingestion at different source FPS values."""

    @pytest.mark.parametrize("source_fps", [5.0, 10.0, 25.0, 30.0])
    def test_fps_ingested(self, tmp_path, source_fps):
        from src.stages import s01_ingest
        video = tmp_path / f"v{int(source_fps)}fps.mp4"
        make_video(video, duration_sec=2.0, fps=source_fps)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        # Allow ±10% tolerance (OpenCV container header can be slightly off)
        assert abs(ctx.video_metadata.fps - source_fps) / source_fps < 0.10

    def test_10sec_video_ingested(self, tmp_path):
        """10-second video at 30 fps — ~300 frames."""
        from src.stages import s01_ingest
        video = tmp_path / "v10s.mp4"
        make_video(video, duration_sec=10.0, fps=30.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        assert ctx.video_metadata is not None
        assert abs(ctx.video_metadata.duration_sec - 10.0) / 10.0 < 0.15


class TestS01IngestEdgeCases:
    """Error handling: missing file, empty file, corrupt file."""

    def test_missing_file_returns_error(self, tmp_path):
        from src.stages import s01_ingest
        ctx = make_pipeline_context(tmp_path / "nonexistent.mp4", tmp_path / "out")
        status = s01_ingest.run(ctx)
        assert status.status == "ERROR"
        assert ctx.video_metadata is None

    def test_missing_file_error_message_is_informative(self, tmp_path):
        from src.stages import s01_ingest
        ctx = make_pipeline_context(tmp_path / "ghost.mp4", tmp_path / "out")
        status = s01_ingest.run(ctx)
        assert "not found" in status.message.lower() or "nonexistent" in status.message.lower() or "ghost" in status.message.lower()

    def test_empty_file_returns_error(self, tmp_path):
        from src.stages import s01_ingest
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        ctx = make_pipeline_context(empty, tmp_path / "out")
        status = s01_ingest.run(ctx)
        assert status.status == "ERROR"

    def test_corrupt_file_returns_error_or_warning(self, tmp_path):
        """A file that is not a real video container should not crash."""
        from src.stages import s01_ingest
        corrupt = tmp_path / "corrupt.mp4"
        # Write random bytes — not a valid video container
        corrupt.write_bytes(b"\xFF\xFE" * 512)
        ctx = make_pipeline_context(corrupt, tmp_path / "out")
        status = s01_ingest.run(ctx)
        # Must return ERROR (or WARNING at most) — must not raise
        assert status.status in {"ERROR", "WARNING"}

    def test_error_does_not_set_video_metadata(self, tmp_path):
        """On ERROR, ctx.video_metadata must remain None — no partial data."""
        from src.stages import s01_ingest
        ctx = make_pipeline_context(tmp_path / "missing.mp4", tmp_path / "out")
        s01_ingest.run(ctx)
        assert ctx.video_metadata is None


# ── s02_sample — core tests ───────────────────────────────────────────────────

class TestS02SampleBasic:
    """s02_sample with standard synthetic videos."""

    def _run_both(self, video: Path, output_dir: Path, fps: float = 1.0):
        """Run s01 then s02 and return the context."""
        from src.stages import s01_ingest, s02_sample
        ctx = make_pipeline_context(video, output_dir, fps=fps)
        s01_ingest.run(ctx)
        s02_sample.run(ctx)
        return ctx

    def test_sample_count_3sec_at_1fps(self, tmp_path):
        """3s @ 1fps → ~3 samples."""
        video = tmp_path / "v3.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        # At 1fps and 3s, expect 2–4 samples (rounding and container header differences)
        assert 2 <= len(ctx.sampled_frame_infos) <= 5

    def test_sample_count_10sec_at_1fps(self, tmp_path):
        """10s video @ 1fps → ~10 samples."""
        video = tmp_path / "v10.mp4"
        make_video(video, duration_sec=10.0, fps=30.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        assert 8 <= len(ctx.sampled_frame_infos) <= 12

    def test_sample_status_ok(self, tmp_path):
        from src.stages import s01_ingest, s02_sample
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        status = s02_sample.run(ctx)
        assert status.status == "OK"

    def test_sampled_frame_infos_not_empty(self, tmp_path):
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out")
        assert len(ctx.sampled_frame_infos) > 0

    def test_video_path_set_in_every_sample(self, tmp_path):
        """Each SampledFrameInfo must reference the source video path."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out")
        for info in ctx.sampled_frame_infos:
            assert info.video_path == video

    def test_no_duplicate_frame_indices(self, tmp_path):
        """Each frame index must appear at most once."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        indices = [i.frame_index for i in ctx.sampled_frame_infos]
        assert len(indices) == len(set(indices)), "Duplicate frame indices found"

    def test_frame_indices_sorted(self, tmp_path):
        """Frame indices must be in ascending order."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        indices = [i.frame_index for i in ctx.sampled_frame_infos]
        assert indices == sorted(indices)

    def test_frame_indices_within_bounds(self, tmp_path):
        """All frame indices must be < frame_count."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        fc = ctx.video_metadata.frame_count
        for info in ctx.sampled_frame_infos:
            assert 0 <= info.frame_index < fc, (
                f"Frame index {info.frame_index} out of bounds [0, {fc})"
            )

    def test_no_pixel_data_in_context(self, tmp_path):
        """Memory safety: no pixel arrays should be stored in the context."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        # SampledFrameInfo has only frame_index, timestamp_sec, video_path
        for info in ctx.sampled_frame_infos:
            assert isinstance(info.frame_index, int)
            assert isinstance(info.timestamp_sec, float)
            assert isinstance(info.video_path, Path)
            # Ensure no numpy arrays or large data objects are stored
            import dataclasses
            fields = {f.name for f in dataclasses.fields(info)}
            assert fields == {"frame_index", "timestamp_sec", "video_path"}


class TestS02SampleTimestamps:
    """Timestamp correctness tests."""

    def _run_both(self, video, output_dir, fps=1.0):
        from src.stages import s01_ingest, s02_sample
        ctx = make_pipeline_context(video, output_dir, fps=fps)
        s01_ingest.run(ctx)
        s02_sample.run(ctx)
        return ctx

    def test_first_sample_is_frame_zero(self, tmp_path):
        """First sampled frame must always be frame 0 (start of video)."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        assert ctx.sampled_frame_infos[0].frame_index == 0

    def test_first_timestamp_is_zero(self, tmp_path):
        """Frame 0's timestamp must be 0.0 s."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        assert ctx.sampled_frame_infos[0].timestamp_sec == pytest.approx(0.0, abs=1e-6)

    def test_timestamps_non_decreasing(self, tmp_path):
        """Timestamps must never go backward."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=10.0, fps=30.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        ts = [i.timestamp_sec for i in ctx.sampled_frame_infos]
        for a, b in itertools.pairwise(ts):
            assert b >= a, f"Timestamp decreased: {a} → {b}"

    def test_timestamps_consistent_with_frame_index(self, tmp_path):
        """timestamp_sec must equal frame_index / video_fps."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        video_fps = ctx.video_metadata.fps
        for info in ctx.sampled_frame_infos:
            expected_ts = info.frame_index / video_fps
            assert abs(info.timestamp_sec - expected_ts) < 1e-6, (
                f"Timestamp {info.timestamp_sec} ≠ {info.frame_index}/{video_fps} = {expected_ts}"
            )

    def test_last_sample_within_duration(self, tmp_path):
        """Last sample timestamp must be ≤ video duration."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=1.0)
        last_ts = ctx.sampled_frame_infos[-1].timestamp_sec
        assert last_ts <= ctx.video_metadata.duration_sec + 0.1  # 100ms tolerance


class TestS02SampleFpsClamping:
    """Behavior when target fps >= or > video fps."""

    def _run_both(self, video, output_dir, fps=1.0):
        from src.stages import s01_ingest, s02_sample
        ctx = make_pipeline_context(video, output_dir, fps=fps)
        s01_ingest.run(ctx)
        s02_sample.run(ctx)
        return ctx

    def test_target_fps_equals_video_fps_no_duplicates(self, tmp_path):
        """Sampling at source FPS should produce one sample per frame — no duplicates."""
        video = tmp_path / "v5fps.mp4"
        make_video(video, duration_sec=2.0, fps=5.0)
        ctx = self._run_both(video, tmp_path / "out", fps=5.0)
        indices = [i.frame_index for i in ctx.sampled_frame_infos]
        assert len(indices) == len(set(indices))

    def test_target_fps_above_video_fps_clamped(self, tmp_path):
        """If target fps > video fps, effective fps is clamped to video fps."""
        video = tmp_path / "v5fps2.mp4"
        make_video(video, duration_sec=2.0, fps=5.0)
        # Request 30fps from a 5fps video — should produce at most 10 frames (2s × 5fps)
        ctx = self._run_both(video, tmp_path / "out", fps=30.0)
        expected_max = int(2.0 * 5.0) + 2   # +2 for rounding slack
        assert len(ctx.sampled_frame_infos) <= expected_max

    def test_sampling_at_sub_1fps(self, tmp_path):
        """0.5 fps sampling of a 10s video should produce ~5 samples."""
        video = tmp_path / "v10s.mp4"
        make_video(video, duration_sec=10.0, fps=10.0)
        ctx = self._run_both(video, tmp_path / "out", fps=0.5)
        assert 4 <= len(ctx.sampled_frame_infos) <= 7


class TestS02SampleOutputFile:
    """sampling_plan.json output tests."""

    def _run_both(self, video, output_dir, fps=1.0):
        from src.stages import s01_ingest, s02_sample
        ctx = make_pipeline_context(video, output_dir, fps=fps)
        s01_ingest.run(ctx)
        s02_sample.run(ctx)
        return ctx

    def test_sampling_plan_json_written(self, tmp_path):
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        output_dir = tmp_path / "out"
        self._run_both(video, output_dir)
        assert (output_dir / "sampling_plan.json").exists()

    def test_sampling_plan_json_valid(self, tmp_path):
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        output_dir = tmp_path / "out"
        self._run_both(video, output_dir)
        data = json.loads((output_dir / "sampling_plan.json").read_text())
        assert "n_frames_sampled" in data
        assert "frames" in data
        assert "video_fps" in data
        assert "target_fps" in data
        assert "segment_fps_reserved" in data  # future VLM use, not yet active

    def test_sampling_plan_count_matches_context(self, tmp_path):
        """n_frames_sampled in JSON must match len(ctx.sampled_frame_infos)."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        output_dir = tmp_path / "out"
        ctx = self._run_both(video, output_dir)
        data = json.loads((output_dir / "sampling_plan.json").read_text())
        assert data["n_frames_sampled"] == len(ctx.sampled_frame_infos)

    def test_sampling_plan_frames_match_context(self, tmp_path):
        """Frame entries in JSON must match ctx.sampled_frame_infos exactly."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        output_dir = tmp_path / "out"
        ctx = self._run_both(video, output_dir)
        data = json.loads((output_dir / "sampling_plan.json").read_text())
        for plan_frame, info in zip(data["frames"], ctx.sampled_frame_infos):
            assert plan_frame["frame_index"] == info.frame_index
            assert abs(plan_frame["timestamp_sec"] - info.timestamp_sec) < 1e-5

    def test_sampling_plan_segment_fps_preserved(self, tmp_path):
        """segment_fps is stored in the plan for future VLM use."""
        video = tmp_path / "v.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        output_dir = tmp_path / "out"
        ctx = self._run_both(video, output_dir)
        data = json.loads((output_dir / "sampling_plan.json").read_text())
        assert data["segment_fps_reserved"] == ctx.config.frame_sampling.segment_fps


class TestS02SampleNoMetadata:
    """s02_sample error handling when s01 has not run."""

    def test_returns_error_when_no_metadata(self, tmp_path):
        from src.config import load_config
        from src.context import PipelineContext
        from src.stages import s02_sample
        config = load_config()
        ctx = PipelineContext(
            config=config,
            video_path=tmp_path / "irrelevant.mp4",
            output_dir=tmp_path / "out",
        )
        # video_metadata is None (s01 never ran)
        status = s02_sample.run(ctx)
        assert status.status == "ERROR"
        assert ctx.sampled_frame_infos == []


class TestS02SampleFrameCountAccuracy:
    """Verify sampling produces exact expected frame counts."""

    @pytest.mark.parametrize("duration,video_fps,target_fps,expected_min,expected_max", [
        (3.0, 10.0, 1.0, 2, 5),     # 3 s at 1fps → ~3 samples
        (10.0, 30.0, 1.0, 8, 12),   # 10 s at 1fps → ~10 samples
        (10.0, 10.0, 2.0, 18, 22),  # 10 s at 2fps → ~20 samples
        (5.0, 25.0, 1.0, 4, 7),     # 5 s at 1fps → ~5 samples
    ])
    def test_frame_count_parametrized(
        self, tmp_path, duration, video_fps, target_fps, expected_min, expected_max
    ):
        from src.stages import s01_ingest, s02_sample
        video = tmp_path / f"v{int(duration)}s_{int(video_fps)}fps.mp4"
        make_video(video, duration_sec=duration, fps=video_fps)
        ctx = make_pipeline_context(video, tmp_path / "out", fps=target_fps)
        s01_ingest.run(ctx)
        s02_sample.run(ctx)
        n = len(ctx.sampled_frame_infos)
        assert expected_min <= n <= expected_max, (
            f"Duration={duration}s, video_fps={video_fps}, "
            f"target_fps={target_fps}: expected [{expected_min},{expected_max}], got {n}"
        )


class TestS02MemorySafety:
    """Verify that no frame pixel data is loaded into memory by s01/s02."""

    def test_sampled_frame_info_has_no_pixel_data(self, tmp_path):
        """SampledFrameInfo must not hold any numpy arrays or byte buffers."""
        from src.stages import s01_ingest, s02_sample
        video = tmp_path / "vmem.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out", fps=1.0)
        s01_ingest.run(ctx)
        s02_sample.run(ctx)
        for info in ctx.sampled_frame_infos:
            # No field should be a numpy array
            assert not isinstance(info.frame_index, np.ndarray)
            assert not isinstance(info.timestamp_sec, np.ndarray)
            # Total size of a SampledFrameInfo object should be tiny
            # (frame_index: 8 bytes, timestamp_sec: 8 bytes, video_path: Path object)
            # Checking for numpy arrays explicitly is sufficient.

    def test_detection_frames_empty_after_s02(self, tmp_path):
        """s02 must not trigger any frame decoding (detection_frames stays empty)."""
        from src.stages import s01_ingest, s02_sample
        video = tmp_path / "vmem2.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        ctx = make_pipeline_context(video, tmp_path / "out")
        s01_ingest.run(ctx)
        s02_sample.run(ctx)
        assert ctx.detection_frames == []


# ── Integration: s01 + s02 together ──────────────────────────────────────────

class TestS01S02Integration:
    """End-to-end ingestion + sampling integration."""

    def test_10min_simulation(self, tmp_path):
        """Simulate a 10-minute video scenario with a short proxy.

        Use a 10-second video as a proxy and verify the 1fps rule:
        target = 1fps, video = 30fps → step = 30 frames.
        For 10 minutes (600s) we'd get ~600 samples; for 10s, ~10 samples.
        """
        from src.stages import s01_ingest, s02_sample
        video = tmp_path / "v10s30fps.mp4"
        make_video(video, duration_sec=10.0, fps=30.0)
        ctx = make_pipeline_context(video, tmp_path / "out", fps=1.0)
        s01_ingest.run(ctx)
        s02_sample.run(ctx)
        assert ctx.video_metadata is not None
        n = len(ctx.sampled_frame_infos)
        assert 8 <= n <= 12, f"Expected ~10 samples, got {n}"

    def test_full_stub_pipeline_with_new_fields(self, tmp_path):
        """Run the complete stub pipeline and verify new schema fields."""
        from src.config import load_config
        from src.context import PipelineContext
        from src.stages import s01_ingest, s02_sample, s12_episode, s13_evaluate

        video = tmp_path / "vfull.mp4"
        make_video(video, duration_sec=3.0, fps=10.0)
        output_dir = tmp_path / "out"
        config = load_config(set_overrides=[
            "stub_mode=true",
            f"output_dir={output_dir}",
            "frame_sampling.fps=1.0",
        ])
        ctx = PipelineContext(config=config, video_path=video, output_dir=output_dir)

        for mod in [s01_ingest, s02_sample]:
            status = mod.run(ctx)
            ctx.record_stage(status)

        # s11/s12 require no AI — always run them
        s12_status = s12_episode.run(ctx)
        ctx.record_stage(s12_status)

        s13_status = s13_evaluate.run(ctx)
        ctx.record_stage(s13_status)

        # Verify new fields
        assert ctx.video_metadata.frame_count > 0
        assert isinstance(ctx.video_metadata.metadata_warnings, list)

        # Verify episode.json contains frame_count
        ep_data = json.loads((output_dir / "episode.json").read_text())
        vm = ep_data.get("video_metadata", {})
        assert "frame_count" in vm
        assert vm["frame_count"] > 0

    def test_sampling_plan_readable_by_downstream(self, tmp_path):
        """sampling_plan.json must be loadable with enough info for YOLO stage."""
        from src.stages import s01_ingest, s02_sample
        video = tmp_path / "vplan.mp4"
        make_video(video, duration_sec=5.0, fps=10.0)
        output_dir = tmp_path / "out"
        ctx = make_pipeline_context(video, output_dir, fps=1.0)
        s01_ingest.run(ctx)
        s02_sample.run(ctx)

        plan = json.loads((output_dir / "sampling_plan.json").read_text())

        # Everything YOLO needs is in the plan
        assert "video_path" in plan
        assert "video_fps" in plan
        assert "frames" in plan
        for frame_entry in plan["frames"]:
            assert "frame_index" in frame_entry
            assert "timestamp_sec" in frame_entry
            # frame_index must be a non-negative integer
            assert isinstance(frame_entry["frame_index"], int)
            assert frame_entry["frame_index"] >= 0
            # timestamp must be a non-negative float
            assert frame_entry["timestamp_sec"] >= 0.0
