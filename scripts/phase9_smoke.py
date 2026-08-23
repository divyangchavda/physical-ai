"""Smoke test for Stage 10 (2-D Trajectory Extraction)."""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.context import PipelineContext
from src.schema.detection import BoundingBox
from src.schema.track import Track, TrackPoint
from src.stages import s10_trajectories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_smoke_test():
    config = load_config()
    config.output_dir = Path("output")
    config.output_dir.mkdir(exist_ok=True)

    ctx = PipelineContext(
        config=config,
        video_path=Path("mock_video.mp4"),
        output_dir=config.output_dir,
    )

    # Synthetic track with 3 points forming a 3-4-5 right triangle
    ctx.tracks = [
        Track(
            track_id=1,
            class_name="cup",
            source="bytetrack",
            start_frame=0,
            end_frame=2,
            start_sec=0.0,
            end_sec=2.0,
            confidence=0.9,
            points=[
                TrackPoint(
                    frame_index=0,
                    timestamp_sec=0.0,
                    bbox=BoundingBox(x1=0.0, y1=0.0, x2=10.0, y2=10.0),  # centroid (5, 5)
                    detection_confidence=0.9,
                ),
                TrackPoint(
                    frame_index=1,
                    timestamp_sec=1.0,
                    bbox=BoundingBox(x1=6.0, y1=0.0, x2=16.0, y2=10.0),  # centroid (11, 5)
                    detection_confidence=0.85,
                ),
                TrackPoint(
                    frame_index=2,
                    timestamp_sec=2.0,
                    bbox=BoundingBox(x1=6.0, y1=8.0, x2=16.0, y2=18.0),  # centroid (11, 13)
                    detection_confidence=0.8,
                ),
            ],
        )
    ]
    original_points_len = len(ctx.tracks[0].points)

    logger.info("Running s10_trajectories...")
    status = s10_trajectories.run(ctx)
    assert status.status == "OK", f"Expected OK, got {status.status}: {status.message}"

    assert len(ctx.trajectories) == 1, f"Expected 1 trajectory, got {len(ctx.trajectories)}"

    traj = ctx.trajectories[0]
    assert traj.coordinate_space == "2D_IMAGE_PIXELS"
    assert traj.trajectory_id == "traj_1"
    assert len(traj.points) == 3

    # Centroid check: first point bbox (0,0)→(10,10) → centroid (5,5)
    assert traj.points[0].x_px == pytest.approx(5.0) if False else abs(traj.points[0].x_px - 5.0) < 0.01
    assert traj.points[0].y_px == pytest.approx(5.0) if False else abs(traj.points[0].y_px - 5.0) < 0.01

    # Distance: segment (5,5)→(11,5) = 6px; segment (11,5)→(11,13) = 8px; total = 14px
    assert traj.total_distance_px is not None
    assert traj.total_distance_px > 0
    expected_dist = 6.0 + 8.0  # = 14.0
    assert abs(traj.total_distance_px - expected_dist) < 0.01, (
        f"Expected total_distance_px={expected_dist}, got {traj.total_distance_px}"
    )

    # Speed: 14px / 2.0 sec = 7.0 px/sec
    assert traj.mean_speed_px_per_sec is not None
    assert abs(traj.mean_speed_px_per_sec - 7.0) < 0.01, (
        f"Expected mean_speed=7.0, got {traj.mean_speed_px_per_sec}"
    )

    # Upstream immutability
    assert len(ctx.tracks[0].points) == original_points_len, "Track was mutated!"

    # Output file
    out_file = config.output_dir / "trajectories.json"
    assert out_file.exists(), "trajectories.json not written"
    import json
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["coordinate_space"] == "2D_IMAGE_PIXELS"

    logger.info("Phase 9 Smoke Test Passed!")
    logger.info("  trajectory_id   : %s", traj.trajectory_id)
    logger.info("  points          : %d", len(traj.points))
    logger.info("  total_distance  : %.2f px", traj.total_distance_px)
    logger.info("  mean_speed      : %.2f px/sec", traj.mean_speed_px_per_sec)
    logger.info("  coordinate_space: %s", traj.coordinate_space)


if __name__ == "__main__":
    run_smoke_test()
