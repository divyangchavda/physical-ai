"""Tests for Stage 10: 2-D Trajectory Extraction."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.context import PipelineContext
from src.models.trajectory_extractor import TrajectoryExtractor
from src.schema.detection import BoundingBox
from src.schema.track import Track, TrackPoint
from src.schema.trajectory import Trajectory2D
from src.stages import s10_trajectories

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bbox(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _make_point(
    frame_index: int,
    timestamp_sec: float,
    x1: float = 10.0,
    y1: float = 10.0,
    x2: float = 30.0,
    y2: float = 30.0,
    detection_confidence: float = 0.9,
    tracking_confidence: float = 1.0,
) -> TrackPoint:
    return TrackPoint(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        bbox=_make_bbox(x1, y1, x2, y2),
        detection_confidence=detection_confidence,
        tracking_confidence=tracking_confidence,
    )


def _make_track(
    track_id: int = 1,
    class_name: str = "cup",
    points: list[TrackPoint] | None = None,
    source: str = "bytetrack",
) -> Track:
    pts = points or []
    start_frame = pts[0].frame_index if pts else 0
    end_frame = pts[-1].frame_index if pts else 0
    start_sec = pts[0].timestamp_sec if pts else 0.0
    end_sec = pts[-1].timestamp_sec if pts else 0.0
    return Track(
        track_id=track_id,
        class_name=class_name,
        points=pts,
        start_frame=start_frame,
        end_frame=end_frame,
        start_sec=start_sec,
        end_sec=end_sec,
        source=source,
        confidence=0.9,
    )


def _make_ctx(stub_mode: bool = False, output_dir: Path | None = None) -> PipelineContext:
    config = PipelineConfig(stub_mode=stub_mode)
    return PipelineContext(
        config=config,
        video_path=Path("dummy.mp4"),
        output_dir=output_dir or Path("output"),
    )


# ---------------------------------------------------------------------------
# 1. One track with points → one trajectory
# ---------------------------------------------------------------------------

def test_one_track_produces_one_trajectory():
    tracks = [_make_track(points=[_make_point(0, 0.0), _make_point(1, 1.0)])]
    result = TrajectoryExtractor().extract(tracks)
    assert len(result) == 1
    assert isinstance(result[0], Trajectory2D)


# ---------------------------------------------------------------------------
# 2. Centroid calculation
# ---------------------------------------------------------------------------

def test_centroid_x_is_midpoint():
    pt = _make_point(0, 0.0, x1=10.0, y1=20.0, x2=30.0, y2=40.0)
    result = TrajectoryExtractor().extract([_make_track(points=[pt])])
    assert result[0].points[0].x_px == pytest.approx(20.0)


def test_centroid_y_is_midpoint():
    pt = _make_point(0, 0.0, x1=10.0, y1=20.0, x2=30.0, y2=40.0)
    result = TrajectoryExtractor().extract([_make_track(points=[pt])])
    assert result[0].points[0].y_px == pytest.approx(30.0)


def test_centroid_non_square_bbox():
    pt = _make_point(0, 0.0, x1=0.0, y1=0.0, x2=100.0, y2=50.0)
    result = TrajectoryExtractor().extract([_make_track(points=[pt])])
    assert result[0].points[0].x_px == pytest.approx(50.0)
    assert result[0].points[0].y_px == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# 3. Coordinate space is always 2D_IMAGE_PIXELS
# ---------------------------------------------------------------------------

def test_coordinate_space_is_2d_image_pixels():
    result = TrajectoryExtractor().extract([
        _make_track(points=[_make_point(0, 0.0)])
    ])
    assert result[0].coordinate_space == "2D_IMAGE_PIXELS"


def test_coordinate_space_cannot_be_changed():
    """Schema enforces Literal['2D_IMAGE_PIXELS'] — any other value must raise."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Trajectory2D(
            trajectory_id="t",
            track_id=1,
            coordinate_space="3D_WORLD",  # type: ignore[arg-type]
            source="test",
        )


# ---------------------------------------------------------------------------
# 4. Track with zero points produces no trajectory
# ---------------------------------------------------------------------------

def test_empty_track_produces_no_trajectory():
    track = _make_track(points=[])
    result = TrajectoryExtractor().extract([track])
    assert result == []


def test_empty_track_does_not_fabricate_point():
    track = _make_track(points=[])
    result = TrajectoryExtractor().extract([track])
    assert not any(len(t.points) > 0 for t in result)


# ---------------------------------------------------------------------------
# 5. Empty ctx.tracks
# ---------------------------------------------------------------------------

def test_empty_tracks_list_produces_empty_result():
    result = TrajectoryExtractor().extract([])
    assert result == []


# ---------------------------------------------------------------------------
# 6. Multiple tracks → multiple trajectories
# ---------------------------------------------------------------------------

def test_multiple_tracks_produce_multiple_trajectories():
    tracks = [
        _make_track(track_id=1, points=[_make_point(0, 0.0), _make_point(1, 1.0)]),
        _make_track(track_id=2, points=[_make_point(0, 0.0), _make_point(1, 1.0)]),
        _make_track(track_id=3, points=[]),  # no points → skipped
    ]
    result = TrajectoryExtractor().extract(tracks)
    assert len(result) == 2
    assert {t.track_id for t in result} == {1, 2}


# ---------------------------------------------------------------------------
# 7. Total distance — two points
# ---------------------------------------------------------------------------

def test_total_distance_two_points():
    pts = [
        _make_point(0, 0.0, x1=0.0,  y1=0.0,  x2=0.01, y2=0.01),  # centroid ≈ (0.005, 0.005)
        _make_point(1, 1.0, x1=12.0, y1=16.0, x2=12.01, y2=16.01), # centroid ≈ (12.005, 16.005)
    ]
    result = TrajectoryExtractor().extract([_make_track(points=pts)])
    expected = math.sqrt(12.0**2 + 16.0**2)  # = 20.0
    assert result[0].total_distance_px == pytest.approx(expected, abs=0.1)


# ---------------------------------------------------------------------------
# 8. Total distance — multiple points
# ---------------------------------------------------------------------------

def test_total_distance_multiple_points_sums_segments():
    # Three collinear points 10px apart each → total 20px
    pts = [
        _make_point(0, 0.0, x1=0.0,  y1=0.0,  x2=0.01, y2=0.01),
        _make_point(1, 1.0, x1=10.0, y1=0.0,  x2=10.01, y2=0.01),
        _make_point(2, 2.0, x1=20.0, y1=0.0,  x2=20.01, y2=0.01),
    ]
    result = TrajectoryExtractor().extract([_make_track(points=pts)])
    assert result[0].total_distance_px == pytest.approx(20.0, abs=0.01)


# ---------------------------------------------------------------------------
# 9. Single point → distance = None
# ---------------------------------------------------------------------------

def test_single_point_distance_is_none():
    result = TrajectoryExtractor().extract([
        _make_track(points=[_make_point(0, 0.0)])
    ])
    assert result[0].total_distance_px is None


def test_single_point_speed_is_none():
    result = TrajectoryExtractor().extract([
        _make_track(points=[_make_point(0, 0.0)])
    ])
    assert result[0].mean_speed_px_per_sec is None


# ---------------------------------------------------------------------------
# 10. Mean speed with valid duration
# ---------------------------------------------------------------------------

def test_mean_speed_correct():
    pts = [
        _make_point(0, 0.0, x1=0.0, y1=0.0, x2=0.01, y2=0.01),
        _make_point(1, 2.0, x1=10.0, y1=0.0, x2=10.01, y2=0.01),  # 10px, 2 sec
    ]
    result = TrajectoryExtractor().extract([_make_track(points=pts)])
    assert result[0].mean_speed_px_per_sec == pytest.approx(10.0 / 2.0, abs=0.01)


# ---------------------------------------------------------------------------
# 11. Mean speed when duration = 0
# ---------------------------------------------------------------------------

def test_mean_speed_none_when_duration_zero():
    pts = [
        _make_point(0, 5.0, x1=0.0, y1=0.0, x2=0.01, y2=0.01),
        _make_point(1, 5.0, x1=10.0, y1=0.0, x2=10.01, y2=0.01),  # same timestamp
    ]
    result = TrajectoryExtractor().extract([_make_track(points=pts)])
    assert result[0].mean_speed_px_per_sec is None


# ---------------------------------------------------------------------------
# 12. Timestamp ordering: sort by frame_index
# ---------------------------------------------------------------------------

def test_points_sorted_by_frame_index_regardless_of_input_order():
    pts = [
        _make_point(5, 5.0),
        _make_point(1, 1.0),
        _make_point(3, 3.0),
    ]
    result = TrajectoryExtractor().extract([_make_track(points=pts)])
    frame_indices = [p.frame_index for p in result[0].points]
    assert frame_indices == sorted(frame_indices)


# ---------------------------------------------------------------------------
# 13. Trajectory ID determinism
# ---------------------------------------------------------------------------

def test_trajectory_id_equals_traj_track_id():
    track = _make_track(track_id=42, points=[_make_point(0, 0.0)])
    result = TrajectoryExtractor().extract([track])
    assert result[0].trajectory_id == "traj_42"


def test_trajectory_id_is_deterministic():
    track = _make_track(track_id=7, points=[_make_point(0, 0.0)])
    r1 = TrajectoryExtractor().extract([track])
    r2 = TrajectoryExtractor().extract([track])
    assert r1[0].trajectory_id == r2[0].trajectory_id


# ---------------------------------------------------------------------------
# 14. Source/provenance preservation
# ---------------------------------------------------------------------------

def test_source_preserved_from_track():
    track = _make_track(source="bytetrack", points=[_make_point(0, 0.0)])
    result = TrajectoryExtractor().extract([track])
    assert result[0].source == "bytetrack"


def test_source_stub_preserved():
    track = _make_track(source="stub", points=[_make_point(0, 0.0)])
    result = TrajectoryExtractor().extract([track])
    assert result[0].source == "stub"


# ---------------------------------------------------------------------------
# 15. Confidence preservation
# ---------------------------------------------------------------------------

def test_confidence_from_detection_confidence():
    pt = _make_point(0, 0.0, detection_confidence=0.75)
    result = TrajectoryExtractor().extract([_make_track(points=[pt])])
    assert result[0].points[0].confidence == pytest.approx(0.75)


def test_confidence_different_per_point():
    pts = [
        _make_point(0, 0.0, detection_confidence=0.9),
        _make_point(1, 1.0, detection_confidence=0.6),
    ]
    result = TrajectoryExtractor().extract([_make_track(points=pts)])
    assert result[0].points[0].confidence == pytest.approx(0.9)
    assert result[0].points[1].confidence == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# 16. Point order preservation
# ---------------------------------------------------------------------------

def test_point_frame_indices_preserved():
    pts = [_make_point(i, float(i)) for i in range(5)]
    result = TrajectoryExtractor().extract([_make_track(points=pts)])
    assert [p.frame_index for p in result[0].points] == list(range(5))


# ---------------------------------------------------------------------------
# 17. Missing frame numbers — no interpolation
# ---------------------------------------------------------------------------

def test_no_interpolation_for_missing_frames():
    pts = [
        _make_point(0, 0.0),
        _make_point(5, 5.0),   # frames 1-4 missing
        _make_point(10, 10.0), # frames 6-9 missing
    ]
    result = TrajectoryExtractor().extract([_make_track(points=pts)])
    assert len(result[0].points) == 3
    assert [p.frame_index for p in result[0].points] == [0, 5, 10]


# ---------------------------------------------------------------------------
# 18. Upstream tracks not mutated
# ---------------------------------------------------------------------------

def test_upstream_tracks_unchanged():
    original_pts = [_make_point(0, 0.0), _make_point(1, 1.0)]
    track = _make_track(points=original_pts)
    original_len = len(track.points)
    original_ids = [p.frame_index for p in track.points]
    TrajectoryExtractor().extract([track])
    assert len(track.points) == original_len
    assert [p.frame_index for p in track.points] == original_ids


# ---------------------------------------------------------------------------
# 19. Deterministic repeated execution
# ---------------------------------------------------------------------------

def test_deterministic_repeated_extraction():
    tracks = [
        _make_track(track_id=1, points=[_make_point(0, 0.0), _make_point(1, 1.0)]),
        _make_track(track_id=2, points=[_make_point(0, 0.0), _make_point(1, 1.0)]),
    ]
    r1 = TrajectoryExtractor().extract(tracks)
    r2 = TrajectoryExtractor().extract(tracks)
    assert [t.trajectory_id for t in r1] == [t.trajectory_id for t in r2]
    assert [t.model_dump_json() for t in r1] == [t.model_dump_json() for t in r2]


# ---------------------------------------------------------------------------
# 20. JSON serialization / deserialization
# ---------------------------------------------------------------------------

def test_trajectory_json_round_trip():
    track = _make_track(points=[_make_point(0, 0.0), _make_point(1, 1.0)])
    result = TrajectoryExtractor().extract([track])
    as_json = result[0].model_dump(mode="json")
    restored = Trajectory2D.model_validate(as_json)
    assert restored.trajectory_id == result[0].trajectory_id
    assert restored.coordinate_space == "2D_IMAGE_PIXELS"


def test_trajectories_json_file_written(tmp_path):
    ctx = _make_ctx(output_dir=tmp_path)
    ctx.tracks = [_make_track(points=[_make_point(0, 0.0), _make_point(1, 1.0)])]
    status = s10_trajectories.run(ctx)
    assert status.status == "OK"
    out = tmp_path / "trajectories.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1


# ---------------------------------------------------------------------------
# 21. Stub mode
# ---------------------------------------------------------------------------

def test_stub_mode_returns_skipped(tmp_path):
    ctx = _make_ctx(stub_mode=True, output_dir=tmp_path)
    ctx.tracks = [_make_track(points=[_make_point(0, 0.0)])]
    status = s10_trajectories.run(ctx)
    assert status.status == "SKIPPED"


def test_stub_mode_no_trajectories_fabricated(tmp_path):
    ctx = _make_ctx(stub_mode=True, output_dir=tmp_path)
    ctx.tracks = [_make_track(points=[_make_point(0, 0.0)])]
    s10_trajectories.run(ctx)
    assert ctx.trajectories == []


def test_stub_mode_writes_empty_json(tmp_path):
    ctx = _make_ctx(stub_mode=True, output_dir=tmp_path)
    ctx.tracks = [_make_track(points=[_make_point(0, 0.0)])]
    s10_trajectories.run(ctx)
    data = json.loads((tmp_path / "trajectories.json").read_text(encoding="utf-8"))
    assert data == []


# ---------------------------------------------------------------------------
# 22. No tracks behavior
# ---------------------------------------------------------------------------

def test_no_tracks_returns_skipped(tmp_path):
    ctx = _make_ctx(output_dir=tmp_path)
    ctx.tracks = []
    status = s10_trajectories.run(ctx)
    assert status.status == "SKIPPED"


def test_no_tracks_writes_empty_json(tmp_path):
    ctx = _make_ctx(output_dir=tmp_path)
    ctx.tracks = []
    s10_trajectories.run(ctx)
    data = json.loads((tmp_path / "trajectories.json").read_text(encoding="utf-8"))
    assert data == []


# ---------------------------------------------------------------------------
# 23. Output file generation
# ---------------------------------------------------------------------------

def test_output_file_written_on_ok(tmp_path):
    ctx = _make_ctx(output_dir=tmp_path)
    ctx.tracks = [_make_track(points=[_make_point(0, 0.0), _make_point(1, 1.0)])]
    s10_trajectories.run(ctx)
    assert (tmp_path / "trajectories.json").exists()


# ---------------------------------------------------------------------------
# 24. Output schema validation
# ---------------------------------------------------------------------------

def test_output_schema_validates(tmp_path):
    ctx = _make_ctx(output_dir=tmp_path)
    ctx.tracks = [_make_track(points=[_make_point(0, 0.0), _make_point(1, 1.0)])]
    s10_trajectories.run(ctx)
    data = json.loads((tmp_path / "trajectories.json").read_text(encoding="utf-8"))
    for entry in data:
        Trajectory2D.model_validate(entry)


# ---------------------------------------------------------------------------
# 25. is_estimated is always True
# ---------------------------------------------------------------------------

def test_is_estimated_always_true():
    result = TrajectoryExtractor().extract([
        _make_track(points=[_make_point(0, 0.0)])
    ])
    assert result[0].is_estimated is True
