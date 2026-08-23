"""Tests for Pydantic schema models (src/schema/)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.schema.detection import BoundingBox, Detection
from src.schema.episode import PhysicalEpisode
from src.schema.evaluation import EvaluationReport
from src.schema.event import ActionType, PhysicalEvent
from src.schema.trajectory import Trajectory2D, TrajectoryPoint2D

# ── BoundingBox ──────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_valid_bbox(self):
        bb = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        assert bb.width == 90
        assert bb.height == 180
        assert bb.area == 90 * 180
        assert bb.cx == 55.0
        assert bb.cy == 110.0

    def test_x2_must_be_gt_x1(self):
        with pytest.raises(ValueError):
            BoundingBox(x1=100, y1=0, x2=50, y2=100)  # x2 < x1

    def test_x2_equal_x1_raises(self):
        with pytest.raises(ValueError):
            BoundingBox(x1=50, y1=0, x2=50, y2=100)  # x2 == x1

    def test_y2_must_be_gt_y1(self):
        with pytest.raises(ValueError):
            BoundingBox(x1=0, y1=100, x2=100, y2=50)  # y2 < y1


# ── Detection ────────────────────────────────────────────────────────────────

class TestDetection:
    def _make_detection(self, confidence=0.75):
        return Detection(
            detection_id="test_id",
            frame_index=0,
            timestamp_sec=1.0,
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            class_id=1,
            class_name="cup",
            confidence=confidence,
            source="test",
        )

    def test_valid_detection(self):
        d = self._make_detection()
        assert d.class_name == "cup"
        assert d.is_estimated is True

    def test_confidence_lower_bound(self):
        with pytest.raises(ValueError):
            self._make_detection(confidence=-0.01)

    def test_confidence_upper_bound(self):
        with pytest.raises(ValueError):
            self._make_detection(confidence=1.01)

    def test_confidence_at_zero_is_valid(self):
        d = self._make_detection(confidence=0.0)
        assert d.confidence == 0.0

    def test_confidence_at_one_is_valid(self):
        d = self._make_detection(confidence=1.0)
        assert d.confidence == 1.0

    def test_json_round_trip(self):
        d = self._make_detection()
        data = json.loads(d.model_dump_json())
        d2 = Detection.model_validate(data)
        assert d2.class_name == d.class_name
        assert d2.confidence == d.confidence


# ── ActionType and PhysicalEvent ─────────────────────────────────────────────

class TestActionType:
    def test_unknown_is_in_vocabulary(self):
        assert ActionType.UNKNOWN in ActionType

    def test_full_vocabulary(self):
        expected = {
            "GRASP", "RELEASE", "PICK", "PLACE", "MOVE",
            "PUSH", "PULL", "OPEN", "CLOSE", "INSERT",
            "REMOVE", "USE_TOOL", "TOUCH", "INSPECT", "UNKNOWN",
        }
        actual = {a.value for a in ActionType}
        assert actual == expected

    def test_no_extra_actions(self):
        """Vocabulary is locked for MVP v1."""
        assert len(ActionType) == 15


class TestPhysicalEvent:
    def _make_event(self, action=ActionType.UNKNOWN, confidence=0.0):
        return PhysicalEvent(
            event_id="ev_001",
            action=action,
            confidence=confidence,
            source="rule_based",
            start_sec=1.0,
            end_sec=3.0,
        )

    def test_default_action_is_unknown(self):
        """UNKNOWN is the correct default when evidence is insufficient."""
        ev = self._make_event()
        assert ev.action == ActionType.UNKNOWN

    def test_default_review_status_is_pending(self):
        """Quality engine sets review_status — not the event extractor."""
        ev = self._make_event()
        assert ev.review_status == "PENDING"

    def test_raw_prediction_fields_preserved(self):
        """action + confidence + source must always be preserved."""
        ev = self._make_event(action=ActionType.GRASP, confidence=0.85)
        assert ev.action == ActionType.GRASP
        assert ev.confidence == 0.85
        assert ev.source == "rule_based"

    def test_is_estimated_is_true_by_default(self):
        ev = self._make_event()
        assert ev.is_estimated is True

    def test_review_status_values(self):
        """All valid review_status values are accepted."""
        for status in ["PENDING", "AUTO_ACCEPT", "HUMAN_REVIEW", "REJECT", "REPROCESS"]:
            ev = self._make_event()
            ev = ev.model_copy(update={"review_status": status})
            assert ev.review_status == status

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            self._make_event(confidence=-0.01)
        with pytest.raises(ValueError):
            self._make_event(confidence=1.01)

    def test_json_round_trip(self):
        ev = self._make_event(action=ActionType.PICK, confidence=0.6)
        data = json.loads(ev.model_dump_json())
        ev2 = PhysicalEvent.model_validate(data)
        assert ev2.action == ev.action
        assert ev2.confidence == ev.confidence


# ── Trajectory2D ─────────────────────────────────────────────────────────────

class TestTrajectory2D:
    def test_coordinate_space_is_2d_image_pixels(self):
        """coordinate_space is locked — cannot be 3D."""
        traj = Trajectory2D(trajectory_id="t1", track_id=0, source="test")
        assert traj.coordinate_space == "2D_IMAGE_PIXELS"

    def test_coordinate_space_cannot_be_changed(self):
        """Setting coordinate_space to a non-2D value must fail."""
        with pytest.raises((ValueError, Exception)):
            Trajectory2D(
                trajectory_id="t1",
                track_id=0,
                source="test",
                coordinate_space="3D",  # type: ignore[arg-type]
            )

    def test_empty_trajectory_is_valid(self):
        traj = Trajectory2D(trajectory_id="t1", track_id=0, source="test")
        assert traj.points == []

    def test_trajectory_with_points(self):
        traj = Trajectory2D(
            trajectory_id="t1",
            track_id=0,
            source="test",
            points=[
                TrajectoryPoint2D(frame_index=0, timestamp_sec=0.0, x_px=10.0, y_px=20.0, confidence=0.9),
                TrajectoryPoint2D(frame_index=1, timestamp_sec=1.0, x_px=15.0, y_px=25.0, confidence=0.8),
            ],
        )
        assert len(traj.points) == 2


# ── PhysicalEpisode ───────────────────────────────────────────────────────────

class TestPhysicalEpisode:
    def test_default_counts_are_zero(self):
        ep = PhysicalEpisode(episode_id="ep_001")
        assert ep.n_detections == 0
        assert ep.n_tracks == 0
        assert ep.n_events == 0

    def test_json_round_trip(self):
        ep = PhysicalEpisode(
            episode_id="ep_001",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            n_frames_sampled=600,
        )
        data = json.loads(ep.model_dump_json())
        ep2 = PhysicalEpisode.model_validate(data)
        assert ep2.episode_id == ep.episode_id
        assert ep2.n_frames_sampled == 600

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError):
            PhysicalEpisode(episode_id="ep_001", n_detections=-1)


# ── EvaluationReport ──────────────────────────────────────────────────────────

class TestEvaluationReport:
    def test_valid_report(self):
        report = EvaluationReport(
            episode_id="ep_001",
            overall_status="PARTIAL",
            warnings=["stub_mode=True"],
        )
        assert report.overall_status == "PARTIAL"
        assert len(report.warnings) == 1

    def test_json_round_trip(self):
        report = EvaluationReport(
            episode_id="ep_001",
            overall_status="PASS",
        )
        data = json.loads(report.model_dump_json())
        report2 = EvaluationReport.model_validate(data)
        assert report2.overall_status == "PASS"
