"""Tests for interface ABCs and stub implementations (src/interfaces/, src/models/)."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from src.interfaces.detector import ObjectDetector
from src.interfaces.pose import PoseEstimator
from src.interfaces.tracker import ObjectTracker
from src.interfaces.vlm import VisionLanguageModel
from src.models.bytetrack_tracker import ByteTrackTracker
from src.models.local_vlm import LocalVLM
from src.models.remote_vlm import RemoteVLM
from src.models.stub_detector import StubDetector
from src.models.stub_tracker import StubTracker
from src.models.stub_vlm import StubLocalVLM, StubRemoteVLM
from src.models.yolo_detector import YOLODetector

# ── ABC instantiation guards ────────────────────────────────────────────────

class TestABCsAreAbstract:
    def test_object_detector_is_abstract(self):
        assert inspect.isabstract(ObjectDetector)

    def test_object_tracker_is_abstract(self):
        assert inspect.isabstract(ObjectTracker)

    def test_vlm_is_abstract(self):
        assert inspect.isabstract(VisionLanguageModel)

    def test_pose_estimator_is_abstract(self):
        assert inspect.isabstract(PoseEstimator)

    def test_cannot_instantiate_detector_abc(self):
        with pytest.raises(TypeError):
            ObjectDetector()  # type: ignore[abstract]

    def test_cannot_instantiate_tracker_abc(self):
        with pytest.raises(TypeError):
            ObjectTracker()  # type: ignore[abstract]

    def test_cannot_instantiate_vlm_abc(self):
        with pytest.raises(TypeError):
            VisionLanguageModel()  # type: ignore[abstract]


# ── StubDetector ────────────────────────────────────────────────────────────

class TestStubDetector:
    def setup_method(self):
        self.det = StubDetector()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_is_object_detector(self):
        assert isinstance(self.det, ObjectDetector)

    def test_load_does_not_raise(self):
        self.det.load()

    def test_detect_returns_empty_list(self):
        """Stub must return an empty list — never fabricate detections."""
        result = self.det.detect(self.frame, frame_index=0, timestamp_sec=0.0)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_returns_list_not_none(self):
        result = self.det.detect(self.frame, 0, 0.0)
        assert result is not None

    def test_unload_does_not_raise(self):
        self.det.load()
        self.det.unload()

    def test_model_name_is_stub(self):
        assert self.det.model_name == "stub"


# ── StubTracker ─────────────────────────────────────────────────────────────

class TestStubTracker:
    def setup_method(self):
        self.tracker = StubTracker()

    def test_is_object_tracker(self):
        assert isinstance(self.tracker, ObjectTracker)

    def test_update_returns_empty_list(self):
        """Stub must return an empty list — never fabricate tracks."""
        result = self.tracker.update([], frame_index=0)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_update_returns_list_not_none(self):
        result = self.tracker.update([], 0)
        assert result is not None

    def test_reset_does_not_raise(self):
        self.tracker.reset()

    def test_backend_name_is_stub(self):
        assert self.tracker.backend_name == "stub"


# ── StubLocalVLM ────────────────────────────────────────────────────────────

class TestStubLocalVLM:
    def setup_method(self):
        self.vlm = StubLocalVLM()

    def test_is_vlm(self):
        assert isinstance(self.vlm, VisionLanguageModel)

    def test_backend_is_local_model(self):
        assert self.vlm.backend == "LOCAL_MODEL"

    def test_analyze_segment_returns_empty_dict(self):
        """Stub must return empty dict — never fabricate physical information."""
        result = self.vlm.analyze_segment(
            video_path=Path("dummy.mp4"),
            start_sec=0.0,
            end_sec=5.0,
            prompt="What is happening?",
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_model_name_is_stub_local(self):
        assert self.vlm.model_name == "stub_local"


# ── StubRemoteVLM ───────────────────────────────────────────────────────────

class TestStubRemoteVLM:
    def setup_method(self):
        self.vlm = StubRemoteVLM()

    def test_is_vlm(self):
        assert isinstance(self.vlm, VisionLanguageModel)

    def test_backend_is_remote_model(self):
        assert self.vlm.backend == "REMOTE_MODEL"

    def test_analyze_segment_returns_empty_dict(self):
        result = self.vlm.analyze_segment(Path("dummy.mp4"), 0.0, 5.0, "prompt")
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_model_name_is_stub_remote(self):
        assert self.vlm.model_name == "stub_remote"


# ── Placeholder implementations ───────────────────────────────────────────

class TestPlaceholders:

    def test_yolo_detector_is_object_detector(self):
        assert isinstance(YOLODetector(), ObjectDetector)

    def test_bytetrack_is_object_tracker(self):
        assert isinstance(ByteTrackTracker(), ObjectTracker)

    def test_local_vlm_analyzes(self):
        vlm = LocalVLM()
        assert isinstance(vlm.analyze_segment(Path("dummy.mp4"), 0.0, 5.0, "prompt"), str)

    def test_remote_vlm_analyzes(self):
        vlm = RemoteVLM(model_name="test")
        assert isinstance(vlm.analyze_segment(Path("dummy.mp4"), 0.0, 5.0, "prompt"), str)

    def test_local_vlm_backend(self):
        assert LocalVLM().backend == "LOCAL_MODEL"

    def test_remote_vlm_backend(self):
        assert RemoteVLM(model_name="test").backend == "REMOTE_MODEL"
