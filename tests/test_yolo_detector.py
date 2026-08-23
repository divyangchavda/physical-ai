"""Unit tests for YOLODetector.

Validates schema mapping, config handling, memory unloading, and error cases
without actually loading real YOLO weights or needing an internet connection.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models.yolo_detector import YOLODetector


@pytest.fixture
def mock_ultralytics(monkeypatch):
    """Mock the ultralytics module so tests can run without the package/model."""
    mock_ultralytics_module = MagicMock()
    mock_yolo_class = MagicMock()
    mock_ultralytics_module.YOLO = mock_yolo_class
    monkeypatch.setitem(sys.modules, "ultralytics", mock_ultralytics_module)
    return mock_yolo_class


def test_lazy_loading(mock_ultralytics):
    """Ensure ultralytics is only imported when load() is called."""
    det = YOLODetector(model_name="yolov8n", device="cpu")
    
    # Model should be None before load
    assert det._model is None
    
    # YOLO constructor shouldn't be called yet
    mock_ultralytics.assert_not_called()
    
    det.load()
    
    # After load, YOLO class is instantiated with the right weights file
    mock_ultralytics.assert_called_once_with("yolov8n.pt")
    assert det._model is not None


def test_detect_valid_objects(mock_ultralytics):
    """Ensure raw YOLO outputs are correctly mapped to our Detection schema."""
    det = YOLODetector(model_name="test_model", confidence=0.6, nms_iou=0.5, device="cpu")
    det.load()

    # Mock inference outputs
    mock_model_instance = mock_ultralytics.return_value
    mock_result = MagicMock()
    mock_result.names = {0: "person", 2: "car"}
    
    mock_box1 = MagicMock()
    mock_xyxy1 = MagicMock()
    mock_xyxy1.tolist.return_value = [10.0, 20.0, 30.0, 40.0]
    mock_box1.xyxy = [mock_xyxy1]
    mock_box1.conf = [0.85]
    mock_box1.cls = [0]
    
    mock_box2 = MagicMock()
    mock_xyxy2 = MagicMock()
    mock_xyxy2.tolist.return_value = [100.0, 100.0, 200.0, 200.0]
    mock_box2.xyxy = [mock_xyxy2]
    mock_box2.conf = [0.95]
    mock_box2.cls = [2]
    
    mock_result.boxes = [mock_box1, mock_box2]
    mock_model_instance.return_value = [mock_result]
    
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = det.detect(frame, frame_index=42, timestamp_sec=2.5)
    
    # YOLO call assertions
    mock_model_instance.assert_called_once_with(
        frame, verbose=False, device="cpu", conf=0.6, iou=0.5
    )
    
    # Output assertions
    assert len(detections) == 2
    
    d1, d2 = detections
    
    assert d1.class_id == 0
    assert d1.class_name == "person"
    assert d1.confidence == 0.85
    assert d1.bbox.x1 == 10.0
    assert d1.bbox.y2 == 40.0
    assert d1.frame_index == 42
    assert d1.timestamp_sec == 2.5
    assert d1.source == "yolo:test_model"
    assert len(d1.detection_id) > 0

    assert d2.class_id == 2
    assert d2.class_name == "car"
    assert d2.confidence == 0.95
    assert d2.bbox.x1 == 100.0


def test_detect_empty_results(mock_ultralytics):
    """Ensure the detector handles frames with zero detections gracefully."""
    det = YOLODetector()
    det.load()
    
    # Mock inference returning no boxes
    mock_model_instance = mock_ultralytics.return_value
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model_instance.return_value = [mock_result]
    
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = det.detect(frame, frame_index=1, timestamp_sec=0.1)
    
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detect_none_boxes(mock_ultralytics):
    """Ensure the detector handles when boxes is literally None."""
    det = YOLODetector()
    det.load()
    
    # Mock inference returning boxes = None
    mock_model_instance = mock_ultralytics.return_value
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model_instance.return_value = [mock_result]
    
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    detections = det.detect(frame, frame_index=1, timestamp_sec=0.1)
    
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detect_without_load_raises_error():
    """Calling detect before load must fail."""
    det = YOLODetector()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="must be load..ed before"):
        det.detect(frame, 0, 0.0)


def test_unload_frees_memory(mock_ultralytics):
    """Test that unload() drops the model reference without crashing."""
    det = YOLODetector(device="cpu")
    det.load()
    assert det._model is not None
    
    det.unload()
    assert det._model is None
    
    # Calling unload multiple times should be safe
    det.unload()
    assert det._model is None
