"""Validation script for GroundingDINO + ByteTrack implementation.

Tests:
1. Import GroundingDINODetector
2. Instantiate detector
3. Load model
4. Process single test frame
5. Verify Detection schema
6. Test ByteTrack with Detection objects
7. Test ByteTrack with empty detection list
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def validate_imports():
    """Test 1: Import all required modules."""
    print("=" * 80)
    print("TEST 1: Import Validation")
    print("=" * 80)
    
    try:
        from src.models.groundingdino_detector import GroundingDINODetector
        print("✓ GroundingDINODetector imported successfully")
        
        from src.models.bytetrack_tracker import ByteTrackTracker
        print("✓ ByteTrackTracker imported successfully")
        
        from src.schema.detection import Detection, BoundingBox
        print("✓ Detection schema imported successfully")
        
        from src.schema.track import Track
        print("✓ Track schema imported successfully")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_detector_instantiation():
    """Test 2: Instantiate GroundingDINO detector."""
    print("=" * 80)
    print("TEST 2: Detector Instantiation")
    print("=" * 80)
    
    try:
        from src.models.groundingdino_detector import GroundingDINODetector
        
        detector = GroundingDINODetector(
            text_prompt="person . box . table .",
            box_threshold=0.20,
            text_threshold=0.25,
            device="cpu",  # Use CPU for validation
        )
        print("✓ GroundingDINODetector instantiated successfully")
        print(f"  - text_prompt: {detector.text_prompt}")
        print(f"  - box_threshold: {detector.box_threshold}")
        print(f"  - text_threshold: {detector.text_threshold}")
        print(f"  - device: {detector.device}")
        print(f"  - model_name: {detector.model_name}")
        print()
        return detector
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_model_loading(detector):
    """Test 3: Load GroundingDINO model."""
    print("=" * 80)
    print("TEST 3: Model Loading")
    print("=" * 80)
    
    try:
        print(f"Loading model from: {detector.model_checkpoint}")
        print(f"Using config: {detector.config_file}")
        
        # Check if files exist
        if not Path(detector.model_checkpoint).exists():
            print(f"✗ Model checkpoint not found: {detector.model_checkpoint}")
            return False
        print(f"✓ Model checkpoint exists")
        
        if not Path(detector.config_file).exists():
            print(f"✗ Config file not found: {detector.config_file}")
            return False
        print(f"✓ Config file exists")
        
        # Load model
        detector.load()
        print("✓ Model loaded successfully")
        print()
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_single_frame_detection(detector):
    """Test 4: Process a single test frame."""
    print("=" * 80)
    print("TEST 4: Single Frame Detection")
    print("=" * 80)
    
    try:
        # Create a dummy frame (640x480, BGR)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some simple shapes to make it detectable
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(frame, (300, 200), (400, 350), (200, 200, 200), -1)
        
        print("Running detection on test frame (640x480)...")
        detections = detector.detect(frame, frame_index=0, timestamp_sec=0.0)
        
        print(f"✓ Detection completed")
        print(f"  - Number of detections: {len(detections)}")
        
        if detections:
            print(f"  - Sample detection:")
            det = detections[0]
            print(f"    * detection_id: {det.detection_id}")
            print(f"    * frame_index: {det.frame_index}")
            print(f"    * timestamp_sec: {det.timestamp_sec}")
            print(f"    * bbox: ({det.bbox.x1:.1f}, {det.bbox.y1:.1f}, {det.bbox.x2:.1f}, {det.bbox.y2:.1f})")
            print(f"    * class_id: {det.class_id}")
            print(f"    * class_name: {det.class_name}")
            print(f"    * confidence: {det.confidence:.3f}")
            print(f"    * source: {det.source}")
            print(f"    * is_estimated: {det.is_estimated}")
        else:
            print(f"  - Note: No objects detected in test frame (expected for simple shapes)")
        
        print()
        return detections
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_detection_schema(detections):
    """Test 5: Verify Detection schema compliance."""
    print("=" * 80)
    print("TEST 5: Detection Schema Validation")
    print("=" * 80)
    
    if not detections:
        print("⚠ No detections to validate (skipping schema check)")
        print()
        return True
    
    try:
        from src.schema.detection import Detection, BoundingBox
        
        for idx, det in enumerate(detections):
            # Check type
            assert isinstance(det, Detection), f"Detection {idx} is not a Detection instance"
            
            # Check required fields
            assert hasattr(det, 'detection_id'), f"Detection {idx} missing detection_id"
            assert hasattr(det, 'frame_index'), f"Detection {idx} missing frame_index"
            assert hasattr(det, 'timestamp_sec'), f"Detection {idx} missing timestamp_sec"
            assert hasattr(det, 'bbox'), f"Detection {idx} missing bbox"
            assert hasattr(det, 'class_id'), f"Detection {idx} missing class_id"
            assert hasattr(det, 'class_name'), f"Detection {idx} missing class_name"
            assert hasattr(det, 'confidence'), f"Detection {idx} missing confidence"
            assert hasattr(det, 'source'), f"Detection {idx} missing source"
            assert hasattr(det, 'is_estimated'), f"Detection {idx} missing is_estimated"
            
            # Check bbox type and values
            assert isinstance(det.bbox, BoundingBox), f"Detection {idx} bbox is not BoundingBox"
            assert det.bbox.x2 > det.bbox.x1, f"Detection {idx} bbox x2 <= x1"
            assert det.bbox.y2 > det.bbox.y1, f"Detection {idx} bbox y2 <= y1"
            
            # Check value ranges
            assert det.frame_index >= 0, f"Detection {idx} frame_index < 0"
            assert det.timestamp_sec >= 0, f"Detection {idx} timestamp_sec < 0"
            assert det.class_id >= 0, f"Detection {idx} class_id < 0"
            assert 0.0 <= det.confidence <= 1.0, f"Detection {idx} confidence out of range"
            assert det.source == "groundingdino", f"Detection {idx} source != 'groundingdino'"
            assert det.is_estimated == True, f"Detection {idx} is_estimated != True"
        
        print(f"✓ All {len(detections)} detections pass schema validation")
        print()
        return True
    except AssertionError as e:
        print(f"✗ Schema validation failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Schema validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_bytetrack_with_detections(detections):
    """Test 6: Test ByteTrack with Detection objects."""
    print("=" * 80)
    print("TEST 6: ByteTrack with Detections")
    print("=" * 80)
    
    try:
        from src.models.bytetrack_tracker import ByteTrackTracker
        
        tracker = ByteTrackTracker()
        print("✓ ByteTrackTracker instantiated")
        
        # Use empty list if no detections
        test_detections = detections if detections else []
        
        print(f"Updating tracker with {len(test_detections)} detections...")
        tracks = tracker.update(test_detections, frame_index=0)
        
        print(f"✓ Tracker update successful")
        print(f"  - Active tracks: {len(tracks)}")
        
        if tracks:
            print(f"  - Sample track:")
            track = tracks[0]
            print(f"    * track_id: {track.track_id}")
            print(f"    * class_name: {track.class_name}")
            print(f"    * points: {len(track.points)}")
        
        print()
        return tracker, tracks
    except Exception as e:
        print(f"✗ ByteTrack test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def validate_bytetrack_empty_detections(tracker):
    """Test 7: Test ByteTrack with empty detection list."""
    print("=" * 80)
    print("TEST 7: ByteTrack with Empty Detections (Interpolation)")
    print("=" * 80)
    
    if tracker is None:
        print("✗ Tracker not available (skipping)")
        print()
        return False
    
    try:
        print("Updating tracker with empty detection list...")
        tracks = tracker.update([], frame_index=1)
        
        print(f"✓ Tracker accepts empty detections")
        print(f"  - Active tracks: {len(tracks)}")
        print(f"  - Note: Tracks may persist or disappear depending on tracker state")
        print()
        return True
    except Exception as e:
        print(f"✗ Empty detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "GROUNDINGDINO + BYTETRACK VALIDATION" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = validate_imports()
    if not results['imports']:
        print("CRITICAL: Import validation failed. Cannot proceed.")
        sys.exit(1)
    
    # Test 2: Instantiation
    detector = validate_detector_instantiation()
    results['instantiation'] = detector is not None
    if not results['instantiation']:
        print("CRITICAL: Detector instantiation failed. Cannot proceed.")
        sys.exit(1)
    
    # Test 3: Model Loading
    results['model_loading'] = validate_model_loading(detector)
    if not results['model_loading']:
        print("CRITICAL: Model loading failed. Cannot proceed.")
        sys.exit(1)
    
    # Test 4: Single Frame Detection
    detections = validate_single_frame_detection(detector)
    results['detection'] = detections is not None
    
    # Test 5: Schema Validation
    results['schema'] = validate_detection_schema(detections) if detections else True
    
    # Test 6: ByteTrack with Detections
    tracker, tracks = validate_bytetrack_with_detections(detections)
    results['bytetrack'] = tracker is not None
    
    # Test 7: ByteTrack with Empty Detections
    results['interpolation'] = validate_bytetrack_empty_detections(tracker)
    
    # Cleanup
    if detector:
        detector.unload()
        print("Detector unloaded")
    
    # Summary
    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 25 + "✓ ALL TESTS PASSED" + " " * 35 + "║")
        print("╚" + "═" * 78 + "╝")
        return 0
    else:
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 25 + "✗ SOME TESTS FAILED" + " " * 34 + "║")
        print("╚" + "═" * 78 + "╝")
        return 1


if __name__ == "__main__":
    sys.exit(main())
