"""Validation script for KalmanSparseTracker.

Tests:
1. Tracker imports
2. Tracker instantiates
3. Detection on frame 0 creates a track
4. Calling update([]) on frames 1-9 produces predicted track points
5. Matching detection on frame 10 updates existing track
6. Same track_id persists across detection gap
7. Object disappears after max_age
8. Different classes don't get incorrectly matched
9. Malformed class names are filtered
"""
from __future__ import annotations

import sys

import numpy as np

from src.schema.detection import BoundingBox, Detection


def test_1_imports():
    """TEST 1: Tracker imports."""
    print("=" * 80)
    print("TEST 1: Imports")
    print("=" * 80)
    
    try:
        from src.models.kalman_sparse_tracker import KalmanSparseTracker, filter_detection
        print("✓ KalmanSparseTracker imported")
        print("✓ filter_detection imported")
        print()
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_instantiation():
    """TEST 2: Tracker instantiates."""
    print("=" * 80)
    print("TEST 2: Instantiation")
    print("=" * 80)
    
    try:
        from src.models.kalman_sparse_tracker import KalmanSparseTracker
        
        tracker = KalmanSparseTracker(
            iou_threshold=0.20,
            max_age=15,
            min_hits=1,
        )
        print(f"✓ Tracker instantiated")
        print(f"  - iou_threshold: {tracker.iou_threshold}")
        print(f"  - max_age: {tracker.max_age}")
        print(f"  - min_hits: {tracker.min_hits}")
        print(f"  - backend_name: {tracker.backend_name}")
        print()
        return tracker
    except Exception as e:
        print(f"✗ Instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_3_create_track(tracker):
    """TEST 3: Detection on frame 0 creates a track."""
    print("=" * 80)
    print("TEST 3: Create Track from Detection")
    print("=" * 80)
    
    try:
        # Create a detection
        detection = Detection(
            detection_id="test_0",
            frame_index=0,
            timestamp_sec=0.0,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            class_id=0,
            class_name="person",
            confidence=0.9,
            source="test",
            is_estimated=False,
        )
        
        # Update tracker
        tracks = tracker.update([detection], frame_index=0)
        
        if len(tracks) == 1:
            track = tracks[0]
            print(f"✓ Track created")
            print(f"  - track_id: {track.track_id}")
            print(f"  - class_name: {track.class_name}")
            print(f"  - points: {len(track.points)}")
            print(f"  - start_frame: {track.start_frame}")
            print()
            return True, track.track_id
        else:
            print(f"✗ Expected 1 track, got {len(tracks)}")
            return False, None
    except Exception as e:
        print(f"✗ Track creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_4_prediction(tracker, track_id):
    """TEST 4: Calling update([]) on frames 1-9 produces predicted points."""
    print("=" * 80)
    print("TEST 4: Kalman Prediction (frames 1-9)")
    print("=" * 80)
    
    try:
        print("Predicting frames 1-9 with empty detection list...")
        for frame_idx in range(1, 10):
            tracks = tracker.update([], frame_index=frame_idx)
            
            if len(tracks) != 1:
                print(f"✗ Expected 1 track on frame {frame_idx}, got {len(tracks)}")
                return False
            
            track = tracks[0]
            if track.track_id != track_id:
                print(f"✗ Track ID changed: {track_id} → {track.track_id}")
                return False
            
            print(f"  Frame {frame_idx:2d}: Track {track.track_id} - {len(track.points)} points")
        
        # Check final track (already updated above in loop)
        track = tracks[0]  # Use last track from loop
        
        if len(track.points) == 10:  # frame 0 + frames 1-9
            print(f"✓ Prediction successful")
            print(f"  - Total points: {len(track.points)}")
            print(f"  - Track spans frames: {track.start_frame} to {track.end_frame}")
            print()
            return True
        else:
            print(f"✗ Expected 10 points, got {len(track.points)}")
            return False
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_update(tracker, track_id):
    """TEST 5: Matching detection on frame 10 updates existing track."""
    print("=" * 80)
    print("TEST 5: Detection Update (frame 10)")
    print("=" * 80)
    
    try:
        # Create detection near previous position
        detection = Detection(
            detection_id="test_10",
            frame_index=10,
            timestamp_sec=10.0 / 30.0,
            bbox=BoundingBox(x1=105, y1=105, x2=205, y2=205),  # Moved slightly
            class_id=0,
            class_name="person",
            confidence=0.85,
            source="test",
            is_estimated=False,
        )
        
        tracks = tracker.update([detection], frame_index=10)
        
        if len(tracks) != 1:
            print(f"✗ Expected 1 track, got {len(tracks)}")
            return False
        
        track = tracks[0]
        
        if track.track_id == track_id:
            print(f"✓ Track updated (not recreated)")
            print(f"  - track_id: {track.track_id} (same)")
            print(f"  - Total points: {len(track.points)}")
            print(f"  - Track spans: {track.start_frame} to {track.end_frame}")
            print()
            return True
        else:
            print(f"✗ Track ID changed: {track_id} → {track.track_id}")
            print(f"  (Detection should have matched existing track)")
            return False
    except Exception as e:
        print(f"✗ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_persistence(tracker, track_id):
    """TEST 6: Track persists across gap."""
    print("=" * 80)
    print("TEST 6: Track Persistence")
    print("=" * 80)
    
    try:
        # Get current tracks
        tracks = tracker.update([], frame_index=11)
        
        if len(tracks) == 1 and tracks[0].track_id == track_id:
            print(f"✓ Track persists after detection gap")
            print(f"  - track_id: {track_id}")
            print(f"  - Total points: {len(tracks[0].points)}")
            print()
            return True
        else:
            print(f"✗ Track lost or ID changed")
            return False
    except Exception as e:
        print(f"✗ Persistence check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_deletion(tracker):
    """TEST 7: Track deleted after max_age."""
    print("=" * 80)
    print("TEST 7: Track Deletion (max_age)")
    print("=" * 80)
    
    try:
        # Reset tracker
        tracker.reset()
        
        # Create a track
        detection = Detection(
            detection_id="test_del",
            frame_index=0,
            timestamp_sec=0.0,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            class_id=0,
            class_name="person",
            confidence=0.9,
            source="test",
            is_estimated=False,
        )
        
        tracks = tracker.update([detection], frame_index=0)
        track_id = tracks[0].track_id
        
        # Advance max_age + 1 frames without detection
        max_age = tracker.max_age
        for i in range(1, max_age + 2):
            tracks = tracker.update([], frame_index=i)
        
        # Track should be deleted
        if len(tracks) == 0:
            print(f"✓ Track deleted after {max_age + 1} frames without detection")
            print()
            return True
        else:
            print(f"✗ Track still exists after {max_age + 1} frames")
            print(f"  - Expected deletion after max_age={max_age}")
            return False
    except Exception as e:
        print(f"✗ Deletion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_class_separation(tracker):
    """TEST 8: Different classes don't match."""
    print("=" * 80)
    print("TEST 8: Class-Aware Matching")
    print("=" * 80)
    
    try:
        # Reset tracker
        tracker.reset()
        
        # Create person track
        det1 = Detection(
            detection_id="test_person",
            frame_index=0,
            timestamp_sec=0.0,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            class_id=0,
            class_name="person",
            confidence=0.9,
            source="test",
            is_estimated=False,
        )
        
        tracks = tracker.update([det1], frame_index=0)
        person_track_id = tracks[0].track_id
        
        # Next frame: cardboard box in similar location
        det2 = Detection(
            detection_id="test_box",
            frame_index=1,
            timestamp_sec=1.0 / 30.0,
            bbox=BoundingBox(x1=105, y1=105, x2=205, y2=205),  # Overlapping
            class_id=1,
            class_name="cardboard box",
            confidence=0.85,
            source="test",
            is_estimated=False,
        )
        
        tracks = tracker.update([det2], frame_index=1)
        
        # Should have 2 tracks: person (predicted) and cardboard box (new)
        if len(tracks) == 2:
            track_ids = {t.track_id for t in tracks}
            classes = {t.class_name for t in tracks}
            
            if "person" in classes and "cardboard box" in classes:
                print(f"✓ Classes kept separate")
                print(f"  - Track 1: person")
                print(f"  - Track 2: cardboard box")
                print()
                return True
            else:
                print(f"✗ Classes: {classes}")
                return False
        else:
            print(f"✗ Expected 2 tracks, got {len(tracks)}")
            return False
    except Exception as e:
        print(f"✗ Class separation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_9_filter_malformed(tracker):
    """TEST 9: Malformed class names filtered."""
    print("=" * 80)
    print("TEST 9: Filter Malformed Class Names")
    print("=" * 80)
    
    try:
        from src.models.kalman_sparse_tracker import filter_detection
        
        # Supported class
        det_good = Detection(
            detection_id="test",
            frame_index=0,
            timestamp_sec=0.0,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            class_id=0,
            class_name="person",
            confidence=0.9,
            source="test",
            is_estimated=False,
        )
        
        # Unsupported classes
        det_empty = Detection(
            detection_id="test",
            frame_index=0,
            timestamp_sec=0.0,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            class_id=0,
            class_name="",
            confidence=0.9,
            source="test",
            is_estimated=False,
        )
        
        det_bad = Detection(
            detection_id="test",
            frame_index=0,
            timestamp_sec=0.0,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
            class_id=0,
            class_name="cardboard box chopper",  # Malformed
            confidence=0.9,
            source="test",
            is_estimated=False,
        )
        
        result_good = filter_detection(det_good)
        result_empty = filter_detection(det_empty)
        result_bad = filter_detection(det_bad)
        
        if result_good is not None and result_empty is None and result_bad is None:
            print(f"✓ Filter working correctly")
            print(f"  - 'person': accepted")
            print(f"  - '' (empty): rejected")
            print(f"  - 'cardboard box chopper': rejected")
            print()
            return True
        else:
            print(f"✗ Filter not working")
            print(f"  - 'person': {result_good is not None}")
            print(f"  - '' (empty): {result_empty is None}")
            print(f"  - 'cardboard box chopper': {result_bad is None}")
            return False
    except Exception as e:
        print(f"✗ Filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_10_boundary_clamping():
    """TEST 10: Boundary clamping for out-of-bound predictions."""
    print("=" * 80)
    print("TEST 10: Boundary Clamping")
    print("=" * 80)
    
    try:
        from src.models.kalman_sparse_tracker import KalmanSparseTracker
        
        # Create tracker with frame dimensions
        frame_width = 1920
        frame_height = 1080
        tracker = KalmanSparseTracker(
            iou_threshold=0.20,
            max_age=15,
            min_hits=1,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        
        # Create detection near right edge with high velocity
        detection = Detection(
            detection_id="test_boundary",
            frame_index=0,
            timestamp_sec=0.0,
            bbox=BoundingBox(x1=1800, y1=500, x2=1900, y2=600),  # Near right edge
            class_id=0,
            class_name="person",
            confidence=0.9,
            source="test",
            is_estimated=False,
        )
        
        tracks = tracker.update([detection], frame_index=0)
        track_id = tracks[0].track_id
        
        # Manually inject high velocity to force out-of-bound prediction
        track = tracker.tracks[track_id]
        track.kf.x[4] = 100.0  # High x velocity (100 pixels per frame)
        track.kf.x[5] = 80.0   # High y velocity
        
        # Predict multiple frames without detection
        print(f"  Initial bbox: x1={detection.bbox.x1:.1f}, y1={detection.bbox.y1:.1f}, "
              f"x2={detection.bbox.x2:.1f}, y2={detection.bbox.y2:.1f}")
        print(f"  Injected velocity: vx=100.0, vy=80.0 pixels/frame")
        print(f"  Frame dimensions: {frame_width}x{frame_height}")
        print()
        
        all_clamped = True
        for i in range(1, 20):
            tracks = tracker.update([], frame_index=i)
            if len(tracks) > 0:
                track = tracks[0]
                last_point = track.points[-1]
                bbox = last_point.bbox
                
                # Check if bbox is within bounds
                within_bounds = (
                    0 <= bbox.x1 <= frame_width and
                    0 <= bbox.y1 <= frame_height and
                    0 <= bbox.x2 <= frame_width and
                    0 <= bbox.y2 <= frame_height
                )
                
                if not within_bounds:
                    print(f"  ✗ Frame {i:2d}: OUT OF BOUNDS - "
                          f"x1={bbox.x1:.1f}, y1={bbox.y1:.1f}, x2={bbox.x2:.1f}, y2={bbox.y2:.1f}")
                    all_clamped = False
                elif i <= 5 or i == 19:  # Show first few and last
                    print(f"  ✓ Frame {i:2d}: CLAMPED - "
                          f"x1={bbox.x1:.1f}, y1={bbox.y1:.1f}, x2={bbox.x2:.1f}, y2={bbox.y2:.1f}")
        
        print()
        if all_clamped:
            print(f"✓ All predictions clamped within bounds [0, {frame_width}] x [0, {frame_height}]")
            print()
            return True
        else:
            print(f"✗ Some predictions exceeded frame boundaries")
            return False
            
    except Exception as e:
        print(f"✗ Boundary clamping test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "KALMAN SPARSE TRACKER VALIDATION" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_1_imports()
    if not results['imports']:
        print("CRITICAL: Import failed. Cannot proceed.")
        return 1
    
    # Test 2: Instantiation
    tracker = test_2_instantiation()
    results['instantiation'] = tracker is not None
    if not results['instantiation']:
        print("CRITICAL: Instantiation failed. Cannot proceed.")
        return 1
    
    # Test 3: Create track
    success, track_id = test_3_create_track(tracker)
    results['create_track'] = success
    if not success:
        print("CRITICAL: Track creation failed. Cannot proceed.")
        return 1
    
    # Test 4: Prediction
    results['prediction'] = test_4_prediction(tracker, track_id)
    
    # Test 5: Update
    results['update'] = test_5_update(tracker, track_id)
    
    # Test 6: Persistence
    results['persistence'] = test_6_persistence(tracker, track_id)
    
    # Test 7: Deletion
    results['deletion'] = test_7_deletion(tracker)
    
    # Test 8: Class separation
    results['class_separation'] = test_8_class_separation(tracker)
    
    # Test 9: Filter malformed
    results['filter_malformed'] = test_9_filter_malformed(tracker)
    
    # Test 10: Boundary clamping
    results['boundary_clamping'] = test_10_boundary_clamping()
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print()
    all_passed = all(results.values())
    
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
