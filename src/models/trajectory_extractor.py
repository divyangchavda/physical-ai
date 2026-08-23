"""Trajectory extractor — converts ByteTrack tracks to 2-D image-space trajectories.

All output trajectories are in 2-D image pixel space.
No 3-D reconstruction. No physical units. No interpolation. No smoothing.
"""
from __future__ import annotations

import math

from src.schema.track import Track
from src.schema.trajectory import Trajectory2D, TrajectoryPoint2D


class TrajectoryExtractor:
    def extract(self, tracks: list[Track]) -> list[Trajectory2D]:
        """Convert tracks to Trajectory2D objects.
        
        - Tracks with zero points are skipped (no fabrication).
        - Points are sorted by frame_index for determinism.
        - Centroids are computed from bounding boxes.
        - Statistics are None when insufficient data.
        """
        result = []
        for track in tracks:
            if not track.points:
                continue
            traj = self._convert(track)
            result.append(traj)
        return result
    
    def _convert(self, track: Track) -> Trajectory2D:
        # Sort by frame_index for determinism
        sorted_points = sorted(track.points, key=lambda p: p.frame_index)
        
        traj_points = [
            TrajectoryPoint2D(
                frame_index=p.frame_index,
                timestamp_sec=p.timestamp_sec,
                x_px=(p.bbox.x1 + p.bbox.x2) / 2.0,
                y_px=(p.bbox.y1 + p.bbox.y2) / 2.0,
                confidence=p.detection_confidence,
            )
            for p in sorted_points
        ]
        
        total_dist, mean_speed = self._compute_stats(traj_points)
        
        return Trajectory2D(
            trajectory_id=f"traj_{track.track_id}",
            track_id=track.track_id,
            points=traj_points,
            source=track.source,
            is_estimated=True,
            total_distance_px=total_dist,
            mean_speed_px_per_sec=mean_speed,
        )
    
    def _compute_stats(
        self, points: list[TrajectoryPoint2D]
    ) -> tuple[float | None, float | None]:
        if len(points) < 2:
            return None, None
        
        total_dist = 0.0
        for i in range(1, len(points)):
            dx = points[i].x_px - points[i - 1].x_px
            dy = points[i].y_px - points[i - 1].y_px
            total_dist += math.sqrt(dx * dx + dy * dy)
        
        duration = points[-1].timestamp_sec - points[0].timestamp_sec
        mean_speed = total_dist / duration if duration > 0 else None
        
        return total_dist, mean_speed
