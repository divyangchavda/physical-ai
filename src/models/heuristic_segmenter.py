"""Deterministic heuristics for Candidate Interaction Segmentation."""
from __future__ import annotations

import math
import uuid
from typing import Any

from src.config import SegmentConfig
from src.schema.detection import BoundingBox
from src.schema.segment import CandidateSegment
from src.schema.track import Track


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ix1 = max(box1.x1, box2.x1)
    iy1 = max(box1.y1, box2.y1)
    ix2 = min(box1.x2, box2.x2)
    iy2 = min(box1.y2, box2.y2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = box1.area + box2.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def check_proximity(
    box1: BoundingBox,
    box2: BoundingBox,
    iou_thresh: float,
    gap_thresh_norm: float,
    frame_width: float,
    frame_height: float,
) -> tuple[bool, dict[str, Any]]:
    """Check spatial proximity between two bounding boxes."""
    # A. Bounding-box overlap
    iou = compute_iou(box1, box2)
    if iou > iou_thresh:
        return True, {"proximity_type": "overlap", "iou": round(iou, 3)}

    # B. Bounding-box gap
    h_gap = 0.0
    if box1.x2 < box2.x1:
        h_gap = box2.x1 - box1.x2
    elif box2.x2 < box1.x1:
        h_gap = box1.x1 - box2.x2

    v_gap = 0.0
    if box1.y2 < box2.y1:
        v_gap = box2.y1 - box1.y2
    elif box2.y2 < box1.y1:
        v_gap = box1.y1 - box2.y2

    gap = math.hypot(h_gap, v_gap)
    norm_dim = max(frame_width, frame_height)
    if norm_dim > 0:
        norm_gap = gap / norm_dim
    else:
        norm_gap = float("inf")

    if norm_gap < gap_thresh_norm:
        return True, {"proximity_type": "gap", "gap_normalized": round(norm_gap, 3)}

    return False, {}


def check_movement(
    track: Track,
    current_frame: int,
    window_frames: int,
    threshold_norm: float,
    frame_height: float,
) -> tuple[bool, float]:
    """Check if track moved significantly over the recent temporal window."""
    target_start_frame = max(0, current_frame - window_frames)
    
    curr_pt = None
    start_pt = None
    
    for p in track.points:
        if p.frame_index == current_frame:
            curr_pt = p
        if target_start_frame <= p.frame_index <= current_frame and start_pt is None:
                start_pt = p
                
    if not curr_pt or not start_pt or start_pt.frame_index == curr_pt.frame_index:
        return False, 0.0

    dx = curr_pt.bbox.cx - start_pt.bbox.cx
    dy = curr_pt.bbox.cy - start_pt.bbox.cy
    displacement = math.hypot(dx, dy)
    
    if frame_height > 0:
        norm_disp = displacement / frame_height
    else:
        norm_disp = 0.0
        
    moved = norm_disp > threshold_norm
    return moved, round(norm_disp, 3)


def _split_long_segment(
    start_sec: float, end_sec: float, max_duration: float
) -> list[tuple[float, float]]:
    """Split [start, end] into consecutive windows of at most *max_duration*."""
    if max_duration <= 0 or (end_sec - start_sec) <= max_duration:
        return [(start_sec, end_sec)]

    total = end_sec - start_sec
    n_windows = math.ceil(total / max_duration)
    width = total / n_windows  # even split reads better than a short tail
    return [
        (start_sec + i * width, start_sec + (i + 1) * width)
        for i in range(n_windows)
    ]


class RawHit:
    """An instantaneous candidate hit."""
    def __init__(self, frame_idx: int, timestamp_sec: float, p_id: int, o_id: int, signals: dict):
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec
        self.person_id = p_id
        self.object_id = o_id
        self.signals = signals


def generate_candidate_segments(
    tracks: list[Track],
    config: SegmentConfig,
    frame_width: int,
    frame_height: int,
    video_duration_sec: float,
) -> list[CandidateSegment]:
    """Identify Candidate Interactions from tracking data using deterministic heuristics."""
    if not tracks:
        return []

    # 1. Group points by frame
    frame_to_points = {}
    for t in tracks:
        for pt in t.points:
            frame_to_points.setdefault(pt.frame_index, []).append((t, pt))

    # 2. Find instant hits
    hits: list[RawHit] = []
    
    for frame_idx in sorted(frame_to_points.keys()):
        points = frame_to_points[frame_idx]
        persons = [(t, p) for t, p in points if t.class_name in config.person_classes]
        objects = [
            (t, p) for t, p in points
            if t.class_name not in config.person_classes
            and t.class_name not in config.background_classes
        ]

        for p_track, p_pt in persons:
            for o_track, o_pt in objects:
                prox_ok, prox_info = check_proximity(
                    p_pt.bbox,
                    o_pt.bbox,
                    config.proximity.iou_threshold,
                    config.proximity.gap_threshold_normalized,
                    float(frame_width),
                    float(frame_height),
                )

                if not prox_ok:
                    continue

                # Either box moving is a deliberately loose gate. This stage is
                # a *candidate generator*: it should over-produce and let the
                # VLM adjudicate, since box geometry alone cannot separate
                # interaction from co-presence. A stricter "the person→object
                # relation must change" gate looks appealing but suppresses
                # carrying, where the two move in lockstep. Precision comes
                # from s06/s11; what matters here is that candidates stay
                # bounded in length (max_segment_duration_sec) and that scene
                # classes are not paired at all (background_classes).
                p_moved, p_disp = check_movement(
                    p_track, frame_idx, config.movement.window_frames,
                    config.movement.threshold, float(frame_height)
                )
                o_moved, o_disp = check_movement(
                    o_track, frame_idx, config.movement.window_frames,
                    config.movement.threshold, float(frame_height)
                )

                if p_moved or o_moved:
                    signals = {**prox_info}
                    signals["movement"] = p_disp if p_moved else o_disp
                    signals["movement_source"] = "person" if p_moved else "object"

                    hits.append(
                        RawHit(
                            frame_idx=frame_idx,
                            timestamp_sec=p_pt.timestamp_sec,
                            p_id=p_track.track_id,
                            o_id=o_track.track_id,
                            signals=signals,
                        )
                    )

    # Fallback: if no person-object interactions found, check solo person movement.
    # Off by default — see SegmentConfig.enable_solo_person_fallback.
    if not hits and config.enable_solo_person_fallback:
        # For solo person, we need to compare across actual sampled frames, not small frame windows
        # The window_frames config is designed for dense video (every frame), but tracks may be sparse
        for frame_idx in sorted(frame_to_points.keys()):
            points = frame_to_points[frame_idx]
            persons = [(t, p) for t, p in points if t.class_name in config.person_classes]
            
            for p_track, p_pt in persons:
                # Find the earliest available point in this track for comparison
                earliest_pt = p_track.points[0] if p_track.points else None
                
                if earliest_pt and earliest_pt.frame_index < frame_idx:
                    dx = p_pt.bbox.cx - earliest_pt.bbox.cx
                    dy = p_pt.bbox.cy - earliest_pt.bbox.cy
                    displacement = math.hypot(dx, dy)
                    norm_disp = displacement / float(frame_height) if frame_height > 0 else 0.0
                    p_moved = norm_disp > config.movement.threshold
                else:
                    p_moved = False
                    norm_disp = 0.0
                
                if p_moved:
                    signals = {
                        "movement": round(norm_disp, 3),
                        "movement_source": "person",
                        "proximity_type": "solo_person"
                    }
                    
                    hits.append(
                        RawHit(
                            frame_idx=frame_idx,
                            timestamp_sec=p_pt.timestamp_sec,
                            p_id=p_track.track_id,
                            o_id=-1,  # sentinel for solo person
                            signals=signals,
                        )
                    )

    if not hits:
        return []

    # 3. Group hits into contiguous clusters (segments)
    # We will simply create a raw segment for each hit, apply padding, then merge overlaps
    padded_segments = []
    for hit in hits:
        start = max(0.0, hit.timestamp_sec - config.temporal_padding_sec)
        end = min(video_duration_sec, hit.timestamp_sec + config.temporal_padding_sec)
        padded_segments.append({
            "start": start,
            "end": end,
            "hits": [hit]
        })
        
    # Sort by start time
    padded_segments.sort(key=lambda s: s["start"])
    
    # 4. Merge overlapping/adjacent
    merged = []
    current = padded_segments[0]
    
    for nxt in padded_segments[1:]:
        # If overlap or gap is within threshold
        if nxt["start"] <= current["end"] + config.merge_gap_sec:
            # Merge
            current["end"] = max(current["end"], nxt["end"])
            current["hits"].extend(nxt["hits"])
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    
    # 5. Convert to CandidateSegment schemas, splitting over-long merges
    results = []
    for m in merged:
        t_ids = set()
        for h in m["hits"]:
            t_ids.add(h.person_id)
            if h.object_id != -1:  # skip sentinel for solo person
                t_ids.add(h.object_id)

        # extract first trigger reason as representative
        trigger_repr = m["hits"][0].signals.get("proximity_type", "unknown")

        windows = _split_long_segment(
            m["start"], m["end"], config.max_segment_duration_sec
        )
        for start_sec, end_sec in windows:
            # Frame range from the hits that actually fall in this window; a
            # split window must not claim the frames of its siblings.
            window_hits = [
                h for h in m["hits"] if start_sec <= h.timestamp_sec <= end_sec
            ] or m["hits"]
            window_t_ids = set()
            for h in window_hits:
                window_t_ids.add(h.person_id)
                if h.object_id != -1:
                    window_t_ids.add(h.object_id)

            seg = CandidateSegment(
                segment_id=f"cand_{len(results):04d}_{uuid.uuid4().hex[:6]}",
                track_ids=sorted(window_t_ids),
                start_frame=min(h.frame_idx for h in window_hits),
                end_frame=max(h.frame_idx for h in window_hits),
                start_sec=start_sec,
                end_sec=end_sec,
                trigger_reason=f"proximity_{trigger_repr}+movement",
                confidence=0.5,  # heuristic candidate; scored later
                source="rule_based",
                status="PENDING"
            )
            results.append(seg)

    return results
