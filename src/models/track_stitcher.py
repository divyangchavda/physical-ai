"""Post-tracking entity resolution — stitch fragmented tracks into entities.

``kalman_sparse_tracker`` has no re-identification path. ``update()`` deletes a
track once ``consecutive_misses`` exceeds its budget, returns only surviving
tracks, and mints a fresh id for every unmatched detection. So one physical
object crossing a detection gap always comes back as a brand-new id, and there
is no mechanism anywhere that can undo that.

tt6 showed the cost: 21 chopper ids, 13 box ids and 5 person ids for a video
containing one chopper, one box and one person. Downstream that is not cosmetic
— ``s07_events._resolve_actor_track`` picks the longest person fragment *per
segment*, so each segment named a different id, and ``graph_builder`` keys nodes
as ``node_track_{id}``, so the delivered graph claimed three separate people.

This pass runs after tracking rather than inside it. Reasons: it touches no
Kalman state, it is deterministic, and it can be tested against a saved
``tracks.json`` with no video and no GPU.

What it does NOT do is force one entity per class. Real cuts, occlusions and
genuine second objects must survive as separate entities — tt6 is one ~8.3s clip
copied four times, so the *correct* answer there is about four entities per
class, with the breaks landing on the copy boundaries.
"""
from __future__ import annotations

import math

from src.schema.detection import BoundingBox
from src.schema.track import Track, TrackPoint


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _center_dist_norm(a: BoundingBox, b: BoundingBox, diagonal: float) -> float:
    """Centre-to-centre distance as a fraction of the frame diagonal.

    IoU alone is not enough for the seam: a small object like the chopper can
    move its own width during a 1s dropout and score 0 IoU while still being
    obviously the same object. Distance catches that; IoU catches the case where
    a large slow object barely moves. Either passing is enough.
    """
    if diagonal <= 0:
        return float("inf")
    return math.hypot(a.cx - b.cx, a.cy - b.cy) / diagonal


def _merge_points(a: list[TrackPoint], b: list[TrackPoint]) -> list[TrackPoint]:
    """Concatenate two point lists, one point per frame, sorted by frame.

    Fragments may overlap by a frame or two at the seam. Keeping the
    higher-confidence observation is the only defensible tie-break: both are
    real observations of the same object at the same instant.
    """
    best: dict[int, TrackPoint] = {}
    for p in (*a, *b):
        prev = best.get(p.frame_index)
        if prev is None or p.detection_confidence > prev.detection_confidence:
            best[p.frame_index] = p
    return [best[k] for k in sorted(best)]


def _pick_entity(
    frag: Track,
    entities: list[Track],
    *,
    max_gap_frames: int,
    max_overlap_frames: int,
    iou_threshold: float,
    max_center_dist_norm: float,
    diagonal: float,
) -> Track | None:
    """Return the entity this fragment continues, or None to start a new one."""
    best: Track | None = None
    for ent in entities:
        if not ent.points:
            continue
        gap = frag.start_frame - ent.end_frame
        # Too far apart in time to be the same object crossing a dropout.
        if gap > max_gap_frames:
            continue
        # Substantial temporal overlap means both were tracked simultaneously,
        # which is evidence of two objects, not one. A frame or two of overlap
        # at the seam is normal and allowed.
        if gap < -max_overlap_frames:
            continue

        a, b = ent.points[-1].bbox, frag.points[0].bbox
        if (
            _iou(a, b) < iou_threshold
            and _center_dist_norm(a, b, diagonal) > max_center_dist_norm
        ):
            continue

        # Most recently active entity wins; lowest id breaks a tie.
        if best is None or (ent.end_frame, -ent.track_id) > (best.end_frame, -best.track_id):
            best = ent
    return best


def stitch_tracks(
    tracks: list[Track],
    frame_width: int,
    frame_height: int,
    *,
    max_gap_frames: int = 45,
    max_overlap_frames: int = 2,
    iou_threshold: float = 0.10,
    max_center_dist_norm: float = 0.15,
) -> tuple[list[Track], dict[int, list[int]]]:
    """Merge fragments of the same physical object into single tracks.

    Returns ``(entities, absorbed)`` where *absorbed* maps each surviving
    track_id to every original id folded into it, itself included. The map is
    written to ``track_merges.json`` so a merge is auditable rather than a
    silent rewrite of the tracking output.

    Fragments are grouped by ``class_name`` and walked in ``start_frame`` order,
    each one attaching to the most recently active compatible entity. Greedy and
    single-pass on purpose: fragments of one object arrive in time order, so
    there is nothing for a global optimiser to recover, and greedy stays
    reproducible.
    """
    if not tracks:
        return [], {}

    diagonal = math.hypot(float(frame_width), float(frame_height))

    by_class: dict[str, list[Track]] = {}
    for t in tracks:
        by_class.setdefault(t.class_name, []).append(t)

    entities: list[Track] = []
    absorbed: dict[int, list[int]] = {}

    for class_name in sorted(by_class):
        # (start_frame, track_id) so the walk order never depends on dict order.
        fragments = sorted(by_class[class_name], key=lambda t: (t.start_frame, t.track_id))
        open_entities: list[Track] = []

        for frag in fragments:
            target = (
                _pick_entity(
                    frag, open_entities,
                    max_gap_frames=max_gap_frames,
                    max_overlap_frames=max_overlap_frames,
                    iou_threshold=iou_threshold,
                    max_center_dist_norm=max_center_dist_norm,
                    diagonal=diagonal,
                )
                if frag.points
                else None
            )

            if target is None:
                entity = frag.model_copy(deep=True)
                open_entities.append(entity)
                absorbed[entity.track_id] = [frag.track_id]
                continue

            merged = _merge_points(target.points, frag.points)
            target.points = merged
            target.start_frame = merged[0].frame_index
            target.end_frame = merged[-1].frame_index
            target.start_sec = merged[0].timestamp_sec
            target.end_sec = merged[-1].timestamp_sec
            absorbed[target.track_id].append(frag.track_id)

        entities.extend(open_entities)

    entities.sort(key=lambda t: t.track_id)
    return entities, absorbed
