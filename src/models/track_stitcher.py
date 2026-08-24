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

It also showed that the fragments are not simply sequential. ``update()``
predicts every live track on every frame and appends the prediction with
``detection_confidence=0.0``, deleting the track only once
``consecutive_misses`` passes ``max(max_age, stride * (max_missed + 1))``. The
unmatched detection at that same frame has already minted the successor id. So a
dying track's ghost tail and its successor's head *always* coexist, up to that
bound — on tt6 the five person fragments overlapped by exactly 15 frames, four
times out of four. Reading that overlap as "two objects were tracked at once"
is wrong; it is one object and an artifact with a formula.

This pass runs after tracking rather than inside it. Reasons: it touches no
Kalman state, it is deterministic, and it can be tested against a saved
``tracks_raw.json`` with no video and no GPU.

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


def _shared_frame_iou(a: Track, b: Track) -> tuple[int, float]:
    """(number of frames both tracks cover, mean IoU on exactly those frames).

    Comparing boxes *at the same frame* is far stronger evidence than comparing
    two endpoints recorded at different instants. Two tracks sitting on the same
    pixels for every frame they coexist are one object seen twice; two objects
    cannot occupy the same space.
    """
    b_boxes = {p.frame_index: p.bbox for p in b.points}
    ious = [
        _iou(p.bbox, b_boxes[p.frame_index])
        for p in a.points
        if p.frame_index in b_boxes
    ]
    if not ious:
        return 0, 0.0
    return len(ious), sum(ious) / len(ious)


def _merge_points(a: list[TrackPoint], b: list[TrackPoint]) -> list[TrackPoint]:
    """Concatenate two point lists, one point per frame, sorted by frame.

    Fragments overlap at the seam, often heavily: a dying track keeps appending
    Kalman-predicted points with ``detection_confidence=0.0`` until it is
    deleted, while its successor is already recording real detections. Keeping
    the higher detection confidence therefore discards the ghost and keeps the
    observation, which is the only defensible tie-break either way.
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
    duplicate_min_iou: float,
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

        n_shared, mean_iou = _shared_frame_iou(ent, frag)

        if gap < -max_overlap_frames:
            # Overlap beyond the tracker's propagation tail, so these two tracks
            # genuinely coexisted rather than one being the other's ghost. They
            # are still the same object if they sat on the same pixels for the
            # whole time they coexisted — that is what a duplicate concurrent
            # track looks like, and what a real second object cannot be.
            if n_shared == 0 or mean_iou < duplicate_min_iou:
                continue
        else:
            a, b = ent.points[-1].bbox, frag.points[0].bbox
            same_place_same_time = n_shared > 0 and mean_iou >= iou_threshold
            if not (
                same_place_same_time
                or _iou(a, b) >= iou_threshold
                or _center_dist_norm(a, b, diagonal) <= max_center_dist_norm
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
    max_overlap_frames: int = 18,
    iou_threshold: float = 0.10,
    max_center_dist_norm: float = 0.15,
    duplicate_min_iou: float = 0.20,
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

    Two different kinds of fragmentation are handled, because tt6 showed both:

    *Sequential.* The object crosses a detection gap and comes back under a new
    id. Bounded by ``max_gap_frames`` and joined on seam geometry.

    *Concurrent.* The dying track's Kalman tail and its successor's head coexist
    for up to ``max_overlap_frames``; past that bound, two tracks that hold the
    same pixels for every shared frame are a duplicate rather than two objects,
    which ``duplicate_min_iou`` decides. Both defaults are derived from the
    tracker's own config in ``s04_track`` rather than chosen here.
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
                    duplicate_min_iou=duplicate_min_iou,
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
