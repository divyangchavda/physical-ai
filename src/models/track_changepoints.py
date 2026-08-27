"""Where in time the tracks change, so events can be timed by geometry.

Why this exists
---------------
Every event a run emits currently inherits its candidate segment's bounds. On
tt7 under config/kaggle_tt7_decoy_b.yaml s05 produces ONE segment, f3..199 =
the whole 6.67s clip, so all seven of the hand-labelled actions fall inside
every event span. tools/score_run.py marks that AMBIGUOUS and says in its own
docstring to read the accuracy as a ceiling. No real number is obtainable from
the run at all until events have distinct timestamps, and s05 cannot supply them
— it cuts on person-object proximity, which changed once in this clip.

So timing has to come from the track geometry. This module finds the frames at
which something about the tracks changed, and nothing else: it does not name
actions, does not decide which change belongs to which verb, and does not touch
the pipeline. It answers one question — *when did the scene change* — and leaves
attribution to the caller.

Every threshold is derived
--------------------------
The rule for this file is that no number in it is a judgement of mine, and none
of it is a tt7 pixel value. There are exactly two derived quantities.

**"Has it moved?" comes from ``detector.nms_iou``.** NMS already encodes the
pipeline's own definition of two boxes being the same box: at IoU >= nms_iou one
is suppressed as a duplicate of the other. Invert that into a displacement. For
two axis-aligned boxes of equal size w x h offset by dx along x alone,

    intersection = (w - dx) * h
    union        = 2wh - (w - dx)h = (w + dx) * h
    IoU          = (w - dx) / (w + dx)

and solving for the offset at which IoU equals a target t,

    dx / w = (1 - t) / (1 + t)

At the configured nms_iou of 0.45 that is 0.55 / 1.45 = 0.379 box-widths: two
boxes further apart than 0.379 of their own width are boxes NMS would have kept
as separate detections, i.e. they are not in the same place. Below it, they are.

Displacement here is 2-D, and the algebra above is 1-D along one axis, so it is
applied per axis: the x offset in box-widths, the y offset in box-heights, and the
larger of the two decides. That is the same question the algebra asks, once for
each direction, and it introduces no new constant. An earlier version normalised
the 2-D distance by the box *diagonal* instead, described as conservative — it is,
but the diagonal of a square box is 1.41 times its width, so the effective
threshold became 0.54 box-widths, a number that appears in no config file. That
is the kind of quietly-invented constant this module exists to avoid.

The budget is per stride step, so gaps of unequal length compare fairly:
``frame_sampling.every_n_frames`` frames of separation are allowed the full
ratio, and the per-frame rate is the ratio divided by the stride. This matters
because a detection can be missed: at stride 3 only 67 of tt7's 200 frames reach
the detector and the observed gaps are not all 3.

**On tt7 the motion signal fires nothing, and that is measured, not assumed.**
The fastest per-axis rate any track reaches is 0.176 per frame (the chopper, on
the single pair ending at f42 — the frame it is carried into the carton) against
a budget of 0.126, and the carton reaches 0.129 on one pair at f12 while it is
still entering the shot. Both are single-pair runs, so ``min_hits`` of 3 — 3
consecutive observed pairs, which at stride 3 is 0.3s — discards them. All twelve
of tt7's change points therefore come from track birth/death and containment.
This code path is exercised by tests but not by tt7's data; on video with faster
motion, or at a smaller stride, it will fire. Do not lower either number to make
it fire on tt7. Both come from the config, and fitting them to one clip is how
the earlier vocabulary work went wrong twice.

**"Is it inside that?" needs no threshold.** Containment is exact: if A is a
subset of B then the clipped intersection equals A's own area to the float,
because every max/min in the clip returns one of A's own coordinates. The one
guard is against a box relabelled as its own contents — a "chopper" detection
that *is* the carton scores containment for no physical reason. The baseline tt7
vocabulary produced four of those with area ratios 1.008 to 1.336. The test is
again nms_iou: if IoU(A, B) >= nms_iou then NMS considered them one detection, so
whatever relation they have is a labelling artifact and not a physical one.

**How long a state must hold comes from ``tracker.min_hits``.** The tracker
already refuses to believe a track exists until min_hits observations support it;
this file applies the same standard to a state, so a single noisy frame cannot
emit a change point. A run shorter than min_hits is not evidence of a change.

Only observed points
--------------------
Callers must pass detection-backed boxes only. Two thirds of ``Track.points`` is
Kalman extrapolation at stride 3, and an earlier geometry pass that read
interpolated boxes reported offsets of six box widths and size ratios of 1.8 —
none of which was in the video. :func:`from_track_dicts` filters on
``detection_confidence > 0`` for exactly that reason.
"""
from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]

# Kinds of change this module can report. Deliberately physical and verb-free:
# naming these ENTER/EXIT/INSERT would be attribution, which is the caller's job.
APPEAR = "APPEAR"
DISAPPEAR = "DISAPPEAR"
MOVE_START = "MOVE_START"
MOVE_STOP = "MOVE_STOP"
ENCLOSE_START = "ENCLOSE_START"
ENCLOSE_END = "ENCLOSE_END"


@dataclass(frozen=True)
class ObservedTrack:
    """One track reduced to the frames on which it was actually detected."""

    track_id: int
    class_name: str
    boxes: dict[int, Box]

    @property
    def frames(self) -> list[int]:
        return sorted(self.boxes)


@dataclass(frozen=True)
class ChangePoint:
    """A frame at which something about the tracks changed."""

    frame: int
    sec: float
    kind: str
    track_id: int
    class_name: str
    other_track_id: int | None = None
    other_class_name: str | None = None
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display only
        other = (
            f" vs {self.other_class_name}#{self.other_track_id}"
            if self.other_track_id is not None else ""
        )
        detail = f"  {self.detail}" if self.detail else ""
        return (
            f"f{self.frame:>3} {self.sec:>5.2f}s {self.kind:<13} "
            f"{self.class_name}#{self.track_id}{other}{detail}"
        )


# ───────────────────────────────────────────────────────────── derived numbers
def still_displacement_ratio(nms_iou: float) -> float:
    """Box-widths of offset at which two equal boxes reach IoU ``nms_iou``.

    See the module docstring for the algebra. Below this offset the pipeline's
    own NMS would call the two boxes one detection, so the object has not moved.
    """
    if not 0.0 <= nms_iou < 1.0:
        raise ValueError(f"nms_iou must be in [0, 1), got {nms_iou}")
    return (1.0 - nms_iou) / (1.0 + nms_iou)


# ─────────────────────────────────────────────────────────────────── geometry
def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return (x2 - x1) * (y2 - y1) if x2 > x1 and y2 > y1 else 0.0


def _iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _centre(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def is_inside(inner: Box, outer: Box, nms_iou: float) -> bool:
    """True when *inner* lies wholly within *outer* and is not *outer* relabelled.

    Exact containment, plus the nms_iou guard described in the module docstring.
    """
    area = _area(inner)
    if area <= 0.0 or _area(outer) <= 0.0:
        return False
    if _iou(inner, outer) >= nms_iou:
        return False  # one detection wearing two labels
    return abs(_intersection(inner, outer) - area) <= 1e-9 * area


def displacement_ratio(a: Box, b: Box) -> float:
    """How far the centre moved from *a* to *b*, in *a*'s own width and height.

    Per axis, larger of the two: the offset-to-IoU algebra in the module docstring
    is one-dimensional, so applying it separately along x and y is the faithful
    2-D generalisation and needs no extra constant.
    """
    width, height = a[2] - a[0], a[3] - a[1]
    (ax, ay), (bx, by) = _centre(a), _centre(b)
    x_ratio = abs(bx - ax) / width if width > 0.0 else 0.0
    y_ratio = abs(by - ay) / height if height > 0.0 else 0.0
    return max(x_ratio, y_ratio)


# ──────────────────────────────────────────────────────────── state debouncing
def _runs(states: list[tuple[int, bool]]) -> list[tuple[bool, list[int]]]:
    """Group ``(frame, state)`` into consecutive same-state runs."""
    out: list[tuple[bool, list[int]]] = []
    for frame, state in states:
        if out and out[-1][0] == state:
            out[-1][1].append(frame)
        else:
            out.append((state, [frame]))
    return out


def _flips(states: list[tuple[int, bool]], min_hits: int) -> list[tuple[int, bool]]:
    """Frames at which a state changed and then held for *min_hits* observations.

    A run shorter than min_hits is discarded, matching the tracker's own rule that
    fewer than min_hits observations are not evidence that something exists. The
    first surviving run establishes the starting state and never emits: a track
    already moving when it appears is reported by APPEAR.

    Discarding a short run can leave two survivors holding the *same* state — a
    single-frame jump between two long still stretches leaves still-then-still —
    so survivors are merged before any flip is emitted. Without that merge the
    blip is reported as a state change in the very case the debounce exists to
    suppress.
    """
    kept = [r for r in _runs(states) if len(r[1]) >= max(1, min_hits)]
    merged: list[tuple[bool, list[int]]] = []
    for state, frames in kept:
        if merged and merged[-1][0] == state:
            merged[-1][1].extend(frames)
        else:
            merged.append((state, list(frames)))
    return [(run[1][0], run[0]) for run in merged[1:]]


# ────────────────────────────────────────────────────────────────── detectors
def motion_changes(
    track: ObservedTrack, *, nms_iou: float, stride: int, min_hits: int, fps: float
) -> list[ChangePoint]:
    """MOVE_START / MOVE_STOP for one track."""
    frames = track.frames
    if len(frames) < 2:
        return []
    budget = still_displacement_ratio(nms_iou) / max(1, stride)

    states: list[tuple[int, bool]] = []
    rates: dict[int, float] = {}
    for earlier, later in zip(frames, frames[1:]):
        gap = later - earlier
        if gap <= 0:
            continue
        rate = displacement_ratio(track.boxes[earlier], track.boxes[later]) / gap
        rates[later] = rate
        states.append((later, rate > budget))

    return [
        ChangePoint(
            frame=frame,
            sec=frame / fps,
            kind=MOVE_START if moving else MOVE_STOP,
            track_id=track.track_id,
            class_name=track.class_name,
            detail=f"rate={rates[frame]:.4f}/frame vs budget {budget:.4f}",
        )
        for frame, moving in _flips(states, min_hits)
    ]


def life_changes(track: ObservedTrack, *, fps: float) -> list[ChangePoint]:
    """APPEAR at the first observed frame, DISAPPEAR after the last.

    The disappearance is reported at the last observed frame rather than a frame
    after it: that is the last moment the object was seen, and inventing a frame
    for its absence would be a claim the data does not make.
    """
    frames = track.frames
    if not frames:
        return []
    return [
        ChangePoint(frame=frames[0], sec=frames[0] / fps, kind=APPEAR,
                    track_id=track.track_id, class_name=track.class_name,
                    detail=f"{len(frames)} observed points"),
        ChangePoint(frame=frames[-1], sec=frames[-1] / fps, kind=DISAPPEAR,
                    track_id=track.track_id, class_name=track.class_name,
                    detail="last observed frame"),
    ]


def containment_changes(
    inner: ObservedTrack, outer: ObservedTrack,
    *, nms_iou: float, min_hits: int, fps: float,
) -> list[ChangePoint]:
    """ENCLOSE_START / ENCLOSE_END for *inner* against *outer*.

    Evaluated on shared observed frames only. Comparing an observed box against
    an interpolated one is what produced the discredited six-box-width offsets.
    """
    shared = sorted(set(inner.boxes) & set(outer.boxes))
    if len(shared) < 2:
        return []
    states = [
        (frame, is_inside(inner.boxes[frame], outer.boxes[frame], nms_iou))
        for frame in shared
    ]
    ratios = {
        frame: _area(inner.boxes[frame]) / _area(outer.boxes[frame])
        for frame in shared if _area(outer.boxes[frame]) > 0.0
    }
    return [
        ChangePoint(
            frame=frame,
            sec=frame / fps,
            kind=ENCLOSE_START if enclosed else ENCLOSE_END,
            track_id=inner.track_id,
            class_name=inner.class_name,
            other_track_id=outer.track_id,
            other_class_name=outer.class_name,
            detail=f"area ratio {ratios.get(frame, float('nan')):.3f}",
        )
        for frame, enclosed in _flips(states, min_hits)
    ]


def find_change_points(
    tracks: list[ObservedTrack],
    *,
    nms_iou: float,
    stride: int,
    min_hits: int,
    fps: float,
    exclude_classes: frozenset[str] = frozenset(),
) -> list[ChangePoint]:
    """Every change point in *tracks*, sorted by frame.

    *exclude_classes* is matched case-insensitively and is intended for the
    background classes a config already names — a "dining table" spans the whole
    clip and every object in the scene is inside it, which is true and useless.
    Passing ``config.segment.background_classes`` keeps that decision in the
    config rather than in this file.
    """
    excluded = {c.strip().lower() for c in exclude_classes}
    kept = [t for t in tracks if t.class_name.strip().lower() not in excluded]

    points: list[ChangePoint] = []
    for track in kept:
        points.extend(life_changes(track, fps=fps))
        points.extend(motion_changes(
            track, nms_iou=nms_iou, stride=stride, min_hits=min_hits, fps=fps,
        ))
    for inner in kept:
        for outer in kept:
            if inner.track_id == outer.track_id:
                continue
            points.extend(containment_changes(
                inner, outer, nms_iou=nms_iou, min_hits=min_hits, fps=fps,
            ))

    points.sort(key=lambda p: (p.frame, p.kind, p.track_id))
    return points


# ─────────────────────────────────────────────────────────────────── adapters
def from_track_dicts(tracks: list[dict]) -> list[ObservedTrack]:
    """Build ObservedTracks from ``tracks.json``, keeping observed points only."""
    out = []
    for t in tracks:
        boxes: dict[int, Box] = {}
        for p in t.get("points", []):
            if (p.get("detection_confidence") or 0.0) <= 0.0:
                continue
            b = p["bbox"]
            boxes[int(p["frame_index"])] = (
                float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]),
            )
        out.append(ObservedTrack(int(t["track_id"]), str(t["class_name"]), boxes))
    return out


def from_fixture(tracks: list[dict]) -> list[ObservedTrack]:
    """Build ObservedTracks from ``tests/fixtures/*_real_detections.json``.

    That file already holds observed points only, as flat
    ``[frame, x1, y1, x2, y2]`` rows — see tools/dump_real_detections.py.
    """
    return [
        ObservedTrack(
            int(t["track_id"]),
            str(t["class_name"]),
            {int(r[0]): (float(r[1]), float(r[2]), float(r[3]), float(r[4]))
             for r in t["real"]},
        )
        for t in tracks
    ]


def from_pipeline_tracks(tracks) -> list[ObservedTrack]:
    """Build ObservedTracks from in-memory :class:`src.schema.track.Track`."""
    return [
        ObservedTrack(
            t.track_id,
            t.class_name,
            {p.frame_index: (p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2)
             for p in t.points if (p.detection_confidence or 0.0) > 0.0},
        )
        for t in tracks
    ]
