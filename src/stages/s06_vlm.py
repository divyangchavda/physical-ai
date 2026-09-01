"""Stage 06 — VLM Semantic Analysis.

Invokes the Vision-Language Model to interpret physical interactions
within Candidate Segments. Produces raw observations (NOT normalized events).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.models.gemini_vlm import GeminiVLM
from src.models.local_vlm import LocalVLM
from src.models.remote_vlm import RemoteVLM
from src.schema.episode import PipelineStageStatus
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus

logger = get_logger(__name__)
STAGE = "s06_vlm"

# Substituted per segment by _render_prompt. A sentinel rather than a format
# field because the schema block below is full of literal braces.
_DURATION_TOKEN = "{{CLIP_DURATION}}"

# Bumped from "v1" when the clip duration was added. Two runs whose observations
# came from different prompts are not comparable, and prompt_version is the only
# field in vlm_observations.json that can tell them apart.
PROMPT_VERSION = "v2"

PROMPT = """You are a physical interaction analyst. Review the provided sequence of video frames, which is a single clip {{CLIP_DURATION}} seconds long.
Identify the physical interaction occurring, if any.
Answer in JSON format ONLY. Do not include markdown formatting, preambles, or explanations.

Schema:
{
  "actor": "description of the person acting, or 'UNKNOWN'",
  "active_hand": "LEFT, RIGHT, BOTH, or 'UNKNOWN'",
  "objects": ["list of primary objects manipulated"],
  "raw_action": "short description of the physical action (e.g., 'picked up the cup')",
  "start_time_sec": <float offset from the clip start, between 0.0 and {{CLIP_DURATION}}, or null>,
  "end_time_sec": <float offset from the clip start, between 0.0 and {{CLIP_DURATION}}, or null>,
  "state_change": "description of object state change (e.g., 'box is open'), or 'UNKNOWN'",
  "visible_facts": "only things directly observable in the frames (e.g., 'hand contacts cup')",
  "inference": "your reasoned interpretation beyond what is directly visible (e.g., 'likely picking up')",
  "uncertainty": "what is occluded, unclear, or ambiguous",
  "confidence": <float between 0.0 and 1.0>
}

Rules:
1. Do not invent information. Use 'UNKNOWN' or null if evidence is insufficient.
2. Distinguish visible facts from inference. visible_facts = what you directly see. inference = what you conclude.
3. Do not identify objects that cannot be visually supported.
4. This clip is {{CLIP_DURATION}} seconds long and nothing outside it is visible to you.
   start_time_sec and end_time_sec are offsets within this clip, not timestamps in a
   longer video, so neither may exceed {{CLIP_DURATION}}. Report null rather than a
   guess if you cannot place the action within the clip.
"""


def _render_prompt(duration_sec: float) -> str:
    """Fill in the clip length the model is being asked to time against.

    The prompt used to ask for a "float offset from segment start" and never say
    what the segment's length was. Measured across twelve tt7 segments at
    max_segment_duration_sec=1.0, gemini-3.1-flash-lite answered with a relative
    end_time_sec of *exactly 2.0* every single time — on clips 0.95s long. It was
    not localising the action; 2.0 is what the field defaults to when the bound
    is unstated.

    Five of those were then discarded outright (see _absolute_timing) and the
    remaining seven all fell back to SEGMENT precision, which is why every event
    spanned its whole window and covered two to four labelled actions.
    """
    return PROMPT.replace(_DURATION_TOKEN, f"{duration_sec:.2f}")


def extract_json(text: str) -> str:
    """Extract a JSON object *or array* from a string that may contain markdown.

    Gemini answers with a bare object for most clips but sometimes returns an
    array — one entry per action it saw. Scraping the first ``{`` to the last
    ``}`` happened to work while those arrays held a single element; on a
    two-element array it yields ``{...}, {...}``, which is not valid JSON and
    failed the whole segment. Pick the container by whichever bracket opens
    first and close with its matching kind.
    """
    text = text.strip()
    match = re.search(
        r'```(?:json)?\s*([\[{].*[\]}])\s*```', text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1)

    # fallback: the earliest of { or [ decides which container we are reading
    obj_start, arr_start = text.find('{'), text.find('[')
    candidates = [(i, o, c) for i, o, c in
                  ((obj_start, '{', '}'), (arr_start, '[', ']')) if i != -1]
    if candidates:
        start, _open, close = min(candidates)
        end = text.rfind(close)
        if end > start:
            return text[start:end + 1]

    return text


def _absolute_timing(
    record: dict, seg_start: float, seg_end: float, fps: float | None = None
) -> tuple[float | None, float | None, str | None]:
    """Convert the VLM's relative offsets to absolute seconds.

    Returns ``(start, end, warning)``. Both bounds come back ``None`` when the
    model's offsets do not fit the clip.

    The prompt now states the clip length (see _render_prompt), which removes the
    known cause of unusable offsets rather than only the symptom. This stays
    regardless: a stated bound is an instruction, not a guarantee, and one
    ignored offset must never cost a usable raw_action. Measured on tt7 at
    ``max_segment_duration_sec=1.0`` under the old prompt: five of seven
    0.95-second segments came back with ``end_time_sec: 2.0``, an offset twice
    the length of the clip it describes.

    Those five used to be discarded outright. RawVLMObservation enforces that
    absolute timestamps lie inside the segment, s06 catches the ValueError, and
    the observation was recorded FAILED -- so a usable ``raw_action`` ("closing
    the lid", "moving the box") was thrown away over a metadata field, and the
    run emitted 2 events where 7 segments had been analysed.

    Dropping both bounds rather than clamping is the honest reading. Clamping
    end to the segment edge invents a boundary the model never reported; an end
    offset of 2.0 on a 0.95s clip says the model's time frame is wrong, which
    discredits its start just as much. ``None`` is a case the pipeline already
    handles -- s07's _timing_precision reports SEGMENT and the event falls back
    to the segment's own bounds.

    One exception, and only one: a bound overshooting by less than the duration of
    a single frame. tt7 is 200 frames at 30fps, so its one whole-clip segment ends
    at 6.666666666666667s; the model, told the clip was "6.67" seconds long by
    _render_prompt's own ``.2f`` rounding, answered 6.67 and the answer was thrown
    away for being 0.0033s past the end. It was the correct answer, rejected by the
    rounding in the question. Within ``1 / fps`` there is no frame to distinguish
    the two values -- the overshoot cannot name a frame the segment does not
    contain -- so the bound is snapped to the segment edge and the snap is
    reported. Beyond one frame period the reasoning above still applies and the
    timing is still dropped: this widens nothing, it only stops a sub-frame
    rounding artifact from costing a whole observation's timing. ``fps`` comes
    from s01's measured video metadata; with no fps supplied nothing is snapped.
    """
    raw_start, raw_end = record.get("start_time_sec"), record.get("end_time_sec")
    if raw_start is None and raw_end is None:
        return None, None, None

    # The schema asks for a float or null, and a model that ignores both writes
    # "UNKNOWN" into the field. That is the same failure as an ill-fitting
    # offset — an unusable timestamp — so it costs the timing, not the answer.
    try:
        start = seg_start + float(raw_start) if raw_start is not None else None
        end = seg_start + float(raw_end) if raw_end is not None else None
    except (TypeError, ValueError):
        return None, None, (
            f"non-numeric offsets {raw_start!r}/{raw_end!r} "
            f"- timing dropped, raw_action kept"
        )

    frame_sec = 1.0 / fps if fps and fps > 0.0 else 0.0
    snapped: list[str] = []

    def _snap(name: str, value: float | None) -> float | None:
        """Pull *value* onto a segment edge it overshoots by under one frame."""
        if value is None or frame_sec <= 0.0:
            return value
        for edge in (seg_start, seg_end):
            if value != edge and abs(value - edge) <= frame_sec:
                # Only ever pulls a value INTO the segment; a bound already
                # inside it is left exactly as the model reported it.
                if (value < seg_start and edge == seg_start) or (
                    value > seg_end and edge == seg_end
                ):
                    snapped.append(
                        f"{name} {value:.6f} -> {edge:.6f} "
                        f"(within one frame, {frame_sec:.6f}s)"
                    )
                    return edge
        return value

    start = _snap("start_time_sec", start)
    end = _snap("end_time_sec", end)

    outside = [
        (name, value)
        for name, value in (("start_time_sec", start), ("end_time_sec", end))
        if value is not None and not (seg_start <= value <= seg_end)
    ]
    # An end before its own start is the same failure wearing a different mask:
    # the offsets are not a coherent interval, so neither is kept.
    if start is not None and end is not None and start > end:
        outside.append(("start_time_sec > end_time_sec", start))

    if outside:
        # Printed at full precision. At 2 decimals the tt7 rejection read
        # "end_time_sec=6.67 outside segment [0.00, 6.67]", which is a
        # self-contradiction: the number that caused it was invisible.
        detail = ", ".join(f"{name}={value:.6f}" for name, value in outside)
        return None, None, (
            f"{detail} outside segment [{seg_start:.6f}, {seg_end:.6f}] "
            f"- timing dropped, raw_action kept"
        )

    return start, end, "; ".join(snapped) if snapped else None


def _write_output(ctx: PipelineContext, status: str = "OK") -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "vlm_observations.json"
    data = [c.model_dump(mode="json") for c in getattr(ctx, "vlm_observations", [])]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Segment bounds are rounded to this many decimals to build the replay join key.
# Bounds come out as 24.900000000000002 from float accumulation, so an exact
# comparison would miss.
_REPLAY_PRECISION = 2


def _bounds_key(start_sec: float, end_sec: float) -> tuple[float, float]:
    return (round(start_sec, _REPLAY_PRECISION), round(end_sec, _REPLAY_PRECISION))


def _replay_observations(
    ctx: PipelineContext, path: str
) -> tuple[list[RawVLMObservation], str | None]:
    """Rebind a prior run's observations onto this run's segments.

    Segment ids cannot be the join key: heuristic_segmenter mints them as
    ``f"cand_{i:04d}_{uuid4().hex[:6]}"``, so the suffix is new every run. The
    time bounds are stable for the same video and are what s07 has to agree
    with, so match on rounded bounds and rewrite segment_id to this run's value.

    Returns ``(observations, error)``. A segment that finds no match is an
    error rather than an omission — replaying nothing silently would present as
    a clean run that happened to produce no events.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return [], f"replay_from file not found: {file_path}"

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], f"replay_from file unreadable: {exc}"

    # s06 writes a bare list; some stages wrap their output in a dict.
    if isinstance(raw, dict):
        raw = raw.get("observations", [])
    if not isinstance(raw, list) or not raw:
        return [], f"replay_from file holds no observations: {file_path}"

    by_bounds: dict[tuple[float, float], list[dict]] = {}
    for record in raw:
        if not isinstance(record, dict):
            continue
        start, end = record.get("segment_start_sec"), record.get("segment_end_sec")
        if start is None or end is None:
            continue
        # A segment may hold several observations (Gemini can return an array),
        # so every match is kept rather than only the first.
        by_bounds.setdefault(_bounds_key(float(start), float(end)), []).append(record)

    observations: list[RawVLMObservation] = []
    missing: list[str] = []
    for seg in ctx.candidate_segments:
        matched = by_bounds.get(_bounds_key(seg.start_sec, seg.end_sec))
        if not matched:
            missing.append(f"{seg.segment_id} [{seg.start_sec:.2f},{seg.end_sec:.2f}]")
            continue
        for record in matched:
            try:
                obs = RawVLMObservation.model_validate(record)
            except ValueError as exc:
                return [], f"replayed observation failed validation: {exc}"
            # Backend and model_name are left as recorded: that is the true
            # provenance of this text. Only the segment binding is rewritten.
            observations.append(obs.model_copy(update={
                "segment_id": seg.segment_id,
                "segment_start_sec": seg.start_sec,
                "segment_end_sec": seg.end_sec,
            }))

    if missing:
        return [], (
            f"replay_from has no observation for {len(missing)} segment(s): "
            + "; ".join(missing)
        )

    return observations, None


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if not hasattr(ctx, "candidate_segments") or ctx.candidate_segments is None:
        msg = "No candidate_segments — s05_segment must run first"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    if not ctx.config.vlm.enabled or ctx.config.stub_mode:
        logger.info("[%s] VLM disabled or stub_mode=True — SKIPPED", STAGE)
        if not hasattr(ctx, "vlm_observations"):
            ctx.vlm_observations = []
        # We can still produce SKIPPED records for each segment if we wanted,
        # but returning empty list is fine for the whole stage being skipped.
        for seg in ctx.candidate_segments:
            obs = RawVLMObservation(
                observation_id=f"obs_{uuid.uuid4().hex[:8]}",
                segment_id=seg.segment_id,
                status=VLMSegmentStatus.SKIPPED,
                error_reason="VLM stage disabled",
                backend="NONE",
                model_name="NONE",
                prompt_version=PROMPT_VERSION,
                segment_start_sec=seg.start_sec,
                segment_end_sec=seg.end_sec
            )
            ctx.vlm_observations.append(obs)
            
        _write_output(ctx, status="SKIPPED")
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="VLM stage skipped",
            duration_sec=time.monotonic() - t0,
        )

    # Replay before any backend is constructed, so no client is built and no
    # request is made.
    replay_path = getattr(ctx.config.vlm, "replay_from", None)
    if replay_path:
        observations, error = _replay_observations(ctx, replay_path)
        if error:
            logger.error("[%s] %s", STAGE, error)
            ctx.vlm_observations = []
            return PipelineStageStatus(
                stage=STAGE, status="ERROR", message=error,
                duration_sec=time.monotonic() - t0,
            )
        ctx.vlm_observations = observations
        _write_output(ctx)
        success_count = sum(
            1 for o in observations if o.status == VLMSegmentStatus.SUCCESS
        )
        logger.info(
            "[%s] REPLAY from %s | %d segments matched by bounds, %d SUCCESS "
            "— no model was called",
            STAGE, replay_path, len(observations), success_count,
        )
        return PipelineStageStatus(
            stage=STAGE, status="OK",
            message=f"Replayed {len(observations)} observations from {replay_path}",
            duration_sec=time.monotonic() - t0,
        )

    # Initialize Backend
    if ctx.config.vlm.backend == "LOCAL_MODEL":
        # config/default.yaml ships backend LOCAL_MODEL / model_name "stub", and
        # src/models/local_vlm.py fabricates a fixed answer without opening the
        # video. A config that sets vlm.enabled=true without naming a backend
        # therefore produces confident nonsense in 0.000s, which a run of tt7 did
        # — "picked up the cup", objects ["white cup"], and there is no cup in
        # that video. It scored EXACT. Nothing objected, so this does.
        if (ctx.config.vlm.model_name or "").lower() == "stub":
            logger.warning(
                "[%s] " + "=" * 60, STAGE,
            )
            logger.warning(
                "[%s] vlm.enabled=true but backend=LOCAL_MODEL and "
                "model_name='stub' — src/models/local_vlm.py is a PLACEHOLDER. "
                "It fabricates an answer WITHOUT READING THE VIDEO. Every "
                "observation below is fiction. Set vlm.backend explicitly "
                "(e.g. GEMINI) if you meant to analyse the video.", STAGE,
            )
            logger.warning(
                "[%s] " + "=" * 60, STAGE,
            )
        vlm = LocalVLM(model_name=ctx.config.vlm.model_name)
    elif ctx.config.vlm.backend == "GEMINI":
        vlm = GeminiVLM(
            model_name=ctx.config.vlm.model_name or "gemini-2.5-flash",
            timeout_sec=ctx.config.vlm.timeout_sec,
        )
    else:
        vlm = RemoteVLM(
            model_name=ctx.config.vlm.model_name,
            api_base_url=ctx.config.vlm.api_base_url,
            timeout_sec=ctx.config.vlm.timeout_sec
        )
        
    ctx.vlm_observations = []

    # Measured by s01 from the container. Used only to size the sub-frame snap in
    # _absolute_timing; absent metadata means no snapping, never a guessed rate.
    fps = ctx.video_metadata.fps if ctx.video_metadata else None

    for seg in ctx.candidate_segments:
        # The model is told this clip's length, so the prompt is per segment.
        prompt_with_test_flags = _render_prompt(seg.end_sec - seg.start_sec)
        if hasattr(ctx, "_test_prompt_flags"):
            prompt_with_test_flags += f"\nTEST_FLAGS: {ctx._test_prompt_flags}"
            
        attempts = 0
        success = False
        last_error = None
        raw_response_text = None
        
        while attempts <= ctx.config.vlm.max_retries and not success:
            attempts += 1
            try:
                raw_response_text = vlm.analyze_segment(
                    video_path=ctx.video_path,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    prompt=prompt_with_test_flags
                )
                
                json_str = extract_json(raw_response_text)
                parsed = json.loads(json_str)

                # A bare object is treated as a one-entry list so both response
                # shapes take a single code path.
                records = parsed if isinstance(parsed, list) else [parsed]
                if not records:
                    raise ValueError("VLM returned an empty observation list")
                if not all(isinstance(r, dict) for r in records):
                    raise ValueError(
                        "VLM observation list contains a non-object entry"
                    )

                # Build and validate every entry before appending any, so a bad
                # second entry cannot leave a half-written segment behind.
                batch: list[RawVLMObservation] = []
                for record in records:
                    data = dict(record)
                    # Convert relative offsets to absolute timestamps BEFORE
                    # validation. Offsets that do not fit the clip cost their
                    # own observation the timing, never the whole answer.
                    start_abs, end_abs, timing_warning = _absolute_timing(
                        data, seg.start_sec, seg.end_sec, fps
                    )
                    if timing_warning:
                        logger.warning(
                            "[%s] segment %s: %s", STAGE, seg.segment_id, timing_warning
                        )
                    # Assigned only where the model supplied the key. Writing
                    # None into an absent key would manufacture the field and
                    # slip past RawVLMObservation's missing-key check, which is
                    # what catches a truncated response.
                    for field in ("start_time_sec", "end_time_sec"):
                        if field in data:
                            data[field] = (
                                start_abs if field == "start_time_sec" else end_abs
                            )

                    batch.append(RawVLMObservation(
                        observation_id=f"obs_{uuid.uuid4().hex[:8]}",
                        segment_id=seg.segment_id,
                        status=VLMSegmentStatus.SUCCESS,
                        backend=vlm.backend,
                        model_name=vlm.model_name,
                        prompt_version=PROMPT_VERSION,
                        segment_start_sec=seg.start_sec,
                        segment_end_sec=seg.end_sec,
                        raw_response=raw_response_text,
                        **data
                    ))

                ctx.vlm_observations.extend(batch)
                success = True
                
            except json.JSONDecodeError as e:
                last_error = f"JSON parsing failed: {e}"
            except ValueError as e:
                last_error = f"Validation failed: {e}"
            except Exception as e:  # noqa: BLE001
                last_error = f"API/Execution failed: {e}"
                
        if not success:
            logger.warning("[%s] Failed to analyze segment %s: %s", STAGE, seg.segment_id, last_error)
            obs = RawVLMObservation(
                observation_id=f"obs_{uuid.uuid4().hex[:8]}",
                segment_id=seg.segment_id,
                status=VLMSegmentStatus.FAILED,
                error_reason=last_error,
                backend=vlm.backend,
                model_name=vlm.model_name,
                prompt_version=PROMPT_VERSION,
                segment_start_sec=seg.start_sec,
                segment_end_sec=seg.end_sec,
                raw_response=raw_response_text
            )
            ctx.vlm_observations.append(obs)

    t_end = time.monotonic()
    duration = t_end - t0

    _write_output(ctx)
    
    success_count = sum(1 for o in ctx.vlm_observations if o.status == VLMSegmentStatus.SUCCESS)
    failed_count = sum(1 for o in ctx.vlm_observations if o.status == VLMSegmentStatus.FAILED)

    # Observations can outnumber segments now that an array response yields one
    # observation per entry, so FAILED is counted rather than subtracted.
    logger.info(
        "[%s] Processed %d segments | %d observations: %d SUCCESS, %d FAILED in %.3fs",
        STAGE, len(ctx.candidate_segments), len(ctx.vlm_observations),
        success_count, failed_count, duration
    )

    msg = f"VLM generated {success_count} successful observations from {len(ctx.candidate_segments)} segments."
    
    return PipelineStageStatus(
        stage=STAGE, status="OK", message=msg, duration_sec=duration
    )
