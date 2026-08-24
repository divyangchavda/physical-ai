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

PROMPT = """You are a physical interaction analyst. Review the provided sequence of video frames representing a short temporal segment.
Identify the physical interaction occurring, if any. 
Answer in JSON format ONLY. Do not include markdown formatting, preambles, or explanations.

Schema:
{
  "actor": "description of the person acting, or 'UNKNOWN'",
  "active_hand": "LEFT, RIGHT, BOTH, or 'UNKNOWN'",
  "objects": ["list of primary objects manipulated"],
  "raw_action": "short description of the physical action (e.g., 'picked up the cup')",
  "start_time_sec": <float offset from segment start, or null>,
  "end_time_sec": <float offset from segment start, or null>,
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
"""

def extract_json(text: str) -> str:
    """Extract JSON from a string that might contain markdown blocks."""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    
    # fallback: try finding the first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
        
    return text


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

    by_bounds: dict[tuple[float, float], dict] = {}
    for record in raw:
        if not isinstance(record, dict):
            continue
        start, end = record.get("segment_start_sec"), record.get("segment_end_sec")
        if start is None or end is None:
            continue
        by_bounds.setdefault(_bounds_key(float(start), float(end)), record)

    observations: list[RawVLMObservation] = []
    missing: list[str] = []
    for seg in ctx.candidate_segments:
        record = by_bounds.get(_bounds_key(seg.start_sec, seg.end_sec))
        if record is None:
            missing.append(f"{seg.segment_id} [{seg.start_sec:.2f},{seg.end_sec:.2f}]")
            continue
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
                prompt_version="v1",
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
    
    for seg in ctx.candidate_segments:
        # Prompt injection for test mocks
        prompt_with_test_flags = PROMPT
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
                data = json.loads(json_str)
                
                # Convert relative offsets to absolute timestamps BEFORE validation
                if data.get("start_time_sec") is not None:
                    data["start_time_sec"] = seg.start_sec + float(data["start_time_sec"])
                if data.get("end_time_sec") is not None:
                    data["end_time_sec"] = seg.start_sec + float(data["end_time_sec"])
                    
                obs = RawVLMObservation(
                    observation_id=f"obs_{uuid.uuid4().hex[:8]}",
                    segment_id=seg.segment_id,
                    status=VLMSegmentStatus.SUCCESS,
                    backend=vlm.backend,
                    model_name=vlm.model_name,
                    prompt_version="v1",
                    segment_start_sec=seg.start_sec,
                    segment_end_sec=seg.end_sec,
                    raw_response=raw_response_text,
                    **data
                )
                
                ctx.vlm_observations.append(obs)
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
                prompt_version="v1",
                segment_start_sec=seg.start_sec,
                segment_end_sec=seg.end_sec,
                raw_response=raw_response_text
            )
            ctx.vlm_observations.append(obs)

    t_end = time.monotonic()
    duration = t_end - t0

    _write_output(ctx)
    
    success_count = sum(1 for o in ctx.vlm_observations if o.status == VLMSegmentStatus.SUCCESS)

    logger.info(
        "[%s] Processed %d segments | %d SUCCESS, %d FAILED in %.3fs",
        STAGE, len(ctx.candidate_segments), success_count, len(ctx.candidate_segments) - success_count, duration
    )
    
    msg = f"VLM generated {success_count} successful observations from {len(ctx.candidate_segments)} segments."
    
    return PipelineStageStatus(
        stage=STAGE, status="OK", message=msg, duration_sec=duration
    )
