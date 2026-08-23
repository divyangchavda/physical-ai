"""Stage 07 — Physical event extraction.

Converts VLM output and rule-based signals into typed PhysicalEvent objects.
SKIPPED in stub mode or when no candidate segments are available.

Rules:
  - UNKNOWN action when evidence is genuinely insufficient.
  - Raw action + confidence + source always preserved.
  - review_status left as PENDING (set later by s10_score).

Output file: output/events.json
Output context: ctx.events (list[PhysicalEvent])
"""
from __future__ import annotations

import json
import time
import uuid

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.schema.episode import PipelineStageStatus
from src.schema.event import PhysicalEvent, ActionType
from src.schema.vlm import VLMSegmentStatus

logger = get_logger(__name__)
STAGE = "s07_events"


def _write_output(ctx: PipelineContext) -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "events.json"
    data = [e.model_dump(mode="json") for e in ctx.events]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _map_raw_action_to_type(raw_action: str) -> ActionType:
    """Map VLM raw_action string to canonical ActionType enum.
    
    Rules:
    - Return UNKNOWN when evidence is insufficient or ambiguous
    - Do not force a match; prefer UNKNOWN over incorrect mapping
    """
    if not raw_action:
        return ActionType.UNKNOWN
    
    raw_lower = raw_action.lower()
    
    # Direct keyword matching (order matters: more specific first)
    if "fold" in raw_lower or "assemble" in raw_lower or "assembling" in raw_lower:
        # Folding/assembling involves multiple primitive actions
        # but typically includes closing/forming - map to generic manipulation
        return ActionType.UNKNOWN  # Too complex for single action type
    
    # Check for whole-word or common patterns to avoid substring conflicts
    if "releasing" in raw_lower or "release" in raw_lower or "let go" in raw_lower:
        return ActionType.RELEASE
    if "inserting" in raw_lower or "insert" in raw_lower:
        return ActionType.INSERT
    if "removing" in raw_lower or "remove" in raw_lower:
        return ActionType.REMOVE
    if "picking" in raw_lower or "pick" in raw_lower:
        return ActionType.PICK
    if "placing" in raw_lower or "place" in raw_lower or "put down" in raw_lower:
        return ActionType.PLACE
    if "grasping" in raw_lower or "grasp" in raw_lower or "grip" in raw_lower or "hold" in raw_lower:
        return ActionType.GRASP
    if "pushing" in raw_lower or "push" in raw_lower:
        return ActionType.PUSH
    if "pulling" in raw_lower or "pull" in raw_lower:
        return ActionType.PULL
    if "opening" in raw_lower or "open" in raw_lower:
        return ActionType.OPEN
    if "closing" in raw_lower or "close" in raw_lower:
        return ActionType.CLOSE
    if "using" in raw_lower or "use" in raw_lower:
        return ActionType.USE_TOOL
    if "touching" in raw_lower or "touch" in raw_lower:
        return ActionType.TOUCH
    if "inspecting" in raw_lower or "inspect" in raw_lower or "examine" in raw_lower:
        return ActionType.INSPECT
    if "moving" in raw_lower or "move" in raw_lower:
        return ActionType.MOVE
    
    return ActionType.UNKNOWN


def _extract_events_from_vlm_observations(ctx: PipelineContext) -> list[PhysicalEvent]:
    """Convert VLM observations to PhysicalEvent objects."""
    events = []
    
    # Build segment to track mapping
    segment_track_map = {}
    for seg in ctx.candidate_segments:
        segment_track_map[seg.segment_id] = seg.track_ids
    
    for obs in ctx.vlm_observations:
        if obs.status != VLMSegmentStatus.SUCCESS:
            logger.debug("[%s] Skipping observation %s with status %s", 
                        STAGE, obs.observation_id, obs.status)
            continue
        
        # Extract track IDs from segment
        track_ids = segment_track_map.get(obs.segment_id, [])
        actor_track_id = track_ids[0] if track_ids else None
        
        # Map raw_action to ActionType
        action_type = _map_raw_action_to_type(obs.raw_action or "")
        
        # Create PhysicalEvent
        event = PhysicalEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            segment_id=obs.segment_id,
            observation_id=obs.observation_id,
            action=action_type,
            confidence=obs.confidence if obs.confidence is not None else 0.0,
            source=f"vlm:{obs.backend.lower()}",
            is_estimated=True,
            actor_track_id=actor_track_id,
            object_track_id=None,  # VLM doesn't provide object track IDs yet
            start_sec=obs.start_time_sec if obs.start_time_sec is not None else obs.segment_start_sec,
            end_sec=obs.end_time_sec if obs.end_time_sec is not None else obs.segment_end_sec,
            attributes={
                "raw_action": obs.raw_action,
                "actor": obs.actor,
                "active_hand": obs.active_hand,
                "objects": obs.objects,
                "state_change": obs.state_change,
                "visible_facts": obs.visible_facts,
                "inference": obs.inference,
                "uncertainty": obs.uncertainty,
                "model_name": obs.model_name,
            },
            review_status="PENDING"
        )
        events.append(event)
        
        logger.debug("[%s] Created event %s: action=%s, confidence=%.2f, time=[%.1f, %.1f]s",
                    STAGE, event.event_id, event.action, event.confidence, 
                    event.start_sec, event.end_sec)
    
    return events


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if ctx.config.stub_mode:
        logger.info("[%s] stub_mode=True — SKIPPED (no events fabricated)", STAGE)
        ctx.events = []
        _write_output(ctx)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: event extraction skipped",
            duration_sec=time.monotonic() - t0,
        )

    if not ctx.candidate_segments:
        logger.info("[%s] No candidate segments — SKIPPED", STAGE)
        ctx.events = []
        _write_output(ctx)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="no candidate segments available",
            duration_sec=time.monotonic() - t0,
        )

    # Extract events from VLM observations
    ctx.events = _extract_events_from_vlm_observations(ctx)
    
    _write_output(ctx)
    
    duration = time.monotonic() - t0
    logger.info("[%s] Extracted %d events from %d VLM observations in %.3fs",
               STAGE, len(ctx.events), len(ctx.vlm_observations), duration)
    
    return PipelineStageStatus(
        stage=STAGE, status="OK",
        message=f"Extracted {len(ctx.events)} physical events",
        duration_sec=duration,
    )
