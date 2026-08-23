"""Object State Inferencer rule-engine."""

import logging
import re
import uuid
from typing import Literal

from src.schema.event import ActionType, PhysicalEvent
from src.schema.state import StateTransition
from src.schema.track import Track
from src.schema.vlm import RawVLMObservation

logger = logging.getLogger(__name__)

class StateInferencer:
    """Infers temporal object-state transitions deterministically based on evidence."""
    
    def __init__(self):
        pass

    def _resolve_identity(
        self, 
        evt: PhysicalEvent, 
        obs: RawVLMObservation | None, 
        tracks: list[Track]
    ) -> tuple[int | None, str | None, Literal["RESOLVED", "UNRESOLVED", "AMBIGUOUS"]]:
        """Resolve object identity conservatively."""
        if evt.object_track_id is not None:
            return evt.object_track_id, None, "RESOLVED"
            
        if not obs or not obs.objects:
            return None, None, "UNRESOLVED"
            
        semantic_label = obs.objects[0]
        
        # Look for tracks in segment matching semantic_label
        matches = [t for t in tracks if semantic_label.lower() in t.class_name.lower()]
        
        if len(matches) == 1:
            return matches[0].track_id, semantic_label, "RESOLVED"
        elif len(matches) > 1:
            return None, semantic_label, "AMBIGUOUS"
            
        return None, semantic_label, "UNRESOLVED"

    def _has_contradiction(self, action: ActionType, obs: RawVLMObservation | None) -> bool:
        """Check for explicit contradictions in VLM/tracking evidence."""
        if not obs:
            return False
            
        all_evidence = f"{obs.visible_facts or ''} {obs.state_change or ''} {obs.inference or ''} {obs.uncertainty or ''}".lower()
        
        if action == ActionType.PICK:
            if re.search(r'\b(stationary|not lifted|no pickup|did not lift|did not move|did not pick)\b', all_evidence):
                return True
        elif action == ActionType.PLACE:
            if re.search(r'\b(kept holding|not placed|no placement|did not place)\b', all_evidence):
                return True
        elif action == ActionType.OPEN:
            if re.search(r'\b(already open|not opened|remained closed)\b', all_evidence):
                return True
        elif action == ActionType.CLOSE and re.search(r'\b(already closed|not closed|remained open)\b', all_evidence):
            return True
                
        return False

    def infer_transitions(
        self, 
        events: list[PhysicalEvent], 
        obs_map: dict[str, RawVLMObservation],
        tracks: list[Track]
    ) -> list[StateTransition]:
        """Infer state transitions from a sequence of events (typically within a segment)."""
        transitions = []
        
        # Sort events by start_sec
        sorted_events = sorted(events, key=lambda e: e.start_sec)
        
        current_states = {}

        for evt in sorted_events:
            obs = obs_map.get(evt.observation_id) if evt.observation_id else None
            
            track_id, sem_label, res_status = self._resolve_identity(evt, obs, tracks)
            obj_key = (track_id, sem_label)
            
            before_state = "UNKNOWN"
            after_state = "UNKNOWN"
            
            # Use tracked state if available
            if obj_key in current_states:
                before_state = current_states[obj_key]
                
            all_evidence = ""
            if obs:
                all_evidence = f"{obs.visible_facts or ''} {obs.state_change or ''} {obs.inference or ''} {obs.uncertainty or ''}".lower()

            # Rule 1: Contradictions force UNKNOWN -> UNKNOWN
            if self._has_contradiction(evt.action, obs):
                before_state = "UNKNOWN"
                after_state = "UNKNOWN"
            else:
                # Rule 2: Evaluate states independently without implied states
                if evt.action in (ActionType.PICK, ActionType.GRASP):
                    # Explicit evidence required for IN_HAND
                    if re.search(r'\b(in hand|held|holding|lifted|pick|grab)', all_evidence):
                        after_state = "IN_HAND"
                    
                    if before_state == "UNKNOWN" and re.search(r'\b(off table|from table|off surface|from surface|picked up from|grabbed from)', all_evidence):
                        before_state = "ON_SURFACE"
                        
                elif evt.action in (ActionType.PLACE, ActionType.RELEASE):
                    # Explicit evidence required for ON_SURFACE
                    if re.search(r'\b(on table|on surface|put|set|place|drop)', all_evidence):
                        after_state = "ON_SURFACE"
                        
                    if before_state == "UNKNOWN" and re.search(r'\b(was holding|from hand|held)', all_evidence):
                        before_state = "IN_HAND"
                        
                elif evt.action == ActionType.OPEN:
                    if re.search(r'\b(is open|opened)\b', all_evidence) or evt.action == ActionType.OPEN:
                        after_state = "OPEN"
                    if before_state == "UNKNOWN" and re.search(r'\b(was closed|from closed|opened the closed)\b', all_evidence):
                        before_state = "CLOSED"
                        
                elif evt.action == ActionType.CLOSE:
                    if re.search(r'\b(is closed|closed|shuts)\b', all_evidence) or evt.action == ActionType.CLOSE:
                        after_state = "CLOSED"
                    if before_state == "UNKNOWN" and re.search(r'\b(was open|from open|closed the open)\b', all_evidence):
                        before_state = "OPEN"

            current_states[obj_key] = after_state

            timing_precision = evt.attributes.get("timing_precision", "EXACT")
            
            # Note: "SEGMENT" is just a string, literal match handles it correctly in pydantic
            if timing_precision not in ["EXACT", "SEGMENT"]:
                timing_precision = "SEGMENT"

            evidence_dict = {}
            if obs:
                evidence_dict["raw_action"] = obs.raw_action
                evidence_dict["visible_facts"] = obs.visible_facts
                evidence_dict["state_change"] = obs.state_change
            
            trans = StateTransition(
                transition_id=f"st_{uuid.uuid4().hex[:8]}",
                track_id=track_id,
                semantic_label=sem_label,
                identity_resolution=res_status,
                from_state=before_state,
                to_state=after_state,
                trigger_event_id=evt.event_id,
                observation_id=evt.observation_id,
                start_sec=evt.start_sec,
                end_sec=evt.end_sec,
                timing_precision=timing_precision,
                confidence=evt.confidence,
                source="rule_based",
                is_estimated=True,
                evidence=evidence_dict
            )
            transitions.append(trans)

        return transitions
