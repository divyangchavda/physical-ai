"""Deterministic normalization of VLM observations into PhysicalEvents."""

import re
import uuid

from src.schema.event import ActionType, PhysicalEvent
from src.schema.vlm import RawVLMObservation


class ActionNormalizer:
    """Deterministically normalizes raw VLM observations into PhysicalEvent objects."""

    def normalize(self, obs: RawVLMObservation) -> list[PhysicalEvent]:
        """Convert a single RawVLMObservation into one or more PhysicalEvents."""
        
        # 1. Handle missing / explicitly unknown raw action
        if not obs.raw_action or obs.raw_action.strip().upper() == "UNKNOWN":
            return [self._build_event(obs, ActionType.UNKNOWN)]
            
        raw = obs.raw_action.lower()
        facts = (obs.visible_facts or "").lower()
        state = (obs.state_change or "").lower()
        inf = (obs.inference or "").lower()
        unc = (obs.uncertainty or "").lower()
        
        # Combine all evidence for general conflict checking
        all_evidence = f"{facts} {inf} {unc}"
        
        # 2. Check for multiple events (simplistic decomposition)
        # Split on ", and ", ", then ", " and ", " then ", or just ","
        if re.search(r'\b(and|then)\b', raw) or ',' in raw:
            splits = re.split(r',\s+(?:and\s+|then\s+)?|\s+and\s+|\s+then\s+', raw)
            splits = [s.strip() for s in splits if s.strip()]
            
            if len(splits) > 1:
                actions = []
                for split in splits:
                    act = self._evaluate_single_action(split, facts, state, inf, unc, all_evidence)
                    actions.append(act)
                
                # If we successfully parsed ALL splits into non-UNKNOWN canonical actions
                if all(a != ActionType.UNKNOWN for a in actions):
                    # Deduplicate consecutive identical actions (e.g., "picked up and lifted")
                    unique_actions = []
                    for a in actions:
                        if not unique_actions or unique_actions[-1] != a:
                            unique_actions.append(a)
                    
                    if len(unique_actions) > 1:
                        return [self._build_event(obs, act, segment_timing=True) for act in unique_actions]
            
            # If decomposition failed or wasn't clean, fallback to UNKNOWN for the whole observation.
            return [self._build_event(obs, ActionType.UNKNOWN)]
            
        # 3. Single event evaluation
        action_type = self._evaluate_single_action(raw, facts, state, inf, unc, all_evidence)
        return [self._build_event(obs, action_type)]

    def _evaluate_single_action(
        self, raw: str, facts: str, state: str, inf: str, unc: str, all_evidence: str
    ) -> ActionType:
        """Evaluate a single action string against the evidence."""
        
        # Check for uncertainty override
        if "cannot see" in unc or "occluded" in unc or "unclear" in unc:
            return ActionType.UNKNOWN
            
        # PICK & GRASP logic
        if re.search(r'\b(pick|lift|raise)', raw):
            # PICK requires explicit upward movement evidence
            if re.search(r'\b(lift|up|upward|rise|raise)', all_evidence):
                # Check for contradiction
                if re.search(r'\b(did not move|no movement|stationary)\b', all_evidence):
                    return ActionType.UNKNOWN
                return ActionType.PICK
            else:
                # "picked up cup" + no lifting evidence -> UNKNOWN
                return ActionType.UNKNOWN

        if re.search(r'\b(grasp|grab|hold|take hold|took hold)', raw):
            # Contradiction check
            if re.search(r'\b(did not grasp|no hold|dropped)', all_evidence):
                return ActionType.UNKNOWN
            # Upgrade check: if evidence explicitly shows lifting, upgrade to PICK
            if re.search(r'\b(lift|up|upward|rise|raise)', facts):
                return ActionType.PICK
            return ActionType.GRASP

        # PLACE
        if re.search(r'\b(put|set|place|drop)', raw):
            if re.search(r'\b(did not put|kept holding|no placement)', all_evidence):
                return ActionType.UNKNOWN
            return ActionType.PLACE

        # OPEN & CLOSE
        if re.search(r'\b(open)', raw):
            if "already open" in state or "already open" in facts:
                return ActionType.UNKNOWN
            return ActionType.OPEN
            
        if re.search(r'\b(close|shut)', raw):
            if "already closed" in state or "already closed" in facts:
                return ActionType.UNKNOWN
            return ActionType.CLOSE
            
        # State-only fallback check: "door is open" without action transition
        if " is open" in raw or " is closed" in raw:
            return ActionType.UNKNOWN

        # MOVE
        if re.search(r'\b(move|slide|carry)', raw):
            # Must confirm the object moved, not just the hand
            if "object was moved" in all_evidence or "moved the object" in all_evidence or "object moves" in all_evidence:
                return ActionType.MOVE
            # False keyword: "moved hand" -> NOT MOVE (unless object movement supported)
            if "moved hand" in raw and not re.search(r'\b(object|it)\b', all_evidence):
                return ActionType.UNKNOWN
            return ActionType.UNKNOWN # require strict evidence for MOVE
            
        # PUSH / PULL
        if re.search(r'\b(push|shove)', raw):
            return ActionType.PUSH
        if re.search(r'\b(pull|drag)', raw):
            return ActionType.PULL
            
        # TOUCH / INSPECT
        if re.search(r'\b(touch|tap)', raw):
            return ActionType.TOUCH
        if re.search(r'\b(look|inspect|examine)', raw):
            if "inspect" in all_evidence or "examine" in all_evidence or "look closely" in all_evidence:
                return ActionType.INSPECT
            return ActionType.UNKNOWN

        # Default fallback
        return ActionType.UNKNOWN

    def _build_event(self, obs: RawVLMObservation, action_type: ActionType, segment_timing: bool = False) -> PhysicalEvent:
        """Construct a PhysicalEvent from a RawVLMObservation and derived ActionType."""
        # Determine timestamps
        timing_precision = "EXACT"
        start_sec = obs.start_time_sec
        end_sec = obs.end_time_sec
        
        if start_sec is None or end_sec is None or segment_timing:
            start_sec = obs.segment_start_sec
            end_sec = obs.segment_end_sec
            timing_precision = "SEGMENT"
            
        # Extract attributes for provenance
        attributes = {
            "vlm_raw_action": obs.raw_action,
            "vlm_visible_facts": obs.visible_facts,
            "vlm_uncertainty": obs.uncertainty,
            "timing_precision": timing_precision
        }
        
        return PhysicalEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            segment_id=obs.segment_id,
            observation_id=obs.observation_id,
            action=action_type,
            confidence=obs.confidence if obs.confidence is not None else 0.0,
            source=f"vlm_normalized:{obs.model_name}",
            is_estimated=True,
            start_sec=start_sec,
            end_sec=end_sec,
            attributes=attributes
        )
