"""Deterministic quality scoring engine for physical events."""
from __future__ import annotations

from collections.abc import Sequence

from src.config import EventConfig
from src.schema.event import ActionType, PhysicalEvent
from src.schema.interaction_graph import GraphEdge, GraphNode
from src.schema.quality import ComponentScores, EventQualityScore, QualityProvenance
from src.schema.state import StateTransition
from src.schema.trajectory import Trajectory2D

EXPECTED_STATE_TRANSITION_ACTIONS = {
    ActionType.OPEN, ActionType.CLOSE, ActionType.PICK, ActionType.PLACE,
    ActionType.INSERT, ActionType.REMOVE, ActionType.GRASP, ActionType.RELEASE
}


class QualityScorer:
    """Calculates deterministic quality scores for events."""

    @staticmethod
    def score_events(
        events: Sequence[PhysicalEvent],
        edges: Sequence[GraphEdge],
        nodes: Sequence[GraphNode],
        transitions: Sequence[StateTransition],
        trajectories: Sequence[Trajectory2D],
        config: EventConfig,
    ) -> list[EventQualityScore]:
        
        edges_by_event = {e.event_id: e for e in edges}
        nodes_by_id = {n.node_id: n for n in nodes}
        transitions_by_event = {}
        for t in transitions:
            if t.trigger_event_id:
                transitions_by_event.setdefault(t.trigger_event_id, []).append(t)
        traj_track_ids = {t.track_id: t.trajectory_id for t in trajectories}
        
        scores = []
        for event in events:
            edge = edges_by_event.get(event.event_id)
            reasons = []
            
            # Action Certainty (0.10)
            if event.action != ActionType.UNKNOWN:
                action_cert = 1.0
            else:
                action_cert = 0.0
                reasons.append("Action is UNKNOWN")
                
            # Actor & Object Resolution (0.20 each)
            actor_res = 0.0
            object_res = 0.0
            if edge:
                if edge.actor_resolution == "RESOLVED":
                    actor_res = 1.0
                elif edge.actor_resolution == "AMBIGUOUS":
                    actor_res = 0.5
                
                if edge.object_resolution == "RESOLVED":
                    object_res = 1.0
                elif edge.object_resolution == "AMBIGUOUS":
                    object_res = 0.5
                    
                if actor_res < 1.0:
                    reasons.append(f"Actor resolution is {edge.actor_resolution}")
                if object_res < 1.0:
                    reasons.append(f"Object resolution is {edge.object_resolution}")
            else:
                reasons.append("No interaction graph edge found")
                
            # State Evidence (0.20)
            event_transitions = transitions_by_event.get(event.event_id, [])
            best_ev = 0.0
            for t in event_transitions:
                from_unk = (t.from_state == "UNKNOWN")
                to_unk = (t.to_state == "UNKNOWN")
                if not from_unk and not to_unk:
                    score = 1.0
                elif from_unk and to_unk:
                    score = 0.0
                else:
                    score = 0.5
                best_ev = max(best_ev, score)

            if event.action in EXPECTED_STATE_TRANSITION_ACTIONS:
                state_ev = best_ev
                if state_ev < 1.0:
                    reasons.append("Missing or partial state transition evidence for expected action")
            elif event_transitions:
                # An action that does not require a transition still may not earn
                # credit for one reading UNKNOWN->UNKNOWN. That free 1.0 is how
                # the weakest event in a run came out with the highest score.
                state_ev = best_ev
                if state_ev < 1.0:
                    reasons.append("State transition attached to this action carries no state evidence")
            else:
                state_ev = 1.0  # Nothing claimed, so nothing to penalize
                
            # Timing Precision (0.10)
            if event.attributes.get("timing_precision") == "EXACT":
                timing_prec = 1.0
            else:
                timing_prec = 0.5
                reasons.append("Timing precision is not EXACT")
                
            # Trajectory Support (0.20)
            traj_support = 0.0
            participants = []
            if edge:
                actor_node = nodes_by_id.get(edge.source_node_id)
                object_node = nodes_by_id.get(edge.target_node_id)
                if actor_node and actor_node.track_id is not None:
                    participants.append(actor_node.track_id)
                if object_node and object_node.track_id is not None:
                    participants.append(object_node.track_id)
            
            if len(participants) == 0:
                traj_support = 0.0
                reasons.append("No resolved participants for trajectory support")
            elif len(participants) == 1:
                if participants[0] in traj_track_ids:
                    traj_support = 1.0
                else:
                    traj_support = 0.0
                    reasons.append("Participant missing trajectory")
            else:
                count = sum(1 for p in participants if p in traj_track_ids)
                if count == 2:
                    traj_support = 1.0
                elif count == 1:
                    traj_support = 0.5
                    reasons.append("One participant missing trajectory")
                else:
                    traj_support = 0.0
                    reasons.append("Both participants missing trajectories")
                    
            composite = (
                action_cert * 0.10 +
                actor_res * 0.20 +
                object_res * 0.20 +
                state_ev * 0.20 +
                timing_prec * 0.10 +
                traj_support * 0.20
            )
            
            # A component reading 0.0 means one leg of the claim has no evidence
            # at all. The weighted average buries that: a factually wrong event
            # with state_evidence=0.0 still scored 0.75 and auto-accepted. A zero
            # caps the tier at HUMAN_REVIEW however good the composite looks.
            zeroed = [
                name for name, value in (
                    ("action_certainty", action_cert),
                    ("actor_resolution", actor_res),
                    ("object_resolution", object_res),
                    ("state_evidence", state_ev),
                    ("timing_precision", timing_prec),
                    ("trajectory_support", traj_support),
                ) if value == 0.0
            ]

            if composite >= config.auto_accept_threshold and not zeroed:
                tier = "AUTO_ACCEPT"
            elif composite >= config.human_review_threshold:
                tier = "HUMAN_REVIEW"
            else:
                tier = "REJECT"

            if zeroed and composite >= config.auto_accept_threshold:
                reasons.append(
                    "Capped at HUMAN_REVIEW: no evidence for " + ", ".join(zeroed)
                )
                
            provenance = QualityProvenance(
                graph_edge_id=edge.edge_id if edge else None,
                state_transition_ids=[t.transition_id for t in transitions_by_event.get(event.event_id, [])],
                trajectory_ids=[traj_track_ids[p] for p in participants if p in traj_track_ids]
            )
            
            score_obj = EventQualityScore(
                event_id=event.event_id,
                vlm_confidence=event.confidence,
                composite_score=composite,
                quality_tier=tier,
                components=ComponentScores(
                    action_certainty=action_cert,
                    actor_resolution=actor_res,
                    object_resolution=object_res,
                    state_evidence=state_ev,
                    timing_precision=timing_prec,
                    trajectory_support=traj_support
                ),
                provenance=provenance,
                reasons=reasons
            )
            scores.append(score_obj)
            
        return scores
