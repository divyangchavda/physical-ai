"""Graph Builder - constructs the deterministic interaction graph from pipeline outputs."""
from __future__ import annotations

import logging
from collections import defaultdict

from src.context import PipelineContext
from src.schema.event import PhysicalEvent
from src.schema.interaction_graph import GraphEdge, GraphNode, NodeRole
from src.schema.state import StateTransition
from src.schema.track import Track
from src.schema.vlm import RawVLMObservation

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds the Interaction Graph by deterministically reconciling events and tracks."""

    def __init__(self) -> None:
        pass

    def build(self, ctx: PipelineContext) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Construct graph nodes and edges from context.

        Returns:
            Tuple of (nodes, edges)
        """
        nodes: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []
        
        # Group tracks by segment
        segment_to_tracks: dict[str, list[Track]] = defaultdict(list)
        for seg in ctx.candidate_segments:
            seg_tracks = [t for t in ctx.tracks if t.track_id in seg.track_ids]
            segment_to_tracks[seg.segment_id] = seg_tracks
            
        # Group observations by observation_id
        obs_map: dict[str, RawVLMObservation] = {
            obs.observation_id: obs for obs in ctx.vlm_observations
        }
        
        # Group state transitions by trigger_event_id
        event_to_states: dict[str, list[StateTransition]] = defaultdict(list)
        for st in ctx.state_transitions:
            if st.trigger_event_id:
                event_to_states[st.trigger_event_id].append(st)
                
        for event in ctx.events:
            # 1. Gather evidence
            obs = obs_map.get(event.observation_id) if event.observation_id else None
            tracks = segment_to_tracks.get(event.segment_id, []) if event.segment_id else []
            state_transitions = event_to_states.get(event.event_id, [])
            
            # 2. Resolve Actor
            actor_track_id, actor_label, actor_resolution = self._resolve_actor(event, obs, tracks)
            
            # 3. Resolve Object(s)
            object_targets = self._resolve_objects(event, obs, tracks, state_transitions)
            
            # 4. Handle Unresolved Actor + Object case
            if actor_track_id is None and not object_targets:
                # If neither is resolved, we still want to preserve an unresolved edge
                # if there is SOME evidence of interaction
                object_targets = [(None, None, "UNRESOLVED")]
            
            # Create Actor Node
            actor_node_id = self._get_node_id(actor_track_id, actor_label, event.observation_id)
            if actor_node_id not in nodes:
                nodes[actor_node_id] = GraphNode(
                    node_id=actor_node_id,
                    role=NodeRole.PERSON,
                    track_id=actor_track_id,
                    semantic_label=actor_label
                )
                
            # Create Edges (one per object target)
            for (obj_track_id, obj_label, obj_resolution) in object_targets:
                # Create Object Node
                obj_node_id = self._get_node_id(obj_track_id, obj_label, event.observation_id)
                if obj_node_id not in nodes:
                    nodes[obj_node_id] = GraphNode(
                        node_id=obj_node_id,
                        role=NodeRole.OBJECT,
                        track_id=obj_track_id,
                        semantic_label=obj_label
                    )
                
                # Match State Transitions for this target
                st_ids = []
                for st in state_transitions:
                    # Match by track_id if resolved, else by label
                    if obj_track_id is not None and st.track_id == obj_track_id or obj_track_id is None and obj_label and st.semantic_label == obj_label:
                        st_ids.append(st.transition_id)
                
                edge_id = f"edge_{event.event_id}_{actor_node_id}_{obj_node_id}"
                
                edges.append(GraphEdge(
                    edge_id=edge_id,
                    source_node_id=actor_node_id,
                    target_node_id=obj_node_id,
                    action=event.action,
                    actor_resolution=actor_resolution,
                    object_resolution=obj_resolution,
                    start_sec=event.start_sec,
                    end_sec=event.end_sec,
                    timing_precision=event.attributes.get("timing_precision", "SEGMENT"),
                    event_id=event.event_id,
                    observation_id=event.observation_id,
                    segment_id=event.segment_id,
                    state_transition_ids=st_ids,
                    confidence=event.confidence
                ))

        return list(nodes.values()), edges
        
    def _get_node_id(self, track_id: int | None, semantic_label: str | None, obs_id: str | None) -> str:
        if track_id is not None:
            return f"node_track_{track_id}"
        label = semantic_label or "unknown"
        o_id = obs_id or "no_obs"
        return f"node_unresolved_{o_id}_{label}"
        
    def _resolve_actor(self, event: PhysicalEvent, obs: RawVLMObservation | None, tracks: list[Track]) -> tuple[int | None, str | None, str]:
        semantic_label = obs.actor if obs else None
        
        # Rule 1: Explicit ID
        if event.actor_track_id is not None:
            return event.actor_track_id, semantic_label, "RESOLVED"
            
        # Rule 2: Multiple visible people -> AMBIGUOUS
        person_tracks = [t for t in tracks if t.class_name == "person"]
        if len(person_tracks) > 1:
            # We don't nearest-neighbor match.
            return None, semantic_label, "AMBIGUOUS"
            
        # Rule 3: Single person track -> UNRESOLVED (not sufficient evidence)
        # We NEVER resolve just because there's 1 person.
        # It must be explicit.
        return None, semantic_label, "UNRESOLVED"

    def _resolve_objects(self, event: PhysicalEvent, obs: RawVLMObservation | None, tracks: list[Track], states: list[StateTransition]) -> list[tuple[int | None, str | None, str]]:
        targets: list[tuple[int | None, str | None, str]] = []
        
        # Rule 1: Explicit ID
        if event.object_track_id is not None:
            # Try to get the label from the track or observation
            label = None
            if obs and obs.objects:
                label = obs.objects[0]
            for t in tracks:
                if t.track_id == event.object_track_id:
                    label = label or t.class_name
            targets.append((event.object_track_id, label, "RESOLVED"))
            return targets
            
        # If no explicit object_track_id, we look for explicitly supported targets.
        # "Explicitly supported" means either we have state transitions for it, 
        # or we have explicit semantic relationships. 
        # But we do NOT just take every object in obs.objects.
        
        # If we have state transitions linked to this event, they are strong evidence of targets
        if states:
            for st in states:
                targets.append((st.track_id, st.semantic_label, st.identity_resolution))
            
            # Deduplicate by track_id/label
            unique_targets = {}
            for t_id, lbl, res in targets:
                key = str(t_id) if t_id is not None else lbl
                unique_targets[key] = (t_id, lbl, res)
            return list(unique_targets.values())
            
        # If no state transitions and no explicit ID, we preserve unresolved semantic objects
        # BUT we must be careful not to expand context objects.
        # For MVP, if there is no explicit PhysicalEvent or State evidence linking to a specific object, 
        # we consider the target UNRESOLVED or AMBIGUOUS.
        
        if obs and obs.objects:
            if len(obs.objects) == 1:
                # Single object in VLM, but no explicit linkage
                targets.append((None, obs.objects[0], "UNRESOLVED"))
            else:
                # Multiple objects in VLM, but no explicit linkage
                # We do NOT create multiple edges unless explicitly supported.
                # So we create a single AMBIGUOUS target.
                targets.append((None, "multiple_ambiguous", "AMBIGUOUS"))
        else:
            targets.append((None, None, "UNRESOLVED"))
            
        return targets
