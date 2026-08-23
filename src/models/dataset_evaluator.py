from src.context import PipelineContext
from src.schema.evaluation import IntegrityIssue


class DatasetEvaluator:
    @staticmethod
    def evaluate(ctx: PipelineContext) -> list[IntegrityIssue]:
        issues: list[IntegrityIssue] = []

        valid_events = {e.event_id for e in ctx.events}
        valid_nodes = {n.node_id for n in ctx.graph_nodes}
        valid_edges = {e.edge_id for e in ctx.graph_edges}
        valid_transitions = {s.transition_id for s in ctx.state_transitions}
        # 1. Referential Integrity & 2. Temporal Consistency
        
        # Events
        for e in ctx.events:
            if e.start_sec > e.end_sec or e.start_sec < 0 or e.end_sec < 0:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="TEMPORAL_CONSISTENCY",
                    message="Event bounds invalid or negative", reference_id=e.event_id
                ))

        # State Transitions
        for s in ctx.state_transitions:
            if s.trigger_event_id and s.trigger_event_id not in valid_events:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="REFERENTIAL_INTEGRITY",
                    message="trigger_event_id not found", reference_id=s.transition_id
                ))
            
            if s.start_sec > s.end_sec or s.start_sec < 0 or s.end_sec < 0:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="TEMPORAL_CONSISTENCY",
                    message="State bounds invalid or negative", reference_id=s.transition_id
                ))
            if s.timing_precision == "SEGMENT":
                issues.append(IntegrityIssue(
                    severity="INFO", dimension="TEMPORAL_CONSISTENCY",
                    message="Timing precision is SEGMENT", reference_id=s.transition_id
                ))
                
            # 3. State Consistency
            if s.from_state == "UNKNOWN" or s.to_state == "UNKNOWN":
                issues.append(IntegrityIssue(
                    severity="INFO", dimension="STATE_CONSISTENCY",
                    message="UNKNOWN state transition", reference_id=s.transition_id
                ))
                
        # Graph Nodes (4. Graph Consistency part 1)
        node_role_map = {n.node_id: n.role for n in ctx.graph_nodes}

        # Graph Edges
        for e in ctx.graph_edges:
            if e.event_id not in valid_events:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="REFERENTIAL_INTEGRITY",
                    message="event_id not found", reference_id=e.edge_id
                ))
            if e.source_node_id not in valid_nodes:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="REFERENTIAL_INTEGRITY",
                    message="source_node_id not found", reference_id=e.edge_id
                ))
            if e.target_node_id not in valid_nodes:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="REFERENTIAL_INTEGRITY",
                    message="target_node_id not found", reference_id=e.edge_id
                ))
            
            # 4. Graph Consistency part 2
            if node_role_map.get(e.source_node_id) != "PERSON":
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="GRAPH_CONSISTENCY",
                    message="Source node must be PERSON", reference_id=e.edge_id
                ))
            if node_role_map.get(e.target_node_id) != "OBJECT":
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="GRAPH_CONSISTENCY",
                    message="Target node must be OBJECT", reference_id=e.edge_id
                ))
                
            if e.actor_resolution in ("UNRESOLVED", "AMBIGUOUS"):
                issues.append(IntegrityIssue(
                    severity="WARNING", dimension="GRAPH_CONSISTENCY",
                    message="Actor identity unresolved", reference_id=e.edge_id
                ))
            if e.object_resolution in ("UNRESOLVED", "AMBIGUOUS"):
                issues.append(IntegrityIssue(
                    severity="WARNING", dimension="GRAPH_CONSISTENCY",
                    message="Object identity unresolved", reference_id=e.edge_id
                ))

            if e.start_sec > e.end_sec or e.start_sec < 0 or e.end_sec < 0:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="TEMPORAL_CONSISTENCY",
                    message="Edge bounds invalid or negative", reference_id=e.edge_id
                ))
            if e.timing_precision == "SEGMENT":
                issues.append(IntegrityIssue(
                    severity="INFO", dimension="TEMPORAL_CONSISTENCY",
                    message="Timing precision is SEGMENT", reference_id=e.edge_id
                ))

        # Quality Scores
        for q in ctx.quality_scores:
            if q.event_id not in valid_events:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="REFERENTIAL_INTEGRITY",
                    message="QualityScore event_id not found", reference_id=q.event_id
                ))

        # Episodes
        for ep in ctx.episodes:
            if ep.start_sec > ep.end_sec or ep.start_sec < 0 or ep.end_sec < 0:
                issues.append(IntegrityIssue(
                    severity="ERROR", dimension="TEMPORAL_CONSISTENCY",
                    message="Episode bounds invalid or negative", reference_id=ep.episode_id
                ))
                
            for ev_id in ep.event_ids:
                if ev_id not in valid_events:
                    issues.append(IntegrityIssue(
                        severity="ERROR", dimension="REFERENTIAL_INTEGRITY",
                        message=f"event_id {ev_id} not found in episode", reference_id=ep.episode_id
                    ))
                else:
                    # 5. Episode Consistency
                    ev = next(e for e in ctx.events if e.event_id == ev_id)
                    if ep.start_sec > ev.start_sec or ep.end_sec < ev.end_sec:
                        issues.append(IntegrityIssue(
                            severity="WARNING", dimension="EPISODE_CONSISTENCY",
                            message="Episode bounds do not contain event bounds", reference_id=ep.episode_id
                        ))
                    
            for eg_id in ep.graph_edge_ids:
                if eg_id not in valid_edges:
                    issues.append(IntegrityIssue(
                        severity="ERROR", dimension="REFERENTIAL_INTEGRITY",
                        message=f"graph_edge_id {eg_id} not found", reference_id=ep.episode_id
                    ))
                    
            for st_id in ep.state_transition_ids:
                if st_id not in valid_transitions:
                    issues.append(IntegrityIssue(
                        severity="ERROR", dimension="REFERENTIAL_INTEGRITY",
                        message=f"state_transition_id {st_id} not found", reference_id=ep.episode_id
                    ))

        return issues
