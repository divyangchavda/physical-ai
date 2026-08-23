"""Smoke test script for Phase 10 — Stage 11 implementation."""
import tempfile
from pathlib import Path

from src.config import PipelineConfig
from src.context import PipelineContext
from src.schema.event import ActionType, PhysicalEvent
from src.schema.interaction_graph import GraphEdge, GraphNode, NodeRole
from src.schema.state import StateTransition
from src.schema.trajectory import Trajectory2D
from src.stages import s11_score


def run_smoke_test():
    print("Running Stage 11 Smoke Test...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config = PipelineConfig(output_dir=tmp_path, stub_mode=False)
        ctx = PipelineContext(config=config, video_path=Path("dummy.mp4"), output_dir=tmp_path)
        
        # High quality event
        e1 = PhysicalEvent(event_id="e_high", action=ActionType.OPEN, source="test", start_sec=0, end_sec=1, attributes={"timing_precision": "EXACT"})
        edge1 = GraphEdge(edge_id="edge_high", source_node_id="n1", target_node_id="n2", action=ActionType.OPEN, start_sec=0, end_sec=1, event_id="e_high", actor_resolution="RESOLVED", object_resolution="RESOLVED", confidence=1.0)
        t1 = StateTransition(transition_id="t1", from_state="CLOSED", to_state="OPEN", trigger_event_id="e_high", start_sec=0, end_sec=1, confidence=1.0, source="test")
        n1 = GraphNode(node_id="n1", role=NodeRole.PERSON, track_id=1)
        n2 = GraphNode(node_id="n2", role=NodeRole.OBJECT, track_id=2)
        traj1 = Trajectory2D(trajectory_id="traj1", track_id=1, source="test")
        traj2 = Trajectory2D(trajectory_id="traj2", track_id=2, source="test")

        # Medium quality event
        e2 = PhysicalEvent(event_id="e_med", action=ActionType.OPEN, source="test", start_sec=0, end_sec=1, attributes={"timing_precision": "SEGMENT"})
        edge2 = GraphEdge(edge_id="edge_med", source_node_id="n3", target_node_id="n4", action=ActionType.OPEN, start_sec=0, end_sec=1, event_id="e_med", actor_resolution="AMBIGUOUS", object_resolution="UNRESOLVED", confidence=1.0)
        t2 = StateTransition(transition_id="t2", from_state="UNKNOWN", to_state="OPEN", trigger_event_id="e_med", start_sec=0, end_sec=1, confidence=1.0, source="test")
        n3 = GraphNode(node_id="n3", role=NodeRole.PERSON, track_id=3)
        traj3 = Trajectory2D(trajectory_id="traj3", track_id=3, source="test")

        # Low quality event
        e3 = PhysicalEvent(event_id="e_low", action=ActionType.UNKNOWN, source="test", start_sec=0, end_sec=1, attributes={"timing_precision": "SEGMENT"})

        ctx.events = [e1, e2, e3]
        ctx.graph_edges = [edge1, edge2]
        ctx.graph_nodes = [n1, n2, n3]
        ctx.state_transitions = [t1, t2]
        ctx.trajectories = [traj1, traj2, traj3]

        status = s11_score.run(ctx)
        
        assert status.status == "OK"
        print("Scores:")
        for score in ctx.quality_scores:
            print(f"  {score.event_id}: {score.quality_tier} (Score: {score.composite_score:.2f})")
            
        assert (tmp_path / "events.json").exists()
        assert (tmp_path / "quality_scores.json").exists()
        print("Smoke test PASSED!")

if __name__ == "__main__":
    run_smoke_test()
