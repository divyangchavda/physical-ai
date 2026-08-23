"""Smoke test for Stage 14."""
import json
from pathlib import Path

from src.config import PipelineConfig
from src.context import PipelineContext
from src.schema.episode import InteractionEpisode
from src.schema.event import ActionType, PhysicalEvent
from src.schema.interaction_graph import GraphEdge, GraphNode, NodeRole
from src.stages.s14_preview import run


def main():
    config = PipelineConfig(stub_mode=False)
    ctx = PipelineContext(
        config=config,
        video_path=Path("dummy.mp4"),
        output_dir=Path("output")
    )
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    
    ctx.events = [
        PhysicalEvent(
            event_id="e1", action=ActionType.PICK, source="test",
            start_sec=0.0, end_sec=1.0, is_estimated=False
        ),
        PhysicalEvent(
            event_id="e2", action=ActionType.UNKNOWN, source="test",
            start_sec=1.0, end_sec=2.0, is_estimated=False
        )
    ]
    ctx.graph_nodes = [
        GraphNode(node_id="n1", role=NodeRole.OBJECT, semantic_label="cup")
    ]
    ctx.graph_edges = [
        GraphEdge(
            edge_id="edge1", source_node_id="n_actor", target_node_id="n1",
            action=ActionType.PICK, start_sec=0.0, end_sec=1.0,
            event_id="e1", confidence=1.0
        )
    ]
    ctx.episodes = [
        InteractionEpisode(
            episode_id="ep1", event_ids=["e1", "e2"], start_sec=0.0, end_sec=2.0,
            timing_precision="EXACT"
        )
    ]
    
    original_events_len = len(ctx.events)
    run(ctx)
    assert len(ctx.events) == original_events_len, "Immutability violated!"
    
    out_file = ctx.output_dir / "preview.json"
    assert out_file.exists(), "preview.json not generated!"
    
    with open(out_file) as f:
        data = json.load(f)
        
    assert data["counts"]["total_events"] == 2
    assert data["counts"]["total_episodes"] == 1
    assert data["timeline"][0]["human_readable_events"] == ["PICK cup", "UNKNOWN"]
    
    print("Smoke test passed.")

if __name__ == "__main__":
    main()
