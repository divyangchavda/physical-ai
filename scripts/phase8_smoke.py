"""Smoke test for Stage 09 (Interaction Graph)."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.context import PipelineContext
from src.schema.event import ActionType, PhysicalEvent
from src.schema.state import StateTransition
from src.stages import s09_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_smoke_test():
    config = load_config(set_overrides=[
        "graph_extraction.enabled=true"
    ])
    config.output_dir = Path("output")
    config.output_dir.mkdir(exist_ok=True)
    
    ctx = PipelineContext(config=config, video_path=Path("mock_video.mp4"), output_dir=config.output_dir)
    
    # Mock data
    ctx.events = [
        PhysicalEvent(
            event_id="evt_1",
            segment_id="seg_1",
            observation_id="obs_1",
            action=ActionType.PICK,
            actor_track_id=1,
            object_track_id=2,
            confidence=0.9,
            source="rule_based",
            start_sec=2.0,
            end_sec=3.5
        )
    ]
    
    ctx.state_transitions = [
        StateTransition(
            transition_id="st_1",
            track_id=2,
            from_state="ON_SURFACE",
            to_state="IN_HAND",
            trigger_event_id="evt_1",
            start_sec=2.0,
            end_sec=3.5,
            confidence=0.9,
            source="rule_based"
        )
    ]
    
    logger.info("Running s09_graph...")
    status = s09_graph.run(ctx)
    assert status.status == "OK"
    
    nodes = ctx.graph_nodes
    edges = ctx.graph_edges
    
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0].action == ActionType.PICK
    assert edges[0].actor_resolution == "RESOLVED"
    assert edges[0].object_resolution == "RESOLVED"
    assert "st_1" in edges[0].state_transition_ids
    
    logger.info("Phase 8 Smoke Test Passed!")

if __name__ == "__main__":
    run_smoke_test()
