from pathlib import Path

from src.config import PipelineConfig
from src.context import PipelineContext
from src.schema.episode import InteractionEpisode
from src.schema.event import PhysicalEvent
from src.schema.interaction_graph import GraphEdge, GraphNode, NodeRole
from src.schema.quality import ComponentScores, EventQualityScore, QualityProvenance
from src.schema.state import StateTransition
from src.stages.s13_evaluate import run


def get_base_ctx():
    return PipelineContext(
        config=PipelineConfig(),
        video_path=Path("dummy.mp4"),
        output_dir=Path("out"),
        events=[
            PhysicalEvent(event_id="e1", action="GRASP", source="vlm", start_sec=1.0, end_sec=2.0)
        ],
        graph_nodes=[
            GraphNode(node_id="n1", role=NodeRole.PERSON),
            GraphNode(node_id="n2", role=NodeRole.OBJECT)
        ],
        graph_edges=[
            GraphEdge(edge_id="ge1", source_node_id="n1", target_node_id="n2", action="GRASP", event_id="e1", start_sec=1.0, end_sec=2.0, confidence=1.0, actor_resolution="RESOLVED", object_resolution="RESOLVED", timing_precision="EXACT")
        ],
        state_transitions=[
            StateTransition(transition_id="st1", from_state="OPEN", to_state="CLOSED", trigger_event_id="e1", start_sec=1.0, end_sec=2.0, source="vlm", confidence=1.0, timing_precision="EXACT")
        ],
        quality_scores=[
            EventQualityScore(event_id="e1", vlm_confidence=1.0, composite_score=1.0, quality_tier="AUTO_ACCEPT", components=ComponentScores(action_certainty=1.0, actor_resolution=1.0, object_resolution=1.0, state_evidence=1.0, timing_precision=1.0, trajectory_support=1.0), provenance=QualityProvenance())
        ],
        episodes=[
            InteractionEpisode(episode_id="ep1", event_ids=["e1"], start_sec=0.0, end_sec=3.0, timing_precision="EXACT", graph_edge_ids=["ge1"], state_transition_ids=["st1"])
        ]
    )

def main():
    print("Running phase12 smoke tests...")
    
    # 1. Seed valid state (asserts HEALTHY)
    ctx1 = get_base_ctx()
    run(ctx1)
    assert ctx1.evaluation.dataset_health == "HEALTHY", f"Expected HEALTHY, got {ctx1.evaluation.dataset_health}"
    print("[PASS] Valid state -> HEALTHY")

    # 2. Inject broken reference (asserts CRITICAL)
    ctx2 = get_base_ctx()
    ctx2.graph_edges[0].event_id = "nonexistent"
    run(ctx2)
    assert ctx2.evaluation.dataset_health == "CRITICAL", f"Expected CRITICAL, got {ctx2.evaluation.dataset_health}"
    print("[PASS] Broken reference -> CRITICAL")

    # 3. Inject temporal inversion (asserts CRITICAL)
    ctx3 = get_base_ctx()
    ctx3.events[0].start_sec = 5.0
    ctx3.events[0].end_sec = 2.0
    run(ctx3)
    assert ctx3.evaluation.dataset_health == "CRITICAL", f"Expected CRITICAL, got {ctx3.evaluation.dataset_health}"
    print("[PASS] Temporal inversion -> CRITICAL")

    print("All smoke tests passed!")

if __name__ == "__main__":
    main()
