from pathlib import Path

from src.config import PipelineConfig
from src.context import PipelineContext
from src.models.dataset_evaluator import DatasetEvaluator
from src.schema.episode import InteractionEpisode
from src.schema.event import PhysicalEvent
from src.schema.interaction_graph import GraphEdge, GraphNode, NodeRole
from src.schema.quality import ComponentScores, EventQualityScore, QualityProvenance
from src.schema.state import StateTransition


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

def test_valid_full_context():
    ctx = get_base_ctx()
    issues = DatasetEvaluator.evaluate(ctx)
    assert len(issues) == 0

def test_broken_reference_trigger_event_id():
    ctx = get_base_ctx()
    ctx.state_transitions[0].trigger_event_id = "bad"
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "REFERENTIAL_INTEGRITY" for i in issues)

def test_broken_reference_graphedge_event_id():
    ctx = get_base_ctx()
    ctx.graph_edges[0].event_id = "bad"
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "REFERENTIAL_INTEGRITY" for i in issues)

def test_broken_reference_graphedge_source_node_id():
    ctx = get_base_ctx()
    ctx.graph_edges[0].source_node_id = "bad"
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "REFERENTIAL_INTEGRITY" for i in issues)

def test_broken_reference_graphedge_target_node_id():
    ctx = get_base_ctx()
    ctx.graph_edges[0].target_node_id = "bad"
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "REFERENTIAL_INTEGRITY" for i in issues)

def test_broken_reference_qualityscore_event_id():
    ctx = get_base_ctx()
    ctx.quality_scores[0].event_id = "bad"
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "REFERENTIAL_INTEGRITY" for i in issues)

def test_broken_reference_episode_event_ids():
    ctx = get_base_ctx()
    ctx.episodes[0].event_ids = ["bad"]
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "REFERENTIAL_INTEGRITY" for i in issues)

def test_broken_reference_episode_graph_edge_ids():
    ctx = get_base_ctx()
    ctx.episodes[0].graph_edge_ids = ["bad"]
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "REFERENTIAL_INTEGRITY" for i in issues)

def test_broken_reference_episode_state_transition_ids():
    ctx = get_base_ctx()
    ctx.episodes[0].state_transition_ids = ["bad"]
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "REFERENTIAL_INTEGRITY" for i in issues)

def test_temporal_inversion_events():
    ctx = get_base_ctx()
    ctx.events[0].start_sec = 2.0
    ctx.events[0].end_sec = 1.0
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_temporal_inversion_state_transitions():
    ctx = get_base_ctx()
    ctx.state_transitions[0].start_sec = 2.0
    ctx.state_transitions[0].end_sec = 1.0
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_temporal_inversion_graph_edges():
    ctx = get_base_ctx()
    ctx.graph_edges[0].start_sec = 2.0
    ctx.graph_edges[0].end_sec = 1.0
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_temporal_inversion_episodes():
    ctx = get_base_ctx()
    ctx.episodes[0].start_sec = 2.0
    ctx.episodes[0].end_sec = 1.0
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_negative_bound_events():
    ctx = get_base_ctx()
    ctx.events[0].start_sec = -1.0
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_negative_bound_state_transitions():
    ctx = get_base_ctx()
    ctx.state_transitions[0].start_sec = -1.0
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_negative_bound_graph_edges():
    ctx = get_base_ctx()
    ctx.graph_edges[0].start_sec = -1.0
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_negative_bound_episodes():
    ctx = get_base_ctx()
    ctx.episodes[0].start_sec = -1.0
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_unknown_state_is_info():
    ctx = get_base_ctx()
    ctx.state_transitions[0].from_state = "UNKNOWN"
    issues = DatasetEvaluator.evaluate(ctx)
    assert not any(i.severity == "ERROR" for i in issues)
    assert any(i.severity == "INFO" and i.dimension == "STATE_CONSISTENCY" for i in issues)

def test_segment_timing_precision_edge_is_info():
    ctx = get_base_ctx()
    ctx.graph_edges[0].timing_precision = "SEGMENT"
    issues = DatasetEvaluator.evaluate(ctx)
    assert not any(i.severity == "ERROR" for i in issues)
    assert any(i.severity == "INFO" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_segment_timing_precision_state_is_info():
    ctx = get_base_ctx()
    ctx.state_transitions[0].timing_precision = "SEGMENT"
    issues = DatasetEvaluator.evaluate(ctx)
    assert not any(i.severity == "ERROR" for i in issues)
    assert any(i.severity == "INFO" and i.dimension == "TEMPORAL_CONSISTENCY" for i in issues)

def test_graph_source_node_not_person():
    ctx = get_base_ctx()
    ctx.graph_nodes[0].role = NodeRole.OBJECT
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "GRAPH_CONSISTENCY" for i in issues)

def test_graph_target_node_not_object():
    ctx = get_base_ctx()
    ctx.graph_nodes[1].role = NodeRole.PERSON
    issues = DatasetEvaluator.evaluate(ctx)
    assert any(i.severity == "ERROR" and i.dimension == "GRAPH_CONSISTENCY" for i in issues)

def test_graph_node_identity_unresolved_is_warning():
    ctx = get_base_ctx()
    ctx.graph_edges[0].actor_resolution = "UNRESOLVED"
    issues = DatasetEvaluator.evaluate(ctx)
    assert not any(i.severity == "ERROR" for i in issues)
    assert any(i.severity == "WARNING" and i.dimension == "GRAPH_CONSISTENCY" for i in issues)

def test_episode_bounds_mismatch_is_warning():
    ctx = get_base_ctx()
    ctx.episodes[0].start_sec = 1.5
    issues = DatasetEvaluator.evaluate(ctx)
    assert not any(i.severity == "ERROR" for i in issues)
    assert any(i.severity == "WARNING" and i.dimension == "EPISODE_CONSISTENCY" for i in issues)

def test_upstream_immutability():
    ctx = get_base_ctx()
    import copy
    ctx_copy = copy.deepcopy(ctx)
    DatasetEvaluator.evaluate(ctx)
    assert ctx == ctx_copy

def test_stub_mode_returns_info():
    from src.stages.s13_evaluate import run
    ctx = get_base_ctx()
    ctx.config.stub_mode = True
    run(ctx)
    assert any(i.severity == "INFO" and i.message == "stub_mode=True, skipped deep evaluation" for i in ctx.evaluation.integrity_issues)
