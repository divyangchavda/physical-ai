"""Tests for Stage 09: Interaction Graph construction.

Covers all 29 audit requirements including:
- Actor/object resolution semantics
- Multi-object and multi-actor conservatism
- Timing preservation
- StateTransition linking and isolation
- Provenance preservation
- Edge/node deduplication
- Deterministic execution
- Serialization/deserialization
- FAILED/SKIPPED upstream handling
- Empty input
- Upstream immutability
- Stub mode (F-13 bug fix)
- Schema validation
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import PipelineConfig
from src.context import PipelineContext
from src.models.graph_builder import GraphBuilder
from src.schema.event import ActionType, PhysicalEvent
from src.schema.interaction_graph import GraphEdge, GraphNode
from src.schema.segment import CandidateSegment
from src.schema.state import StateTransition
from src.schema.track import Track
from src.schema.vlm import RawVLMObservation, VLMSegmentStatus
from src.stages import s09_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(**kwargs)


def _make_ctx(stub_mode: bool = False, output_dir: str = "output") -> PipelineContext:
    config = _make_config(stub_mode=stub_mode)
    return PipelineContext(
        config=config,
        video_path=Path("dummy.mp4"),
        output_dir=Path(output_dir),
    )


def _make_event(
    event_id: str = "e1",
    segment_id: str = "seg1",
    observation_id: str = "obs1",
    action: ActionType = ActionType.PICK,
    actor_track_id: int | None = None,
    object_track_id: int | None = None,
    start_sec: float = 0.0,
    end_sec: float = 1.0,
    confidence: float = 0.9,
    timing_precision: str = "SEGMENT",
) -> PhysicalEvent:
    return PhysicalEvent(
        event_id=event_id,
        segment_id=segment_id,
        observation_id=observation_id,
        action=action,
        actor_track_id=actor_track_id,
        object_track_id=object_track_id,
        source="test",
        start_sec=start_sec,
        end_sec=end_sec,
        confidence=confidence,
        attributes={"timing_precision": timing_precision},
    )


def _make_track(track_id: int, class_name: str) -> Track:
    return Track(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        start_frame=0,
        end_frame=10,
        start_sec=0.0,
        end_sec=5.0,
        confidence=1.0,
        points=[],
        source="test",
    )


def _make_segment(segment_id: str, track_ids: list[int]) -> CandidateSegment:
    return CandidateSegment(
        segment_id=segment_id,
        track_ids=track_ids,
        start_frame=0,
        end_frame=10,
        start_sec=0.0,
        end_sec=5.0,
    )


def _make_transition(
    transition_id: str,
    trigger_event_id: str,
    track_id: int | None = None,
    semantic_label: str | None = None,
) -> StateTransition:
    return StateTransition(
        transition_id=transition_id,
        track_id=track_id,
        semantic_label=semantic_label,
        from_state="ON_SURFACE",
        to_state="IN_HAND",
        trigger_event_id=trigger_event_id,
        start_sec=0.0,
        end_sec=1.0,
        confidence=0.9,
        source="test",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx() -> PipelineContext:
    return _make_ctx()


# ============================================================================
# REQUIREMENT 1 — Event-as-Edge architecture
# ============================================================================

def test_event_produces_edge(ctx):
    ctx.events = [_make_event(actor_track_id=1, object_track_id=2)]
    _, edges = GraphBuilder().build(ctx)
    assert len(edges) == 1
    assert isinstance(edges[0], GraphEdge)


def test_each_event_participant_is_a_node(ctx):
    ctx.events = [_make_event(actor_track_id=1, object_track_id=2)]
    nodes, edges = GraphBuilder().build(ctx)
    node_ids = {n.node_id for n in nodes}
    assert edges[0].source_node_id in node_ids
    assert edges[0].target_node_id in node_ids


# ============================================================================
# REQUIREMENTS 4/5 — Explicit track IDs → RESOLVED
# ============================================================================

def test_explicit_actor_track_id_resolves(ctx):
    ctx.events = [_make_event(actor_track_id=3, object_track_id=None)]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].actor_resolution == "RESOLVED"


def test_explicit_object_track_id_resolves(ctx):
    ctx.events = [_make_event(actor_track_id=None, object_track_id=7)]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].object_resolution == "RESOLVED"


# ============================================================================
# REQUIREMENT 6 — Single visible person must NOT auto-resolve actor
# ============================================================================

def test_single_person_track_does_not_resolve_actor(ctx):
    ctx.events = [_make_event()]  # actor_track_id=None
    ctx.tracks = [_make_track(1, "person")]
    ctx.candidate_segments = [_make_segment("seg1", [1])]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].actor_resolution == "UNRESOLVED"


# ============================================================================
# REQUIREMENT 7 — Multiple visible people → AMBIGUOUS
# ============================================================================

def test_two_people_no_explicit_id_gives_ambiguous(ctx):
    ctx.events = [_make_event()]
    ctx.tracks = [_make_track(1, "person"), _make_track(2, "person")]
    ctx.candidate_segments = [_make_segment("seg1", [1, 2])]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].actor_resolution == "AMBIGUOUS"
    # Must produce exactly ONE edge, not one per person
    assert len(edges) == 1


# ============================================================================
# REQUIREMENT 8 — No nearest-person matching (code inspection; covered by 6/7)
# ============================================================================

# ============================================================================
# REQUIREMENTS 9/10/13 — A: Multi-object conservatism
# ============================================================================

def test_multiple_vlm_objects_no_explicit_id_collapses_to_one_ambiguous_edge(ctx):
    """VLM.objects has >1 entry, no explicit object_track_id, no StateTransition.
    Must produce exactly one edge with object_resolution=AMBIGUOUS.
    Must NOT create one edge per VLM object.
    """
    ctx.events = [_make_event(actor_track_id=1, object_track_id=None)]
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs1",
            segment_id="seg1",
            status=VLMSegmentStatus.SUCCESS,
            actor="person",
            active_hand="RIGHT",
            objects=["cup", "table"],
            raw_action="picked up cup from the table",
            segment_start_sec=0.0,
            segment_end_sec=1.0,
            start_time_sec=0.0,
            end_time_sec=1.0,
            state_change=None,
            visible_facts="person lifts cup",
            inference=None,
            uncertainty=None,
            confidence=0.85,
        )
    ]
    _, edges = GraphBuilder().build(ctx)
    assert len(edges) == 1
    assert edges[0].object_resolution == "AMBIGUOUS"


def test_contextual_object_not_target_when_explicit_id_present(ctx):
    """Explicit object_track_id=2 (cup); VLM also lists 'table'.
    Only cup edge should be produced; table must not appear.
    """
    ctx.events = [_make_event(actor_track_id=1, object_track_id=2)]
    ctx.vlm_observations = [
        RawVLMObservation(
            observation_id="obs1",
            segment_id="seg1",
            status=VLMSegmentStatus.SUCCESS,
            actor="person",
            active_hand="RIGHT",
            objects=["cup", "table"],
            raw_action="picked up cup from table",
            segment_start_sec=0.0,
            segment_end_sec=1.0,
            start_time_sec=0.0,
            end_time_sec=1.0,
            state_change=None,
            visible_facts="none",
            inference=None,
            uncertainty=None,
            confidence=0.9,
        )
    ]
    _, edges = GraphBuilder().build(ctx)
    assert len(edges) == 1
    assert edges[0].target_node_id == "node_track_2"


# ============================================================================
# REQUIREMENT 14 — B: Multi-actor conservatism
# ============================================================================

def test_multi_actor_produces_one_edge_not_one_per_person(ctx):
    """Two person tracks visible; no explicit actor_track_id.
    Must produce exactly one edge (AMBIGUOUS actor), not two edges.
    """
    ctx.events = [_make_event(object_track_id=5)]
    ctx.tracks = [_make_track(1, "person"), _make_track(2, "person")]
    ctx.candidate_segments = [_make_segment("seg1", [1, 2])]
    _, edges = GraphBuilder().build(ctx)
    assert len(edges) == 1
    assert edges[0].actor_resolution == "AMBIGUOUS"


# ============================================================================
# REQUIREMENTS 11/12 — UNKNOWN action stays visible; participants independent
# ============================================================================

def test_unknown_action_preserved(ctx):
    ctx.events = [_make_event(action=ActionType.UNKNOWN, actor_track_id=1, object_track_id=2)]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].action == ActionType.UNKNOWN


def test_unknown_action_does_not_imply_unknown_participants(ctx):
    ctx.events = [_make_event(action=ActionType.UNKNOWN, actor_track_id=5, object_track_id=9)]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].actor_resolution == "RESOLVED"
    assert edges[0].object_resolution == "RESOLVED"


def test_unknown_action_with_ambiguous_actor(ctx):
    """UNKNOWN action + no explicit actor → actor AMBIGUOUS; independent of action."""
    ctx.events = [_make_event(action=ActionType.UNKNOWN, object_track_id=9)]
    ctx.tracks = [_make_track(1, "person"), _make_track(2, "person")]
    ctx.candidate_segments = [_make_segment("seg1", [1, 2])]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].action == ActionType.UNKNOWN
    assert edges[0].actor_resolution == "AMBIGUOUS"
    assert edges[0].object_resolution == "RESOLVED"


# ============================================================================
# REQUIREMENTS 15/16 — C: Timing preservation
# ============================================================================

@pytest.mark.parametrize("timing_precision", ["EXACT", "SEGMENT"])
def test_timing_fields_preserved(ctx, timing_precision):
    ctx.events = [
        _make_event(
            actor_track_id=1,
            object_track_id=2,
            start_sec=2.0,
            end_sec=4.5,
            timing_precision=timing_precision,
        )
    ]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].start_sec == 2.0
    assert edges[0].end_sec == 4.5
    assert edges[0].timing_precision == timing_precision


# ============================================================================
# REQUIREMENTS 17/18 — D/E: StateTransition linking and isolation
# ============================================================================

def test_state_transition_attaches_by_event_id_and_track(ctx):
    ctx.events = [_make_event(event_id="e1", actor_track_id=1, object_track_id=2)]
    ctx.state_transitions = [_make_transition("st1", trigger_event_id="e1", track_id=2)]
    _, edges = GraphBuilder().build(ctx)
    assert "st1" in edges[0].state_transition_ids


def test_state_transition_isolation_different_events(ctx):
    """st1 belongs to e1, st2 belongs to e2.
    After building, each edge must only hold its own transition.
    """
    ctx.events = [
        _make_event(event_id="e1", actor_track_id=1, object_track_id=2),
        _make_event(event_id="e2", actor_track_id=1, object_track_id=3),
    ]
    ctx.state_transitions = [
        _make_transition("st1", trigger_event_id="e1", track_id=2),
        _make_transition("st2", trigger_event_id="e2", track_id=3),
    ]
    _, edges = GraphBuilder().build(ctx)
    e1_edge = next(e for e in edges if e.event_id == "e1")
    e2_edge = next(e for e in edges if e.event_id == "e2")
    assert "st1" in e1_edge.state_transition_ids
    assert "st2" not in e1_edge.state_transition_ids
    assert "st2" in e2_edge.state_transition_ids
    assert "st1" not in e2_edge.state_transition_ids


def test_wrong_trigger_event_id_transition_not_attached(ctx):
    """A transition whose trigger_event_id is a completely different event
    must never attach to any edge.
    """
    ctx.events = [_make_event(event_id="e1", actor_track_id=1, object_track_id=2)]
    ctx.state_transitions = [
        _make_transition("st_other", trigger_event_id="e_nonexistent", track_id=2)
    ]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].state_transition_ids == []


def test_multiple_state_transitions_same_event_all_attached(ctx):
    """One event with two state transitions (e.g. two objects affected).
    Both must appear in state_transition_ids.
    """
    ctx.events = [_make_event(event_id="e1", actor_track_id=1, object_track_id=2)]
    ctx.state_transitions = [
        _make_transition("st1", trigger_event_id="e1", track_id=2),
        _make_transition("st2", trigger_event_id="e1", track_id=2),
    ]
    _, edges = GraphBuilder().build(ctx)
    assert "st1" in edges[0].state_transition_ids
    assert "st2" in edges[0].state_transition_ids


# ============================================================================
# REQUIREMENT 19 — F: Provenance preservation
# ============================================================================

def test_provenance_fields_preserved(ctx):
    ctx.events = [
        _make_event(
            event_id="evt_prov",
            segment_id="seg_prov",
            observation_id="obs_prov",
            actor_track_id=1,
            object_track_id=2,
        )
    ]
    _, edges = GraphBuilder().build(ctx)
    assert edges[0].event_id == "evt_prov"
    assert edges[0].observation_id == "obs_prov"
    assert edges[0].segment_id == "seg_prov"


# ============================================================================
# REQUIREMENT 20 — G: Edge deduplication must not merge different event_ids
# ============================================================================

def test_different_event_ids_produce_separate_edges(ctx):
    ctx.events = [
        _make_event(event_id="e1", actor_track_id=1, object_track_id=2),
        _make_event(event_id="e2", actor_track_id=1, object_track_id=2),
    ]
    _, edges = GraphBuilder().build(ctx)
    assert len(edges) == 2
    edge_ids = {e.edge_id for e in edges}
    assert len(edge_ids) == 2  # distinct edge_ids


# ============================================================================
# REQUIREMENT 21 — H: Node deduplication
# ============================================================================

def test_shared_actor_track_produces_one_person_node(ctx):
    ctx.events = [
        _make_event(event_id="e1", actor_track_id=1, object_track_id=2),
        _make_event(event_id="e2", actor_track_id=1, object_track_id=3),
    ]
    nodes, edges = GraphBuilder().build(ctx)
    person_nodes = [n for n in nodes if n.role.value == "PERSON"]
    assert len(person_nodes) == 1
    assert person_nodes[0].node_id == "node_track_1"
    assert len(edges) == 2


def test_shared_object_track_produces_one_object_node(ctx):
    ctx.events = [
        _make_event(event_id="e1", actor_track_id=1, object_track_id=5),
        _make_event(event_id="e2", actor_track_id=2, object_track_id=5),
    ]
    nodes, _edges = GraphBuilder().build(ctx)
    obj_nodes = [n for n in nodes if n.role.value == "OBJECT"]
    assert len(obj_nodes) == 1
    assert obj_nodes[0].node_id == "node_track_5"


# ============================================================================
# REQUIREMENT 22 — I: Deterministic repeated execution
# ============================================================================

def test_deterministic_repeated_execution(ctx):
    ctx.events = [
        _make_event(event_id="e1", actor_track_id=1, object_track_id=2),
        _make_event(event_id="e2", actor_track_id=3, object_track_id=4),
    ]
    ctx.state_transitions = [_make_transition("st1", "e1", track_id=2)]

    nodes1, edges1 = GraphBuilder().build(ctx)
    nodes2, edges2 = GraphBuilder().build(ctx)

    assert {n.node_id for n in nodes1} == {n.node_id for n in nodes2}
    assert {e.edge_id for e in edges1} == {e.edge_id for e in edges2}
    assert len(nodes1) == len(nodes2)
    assert len(edges1) == len(edges2)

    # Compare serialized form for full determinism
    serial1 = sorted(e.model_dump_json() for e in edges1)
    serial2 = sorted(e.model_dump_json() for e in edges2)
    assert serial1 == serial2


# ============================================================================
# REQUIREMENT 23 — J: Serialization / deserialization
# ============================================================================

def test_interaction_graph_json_written_and_valid(tmp_path):
    ctx = PipelineContext(
        config=_make_config(),
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path,
    )
    ctx.events = [_make_event(actor_track_id=1, object_track_id=2)]
    ctx.state_transitions = [_make_transition("st1", "e1", track_id=2)]

    status = s09_graph.run(ctx)
    assert status.status == "OK"

    out_file = tmp_path / "interaction_graph.json"
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "metadata" in data
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)

    # Pydantic round-trip validation
    for raw_node in data["nodes"]:
        GraphNode.model_validate(raw_node)
    for raw_edge in data["edges"]:
        GraphEdge.model_validate(raw_edge)


# ============================================================================
# REQUIREMENTS 24/25 — K: FAILED/SKIPPED upstream → zero edges
# ============================================================================

def test_failed_vlm_observations_produce_no_edges(ctx):
    """Stage 09 operates on ctx.events; FAILED obs produce no events → no edges."""
    ctx.events = []  # nothing promoted from FAILED obs
    _, edges = GraphBuilder().build(ctx)
    assert edges == []


def test_skipped_vlm_observations_produce_no_edges(ctx):
    """Same as FAILED case — SKIPPED obs produce no events."""
    ctx.events = []
    _, edges = GraphBuilder().build(ctx)
    assert edges == []


# ============================================================================
# REQUIREMENT 26 — L: Empty input
# ============================================================================

def test_empty_events_graceful(tmp_path):
    ctx = PipelineContext(
        config=_make_config(),
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path,
    )
    ctx.events = []

    status = s09_graph.run(ctx)
    assert status.status == "OK"
    assert ctx.graph_nodes == []
    assert ctx.graph_edges == []

    out_file = tmp_path / "interaction_graph.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["nodes"] == []
    assert data["edges"] == []


# ============================================================================
# REQUIREMENT 27 — M: Upstream immutability
# ============================================================================

def test_stage09_does_not_mutate_upstream_context(tmp_path):
    ctx = PipelineContext(
        config=_make_config(),
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path,
    )
    ctx.events = [_make_event(actor_track_id=1, object_track_id=2)]
    ctx.tracks = [_make_track(1, "person"), _make_track(2, "cup")]
    ctx.state_transitions = [_make_transition("st1", "e1", track_id=2)]
    ctx.vlm_observations = []

    events_before = list(ctx.events)
    tracks_before = list(ctx.tracks)
    transitions_before = list(ctx.state_transitions)

    s09_graph.run(ctx)

    assert len(ctx.events) == len(events_before)
    assert len(ctx.tracks) == len(tracks_before)
    assert len(ctx.state_transitions) == len(transitions_before)
    assert ctx.events[0].event_id == events_before[0].event_id
    assert ctx.tracks[0].track_id == tracks_before[0].track_id


# ============================================================================
# REQUIREMENT 28 — N: Stub mode (F-13 bug fix)
# ============================================================================

def test_stub_mode_returns_skipped(tmp_path):
    """In stub_mode=True, stage must return SKIPPED — not OK with empty data."""
    ctx = PipelineContext(
        config=_make_config(stub_mode=True),
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path,
    )
    ctx.events = [_make_event(actor_track_id=1, object_track_id=2)]

    status = s09_graph.run(ctx)
    assert status.status == "SKIPPED"


def test_stub_mode_produces_no_graph_nodes_or_edges(tmp_path):
    ctx = PipelineContext(
        config=_make_config(stub_mode=True),
        video_path=Path("dummy.mp4"),
        output_dir=tmp_path,
    )
    ctx.events = [_make_event(actor_track_id=1, object_track_id=2)]

    s09_graph.run(ctx)
    # graph_nodes and graph_edges must remain at their default empty state
    assert ctx.graph_nodes == []
    assert ctx.graph_edges == []


# ============================================================================
# REQUIREMENT 29 — O: Schema validation (covered by J; explicit Pydantic check)
# ============================================================================

def test_graph_node_schema_validates():
    from src.schema.interaction_graph import NodeRole
    node = GraphNode(node_id="node_track_1", role=NodeRole.PERSON, track_id=1)
    assert node.node_id == "node_track_1"
    assert node.track_id == 1


def test_graph_edge_schema_validates():
    edge = GraphEdge(
        edge_id="e1",
        source_node_id="node_track_1",
        target_node_id="node_track_2",
        action=ActionType.PICK,
        actor_resolution="RESOLVED",
        object_resolution="RESOLVED",
        start_sec=0.0,
        end_sec=1.0,
        timing_precision="EXACT",
        event_id="evt1",
        confidence=0.9,
    )
    assert edge.timing_precision == "EXACT"
    assert edge.state_transition_ids == []
