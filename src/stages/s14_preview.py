"""Stage 14 — Preview JSON report rendering."""
from __future__ import annotations

import time

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.schema.episode import PipelineStageStatus
from src.schema.event import ActionType
from src.schema.preview import (
    EpisodeSummary,
    EvaluationSummary,
    PreviewCounts,
    PreviewReport,
    QualityDistribution,
)

logger = get_logger(__name__)
STAGE = "s14_preview"


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if ctx.config.stub_mode:
        logger.info("[%s] stub_mode=True — SKIPPED (no preview rendered)", STAGE)
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: preview rendering skipped",
            duration_sec=time.monotonic() - t0,
        )

    # PreviewCounts
    counts = PreviewCounts(
        total_events=len(ctx.events),
        total_episodes=len(ctx.episodes),
        total_state_transitions=len(ctx.state_transitions),
        total_graph_nodes=len(ctx.graph_nodes),
        total_graph_edges=len(ctx.graph_edges),
        total_trajectories=len(ctx.trajectories)
    )

    # QualityDistribution
    q_dist = QualityDistribution()
    for q in ctx.quality_scores:
        t = q.quality_tier
        if t == "HIGH" or t == "AUTO_ACCEPT":
            q_dist.high += 1
        elif t == "MEDIUM" or t == "HUMAN_REVIEW":
            q_dist.medium += 1
        elif t == "LOW":
            q_dist.low += 1
        elif t == "REJECT" or t == "REJECTED":
            q_dist.rejected += 1

    # EvaluationSummary
    if ctx.evaluation is None:
        eval_sum = EvaluationSummary(
            dataset_health="NOT_AVAILABLE",
            error_count=0,
            warning_count=0,
            info_count=0
        )
    else:
        errors = sum(1 for i in ctx.evaluation.integrity_issues if i.severity == "ERROR")
        warnings = sum(1 for i in ctx.evaluation.integrity_issues if i.severity == "WARNING")
        infos = sum(1 for i in ctx.evaluation.integrity_issues if i.severity == "INFO")
        
        eval_sum = EvaluationSummary(
            dataset_health=ctx.evaluation.dataset_health,
            error_count=errors,
            warning_count=warnings,
            info_count=infos
        )

    # Timeline (Episodes)
    timeline = []
    
    event_dict = {e.event_id: e for e in ctx.events}
    node_dict = {n.node_id: n for n in ctx.graph_nodes}
    edge_dict = {e.event_id: e for e in ctx.graph_edges}

    for ep in ctx.episodes:
        hr_events = []
        for eid in ep.event_ids:
            if eid not in event_dict:
                continue
            ev = event_dict[eid]
            if ev.action == ActionType.UNKNOWN:
                hr_events.append("UNKNOWN")
                continue
                
            action_str = ev.action.value
            obj_label = None
            if eid in edge_dict:
                edge = edge_dict[eid]
                tgt_id = edge.target_node_id
                if tgt_id in node_dict:
                    obj_label = node_dict[tgt_id].semantic_label
            
            if obj_label:
                hr_events.append(f"{action_str} {obj_label}")
            else:
                hr_events.append(action_str)
                
        ep_sum = EpisodeSummary(
            episode_id=ep.episode_id,
            start_sec=ep.start_sec,
            end_sec=ep.end_sec,
            timing_precision=ep.timing_precision,
            quality_tier=ep.episode_quality_tier,
            event_ids=ep.event_ids,
            human_readable_events=hr_events
        )
        timeline.append(ep_sum)

    report = PreviewReport(
        dataset_health=eval_sum.dataset_health,
        counts=counts,
        quality_distribution=q_dist,
        evaluation_summary=eval_sum,
        timeline=timeline
    )

    out_path = ctx.output_dir / "preview.json"
    with open(out_path, "w") as f:
        f.write(report.model_dump_json(indent=2))

    return PipelineStageStatus(
        stage=STAGE, status="OK",
        message="Generated preview.json",
        duration_sec=time.monotonic() - t0,
    )
