"""Stage 09: Interaction Graph construction."""
import json
import logging
import time

from src.context import PipelineContext, PipelineStageStatus
from src.models.graph_builder import GraphBuilder

STAGE = "s09_graph"
logger = logging.getLogger(__name__)


def _write_output(ctx: PipelineContext, status: str) -> None:
    """Serialize the extracted interaction graph to interaction_graph.json."""
    out_file = ctx.output_dir / "interaction_graph.json"
    
    output_data = {
        "metadata": {
            "video_path": str(ctx.video_path),
            "stage": STAGE,
            "status": status,
        },
        "nodes": [n.model_dump() for n in ctx.graph_nodes],
        "edges": [e.model_dump() for e in ctx.graph_edges]
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


def run(ctx: PipelineContext) -> PipelineStageStatus:
    """Construct the deterministic interaction graph from events and tracks."""
    t0 = time.time()

    if ctx.config.stub_mode:
        logger.info("[%s] stub_mode=True — SKIPPED (no graph fabricated)", STAGE)
        return PipelineStageStatus(
            stage=STAGE,
            status="SKIPPED",
            message="stub_mode: interaction graph construction skipped",
            execution_time_sec=time.time() - t0,
        )

    if not ctx.config.graph_extraction.enabled:
        logger.info("[%s] Graph extraction disabled", STAGE)
        return PipelineStageStatus(stage=STAGE, status="SKIPPED", message="Disabled in config")
        
    try:
        builder = GraphBuilder()
        nodes, edges = builder.build(ctx)
        
        ctx.graph_nodes = nodes
        ctx.graph_edges = edges
        
        _write_output(ctx, "SUCCESS")
        
        logger.info("[%s] Constructed %d nodes and %d edges", STAGE, len(nodes), len(edges))
        
        return PipelineStageStatus(
            stage=STAGE,
            status="OK",
            message=f"Graph constructed ({len(nodes)} nodes, {len(edges)} edges)",
            execution_time_sec=time.time() - t0,
        )
        
    except Exception as e:
        logger.exception("[%s] Graph extraction failed", STAGE)
        return PipelineStageStatus(
            stage=STAGE,
            status="ERROR",
            message=str(e),
            execution_time_sec=time.time() - t0,
        )
