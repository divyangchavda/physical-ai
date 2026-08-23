"""Stage 13 — Pipeline evaluation.

Examines stage statuses and data quality to produce an EvaluationReport.
Always runs. Records issues and warnings without modifying any data.

Output file: output/evaluation.json
Output context: ctx.evaluation (EvaluationReport)
"""
from __future__ import annotations

import json
import time

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.schema.episode import PipelineStageStatus
from src.schema.evaluation import EvaluationReport, IntegrityIssue, StageEvaluation

logger = get_logger(__name__)
STAGE = "s13_evaluate"


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    episode_id = ctx.episode.episode_id if ctx.episode else "unknown"

    stage_evals: list[StageEvaluation] = []
    issues: list[str] = []
    warnings: list[str] = []

    # Evaluate each completed stage
    for ss in ctx.stage_statuses:
        if ss.status == "OK":
            stage_evals.append(StageEvaluation(stage=ss.stage, status="OK"))
        elif ss.status == "WARNING":
            stage_evals.append(StageEvaluation(
                stage=ss.stage, status="WARNING", message=ss.message
            ))
            warnings.append(f"{ss.stage}: {ss.message}")
        elif ss.status == "SKIPPED":
            stage_evals.append(StageEvaluation(
                stage=ss.stage, status="SKIPPED", message=ss.message
            ))
        else:
            stage_evals.append(StageEvaluation(
                stage=ss.stage, status="ERROR", message=ss.message
            ))
            issues.append(f"{ss.stage}: {ss.message}")

    # Data quality checks
    n_errors = sum(1 for ss in ctx.stage_statuses if ss.status == "ERROR")
    n_warnings = sum(1 for ss in ctx.stage_statuses if ss.status == "WARNING")
    n_skipped = sum(1 for ss in ctx.stage_statuses if ss.status == "SKIPPED")
    n_ok = sum(1 for ss in ctx.stage_statuses if ss.status == "OK")

    if n_errors > 0:
        overall = "FAIL"
    elif n_skipped > 0 and (n_ok + n_warnings) > 0:
        overall = "PARTIAL"
    elif n_warnings > 0 and n_skipped == 0 and n_errors == 0:
        overall = "PARTIAL"  # all ran but some produced warnings
    elif n_skipped == len(ctx.stage_statuses):
        overall = "SKIPPED"
    else:
        overall = "PASS"

    if ctx.video_metadata is None:
        warnings.append("Video metadata is missing — s01_ingest may have failed")

    if not ctx.sampled_frame_infos:
        warnings.append("No frames were sampled — pipeline produced no data")

    if ctx.video_metadata and ctx.video_metadata.metadata_warnings:
        for mw in ctx.video_metadata.metadata_warnings:
            warnings.append(f"video metadata: {mw}")

    from src.models.dataset_evaluator import DatasetEvaluator
    
    if ctx.config.stub_mode:
        warnings.append(
            "stub_mode=True: all heavy AI stages were skipped. "
            "No real detections, tracks, or events were produced."
        )
        
    # Run evaluator
    integrity_issues = DatasetEvaluator.evaluate(ctx)
    if ctx.config.stub_mode:
        integrity_issues.append(IntegrityIssue(
            severity="INFO", dimension="QUALITY_CONSISTENCY",
            message="stub_mode=True, skipped deep evaluation", reference_id="system"
        ))
        
    dataset_health = "HEALTHY"
    if any(i.severity == "ERROR" for i in integrity_issues):
        dataset_health = "CRITICAL"
    elif any(i.severity == "WARNING" for i in integrity_issues):
        dataset_health = "WARNING"

    report = EvaluationReport(
        episode_id=episode_id,
        overall_status=overall,
        stage_evaluations=stage_evals,
        dataset_health=dataset_health,
        integrity_issues=integrity_issues,
        issues=issues,
        warnings=warnings,
    )

    ctx.evaluation = report

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "evaluation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2)

    logger.info(
        "[%s] overall=%s health=%s | OK=%d WARNING=%d SKIPPED=%d ERROR=%d | issues=%d warnings=%d",
        STAGE, overall, dataset_health, n_ok, n_warnings, n_skipped, n_errors, len(issues), len(warnings),
    )
    return PipelineStageStatus(
        stage=STAGE, status="OK", duration_sec=time.monotonic() - t0
    )
