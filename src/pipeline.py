"""Physical Data Compiler — main pipeline entry point.

Usage:
    python -m src.pipeline <video_path> [options]

Run with --stub to use zero-dependency stubs for all heavy AI stages.
This is the correct mode for foundation testing before real models are wired in.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src.config import load_config
from src.context import PipelineContext
from src.logging_utils import configure_root_logger, get_logger
from src.stages import (
    s01_ingest,
    s02_sample,
    s03_detect,
    s04_track,
    s05_segment,
    s06_vlm,
    s07_events,
    s08_states,
    s09_graph,
    s10_trajectories,
    s11_score,
    s12_episode,
    s13_evaluate,
    s14_preview,
)

logger = get_logger(__name__)


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env file at the repo root into os.environ.

    The VLM reads GEMINI_API_KEY from the environment only. Without this, a
    .env file sitting in the repo is silently ignored. Existing environment
    variables always win, so Kaggle Secrets (exported by the notebook) are
    never overwritten by a stale local .env.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logger.warning("Could not read %s: %s", env_path, exc)


# Ordered list of pipeline stages
_STAGE_MODULES = [
    s01_ingest,
    s02_sample,
    s03_detect,
    s04_track,
    s05_segment,
    s06_vlm,
    s07_events,
    s08_states,
    s09_graph,
    s10_trajectories,
    s11_score,
    s12_episode,
    s13_evaluate,
    s14_preview,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description=(
            "Physical Data Compiler — converts a physical-world video into "
            "structured physical data (local-first, sequential pipeline)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Foundation smoke test (all heavy stages use stubs)
  python -m src.pipeline input/video.mp4 --stub

  # Real run (requires YOLO and ByteTrack to be implemented)
  python -m src.pipeline input/video.mp4

  # Custom config + override
  python -m src.pipeline input/video.mp4 --config my_config.yaml \\
      --set frame_sampling.fps=2.0

Output files:
  output/detections.json
  output/tracks.json
  output/candidate_segments.json
  output/vlm_observations.json
  output/events.json
  output/states.json
  output/interaction_graph.json
  output/trajectories.json
  output/quality_scores.json
  output/episodes.json
  output/episode.json
  output/evaluation.json
  output/preview.json
""",
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Input video file (MP4, AVI, MOV, etc.)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a YAML config file (overrides default.yaml)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Run all heavy AI stages as zero-dependency stubs. "
            "Use this for foundation testing before real models are implemented."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        metavar="KEY=VALUE",
        help=(
            "Override a config value (repeatable). "
            "Example: --set frame_sampling.fps=2.0 --set vlm.enabled=false"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (overrides config output_dir)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and return an exit code (0=success, 1=error)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    _load_dotenv()

    # ── Config ───────────────────────────────────────────────────────────────
    overrides: list[str] = list(args.overrides or [])
    if args.stub:
        overrides.append("stub_mode=true")
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")

    try:
        config = load_config(yaml_path=args.config, set_overrides=overrides)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    # Auto-generate timestamped output directory if user didn't explicitly override it
    if not args.output_dir:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = config.output_dir / f"run_{timestamp}"

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = config.output_dir / "pipeline.log"
    configure_root_logger(level=log_level, log_file=log_file)

    # ── Banner ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Physical Data Compiler v%s", _pipeline_version())
    logger.info("Input : %s", args.video)
    logger.info("Output: %s", config.output_dir.resolve())
    logger.info("Log   : %s", log_file.resolve())
    logger.info("Stub  : %s", config.stub_mode)
    logger.info("=" * 60)

    # ── Context ──────────────────────────────────────────────────────────────
    ctx = PipelineContext(
        config=config,
        video_path=args.video.resolve(),
        output_dir=config.output_dir.resolve(),
    )

    # ── Run stages ───────────────────────────────────────────────────────────
    for stage_mod in _STAGE_MODULES:
        stage_name = getattr(stage_mod, "STAGE", stage_mod.__name__)
        try:
            status = stage_mod.run(ctx)
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user (KeyboardInterrupt).")
            raise
        except Exception as exc:
            logger.exception("Unhandled exception in stage %s", stage_name)
            from src.schema.episode import PipelineStageStatus
            status = PipelineStageStatus(
                stage=stage_name, status="ERROR", message=str(exc)
            )
        ctx.record_stage(status)
        _log_stage_result(stage_name, status.status)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Pipeline complete. Outputs in: %s", config.output_dir.resolve())
    if ctx.evaluation:
        logger.info("Overall status: %s", ctx.evaluation.overall_status)
    logger.info("=" * 60)

    errors = [s for s in ctx.stage_statuses if s.status == "ERROR"]
    return 1 if errors else 0


def _log_stage_result(stage_name: str, status: str) -> None:
    symbol = {"OK": "[OK]", "SKIPPED": "[SKIP]", "ERROR": "[ERR]"}.get(status, "[?]")
    logger.info("  [%s] %s %s", status.ljust(7), symbol, stage_name)


def _pipeline_version() -> str:
    try:
        from src import __version__
        return __version__
    except ImportError:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
