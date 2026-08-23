"""Structured logging utilities for the Physical Data Compiler pipeline."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with consistent formatting.

    Call as ``logger = get_logger(__name__)`` at module level.

    Handlers live only on the root logger. Attaching one here as well would
    emit every record twice — once from this handler, once after propagating
    to root — and would also bypass the run's log file.
    """
    if not logging.getLogger().handlers:
        # Nothing has configured logging yet (e.g. a standalone tools/ script,
        # or module import before the pipeline sets up its run log). Give root
        # a stdout handler so records are not swallowed. A later
        # configure_root_logger(force=True) replaces it cleanly.
        configure_root_logger(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def configure_root_logger(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure the root logger for a pipeline run.

    Args:
        level: logging level (e.g. logging.DEBUG, logging.INFO).
        log_file: optional file path to write logs to (in addition to stdout).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
