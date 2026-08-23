"""Stage 12 — Episode assembly.

Aggregates all stage outputs into a single PhysicalEpisode root object
and writes episode.json. Always runs regardless of other stage statuses.

Summary counts reflect actual data — never fabricated:
  - 0 for a SKIPPED stage.
  - Accurate count for an OK stage.

Output file: output/episode.json
Output context: ctx.episode (PhysicalEpisode)
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

from src.context import PipelineContext
from src.logging_utils import get_logger
from src.schema.episode import PhysicalEpisode, PipelineStageStatus

logger = get_logger(__name__)
STAGE = "s12_episode"


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    episode_id = f"ep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    n_frames_with_det = sum(
        1 for df in ctx.detection_frames if df.detections
    )

    episode = PhysicalEpisode(
        episode_id=episode_id,
        created_at=datetime.now(timezone.utc),
        video_metadata=ctx.video_metadata,
        stage_statuses=list(ctx.stage_statuses),  # snapshot before this stage appends
        n_frames_sampled=len(ctx.sampled_frame_infos),
        n_frames_with_detections=n_frames_with_det,
        n_detections=sum(len(df.detections) for df in ctx.detection_frames),
        n_tracks=len(ctx.tracks),
        n_candidate_segments=len(ctx.candidate_segments),
        n_events=len(ctx.events),
        n_state_transitions=len(ctx.state_transitions),
        n_trajectories=len(ctx.trajectories),
    )

    ctx.episode = episode

    if ctx.config.episode.enabled:
        from src.models.episode_assembler import EpisodeAssembler
        assembler = EpisodeAssembler(config=ctx.config.episode)
        ctx.episodes = assembler.assemble(ctx.events)
        out_path_episodes = ctx.output_dir / "episodes.json"
        with open(out_path_episodes, "w", encoding="utf-8") as f:
            json.dump([e.model_dump(mode="json") for e in ctx.episodes], f, indent=2, default=str)

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "episode.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(episode.model_dump(mode="json"), f, indent=2, default=str)

    logger.info(
        "[%s] episode_id=%s | frames=%d detections=%d tracks=%d events=%d",
        STAGE,
        episode_id,
        episode.n_frames_sampled,
        episode.n_detections,
        episode.n_tracks,
        episode.n_events,
    )
    return PipelineStageStatus(
        stage=STAGE, status="OK", duration_sec=time.monotonic() - t0
    )
