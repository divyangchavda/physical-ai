"""Schema package — Pydantic models for all inter-stage data structures.

Each module owns one logical data concept:
    detection   → BoundingBox, Detection, DetectionFrame
    track       → TrackPoint, Track
    segment     → CandidateSegment
    event       → ActionType (enum), PhysicalEvent
    state       → ObjectState, StateTransition
    trajectory  → TrajectoryPoint2D, Trajectory2D  (2-D only, never 3-D)
    episode     → VideoMetadata, PipelineStageStatus, PhysicalEpisode
    evaluation  → StageEvaluation, EvaluationReport
"""
