"""Pipeline stages package.

Each stage module exposes a single ``run(ctx: PipelineContext) -> PipelineStageStatus``
function. Stages mutate the context and return their execution status.

Stage contract:
  - Always returns a PipelineStageStatus (never raises).
  - Sets status="SKIPPED" when stub_mode=True or when the stage is disabled.
  - SKIPPED stages write an empty array [] to their output file.
  - Never fabricate data in any status.
  - Errors are caught by the orchestrator in pipeline.py.
"""
