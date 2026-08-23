# Physical Data Compiler

A **local-first Python CLI** that converts physical-world videos into structured physical data.

> **Foundation phase** — architecture and interfaces only.
> Real YOLO inference, ByteTrack, and VLM are not yet wired in.
> Run with `--stub` for end-to-end testing.

---

## Quick Start

```bash
# 1. Install core dependencies
pip install -r requirements.txt

# Install PyTorch separately (match your CUDA version):
# https://pytorch.org/get-started/locally/

# 2. Place your video in input/
mkdir input
cp your_video.mp4 input/video.mp4

# 3. Run the pipeline (stub mode — foundation testing)
python -m src.pipeline input/video.mp4 --stub

# 4. Check outputs
ls output/
```

---

## Hardware Requirements

| Resource | Minimum |
|---|---|
| RAM | 8 GB |
| SSD | ~520 GB |
| GPU (optional) | NVIDIA 4 GB VRAM |
| Python | 3.10+ |

The pipeline runs on CPU if no GPU is available. Heavy AI stages are optional.

---

## CLI Reference

```
python -m src.pipeline <video> [options]

Arguments:
  video                 Input video file (MP4, AVI, MOV, etc.)

Options:
  --config PATH         YAML config file (overrides default.yaml)
  --stub                Run all heavy AI stages as stubs (no inference)
  --set KEY=VALUE       Override a config value (repeatable)
  --output-dir DIR      Output directory
  --verbose             Enable DEBUG logging
  -h, --help            Show this help message
```

### Examples

```bash
# Stub mode (foundation smoke test)
python -m src.pipeline input/video.mp4 --stub

# Custom config + override
python -m src.pipeline input/video.mp4 \
    --config my_config.yaml \
    --set frame_sampling.fps=2.0

# Output to a specific directory
python -m src.pipeline input/video.mp4 --stub --output-dir /tmp/run1
```

---

## Pipeline Stages

| Stage | File | Status | Description |
|---|---|---|---|
| s01 | `s01_ingest.py` | ✅ Implemented | Video metadata (OpenCV) |
| s02 | `s02_sample.py` | ✅ Implemented | Frame sampling (arithmetic) |
| s03 | `s03_detect.py` | 🔲 Stub | Object detection (YOLO TBD) |
| s04 | `s04_track.py` | 🔲 Stub | Object tracking (ByteTrack TBD) |
| s05 | `s05_segment.py` | 🔲 Stub | Candidate segment identification |
| s06 | `s06_vlm.py` | 🔲 Stub | VLM semantic analysis (model TBD) |
| s07 | `s07_events.py` | 🔲 Stub | Physical event extraction |
| s08 | `s08_states.py` | 🔲 Stub | State transitions |
| s09 | `s09_trajectories.py` | 🔲 Stub | 2-D trajectory extraction |
| s10 | `s10_score.py` | 🔲 Stub | Quality / confidence scoring |
| s11 | `s11_episode.py` | ✅ Implemented | Episode assembly → episode.json |
| s12 | `s12_evaluate.py` | ✅ Implemented | Evaluation → evaluation.json |
| s13 | `s13_preview.py` | 🔲 Stub | Preview video rendering |

---

## Expected Output Files

```
output/
├── detections.json         # DetectionFrame[] (empty in stub mode)
├── tracks.json             # Track[] (empty in stub mode)
├── candidate_segments.json # CandidateSegment[] (empty in stub mode)
├── events.json             # PhysicalEvent[] (empty in stub mode)
├── states.json             # {object_states, state_transitions} (empty in stub mode)
├── trajectories.json       # Trajectory2D[] (empty in stub mode)
├── episode.json            # PhysicalEpisode (always written)
├── evaluation.json         # EvaluationReport (always written)
└── preview.mp4             # (written when s13 is implemented)
```

---

## Project Structure

```
Physical Data Compiler/
├── input/                     # Drop your video here (git-ignored)
├── output/                    # Pipeline outputs (git-ignored)
├── config/
│   └── default.yaml           # Default configuration
├── src/
│   ├── pipeline.py            # Entry point: python -m src.pipeline
│   ├── config.py              # Configuration system (3-layer: YAML→CLI)
│   ├── context.py             # PipelineContext (shared stage state)
│   ├── logging_utils.py       # Structured logging
│   ├── interfaces/            # Abstract base classes (replaceable models)
│   │   ├── detector.py        # ObjectDetector ABC
│   │   ├── tracker.py         # ObjectTracker ABC
│   │   ├── vlm.py             # VisionLanguageModel ABC
│   │   └── pose.py            # PoseEstimator ABC
│   ├── models/                # Concrete implementations
│   │   ├── stub_detector.py   # StubDetector (zero-dep, always works)
│   │   ├── stub_tracker.py    # StubTracker
│   │   ├── stub_vlm.py        # StubLocalVLM, StubRemoteVLM
│   │   ├── yolo_detector.py   # YOLODetector (placeholder)
│   │   ├── bytetrack_tracker.py # ByteTrackTracker (placeholder)
│   │   ├── local_vlm.py       # LocalVLM (model TBD)
│   │   └── remote_vlm.py      # RemoteVLM (provider TBD)
│   ├── schema/                # Pydantic data models
│   │   ├── detection.py       # BoundingBox, Detection, DetectionFrame
│   │   ├── track.py           # TrackPoint, Track
│   │   ├── segment.py         # CandidateSegment
│   │   ├── event.py           # ActionType (enum), PhysicalEvent
│   │   ├── state.py           # ObjectState, StateTransition
│   │   ├── trajectory.py      # Trajectory2D (2-D only, never 3-D)
│   │   ├── episode.py         # PhysicalEpisode (root output)
│   │   └── evaluation.py      # EvaluationReport
│   └── stages/                # One file per pipeline stage
│       ├── s01_ingest.py  through  s13_preview.py
└── tests/
    ├── test_config.py
    ├── test_interfaces.py
    ├── test_schema.py
    └── test_stages_smoke.py   # Full end-to-end stub run
```

---

## Configuration

All values are overridable. See [`config/default.yaml`](config/default.yaml) for full documentation.

Key settings:

| Setting | Default | Notes |
|---|---|---|
| `frame_sampling.fps` | `1.0` | 1 fps ≈ 600 frames/10 min (spec default) |
| `frame_sampling.max_frames` | `1200` | Hard cap |
| `frame_sampling.segment_fps` | `5.0` | Higher fps for VLM segment re-sampling |
| `detector.model` | `yolov8n` | Configurable; ABC is model-agnostic |
| `detector.confidence` | `0.35` | Min detection confidence |
| `vlm.enabled` | `false` | Heavy; disabled by default |
| `vlm.backend` | `LOCAL_MODEL` | `LOCAL_MODEL` or `REMOTE_MODEL` |
| `stub_mode` | `false` | `true` = all heavy stages use stubs |
| `pose.enabled` | `false` | Optional; disabled by default |

---

## Data Integrity Rules

These rules are enforced by the schema layer and the pipeline contract:

1. **No fabricated data.** Stubs return empty lists/dicts. Never invent detections, events, trajectories, or metrics.
2. **Trajectories are 2-D only.** `Trajectory2D.coordinate_space` is locked to `"2D_IMAGE_PIXELS"` by Pydantic. No code path can claim 3-D.
3. **Raw predictions are preserved.** `PhysicalEvent` stores `action + confidence + source` as-is. The quality engine (`s10`) assigns `review_status` — it never modifies raw fields.
4. **UNKNOWN means unknown.** Use `ActionType.UNKNOWN` when evidence is genuinely insufficient. Do not force an action to avoid UNKNOWN.
5. **SKIPPED stages are explicit.** A skipped stage writes empty output and records `status="SKIPPED"`. It never pretends to have run.

---

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
python -m pytest tests/ -v

# Lint
python -m ruff check src/ tests/

# Syntax check entry point
python -m py_compile src/pipeline.py

# Verify --help works
python -m src.pipeline --help

# Smoke test (creates synthetic video, runs full pipeline)
python -m pytest tests/test_stages_smoke.py -v
```

---

## Action Vocabulary (MVP v1, locked)

```
GRASP   RELEASE  PICK    PLACE   MOVE
PUSH    PULL     OPEN    CLOSE   INSERT
REMOVE  USE_TOOL TOUCH   INSPECT UNKNOWN
```

`UNKNOWN` is the default and must be used when the system cannot determine the action.

---

## Adding a New Model Backend

To swap the detector (example):

1. Create `src/models/my_detector.py` inheriting `ObjectDetector`.
2. Implement `load()`, `detect()`, `unload()`, `model_name`.
3. In `s03_detect.py`, instantiate `MyDetector` instead of `YOLODetector`.
4. No other files need to change.

The same pattern applies to trackers (`ObjectTracker`) and VLMs (`VisionLanguageModel`).
