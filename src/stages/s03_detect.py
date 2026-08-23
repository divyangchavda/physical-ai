"""Stage 03 — Object detection.

Runs the configured ObjectDetector on all sampled frames.
SKIPPED in stub mode — produces an empty-but-valid detections.json.

Output file: output/detections.json
Output context: ctx.detection_frames (list[DetectionFrame])
"""
from __future__ import annotations

import json
import time

import cv2

from src.context import PipelineContext
from src.interfaces.detector import ObjectDetector
from src.logging_utils import get_logger
from src.models.stub_detector import StubDetector
from src.schema.detection import DetectionFrame
from src.schema.episode import PipelineStageStatus

logger = get_logger(__name__)
STAGE = "s03_detect"


def _build_detector(ctx: PipelineContext) -> ObjectDetector:
    """Return the appropriate detector for the current config."""
    if ctx.config.stub_mode:
        return StubDetector()
    
    # Determine device
    device = ctx.config.detector.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    
    # Get detector backend (default to yolov8)
    backend = getattr(ctx.config.detector, 'backend', 'yolov8')
    
    try:
        if backend == 'groundingdino_hf':
            from src.models.groundingdino_hf_detector import (
                DEFAULT_MODEL_ID,
                GroundingDINOHFDetector,
            )

            text_prompt = getattr(ctx.config.detector, 'text_prompt', None)
            if not text_prompt:
                logger.warning(
                    "[%s] groundingdino_hf backend selected but no text_prompt "
                    "provided; falling back to StubDetector", STAGE
                )
                return StubDetector()

            # `model` doubles as the Hub id here; ignore the yolo default.
            model_id = getattr(ctx.config.detector, 'model', None)
            if not model_id or model_id.startswith('yolo'):
                model_id = DEFAULT_MODEL_ID

            return GroundingDINOHFDetector(
                text_prompt=text_prompt,
                box_threshold=ctx.config.detector.confidence,
                text_threshold=getattr(ctx.config.detector, 'text_threshold', 0.25),
                device=device,
                model_id=model_id,
                nms_iou=ctx.config.detector.nms_iou,
                drop_unlabeled=getattr(ctx.config.detector, 'drop_unlabeled', True),
            )

        elif backend == 'groundingdino':
            from src.models.groundingdino_detector import GroundingDINODetector
            
            # Get text prompt from config
            text_prompt = getattr(ctx.config.detector, 'text_prompt', None)
            if not text_prompt:
                logger.warning(
                    "[%s] GroundingDINO backend selected but no text_prompt provided; "
                    "falling back to StubDetector", STAGE
                )
                return StubDetector()
            
            text_threshold = getattr(ctx.config.detector, 'text_threshold', 0.25)
            
            return GroundingDINODetector(
                text_prompt=text_prompt,
                box_threshold=ctx.config.detector.confidence,
                text_threshold=text_threshold,
                device=device,
                model_checkpoint=getattr(ctx.config.detector, 'model_checkpoint', None),
                config_file=getattr(ctx.config.detector, 'config_file', None),
            )
        
        elif backend == 'yolo-world':
            from src.models.yoloworld_detector import YOLOWorldDetector
            
            # Get vocabulary from config
            vocabulary = getattr(ctx.config.detector, 'vocabulary', [])
            if not vocabulary:
                logger.warning(
                    "[%s] YOLO-World backend selected but no vocabulary provided; "
                    "falling back to YOLODetector", STAGE
                )
                from src.models.yolo_detector import YOLODetector
                return YOLODetector(
                    model_name=ctx.config.detector.model,
                    confidence=ctx.config.detector.confidence,
                    nms_iou=ctx.config.detector.nms_iou,
                    device=device,
                )
            
            return YOLOWorldDetector(
                model_name=ctx.config.detector.model,
                vocabulary=vocabulary,
                confidence=ctx.config.detector.confidence,
                nms_iou=ctx.config.detector.nms_iou,
                device=device,
            )
        else:
            # Default: YOLOv8
            from src.models.yolo_detector import YOLODetector
            return YOLODetector(
                model_name=ctx.config.detector.model,
                confidence=ctx.config.detector.confidence,
                nms_iou=ctx.config.detector.nms_iou,
                device=device,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[%s] Cannot instantiate detector (%s); falling back to StubDetector",
            STAGE, exc,
        )
        return StubDetector()


def _write_output(ctx: PipelineContext, status: str = "OK") -> None:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = ctx.output_dir / "detections.json"
    data = [df.model_dump(mode="json") for df in ctx.detection_frames]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run(ctx: PipelineContext) -> PipelineStageStatus:
    t0 = time.monotonic()

    if ctx.config.stub_mode:
        logger.info(
            "[%s] stub_mode=True — SKIPPED (no detections fabricated)", STAGE
        )
        ctx.detection_frames = []
        _write_output(ctx, status="SKIPPED")
        return PipelineStageStatus(
            stage=STAGE, status="SKIPPED",
            message="stub_mode: detection stage skipped",
            duration_sec=time.monotonic() - t0,
        )

    if not ctx.sampled_frame_infos:
        msg = "No sampled frames — s02_sample must run first"
        logger.error("[%s] %s", STAGE, msg)
        return PipelineStageStatus(stage=STAGE, status="ERROR", message=msg)

    detector = _build_detector(ctx)
    try:
        detector.load()
    except NotImplementedError as exc:
        logger.warning("[%s] %s — using StubDetector", STAGE, exc)
        detector = StubDetector()

    # Use the video path embedded in the sampling plan when available.
    # SampledFrameInfo.video_path defaults to Path() (empty) when constructed
    # by older code; fall back to ctx.video_path in that case.
    _first_info = ctx.sampled_frame_infos[0] if ctx.sampled_frame_infos else None
    _video_path = (
        _first_info.video_path
        if _first_info is not None and _first_info.video_path.parts
        else ctx.video_path
    )
    cap = cv2.VideoCapture(str(_video_path))
    detection_frames: list[DetectionFrame] = []
    
    inference_time_sec = 0.0
    frames_processed = 0

    try:
        for info in ctx.sampled_frame_infos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(info.frame_index))
            ret, frame = cap.read()
            if not ret:
                detection_frames.append(DetectionFrame(
                    frame_index=info.frame_index,
                    timestamp_sec=info.timestamp_sec,
                    status="ERROR",
                    message="Could not read frame from video",
                ))
                continue
            
            t_inf0 = time.monotonic()
            detections = detector.detect(frame, info.frame_index, info.timestamp_sec)
            t_inf1 = time.monotonic()
            
            inference_time_sec += (t_inf1 - t_inf0)
            frames_processed += 1
            
            detection_frames.append(DetectionFrame(
                frame_index=info.frame_index,
                timestamp_sec=info.timestamp_sec,
                detections=detections,
                status="OK",
            ))
    finally:
        cap.release()
        detector.unload()

    ctx.detection_frames = detection_frames
    _write_output(ctx)

    n_det = sum(len(df.detections) for df in detection_frames)
    avg_inf = inference_time_sec / frames_processed if frames_processed > 0 else 0.0
    det_per_frame = n_det / frames_processed if frames_processed > 0 else 0.0
    
    logger.info(
        "[%s] %d/%d frames processed | detections: %d total (%.2f/frame) | inference: %.2fs total (%.3fs avg/frame)",
        STAGE, frames_processed, len(ctx.sampled_frame_infos), n_det, det_per_frame, inference_time_sec, avg_inf
    )
    
    msg = f"Processed {frames_processed} frames, {n_det} detections. Inference: {inference_time_sec:.2f}s total, {avg_inf:.3f}s avg/frame."
    
    return PipelineStageStatus(
        stage=STAGE, status="OK", message=msg, duration_sec=time.monotonic() - t0
    )
