"""GroundingDINO detector implementation.

Status: IMPLEMENTED

Integrates GroundingDINO for open-vocabulary object detection using text prompts.
Uses the pre-trained model checkpoint for zero-shot detection.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.interfaces.detector import ObjectDetector
from src.logging_utils import get_logger
from src.schema.detection import BoundingBox, Detection

logger = get_logger(__name__)


class GroundingDINODetector(ObjectDetector):
    """GroundingDINO-based object detector for open-vocabulary detection.
    
    Uses text prompts to detect arbitrary objects without fine-tuning.
    """

    def __init__(
        self,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda",
        model_checkpoint: str | None = None,
        config_file: str | None = None,
    ) -> None:
        """Initialize GroundingDINO detector.
        
        Args:
            text_prompt: Text prompt for detection (e.g., "person . box . table .")
            box_threshold: Minimum confidence for box detection
            text_threshold: Minimum confidence for text matching
            device: "cuda" or "cpu"
            model_checkpoint: Path to model weights (default: ~/GroundingDINO/groundingdino_swint_ogc.pth)
            config_file: Path to config file (default: ~/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py)
        """
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        
        # Default paths
        home = Path.home()
        self.model_checkpoint = model_checkpoint or str(home / "GroundingDINO" / "groundingdino_swint_ogc.pth")
        self.config_file = config_file or str(home / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
        
        self.model = None
        self._class_name_to_id: dict[str, int] = {}
        
        # Parse class names from text prompt
        # GroundingDINO expects prompts like "person . box . table ."
        self._parse_class_names()

    def _parse_class_names(self) -> None:
        """Extract class names from text prompt and assign IDs."""
        # Split by period and strip whitespace
        parts = [p.strip() for p in self.text_prompt.split('.')]
        # Filter out empty strings
        class_names = [p for p in parts if p]
        
        # Assign sequential IDs
        for idx, name in enumerate(class_names):
            self._class_name_to_id[name.lower()] = idx
        
        logger.info(
            "GroundingDINO: Parsed %d classes from prompt: %s",
            len(class_names), class_names
        )

    def load(self) -> None:
        """Load GroundingDINO model."""
        if self.model is not None:
            logger.warning("GroundingDINO model already loaded")
            return
        
        try:
            # Import GroundingDINO modules
            from groundingdino.util.inference import load_model
            
            logger.info("Loading GroundingDINO model from: %s", self.model_checkpoint)
            logger.info("Using config: %s", self.config_file)
            
            # Verify files exist
            if not Path(self.model_checkpoint).exists():
                raise FileNotFoundError(f"Model checkpoint not found: {self.model_checkpoint}")
            if not Path(self.config_file).exists():
                raise FileNotFoundError(f"Config file not found: {self.config_file}")
            
            # Load model
            self.model = load_model(self.config_file, self.model_checkpoint, device=self.device)
            
            logger.info(
                "GroundingDINO loaded successfully (device=%s, box_threshold=%.2f, text_threshold=%.2f)",
                self.device, self.box_threshold, self.text_threshold
            )
            
        except ImportError as e:
            raise RuntimeError(
                "GroundingDINO is not installed. Please install it from: "
                "https://github.com/IDEA-Research/GroundingDINO"
            ) from e
        except Exception as e:
            logger.exception("Failed to load GroundingDINO model")
            raise RuntimeError(f"Failed to load GroundingDINO: {e}") from e

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_sec: float,
    ) -> list[Detection]:
        """Run GroundingDINO detection on a single frame.
        
        Args:
            frame: HxWxC uint8 BGR image (OpenCV format)
            frame_index: 0-based frame index
            timestamp_sec: timestamp in seconds
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            from groundingdino.util.inference import predict
            import groundingdino.datasets.transforms as T
            
            # Convert BGR to RGB
            frame_rgb = frame[:, :, ::-1].copy()
            
            # Convert to PIL Image
            image_pil = Image.fromarray(frame_rgb)
            
            # Apply GroundingDINO transforms
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed, _ = transform(image_pil, None)
            
            # Run detection
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )
            
            # Convert to Detection objects
            detections = []
            h, w = frame.shape[:2]
            
            for idx, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
                # GroundingDINO returns boxes in [cx, cy, w, h] format normalized to [0, 1]
                cx, cy, box_w, box_h = box.tolist()
                
                # Convert to pixel coordinates
                cx_px = cx * w
                cy_px = cy * h
                w_px = box_w * w
                h_px = box_h * h
                
                # Convert to [x1, y1, x2, y2]
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                x2 = cx_px + w_px / 2
                y2 = cy_px + h_px / 2
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Clean up phrase (remove trailing punctuation)
                class_name = phrase.strip().rstrip('.')
                
                # Get class ID (use mapping or default to 0)
                class_id = self._class_name_to_id.get(class_name.lower(), 0)
                
                # Convert logit to confidence (sigmoid is already applied by GroundingDINO)
                confidence = float(logit)
                
                detection = Detection(
                    detection_id=f"dino_{frame_index}_{uuid.uuid4().hex[:8]}",
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    source="groundingdino",
                    is_estimated=True,
                )
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.exception("GroundingDINO detection failed on frame %d", frame_index)
            # Return empty list on error (never crash the pipeline)
            return []

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("GroundingDINO model unloaded")

    @property
    def model_name(self) -> str:
        """Return model identifier for provenance."""
        return "groundingdino"
