"""PoseEstimator abstract base class (optional component).

Disabled by default (``pose.enabled: false`` in config).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class PoseEstimator(ABC):
    """Interface for optional pose / hand-keypoint estimators.

    Contract:
      - ``estimate()`` never raises; it returns None on failure.
      - Returns None if no pose is detected or if the estimator is disabled.
      - Never fabricate keypoints.
    """

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> dict[str, Any] | None:
        """Estimate pose or hand keypoints in a single frame.

        Args:
            frame: HxWxC uint8 BGR image (OpenCV convention).

        Returns:
            Dict with keypoint data, or ``None`` if no pose detected.
        """

    @property
    @abstractmethod
    def estimator_name(self) -> str:
        """Human-readable estimator identifier."""
