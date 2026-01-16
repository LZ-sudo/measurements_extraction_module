"""
Mediapipe Detection Module

This module provides segmentation classes for hair and body using MediaPipe Tasks API.
Pose and hand detection have been moved to rtmlib_detection for higher accuracy.
"""

from .hair_segmentation import HairSegmenter, segment_hair
from .body_segmentation import BodySegmenter, segment_body

__all__ = [
    'HairSegmenter',
    'BodySegmenter',
    'segment_hair',
    'segment_body'
]
