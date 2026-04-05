"""
Mediapipe Detection Module

This module provides:
- Body and hair segmentation using MediaPipe Tasks API
- Head width detection using MediaPipe Pose (landmarks 7 & 8)
- Face detection for hair segmentation support

Pose and hand detection for other body measurements use vitpose_detection for higher accuracy.
"""

from .hair_segmentation import HairSegmenter, segment_hair
from .head_detection import HeadWidthDetector, detect_head_width
from .hand_detection import HandLandmarkDetector, detect_hands

__all__ = [
    'HairSegmenter',
    'HeadWidthDetector',
    'HandLandmarkDetector',
    'segment_hair',
    'detect_head_width',
    'detect_hands',
]
