"""
Mediapipe Detection Module

This module provides:
- Body and hair segmentation using MediaPipe Tasks API
- Head width detection using MediaPipe Pose (landmarks 7 & 8)
- Face detection for hair segmentation support

Pose and hand detection for other body measurements use vitpose_detection for higher accuracy.
"""

from .hair_segmentation import HairSegmenter, segment_hair
from .face_detection import FaceDetector, detect_face
from .head_detection import HeadWidthDetector, detect_head_width

__all__ = [
    'HairSegmenter',
    'FaceDetector',
    'HeadWidthDetector',
    'segment_hair',
    'detect_face',
    'detect_head_width'
]
