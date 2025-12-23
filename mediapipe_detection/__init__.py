"""
Mediapipe Detection Module

This module provides detection classes for pose, face, hand landmarks,
and segmentation for hair and body.
"""

from .pose_detection import PoseDetector, detect_pose
from .face_detection import FaceDetector, detect_face
from .hand_detection import HandDetector, detect_hands
from .hair_segmentation import HairSegmenter, segment_hair
from .body_segmentation import BodySegmenter, segment_body

__all__ = [
    'PoseDetector',
    'FaceDetector',
    'HandDetector',
    'HairSegmenter',
    'BodySegmenter',
    'detect_pose',
    'detect_face',
    'detect_hands',
    'segment_hair',
    'segment_body'
]
