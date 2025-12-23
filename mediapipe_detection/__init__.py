"""
Mediapipe Detection Module

This module provides detection classes for pose, face, hand landmarks, and hair segmentation.
"""

from .pose_detection import PoseDetector, detect_pose
from .face_detection import FaceDetector, detect_face
from .hand_detection import HandDetector, detect_hands
from .hair_segmentation import HairSegmenter, segment_hair

__all__ = [
    'PoseDetector',
    'FaceDetector',
    'HandDetector',
    'HairSegmenter',
    'detect_pose',
    'detect_face',
    'detect_hands',
    'segment_hair'
]
