"""
ViTPose Detection Module

This module provides pose and hand detection using ViTPose models.
ViTPose offers higher accuracy than RTMLib, especially for challenging
poses and clothing occlusion scenarios.

Classes:
    PoseDetector: Detects body pose landmarks (133 wholebody keypoints).
    HandDetector: Detects hand landmarks from the wholebody model.

Functions:
    detect_pose: Convenience function for pose detection from image file.
    detect_hands: Convenience function for hand detection from image file.

Example:
    >>> from vitpose_detection import PoseDetector
    >>> detector = PoseDetector(model_name='h')  # Use ViTPose-Huge
    >>> landmarks = detector.detect(image)
"""

from .config_model import (
    get_model_paths,
    get_cache_dir,
    DEFAULT_VITPOSE_MODEL,
    DEFAULT_YOLO_MODEL,
)
from .pose_detection import PoseDetector, detect_pose
from .hand_detection import HandDetector, detect_hands

__all__ = [
    # Classes
    'PoseDetector',
    'HandDetector',
    # Functions
    'detect_pose',
    'detect_hands',
    'get_model_paths',
    'get_cache_dir',
    # Constants
    'DEFAULT_VITPOSE_MODEL',
    'DEFAULT_YOLO_MODEL',
]
