"""
RTMLib Detection Module

This module provides pose and hand detection using RTMLib (RTMW/DWPose).
It offers higher accuracy than MediaPipe for pose estimation with 133 wholebody keypoints.
"""

from .pose_detection import PoseDetector, detect_pose
from .hand_detection import HandDetector, detect_hands

__all__ = ['PoseDetector', 'detect_pose', 'HandDetector', 'detect_hands']
