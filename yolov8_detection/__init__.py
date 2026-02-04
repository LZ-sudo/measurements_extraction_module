"""
YOLOv8 Detection Module

This module provides detection capabilities using YOLOv8 models.
Currently includes head detection for height measurement.
"""

from .head_detection import HeadDetector, detect_head

__all__ = ['HeadDetector', 'detect_head']
