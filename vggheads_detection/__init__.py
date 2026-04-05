"""
VGGHeads Detection Module

Provides head detection using VGGHeads (MIT license), replacing YOLOv8.
Model weights are downloaded automatically from HuggingFace on first use.
"""

from .head_detection import HeadDetector, detect_head

__all__ = ["HeadDetector", "detect_head"]
