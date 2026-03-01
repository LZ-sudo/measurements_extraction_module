"""
ViTPose Movement Accuracy Module

Evaluates how accurately an animated model replicates the movement
of a subject in a ground truth video using pose-based metrics.
"""

from .movement_evaluator import MovementEvaluator
from .body_region_config import REGION_REGISTRY, DEFAULT_REGIONS

__all__ = ["MovementEvaluator", "REGION_REGISTRY", "DEFAULT_REGIONS"]
