"""
Calibration Configuration
=========================

This module contains all configuration parameters for the backdrop calibration system,
including OCR settings, line detection parameters, and expected measurements.

Author: Calibration Module
Date: 2025
"""

# ============================================================================
# BACKDROP MEASUREMENT CONFIGURATION
# ============================================================================

# Expected detections on the backdrop
EXPECTED_LINES = 20          # Number of bold 10cm demarcation lines
EXPECTED_NUMBERS = 20        # Number of measurement labels (10, 20, 30...200)
NUMBER_RANGE = (10, 200)     # Min and max numbers on backdrop (cm)
CM_INTERVAL = 10             # Interval between consecutive numbers (cm)

# ============================================================================
# CORNER DETECTION CONFIGURATION
# ============================================================================

# OCR configuration for corner number detection
CORNER_DETECTION_CONFIG = {
    'scale_factors': [3.0, 3.5, 4.0, 4.5],           # Scale factors for OCR
    'preprocess_methods': ['clahe', 'adaptive'],      # Preprocessing methods
    'psm_modes': [6, 7, 11],                         # Tesseract PSM modes
}

# Fallback OCR configuration (used if region-based search fails)
CORNER_FALLBACK_CONFIG = {
    'scale_factors': [3.0, 3.5, 4.0, 2.5, 4.5, 5.0, 2.0, 1.5],
    'preprocess_methods': ['clahe', 'adaptive', 'otsu', 'morph'],
    'psm_modes': [11, 6, 7, 12],
}

# ROI margins around detected corners (pixels)
ROI_MARGIN_X = 80  # Horizontal margin to include full number width + backdrop edge
ROI_MARGIN_Y = 60  # Vertical margin to include full number height + backdrop edge

# Corner geometry validation thresholds (percentage of image dimensions)
CORNER_VALIDATION_THRESHOLD_TOP = 0.4     # Top corners should be in upper 40%
CORNER_VALIDATION_THRESHOLD_BOTTOM = 0.6  # Bottom corners should be in lower 60%
CORNER_VALIDATION_THRESHOLD_LEFT = 0.4    # Left corners should be in left 40%
CORNER_VALIDATION_THRESHOLD_RIGHT = 0.6   # Right corners should be in right 60%

# ============================================================================
# NUMBER DETECTION (OCR) CONFIGURATION
# ============================================================================

# OCR region width (percentage of image width to scan on left/right)
NUMBER_DETECTION_REGION_WIDTH = 0.30

# OCR scale factor for better text recognition
NUMBER_DETECTION_SCALE_FACTOR = 4.0

# Preprocessing methods to try for number detection
NUMBER_DETECTION_PREPROCESS_METHODS = ['clahe', 'adaptive', 'otsu']

# Tesseract PSM modes to try
NUMBER_DETECTION_PSM_MODES = [11, 6, 7]

# ============================================================================
# LINE DETECTION CONFIGURATION
# ============================================================================

# Minimum line thickness (pixels) to be considered a bold demarcation line
LINE_MIN_THICKNESS = 3.0

# Canny edge detection thresholds (low, high) - multiple configs for robustness
LINE_CANNY_CONFIGS = [
    (30, 100),   # Original - good for clear lines
    (50, 150),   # Higher threshold - better for bold lines
    (40, 120),   # Medium threshold
    (20, 80),    # Lower threshold - catch fainter lines
]

# Hough Transform parameters (as percentage of image width)
LINE_HOUGH_THRESHOLD = 0.12      # Minimum votes (12% of width)
LINE_HOUGH_MIN_LENGTH = 0.30     # Minimum line length (30% of width)
LINE_HOUGH_MAX_GAP = 60          # Maximum gap between line segments (pixels)
LINE_MIN_LENGTH_VALIDATION = 0.35  # Minimum line length for validation (35% of width)

# Line deduplication threshold (pixels)
LINE_DEDUP_THRESHOLD = 15  # Lines within 15px are considered duplicates

# Line thickness measurement search range (pixels)
LINE_THICKNESS_SEARCH_RANGE = 15

# ============================================================================
# NUMBER-LINE MATCHING CONFIGURATION
# ============================================================================

# Maximum distance (pixels) between a number and line to be considered a match
# This should be strict (10-15px) to ensure only truly intersecting lines are matched
NUMBER_LINE_MATCH_DISTANCE = 15

# Minimum length of consecutive sequence to use for calibration
MIN_CONSECUTIVE_SEQUENCE_LENGTH = 3

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Colors for visualization (BGR format)
VIZ_COLOR_CORNER = (255, 0, 0)      # Blue for corners
VIZ_COLOR_ROI = (0, 255, 0)         # Green for ROI boundary
VIZ_COLOR_NUMBER = (255, 0, 255)    # Magenta for numbers
VIZ_COLOR_LINE = (0, 255, 255)      # Yellow for lines

# Marker sizes
VIZ_CORNER_RADIUS = 8
VIZ_NUMBER_RADIUS = 6
VIZ_LINE_THICKNESS = 2
VIZ_ROI_THICKNESS = 3
