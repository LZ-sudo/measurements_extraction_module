"""
ArUco Marker-Based Calibration Configuration
==============================================

This module contains all configuration parameters for ArUco marker-based backdrop calibration,
including marker specifications, detection parameters, and quality thresholds.
"""

import cv2

# ============================================================================
# ARUCO MARKER CONFIGURATION
# ============================================================================

# Default marker size (should match printed marker size in cm)
# 12cm markers recommended for better accuracy (more pixels per marker)
DEFAULT_MARKER_SIZE_CM = 12.0

# ArUco dictionary type
# DICT_4X4_50: 4×4 bit markers, 50 unique IDs (IDs 0-49)
# This provides good balance between detection reliability and uniqueness
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Required marker IDs for backdrop corners
# These markers must be detected for successful calibration
REQUIRED_MARKER_IDS = {0, 1, 2, 3}

# Marker placement convention:
# - Marker 0 (ID: 0) → Top-Left corner
# - Marker 1 (ID: 1) → Top-Right corner
# - Marker 2 (ID: 2) → Bottom-Left corner
# - Marker 3 (ID: 3) → Bottom-Right corner

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

# Confidence is determined by marker size variation (coefficient of variation)
# Lower variation = higher confidence = more consistent perspective

# High confidence: marker sizes vary by less than 2%
CONFIDENCE_HIGH_THRESHOLD = 0.02

# Medium confidence: marker sizes vary by less than 5%
CONFIDENCE_MEDIUM_THRESHOLD = 0.05

# Above 5% variation is considered low confidence
# (May indicate perspective distortion, markers not flat, or measurement errors)

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Colors for visualization (BGR format)
VIZ_COLOR_MARKER = (0, 255, 0)       # Green for detected markers
VIZ_COLOR_BACKDROP = (255, 0, 0)     # Blue for backdrop corners
VIZ_COLOR_CENTER = (255, 0, 0)       # Blue for marker centers
VIZ_COLOR_TEXT = (0, 0, 255)         # Red for text overlays

# Marker and corner sizes
VIZ_CORNER_RADIUS = 8
VIZ_BACKDROP_LINE_THICKNESS = 3
VIZ_TEXT_SCALE = 0.8
VIZ_TEXT_THICKNESS = 2

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

# Minimum number of markers required for calibration
MIN_MARKERS_REQUIRED = 4

# Maximum allowed marker size variation for acceptance
# If variation exceeds this, calibration may still succeed but with "low" confidence
MAX_MARKER_VARIATION_FOR_ACCEPTANCE = 0.15  # 15%

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Default visualization output filename
DEFAULT_VISUALIZATION_FILENAME = "aruco_backdrop_detection.jpg"

# Calibration method identifier (for JSON output)
CALIBRATION_METHOD = "aruco_markers"
