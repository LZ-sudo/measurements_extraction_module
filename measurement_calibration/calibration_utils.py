"""
ArUco-Based Calibration Utilities Module

This module contains utilities for ArUco marker-based backdrop calibration:
- ArUco marker detection and corner extraction
- Marker size calculation for scale determination
- Backdrop corner extraction from marker positions

Used by calibrate_backdrop.py for robust, accurate calibration.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# ARUCO MARKER DETECTION UTILITIES
# ============================================================================

def detect_aruco_markers(
    image: np.ndarray,
    aruco_dict_type: int = cv2.aruco.DICT_4X4_50
) -> Tuple[Optional[List], Optional[np.ndarray], List]:
    """
    Detect ArUco markers in the image.

    Args:
        image: Input image (BGR)
        aruco_dict_type: ArUco dictionary type

    Returns:
        Tuple of (corners, ids, rejected):
            - corners: List of detected marker corners (each marker has 4 corners)
            - ids: Array of detected marker IDs
            - rejected: List of rejected marker candidates
    """
    # Get ArUco dictionary and detection parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters()

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=aruco_params
    )

    return corners, ids, rejected


def get_marker_center(corners: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the center point of a marker from its corners.

    Args:
        corners: 4x2 array of marker corner coordinates

    Returns:
        (x, y) center coordinates
    """
    center_x = np.mean(corners[:, 0])
    center_y = np.mean(corners[:, 1])
    return float(center_x), float(center_y)


def calculate_marker_size_pixels(corners: np.ndarray) -> float:
    """
    Calculate the size of a marker in pixels.

    Computes the average of all 4 side lengths for robust size estimation.

    Args:
        corners: 4x2 array of marker corner coordinates

    Returns:
        Average side length in pixels
    """
    # Calculate all 4 side lengths
    side_lengths = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        length = np.linalg.norm(p2 - p1)
        side_lengths.append(length)

    # Return average
    return float(np.mean(side_lengths))


def organize_markers_by_id(
    corners: List,
    ids: np.ndarray,
    required_ids: set = {0, 1, 2, 3}
) -> Optional[Dict[int, np.ndarray]]:
    """
    Organize detected markers by their IDs and verify all required markers are present.

    Args:
        corners: List of marker corners from ArUco detection
        ids: Array of marker IDs from ArUco detection
        required_ids: Set of required marker IDs (default: {0, 1, 2, 3})

    Returns:
        Dictionary mapping marker ID to corners, or None if not all required markers found
    """
    if ids is None or len(ids) < len(required_ids):
        return None

    detected_ids = set(ids.flatten().tolist())

    # Check if all required markers are detected
    if not required_ids.issubset(detected_ids):
        return None

    # Create mapping
    marker_dict = {}
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in required_ids:
            # corners[i] shape is (1, 4, 2), extract to (4, 2)
            marker_dict[marker_id] = corners[i][0]

    return marker_dict


def get_backdrop_corners_from_markers(
    marker_dict: Dict[int, np.ndarray]
) -> np.ndarray:
    """
    Get backdrop corner positions from marker centers.

    Marker placement convention:
    - Marker 0 (ID: 0) → Top-Left
    - Marker 1 (ID: 1) → Top-Right
    - Marker 2 (ID: 2) → Bottom-Left
    - Marker 3 (ID: 3) → Bottom-Right

    Args:
        marker_dict: Dictionary mapping marker ID to corners

    Returns:
        4x2 array of backdrop corners [top-left, top-right, bottom-right, bottom-left]
    """
    # Get center of each marker
    top_left = get_marker_center(marker_dict[0])
    top_right = get_marker_center(marker_dict[1])
    bottom_left = get_marker_center(marker_dict[2])
    bottom_right = get_marker_center(marker_dict[3])

    # Return in standard order: TL, TR, BR, BL
    corners = np.array([
        top_left,
        top_right,
        bottom_right,
        bottom_left
    ], dtype=np.float32)

    return corners


def calculate_calibration_from_markers(
    marker_dict: Dict[int, np.ndarray],
    marker_size_cm: float,
    image_height: int
) -> Dict:
    """
    Calculate calibration data from detected ArUco markers.

    Args:
        marker_dict: Dictionary mapping marker ID to corners
        marker_size_cm: Physical size of markers in cm
        image_height: Original image height in pixels

    Returns:
        Dictionary containing calibration data
    """
    # Calculate marker sizes for all 4 markers and average
    marker_sizes_px = []
    for marker_id in sorted(marker_dict.keys()):
        marker_corners = marker_dict[marker_id]
        size_px = calculate_marker_size_pixels(marker_corners)
        marker_sizes_px.append(size_px)

    avg_marker_size_px = np.mean(marker_sizes_px)
    marker_size_variation = np.std(marker_sizes_px) / avg_marker_size_px if avg_marker_size_px > 0 else 0

    # Calculate pixels per cm
    pixels_per_cm = avg_marker_size_px / marker_size_cm

    # Calculate cm_per_normalized_unit
    # normalized_unit = 1.0 means full image height
    # pixels_per_cm tells us how many pixels = 1 cm
    # So: cm_per_normalized_unit = image_height_pixels / pixels_per_cm
    cm_per_normalized_unit = image_height / pixels_per_cm

    # Determine confidence based on marker size variation
    if marker_size_variation < 0.02:
        confidence = "high"
    elif marker_size_variation < 0.05:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "cm_per_normalized_unit": float(cm_per_normalized_unit),
        "pixels_per_cm": float(pixels_per_cm),
        "detected_markers": int(len(marker_dict)),
        "detected_marker_ids": [int(k) for k in sorted(marker_dict.keys())],
        "marker_size_cm": float(marker_size_cm),
        "marker_size_pixels_avg": float(avg_marker_size_px),
        "marker_size_pixels_individual": [float(s) for s in marker_sizes_px],
        "marker_size_variation_cv": float(marker_size_variation),
        "confidence": confidence
    }


def get_roi_bounds_from_corners(backdrop_corners: np.ndarray) -> List[int]:
    """
    Calculate ROI bounding box from backdrop corners.

    Args:
        backdrop_corners: 4x2 array of backdrop corner coordinates

    Returns:
        List of [x_min, y_min, width, height]
    """
    x_coords = backdrop_corners[:, 0]
    y_coords = backdrop_corners[:, 1]

    x_min = int(np.min(x_coords))
    y_min = int(np.min(y_coords))
    x_max = int(np.max(x_coords))
    y_max = int(np.max(y_coords))

    return [x_min, y_min, x_max - x_min, y_max - y_min]
