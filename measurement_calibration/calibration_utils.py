"""
Calibration Utilities Module

This module contains shared utilities for measurement backdrop calibration:
- Detection parameter configurations and bounds
- Line detection methods
- OCR detection methods
- Calibration calculation utilities

Used by both calibrate_backdrop.py (optimization) and extract_norm_per_cm.py (application).
"""

import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, List, Tuple, Optional


# ============================================================================
# PARAMETER CONFIGURATIONS
# ============================================================================

# Default detection parameters (fallback when no optimization is available)
DEFAULT_PARAMS = {
    'canny_low': 50,
    'canny_high': 150,
    'hough_threshold': 106,
    'min_line_length_ratio': 0.47,
    'max_line_gap': 25,
    'line_scale_factor': 1.0,
    'ocr_scale_factor': 1.0,
    'ocr_region_width': 0.15,
    'ocr_psm_mode': 11
}

# Line detection parameter bounds for optimization (Stage 1)
LINE_PARAM_BOUNDS = np.array([
    [20, 100],    # canny_low: Lower threshold for Canny edge detection
    [80, 250],    # canny_high: Upper threshold for Canny edge detection
    [0.20, 0.55], # min_line_length_ratio: Minimum line length as ratio of image width
    [40, 250],    # hough_threshold: Threshold for Hough line detection
    [1.0, 3.5]    # line_scale_factor: Image upscaling for better line detection
])

# OCR parameter configurations to test (Stage 2)
# Format: (scale_factor, region_width, psm_mode)
OCR_CONFIGS = [
    # Standard region width (0.20) with varying scales
    (1.5, 0.20, 11), (2.0, 0.20, 11), (2.5, 0.20, 11), (3.0, 0.20, 11),
    (3.5, 0.20, 11), (4.0, 0.20, 11), (4.5, 0.20, 11), (5.0, 0.20, 11),
    # Narrow region for distant text
    (3.0, 0.15, 11), (4.0, 0.15, 11), (5.0, 0.15, 11),
    # Wide region for offset backdrops
    (2.5, 0.25, 11), (3.0, 0.25, 11), (3.5, 0.25, 11),
    # Alternative PSM modes (6: single uniform block, 12: sparse text with OSD)
    (3.0, 0.20, 6), (4.0, 0.20, 6),
    (3.0, 0.20, 12), (4.0, 0.20, 12)
]


# ============================================================================
# LINE DETECTION UTILITIES
# ============================================================================

def detect_lines_with_params(
    image: np.ndarray,
    canny_low: int,
    canny_high: int,
    hough_threshold: int,
    min_line_length_ratio: float,
    max_line_gap: int,
    line_scale_factor: float = 1.0,
    validate_thickness: bool = True,
    min_thickness: float = 2.0
) -> Tuple[List[int], List[float]]:
    """
    Detect horizontal lines with given parameters.

    Args:
        image: Input image as numpy array (BGR format)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        hough_threshold: Threshold for Hough line detection
        min_line_length_ratio: Minimum line length as ratio of image width
        max_line_gap: Maximum gap between line segments
        line_scale_factor: Scale factor for upscaling image before line detection
        validate_thickness: If True, validate line thickness
        min_thickness: Minimum line thickness in pixels

    Returns:
        Tuple of (y_coordinates, thicknesses)
    """
    h, w = image.shape[:2]

    # Upscale image if needed for better line detection
    if line_scale_factor > 1.0:
        scaled_image = cv2.resize(
            image,
            (int(w * line_scale_factor), int(h * line_scale_factor)),
            interpolation=cv2.INTER_CUBIC
        )
    else:
        scaled_image = image

    # Edge detection
    gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)

    # Line detection using Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,
        minLineLength=int(scaled_image.shape[1] * min_line_length_ratio),
        maxLineGap=max_line_gap
    )

    if lines is None:
        return [], []

    # Extract y-coordinates of horizontal lines
    y_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 3 or angle > 177:  # Horizontal lines only
            y_coords.append(int((y1 + y2) / 2))

    # Scale back to original coordinates if image was upscaled
    if line_scale_factor > 1.0:
        y_coords = [int(y / line_scale_factor) for y in y_coords]

    # Remove duplicates within 5 pixels
    y_coords = sorted(set(y_coords))
    filtered_coords = []
    for i, y in enumerate(y_coords):
        if i == 0 or y - filtered_coords[-1] > 5:
            filtered_coords.append(y)

    # Measure line thickness for validation (on original image)
    thicknesses = []
    if validate_thickness:
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        validated_coords = []
        for y in filtered_coords:
            thickness = measure_line_thickness(original_gray, y)
            if thickness >= min_thickness:  # Only keep lines above minimum thickness
                validated_coords.append(y)
                thicknesses.append(thickness)
        return validated_coords, thicknesses
    else:
        return filtered_coords, [0.0] * len(filtered_coords)


def measure_line_thickness(gray_image: np.ndarray, y: int, search_range: int = 10) -> float:
    """
    Measure the thickness of a horizontal line at a given y-coordinate.

    Args:
        gray_image: Grayscale image
        y: Y-coordinate of the line
        search_range: Vertical search range around y

    Returns:
        Estimated line thickness in pixels
    """
    h, w = gray_image.shape

    # Ensure y is within bounds
    y = max(search_range, min(h - search_range - 1, y))

    # Extract horizontal profile around the line
    profile = gray_image[y - search_range:y + search_range + 1, :]

    # Average across width to get vertical intensity profile
    vertical_profile = np.mean(profile, axis=1)

    # Find the darkest point (should be the line center)
    center_idx = np.argmin(vertical_profile)

    # Measure thickness by finding where intensity rises above threshold
    # Threshold: midpoint between min and mean intensity
    threshold = (np.min(vertical_profile) + np.mean(vertical_profile)) / 2

    # Count pixels below threshold (dark pixels = line)
    thickness = np.sum(vertical_profile < threshold)

    return float(thickness)


# ============================================================================
# OCR DETECTION UTILITIES
# ============================================================================

def detect_numbers(
    image: np.ndarray,
    scale_factor: float = 1.0,
    region_width: float = 0.15,
    psm_mode: int = 11,
    number_range: Tuple[int, int] = (10, 200)
) -> List[Tuple[int, int]]:
    """
    Detect measurement numbers using OCR.

    Args:
        image: Input image as numpy array (BGR format)
        scale_factor: Upscaling factor for small text (1.0-5.0)
        region_width: Width of left region to scan (0.15-0.30)
        psm_mode: Tesseract PSM mode (6, 11, or 12)
        number_range: Valid number range as (min, max) tuple

    Returns:
        List of (number, y_position) tuples
    """
    h, w = image.shape[:2]
    left_region = image[:, :int(w * region_width)]

    # Upscale if needed for better OCR on small text
    if scale_factor > 1.0:
        region_h, region_w = left_region.shape[:2]
        left_region = cv2.resize(
            left_region,
            (int(region_w * scale_factor), int(region_h * scale_factor)),
            interpolation=cv2.INTER_CUBIC
        )

    # Otsu thresholding
    gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is dark
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Tesseract with configurable PSM mode
    custom_config = f'--psm {psm_mode} --oem 3 -c tessedit_char_whitelist=0123456789'
    data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)

    # Extract numbers
    numbers_with_pos = []
    seen_numbers = set()

    min_num, max_num = number_range
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = float(data['conf'][i])
        if conf < 0:
            conf = 0

        if conf >= 0:
            clean_text = re.sub(r'[^\d]', '', text)
            if clean_text:
                try:
                    number = int(clean_text)
                    # Range check based on backdrop config
                    if min_num <= number <= max_num and number not in seen_numbers:
                        seen_numbers.add(number)
                        y = data['top'][i] + data['height'][i] // 2
                        # Scale back to original coordinates if upscaled
                        if scale_factor > 1.0:
                            y = int(y / scale_factor)
                        numbers_with_pos.append((number, y))
                except ValueError:
                    pass

    numbers_with_pos.sort(key=lambda x: x[1])
    return numbers_with_pos


# ============================================================================
# CALIBRATION CALCULATION UTILITIES
# ============================================================================

def calculate_calibration(
    lines: List[int],
    numbers: List[Tuple[int, int]],
    image_height: int
) -> Dict:
    """
    Calculate calibration data from detected lines and numbers.

    Args:
        lines: List of y-coordinates of detected lines
        numbers: List of (number, y_position) tuples
        image_height: Height of the image in pixels

    Returns:
        Dictionary containing calibration information:
        {
            "cm_per_normalized_unit": float,
            "pixels_per_cm": float,
            "detected_lines": int,
            "detected_numbers": int,
            "line_spacing_pixels": float,
            "line_spacing_cv": float,
            "cm_interval": int,
            "confidence": str
        }
    """
    if len(lines) >= 2 and len(numbers) >= 2:
        # Calculate average spacing between consecutive lines
        line_spacings = []
        for i in range(len(lines) - 1):
            spacing = lines[i + 1] - lines[i]
            line_spacings.append(spacing)

        avg_line_spacing_pixels = np.mean(line_spacings)
        spacing_cv = np.std(line_spacings) / avg_line_spacing_pixels if avg_line_spacing_pixels > 0 else 0

        # Determine cm interval between lines using OCR numbers
        number_diffs = []
        for i in range(len(numbers) - 1):
            num_diff = numbers[i + 1][0] - numbers[i][0]
            number_diffs.append(num_diff)

        # Most common difference should be the cm interval
        if number_diffs:
            cm_interval = abs(int(np.median(number_diffs)))
        else:
            # Default assumption: 10cm intervals
            cm_interval = 10

        # Calculate pixels per cm
        pixels_per_cm = avg_line_spacing_pixels / cm_interval

        # Calculate normalized units (0-1 range) per cm
        # Normalized unit = 1 / image_height
        normalized_unit_per_pixel = 1.0 / image_height
        cm_per_normalized_unit = 1.0 / (pixels_per_cm * normalized_unit_per_pixel)

        # Determine confidence
        if len(lines) >= 10 and len(numbers) >= 5:
            confidence = "high"
        elif len(lines) >= 5 and len(numbers) >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        calibration_data = {
            "cm_per_normalized_unit": float(cm_per_normalized_unit),
            "pixels_per_cm": float(pixels_per_cm),
            "detected_lines": len(lines),
            "detected_numbers": len(numbers),
            "line_spacing_pixels": float(avg_line_spacing_pixels),
            "line_spacing_cv": float(spacing_cv),
            "cm_interval": cm_interval,
            "confidence": confidence
        }
    else:
        # Not enough data
        calibration_data = {
            "cm_per_normalized_unit": None,
            "pixels_per_cm": None,
            "detected_lines": len(lines),
            "detected_numbers": len(numbers),
            "line_spacing_pixels": None,
            "line_spacing_cv": None,
            "cm_interval": None,
            "confidence": "failed",
            "error": "Not enough lines or numbers detected"
        }

    return calibration_data


def calculate_score(
    lines: List[int],
    numbers: List[Tuple[int, int]],
    expected_lines: int,
    expected_numbers: int
) -> float:
    """
    Calculate a quality score for detection results.

    Scoring breakdown:
    - Line count accuracy: 50% weight
    - Line spacing consistency: 30% weight
    - Number count accuracy: 20% weight

    Args:
        lines: List of y-coordinates of detected lines
        numbers: List of (number, y_position) tuples
        expected_lines: Expected number of lines
        expected_numbers: Expected number of numbers

    Returns:
        Score between 0.0 and 1.0
    """
    scores = []

    # Line count accuracy (50% weight)
    if expected_lines > 0:
        line_error = abs(len(lines) - expected_lines) / expected_lines
        line_accuracy = max(0, 1 - line_error)
        scores.append(line_accuracy * 0.50)
    else:
        scores.append(0)

    # Line spacing consistency (30% weight)
    if len(lines) >= 2:
        spacings = np.diff(sorted(lines))
        spacing_cv = np.std(spacings) / np.mean(spacings)
        # Normalize: CV of 0.3 or higher = score of 0
        spacing_score = max(0, 1 - spacing_cv * 3.33)
        scores.append(spacing_score * 0.30)
    else:
        scores.append(0)

    # Number count accuracy (20% weight)
    if expected_numbers > 0:
        number_error = abs(len(numbers) - expected_numbers) / expected_numbers
        number_accuracy = max(0, 1 - number_error)
        scores.append(number_accuracy * 0.20)
    else:
        scores.append(0)

    return sum(scores)
