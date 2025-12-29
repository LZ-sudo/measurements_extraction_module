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
# CORNER DETECTION UTILITIES
# ============================================================================

def find_corner_numbers_adaptive(
    image: np.ndarray,
    top_number: int,
    bottom_number: int,
    number_range: Tuple[int, int],
    cm_interval: int,
    debug: bool = False
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Adaptively search for four corner numbers using multiple OCR configurations.

    This method tries different combinations of:
    - Scale factors (expanded range for better coverage)
    - Preprocessing methods (clahe, adaptive, otsu, morph)
    - PSM modes (11, 6, 7, 12)
    - Region-based search focusing on likely corner areas

    Args:
        image: Input image
        top_number: Number at top corners (e.g., 200)
        bottom_number: Number at bottom corners (e.g., 10)
        number_range: Valid number range as (min, max) tuple
        cm_interval: Valid number interval (default: 10)
        debug: Enable debug output

    Returns:
        Tuple of (top_left, top_right, bottom_left, bottom_right) positions,
        where each position is (x, y), or None if all four corners not found
    """
    h, w = image.shape[:2]

    # First, try region-based search (more targeted)
    corners = find_corners_by_region(image, top_number, bottom_number, number_range, cm_interval)
    if corners is not None:
        if debug:
            print("  Found corners using region-based search")
        return corners

    # Fallback: Try different OCR configurations on full image
    # Expanded scale range for better detection at different distances
    scale_factors = [3.0, 3.5, 4.0, 2.5, 4.5, 5.0, 2.0, 1.5]
    preprocess_methods = ['clahe', 'adaptive', 'otsu', 'morph']
    psm_modes = [11, 6, 7, 12]

    for scale_factor in scale_factors:
        for preprocess_method in preprocess_methods:
            for psm_mode in psm_modes:
                # Detect numbers with this configuration using shared utility
                numbers_with_positions = detect_numbers_with_config(
                    image, scale_factor, preprocess_method, psm_mode, number_range, cm_interval
                )

                # Try to identify the four corner numbers
                corners = identify_corner_numbers(
                    numbers_with_positions, top_number, bottom_number, w, h
                )

                if corners is not None:
                    if debug:
                        print(f"  Found corners with: scale={scale_factor}, "
                              f"preprocess={preprocess_method}, psm={psm_mode}")
                    return corners

    # If we couldn't find all four corners
    return None


def find_corners_by_region(
    image: np.ndarray,
    top_number: int,
    bottom_number: int,
    number_range: Tuple[int, int],
    cm_interval: int
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Search for corner numbers by focusing on specific image regions.

    This method divides the image into quadrants and searches each region
    with targeted OCR configurations, which is more robust when the backdrop
    is partially occluded.

    Args:
        image: Input image
        top_number: Number at top corners (e.g., 200)
        bottom_number: Number at bottom corners (e.g., 10)
        number_range: Valid number range as (min, max) tuple
        cm_interval: Valid number interval

    Returns:
        Tuple of corner positions or None
    """
    h, w = image.shape[:2]

    # Define search regions for each corner (with generous overlap)
    # Format: (x_start, y_start, x_end, y_end)
    regions = {
        'top_left': (0, 0, w // 3, h // 3),
        'top_right': (2 * w // 3, 0, w, h // 3),
        'bottom_left': (0, 2 * h // 3, w // 3, h),
        'bottom_right': (2 * w // 3, 2 * h // 3, w, h)
    }

    # Try focused OCR on each region
    scale_factors = [3.0, 3.5, 4.0, 4.5]
    preprocess_methods = ['clahe', 'adaptive']
    psm_modes = [6, 7, 11]

    corner_candidates = {
        'top_left': [],
        'top_right': [],
        'bottom_left': [],
        'bottom_right': []
    }

    for region_name, (x1, y1, x2, y2) in regions.items():
        region_img = image[y1:y2, x1:x2]
        expected_num = top_number if 'top' in region_name else bottom_number

        for scale_factor in scale_factors:
            for preprocess_method in preprocess_methods:
                for psm_mode in psm_modes:
                    numbers = detect_numbers_with_config(
                        region_img, scale_factor, preprocess_method,
                        psm_mode, number_range, cm_interval
                    )

                    # Find matching numbers in this region
                    for num, x, y in numbers:
                        if num == expected_num:
                            # Convert to global coordinates
                            global_x = x1 + x
                            global_y = y1 + y
                            corner_candidates[region_name].append((global_x, global_y))

    # Check if we found at least one candidate for each corner
    if not all(len(candidates) > 0 for candidates in corner_candidates.values()):
        return None

    # Select the best candidate for each corner
    # For top corners: prefer leftmost/rightmost respectively
    # For bottom corners: prefer leftmost/rightmost respectively
    top_left = min(corner_candidates['top_left'], key=lambda p: p[0])
    top_right = max(corner_candidates['top_right'], key=lambda p: p[0])
    bottom_left = min(corner_candidates['bottom_left'], key=lambda p: p[0])
    bottom_right = max(corner_candidates['bottom_right'], key=lambda p: p[0])

    # Validate the corners make geometric sense
    if not validate_corner_geometry(top_left, top_right, bottom_left, bottom_right, w, h):
        return None

    return (top_left, top_right, bottom_left, bottom_right)


def validate_corner_geometry(
    top_left: Tuple[int, int],
    top_right: Tuple[int, int],
    bottom_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    image_width: int,
    image_height: int
) -> bool:
    """
    Validate that corner positions make geometric sense.
    Uses relaxed percentage-based thresholds instead of strict midpoint division.

    Args:
        top_left, top_right, bottom_left, bottom_right: Corner positions (x, y)
        image_width, image_height: Image dimensions

    Returns:
        True if corners are geometrically valid
    """
    # Use 40/60 split instead of strict 50/50 for more flexibility
    y_threshold_top = image_height * 0.4
    y_threshold_bottom = image_height * 0.6
    x_threshold_left = image_width * 0.4
    x_threshold_right = image_width * 0.6

    # Top corners should be in upper 40% of image
    if not (top_left[1] < y_threshold_top and top_right[1] < y_threshold_top):
        return False

    # Bottom corners should be in lower 60% of image
    if not (bottom_left[1] > y_threshold_bottom and bottom_right[1] > y_threshold_bottom):
        return False

    # Left corners should be in left 40% of image
    if not (top_left[0] < x_threshold_left and bottom_left[0] < x_threshold_left):
        return False

    # Right corners should be in right 60% of image
    if not (top_right[0] > x_threshold_right and bottom_right[0] > x_threshold_right):
        return False

    # Top-right should be to the right of top-left
    if top_right[0] <= top_left[0]:
        return False

    # Bottom-right should be to the right of bottom-left
    if bottom_right[0] <= bottom_left[0]:
        return False

    return True


def identify_corner_numbers(
    numbers_with_positions: List[Tuple[int, int, int]],
    top_number: int,
    bottom_number: int,
    image_width: int,
    image_height: int
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Identify the four corner numbers from detected numbers.

    Args:
        numbers_with_positions: List of (number, x, y) tuples
        top_number: Expected number at top corners (e.g., 200)
        bottom_number: Expected number at bottom corners (e.g., 10)
        image_width: Width of image
        image_height: Height of image

    Returns:
        (top_left, top_right, bottom_left, bottom_right) positions or None
    """
    # Find all instances of top and bottom numbers
    top_numbers = [(x, y) for num, x, y in numbers_with_positions if num == top_number]
    bottom_numbers = [(x, y) for num, x, y in numbers_with_positions if num == bottom_number]

    # Need at least 2 of each
    if len(top_numbers) < 2 or len(bottom_numbers) < 2:
        return None

    # Find the two top numbers that are furthest apart (left and right)
    top_numbers_sorted = sorted(top_numbers, key=lambda pos: pos[0])
    top_left = top_numbers_sorted[0]
    top_right = top_numbers_sorted[-1]

    # Find the two bottom numbers that are furthest apart (left and right)
    bottom_numbers_sorted = sorted(bottom_numbers, key=lambda pos: pos[0])
    bottom_left = bottom_numbers_sorted[0]
    bottom_right = bottom_numbers_sorted[-1]

    # Use the more flexible validation method
    if not validate_corner_geometry(
        top_left, top_right, bottom_left, bottom_right, image_width, image_height
    ):
        return None

    return (top_left, top_right, bottom_left, bottom_right)


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

    # Edge detection with bilateral filtering to reduce noise while preserving edges
    gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, canny_low, canny_high, apertureSize=3)

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

    # Extract y-coordinates of horizontal lines with length validation
    y_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 3 or angle > 177:  # Horizontal lines only
            # Double-check line length (longer lines are more reliable)
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if line_length > scaled_image.shape[1] * min_line_length_ratio:
                y_coords.append(int((y1 + y2) / 2))

    # Scale back to original coordinates if image was upscaled
    if line_scale_factor > 1.0:
        y_coords = [int(y / line_scale_factor) for y in y_coords]

    # Remove duplicates within 10 pixels (increased from 5 to avoid detecting thin/thick line pairs)
    y_coords = sorted(set(y_coords))
    filtered_coords = []
    for i, y in enumerate(y_coords):
        if i == 0 or y - filtered_coords[-1] > 10:
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


def detect_lines_morphological(
    image: np.ndarray,
    expected_lines: int = 20,
    min_thickness: float = 3.0
) -> List[int]:
    """
    Detect bold horizontal lines using Hough Transform on corrected image.

    On a perspective-corrected image, we can use simpler, more reliable detection.
    This method specifically targets ONLY bold lines by using strict thickness validation.

    Strategy:
    - Cast a wide net with lenient Hough parameters to find all lines
    - Measure thickness of each line accurately
    - Keep ONLY the thickest lines (the bold 10cm demarcations)

    Args:
        image: Perspective-corrected backdrop image
        expected_lines: Expected number of bold demarcation lines
        min_thickness: Minimum line thickness in pixels (default 3.0 for bold lines)

    Returns:
        List of y-coordinates for detected bold lines
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try multiple edge detection thresholds to find ALL lines
    detected_lines = []

    # Multiple Canny thresholds to cast a wide net
    canny_configs = [
        (30, 100),   # Original - good for clear lines
        (50, 150),   # Higher threshold - better for bold lines
        (40, 120),   # Medium threshold
        (20, 80),    # Lower threshold - catch fainter lines
    ]

    for canny_low, canny_high in canny_configs:
        # Bilateral filter for noise reduction
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Edge detection
        edges = cv2.Canny(filtered, canny_low, canny_high)

        # Use Hough Transform with lenient parameters to find ALL lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=int(w * 0.12),  # Lenient to find all lines
            minLineLength=int(w * 0.30),  # Lenient to find all lines
            maxLineGap=60  # Allow gaps
        )

        if lines is not None:
            # Extract y-coordinates of horizontal lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is horizontal (angle < 3 degrees)
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 3 or angle > 177:
                    # Validate line length
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if line_length >= w * 0.35:
                        y_center = int((y1 + y2) / 2)
                        detected_lines.append(y_center)

    # Sort and remove duplicates within 15 pixels
    y_coords = sorted(set(detected_lines))
    filtered_coords = []
    for y in y_coords:
        if not filtered_coords or y - filtered_coords[-1] > 15:
            filtered_coords.append(y)

    # CRITICAL: Measure thickness of ALL detected lines
    all_coords_with_thickness = []
    for y in filtered_coords:
        thickness = measure_line_thickness(gray, y, search_range=15)
        all_coords_with_thickness.append((y, thickness))

    # Sort by thickness descending
    all_coords_with_thickness.sort(key=lambda x: x[1], reverse=True)

    # Strategy: Keep only the thickest lines (bold demarcations)
    # First, apply minimum thickness filter
    thick_lines = [(y, t) for y, t in all_coords_with_thickness if t >= min_thickness]

    # If we have more than expected, keep only the thickest ones
    if len(thick_lines) >= expected_lines:
        # Take exactly the expected number of thickest lines
        validated_coords = [y for y, _ in thick_lines[:expected_lines]]
    else:
        # If we don't have enough thick lines, we may need to relax threshold slightly
        # But still prioritize thickness - take the thickest lines available
        validated_coords = [y for y, _ in all_coords_with_thickness[:expected_lines]]

    # Sort by position (y-coordinate) for proper ordering
    validated_coords.sort()

    return validated_coords


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
    number_range: Tuple[int, int] = (10, 200),
    use_preprocessing: str = 'otsu',
    cm_interval: int = 10
) -> List[Tuple[int, int]]:
    """
    Detect measurement numbers using OCR with advanced preprocessing.

    Args:
        image: Input image as numpy array (BGR format)
        scale_factor: Upscaling factor for small text (1.0-5.0)
        region_width: Width of left region to scan (0.15-0.30)
        psm_mode: Tesseract PSM mode (6, 11, or 12)
        number_range: Valid number range as (min, max) tuple
        use_preprocessing: Preprocessing method ('otsu', 'adaptive', 'morph', 'clahe')
        cm_interval: Valid number interval (default: 10 for 10, 20, 30, ...)

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

    # Convert to grayscale
    gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing based on method
    if use_preprocessing == 'adaptive':
        # Adaptive thresholding - best for uneven lighting
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
    elif use_preprocessing == 'morph':
        # Morphological operations + Otsu
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        _, binary = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif use_preprocessing == 'clahe':
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # otsu (default)
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
                    # Range check and interval validation based on backdrop config
                    if min_num <= number <= max_num and number % cm_interval == 0 and number not in seen_numbers:
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


def detect_numbers_ocr(
    image: np.ndarray,
    number_range: Tuple[int, int] = (10, 200),
    cm_interval: int = 10,
    expected_numbers: int = 20,
    region_width: float = 0.30
) -> List[Tuple[int, int]]:
    """
    Detect numbers using OCR on both left and right regions.

    Args:
        image: Perspective-corrected backdrop image
        number_range: Valid number range as (min, max) tuple
        cm_interval: Valid number interval (default: 10)
        expected_numbers: Expected number of numbers to detect
        region_width: Width of each region to scan (left and right)

    Returns:
        List of (number, y_position) tuples
    """
    h, w = image.shape[:2]
    min_num, max_num = number_range

    # Try both left and right regions (numbers appear on both sides)
    all_numbers = {}  # Use dict to deduplicate by number value

    for region_name, region in [
        ("left", image[:, :int(w * region_width)]),
        ("right", image[:, -int(w * region_width):])
    ]:
        # Upscale for better OCR (4x for better results on corrected image)
        scale_factor = 4.0
        region_h, region_w = region.shape[:2]
        scaled = cv2.resize(
            region,
            (int(region_w * scale_factor), int(region_h * scale_factor)),
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to grayscale
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        # Try multiple preprocessing methods
        for preprocess_method in ['clahe', 'adaptive', 'otsu']:
            if preprocess_method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif preprocess_method == 'adaptive':
                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 21, 10
                )
            else:  # otsu
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Invert if needed
            if np.mean(binary) < 127:
                binary = cv2.bitwise_not(binary)

            # Try multiple PSM modes
            for psm_mode in [11, 6, 7]:
                custom_config = f'--psm {psm_mode} --oem 3 -c tessedit_char_whitelist=0123456789'

                try:
                    data = pytesseract.image_to_data(
                        binary,
                        config=custom_config,
                        output_type=pytesseract.Output.DICT
                    )

                    for i in range(len(data['text'])):
                        text = data['text'][i].strip()
                        conf = float(data['conf'][i])

                        if conf >= 0 and text.isdigit():
                            number = int(text)

                            # Validate: must be in range AND a multiple of cm_interval
                            if min_num <= number <= max_num and number % cm_interval == 0:
                                y = data['top'][i] + data['height'][i] // 2
                                y_original = int(y / scale_factor)

                                # Keep highest confidence detection for each number
                                if number not in all_numbers or conf > all_numbers[number][1]:
                                    all_numbers[number] = (y_original, conf)
                except Exception:
                    continue

            # Early exit if we found enough numbers
            if len(all_numbers) >= expected_numbers * 0.9:
                break

        if len(all_numbers) >= expected_numbers * 0.9:
            break

    # Convert back to list format
    numbers_with_pos = [(num, y_pos) for num, (y_pos, conf) in all_numbers.items()]
    numbers_with_pos.sort(key=lambda x: x[1])
    return numbers_with_pos


def detect_numbers_with_config(
    image: np.ndarray,
    scale_factor: float,
    preprocess_method: str,
    psm_mode: int,
    number_range: Tuple[int, int] = (10, 200),
    cm_interval: int = 10
) -> List[Tuple[int, int, int]]:
    """
    Detect numbers with a specific OCR configuration.

    This utility function allows testing different OCR settings to find
    the optimal configuration for a given image.

    Args:
        image: Input image (BGR format)
        scale_factor: Upscaling factor (e.g., 2.0, 2.5, 3.0, 4.0)
        preprocess_method: Preprocessing method ('clahe', 'adaptive', 'otsu', 'morph')
        psm_mode: Tesseract PSM mode (6, 7, 11, 12)
        number_range: Valid number range as (min, max) tuple
        cm_interval: Valid number interval (default: 10 for 10, 20, 30, ...)

    Returns:
        List of (number, x_position, y_position) tuples
    """
    h, w = image.shape[:2]
    min_num, max_num = number_range

    # Upscale image
    scaled = cv2.resize(
        image,
        (int(w * scale_factor), int(h * scale_factor)),
        interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    # Apply preprocessing
    if preprocess_method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif preprocess_method == 'adaptive':
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
    elif preprocess_method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # morph
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Invert if needed
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Run OCR
    custom_config = f'--psm {psm_mode} --oem 3 -c tessedit_char_whitelist=0123456789'

    try:
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)

        # Use dict to deduplicate - keep highest confidence for each valid number
        seen_numbers = {}
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])

            if conf >= 0 and text.isdigit():
                number = int(text)

                # Validate: must be in range AND a multiple of cm_interval
                if min_num <= number <= max_num and number % cm_interval == 0:
                    # Get center position (scale back to original)
                    x = (data['left'][i] + data['width'][i] // 2) / scale_factor
                    y = (data['top'][i] + data['height'][i] // 2) / scale_factor

                    # Keep highest confidence detection for each number
                    if number not in seen_numbers or conf > seen_numbers[number][2]:
                        seen_numbers[number] = (int(x), int(y), conf)

        # Convert to list format (number, x, y)
        numbers_with_positions = [(num, x, y) for num, (x, y, _) in seen_numbers.items()]
        return numbers_with_positions
    except Exception:
        return []


# # ============================================================================
# # CALIBRATION CALCULATION UTILITIES
# # ============================================================================

# def calculate_calibration(
#     lines: List[int],
#     numbers: List[Tuple[int, int]],
#     image_height: int
# ) -> Dict:
#     """
#     Calculate calibration data from detected lines and numbers.

#     Args:
#         lines: List of y-coordinates of detected lines
#         numbers: List of (number, y_position) tuples
#         image_height: Height of the image in pixels

#     Returns:
#         Dictionary containing calibration information:
#         {
#             "cm_per_normalized_unit": float,
#             "pixels_per_cm": float,
#             "detected_lines": int,
#             "detected_numbers": int,
#             "line_spacing_pixels": float,
#             "line_spacing_cv": float,
#             "cm_interval": int,
#             "confidence": str
#         }
#     """
#     if len(lines) >= 2 and len(numbers) >= 2:
#         # Calculate average spacing between consecutive lines
#         line_spacings = []
#         for i in range(len(lines) - 1):
#             spacing = lines[i + 1] - lines[i]
#             line_spacings.append(spacing)

#         avg_line_spacing_pixels = np.mean(line_spacings)
#         spacing_cv = np.std(line_spacings) / avg_line_spacing_pixels if avg_line_spacing_pixels > 0 else 0

#         # Determine cm interval between lines using OCR numbers
#         number_diffs = []
#         for i in range(len(numbers) - 1):
#             num_diff = numbers[i + 1][0] - numbers[i][0]
#             number_diffs.append(num_diff)

#         # Most common difference should be the cm interval
#         if number_diffs:
#             cm_interval = abs(int(np.median(number_diffs)))
#         else:
#             # Default assumption: 10cm intervals
#             cm_interval = 10

#         # Calculate pixels per cm
#         pixels_per_cm = avg_line_spacing_pixels / cm_interval

#         # Calculate normalized units (0-1 range) per cm
#         # Normalized unit = 1 / image_height
#         normalized_unit_per_pixel = 1.0 / image_height
#         cm_per_normalized_unit = 1.0 / (pixels_per_cm * normalized_unit_per_pixel)

#         # Determine confidence
#         if len(lines) >= 10 and len(numbers) >= 5:
#             confidence = "high"
#         elif len(lines) >= 5 and len(numbers) >= 3:
#             confidence = "medium"
#         else:
#             confidence = "low"

#         calibration_data = {
#             "cm_per_normalized_unit": float(cm_per_normalized_unit),
#             "pixels_per_cm": float(pixels_per_cm),
#             "detected_lines": len(lines),
#             "detected_numbers": len(numbers),
#             "line_spacing_pixels": float(avg_line_spacing_pixels),
#             "line_spacing_cv": float(spacing_cv),
#             "cm_interval": cm_interval,
#             "confidence": confidence
#         }
#     else:
#         # Not enough data
#         calibration_data = {
#             "cm_per_normalized_unit": None,
#             "pixels_per_cm": None,
#             "detected_lines": len(lines),
#             "detected_numbers": len(numbers),
#             "line_spacing_pixels": None,
#             "line_spacing_cv": None,
#             "cm_interval": None,
#             "confidence": "failed",
#             "error": "Not enough lines or numbers detected"
#         }

#     return calibration_data


# def calculate_score(
#     lines: List[int],
#     numbers: List[Tuple[int, int]],
#     expected_lines: int,
#     expected_numbers: int
# ) -> float:
#     """
#     Calculate a quality score for detection results.

#     Scoring breakdown:
#     - Line count accuracy: 50% weight
#     - Line spacing consistency: 30% weight
#     - Number count accuracy: 20% weight

#     Args:
#         lines: List of y-coordinates of detected lines
#         numbers: List of (number, y_position) tuples
#         expected_lines: Expected number of lines
#         expected_numbers: Expected number of numbers

#     Returns:
#         Score between 0.0 and 1.0
#     """
#     scores = []

#     # Line count accuracy (50% weight)
#     if expected_lines > 0:
#         line_error = abs(len(lines) - expected_lines) / expected_lines
#         line_accuracy = max(0, 1 - line_error)
#         scores.append(line_accuracy * 0.50)
#     else:
#         scores.append(0)

#     # Line spacing consistency (30% weight)
#     if len(lines) >= 2:
#         spacings = np.diff(sorted(lines))
#         spacing_cv = np.std(spacings) / np.mean(spacings)
#         # Normalize: CV of 0.3 or higher = score of 0
#         spacing_score = max(0, 1 - spacing_cv * 3.33)
#         scores.append(spacing_score * 0.30)
#     else:
#         scores.append(0)

#     # Number count accuracy (20% weight)
#     if expected_numbers > 0:
#         number_error = abs(len(numbers) - expected_numbers) / expected_numbers
#         number_accuracy = max(0, 1 - number_error)
#         scores.append(number_accuracy * 0.20)
#     else:
#         scores.append(0)

#     return sum(scores)


# ============================================================================
# NUMBER-LINE MATCHING UTILITIES
# ============================================================================

def match_numbers_to_lines(
    numbers: List[Tuple[int, int]],
    lines: List[int],
    cm_interval: int = 10,
    max_distance: int = 15,
    min_sequence_length: int = 3,
    debug: bool = False
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    Match detected numbers with nearby lines and find the longest consecutive sequence.

    This strategy uses numbers as ground truth anchors - lines that align with numbers
    are definitely the bold 10cm demarcation lines, not thin lines.

    IMPORTANT: max_distance should be strict (10-15px) to ensure only numbers with
    truly intersecting bold lines are matched, not numbers near thin lines.

    Args:
        numbers: List of (number, y_position) tuples
        lines: List of line y-positions
        cm_interval: Interval between consecutive numbers (default 10cm)
        max_distance: Maximum vertical distance (in pixels) for a line to match a number
                     (default 15px for strict matching)
        min_sequence_length: Minimum length of consecutive sequence to keep (default 3)
        debug: Enable debug output

    Returns:
        Tuple of (all_matched_pairs, longest_consecutive_sequence)
        where each item is a list of (number, number_y, line_y) tuples
    """
    if not numbers or not lines:
        return [], []

    # Step 1: Match each number with the closest line (if within max_distance)
    matched_pairs = []
    unmatched_numbers = []

    for num, num_y in numbers:
        # Find the closest line to this number
        closest_line = min(lines, key=lambda line_y: abs(line_y - num_y))
        distance = abs(closest_line - num_y)

        if distance <= max_distance:
            matched_pairs.append((num, num_y, closest_line))
        else:
            unmatched_numbers.append((num, distance))

    if debug:
        if unmatched_numbers:
            print(f"  Note: {len(unmatched_numbers)} numbers had no nearby line (>{max_distance}px away)")
            for num, dist in unmatched_numbers[:5]:  # Show first 5
                print(f"        Number {num}: closest line {dist:.1f}px away")
        # Also show matched pairs for debugging
        if matched_pairs:
            print(f"  Matched pairs (number: distance to line):")
            for num, num_y, line_y in matched_pairs[:10]:  # Show first 10
                dist = abs(line_y - num_y)
                print(f"        {num}cm: {dist:.1f}px")

    if not matched_pairs:
        return [], []

    # Sort by number value
    matched_pairs.sort(key=lambda x: x[0])

    # Step 2: Find longest consecutive sequence
    # Numbers should be spaced by cm_interval (e.g., 10, 20, 30...)
    sequences = []
    current_sequence = [matched_pairs[0]]

    for i in range(1, len(matched_pairs)):
        prev_num, _, _ = current_sequence[-1]
        curr_num, curr_num_y, curr_line_y = matched_pairs[i]

        # Check if this number is consecutive (differs by cm_interval)
        if curr_num == prev_num + cm_interval:
            current_sequence.append((curr_num, curr_num_y, curr_line_y))
        else:
            # Sequence broken, start a new one
            if len(current_sequence) >= min_sequence_length:
                sequences.append(current_sequence)
            current_sequence = [(curr_num, curr_num_y, curr_line_y)]

    # Don't forget the last sequence
    if len(current_sequence) >= min_sequence_length:
        sequences.append(current_sequence)

    # Step 3: Select the longest sequence
    if sequences:
        longest_sequence = max(sequences, key=len)
    else:
        # Fallback: if no sequence of min_sequence_length+, use all matched pairs
        longest_sequence = matched_pairs if len(matched_pairs) >= 2 else []

    return matched_pairs, longest_sequence
