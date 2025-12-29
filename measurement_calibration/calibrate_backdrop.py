"""
Corner-Based Backdrop Calibration

This script uses a corner-detection approach optimized for fixed-design backdrops:

1. CORNER NUMBER DETECTION
   - Adaptively searches for four corner numbers (200, 200, 10, 10)
   - Tries multiple OCR configurations (scale, preprocessing, PSM modes)
   - Defines backdrop boundary based on detected corner positions

2. ROI-BASED DETECTION ON ORIGINAL IMAGE
   - Extracts region of interest from ORIGINAL image (no warping)
   - Detects lines and numbers within ROI
   - Converts ROI coordinates back to original image coordinates

3. CALIBRATION RELATIVE TO FULL IMAGE
   - Calculates cm_per_normalized_unit using ORIGINAL image height
   - Ensures calibration is consistent across all images regardless of backdrop size

This approach handles:
- Background clutter (only processes detected backdrop region)
- Varying distances (adaptive OCR finds corners at any scale)
- Different lighting conditions (multiple preprocessing methods)
- Angled views (corner detection works from any angle)
"""

import cv2
import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Optional
import pytesseract
from pathlib import Path

# Import shared calibration utilities
from calibration_utils import detect_numbers_with_config


class PerspectiveBackdropCalibrator:
    """
    Calibrator using corner-based backdrop detection.

    This approach eliminates the need for parameter optimization by:
    1. Detecting backdrop corners using adaptive OCR (searches for corner numbers)
    2. Extracting ROI from original image (no warping/perspective correction)
    3. Detecting lines and numbers within ROI
    4. Calculating calibration relative to full original image dimensions
    """

    def __init__(
        self,
        target_config: Dict,
        tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        debug: bool = False,
        visualization_dir: Optional[str] = None
    ):
        """
        Initialize calibrator.

        Args:
            target_config: Expected detection configuration
            tesseract_cmd: Path to Tesseract executable
            debug: Enable debug output
            visualization_dir: Directory to save visualization images
        """
        self.expected_lines = target_config['expected_lines']
        self.expected_numbers = target_config['expected_numbers']
        self.number_range = tuple(target_config['number_range'])
        self.cm_interval = target_config.get('cm_interval', 10)

        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.debug = debug
        self.visualization_dir = visualization_dir

        if visualization_dir:
            Path(visualization_dir).mkdir(parents=True, exist_ok=True)

    def detect_backdrop_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the backdrop by finding four corner numbers: 200, 200, 10, 10.

        Key insight: The backdrop corners are defined by specific numbers:
        - Top corners: two "200"s (left and right)
        - Bottom corners: two "10"s (left and right)

        This method tries different OCR configurations until all four corners are found.

        Args:
            image: Input image

        Returns:
            4-point array of backdrop corners (top-left, top-right, bottom-right, bottom-left)
            or None if backdrop not found
        """
        h, w = image.shape[:2]
        top_number = self.number_range[1]  # 200
        bottom_number = self.number_range[0]  # 10

        # Try to find all four corner numbers with adaptive OCR
        corner_positions = self._find_corner_numbers_adaptive(image, top_number, bottom_number)

        if corner_positions is None:
            if self.debug:
                print("  Warning: Could not find all four corner numbers")
            return None

        top_left, top_right, bottom_left, bottom_right = corner_positions

        if self.debug:
            print(f"  Found all 4 corner numbers:")
            print(f"    Top-left {top_number}: {top_left}")
            print(f"    Top-right {top_number}: {top_right}")
            print(f"    Bottom-left {bottom_number}: {bottom_left}")
            print(f"    Bottom-right {bottom_number}: {bottom_right}")

        # Add generous margins to include full backdrop beyond the corner numbers
        # Corner positions are at number centers, so we need extra space for:
        # 1. The width/height of the numbers themselves (~30-50px)
        # 2. Any backdrop border beyond the numbers
        # 3. Some padding for safety
        margin_x = 80  # Increased from 40 to include full number width + backdrop edge
        margin_y = 60  # Increased from 30 to include full number height + backdrop edge

        x_left = max(0, top_left[0] - margin_x)
        x_right = min(w - 1, top_right[0] + margin_x)
        y_top = max(0, min(top_left[1], top_right[1]) - margin_y)
        y_bottom = min(h - 1, max(bottom_left[1], bottom_right[1]) + margin_y)

        # Create quadrilateral corners
        corners = np.array([
            [x_left, y_top],
            [x_right, y_top],
            [x_right, y_bottom],
            [x_left, y_bottom]
        ], dtype=np.float32)

        return corners

    def _find_corner_numbers_adaptive(
        self,
        image: np.ndarray,
        top_number: int,
        bottom_number: int
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """
        Adaptively search for four corner numbers using multiple OCR configurations.

        This method tries different combinations of:
        - Scale factors (1.5x to 5.0x) - expanded range for better coverage
        - Preprocessing methods (clahe, adaptive, otsu, morph)
        - PSM modes (11, 6, 7, 12)
        - Region-based search focusing on likely corner areas

        Args:
            image: Input image
            top_number: Number at top corners (e.g., 200)
            bottom_number: Number at bottom corners (e.g., 10)

        Returns:
            Tuple of (top_left, top_right, bottom_left, bottom_right) positions,
            where each position is (x, y), or None if all four corners not found
        """
        h, w = image.shape[:2]

        # First, try region-based search (more targeted)
        corners = self._find_corners_by_region(image, top_number, bottom_number)
        if corners is not None:
            if self.debug:
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
                        image, scale_factor, preprocess_method, psm_mode, self.number_range, self.cm_interval
                    )

                    # Try to identify the four corner numbers
                    corners = self._identify_corner_numbers(
                        numbers_with_positions, top_number, bottom_number, w, h
                    )

                    if corners is not None:
                        if self.debug:
                            print(f"  Found corners with: scale={scale_factor}, "
                                  f"preprocess={preprocess_method}, psm={psm_mode}")
                        return corners

        # If we couldn't find all four corners
        return None

    def _find_corners_by_region(
        self,
        image: np.ndarray,
        top_number: int,
        bottom_number: int
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
                            psm_mode, self.number_range, self.cm_interval
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
        if not self._validate_corner_geometry(top_left, top_right, bottom_left, bottom_right, w, h):
            return None

        return (top_left, top_right, bottom_left, bottom_right)

    def _validate_corner_geometry(
        self,
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

    def _identify_corner_numbers(
        self,
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
        if not self._validate_corner_geometry(
            top_left, top_right, bottom_left, bottom_right, image_width, image_height
        ):
            return None

        return (top_left, top_right, bottom_left, bottom_right)


    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners as: top-left, top-right, bottom-right, bottom-left.

        Args:
            corners: Array of 4 corner points

        Returns:
            Ordered corner array
        """
        # Calculate center
        center = corners.mean(axis=0)

        # Separate into top/bottom based on y-coordinate
        top = corners[corners[:, 1] < center[1]]
        bottom = corners[corners[:, 1] >= center[1]]

        # Order top: left to right
        top = top[top[:, 0].argsort()]
        # Order bottom: left to right
        bottom = bottom[bottom[:, 0].argsort()]

        # Return ordered: TL, TR, BR, BL
        return np.array([
            top[0],      # top-left
            top[1],      # top-right
            bottom[1],   # bottom-right
            bottom[0]    # bottom-left
        ], dtype=np.float32)

    def apply_perspective_transform(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        target_width: int = 800,
        target_height: int = 2000
    ) -> np.ndarray:
        """
        Apply perspective transform to get straight-on view of backdrop.

        Args:
            image: Input image
            corners: 4-point array of backdrop corners
            target_width: Width of output image
            target_height: Height of output image

        Returns:
            Perspective-corrected image
        """
        # Define destination points (straight-on rectangle)
        dst = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(corners, dst)

        # Apply transform
        warped = cv2.warpPerspective(image, matrix, (target_width, target_height))

        return warped

    def detect_lines_morphological(
        self,
        image: np.ndarray,
        min_thickness: float = 2.5
    ) -> List[int]:
        """
        Detect bold horizontal lines using Hough Transform on corrected image.

        On a perspective-corrected image, we can use simpler, more reliable detection.
        This method specifically targets bold lines by using thickness validation.

        Args:
            image: Perspective-corrected backdrop image
            min_thickness: Minimum line thickness in pixels (default 2.5 for bold lines)

        Returns:
            List of y-coordinates for detected bold lines
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try multiple edge detection thresholds to find bold lines
        detected_lines = []

        # Expanded Canny thresholds to detect lines at various intensities and conditions
        canny_configs = [
            (30, 100),   # Original - good for clear lines
            (50, 150),   # Higher threshold - better for bold lines
            (40, 120),   # Medium threshold
            (20, 80),    # Lower threshold - catch fainter lines
            (60, 180),   # Very high threshold - only strongest edges
        ]

        for canny_low, canny_high in canny_configs:
            # Bilateral filter for noise reduction
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)

            # Edge detection
            edges = cv2.Canny(filtered, canny_low, canny_high)

            # Use Hough Transform to detect long horizontal lines
            # More lenient parameters to ensure we find all 20 lines
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=int(w * 0.12),  # More lenient: 12% (was 15%)
                minLineLength=int(w * 0.30),  # More lenient: 30% (was 35%)
                maxLineGap=60  # Allow even larger gaps: 60px (was 50px)
            )

            if lines is not None:
                # Extract y-coordinates of horizontal lines
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Check if line is horizontal (angle < 3 degrees)
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if angle < 3 or angle > 177:
                        # Validate line length - slightly more lenient
                        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if line_length >= w * 0.35:  # 35% (was 40%)
                            y_center = int((y1 + y2) / 2)
                            detected_lines.append(y_center)

        # Sort and remove duplicates within 15 pixels
        y_coords = sorted(set(detected_lines))
        filtered_coords = []
        for y in y_coords:
            if not filtered_coords or y - filtered_coords[-1] > 15:
                filtered_coords.append(y)

        # Validate line thickness - use more relaxed threshold (2.5 instead of 3.0)
        validated_coords = []
        thicknesses = []
        for y in filtered_coords:
            thickness = self._measure_line_thickness(gray, y)
            if thickness >= min_thickness:
                validated_coords.append(y)
                thicknesses.append(thickness)

        # Additional filtering: if we have many lines, keep only the thickest ones
        if len(validated_coords) > self.expected_lines * 1.2:
            # Sort by thickness descending and keep top expected_lines
            coords_with_thickness = list(zip(validated_coords, thicknesses))
            coords_with_thickness.sort(key=lambda x: x[1], reverse=True)
            # Keep the thickest lines up to expected count
            validated_coords = [y for y, _ in coords_with_thickness[:self.expected_lines]]
            validated_coords.sort()  # Sort by position

        return validated_coords

    def _measure_line_thickness(self, gray_image: np.ndarray, y: int, search_range: int = 15) -> float:
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
        threshold = (np.min(vertical_profile) + np.mean(vertical_profile)) / 2

        # Count pixels below threshold (dark pixels = line)
        thickness = np.sum(vertical_profile < threshold)

        return float(thickness)

    def detect_numbers_ocr(
        self,
        image: np.ndarray,
        region_width: float = 0.30
    ) -> List[Tuple[int, int]]:
        """
        Detect numbers using OCR on both left and right regions.

        Args:
            image: Perspective-corrected backdrop image
            region_width: Width of each region to scan (left and right)

        Returns:
            List of (number, y_position) tuples
        """
        h, w = image.shape[:2]
        min_num, max_num = self.number_range

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
                                if min_num <= number <= max_num and number % self.cm_interval == 0:
                                    y = data['top'][i] + data['height'][i] // 2
                                    y_original = int(y / scale_factor)

                                    # Keep highest confidence detection for each number
                                    if number not in all_numbers or conf > all_numbers[number][1]:
                                        all_numbers[number] = (y_original, conf)
                    except Exception:
                        continue

                # Early exit if we found enough numbers
                if len(all_numbers) >= self.expected_numbers * 0.9:
                    break

            if len(all_numbers) >= self.expected_numbers * 0.9:
                break

        # Convert back to list format
        numbers_with_pos = [(num, y_pos) for num, (y_pos, conf) in all_numbers.items()]
        numbers_with_pos.sort(key=lambda x: x[1])
        return numbers_with_pos

    def calculate_calibration(
        self,
        lines: List[int],
        numbers: List[Tuple[int, int]],
        image_height: int
    ) -> Dict:
        """Calculate calibration data from detected lines and numbers."""
        if len(lines) >= 2 and len(numbers) >= 2:
            # Calculate spacing
            line_spacings = np.diff(sorted(lines))
            avg_spacing = np.mean(line_spacings)
            spacing_cv = np.std(line_spacings) / avg_spacing

            # Determine cm interval
            number_diffs = [numbers[i+1][0] - numbers[i][0] for i in range(len(numbers)-1)]
            cm_interval = abs(int(np.median(number_diffs))) if number_diffs else 10

            # Calculate pixels per cm
            pixels_per_cm = avg_spacing / cm_interval

            # Calculate normalized units per cm
            normalized_unit_per_pixel = 1.0 / image_height
            cm_per_normalized_unit = 1.0 / (pixels_per_cm * normalized_unit_per_pixel)

            # Determine confidence
            if len(lines) >= 15 and len(numbers) >= 15:
                confidence = "high"
            elif len(lines) >= 10 and len(numbers) >= 10:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "cm_per_normalized_unit": float(cm_per_normalized_unit),
                "pixels_per_cm": float(pixels_per_cm),
                "detected_lines": len(lines),
                "detected_numbers": len(numbers),
                "line_spacing_pixels": float(avg_spacing),
                "line_spacing_cv": float(spacing_cv),
                "cm_interval": cm_interval,
                "confidence": confidence
            }
        else:
            return {
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

    def calibrate(self, image: np.ndarray) -> Dict:
        """
        Calibrate backdrop using perspective-aware detection.

        Args:
            image: Input image

        Returns:
            Calibration result dictionary
        """
        # Store original image dimensions - calibration MUST be relative to full image
        h_original, w_original = image.shape[:2]

        if self.debug:
            print("="*70)
            print("PERSPECTIVE-AWARE CALIBRATION")
            print("="*70)
            print(f"Target: {self.expected_lines} lines, {self.expected_numbers} numbers")
            print(f"Original image: {w_original}x{h_original}")
            print()

        # Stage 1: Detect backdrop region (defines ROI in original image)
        if self.debug:
            print("Stage 1: Detecting backdrop boundary...")

        corners = self.detect_backdrop_region(image)

        if corners is None:
            # Fallback: use entire image
            if self.debug:
                print("  WARNING: Backdrop boundary not found, using entire image")
            roi = image
            roi_bounds = (0, 0, w_original, h_original)  # (x, y, w, h)
        else:
            if self.debug:
                print(f"  [OK] Found backdrop quadrilateral")

            # Extract ROI from original image (no warping!)
            # Get bounding box of the detected corners
            x_coords = corners[:, 0]
            y_coords = corners[:, 1]
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

            roi = image[y_min:y_max, x_min:x_max]
            roi_bounds = (x_min, y_min, x_max - x_min, y_max - y_min)

            if self.debug:
                print(f"  [OK] ROI extracted: {roi.shape[1]}x{roi.shape[0]} at ({x_min}, {y_min})")

        # Stage 2: Detect numbers within ROI (do this FIRST)
        if self.debug:
            print("Stage 2: Detecting numbers with OCR in ROI...")

        numbers_roi = self.detect_numbers_ocr(roi)

        # Convert ROI coordinates back to original image coordinates
        numbers = [(num, y_pos + roi_bounds[1]) for num, y_pos in numbers_roi]

        if self.debug:
            print(f"  [OK] Detected {len(numbers)} numbers")

        # Stage 3: Detect lines within ROI
        if self.debug:
            print("Stage 3: Detecting horizontal lines in ROI...")

        lines_roi = self.detect_lines_morphological(roi)

        # Convert ROI coordinates back to original image coordinates
        lines = [y_pos + roi_bounds[1] for y_pos in lines_roi]

        if self.debug:
            print(f"  [OK] Detected {len(lines)} lines via Hough Transform")

        # If line detection failed but we have good OCR, use number positions as line positions
        if len(lines) < self.expected_lines * 0.5 and len(numbers) >= self.expected_lines * 0.7:
            if self.debug:
                print(f"  Using number y-positions as line positions (more reliable)")
            lines = [y_pos for _, y_pos in numbers]

            if self.debug:
                print(f"  [OK] Using {len(lines)} number-based line positions")

        # Stage 5: Calculate calibration
        # CRITICAL: Use original image height, NOT warped image height
        # The calibration must be relative to the full original image for normalized coordinates
        if self.debug:
            print("Stage 5: Calculating calibration...")

        calibration_data = self.calculate_calibration(lines, numbers, h_original)

        if self.debug:
            print()
            print("="*70)
            print("CALIBRATION RESULT")
            print("="*70)
            print(f"Detected: {len(lines)}/{self.expected_lines} lines, "
                  f"{len(numbers)}/{self.expected_numbers} numbers")

            if calibration_data.get('cm_per_normalized_unit'):
                print(f"Calibration: {calibration_data['cm_per_normalized_unit']:.2f} cm/normalized_unit")
                print(f"Pixels per cm: {calibration_data['pixels_per_cm']:.2f}")
                print(f"Line spacing: {calibration_data['line_spacing_pixels']:.2f} px "
                      f"(CV: {calibration_data['line_spacing_cv']:.3f})")
                print(f"Confidence: {calibration_data['confidence']}")
            else:
                print(f"Calibration: FAILED - {calibration_data.get('error', 'Unknown error')}")
            print("="*70)

        # Save visualization if requested
        if self.visualization_dir:
            self._save_visualization(image, corners, lines, numbers, calibration_data)

        return {
            "calibration_data": calibration_data,
            "method": "corner_detection",
            "backdrop_corners": corners.tolist() if corners is not None else None,
            "roi_bounds": roi_bounds
        }

    def _save_visualization(
        self,
        original: np.ndarray,
        corners: Optional[np.ndarray],
        lines: List[int],
        numbers: List[Tuple[int, int]],
        calibration_data: Dict
    ):
        """Save visualization image showing detections on original image."""
        # Single visualization: Original image with backdrop boundary, detected lines and numbers
        vis = original.copy()
        h, w = vis.shape[:2]

        # Draw backdrop boundary if detected
        if corners is not None:
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
            for i, corner in enumerate(corners):
                cv2.circle(vis, tuple(corner.astype(int)), 8, (255, 0, 0), -1)

        # Draw detected lines (in original image coordinates)
        for y in lines:
            cv2.line(vis, (0, y), (w, y), (0, 255, 255), 2)

        # Draw detected numbers (in original image coordinates)
        for num, y in numbers:
            cv2.circle(vis, (int(w * 0.15), y), 6, (255, 0, 255), -1)
            cv2.putText(
                vis, str(num), (int(w * 0.18), y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
            )

        # Add info text
        info_text = [
            f"Lines: {calibration_data['detected_lines']}/{self.expected_lines}",
            f"Numbers: {calibration_data['detected_numbers']}/{self.expected_numbers}",
            f"Confidence: {calibration_data['confidence']}"
        ]

        if calibration_data.get('cm_per_normalized_unit'):
            info_text.append(f"Cal: {calibration_data['cm_per_normalized_unit']:.2f} cm/norm")

        for i, text in enumerate(info_text):
            cv2.putText(
                vis, text, (10, 40 + i*35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        cv2.imwrite(
            str(Path(self.visualization_dir) / "1_backdrop_detection.jpg"),
            vis
        )

        if self.debug:
            print(f"\nVisualization saved to {self.visualization_dir}/1_backdrop_detection.jpg")


def calibrate_backdrop(
    image_path: str,
    config_path: str,
    output_calibration_path: Optional[str] = None,
    visualization_dir: Optional[str] = None,
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    debug: bool = False
) -> Dict:
    """
    Calibrate a backdrop image using perspective-aware detection.

    Args:
        image_path: Path to input image
        config_path: Path to backdrop config JSON
        output_calibration_path: Optional path to save calibration data
        visualization_dir: Optional directory to save visualization images
        tesseract_cmd: Path to Tesseract executable
        debug: Print debug information

    Returns:
        Calibration result dictionary
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        target_config = json.load(f)

    # Perform calibration
    calibrator = PerspectiveBackdropCalibrator(
        target_config,
        tesseract_cmd,
        debug,
        visualization_dir
    )
    result = calibrator.calibrate(image)

    # Add image path to result
    result['image_path'] = image_path

    # Save calibration data
    if output_calibration_path:
        with open(output_calibration_path, 'w') as f:
            json.dump(result, f, indent=2)
        if debug:
            print(f"\nCalibration data saved to {output_calibration_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate measurement backdrop using perspective-aware detection"
    )
    parser.add_argument("input_image", type=str, help="Path to backdrop image")
    parser.add_argument("config", type=str, help="Path to backdrop configuration JSON")
    parser.add_argument(
        "-c", "--calibration-output",
        type=str,
        help="Path to save calibration data JSON"
    )
    parser.add_argument(
        "-v", "--visualization-dir",
        type=str,
        help="Directory to save visualization images"
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Path to Tesseract executable"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    # Perform calibration
    result = calibrate_backdrop(
        args.input_image,
        args.config,
        args.calibration_output,
        args.visualization_dir,
        args.tesseract_cmd,
        debug=args.debug
    )

    # Print summary (if not in debug mode)
    if not args.debug:
        print("\n" + "="*70)
        print("CALIBRATION SUMMARY")
        print("="*70)
        cal_data = result['calibration_data']
        print(f"Detected: {cal_data['detected_lines']} lines, {cal_data['detected_numbers']} numbers")

        if cal_data.get('cm_per_normalized_unit'):
            print(f"Calibration: {cal_data['cm_per_normalized_unit']:.2f} cm/normalized_unit")
            print(f"Pixels per cm: {cal_data['pixels_per_cm']:.2f}")
            print(f"Confidence: {cal_data['confidence']}")
        else:
            print(f"Calibration: FAILED - {cal_data.get('error', 'Unknown error')}")

        print("="*70)
