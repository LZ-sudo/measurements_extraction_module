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
from calibration_utils import (
    match_numbers_to_lines,
    find_corner_numbers_adaptive,
    detect_lines_morphological,
    detect_numbers_ocr
)
import calibration_config as config
from camera_calibration import CameraCalibrator


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
        tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        debug: bool = False,
        visualization_dir: Optional[str] = None
    ):
        """
        Initialize calibrator.

        Configuration is loaded from calibration_config module.

        Args:
            tesseract_cmd: Path to Tesseract executable
            debug: Enable debug output
            visualization_dir: Directory to save visualization images
        """
        # Load configuration from config module
        self.expected_lines = config.EXPECTED_LINES
        self.expected_numbers = config.EXPECTED_NUMBERS
        self.number_range = config.NUMBER_RANGE
        self.cm_interval = config.CM_INTERVAL

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

        # Try to find all four corner numbers with adaptive OCR (using utility function)
        corner_positions = find_corner_numbers_adaptive(
            image, top_number, bottom_number, self.number_range, self.cm_interval, self.debug
        )

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
        margin_x = config.ROI_MARGIN_X
        margin_y = config.ROI_MARGIN_Y

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


    def calculate_calibration(
        self,
        lines: List[int],
        numbers: List[Tuple[int, int]],
        image_height: int,
        sequence_length: int = 0
    ) -> Dict:
        """
        Calculate calibration data from detected lines and numbers.

        Args:
            lines: List of line y-positions used for calibration
            numbers: List of (number, y_position) tuples used for calibration
            image_height: Original image height for normalized coordinates
            sequence_length: Length of the consecutive sequence (0 if not using sequence)

        Returns:
            Calibration result dictionary
        """
        if len(lines) >= 2 and len(numbers) >= 2:
            # Calculate spacing between consecutive lines
            line_spacings = np.diff(sorted(lines))
            avg_spacing = np.mean(line_spacings)
            spacing_cv = np.std(line_spacings) / avg_spacing if avg_spacing > 0 else 0

            # Determine cm interval from numbers
            number_diffs = [numbers[i+1][0] - numbers[i][0] for i in range(len(numbers)-1)]
            cm_interval = abs(int(np.median(number_diffs))) if number_diffs else 10

            # Calculate pixels per cm
            pixels_per_cm = avg_spacing / cm_interval

            # Calculate normalized units per cm (for normalized coordinates 0-1)
            normalized_unit_per_pixel = 1.0 / image_height
            cm_per_normalized_unit = 1.0 / (pixels_per_cm * normalized_unit_per_pixel)

            # Determine confidence based on sequence quality, not just counts
            # Key factors:
            # 1. Sequence length (longer = more reliable)
            # 2. Spacing consistency (lower CV = more reliable)
            confidence = self._calculate_confidence(sequence_length, spacing_cv)

            return {
                "cm_per_normalized_unit": float(cm_per_normalized_unit),
                "pixels_per_cm": float(pixels_per_cm),
                "detected_lines": len(lines),
                "detected_numbers": len(numbers),
                "line_spacing_pixels": float(avg_spacing),
                "line_spacing_cv": float(spacing_cv),
                "cm_interval": cm_interval,
                "sequence_length": sequence_length,
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
                "sequence_length": 0,
                "confidence": "failed",
                "error": "Not enough lines or numbers detected"
            }

    def _calculate_confidence(self, sequence_length: int, spacing_cv: float) -> str:
        """
        Calculate confidence based on consecutive sequence quality.

        A good calibration has:
        - Long consecutive sequence (8+ pairs = excellent, 5-7 = good, 3-4 = acceptable)
        - Low spacing variation (CV < 0.05 = excellent, < 0.1 = good, < 0.2 = acceptable)

        Args:
            sequence_length: Number of consecutive number-line pairs
            spacing_cv: Coefficient of variation for line spacing

        Returns:
            Confidence level: "high", "medium", "low", or "failed"
        """
        if sequence_length == 0:
            # Fallback mode (no sequence matching used)
            return "low"

        # Primary factor: sequence length
        if sequence_length >= 8:
            # Excellent sequence length
            if spacing_cv < 0.05:
                return "high"
        elif sequence_length >= 5:
            # Good sequence length
            if spacing_cv < 0.05:
                return "high"
            elif spacing_cv < 0.1:
                return "medium"
            else:
                return "low"
        elif sequence_length >= 3:
            # Minimum acceptable sequence length
            if spacing_cv < 0.05:
                return "medium"
            elif spacing_cv < 0.1:
                return "low"
            else:
                return "low"
        else:
            return "failed"

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

        numbers_roi = detect_numbers_ocr(
            roi,
            number_range=self.number_range,
            cm_interval=self.cm_interval,
            expected_numbers=self.expected_numbers,
            region_width=config.NUMBER_DETECTION_REGION_WIDTH
        )

        # Convert ROI coordinates back to original image coordinates
        numbers = [(num, y_pos + roi_bounds[1]) for num, y_pos in numbers_roi]

        if self.debug:
            print(f"  [OK] Detected {len(numbers)} numbers")

        # Stage 3: Detect lines within ROI
        if self.debug:
            print("Stage 3: Detecting horizontal lines in ROI...")

        lines_roi = detect_lines_morphological(
            roi,
            expected_lines=self.expected_lines,
            min_thickness=config.LINE_MIN_THICKNESS
        )

        # Convert ROI coordinates back to original image coordinates
        lines = [y_pos + roi_bounds[1] for y_pos in lines_roi]

        if self.debug:
            print(f"  [OK] Detected {len(lines)} lines via Hough Transform")

        # Stage 4: Match numbers with lines and find longest consecutive sequence
        if self.debug:
            print("Stage 4: Matching numbers with lines...")

        matched_pairs, longest_sequence = match_numbers_to_lines(
            numbers, lines,
            cm_interval=self.cm_interval,
            max_distance=config.NUMBER_LINE_MATCH_DISTANCE,
            min_sequence_length=config.MIN_CONSECUTIVE_SEQUENCE_LENGTH,
            debug=self.debug
        )

        if self.debug:
            print(f"  [OK] Matched {len(matched_pairs)} number-line pairs")
            print(f"  [OK] Longest consecutive sequence: {len(longest_sequence)} pairs")
            if longest_sequence:
                first_num = longest_sequence[0][0]
                last_num = longest_sequence[-1][0]
                print(f"      Range: {first_num} to {last_num} cm")

        # Use the matched pairs for calibration (prioritize longest sequence)
        if longest_sequence:
            numbers_for_cal = [(num, y) for num, y, _ in longest_sequence]
            lines_for_cal = [line_y for _, _, line_y in longest_sequence]
            sequence_length = len(longest_sequence)
        else:
            # Fallback: use all detected numbers and lines if matching failed
            numbers_for_cal = numbers
            lines_for_cal = lines
            sequence_length = 0

        # Stage 5: Calculate calibration
        # CRITICAL: Use original image height, NOT warped image height
        # The calibration must be relative to the full original image for normalized coordinates
        if self.debug:
            print("Stage 5: Calculating calibration...")

        calibration_data = self.calculate_calibration(
            lines_for_cal, numbers_for_cal, h_original, sequence_length
        )

        if self.debug:
            print()
            print("="*70)
            print("CALIBRATION RESULT")
            print("="*70)
            print(f"Detected: {len(lines)}/{self.expected_lines} lines, "
                  f"{len(numbers)}/{self.expected_numbers} numbers")
            print(f"Consecutive sequence: {calibration_data.get('sequence_length', 0)} pairs")

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
    output_calibration_path: Optional[str] = None,
    visualization_dir: Optional[str] = None,
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    debug: bool = False
) -> Dict:
    """
    Calibrate a backdrop image using perspective-aware detection.

    Configuration is loaded from calibration_config module (no JSON file needed).

    Args:
        image_path: Path to input image
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

    # Perform calibration (config loaded from calibration_config.py)
    calibrator = PerspectiveBackdropCalibrator(
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

    # Perform calibration (config is now loaded from calibration_config.py)
    result = calibrate_backdrop(
        args.input_image,
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
