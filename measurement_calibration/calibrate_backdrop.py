"""
Automatic Backdrop Calibration Script

This script calibrates line detection parameters for a measurement backdrop image.
It uses Latin Hypercube Sampling (LHS) to efficiently explore the parameter space
and find the optimal configuration for different lighting conditions and camera setups.

Calibration Strategy:
1. Latin Hypercube Sampling across 4D parameter space (Canny low/high, minLineLength, Hough threshold)
2. Line thickness validation to ensure only bold demarcation lines are detected
3. Score by line count accuracy, spacing consistency, and number detection
4. Generate visualization showing detected lines and numbers
5. Select best overall configuration

Features:
- Efficient parameter exploration (70% fewer samples than grid search for same accuracy)
- Line thickness measurement and validation
- Visual output showing all detections with color coding
- Detailed parameter tuning information
"""

import cv2
import numpy as np
import json
import argparse
from typing import Dict, List, Tuple, Optional
import pytesseract
import re
from scipy.stats import qmc
from pathlib import Path


class BackdropCalibrator:
    """
    Latin Hypercube Sampling calibration for measurement backdrop images.
    Uses LHS to efficiently explore parameter space and find optimal detection settings.
    """

    def __init__(
        self,
        target_config: Dict,
        tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        debug: bool = False,
        visualization_dir: Optional[str] = None
    ):
        """
        Initialize the Backdrop Calibrator.

        Args:
            target_config: Dictionary with expected_lines, expected_numbers, number_range, cm_interval
            tesseract_cmd: Path to Tesseract executable
            debug: If True, print detailed calibration progress
            visualization_dir: Directory to save visualization images (None to skip)
        """
        self.target_config = target_config
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.debug = debug
        self.visualization_dir = visualization_dir
        if visualization_dir:
            Path(visualization_dir).mkdir(parents=True, exist_ok=True)

    def _measure_line_thickness(self, image: np.ndarray, y_coord: int, window: int = 20) -> float:
        """
        Measure the thickness of a horizontal line at given y-coordinate.

        Args:
            image: Grayscale image
            y_coord: Y-coordinate of the line
            window: Vertical window size around the line

        Returns:
            Average thickness in pixels
        """
        h, w = image.shape[:2]
        y_start = max(0, y_coord - window)
        y_end = min(h, y_coord + window)

        # Extract horizontal strip around the line
        strip = image[y_start:y_end, :]

        # For each column, measure the vertical extent of dark pixels
        thicknesses = []
        for col in range(0, w, 10):  # Sample every 10 pixels for efficiency
            column = strip[:, col]
            # Threshold to find dark pixels (lines are typically dark)
            dark_pixels = column < 128

            # Find continuous dark regions
            if np.any(dark_pixels):
                # Count consecutive dark pixels around center
                center = window
                thickness = 0
                # Count upward from center
                for i in range(center, -1, -1):
                    if i < len(dark_pixels) and dark_pixels[i]:
                        thickness += 1
                    else:
                        break
                # Count downward from center
                for i in range(center + 1, len(dark_pixels)):
                    if dark_pixels[i]:
                        thickness += 1
                    else:
                        break

                if thickness > 0:
                    thicknesses.append(thickness)

        return np.median(thicknesses) if thicknesses else 0

    def _detect_lines_with_params(
        self,
        image: np.ndarray,
        canny_low: int,
        canny_high: int,
        hough_threshold: int,
        min_line_length_ratio: float,
        max_line_gap: int,
        validate_thickness: bool = True,
        min_thickness: float = 2.0
    ) -> Tuple[List[int], List[float]]:
        """
        Detect horizontal lines with given parameters.

        Returns:
            Tuple of (y_coordinates, thicknesses)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=hough_threshold,
            minLineLength=int(image.shape[1] * min_line_length_ratio),
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

        # Remove duplicates within 5 pixels
        y_coords = sorted(set(y_coords))
        filtered_coords = []
        for i, y in enumerate(y_coords):
            if i == 0 or y - filtered_coords[-1] > 5:
                filtered_coords.append(y)

        # Measure line thickness for validation
        thicknesses = []
        if validate_thickness:
            validated_coords = []
            for y in filtered_coords:
                thickness = self._measure_line_thickness(gray, y)
                if thickness >= min_thickness:  # Only keep lines above minimum thickness
                    validated_coords.append(y)
                    thicknesses.append(thickness)
            return validated_coords, thicknesses
        else:
            return filtered_coords, [0.0] * len(filtered_coords)

    def _detect_numbers(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect measurement numbers using OCR.
        Uses proven Otsu + PSM 11 configuration from extract_norm_per_cm.py
        """
        h, w = image.shape[:2]
        left_region = image[:, :int(w * 0.15)]

        # Otsu thresholding
        gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if background is dark
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # Tesseract PSM 11 (sparse text)
        custom_config = r'--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)

        # Extract numbers
        numbers_with_pos = []
        seen_numbers = set()

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text:
                continue

            conf = float(data['conf'][i])
            if conf < 0:
                conf = 0

            if conf >= 0:
                clean_text = re.sub(r'[^\d]', '', text)
                if clean_text:
                    try:
                        number = int(clean_text)
                        # Range 20-220 from extract_norm_per_cm.py
                        if 20 <= number <= 220 and number not in seen_numbers:
                            seen_numbers.add(number)
                            y = data['top'][i] + data['height'][i] // 2
                            numbers_with_pos.append((number, y))
                    except ValueError:
                        pass

        numbers_with_pos.sort(key=lambda x: x[1])
        return numbers_with_pos

    def _visualize_detection(
        self,
        image: np.ndarray,
        lines: List[int],
        line_thicknesses: List[float],
        numbers: List[Tuple[int, int]],
        params: Dict,
        filename: str
    ):
        """
        Create visualization showing detected lines and numbers.

        Args:
            image: Original image
            lines: List of y-coordinates of detected lines
            line_thicknesses: List of measured thicknesses for each line
            numbers: List of (number, y_position) tuples
            params: Detection parameters used
            filename: Output filename
        """
        if not self.visualization_dir:
            return

        # Create a copy for visualization
        vis_image = image.copy()
        h, w = vis_image.shape[:2]

        # Draw detected lines
        for y, thickness in zip(lines, line_thicknesses):
            # Color code by thickness: green for thick lines, yellow for thin
            color = (0, 255, 0) if thickness >= 3.0 else (0, 255, 255)
            cv2.line(vis_image, (0, y), (w, y), color, 2)
            # Add thickness label
            cv2.putText(vis_image, f"{thickness:.1f}px", (w - 100, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw detected numbers
        for number, y in numbers:
            x = int(w * 0.15)  # Right edge of OCR region
            cv2.circle(vis_image, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(vis_image, str(number), (x + 10, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw OCR region boundary
        cv2.line(vis_image, (int(w * 0.15), 0), (int(w * 0.15), h), (128, 128, 128), 1)

        # Add parameter info
        info_y = 30
        param_text = [
            f"Lines: {len(lines)}/{self.target_config['expected_lines']}",
            f"Numbers: {len(numbers)}/{self.target_config['expected_numbers']}",
            f"Canny: ({params['canny_low']}, {params['canny_high']})",
            f"Hough: {params['hough_threshold']}",
            f"MinLen: {params['min_line_length_ratio']:.2f}",
            f"Thickness range: {min(line_thicknesses):.1f}-{max(line_thicknesses):.1f}px" if line_thicknesses else "N/A"
        ]

        for text in param_text:
            cv2.putText(vis_image, text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            info_y += 25

        # Save visualization
        output_path = Path(self.visualization_dir) / filename
        cv2.imwrite(str(output_path), vis_image)
        if self.debug:
            print(f"  Saved visualization: {output_path}")


    def _calculate_score(
        self,
        lines: List[int],
        numbers: List[Tuple[int, int]]
    ) -> float:
        """
        Calculate quality score for detected lines and numbers.

        Scoring:
        - Line count accuracy: 50% weight (most important)
        - Line spacing consistency: 30% weight (critical for calibration)
        - Number detection: 20% weight (validation only)
        """
        target = self.target_config
        scores = []

        # Criterion 1: Line count accuracy (50% weight)
        if target['expected_lines'] > 0:
            line_error = abs(len(lines) - target['expected_lines']) / target['expected_lines']
            line_accuracy = max(0, 1 - line_error)
            scores.append(line_accuracy * 0.50)
        else:
            scores.append(0)

        # Criterion 2: Line spacing consistency (30% weight)
        if len(lines) >= 2:
            spacings = np.diff(sorted(lines))
            spacing_cv = np.std(spacings) / np.mean(spacings)
            # Good spacing: CV < 0.15, Acceptable: CV < 0.30
            spacing_score = max(0, 1 - spacing_cv * 3.33)  # Maps CV=0.30 to score=0
            scores.append(spacing_score * 0.30)
        else:
            scores.append(0)

        # Criterion 3: Number detection (20% weight)
        if target['expected_numbers'] > 0:
            number_accuracy = min(len(numbers) / target['expected_numbers'], 1.0)
            scores.append(number_accuracy * 0.20)
        else:
            scores.append(0)

        return sum(scores)

    def calibrate(self, image: np.ndarray, n_samples: int = 500) -> Dict:
        """
        Perform calibration using Latin Hypercube Sampling.

        Parameter Ranges:
        - canny_low: [20, 100]
        - canny_high: [80, 250]
        - min_line_length_ratio: [0.25, 0.60]
        - hough_threshold: [40, 250]

        Args:
            image: Input image
            n_samples: Number of parameter combinations to test (default: 500)

        Returns:
            Dictionary with optimal_params, calibration_data, score, validated
        """
        if self.debug:
            print("="*70)
            print("LATIN HYPERCUBE SAMPLING CALIBRATION")
            print("="*70)
            print(f"Target: {self.target_config['expected_lines']} lines, "
                  f"{self.target_config['expected_numbers']} numbers")
            print(f"\nTesting {n_samples} parameter combinations using LHS...")
            print()

        # Detect numbers once (independent of line parameters)
        numbers = self._detect_numbers(image)

        if self.debug:
            print(f"Number Detection: {len(numbers)}/{self.target_config['expected_numbers']} numbers")
            print(f"  Detected: {[n[0] for n in numbers]}")
            print()

        max_line_gap = 25  # Fixed
        best_overall_score = -1
        best_overall_result = None

        # Latin Hypercube Sampling
        # Parameter bounds: [canny_low, canny_high, min_line_length_ratio, hough_threshold]
        param_bounds = np.array([
            [20, 100],   # canny_low
            [80, 250],   # canny_high
            [0.25, 0.60], # min_line_length_ratio
            [40, 250]    # hough_threshold
        ])

        # Generate LHS samples
        sampler = qmc.LatinHypercube(d=4, seed=42)
        lhs_samples = sampler.random(n=n_samples)
        # Scale to parameter bounds
        lhs_scaled = qmc.scale(lhs_samples, param_bounds[:, 0], param_bounds[:, 1])

        if self.debug:
            print("="*70)
            print("PARAMETER SEARCH RESULTS")
            print("="*70)
            print(f"{'#':<4} {'Canny':^12} {'Hough':>6} {'minLen':>6} {'Lines':>6} {'Thick':>8} {'Score':>7}")
            print("-"*70)

        for i, params_array in enumerate(lhs_scaled):
            canny_low = int(params_array[0])
            canny_high = int(params_array[1])
            min_len_ratio = params_array[2]
            hough_thresh = int(params_array[3])

            # Skip invalid combinations
            if canny_high <= canny_low * 1.5:
                continue

            # Detect lines with these parameters
            lines, thicknesses = self._detect_lines_with_params(
                image, canny_low, canny_high, hough_thresh,
                min_len_ratio, max_line_gap
            )

            # Calculate score
            score = self._calculate_score(lines, numbers)

            avg_thickness = np.mean(thicknesses) if thicknesses else 0

            if self.debug:
                print(f"{i+1:<4} ({canny_low:2},{canny_high:3}) "
                      f"{hough_thresh:6} {min_len_ratio:6.2f} {len(lines):6} "
                      f"{avg_thickness:6.1f}px {score:7.3f}")

            # Track best result
            if score > best_overall_score:
                best_overall_score = score
                best_overall_result = {
                    'params': {
                        'canny_low': canny_low,
                        'canny_high': canny_high,
                        'min_line_length_ratio': min_len_ratio,
                        'max_line_gap': max_line_gap,
                        'hough_threshold': hough_thresh
                    },
                    'lines': lines,
                    'line_thicknesses': thicknesses,
                    'numbers': numbers,
                    'score': score
                }

        if self.debug:
            print("="*70)
            print()

        # Extract best results
        lines = best_overall_result['lines']
        line_thicknesses = best_overall_result.get('line_thicknesses', [])
        numbers = best_overall_result['numbers']
        params = best_overall_result['params']
        score = best_overall_result['score']

        # Create visualization of best result
        if self.visualization_dir:
            self._visualize_detection(
                image, lines, line_thicknesses, numbers, params,
                filename="best_detection.jpg"
            )

        # Calculate line spacing statistics
        if len(lines) >= 2:
            spacings = np.diff(sorted(lines))
            avg_spacing = np.mean(spacings)
            spacing_std = np.std(spacings)
            spacing_cv = spacing_std / avg_spacing if avg_spacing > 0 else float('inf')
        else:
            avg_spacing = 0
            spacing_cv = float('inf')

        # Validation
        line_accuracy = abs(len(lines) - self.target_config['expected_lines']) / self.target_config['expected_lines']
        number_accuracy = len(numbers) / self.target_config['expected_numbers']

        validated = (
            line_accuracy <= 0.10 and  # Within 10% of target lines
            number_accuracy >= 0.80 and  # At least 80% of numbers
            len(lines) >= 2
        )

        # Calculate calibration metrics
        h, w = image.shape[:2]

        if len(lines) >= 2 and len(numbers) >= 2:
            avg_line_spacing = np.mean(np.diff(sorted(lines)))
            pixels_per_cm = avg_line_spacing / self.target_config['cm_interval']
            normalized_unit_per_pixel = 1.0 / h
            cm_per_normalized_unit = 1.0 / (pixels_per_cm * normalized_unit_per_pixel)

            # Confidence scoring
            if validated and spacing_cv < 0.15 and line_accuracy <= 0.05:
                confidence = "high"
            elif validated and spacing_cv < 0.30 and line_accuracy <= 0.10:
                confidence = "medium"
            elif validated:
                confidence = "acceptable"
            else:
                confidence = "needs_review"

            calibration_data = {
                "cm_per_normalized_unit": float(cm_per_normalized_unit),
                "pixels_per_cm": float(pixels_per_cm),
                "detected_lines": len(lines),
                "detected_numbers": len(numbers),
                "line_spacing_pixels": float(avg_line_spacing),
                "line_spacing_cv": float(spacing_cv),
                "cm_interval": self.target_config['cm_interval'],
                "confidence": confidence
            }
        else:
            calibration_data = {
                "cm_per_normalized_unit": None,
                "pixels_per_cm": None,
                "detected_lines": len(lines),
                "detected_numbers": len(numbers),
                "line_spacing_pixels": None,
                "line_spacing_cv": None,
                "cm_interval": self.target_config['cm_interval'],
                "confidence": "failed",
                "error": "Insufficient lines or numbers detected"
            }
            validated = False

        result = {
            "optimal_params": params,
            "calibration_data": calibration_data,
            "score": float(score),
            "validated": validated
        }

        if self.debug:
            print("="*70)
            print("CALIBRATION RESULT")
            print("="*70)
            print(f"Best Score: {score:.3f}")
            print(f"Validated: {validated}")
            print(f"\nDetected: {len(lines)} lines, {len(numbers)} numbers")
            print(f"Line spacing: {avg_spacing:.2f} px (CV: {spacing_cv:.3f})")
            print(f"\nOptimal Parameters:")
            print(f"  Canny: ({params['canny_low']}, {params['canny_high']})")
            print(f"  Hough threshold: {params['hough_threshold']}")
            print(f"  Min line length ratio: {params['min_line_length_ratio']:.2f}")
            print(f"  Max line gap: {params['max_line_gap']}")

            if calibration_data.get('cm_per_normalized_unit'):
                print(f"\nCalibration: {calibration_data['cm_per_normalized_unit']:.2f} cm/normalized_unit")
                print(f"Pixels per cm: {calibration_data['pixels_per_cm']:.2f}")
                print(f"Confidence: {calibration_data['confidence']}")
            print("="*70)

        return result


def calibrate_backdrop(
    image_path: str,
    config_path: str,
    output_params_path: Optional[str] = None,
    output_calibration_path: Optional[str] = None,
    visualization_dir: Optional[str] = None,
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    n_samples: int = 500,
    debug: bool = False
) -> Dict:
    """
    Convenience function to calibrate a backdrop image using Latin Hypercube Sampling.

    Args:
        image_path: Path to input image
        config_path: Path to backdrop config JSON
        output_params_path: Optional path to save optimal parameters
        output_calibration_path: Optional path to save calibration data
        visualization_dir: Optional directory to save visualization images
        tesseract_cmd: Path to Tesseract executable
        n_samples: Number of LHS samples to test (default: 500)
        debug: Print debug information
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        target_config = json.load(f)

    # Perform calibration
    calibrator = BackdropCalibrator(target_config, tesseract_cmd, debug, visualization_dir)
    result = calibrator.calibrate(image, n_samples=n_samples)

    # Save optimal parameters
    if output_params_path:
        with open(output_params_path, 'w') as f:
            json.dump(result['optimal_params'], f, indent=2)
        if debug:
            print(f"\nOptimal parameters saved to {output_params_path}")

    # Save calibration data
    if output_calibration_path:
        with open(output_calibration_path, 'w') as f:
            json.dump(result['calibration_data'], f, indent=2)
        if debug:
            print(f"Calibration data saved to {output_calibration_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically calibrate measurement backdrop detection parameters"
    )
    parser.add_argument("input_image", type=str, help="Path to backdrop image")
    parser.add_argument("config", type=str, help="Path to backdrop configuration JSON")
    parser.add_argument(
        "-p", "--params-output",
        type=str,
        help="Path to save optimal detection parameters JSON"
    )
    parser.add_argument(
        "-c", "--calibration-output",
        type=str,
        help="Path to save calibration data JSON"
    )
    parser.add_argument(
        "-v", "--visualization-dir",
        type=str,
        help="Directory to save visualization images showing detected lines and numbers"
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Path to Tesseract executable"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples for Latin Hypercube Sampling (default: 500)"
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
        args.params_output,
        args.calibration_output,
        args.visualization_dir,
        args.tesseract_cmd,
        n_samples=args.samples,
        debug=args.debug
    )

    # Print summary (if not in debug mode, which already printed)
    if not args.debug:
        print("\n" + "="*70)
        print("CALIBRATION SUMMARY")
        print("="*70)
        print(f"Score: {result['score']:.3f}")
        print(f"Validated: {result['validated']}")

        cal_data = result['calibration_data']
        print(f"\nDetected: {cal_data['detected_lines']} lines, "
              f"{cal_data['detected_numbers']} numbers")

        if cal_data.get('cm_per_normalized_unit'):
            print(f"\nCalibration: {cal_data['cm_per_normalized_unit']:.2f} cm/normalized_unit")
            print(f"Pixels per cm: {cal_data['pixels_per_cm']:.2f}")
            print(f"Line spacing: {cal_data['line_spacing_pixels']:.2f} pixels (CV: {cal_data.get('line_spacing_cv', 0):.3f})")
            print(f"Confidence: {cal_data['confidence']}")
        else:
            print(f"\nCalibration: FAILED - {cal_data.get('error', 'Unknown error')}")

        print(f"\nOptimal Parameters:")
        for key, value in result['optimal_params'].items():
            print(f"  {key}: {value}")
        print("="*70)
