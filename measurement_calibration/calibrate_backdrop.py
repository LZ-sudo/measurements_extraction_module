"""
Automatic Backdrop Calibration Script

This script calibrates line detection and OCR parameters for a measurement backdrop image.
It uses a two-stage optimization approach with Latin Hypercube Sampling (LHS) to efficiently
find the optimal configuration for different lighting conditions and camera setups.

Two-Stage Calibration Strategy:
1. STAGE 1 (Fast): Optimize line detection parameters using LHS
   - 5D parameter space: Canny low/high, minLineLength, Hough threshold, line upscaling
   - Score based on line count accuracy and spacing consistency
   - Tests 500+ combinations quickly (no OCR overhead)

2. STAGE 2 (Slower): Optimize OCR parameters with best line detection
   - Test ~20 OCR configurations: scale factor, region width, PSM mode
   - Combines line detection results with number detection
   - Final score weights: 50% lines, 30% spacing, 20% numbers

Features:
- Efficient two-stage optimization (fast line detection, then OCR)
- Multi-scale OCR for distant/small text (1.5x-5.0x upscaling)
- Configurable OCR region width and PSM modes
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
from scipy.stats import qmc
from pathlib import Path

# Import shared utilities
from calibration_utils import (
    LINE_PARAM_BOUNDS,
    OCR_CONFIGS,
    detect_lines_with_params,
    detect_numbers,
    calculate_calibration,
    calculate_score
)


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



    def calibrate(self, image: np.ndarray, n_samples: int = 500) -> Dict:
        """
        Perform two-stage calibration using Latin Hypercube Sampling.

        Stage 1: Optimize line detection parameters (fast)
        Stage 2: Optimize OCR parameters with best line detection (slower)

        Args:
            image: Input image
            n_samples: Number of parameter combinations to test for line detection (default: 500)

        Returns:
            Dictionary with optimal_params, calibration_data, score, validated
        """
        if self.debug:
            print("="*70)
            print("TWO-STAGE CALIBRATION")
            print("="*70)
            print(f"Target: {self.target_config['expected_lines']} lines, "
                  f"{self.target_config['expected_numbers']} numbers")
            print()

        # ==================================================================
        # STAGE 1: OPTIMIZE LINE DETECTION PARAMETERS
        # ==================================================================
        if self.debug:
            print("="*70)
            print("STAGE 1: LINE DETECTION OPTIMIZATION (Fast)")
            print("="*70)
            print(f"Testing {n_samples} line detection parameter combinations...")
            print()

        max_line_gap = 25  # Fixed
        best_line_score = -1
        best_line_result = None

        # Use parameter bounds from calibration_utils
        sampler = qmc.LatinHypercube(d=5, seed=42)
        lhs_samples = sampler.random(n=n_samples)
        lhs_scaled = qmc.scale(lhs_samples, LINE_PARAM_BOUNDS[:, 0], LINE_PARAM_BOUNDS[:, 1])

        if self.debug:
            print(f"{'#':<4} {'Canny':^12} {'Hough':>6} {'minLen':>6} {'LnScl':^6} {'Lines':>6} {'Spacing':>8}")
            print("-"*70)

        for i, params_array in enumerate(lhs_scaled):
            canny_low = int(params_array[0])
            canny_high = int(params_array[1])
            min_len_ratio = params_array[2]
            hough_thresh = int(params_array[3])
            line_scale = params_array[4]

            # Skip invalid combinations
            if canny_high <= canny_low * 1.5:
                continue

            # Detect lines with these parameters
            lines, thicknesses = detect_lines_with_params(
                image, canny_low, canny_high, hough_thresh,
                min_len_ratio, max_line_gap, line_scale
            )

            # Score based on line count and spacing consistency only (no numbers yet)
            line_score = 0
            if len(lines) > 0:
                # Line count accuracy (60% weight)
                line_error = abs(len(lines) - self.target_config['expected_lines']) / self.target_config['expected_lines']
                line_accuracy = max(0, 1 - line_error)
                line_score += line_accuracy * 0.60

                # Line spacing consistency (40% weight)
                if len(lines) >= 2:
                    spacings = np.diff(sorted(lines))
                    spacing_cv = np.std(spacings) / np.mean(spacings)
                    spacing_score = max(0, 1 - spacing_cv * 3.33)
                    line_score += spacing_score * 0.40

            if self.debug and (i % 50 == 0 or line_score > best_line_score):
                spacing_cv_val = np.std(np.diff(sorted(lines))) / np.mean(np.diff(sorted(lines))) if len(lines) >= 2 else 0
                print(f"{i+1:<4} ({canny_low:2},{canny_high:3}) "
                      f"{hough_thresh:6} {min_len_ratio:6.2f} {line_scale:5.1f}x {len(lines):6} "
                      f"{spacing_cv_val:7.3f}")

            # Track best line detection result
            if line_score > best_line_score:
                best_line_score = line_score
                best_line_result = {
                    'params': {
                        'canny_low': canny_low,
                        'canny_high': canny_high,
                        'min_line_length_ratio': min_len_ratio,
                        'max_line_gap': max_line_gap,
                        'hough_threshold': hough_thresh,
                        'line_scale_factor': line_scale
                    },
                    'lines': lines,
                    'line_thicknesses': thicknesses,
                    'score': line_score
                }

        if self.debug:
            print("="*70)
            print(f"Best line detection: {len(best_line_result['lines'])} lines, score: {best_line_score:.3f}")
            print()

        # ==================================================================
        # STAGE 2: OPTIMIZE OCR PARAMETERS WITH FIXED LINE DETECTION
        # ==================================================================
        if self.debug:
            print("="*70)
            print("STAGE 2: OCR OPTIMIZATION (Slower)")
            print("="*70)
            print("Testing OCR parameter combinations with best line detection...")
            print()

        # Use OCR configurations from calibration_utils

        best_overall_score = -1
        best_overall_result = None

        if self.debug:
            print(f"{'#':<4} {'Scale':>5} {'Region':>6} {'PSM':>4} {'Nums':>5} {'Score':>7}")
            print("-"*70)

        for i, (ocr_scale, ocr_region, psm_mode) in enumerate(OCR_CONFIGS):
            # Detect numbers with this OCR configuration
            numbers = detect_numbers(
                image, ocr_scale, ocr_region, psm_mode,
                number_range=tuple(self.target_config['number_range'])
            )

            # Calculate final score combining lines and numbers
            final_score = calculate_score(
                best_line_result['lines'], numbers,
                self.target_config['expected_lines'],
                self.target_config['expected_numbers']
            )

            if self.debug:
                print(f"{i+1:<4} {ocr_scale:5.1f}x {ocr_region:6.2f} {psm_mode:4} {len(numbers):5} {final_score:7.3f}")

            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_result = {
                    'params': {
                        **best_line_result['params'],
                        'ocr_scale_factor': ocr_scale,
                        'ocr_region_width': ocr_region,
                        'ocr_psm_mode': psm_mode
                    },
                    'lines': best_line_result['lines'],
                    'line_thicknesses': best_line_result['line_thicknesses'],
                    'numbers': numbers,
                    'score': final_score
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

        # Calculate calibration metrics using utility function
        h, w = image.shape[:2]
        calibration_data = calculate_calibration(lines, numbers, h)

        # Override confidence based on validation
        if validated and spacing_cv < 0.15 and line_accuracy <= 0.05:
            calibration_data["confidence"] = "high"
        elif validated and spacing_cv < 0.30 and line_accuracy <= 0.10:
            calibration_data["confidence"] = "medium"
        elif validated:
            calibration_data["confidence"] = "acceptable"
        elif calibration_data.get("cm_per_normalized_unit"):
            calibration_data["confidence"] = "needs_review"

        result = {
            "optimal_params": params,
            "calibration_data": calibration_data,
            "score": float(score),
            "validated": validated
        }

        if self.debug:
            print("="*70)
            print("FINAL CALIBRATION RESULT")
            print("="*70)
            print(f"Best Score: {score:.3f}")
            print(f"Validated: {validated}")
            print(f"\nDetected: {len(lines)} lines, {len(numbers)} numbers")
            print(f"Line spacing: {avg_spacing:.2f} px (CV: {spacing_cv:.3f})")
            print(f"\nOptimal Line Detection Parameters:")
            print(f"  Canny: ({params['canny_low']}, {params['canny_high']})")
            print(f"  Hough threshold: {params['hough_threshold']}")
            print(f"  Min line length ratio: {params['min_line_length_ratio']:.2f}")
            print(f"  Max line gap: {params['max_line_gap']}")
            print(f"  Line scale factor: {params['line_scale_factor']:.1f}x")
            print(f"\nOptimal OCR Parameters:")
            print(f"  OCR scale factor: {params['ocr_scale_factor']:.1f}x")
            print(f"  OCR region width: {params['ocr_region_width']:.2f}")
            print(f"  OCR PSM mode: {params['ocr_psm_mode']}")

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
