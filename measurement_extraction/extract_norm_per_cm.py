"""
Measurement Calibration Extraction Module

This script extracts the calibration factor (normalized units per cm) from images
with measurement backdrops. It uses OpenCV for line detection and Tesseract OCR
to read the measurement markings.
"""

import cv2
import numpy as np
import pytesseract
import json
import re
from typing import Dict, List, Tuple, Optional


class MeasurementCalibrator:
    """
    A class to extract calibration information from measurement backdrop images.
    """

    def __init__(self, tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe", debug: bool = False):
        """
        Initialize the Measurement Calibrator.

        Args:
            tesseract_cmd: Path to Tesseract executable.
            debug: If True, print debug information during processing.
        """
        # Set Tesseract command path
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.debug = debug

    def _detect_horizontal_lines(self, image: np.ndarray) -> List[int]:
        """
        Detect horizontal lines in the image (bold 10cm demarcation lines only).

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            List of y-coordinates of detected horizontal lines, sorted.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection - use higher thresholds to detect only bold lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough Line Transform
        # Higher threshold to detect only BOLD lines (10cm demarcations)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=106,  # Higher threshold for bold lines only
            minLineLength=image.shape[1] * 0.47,  # At least 47% image width
            maxLineGap=25
        )

        if lines is None:
            return []

        # Extract y-coordinates of horizontal lines
        y_coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is approximately horizontal
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 3 or angle > 177:  # Strict horizontal check
                # Use average y-coordinate
                y_coords.append(int((y1 + y2) / 2))

        # Remove duplicate y-coordinates (within 5 pixels)
        y_coords = sorted(set(y_coords))
        filtered_coords = []
        for i, y in enumerate(y_coords):
            if i == 0 or y - filtered_coords[-1] > 5:
                filtered_coords.append(y)

        if self.debug:
            print(f"\nDebug: Detected {len(filtered_coords)} bold horizontal lines (10cm demarcations)")

        return filtered_coords

    def _extract_numbers_with_positions(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract numbers and their vertical positions from the image using OCR.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            List of tuples (number, y_position) sorted by y_position.
        """
        h, w = image.shape[:2]

        # Extract left portion of image where numbers are located
        left_region = image[:, :int(w * 0.15)]  # Left 15% of image

        # Preprocess for Tesseract - convert to grayscale and enhance contrast
        gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to make numbers clearer
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if background is dark (most pixels are black)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # Use Tesseract to detect text with bounding boxes
        # --psm 11: Sparse text. Find as much text as possible in no particular order
        # --oem 3: Use default OCR Engine mode
        # Configuration for better digit recognition
        custom_config = r'--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'

        # Get detailed data including bounding boxes
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)

        # Extract numbers with their positions
        numbers_with_pos = []
        seen_numbers = set()  # Track unique numbers to avoid duplicates

        if self.debug:
            print(f"\nDebug: Tesseract detected {len(data['text'])} text regions")

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            # Tesseract returns -1 for confidence when it's uncertain, convert to 0
            conf = float(data['conf'][i])
            if conf < 0:
                conf = 0

            if self.debug and text:
                print(f"  OCR: '{text}' (confidence: {conf:.1f})")

            # Filter for valid numbers
            # Lower threshold to 0 to accept all detections (we'll validate by range instead)
            if text and conf >= 0:
                # Try to parse as number (remove any non-digit characters)
                clean_text = re.sub(r'[^\d]', '', text)

                if clean_text:
                    try:
                        number = int(clean_text)

                        # Only keep numbers that look like measurement markings (20-220 range)
                        # and avoid duplicate detections
                        if 20 <= number <= 220 and number not in seen_numbers:
                            seen_numbers.add(number)

                            # Get vertical center of bounding box
                            y = data['top'][i] + data['height'][i] // 2
                            numbers_with_pos.append((number, y))

                            if self.debug:
                                print(f"    [OK] Accepted: {number} at y={y}")
                        elif self.debug and number in seen_numbers:
                            print(f"    [X] Duplicate: {number}")
                        elif self.debug:
                            print(f"    [X] Out of range: {number}")
                    except ValueError:
                        if self.debug:
                            print(f"    [X] Could not parse: '{clean_text}'")

        # Sort by y-position
        numbers_with_pos.sort(key=lambda x: x[1])

        if self.debug:
            print(f"\nDebug: Final detected numbers: {[n[0] for n in numbers_with_pos]}")

        return numbers_with_pos

    def calibrate(self, image: np.ndarray) -> Dict:
        """
        Extract calibration from measurement backdrop image.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary containing calibration information:
            {
                "cm_per_normalized_unit": float,
                "detected_lines": int,
                "detected_numbers": int,
                "line_spacing_pixels": float,
                "confidence": str
            }
        """
        h, w = image.shape[:2]

        # Detect horizontal lines
        line_y_coords = self._detect_horizontal_lines(image)

        # Extract numbers with positions
        numbers_with_pos = self._extract_numbers_with_positions(image)

        # Calculate calibration
        calibration_data = {}

        if len(line_y_coords) >= 2 and len(numbers_with_pos) >= 2:
            # Calculate average spacing between consecutive lines
            line_spacings = []
            for i in range(len(line_y_coords) - 1):
                spacing = line_y_coords[i + 1] - line_y_coords[i]
                line_spacings.append(spacing)

            avg_line_spacing_pixels = np.mean(line_spacings)

            # Try to determine cm interval between lines using OCR numbers
            number_diffs = []
            for i in range(len(numbers_with_pos) - 1):
                num_diff = numbers_with_pos[i + 1][0] - numbers_with_pos[i][0]
                number_diffs.append(num_diff)

            # Most common difference should be the cm interval
            # Take absolute value to handle both ascending and descending orders
            if number_diffs:
                cm_interval = abs(int(np.median(number_diffs)))
            else:
                # Default assumption: 10cm intervals
                cm_interval = 10

            # Calculate pixels per cm
            pixels_per_cm = avg_line_spacing_pixels / cm_interval

            # Calculate normalized units (0-1 range) per cm
            # Normalized unit = 1 / image_height
            normalized_unit_per_pixel = 1.0 / h
            cm_per_normalized_unit = 1.0 / (pixels_per_cm * normalized_unit_per_pixel)

            # Determine confidence
            if len(line_y_coords) >= 10 and len(numbers_with_pos) >= 5:
                confidence = "high"
            elif len(line_y_coords) >= 5 and len(numbers_with_pos) >= 3:
                confidence = "medium"
            else:
                confidence = "low"

            calibration_data = {
                "cm_per_normalized_unit": float(cm_per_normalized_unit),
                "pixels_per_cm": float(pixels_per_cm),
                "detected_lines": len(line_y_coords),
                "detected_numbers": len(numbers_with_pos),
                "line_spacing_pixels": float(avg_line_spacing_pixels),
                "cm_interval": cm_interval,
                "confidence": confidence
            }
        else:
            # Not enough data
            calibration_data = {
                "cm_per_normalized_unit": None,
                "pixels_per_cm": None,
                "detected_lines": len(line_y_coords),
                "detected_numbers": len(numbers_with_pos),
                "line_spacing_pixels": None,
                "cm_interval": None,
                "confidence": "failed",
                "error": "Not enough lines or numbers detected"
            }

        return calibration_data


def extract_calibration(
    image_path: str,
    output_path: Optional[str] = None,
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    debug: bool = False
) -> Dict:
    """
    Convenience function to extract calibration from an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        tesseract_cmd: Path to Tesseract executable.
        debug: If True, print debug information during processing.

    Returns:
        Dictionary containing calibration information.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create calibrator and extract calibration
    calibrator = MeasurementCalibrator(tesseract_cmd=tesseract_cmd, debug=debug)
    calibration_data = calibrator.calibrate(image)

    # Save output if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"Calibration data saved to {output_path}")

    return calibration_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract calibration (cm per normalized unit) from measurement backdrop image"
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument(
        "--tesseract-cmd",
        type=str,
        default=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        help="Path to Tesseract executable"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output showing OCR detections"
    )

    args = parser.parse_args()

    # Extract calibration
    calibration_data = extract_calibration(
        args.input_image,
        args.output,
        args.tesseract_cmd,
        args.debug
    )

    # Print the results
    print("\nCalibration Data:")
    print(json.dumps(calibration_data, indent=2))

    # Print interpretation
    if calibration_data.get("cm_per_normalized_unit"):
        print(f"\n[OK] Calibration successful!")
        print(f"  Confidence: {calibration_data['confidence']}")
        print(f"  1 normalized unit = {calibration_data['cm_per_normalized_unit']:.2f} cm")
        print(f"  1 cm = {1/calibration_data['cm_per_normalized_unit']:.6f} normalized units")
    else:
        print(f"\n[FAILED] Calibration failed: {calibration_data.get('error', 'Unknown error')}")
