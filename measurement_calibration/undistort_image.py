"""
Image Undistortion Utility

Applies lens distortion correction to an image using camera calibration data.

Usage:
    python undistort_image.py <input_image> <camera_calibration_json> [-o <output_path>]

Example:
    python undistort_image.py photo.jpg camera_calibration.json -o photo_undistorted.jpg
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from camera_calibration import CameraCalibrator


def undistort_image(
    input_path: str,
    calibration_path: str,
    output_path: str = None
) -> str:
    """
    Undistort an image using camera calibration data.

    Args:
        input_path: Path to input image
        calibration_path: Path to camera calibration JSON file
        output_path: Path for output image (default: <input>_undistorted.<ext>)

    Returns:
        Path to the undistorted image
    """
    import cv2

    # Load and validate input image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_path}")

    # Load camera calibration
    calibrator = CameraCalibrator()
    calibrator.load_calibration(calibration_path)

    # Undistort image
    undistorted = calibrator.undistort_image(image)

    # Determine output path
    if output_path is None:
        input_stem = Path(input_path).stem
        input_suffix = Path(input_path).suffix
        input_dir = Path(input_path).parent
        output_path = str(input_dir / f"{input_stem}_undistorted{input_suffix}")

    # Save undistorted image
    cv2.imwrite(output_path, undistorted)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Undistort an image using camera calibration data"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "calibration",
        type=str,
        help="Path to camera calibration JSON file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path for output image (default: <input>_undistorted.<ext>)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found: {args.input_image}")
        sys.exit(1)

    if not os.path.exists(args.calibration):
        print(f"Error: Calibration file not found: {args.calibration}")
        sys.exit(1)

    # Run undistortion
    try:
        output_path = undistort_image(
            args.input_image,
            args.calibration,
            args.output
        )
        print(f"Undistorted image saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
