"""
ArUco Marker-Based Backdrop Calibration

This script uses ArUco markers for robust, accurate backdrop calibration:

ADVANTAGES OVER LINE/OCR DETECTION:
- Works on both original and undistorted images (solves camera calibration chicken-and-egg problem)
- Sub-pixel corner detection accuracy
- No OCR required (marker size is known from printing)
- Robust to lighting variations, blur, and distortion
- Industry-standard approach used in robotics, AR, and computer vision

SETUP REQUIRED:
1. Print ArUco markers using generate_aruco_markers.py
2. Attach the 4 markers (IDs 0-3) to the backdrop corners
3. Ensure markers are flat, visible, and exactly the printed size

CALIBRATION PROCESS:
1. Detect all 4 ArUco markers (IDs 0, 1, 2, 3)
2. Extract marker corners with sub-pixel accuracy
3. Calculate scale from known marker size
4. Create homography matrix for perspective transformation
5. Output calibration data compatible with existing measurement pipeline
"""

import cv2
import numpy as np
import json
import argparse
from typing import Dict, Optional
from pathlib import Path

# Import configuration and utilities
import calibration_config as config
from calibration_utils import (
    detect_aruco_markers,
    organize_markers_by_id,
    get_backdrop_corners_from_markers,
    calculate_calibration_from_markers,
    get_roi_bounds_from_corners
)
from camera_calibration import CameraCalibrator


class ArUcoBackdropCalibrator:
    """
    Calibrator using ArUco markers for robust backdrop detection.

    This approach provides:
    - Reliable detection on both original and undistorted images
    - Sub-pixel accuracy for marker corners
    - Known scale from printed marker size
    - Automatic perspective correction via homography
    """

    def __init__(
        self,
        marker_details: Optional[Dict] = None,
        debug: bool = False,
        visualization_dir: Optional[str] = None,
        camera_calibration_path: Optional[str] = None
    ):
        """
        Initialize ArUco-based calibrator.

        Args:
            marker_details: Optional dict containing marker_size_cm and marker_positions_cm
                          (if None, uses defaults from config)
            debug: Enable debug output
            visualization_dir: Directory to save visualization images
            camera_calibration_path: Optional path to camera calibration JSON file
        """
        # Extract marker size and positions from marker_details or use config defaults
        if marker_details:
            self.marker_size_cm = marker_details.get('marker_size_cm', config.DEFAULT_MARKER_SIZE_CM)
            self.marker_positions = marker_details.get('marker_positions_cm', config.DEFAULT_MARKER_POSITIONS)
        else:
            self.marker_size_cm = config.DEFAULT_MARKER_SIZE_CM
            self.marker_positions = config.DEFAULT_MARKER_POSITIONS
        self.debug = debug
        self.visualization_dir = visualization_dir

        if visualization_dir:
            Path(visualization_dir).mkdir(parents=True, exist_ok=True)

        # Load camera calibration if provided
        self.camera_calibrator = None
        if camera_calibration_path:
            self.camera_calibrator = CameraCalibrator()
            self.camera_calibrator.load_calibration(camera_calibration_path)
            if self.debug:
                print(f"Loaded camera calibration from: {camera_calibration_path}")

    def calibrate(self, image: np.ndarray) -> Dict:
        """
        Calibrate backdrop using ArUco markers.

        Args:
            image: Input image (BGR)

        Returns:
            Calibration result dictionary
        """
        if self.debug:
            print("=" * 70)
            print("ARUCO MARKER-BASED BACKDROP CALIBRATION")
            print("=" * 70)
            print(f"Expected marker size: {self.marker_size_cm} cm")
            print(f"Required markers: IDs {sorted(config.REQUIRED_MARKER_IDS)}")
            print(f"Marker positions (cm from floor):")
            print(f"  Top-Left (ID 0):     x={self.marker_positions['top_left']['x']}, y={self.marker_positions['top_left']['y']}")
            print(f"  Top-Right (ID 1):    x={self.marker_positions['top_right']['x']}, y={self.marker_positions['top_right']['y']}")
            print(f"  Bottom-Left (ID 2):  x={self.marker_positions['bottom_left']['x']}, y={self.marker_positions['bottom_left']['y']}")
            print(f"  Bottom-Right (ID 3): x={self.marker_positions['bottom_right']['x']}, y={self.marker_positions['bottom_right']['y']}")
            print()

        # Store original image dimensions
        h_original, w_original = image.shape[:2]

        # Stage 0: Apply camera calibration (undistortion) if available
        if self.camera_calibrator is not None:
            if self.debug:
                print("Stage 0: Applying camera undistortion...")

            image = self.camera_calibrator.undistort_image(image)

            if self.debug:
                print("  [OK] Image undistorted using camera calibration")
                print()
        else:
            if self.debug:
                print("WARNING: No camera calibration provided")
                print("  Lens distortion may cause measurement inaccuracies")
                print("  For best results, calibrate camera first")
                print()

        # Stage 1: Detect ArUco markers
        if self.debug:
            print("Stage 1: Detecting ArUco markers...")

        corners, ids, rejected = detect_aruco_markers(image, config.ARUCO_DICT_TYPE)

        if ids is None:
            if self.debug:
                print("  ERROR: No ArUco markers detected!")
                print("  Check that markers are visible, flat, and in focus")

            return {
                "calibration_data": {
                    "cm_per_normalized_unit": None,
                    "pixels_per_cm": None,
                    "detected_markers": 0,
                    "confidence": "failed",
                    "error": "No ArUco markers detected"
                },
                "method": config.CALIBRATION_METHOD,
                "backdrop_corners": None,
                "roi_bounds": None
            }

        if self.debug:
            print(f"  [OK] Detected {len(ids)} ArUco markers")
            print(f"      Marker IDs: {ids.flatten().tolist()}")

        # Stage 2: Verify all required markers are present
        if self.debug:
            print("Stage 2: Verifying required markers...")

        marker_dict = organize_markers_by_id(corners, ids, config.REQUIRED_MARKER_IDS)

        if marker_dict is None:
            detected_ids = ids.flatten().tolist() if ids is not None else []
            missing_ids = list(config.REQUIRED_MARKER_IDS - set(detected_ids))

            if self.debug:
                print(f"  ERROR: Missing required markers: {missing_ids}")
                print(f"  Detected IDs: {detected_ids}")
                print(f"  Ensure all {len(config.REQUIRED_MARKER_IDS)} corner markers are visible")

            return {
                "calibration_data": {
                    "cm_per_normalized_unit": None,
                    "pixels_per_cm": None,
                    "detected_markers": len(detected_ids),
                    "detected_marker_ids": detected_ids,
                    "missing_marker_ids": missing_ids,
                    "confidence": "failed",
                    "error": f"Missing markers: {missing_ids}"
                },
                "method": config.CALIBRATION_METHOD,
                "backdrop_corners": None,
                "roi_bounds": None
            }

        if self.debug:
            print(f"  [OK] All {len(config.REQUIRED_MARKER_IDS)} required markers present")

        # Stage 3: Calculate calibration from markers
        if self.debug:
            print("Stage 3: Calculating calibration from markers...")

        calibration_data = calculate_calibration_from_markers(
            marker_dict,
            self.marker_size_cm,
            h_original,
            self.marker_positions
        )

        if self.debug:
            print(f"  Marker sizes (pixels):")
            for marker_id, size_px in zip(sorted(marker_dict.keys()),
                                         calibration_data['marker_size_pixels_individual']):
                print(f"    Marker {marker_id}: {size_px:.2f} px")
            print(f"  Average: {calibration_data['marker_size_pixels_avg']:.2f} px")
            print(f"  Variation (CV): {calibration_data['marker_size_variation_cv']:.4f}")

            # Show detailed calibration breakdown
            print(f"  Pixels per cm calculation:")
            print(f"    From marker size: {calibration_data['pixels_per_cm_from_marker_size']:.2f}")
            print(f"    From horizontal distance: {calibration_data['pixels_per_cm_horizontal']:.2f}")
            print(f"    From vertical distance: {calibration_data['pixels_per_cm_vertical']:.2f}")
            print(f"    Final (averaged): {calibration_data['pixels_per_cm']:.2f}")
            print(f"    Method: {calibration_data['pixels_per_cm_method']}")

        # Stage 4: Get backdrop corners
        if self.debug:
            print("Stage 4: Extracting backdrop corners...")

        backdrop_corners = get_backdrop_corners_from_markers(marker_dict)

        if self.debug:
            corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            for i, (x, y) in enumerate(backdrop_corners):
                print(f"  {corner_names[i]}: ({x:.1f}, {y:.1f})")

        # Calculate ROI bounds
        roi_bounds = get_roi_bounds_from_corners(backdrop_corners)

        if self.debug:
            print()
            print("=" * 70)
            print("CALIBRATION RESULT")
            print("=" * 70)
            print(f"Detected: {calibration_data['detected_markers']}/{len(config.REQUIRED_MARKER_IDS)} markers")
            print(f"Marker IDs: {calibration_data['detected_marker_ids']}")
            print(f"Calibration: {calibration_data['cm_per_normalized_unit']:.2f} cm/normalized_unit")
            print(f"Pixels per cm: {calibration_data['pixels_per_cm']:.2f}")
            print(f"Marker size variation: {calibration_data['marker_size_variation_cv']:.4f}")
            print(f"Confidence: {calibration_data['confidence']}")
            print("=" * 70)

        # Save visualization if requested
        if self.visualization_dir:
            self._save_visualization(image, marker_dict, backdrop_corners, calibration_data)

        return {
            "calibration_data": calibration_data,
            "method": config.CALIBRATION_METHOD,
            "backdrop_corners": backdrop_corners.tolist(),
            "marker_positions": self.marker_positions,
            "roi_bounds": roi_bounds
        }

    def _save_visualization(
        self,
        image: np.ndarray,
        marker_dict: Dict[int, np.ndarray],
        backdrop_corners: np.ndarray,
        calibration_data: Dict
    ):
        """Save visualization image showing detected markers and backdrop corners."""
        vis = image.copy()

        # Draw detected markers
        corners_list = [marker_dict[i].reshape(1, 4, 2).astype(np.int32)
                       for i in sorted(marker_dict.keys())]
        ids_array = np.array([[i] for i in sorted(marker_dict.keys())])
        cv2.aruco.drawDetectedMarkers(vis, corners_list, ids_array,
                                     borderColor=config.VIZ_COLOR_MARKER)

        # Draw backdrop corners
        pts = backdrop_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], True, config.VIZ_COLOR_BACKDROP,
                     config.VIZ_BACKDROP_LINE_THICKNESS)

        # Draw corner labels
        corner_names = ["TL (0)", "TR (1)", "BR (3)", "BL (2)"]
        for i, (corner, name) in enumerate(zip(backdrop_corners, corner_names)):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(vis, (x, y), config.VIZ_CORNER_RADIUS,
                      config.VIZ_COLOR_CENTER, -1)
            cv2.putText(
                vis, name, (x + 15, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.VIZ_COLOR_CENTER, 2
            )

        # Add info text
        info_text = [
            f"Markers: {calibration_data['detected_markers']}/{len(config.REQUIRED_MARKER_IDS)} detected",
            f"Cal: {calibration_data['cm_per_normalized_unit']:.2f} cm/norm",
            f"Pixels/cm: {calibration_data['pixels_per_cm']:.2f}",
            f"Confidence: {calibration_data['confidence']}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(
                vis, text, (10, 40 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX, config.VIZ_TEXT_SCALE,
                config.VIZ_COLOR_TEXT, config.VIZ_TEXT_THICKNESS
            )

        output_path = Path(self.visualization_dir) / config.DEFAULT_VISUALIZATION_FILENAME
        cv2.imwrite(str(output_path), vis)

        if self.debug:
            print(f"\nVisualization saved to {output_path}")


def calibrate_backdrop(
    image_path: str,
    output_calibration_path: Optional[str] = None,
    visualization_dir: Optional[str] = None,
    camera_calibration_path: Optional[str] = None,
    marker_details_path: Optional[str] = None,
    debug: bool = False
) -> Dict:
    """
    Calibrate a backdrop image using ArUco markers.

    Args:
        image_path: Path to input image
        output_calibration_path: Optional path to save calibration data
        visualization_dir: Optional directory to save visualization images
        camera_calibration_path: Optional path to camera calibration JSON file
        marker_details_path: Optional path to JSON file with marker size and positions
        debug: Print debug information

    Returns:
        Calibration result dictionary
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Load marker details if provided
    marker_details = None
    if marker_details_path:
        with open(marker_details_path, 'r', encoding='utf-8') as f:
            marker_details = json.load(f)
        if debug:
            print(f"Loaded marker details from: {marker_details_path}")
            print(f"  Marker size: {marker_details.get('marker_size_cm')} cm")
            print(f"  Marker positions: {list(marker_details.get('marker_positions_cm', {}).keys())}")

    # Perform calibration
    calibrator = ArUcoBackdropCalibrator(
        marker_details=marker_details,
        debug=debug,
        visualization_dir=visualization_dir,
        camera_calibration_path=camera_calibration_path
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
        description="Calibrate measurement backdrop using ArUco markers"
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
        "--camera-calibration",
        type=str,
        help="Path to camera calibration JSON file (for lens distortion correction)"
    )
    parser.add_argument(
        "--marker-details",
        type=str,
        help="Path to JSON file containing marker_size_cm and marker_positions_cm"
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
        output_calibration_path=args.calibration_output,
        visualization_dir=args.visualization_dir,
        camera_calibration_path=args.camera_calibration,
        marker_details_path=args.marker_details,
        debug=args.debug
    )

    # Print summary (if not in debug mode)
    if not args.debug:
        print("\n" + "=" * 70)
        print("CALIBRATION SUMMARY")
        print("=" * 70)
        cal_data = result['calibration_data']

        if cal_data.get('cm_per_normalized_unit'):
            print(f"Detected: {cal_data['detected_markers']}/{len(config.REQUIRED_MARKER_IDS)} markers")
            print(f"Marker IDs: {cal_data.get('detected_marker_ids', [])}")
            print(f"Calibration: {cal_data['cm_per_normalized_unit']:.2f} cm/normalized_unit")
            print(f"Pixels per cm: {cal_data['pixels_per_cm']:.2f}")
            print(f"Confidence: {cal_data['confidence']}")
        else:
            print(f"Calibration: FAILED - {cal_data.get('error', 'Unknown error')}")
            if 'missing_marker_ids' in cal_data:
                print(f"Missing markers: {cal_data['missing_marker_ids']}")

        print("=" * 70)
