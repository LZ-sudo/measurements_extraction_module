"""
Camera Calibration Module
=========================

This module provides camera calibration functionality to correct lens distortion
and perspective effects for accurate measurements.

Two calibration approaches are supported:
1. Checkerboard-based calibration (standard OpenCV approach)
2. Backdrop-based calibration (using known backdrop measurements)

Author: Calibration Module
Date: 2025
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob


class CameraCalibrator:
    """
    Handles camera calibration using checkerboard patterns or backdrop measurements.
    """

    def __init__(self):
        """Initialize the camera calibrator."""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_error = None

    def calibrate_from_checkerboard(
        self,
        image_paths: List[str],
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size_mm: float = 25.0,
        visualize: bool = False
    ) -> Dict:
        """
        Calibrate camera using checkerboard pattern images.

        Args:
            image_paths: List of paths to checkerboard calibration images
            checkerboard_size: Inner corner count (columns, rows)
            square_size_mm: Physical size of each checkerboard square in mm
            visualize: Whether to show detected corners

        Returns:
            Dictionary with calibration results
        """
        # Prepare object points (3D points in real world space)
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size_mm

        # Arrays to store object points and image points
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        successful_images = []
        failed_images = []

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                failed_images.append(img_path)
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

            if ret:
                obj_points.append(objp)

                # Refine corner positions for sub-pixel accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners_refined)

                successful_images.append(img_path)

                if visualize:
                    # Draw and display corners
                    img_viz = img.copy()
                    cv2.drawChessboardCorners(img_viz, checkerboard_size, corners_refined, ret)
                    cv2.imshow(f'Corners: {Path(img_path).name}', img_viz)
                    cv2.waitKey(500)
            else:
                failed_images.append(img_path)

        if visualize:
            cv2.destroyAllWindows()

        if len(obj_points) < 3:
            return {
                "success": False,
                "error": f"Insufficient valid images. Found {len(obj_points)}, need at least 3",
                "successful_images": successful_images,
                "failed_images": failed_images
            }

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            gray.shape[::-1],
            None,
            None
        )

        # Calculate reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            img_points_reprojected, _ = cv2.projectPoints(
                obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(img_points[i], img_points_reprojected, cv2.NORM_L2) / len(img_points_reprojected)
            total_error += error

        mean_error = total_error / len(obj_points)

        # Store calibration parameters
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calibration_error = mean_error

        return {
            "success": True,
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "reprojection_error": mean_error,
            "num_successful_images": len(successful_images),
            "num_failed_images": len(failed_images),
            "successful_images": successful_images,
            "failed_images": failed_images
        }

    def calibrate_from_backdrop(
        self,
        backdrop_calibration_path: str,
        backdrop_image_path: str,
        backdrop_width_cm: float = 200.0,
        backdrop_height_cm: float = 200.0
    ) -> Dict:
        """
        Calibrate camera using backdrop with known measurements.

        This uses the detected corner positions from your backdrop calibration
        to compute perspective transform and approximate distortion correction.

        Args:
            backdrop_calibration_path: Path to backdrop calibration JSON
            backdrop_image_path: Path to backdrop image
            backdrop_width_cm: Physical width of backdrop in cm
            backdrop_height_cm: Physical height of backdrop in cm

        Returns:
            Dictionary with calibration results
        """
        # Load backdrop calibration data
        with open(backdrop_calibration_path, 'r') as f:
            backdrop_data = json.load(f)

        if "backdrop_corners" not in backdrop_data:
            return {
                "success": False,
                "error": "No backdrop corners found in calibration data"
            }

        # Load image to get dimensions
        img = cv2.imread(backdrop_image_path)
        if img is None:
            return {
                "success": False,
                "error": f"Could not load image: {backdrop_image_path}"
            }

        h, w = img.shape[:2]

        # Get detected corner positions (in pixels)
        corners_detected = np.array(backdrop_data["backdrop_corners"], dtype=np.float32)

        # Define known physical positions (in cm)
        # Assuming corners are: top-left, top-right, bottom-right, bottom-left
        corners_physical = np.array([
            [0, 0],                                    # Top-left
            [backdrop_width_cm, 0],                    # Top-right
            [backdrop_width_cm, backdrop_height_cm],   # Bottom-right
            [0, backdrop_height_cm]                    # Bottom-left
        ], dtype=np.float32)

        # For perspective-only correction (no lens distortion modeling)
        # we compute a homography matrix
        homography, _ = cv2.findHomography(corners_detected, corners_physical)

        # Estimate camera matrix (assuming principal point at image center)
        # This is a rough approximation
        fx = fy = max(w, h)  # Focal length approximation
        cx, cy = w / 2, h / 2  # Principal point at center

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # For backdrop-based calibration, we can't reliably estimate
        # radial/tangential distortion without multiple views
        # So we'll use perspective transform only
        dist_coeffs = np.zeros(5, dtype=np.float32)

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.homography = homography

        return {
            "success": True,
            "method": "backdrop_perspective",
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "homography": homography.tolist(),
            "note": "Backdrop calibration provides perspective correction only. For full lens distortion correction, use checkerboard calibration."
        }

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply distortion correction to an image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated. Run calibration first.")

        h, w = image.shape[:2]

        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            1,  # alpha=1 keeps all pixels (may have black borders)
            (w, h)
        )

        # Undistort
        undistorted = cv2.undistort(
            image,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_matrix
        )

        # Crop to ROI if desired (removes black borders)
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]

        return undistorted

    def save_calibration(self, output_path: str):
        """
        Save calibration parameters to JSON file.

        Args:
            output_path: Path to save calibration JSON
        """
        if self.camera_matrix is None:
            raise ValueError("No calibration data to save")

        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "calibration_error": self.calibration_error
        }

        if hasattr(self, 'homography'):
            calibration_data["homography"] = self.homography.tolist()

        with open(output_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)

        print(f"Calibration saved to: {output_path}")

    def load_calibration(self, calibration_path: str):
        """
        Load calibration parameters from JSON file.

        Args:
            calibration_path: Path to calibration JSON
        """
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)

        self.camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float32)
        self.dist_coeffs = np.array(calibration_data["distortion_coefficients"], dtype=np.float32)

        if "calibration_error" in calibration_data:
            self.calibration_error = calibration_data["calibration_error"]

        if "homography" in calibration_data:
            self.homography = np.array(calibration_data["homography"], dtype=np.float32)

        print(f"Calibration loaded from: {calibration_path}")


def create_calibration_from_images(
    calibration_images_dir: str,
    output_path: str,
    checkerboard_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    visualize: bool = False
) -> Dict:
    """
    Convenience function to calibrate from a directory of checkerboard images.

    Args:
        calibration_images_dir: Directory containing checkerboard images
        output_path: Path to save calibration JSON
        checkerboard_size: Inner corner count (columns, rows)
        square_size_mm: Physical size of each checkerboard square in mm
        visualize: Whether to show detected corners

    Returns:
        Calibration results dictionary
    """
    # Find all images in directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(str(Path(calibration_images_dir) / ext)))

    if not image_paths:
        return {
            "success": False,
            "error": f"No images found in {calibration_images_dir}"
        }

    print(f"Found {len(image_paths)} calibration images")

    # Calibrate
    calibrator = CameraCalibrator()
    results = calibrator.calibrate_from_checkerboard(
        image_paths,
        checkerboard_size,
        square_size_mm,
        visualize
    )

    # Save if successful
    if results["success"]:
        calibrator.save_calibration(output_path)

    return results
