"""
Camera Calibration Module
=========================

Simple module to load and apply camera calibration for undistorting images.
"""

import cv2
import numpy as np
import json


class CameraCalibrator:
    """Handles loading and applying camera calibration."""

    def __init__(self):
        """Initialize the camera calibrator."""
        self.camera_matrix = None
        self.dist_coeffs = None

    def load_calibration(self, calibration_path: str):
        """
        Load calibration parameters from JSON file.

        Args:
            calibration_path: Path to calibration JSON file
        """
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)

        self.camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float32)
        self.dist_coeffs = np.array(calibration_data["distortion_coefficients"], dtype=np.float32)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply distortion correction to an image.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated. Load calibration first.")

        h, w = image.shape[:2]

        # Get optimal new camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.dist_coeffs,
            (w, h),
            1,  # alpha=1 keeps all pixels
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

        return undistorted
