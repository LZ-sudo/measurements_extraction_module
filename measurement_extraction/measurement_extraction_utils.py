"""
Measurement Extraction Utilities

This module provides utility functions to convert normalized landmark coordinates
from MediaPipe detection outputs into real-world measurements (in cm) using
calibration data from the backdrop.

Key concepts:
- MediaPipe normalized coordinates: x and y are in range [0, 1]
  - x is normalized by image width
  - y is normalized by image height
- Calibration data provides cm_per_normalized_unit for vertical axis
- For horizontal measurements, we derive the scale using image aspect ratio
- Assumes square pixels (valid for modern cameras)
"""

import cv2
import numpy as np
import os
import sys
import tempfile
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import copy

# Add parent directory to path for config imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from measurement_calibration import calibration_config as config


# ============================================================================
# IMAGE PREPARATION UTILITIES
# ============================================================================

def prepare_image_for_measurement(
    image_path: str,
    camera_calibrator=None
) -> Tuple[str, Optional[str]]:
    """
    Prepare image for measurement extraction.

    If camera calibration is provided, undistorts the image and saves to temporary file.
    Otherwise returns the original image path.

    Args:
        image_path: Path to original image
        camera_calibrator: Optional CameraCalibrator instance for undistortion

    Returns:
        Tuple of (path_to_use, temp_path_to_cleanup):
            - path_to_use: Path to the image to use for detections
            - temp_path_to_cleanup: Path to temporary file (None if no temp file created)
    """
    if camera_calibrator is None:
        return image_path, None

    # Load and undistort image
    print("\nUndistorting image using camera calibration...")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    undistorted = camera_calibrator.undistort_image(image)

    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    image_name = Path(image_path).stem
    temp_path = os.path.join(temp_dir, f"{image_name}_undistorted.jpg")

    cv2.imwrite(temp_path, undistorted)
    print(f"Undistorted image saved to: {temp_path}")

    return temp_path, temp_path


def cleanup_temporary_image(temp_image_path: Optional[str]):
    """
    Clean up temporary undistorted image file.

    Args:
        temp_image_path: Path to temporary file to remove (can be None)
    """
    if temp_image_path and os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
            print(f"\nCleaned up temporary undistorted image")
        except Exception as e:
            print(f"\nWarning: Could not remove temporary file {temp_image_path}: {e}")


# ============================================================================
# PLANE SHIFTING UTILITIES (for depth correction)
# ============================================================================

def measure_height_in_pixels(
    body_data: Dict,
    pose_data: Optional[Dict],
    image_height: int
) -> Optional[float]:
    """
    Measure subject height in pixels from body segmentation and pose data.

    Uses body segmentation for the top (head) and pose heel landmarks for the
    bottom (feet). This is used to calculate the plane shift factor for depth
    correction.

    Args:
        body_data: Body detection data containing height information
        pose_data: Pose detection data with heel landmarks
        image_height: Image height in pixels

    Returns:
        Height in pixels, or None if data is invalid
    """
    try:
        height_info = body_data.get('height', {})
        y_top = height_info.get('top', {}).get('y')
        y_bottom_seg = height_info.get('bottom', {}).get('y')

        if y_top is None or y_bottom_seg is None:
            return None

        # Use segmentation bottom as default
        y_bottom = y_bottom_seg

        # Prefer heel landmarks if available (more accurate)
        if pose_data and 'heel_landmarks' in pose_data:
            heel_data = pose_data['heel_landmarks']
            left_heel = heel_data.get('left')
            right_heel = heel_data.get('right')

            if left_heel and right_heel:
                y_bottom = (left_heel['y'] + right_heel['y']) / 2

        # Convert normalized coordinates to pixels
        height_pixels = abs(y_bottom - y_top) * image_height
        return height_pixels

    except (KeyError, TypeError):
        return None


def calculate_shift_factor(
    known_height_cm: float,
    measured_height_px: float,
    pixels_per_cm_backdrop: float
) -> float:
    """
    Calculate the plane shift factor to move calibration from backdrop to subject plane.

    The shift factor represents the ratio D/d1 where:
    - D is the camera-to-backdrop distance
    - d1 is the camera-to-subject distance

    This factor is used to adjust the backdrop calibration to the subject's plane,
    correcting for the depth offset that causes measurement overestimation.

    Args:
        known_height_cm: Subject's known height in centimeters
        measured_height_px: Subject's measured height in pixels (from landmarks)
        pixels_per_cm_backdrop: Pixels per cm at the backdrop plane (from ArUco calibration)

    Returns:
        Shift factor to multiply with pixels_per_cm_backdrop
    """
    expected_height_px = known_height_cm * pixels_per_cm_backdrop
    shift_factor = measured_height_px / expected_height_px
    return shift_factor


def apply_calibration_shift(
    calibration_data: Dict,
    shift_factor: float,
    known_height_cm: float,
    body_data: Optional[Dict] = None,
    pose_data: Optional[Dict] = None,
    image_path: Optional[str] = None,
    iterate: bool = True,
    tolerance_cm: float = 0.1,
    max_iterations: int = 10
) -> Dict:
    """
    Apply shift factor to calibration data, returning shifted calibration.

    Creates a modified copy of the calibration data with adjusted values
    for the subject's plane. Original backdrop values are preserved.

    When iterate=True, performs iterative refinement using the perspective-corrected
    height measurement to converge the shift factor until the measured height
    matches the known height within the specified tolerance.

    Args:
        calibration_data: Original calibration data from ArUco backdrop calibration
        shift_factor: Initial shift factor from calculate_shift_factor()
        known_height_cm: Subject's known height (stored for reference)
        body_data: Body detection data (required if iterate=True)
        pose_data: Pose detection data (required if iterate=True)
        image_path: Path to image (required if iterate=True)
        iterate: Whether to perform iterative refinement (default: False)
        tolerance_cm: Convergence tolerance in cm (default: 0.1 cm)
        max_iterations: Maximum iterations (default: 10)

    Returns:
        New calibration dict with shifted values applied
    """

    # Perform iterative refinement if requested
    if iterate and body_data is not None and image_path is not None:
        print(f"\n   Iterative plane shift refinement:")

        for iteration in range(max_iterations):
            # Create temporary shifted calibration
            temp_calibration = copy.deepcopy(calibration_data)
            original_pixels_per_cm = calibration_data['calibration_data']['pixels_per_cm']
            original_cm_per_normalized = calibration_data['calibration_data']['cm_per_normalized_unit']

            temp_calibration['calibration_data']['pixels_per_cm'] = original_pixels_per_cm * shift_factor
            temp_calibration['calibration_data']['cm_per_normalized_unit'] = original_cm_per_normalized / shift_factor

            # Measure height using perspective-corrected method
            # extract_height is defined later in this file, so we call it directly
            measured_height_cm = extract_height(
                body_data, temp_calibration, image_path, pose_data
            )

            if measured_height_cm is None:
                print(f"      Iteration {iteration + 1}: Height extraction failed, stopping")
                break

            # Apply perspective correction (divide by current shift factor)
            corrected_height_cm = measured_height_cm / shift_factor

            # Calculate error
            error = corrected_height_cm - known_height_cm

            print(f"      Iteration {iteration + 1}: height={corrected_height_cm:.2f} cm, "
                  f"error={error:+.2f} cm, shift={shift_factor:.6f}")

            # Check convergence
            if abs(error) < tolerance_cm:
                print(f"      Converged after {iteration + 1} iteration(s)")
                break

            # Adjust shift factor proportionally
            adjustment = corrected_height_cm / known_height_cm
            shift_factor = shift_factor * adjustment

    # Apply final shift factor
    shifted_calibration = copy.deepcopy(calibration_data)

    # Get original backdrop values
    original_pixels_per_cm = calibration_data['calibration_data']['pixels_per_cm']
    original_cm_per_normalized = calibration_data['calibration_data']['cm_per_normalized_unit']

    # Calculate shifted values
    shifted_pixels_per_cm = original_pixels_per_cm * shift_factor
    shifted_cm_per_normalized = original_cm_per_normalized / shift_factor

    # Update calibration data with shifted values
    shifted_calibration['calibration_data']['pixels_per_cm'] = shifted_pixels_per_cm
    shifted_calibration['calibration_data']['cm_per_normalized_unit'] = shifted_cm_per_normalized

    # Store shift metadata for reference
    shifted_calibration['plane_shift'] = {
        'applied': True,
        'shift_factor': shift_factor,
        'known_height_cm': known_height_cm,
        'original_pixels_per_cm': original_pixels_per_cm,
        'original_cm_per_normalized_unit': original_cm_per_normalized,
        'shifted_pixels_per_cm': shifted_pixels_per_cm,
        'shifted_cm_per_normalized_unit': shifted_cm_per_normalized
    }

    return shifted_calibration


# ============================================================================
# PERSPECTIVE TRANSFORMATION UTILITIES
# ============================================================================

def create_perspective_transform(calibration_data: Dict) -> Optional[np.ndarray]:
    """
    Create homography matrix for perspective transformation from backdrop calibration.

    This transforms image coordinates to real-world backdrop coordinates (in cm),
    correcting for perspective distortion and camera angle.

    Args:
        calibration_data: Calibration data containing backdrop_corners and optionally marker_positions

    Returns:
        3x3 homography matrix, or None if corners not available
    """
    if 'backdrop_corners' not in calibration_data:
        return None

    # Detected corners in image (pixels)
    corners_image = np.array(calibration_data['backdrop_corners'], dtype=np.float32)

    # Get physical marker positions from calibration data, or use defaults from config
    # marker_positions defines where markers are physically placed (in cm from floor)
    marker_positions = calibration_data.get('marker_positions', config.DEFAULT_MARKER_POSITIONS)

    # Corresponding physical positions on backdrop (cm)
    # Order matches backdrop_corners: [top-left, top-right, bottom-right, bottom-left]
    corners_world = np.array([
        [marker_positions['top_left']['x'], marker_positions['top_left']['y']],
        [marker_positions['top_right']['x'], marker_positions['top_right']['y']],
        [marker_positions['bottom_right']['x'], marker_positions['bottom_right']['y']],
        [marker_positions['bottom_left']['x'], marker_positions['bottom_left']['y']]
    ], dtype=np.float32)

    # Compute homography matrix
    H, _ = cv2.findHomography(corners_image, corners_world)

    return H


def transform_point_to_world(
    x_normalized: float,
    y_normalized: float,
    H: np.ndarray,
    image_width: int,
    image_height: int
) -> Tuple[float, float]:
    """
    Transform a normalized image coordinate to world coordinates using homography.

    Args:
        x_normalized: X coordinate in normalized image space (0-1)
        y_normalized: Y coordinate in normalized image space (0-1)
        H: Homography matrix from create_perspective_transform()
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Tuple of (x_cm, y_cm) in world coordinates (backdrop coordinate system)
    """
    # Convert normalized to pixel coordinates
    x_px = x_normalized * image_width
    y_px = y_normalized * image_height

    # Apply perspective transform
    point_world = cv2.perspectiveTransform(
        np.array([[[x_px, y_px]]], dtype=np.float32),
        H
    )[0][0]

    return float(point_world[0]), float(point_world[1])


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Extract image dimensions from an image file.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height) in pixels

    Raises:
        ValueError: If image cannot be read
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    height, width = image.shape[:2]
    return width, height


def calculate_aspect_ratio(image_path: str) -> float:
    """
    Calculate the aspect ratio (width / height) of an image.

    Args:
        image_path: Path to the image file

    Returns:
        Aspect ratio (width / height)
    """
    width, height = get_image_dimensions(image_path)
    return width / height


def calculate_vertical_distance(
    y1: float,
    y2: float,
    cm_per_normalized_unit: float
) -> float:
    """
    Calculate vertical distance between two y-coordinates in cm.

    Used for measurements that are purely vertical (height, hair length).

    LEGACY METHOD: This uses simple scaling and doesn't account for perspective.
    For more accurate measurements, use calculate_vertical_distance_perspective().

    Args:
        y1: First y-coordinate (normalized, 0-1 range)
        y2: Second y-coordinate (normalized, 0-1 range)
        cm_per_normalized_unit: Calibration factor for vertical axis

    Returns:
        Distance in centimeters
    """
    return abs(y2 - y1) * cm_per_normalized_unit


def calculate_vertical_distance_perspective(
    y1: float,
    y2: float,
    x: float,
    calibration_data: Dict,
    image_width: int,
    image_height: int
) -> Optional[float]:
    """
    Calculate vertical distance with perspective correction.

    This method transforms coordinates to real-world backdrop space,
    correcting for perspective distortion and camera angle. This is
    significantly more accurate than simple scaling, especially when
    the floor is visible in the image.

    Args:
        y1: First y-coordinate (normalized, 0-1 range)
        y2: Second y-coordinate (normalized, 0-1 range)
        x: X-coordinate for both points (normalized, 0-1 range)
        calibration_data: Full calibration data with backdrop_corners
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Distance in centimeters, or None if perspective transform unavailable
    """
    # Create perspective transform
    H = create_perspective_transform(calibration_data)
    if H is None:
        return None

    # Transform both points to world coordinates
    x1_world, y1_world = transform_point_to_world(x, y1, H, image_width, image_height)
    x2_world, y2_world = transform_point_to_world(x, y2, H, image_width, image_height)

    # Calculate vertical distance in world coordinates (cm)
    vertical_distance_cm = abs(y2_world - y1_world)

    return vertical_distance_cm


def calculate_2d_distance(
    landmark1: Dict[str, float],
    landmark2: Dict[str, float],
    cm_per_normalized_unit_y: float,
    aspect_ratio: float
) -> float:
    """
    Calculate 2D Euclidean distance between two landmarks in cm.

    Used for measurements that span both x and y dimensions (shoulder width,
    arm length, etc.). The z-coordinate is ignored as it represents depth
    which is not calibrated.

    Args:
        landmark1: First landmark with 'x' and 'y' keys (normalized coordinates)
        landmark2: Second landmark with 'x' and 'y' keys (normalized coordinates)
        cm_per_normalized_unit_y: Calibration factor for vertical axis (from calibration data)
        aspect_ratio: Image aspect ratio (width / height)

    Returns:
        Distance in centimeters
    """
    # Calculate horizontal calibration scale based on aspect ratio
    # Since x is normalized by width and y by height, we need to account for this
    cm_per_normalized_unit_x = cm_per_normalized_unit_y * aspect_ratio

    # Calculate differences in normalized coordinates
    dx_normalized = abs(landmark2['x'] - landmark1['x'])
    dy_normalized = abs(landmark2['y'] - landmark1['y'])

    # Convert to centimeters
    dx_cm = dx_normalized * cm_per_normalized_unit_x
    dy_cm = dy_normalized * cm_per_normalized_unit_y

    # Calculate Euclidean distance
    distance_cm = np.sqrt(dx_cm**2 + dy_cm**2)

    return float(distance_cm)


def extract_height(
    body_data: Dict,
    calibration_data: Dict,
    image_path: str,
    pose_data: Optional[Dict] = None
) -> Optional[float]:
    """
    Extract height measurement from body detection data with perspective correction.

    Uses body segmentation for the top (head) and pose heel landmarks for the bottom (feet).
    This hybrid approach avoids including floor in the measurement while maintaining accuracy.

    Args:
        body_data: Body detection data containing height information
        calibration_data: Calibration data with backdrop_corners and cm_per_normalized_unit
        image_path: Path to the image (needed for dimensions)
        pose_data: Optional pose detection data with heel landmarks for feet position

    Returns:
        Height in centimeters, or None if data is invalid
    """
    try:
        height_info = body_data.get('height', {})
        y_top = height_info.get('top', {}).get('y')
        y_bottom_seg = height_info.get('bottom', {}).get('y')

        if y_top is None or y_bottom_seg is None:
            return None

        # Use segmentation bottom as default, but prefer heel landmarks if available
        y_bottom = y_bottom_seg
        x_bottom = height_info.get('bottom', {}).get('x', 0.5)

        # Try to get more accurate foot position from pose heel landmarks
        if pose_data and 'heel_landmarks' in pose_data:
            heel_data = pose_data['heel_landmarks']
            left_heel = heel_data.get('left')
            right_heel = heel_data.get('right')

            if left_heel and right_heel:
                # Use average of both heels for more robust measurement
                y_bottom = (left_heel['y'] + right_heel['y']) / 2
                x_bottom = (left_heel['x'] + right_heel['x']) / 2
                print(f"  Using pose heel landmarks for feet position (y={y_bottom:.4f})")
                print(f"  Segmentation bottom was at y={y_bottom_seg:.4f}")
            else:
                print(f"  WARNING: Heel landmarks incomplete, using segmentation bottom")
        else:
            print(f"  WARNING: No heel landmarks available, using segmentation bottom")

        # Get x-coordinate for top
        x_top = height_info.get('top', {}).get('x', 0.5)
        x_avg = (x_top + x_bottom) / 2

        print(f"  Height measurement from y_top={y_top:.4f} to y_bottom={y_bottom:.4f}")

        # Try to use perspective-corrected method
        if 'backdrop_corners' in calibration_data:
            # Get image dimensions
            image_width, image_height = get_image_dimensions(image_path)

            # Calculate with perspective correction
            height_perspective = calculate_vertical_distance_perspective(
                y_top, y_bottom, x_avg,
                calibration_data,
                image_width, image_height
            )

            if height_perspective is not None:
                print(f"  Using perspective-corrected height: {height_perspective:.2f} cm")
                return height_perspective

        # Fallback to legacy method if perspective correction unavailable
        print(f"  WARNING: Using legacy height calculation (no perspective correction)")
        cm_per_normalized_unit = calibration_data['calibration_data']['cm_per_normalized_unit']
        return calculate_vertical_distance(y_top, y_bottom, cm_per_normalized_unit)

    except (KeyError, TypeError) as e:
        print(f"Error extracting height: {e}")
        return None


def extract_hair_length(
    hair_data: Dict,
    calibration_data: Dict,
    image_path: str
) -> Optional[float]:
    """
    Extract hair length measurement from hair segmentation data.

    Args:
        hair_data: Hair segmentation data containing hair_length information
        calibration_data: Calibration data with cm_per_normalized_unit
        image_path: Path to the image (for reference, not used for vertical measurements)

    Returns:
        Hair length in centimeters, or None if data is invalid
    """
    try:
        hair_info = hair_data.get('hair_length', {})
        y_top = hair_info.get('top', {}).get('y')
        y_bottom = hair_info.get('bottom', {}).get('y')

        if y_top is None or y_bottom is None:
            return None

        cm_per_normalized_unit = calibration_data['calibration_data']['cm_per_normalized_unit']

        return calculate_vertical_distance(y_top, y_bottom, cm_per_normalized_unit)

    except (KeyError, TypeError) as e:
        print(f"Error extracting hair length: {e}")
        return None


def extract_pose_measurements(
    pose_data: Dict,
    calibration_data: Dict,
    image_path: str
) -> Dict[str, Optional[float]]:
    """
    Extract all pose measurements from pose detection data.

    For bilateral measurements (left/right), returns the average.

    Args:
        pose_data: Pose detection data with landmark coordinates
        calibration_data: Calibration data with cm_per_normalized_unit
        image_path: Path to the image (needed for aspect ratio calculation)

    Returns:
        Dictionary with measurement names as keys and values in cm
    """
    measurements = {}

    try:
        cm_per_normalized_unit_y = calibration_data['calibration_data']['cm_per_normalized_unit']
        aspect_ratio = calculate_aspect_ratio(image_path)

        # Width measurements (single measurement, not bilateral)
        width_measurements = ['head_width', 'shoulder_width', 'hip_width']

        for measurement_name in width_measurements:
            if measurement_name in pose_data:
                landmarks = pose_data[measurement_name]
                # Get the two landmarks (should be exactly 2 for width measurements)
                landmark_keys = [k for k in landmarks.keys() if k.startswith('landmark_')]

                if len(landmark_keys) == 2:
                    landmark1 = landmarks[landmark_keys[0]]
                    landmark2 = landmarks[landmark_keys[1]]

                    distance = calculate_2d_distance(
                        landmark1, landmark2,
                        cm_per_normalized_unit_y,
                        aspect_ratio
                    )
                    measurements[measurement_name] = distance
                else:
                    measurements[measurement_name] = None

        # Bilateral measurements (left and right) - need to average
        bilateral_measurements = [
            'upper_arm_length',
            'forearm_length',
            'upper_leg_length',
            'lower_leg_length',
            'shoulder_to_waist'
        ]

        for measurement_name in bilateral_measurements:
            if measurement_name in pose_data:
                measurement_data = pose_data[measurement_name]

                left_distance = None
                right_distance = None

                # Extract left measurement
                if 'left' in measurement_data:
                    left_landmarks = measurement_data['left']
                    landmark_keys = [k for k in left_landmarks.keys() if k.startswith('landmark_')]

                    if len(landmark_keys) == 2:
                        landmark1 = left_landmarks[landmark_keys[0]]
                        landmark2 = left_landmarks[landmark_keys[1]]
                        left_distance = calculate_2d_distance(
                            landmark1, landmark2,
                            cm_per_normalized_unit_y,
                            aspect_ratio
                        )

                # Extract right measurement
                if 'right' in measurement_data:
                    right_landmarks = measurement_data['right']
                    landmark_keys = [k for k in right_landmarks.keys() if k.startswith('landmark_')]

                    if len(landmark_keys) == 2:
                        landmark1 = right_landmarks[landmark_keys[0]]
                        landmark2 = right_landmarks[landmark_keys[1]]
                        right_distance = calculate_2d_distance(
                            landmark1, landmark2,
                            cm_per_normalized_unit_y,
                            aspect_ratio
                        )

                # Average the left and right measurements
                if left_distance is not None and right_distance is not None:
                    measurements[measurement_name] = (left_distance + right_distance) / 2
                elif left_distance is not None:
                    measurements[measurement_name] = left_distance
                elif right_distance is not None:
                    measurements[measurement_name] = right_distance
                else:
                    measurements[measurement_name] = None

    except (KeyError, TypeError) as e:
        print(f"Error extracting pose measurements: {e}")

    return measurements


def extract_hand_measurements(
    hand_data: Dict,
    calibration_data: Dict,
    image_path: str
) -> Optional[float]:
    """
    Extract hand length measurement from hand detection data.

    If multiple hands are detected, returns the average hand length.

    Args:
        hand_data: Hand detection data with landmark coordinates
        calibration_data: Calibration data with cm_per_normalized_unit
        image_path: Path to the image (needed for aspect ratio calculation)

    Returns:
        Average hand length in centimeters, or None if data is invalid
    """
    try:
        cm_per_normalized_unit_y = calibration_data['calibration_data']['cm_per_normalized_unit']
        aspect_ratio = calculate_aspect_ratio(image_path)

        hands = hand_data.get('hands', [])
        if not hands:
            return None

        hand_lengths = []

        for hand in hands:
            hand_length_data = hand.get('hand_length', {})

            # Get the two landmarks for hand length
            landmark_keys = [k for k in hand_length_data.keys() if k.startswith('landmark_')]

            if len(landmark_keys) == 2:
                landmark1 = hand_length_data[landmark_keys[0]]
                landmark2 = hand_length_data[landmark_keys[1]]

                distance = calculate_2d_distance(
                    landmark1, landmark2,
                    cm_per_normalized_unit_y,
                    aspect_ratio
                )
                hand_lengths.append(distance)

        # Return average of all detected hands
        if hand_lengths:
            return sum(hand_lengths) / len(hand_lengths)
        else:
            return None

    except (KeyError, TypeError) as e:
        print(f"Error extracting hand measurements: {e}")
        return None


def extract_neck_length(
    pose_data: Dict,
    calibration_data: Dict,
    image_path: str
) -> Optional[float]:
    """
    Extract neck length measurement from pose detection data.

    Note: This function assumes neck_length data exists in pose_data.
    If not present in the current schema, this will need to be adjusted
    based on which landmarks define neck length.

    Args:
        pose_data: Pose detection data with landmark coordinates
        calibration_data: Calibration data with cm_per_normalized_unit
        image_path: Path to the image (needed for aspect ratio calculation)

    Returns:
        Neck length in centimeters, or None if data is invalid
    """
    try:
        cm_per_normalized_unit_y = calibration_data['calibration_data']['cm_per_normalized_unit']
        aspect_ratio = calculate_aspect_ratio(image_path)

        if 'neck_length' not in pose_data:
            return None

        neck_data = pose_data['neck_length']

        # Handle bilateral neck measurement if present
        if 'left' in neck_data and 'right' in neck_data:
            # Average left and right
            left_landmarks = neck_data['left']
            right_landmarks = neck_data['right']

            left_keys = [k for k in left_landmarks.keys() if k.startswith('landmark_')]
            right_keys = [k for k in right_landmarks.keys() if k.startswith('landmark_')]

            left_distance = None
            right_distance = None

            if len(left_keys) == 2:
                landmark1 = left_landmarks[left_keys[0]]
                landmark2 = left_landmarks[left_keys[1]]
                left_distance = calculate_2d_distance(
                    landmark1, landmark2,
                    cm_per_normalized_unit_y,
                    aspect_ratio
                )

            if len(right_keys) == 2:
                landmark1 = right_landmarks[right_keys[0]]
                landmark2 = right_landmarks[right_keys[1]]
                right_distance = calculate_2d_distance(
                    landmark1, landmark2,
                    cm_per_normalized_unit_y,
                    aspect_ratio
                )

            if left_distance and right_distance:
                return (left_distance + right_distance) / 2
            elif left_distance:
                return left_distance
            elif right_distance:
                return right_distance
            else:
                return None
        else:
            # Single neck measurement
            landmark_keys = [k for k in neck_data.keys() if k.startswith('landmark_')]

            if len(landmark_keys) == 2:
                landmark1 = neck_data[landmark_keys[0]]
                landmark2 = neck_data[landmark_keys[1]]

                return calculate_2d_distance(
                    landmark1, landmark2,
                    cm_per_normalized_unit_y,
                    aspect_ratio
                )
            else:
                return None

    except (KeyError, TypeError) as e:
        print(f"Error extracting neck length: {e}")
        return None
