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
from typing import Dict, Tuple, Optional, List
from pathlib import Path


def create_perspective_transform(calibration_data: Dict) -> Optional[np.ndarray]:
    """
    Create homography matrix for perspective transformation from backdrop calibration.

    This transforms image coordinates to real-world backdrop coordinates (in cm),
    correcting for perspective distortion and camera angle.

    Args:
        calibration_data: Calibration data containing backdrop_corners

    Returns:
        3x3 homography matrix, or None if corners not available
    """
    if 'backdrop_corners' not in calibration_data:
        return None

    # Detected corners in image (pixels)
    corners_image = np.array(calibration_data['backdrop_corners'], dtype=np.float32)

    # Corresponding physical positions on backdrop (cm)
    # Backdrop coordinate system: origin at bottom-left
    # Top-left corner = (0cm, 200cm) - at 200cm height line
    # Top-right corner = (200cm, 200cm) - at 200cm height line
    # Bottom-right corner = (200cm, 10cm) - at 10cm height line
    # Bottom-left corner = (0cm, 10cm) - at 10cm height line
    corners_world = np.array([
        [0, 200],      # Top-left: 200cm height, 0cm from left
        [200, 200],    # Top-right: 200cm height, 200cm from left
        [200, 10],     # Bottom-right: 10cm height, 200cm from left
        [0, 10]        # Bottom-left: 10cm height, 0cm from left
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

    This function now uses perspective transformation to correct for camera angle
    and floor inclusion. It prefers pose ankle landmarks over segmentation bottom
    to avoid measuring to the floor.

    Args:
        body_data: Body detection data containing height information
        calibration_data: Calibration data with backdrop_corners and cm_per_normalized_unit
        image_path: Path to the image (needed for dimensions)
        pose_data: Optional pose detection data with ankle landmarks for feet position

    Returns:
        Height in centimeters, or None if data is invalid
    """
    try:
        height_info = body_data.get('height', {})
        y_top = height_info.get('top', {}).get('y')
        y_bottom_seg = height_info.get('bottom', {}).get('y')

        if y_top is None or y_bottom_seg is None:
            return None

        # Try to get more accurate foot position from pose ankles (landmarks 27, 28)
        y_bottom = y_bottom_seg  # Default to segmentation
        x_bottom = 0.5  # Default to center

        if pose_data and 'lower_leg_length' in pose_data:
            # Get ankle landmarks (more accurate than segmentation bottom)
            left_ankle = pose_data['lower_leg_length'].get('left', {}).get('landmark_27')
            right_ankle = pose_data['lower_leg_length'].get('right', {}).get('landmark_28')

            if left_ankle and right_ankle:
                # Use average of both ankles for more robust measurement
                y_bottom = (left_ankle['y'] + right_ankle['y']) / 2
                x_bottom = (left_ankle['x'] + right_ankle['x']) / 2
                print(f"  Using pose ankle landmarks for feet position (y={y_bottom:.4f})")
                print(f"  Segmentation bottom was at y={y_bottom_seg:.4f} (floor included)")
            else:
                print(f"  WARNING: Ankle landmarks incomplete, using segmentation bottom")
        else:
            print(f"  WARNING: No pose data, using segmentation bottom (may include floor)")

        # Get x-coordinate for top
        x_top = height_info.get('top', {}).get('x', 0.5)
        x_avg = (x_top + x_bottom) / 2

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
            'lower_leg_length'
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
