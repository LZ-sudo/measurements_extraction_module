"""
Hand Landmarks Detection Module using easyViTPose

This module provides hand detection using ViTPose wholebody models.
Hand keypoints are extracted from the 133 wholebody keypoints (indices 91-132).
Output format is compatible with MediaPipe hand detection for seamless integration.
"""

import cv2
import numpy as np
import json
from typing import Dict, Optional, List

# Import model configuration - handles both package import and standalone script
try:
    from .config_model import (
        get_model_paths,
        DEFAULT_VITPOSE_MODEL,
        DEFAULT_YOLO_MODEL,
    )
except ImportError:
    # Running as standalone script
    from config_model import (
        get_model_paths,
        DEFAULT_VITPOSE_MODEL,
        DEFAULT_YOLO_MODEL,
    )


# ============================================================================
# Hand Keypoint Configuration
# ============================================================================

# COCO-WholeBody hand keypoint indices
# Left hand: 91-111 (21 keypoints)
# Right hand: 112-132 (21 keypoints)
LEFT_HAND_START = 91
RIGHT_HAND_START = 112
HAND_KEYPOINT_COUNT = 21

# Hand keypoint names (same order for both hands)
HAND_KEYPOINT_NAMES = [
    'wrist',           # 0
    'thumb_cmc',       # 1
    'thumb_mcp',       # 2
    'thumb_ip',        # 3
    'thumb_tip',       # 4
    'index_mcp',       # 5
    'index_pip',       # 6
    'index_dip',       # 7
    'index_tip',       # 8
    'middle_mcp',      # 9
    'middle_pip',      # 10
    'middle_dip',      # 11
    'middle_tip',      # 12
    'ring_mcp',        # 13
    'ring_pip',        # 14
    'ring_dip',        # 15
    'ring_tip',        # 16
    'pinky_mcp',       # 17
    'pinky_pip',       # 18
    'pinky_dip',       # 19
    'pinky_tip',       # 20
]


class HandDetector:
    """
    A class to detect hand landmarks using ViTPose wholebody models.
    Output format is compatible with MediaPipe for seamless integration.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VITPOSE_MODEL,
        yolo_model: str = DEFAULT_YOLO_MODEL,
        device: Optional[str] = None,
        yolo_size: int = 320,
        min_hand_confidence: float = 0.3
    ):
        """
        Initialize the Hand Detector using ViTPose.

        Args:
            model_name: ViTPose model size - 's', 'b', 'l', or 'h'.
                       Use 'h' for maximum accuracy.
            yolo_model: YOLO model for person detection.
            device: Device to run on - 'cpu', 'cuda', 'mps', or None for auto-detect.
            yolo_size: YOLO input image size.
            min_hand_confidence: Minimum average confidence to consider a hand detected.
        """
        from easy_ViTPose import VitInference

        # Get model paths (downloads if necessary)
        vitpose_path, yolo_path = get_model_paths(model_name, yolo_model)

        # Initialize ViTPose model
        # Note: model_path and yolo_path are positional arguments
        self.model = VitInference(
            vitpose_path,
            yolo_path,
            model_name=model_name,
            yolo_size=yolo_size,
            is_video=False,
            device=device
        )
        self.model_name = model_name
        self.device = device
        self.min_hand_confidence = min_hand_confidence

    def _select_main_person(self, keypoints_dict: dict, image_height: int, image_width: int) -> Optional[np.ndarray]:
        """
        Select the main person from detected people based on bounding box area.

        Args:
            keypoints_dict: Dictionary from ViTPose {person_id: keypoints}.
            image_height: Image height.
            image_width: Image width.

        Returns:
            Keypoints array for the main person, or None if no person detected.
        """
        if not keypoints_dict:
            return None

        best_person = None
        best_area = 0

        for person_id, kpts in keypoints_dict.items():
            # Calculate bounding box from body keypoints (first 23 keypoints)
            body_kpts = kpts[:23]
            valid_kpts = body_kpts[body_kpts[:, 2] > 0.3]
            if len(valid_kpts) < 5:
                continue

            # kpts format is (y, x, score)
            y_coords = valid_kpts[:, 0]
            x_coords = valid_kpts[:, 1]

            min_x, max_x = x_coords.min(), x_coords.max()
            min_y, max_y = y_coords.min(), y_coords.max()

            area = (max_x - min_x) * (max_y - min_y)

            if area > best_area:
                best_area = area
                best_person = kpts

        return best_person

    def _get_hand_keypoints(
        self,
        keypoints: np.ndarray,
        start_idx: int,
        image_height: int,
        image_width: int
    ) -> List[Dict[str, float]]:
        """
        Extract hand keypoints from wholebody keypoints array.

        Note: easyViTPose returns keypoints as (y, x, score), so we swap x and y.

        Args:
            keypoints: Array of all 133 keypoint coordinates as (y, x, score).
            start_idx: Starting index for hand (91 for left, 112 for right).
            image_height: Image height for normalization.
            image_width: Image width for normalization.

        Returns:
            List of 21 hand landmarks with normalized coordinates.
        """
        hand_landmarks = []
        for i in range(HAND_KEYPOINT_COUNT):
            idx = start_idx + i
            # easyViTPose format: (y, x, score)
            y_px, x_px, confidence = keypoints[idx]

            hand_landmarks.append({
                'x': float(x_px / image_width),
                'y': float(y_px / image_height),
                'z': 0.0,
                'visibility': float(confidence)
            })

        return hand_landmarks

    def _get_hand_confidence(self, keypoints: np.ndarray, start_idx: int) -> float:
        """
        Calculate average confidence for a hand.

        Args:
            keypoints: Array of keypoints (y, x, score).
            start_idx: Starting index for hand (91 for left, 112 for right).

        Returns:
            Average confidence score for the hand.
        """
        hand_scores = keypoints[start_idx:start_idx + HAND_KEYPOINT_COUNT, 2]
        return float(np.mean(hand_scores))

    def _is_hand_detected(self, keypoints: np.ndarray, start_idx: int) -> bool:
        """
        Check if a hand is detected based on confidence threshold.

        Args:
            keypoints: Array of keypoints.
            start_idx: Starting index for hand.

        Returns:
            True if hand is detected with sufficient confidence.
        """
        avg_confidence = self._get_hand_confidence(keypoints, start_idx)
        return avg_confidence >= self.min_hand_confidence

    def visualize(self, image: np.ndarray, kpt_thr: float = 0.3) -> np.ndarray:
        """
        Draw hand landmarks on the image for visualization.
        Only draws hand keypoints and connections, not the full body skeleton.

        Args:
            image: Input image as numpy array (BGR format).
            kpt_thr: Minimum confidence threshold for drawing keypoints.

        Returns:
            Image with hand landmarks drawn on it.
        """
        h, w = image.shape[:2]
        annotated_image = image.copy()

        # Convert BGR to RGB for ViTPose
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        keypoints_dict = self.model.inference(img_rgb)

        if not keypoints_dict:
            return annotated_image

        person_kpts = self._select_main_person(keypoints_dict, h, w)

        if person_kpts is None:
            return annotated_image

        # Hand connection pairs (finger bones)
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),    # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm connections
        ]

        # Colors for each hand
        left_color = (255, 128, 0)   # Orange for left hand
        right_color = (0, 128, 255)  # Blue for right hand

        # Draw left hand if detected
        if self._is_hand_detected(person_kpts, LEFT_HAND_START):
            self._draw_hand(
                annotated_image, person_kpts,
                LEFT_HAND_START, hand_connections, left_color, kpt_thr
            )
            # Add label - get wrist position (y, x, score)
            y_px, x_px, _ = person_kpts[LEFT_HAND_START]
            cv2.putText(annotated_image, "Left", (int(x_px) - 30, int(y_px) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)

        # Draw right hand if detected
        if self._is_hand_detected(person_kpts, RIGHT_HAND_START):
            self._draw_hand(
                annotated_image, person_kpts,
                RIGHT_HAND_START, hand_connections, right_color, kpt_thr
            )
            # Add label
            y_px, x_px, _ = person_kpts[RIGHT_HAND_START]
            cv2.putText(annotated_image, "Right", (int(x_px) - 30, int(y_px) - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)

        return annotated_image

    def _draw_hand(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        start_idx: int,
        connections: list,
        color: tuple,
        kpt_thr: float
    ):
        """
        Draw a single hand's landmarks and connections.

        Args:
            image: Image to draw on (modified in place).
            keypoints: All keypoints array (y, x, score).
            start_idx: Starting index for this hand (91 for left, 112 for right).
            connections: List of (start, end) tuples for bone connections.
            color: BGR color tuple for this hand.
            kpt_thr: Minimum confidence threshold.
        """
        # Draw connections (bones)
        for start, end in connections:
            idx1 = start_idx + start
            idx2 = start_idx + end

            # keypoints format: (y, x, score)
            if keypoints[idx1, 2] >= kpt_thr and keypoints[idx2, 2] >= kpt_thr:
                pt1 = (int(keypoints[idx1, 1]), int(keypoints[idx1, 0]))  # (x, y)
                pt2 = (int(keypoints[idx2, 1]), int(keypoints[idx2, 0]))  # (x, y)
                cv2.line(image, pt1, pt2, color, 2)

        # Draw keypoints as circles
        for i in range(HAND_KEYPOINT_COUNT):
            idx = start_idx + i
            # keypoints format: (y, x, score)
            if keypoints[idx, 2] >= kpt_thr:
                x, y = int(keypoints[idx, 1]), int(keypoints[idx, 0])
                # Larger circle for wrist and fingertips
                radius = 5 if i in [0, 4, 8, 12, 16, 20] else 3
                cv2.circle(image, (x, y), radius, color, -1)
                cv2.circle(image, (x, y), radius, (255, 255, 255), 1)  # White outline

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect hand landmarks in an image and extract measurement landmark pairs.
        Output format matches MediaPipe hand detection for compatibility.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary containing hand landmarks matching MediaPipe's output format:
            {
                "hands": [
                    {
                        "handedness": "Left" or "Right",
                        "confidence": 0.95,
                        "hand_length": {
                            "landmark_0": {"x": ..., "y": ..., "z": ...},
                            "landmark_12": {"x": ..., "y": ..., "z": ...}
                        }
                    },
                    ...
                ]
            }
        """
        h, w = image.shape[:2]

        # Convert BGR to RGB for ViTPose
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        keypoints_dict = self.model.inference(img_rgb)

        # Initialize result dictionary
        hand_data = {"hands": []}

        if not keypoints_dict:
            return hand_data

        # Select main person
        person_kpts = self._select_main_person(keypoints_dict, h, w)

        if person_kpts is None:
            return hand_data

        # Check and extract left hand
        if self._is_hand_detected(person_kpts, LEFT_HAND_START):
            left_landmarks = self._get_hand_keypoints(
                person_kpts, LEFT_HAND_START, h, w
            )
            confidence = self._get_hand_confidence(person_kpts, LEFT_HAND_START)

            # Hand length measurement: wrist (0) to middle finger tip (12)
            hand_info = {
                "handedness": "Left",
                "confidence": confidence,
                "hand_length": {
                    "landmark_0": left_landmarks[0],   # Wrist
                    "landmark_12": left_landmarks[12]  # Middle finger tip
                }
            }
            hand_data["hands"].append(hand_info)

        # Check and extract right hand
        if self._is_hand_detected(person_kpts, RIGHT_HAND_START):
            right_landmarks = self._get_hand_keypoints(
                person_kpts, RIGHT_HAND_START, h, w
            )
            confidence = self._get_hand_confidence(person_kpts, RIGHT_HAND_START)

            # Hand length measurement: wrist (0) to middle finger tip (12)
            hand_info = {
                "handedness": "Right",
                "confidence": confidence,
                "hand_length": {
                    "landmark_0": right_landmarks[0],   # Wrist
                    "landmark_12": right_landmarks[12]  # Middle finger tip
                }
            }
            hand_data["hands"].append(hand_info)

        return hand_data

    def close(self):
        """Release resources."""
        if hasattr(self.model, 'reset'):
            self.model.reset()


def detect_hands(
    image_path: str,
    output_path: Optional[str] = None,
    visualize_path: Optional[str] = None,
    model_name: str = DEFAULT_VITPOSE_MODEL,
    device: Optional[str] = None
) -> Dict:
    """
    Convenience function to detect hand landmarks in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        visualize_path: Optional path to save visualization image.
        model_name: ViTPose model size - 's', 'b', 'l', or 'h'.
        device: Device to run on - 'cpu', 'cuda', 'mps', or None for auto-detect.

    Returns:
        Dictionary containing hand landmarks for measurements.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = HandDetector(model_name=model_name, device=device)
    hand_data = detector.detect(image)

    # Generate visualization if requested
    if visualize_path:
        annotated_image = detector.visualize(image)
        cv2.imwrite(visualize_path, annotated_image)
        print(f"Visualization saved to {visualize_path}")

    detector.close()

    # Save output if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(hand_data, f, indent=2)
        print(f"Hand landmark coordinates saved to {output_path}")

    return hand_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect hand landmarks using ViTPose model"
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument(
        "-v", "--visualize", type=str,
        help="Path to save visualization image with landmarks"
    )
    parser.add_argument(
        "--model", type=str, default="h",
        choices=["s", "b", "l", "h"],
        help="ViTPose model size (default: h for maximum accuracy)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto-detect)"
    )

    args = parser.parse_args()

    # Detect hands and extract landmark pairs
    hand_data = detect_hands(
        args.input_image,
        args.output,
        args.visualize,
        args.model,
        args.device
    )

    # Print the results
    print("\nHand Landmarks (ViTPose):")
    print(json.dumps(hand_data, indent=2))
