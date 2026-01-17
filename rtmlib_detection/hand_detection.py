"""
Hand Landmarks Detection Module using RTMLib (RTMW)

This module provides hand detection using RTMLib's RTMW wholebody model.
Hand keypoints are extracted from the 133 wholebody keypoints (indices 91-132).
Output format is compatible with MediaPipe hand detection for seamless integration.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, List

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
    A class to detect hand landmarks using RTMLib's RTMW wholebody model.
    Output format is compatible with MediaPipe for seamless integration.
    """

    def __init__(
        self,
        mode: str = 'performance',
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        min_hand_confidence: float = 0.3
    ):
        """
        Initialize the Hand Detector using RTMLib.

        Args:
            mode: Model mode - 'performance' (best accuracy), 'balanced', or 'lightweight'
            backend: Inference backend - 'onnxruntime', 'opencv', or 'openvino'
            device: Device to run on - 'cpu', 'cuda', or 'mps'
            min_hand_confidence: Minimum average confidence to consider a hand detected
        """
        from rtmlib import Wholebody

        self.wholebody = Wholebody(
            to_openpose=False,
            mode=mode,
            backend=backend,
            device=device
        )
        self.mode = mode
        self.device = device
        self.min_hand_confidence = min_hand_confidence

    def _get_hand_keypoints(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        start_idx: int,
        image_height: int,
        image_width: int
    ) -> List[Dict[str, float]]:
        """
        Extract hand keypoints from wholebody keypoints array.

        Args:
            keypoints: Array of all 133 keypoint coordinates
            scores: Array of confidence scores
            start_idx: Starting index for hand (91 for left, 112 for right)
            image_height: Image height for normalization
            image_width: Image width for normalization

        Returns:
            List of 21 hand landmarks with normalized coordinates
        """
        hand_landmarks = []
        for i in range(HAND_KEYPOINT_COUNT):
            idx = start_idx + i
            x_px, y_px = keypoints[idx]
            confidence = float(scores[idx]) if idx < len(scores) else 0.0

            hand_landmarks.append({
                'x': float(x_px / image_width),
                'y': float(y_px / image_height),
                'z': 0.0,
                'visibility': confidence
            })

        return hand_landmarks

    def _get_hand_confidence(self, scores: np.ndarray, start_idx: int) -> float:
        """
        Calculate average confidence for a hand.

        Args:
            scores: Array of confidence scores
            start_idx: Starting index for hand (91 for left, 112 for right)

        Returns:
            Average confidence score for the hand
        """
        hand_scores = scores[start_idx:start_idx + HAND_KEYPOINT_COUNT]
        return float(np.mean(hand_scores))

    def _is_hand_detected(self, scores: np.ndarray, start_idx: int) -> bool:
        """
        Check if a hand is detected based on confidence threshold.

        Args:
            scores: Array of confidence scores
            start_idx: Starting index for hand

        Returns:
            True if hand is detected with sufficient confidence
        """
        avg_confidence = self._get_hand_confidence(scores, start_idx)
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

        keypoints, scores = self.wholebody(image)

        if keypoints is None or len(keypoints) == 0:
            return annotated_image

        person_kpts = keypoints[0]
        person_scores = scores[0] if scores is not None and len(scores) > 0 else np.zeros(133)

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
        if self._is_hand_detected(person_scores, LEFT_HAND_START):
            self._draw_hand(
                annotated_image, person_kpts, person_scores,
                LEFT_HAND_START, hand_connections, left_color, kpt_thr
            )
            # Add label
            wrist_idx = LEFT_HAND_START
            wrist_x, wrist_y = int(person_kpts[wrist_idx][0]), int(person_kpts[wrist_idx][1])
            cv2.putText(annotated_image, "Left", (wrist_x - 30, wrist_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)

        # Draw right hand if detected
        if self._is_hand_detected(person_scores, RIGHT_HAND_START):
            self._draw_hand(
                annotated_image, person_kpts, person_scores,
                RIGHT_HAND_START, hand_connections, right_color, kpt_thr
            )
            # Add label
            wrist_idx = RIGHT_HAND_START
            wrist_x, wrist_y = int(person_kpts[wrist_idx][0]), int(person_kpts[wrist_idx][1])
            cv2.putText(annotated_image, "Right", (wrist_x - 30, wrist_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)

        return annotated_image

    def _draw_hand(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        scores: np.ndarray,
        start_idx: int,
        connections: list,
        color: tuple,
        kpt_thr: float
    ):
        """
        Draw a single hand's landmarks and connections.

        Args:
            image: Image to draw on (modified in place).
            keypoints: All keypoints array.
            scores: All scores array.
            start_idx: Starting index for this hand (91 for left, 112 for right).
            connections: List of (start, end) tuples for bone connections.
            color: BGR color tuple for this hand.
            kpt_thr: Minimum confidence threshold.
        """
        # Draw connections (bones)
        for start, end in connections:
            idx1 = start_idx + start
            idx2 = start_idx + end

            if scores[idx1] >= kpt_thr and scores[idx2] >= kpt_thr:
                pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                cv2.line(image, pt1, pt2, color, 2)

        # Draw keypoints as circles
        for i in range(HAND_KEYPOINT_COUNT):
            idx = start_idx + i
            if scores[idx] >= kpt_thr:
                x, y = int(keypoints[idx][0]), int(keypoints[idx][1])
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

        # Run wholebody detection
        keypoints, scores = self.wholebody(image)

        # Initialize result dictionary
        hand_data = {"hands": []}

        if keypoints is None or len(keypoints) == 0:
            return hand_data

        # Get first person's keypoints
        person_kpts = keypoints[0]
        person_scores = scores[0] if scores is not None and len(scores) > 0 else np.zeros(133)

        # Check and extract left hand
        if self._is_hand_detected(person_scores, LEFT_HAND_START):
            left_landmarks = self._get_hand_keypoints(
                person_kpts, person_scores, LEFT_HAND_START, h, w
            )
            confidence = self._get_hand_confidence(person_scores, LEFT_HAND_START)

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
        if self._is_hand_detected(person_scores, RIGHT_HAND_START):
            right_landmarks = self._get_hand_keypoints(
                person_kpts, person_scores, RIGHT_HAND_START, h, w
            )
            confidence = self._get_hand_confidence(person_scores, RIGHT_HAND_START)

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
        # RTMLib doesn't require explicit cleanup
        pass


def detect_hands(
    image_path: str,
    output_path: Optional[str] = None,
    visualize_path: Optional[str] = None,
    mode: str = 'performance',
    device: str = 'cpu'
) -> Dict:
    """
    Convenience function to detect hand landmarks in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        visualize_path: Optional path to save visualization image.
        mode: Model mode - 'performance', 'balanced', or 'lightweight'
        device: Device to run on - 'cpu', 'cuda', or 'mps'

    Returns:
        Dictionary containing hand landmarks for measurements.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = HandDetector(mode=mode, device=device)
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
        description="Detect hand landmarks using RTMLib RTMW model"
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument(
        "-v", "--visualize", type=str,
        help="Path to save visualization image with landmarks"
    )
    parser.add_argument(
        "--mode", type=str, default="performance",
        choices=["performance", "balanced", "lightweight"],
        help="Model mode (default: performance for best accuracy)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu)"
    )

    args = parser.parse_args()

    # Detect hands and extract landmark pairs
    hand_data = detect_hands(
        args.input_image,
        args.output,
        args.visualize,
        args.mode,
        args.device
    )

    # Print the results
    print("\nHand Landmarks (RTMLib RTMW):")
    print(json.dumps(hand_data, indent=2))
