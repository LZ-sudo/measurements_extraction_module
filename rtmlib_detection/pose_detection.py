"""
Pose Detection Module using RTMLib (RTMW)

This module provides pose detection using RTMLib's RTMW model with 133 wholebody keypoints.
Output format is compatible with MediaPipe pose detection for seamless integration.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# COCO-WholeBody to MediaPipe landmark mapping
# COCO-WholeBody indices -> MediaPipe indices
WHOLEBODY_TO_MEDIAPIPE = {
    # Ears
    3: 7,   # left_ear
    4: 8,   # right_ear
    # Shoulders
    5: 11,  # left_shoulder
    6: 12,  # right_shoulder
    # Elbows
    7: 13,  # left_elbow
    8: 14,  # right_elbow
    # Wrists
    9: 15,  # left_wrist
    10: 16, # right_wrist
    # Hips
    11: 23, # left_hip
    12: 24, # right_hip
    # Knees
    13: 25, # left_knee
    14: 26, # right_knee
    # Ankles
    15: 27, # left_ankle
    16: 28, # right_ankle
    # Heels (from foot keypoints)
    19: 29, # left_heel
    22: 30, # right_heel
}

# COCO-WholeBody keypoint names for reference
WHOLEBODY_KEYPOINT_NAMES = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle',
    17: 'left_big_toe', 18: 'left_small_toe', 19: 'left_heel',
    20: 'right_big_toe', 21: 'right_small_toe', 22: 'right_heel'
}


class PoseDetector:
    """
    A class to detect pose landmarks using RTMLib's RTMW model.
    Output format is compatible with MediaPipe for seamless integration.
    """

    def __init__(
        self,
        mode: str = 'performance',
        backend: str = 'onnxruntime',
        device: str = 'cpu',
        to_openpose: bool = False
    ):
        """
        Initialize the Pose Detector using RTMLib.

        Args:
            mode: Model mode - 'performance' (best accuracy), 'balanced', or 'lightweight'
            backend: Inference backend - 'onnxruntime', 'opencv', or 'openvino'
            device: Device to run on - 'cpu', 'cuda', or 'mps'
            to_openpose: Whether to use OpenPose-style output (False for mmpose-style)
        """
        from rtmlib import Wholebody

        self.wholebody = Wholebody(
            to_openpose=to_openpose,
            mode=mode,
            backend=backend,
            device=device
        )
        self.mode = mode
        self.device = device

    def _get_landmark_coords(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        idx: int,
        image_height: int,
        image_width: int
    ) -> Dict[str, float]:
        """
        Extract normalized landmark coordinates from keypoints array.

        Args:
            keypoints: Array of keypoint coordinates (N, 2) in pixel space
            scores: Array of confidence scores (N,)
            idx: Keypoint index
            image_height: Image height for normalization
            image_width: Image width for normalization

        Returns:
            Dictionary with normalized x, y, z coordinates and visibility
        """
        x_px, y_px = keypoints[idx]
        confidence = float(scores[idx]) if idx < len(scores) else 0.0

        return {
            'x': float(x_px / image_width),
            'y': float(y_px / image_height),
            'z': 0.0,  # RTMLib doesn't provide z-coordinate
            'visibility': confidence
        }

    def visualize(self, image: np.ndarray) -> np.ndarray:
        """
        Draw pose landmarks on the image for visualization.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Image with landmarks drawn on it.
        """
        from rtmlib import draw_skeleton

        keypoints, scores = self.wholebody(image)

        if keypoints is not None and len(keypoints) > 0:
            return draw_skeleton(image.copy(), keypoints, scores, kpt_thr=0.3)

        return image.copy()

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect pose landmarks in an image and extract measurement landmark pairs.
        Output format matches MediaPipe pose detection for compatibility.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary containing normalized landmark coordinates for measurements,
            matching MediaPipe's output format.
        """
        h, w = image.shape[:2]

        # Run wholebody detection
        keypoints, scores = self.wholebody(image)

        # Initialize result dictionary
        landmark_pairs = {}

        if keypoints is None or len(keypoints) == 0:
            return landmark_pairs

        # Get first person's keypoints (keypoints shape is (num_people, 133, 2))
        person_kpts = keypoints[0]
        person_scores = scores[0] if scores is not None and len(scores) > 0 else np.zeros(133)

        # Helper function to get landmark coords
        def get_coords(idx: int) -> Dict[str, float]:
            return self._get_landmark_coords(person_kpts, person_scores, idx, h, w)

        # Head width: COCO-WholeBody 3 (left ear) and 4 (right ear)
        # Maps to MediaPipe landmark_7 and landmark_8
        landmark_pairs["head_width"] = {
            "landmark_7": get_coords(3),   # left_ear
            "landmark_8": get_coords(4)    # right_ear
        }

        # Shoulder width: COCO-WholeBody 5 (left shoulder) and 6 (right shoulder)
        # Maps to MediaPipe landmark_11 and landmark_12
        landmark_pairs["shoulder_width"] = {
            "landmark_11": get_coords(5),  # left_shoulder
            "landmark_12": get_coords(6)   # right_shoulder
        }

        # Hip width: COCO-WholeBody 11 (left hip) and 12 (right hip)
        # Maps to MediaPipe landmark_23 and landmark_24
        landmark_pairs["hip_width"] = {
            "landmark_23": get_coords(11), # left_hip
            "landmark_24": get_coords(12)  # right_hip
        }

        # Upper arm length: shoulder to elbow
        # Left: 5->7 (COCO) = 11->13 (MediaPipe)
        # Right: 6->8 (COCO) = 12->14 (MediaPipe)
        landmark_pairs["upper_arm_length"] = {
            "left": {
                "landmark_11": get_coords(5),  # left_shoulder
                "landmark_13": get_coords(7)   # left_elbow
            },
            "right": {
                "landmark_12": get_coords(6),  # right_shoulder
                "landmark_14": get_coords(8)   # right_elbow
            }
        }

        # Forearm length: elbow to wrist
        # Left: 7->9 (COCO) = 13->15 (MediaPipe)
        # Right: 8->10 (COCO) = 14->16 (MediaPipe)
        landmark_pairs["forearm_length"] = {
            "left": {
                "landmark_13": get_coords(7),  # left_elbow
                "landmark_15": get_coords(9)   # left_wrist
            },
            "right": {
                "landmark_14": get_coords(8),  # right_elbow
                "landmark_16": get_coords(10)  # right_wrist
            }
        }

        # Upper leg length: hip to knee
        # Left: 11->13 (COCO) = 23->25 (MediaPipe)
        # Right: 12->14 (COCO) = 24->26 (MediaPipe)
        landmark_pairs["upper_leg_length"] = {
            "left": {
                "landmark_23": get_coords(11), # left_hip
                "landmark_25": get_coords(13)  # left_knee
            },
            "right": {
                "landmark_24": get_coords(12), # right_hip
                "landmark_26": get_coords(14)  # right_knee
            }
        }

        # Lower leg length: knee to ankle
        # Left: 13->15 (COCO) = 25->27 (MediaPipe)
        # Right: 14->16 (COCO) = 26->28 (MediaPipe)
        landmark_pairs["lower_leg_length"] = {
            "left": {
                "landmark_25": get_coords(13), # left_knee
                "landmark_27": get_coords(15)  # left_ankle
            },
            "right": {
                "landmark_26": get_coords(14), # right_knee
                "landmark_28": get_coords(16)  # right_ankle
            }
        }

        # Shoulder to waist: shoulder to hip
        # Left: 5->11 (COCO) = 11->23 (MediaPipe)
        # Right: 6->12 (COCO) = 12->24 (MediaPipe)
        landmark_pairs["shoulder_to_waist"] = {
            "left": {
                "landmark_11": get_coords(5),  # left_shoulder
                "landmark_23": get_coords(11)  # left_hip
            },
            "right": {
                "landmark_12": get_coords(6),  # right_shoulder
                "landmark_24": get_coords(12)  # right_hip
            }
        }

        # Heel landmarks for height measurement
        # COCO-WholeBody 19 (left heel) and 22 (right heel)
        # Maps to MediaPipe landmark_29 and landmark_30
        landmark_pairs["heel_landmarks"] = {
            "left": get_coords(19),   # left_heel
            "right": get_coords(22)   # right_heel
        }

        return landmark_pairs

    def close(self):
        """Release resources."""
        # RTMLib doesn't require explicit cleanup
        pass


def detect_pose(
    image_path: str,
    output_path: Optional[str] = None,
    visualize_path: Optional[str] = None,
    mode: str = 'performance',
    device: str = 'cpu'
) -> Dict:
    """
    Convenience function to detect pose landmarks in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        visualize_path: Optional path to save visualization image.
        mode: Model mode - 'performance', 'balanced', or 'lightweight'
        device: Device to run on - 'cpu', 'cuda', or 'mps'

    Returns:
        Dictionary containing landmark pairs for body measurements.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = PoseDetector(mode=mode, device=device)
    landmark_pairs = detector.detect(image)

    # Generate visualization if requested
    if visualize_path:
        annotated_image = detector.visualize(image)
        cv2.imwrite(visualize_path, annotated_image)
        print(f"Visualization saved to {visualize_path}")

    detector.close()

    # Save output if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(landmark_pairs, f, indent=2)
        print(f"Landmark coordinates saved to {output_path}")

    return landmark_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect pose landmarks using RTMLib RTMW model"
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

    # Detect pose and extract landmark pairs
    landmark_pairs = detect_pose(
        args.input_image,
        args.output,
        args.visualize,
        args.mode,
        args.device
    )

    # Print the results
    print("\nPose Landmark Pairs (RTMLib RTMW):")
    print(json.dumps(landmark_pairs, indent=2))
