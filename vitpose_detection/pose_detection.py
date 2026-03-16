"""
Pose Detection Module using easyViTPose

This module provides pose detection using ViTPose models with 133 wholebody keypoints.
Output format is compatible with MediaPipe pose detection for seamless integration
with the existing measurement extraction pipeline.

ViTPose offers higher accuracy than RTMLib, especially for challenging poses
and clothing occlusion scenarios.
"""

import cv2
import numpy as np
import json
from typing import Dict, Optional

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
# Keypoint Mappings
# ============================================================================


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
    A class to detect pose landmarks using ViTPose models.
    Output format is compatible with MediaPipe for seamless integration.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_VITPOSE_MODEL,
        yolo_model: str = DEFAULT_YOLO_MODEL,
        device: Optional[str] = None,
        yolo_size: int = 320
    ):
        """
        Initialize the Pose Detector using ViTPose.

        Args:
            model_name: ViTPose model size - 's' (small), 'b' (base), 'l' (large), 'h' (huge).
                       Use 'h' for maximum accuracy.
            yolo_model: YOLO model for person detection ('yolov8s' or 'yolov8m').
            device: Device to run on - 'cpu', 'cuda', or 'mps'. None for auto-detect.
            yolo_size: YOLO input image size (larger = better detection, slower).
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

    def _get_landmark_coords(
        self,
        keypoints: np.ndarray,
        idx: int,
        image_height: int,
        image_width: int
    ) -> Dict[str, float]:
        """
        Extract normalized landmark coordinates from keypoints array.

        Note: easyViTPose returns keypoints as (y, x, score), so we swap x and y.

        Args:
            keypoints: Array of keypoint coordinates (N, 3) as (y, x, score).
            idx: Keypoint index.
            image_height: Image height for normalization.
            image_width: Image width for normalization.

        Returns:
            Dictionary with normalized x, y, z coordinates and visibility.
        """
        # easyViTPose format: (y, x, score)
        y_px, x_px, confidence = keypoints[idx]

        return {
            'x': float(x_px / image_width),
            'y': float(y_px / image_height),
            'z': 0.0,  # ViTPose doesn't provide z-coordinate
            'visibility': float(confidence)
        }

    def _select_main_person(self, keypoints_dict: dict, image_height: int, image_width: int) -> Optional[np.ndarray]:
        """
        Select the main person from detected people.

        Strategy: Select the person with the largest bounding box area,
        which typically corresponds to the person closest to the camera.

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
            # Calculate bounding box from keypoints
            # Filter out low-confidence keypoints
            valid_kpts = kpts[kpts[:, 2] > 0.3]
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

    def visualize(self, image: np.ndarray, kpt_thr: float = 0.3) -> np.ndarray:
        """
        Draw pose landmarks on the image for visualization.
        Uses ViTPose's built-in visualization with custom landmark labels.

        Args:
            image: Input image as numpy array (BGR format).
            kpt_thr: Minimum confidence threshold for drawing keypoints.

        Returns:
            Image with landmarks drawn on it.
        """
        # Convert BGR to RGB for ViTPose
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        keypoints_dict = self.model.inference(img_rgb)

        if not keypoints_dict:
            return image.copy()

        # Get the visualization from ViTPose
        annotated_image = self.model.draw(show_yolo=False)

        # Convert back to BGR
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Add custom labels for measurement-relevant keypoints
        h, w = image.shape[:2]
        person_kpts = self._select_main_person(keypoints_dict, h, w)

        if person_kpts is not None:
            # Measurement-relevant keypoints with their names
            # Format: (COCO-WholeBody index, MediaPipe index, name)
            measurement_keypoints = [
                (3, 7, "L_ear"),
                (4, 8, "R_ear"),
                (5, 11, "L_shoulder"),
                (6, 12, "R_shoulder"),
                (7, 13, "L_elbow"),
                (8, 14, "R_elbow"),
                (9, 15, "L_wrist"),
                (10, 16, "R_wrist"),
                (11, 23, "L_hip"),
                (12, 24, "R_hip"),
                (13, 25, "L_knee"),
                (14, 26, "R_knee"),
                (15, 27, "L_ankle"),
                (16, 28, "R_ankle"),
                (19, 29, "L_heel"),
                (22, 30, "R_heel"),
            ]

            for coco_idx, mp_idx, name in measurement_keypoints:
                # kpts format is (y, x, score)
                y_px, x_px, score = person_kpts[coco_idx]
                if score >= kpt_thr:
                    x, y = int(x_px), int(y_px)
                    # Draw filled circle
                    cv2.circle(annotated_image, (x, y), 6, (0, 255, 0), -1)
                    # Draw white outline
                    cv2.circle(annotated_image, (x, y), 6, (255, 255, 255), 1)
                    # Add label with MediaPipe index
                    cv2.putText(annotated_image, f"{mp_idx}", (x + 8, y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return annotated_image

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

        # Convert BGR to RGB for ViTPose
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        keypoints_dict = self.model.inference(img_rgb)

        # Initialize result dictionary
        landmark_pairs = {}

        if not keypoints_dict:
            return landmark_pairs

        # Select main person
        person_kpts = self._select_main_person(keypoints_dict, h, w)

        if person_kpts is None:
            return landmark_pairs

        # Helper function to get landmark coords
        def get_coords(idx: int) -> Dict[str, float]:
            return self._get_landmark_coords(person_kpts, idx, h, w)

        # Note: head_width uses MediaPipe pose detection for better accuracy
        # See mediapipe_detection/head_detection.py (HeadWidthDetector)

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
        # Reset model state if needed
        if hasattr(self.model, 'reset'):
            self.model.reset()


def detect_pose(
    image_path: str,
    output_path: Optional[str] = None,
    visualize_path: Optional[str] = None,
    model_name: str = DEFAULT_VITPOSE_MODEL,
    device: Optional[str] = None
) -> Dict:
    """
    Convenience function to detect pose landmarks in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        visualize_path: Optional path to save visualization image.
        model_name: ViTPose model size - 's', 'b', 'l', or 'h' (default: 'h' for max accuracy).
        device: Device to run on - 'cpu', 'cuda', 'mps', or None for auto-detect.

    Returns:
        Dictionary containing landmark pairs for body measurements.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = PoseDetector(model_name=model_name, device=device)
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
        description="Detect pose landmarks using ViTPose model"
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
        choices=["cpu", "cuda"],
        help="Device to run inference on (default: auto-detect)"
    )

    args = parser.parse_args()

    # Detect pose and extract landmark pairs
    landmark_pairs = detect_pose(
        args.input_image,
        args.output,
        args.visualize,
        args.model,
        args.device
    )

    # Print the results
    print("\nPose Landmark Pairs (ViTPose):")
    print(json.dumps(landmark_pairs, indent=2))
