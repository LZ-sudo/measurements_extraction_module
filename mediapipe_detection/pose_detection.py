"""
Pose Detection Module using Google Mediapipe

This script detects pose landmarks in an image using Google's Mediapipe Pose solution.
It extracts specific landmark pairs needed for body measurements and returns them as JSON.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from typing import Dict, Optional


class PoseDetector:
    """
    A class to detect and visualize pose landmarks using Mediapipe.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the Pose Detector.

        Args:
            model_path: Path to pose landmarker model file. If None, downloads default model.
            min_pose_detection_confidence: Minimum confidence for pose detection.
            min_pose_presence_confidence: Minimum confidence for pose presence.
            min_tracking_confidence: Minimum confidence for pose tracking.
        """
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Download default model if not provided
        if model_path is None:
            model_path = self._download_default_model()

        # Create pose landmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _download_default_model(self) -> str:
        """Get the default pose landmarker model path."""
        import os

        # Check in mediapipe_task_files directory first
        task_files_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mediapipe_task_files", "pose_landmarker_heavy.task")

        if os.path.exists(task_files_path):
            return task_files_path

        # Fall back to current directory
        local_path = "pose_landmarker_heavy.task"
        if os.path.exists(local_path):
            return local_path

        # If not found, download it
        import urllib.request
        model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

        print(f"Downloading pose landmarker model...")
        urllib.request.urlretrieve(model_url, local_path)
        print(f"Model downloaded to {local_path}")

        return local_path

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect pose landmarks in an image and extract measurement landmark pairs.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary containing normalized landmark coordinates for measurements:
            {
                "head_width": {"landmark_7": {"x": ..., "y": ..., "z": ...}, "landmark_8": {...}},
                "shoulder_width": {"landmark_11": {...}, "landmark_12": {...}},
                "hip_width": {"landmark_23": {...}, "landmark_24": {...}},
                "upper_arm_length": {
                    "left": {"landmark_11": {...}, "landmark_13": {...}},
                    "right": {"landmark_12": {...}, "landmark_14": {...}}
                },
                "forearm_length": {
                    "left": {"landmark_13": {...}, "landmark_15": {...}},
                    "right": {"landmark_14": {...}, "landmark_16": {...}}
                }
            }
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Process the image
        results = self.landmarker.detect(mp_image)

        # Initialize result dictionary
        landmark_pairs = {}

        # Extract landmark pairs if detected
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]  # Get first person's landmarks

            # Helper function to extract landmark coordinates
            def get_landmark_coords(idx: int) -> Dict:
                lm = pose_landmarks[idx]
                return {"x": lm.x, "y": lm.y, "z": lm.z}

            # Head width: landmarks 7 (left ear) and 8 (right ear)
            landmark_pairs["head_width"] = {
                "landmark_7": get_landmark_coords(7),
                "landmark_8": get_landmark_coords(8)
            }

            # Shoulder width: landmarks 11 (left shoulder) and 12 (right shoulder)
            landmark_pairs["shoulder_width"] = {
                "landmark_11": get_landmark_coords(11),
                "landmark_12": get_landmark_coords(12)
            }

            # Hip width: landmarks 23 (left hip) and 24 (right hip)
            landmark_pairs["hip_width"] = {
                "landmark_23": get_landmark_coords(23),
                "landmark_24": get_landmark_coords(24)
            }

            # Upper arm length: 11 to 13 (left) and 12 to 14 (right)
            landmark_pairs["upper_arm_length"] = {
                "left": {
                    "landmark_11": get_landmark_coords(11),  # Left shoulder
                    "landmark_13": get_landmark_coords(13)   # Left elbow
                },
                "right": {
                    "landmark_12": get_landmark_coords(12),  # Right shoulder
                    "landmark_14": get_landmark_coords(14)   # Right elbow
                }
            }

            # Forearm length: 13 to 15 (left) and 14 to 16 (right)
            landmark_pairs["forearm_length"] = {
                "left": {
                    "landmark_13": get_landmark_coords(13),  # Left elbow
                    "landmark_15": get_landmark_coords(15)   # Left wrist
                },
                "right": {
                    "landmark_14": get_landmark_coords(14),  # Right elbow
                    "landmark_16": get_landmark_coords(16)   # Right wrist
                }
            }

        return landmark_pairs

    def close(self):
        """Release resources."""
        self.landmarker.close()


def detect_pose(image_path: str, output_path: Optional[str] = None) -> Dict:
    """
    Convenience function to detect pose landmarks in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.

    Returns:
        Dictionary containing landmark pairs for body measurements.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = PoseDetector()
    landmark_pairs = detector.detect(image)
    detector.close()

    # Save output if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(landmark_pairs, f, indent=2)
        print(f"Landmark coordinates saved to {output_path}")

    return landmark_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect pose landmarks in an image and extract measurement pairs")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")

    args = parser.parse_args()

    # Detect pose and extract landmark pairs
    landmark_pairs = detect_pose(args.input_image, args.output)

    # Print the results
    print("\nPose Landmark Pairs:")
    print(json.dumps(landmark_pairs, indent=2))
