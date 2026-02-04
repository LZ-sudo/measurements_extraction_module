"""
Head Width Detection Module using Google Mediapipe

This module detects head width landmarks (ears) using MediaPipe Pose Landmarker.
Focused specifically on landmarks 7 (left ear) and 8 (right ear) for head width measurement.

Other body measurements use RTMLib for higher accuracy.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from typing import Dict, Optional
import os


class HeadWidthDetector:
    """
    A class to detect head width landmarks using MediaPipe Pose.
    Focuses on landmarks 7 (left ear) and 8 (right ear).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the Head Width Detector.

        Args:
            model_path: Path to pose landmarker model file. If None, uses default.
            min_pose_detection_confidence: Minimum confidence for pose detection.
            min_pose_presence_confidence: Minimum confidence for pose presence.
            min_tracking_confidence: Minimum confidence for pose tracking.
        """
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Get default model if not provided
        if model_path is None:
            model_path = self._get_default_model()

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

    def _get_default_model(self) -> str:
        """Get the default pose landmarker model path."""
        # Check in weight_files directory first
        task_files_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "weight_files",
            "pose_landmarker_heavy.task"
        )

        if os.path.exists(task_files_path):
            return task_files_path

        # Fall back to current directory
        local_path = "pose_landmarker_heavy.task"
        if os.path.exists(local_path):
            return local_path

        raise FileNotFoundError(
            f"Pose landmarker model not found at {task_files_path}. "
            "Please download pose_landmarker_heavy.task from MediaPipe models."
        )

    def visualize(self, image: np.ndarray) -> np.ndarray:
        """
        Draw head width landmarks on the image for visualization.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Image with ear landmarks drawn on it.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect landmarks
        results = self.landmarker.detect(mp_image)

        # Create a copy for drawing
        annotated_image = image.copy()
        h, w = image.shape[:2]

        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]

            # Draw ear landmarks (7 and 8)
            ear_indices = [(7, "L_ear"), (8, "R_ear")]
            for idx, name in ear_indices:
                lm = pose_landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Draw filled circle
                cv2.circle(annotated_image, (cx, cy), 8, (0, 255, 0), -1)
                # Draw white outline
                cv2.circle(annotated_image, (cx, cy), 8, (255, 255, 255), 2)
                # Add label
                cv2.putText(annotated_image, f"{idx} ({name})", (cx + 12, cy - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw line between ears to show head width
            left_ear = pose_landmarks[7]
            right_ear = pose_landmarks[8]
            pt1 = (int(left_ear.x * w), int(left_ear.y * h))
            pt2 = (int(right_ear.x * w), int(right_ear.y * h))
            cv2.line(annotated_image, pt1, pt2, (255, 0, 255), 2)

            # Add head width label
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.putText(annotated_image, "Head Width", (mid_x - 40, mid_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        return annotated_image

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect head width landmarks in an image.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary containing head width landmark coordinates:
            {
                "head_width": {
                    "landmark_7": {"x": ..., "y": ..., "z": ...},
                    "landmark_8": {"x": ..., "y": ..., "z": ...}
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

        # Extract ear landmarks if detected
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks[0]

            def get_landmark_coords(idx: int) -> Dict:
                lm = pose_landmarks[idx]
                return {"x": lm.x, "y": lm.y, "z": lm.z}

            # Head width: landmarks 7 (left ear) and 8 (right ear)
            landmark_pairs["head_width"] = {
                "landmark_7": get_landmark_coords(7),
                "landmark_8": get_landmark_coords(8)
            }

        return landmark_pairs

    def close(self):
        """Release resources."""
        self.landmarker.close()


def detect_head_width(
    image_path: str,
    output_path: Optional[str] = None,
    visualize_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to detect head width landmarks in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        visualize_path: Optional path to save visualization image.

    Returns:
        Dictionary containing head width landmark coordinates.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = HeadWidthDetector()
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
        print(f"Head width coordinates saved to {output_path}")

    return landmark_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect head width landmarks using MediaPipe Pose"
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument("-v", "--visualize", type=str, help="Path to save visualization image with landmarks")

    args = parser.parse_args()

    # Detect head width landmarks
    landmark_pairs = detect_head_width(args.input_image, args.output, args.visualize)

    # Print the results
    print("\nHead Width Landmarks (MediaPipe):")
    print(json.dumps(landmark_pairs, indent=2))
