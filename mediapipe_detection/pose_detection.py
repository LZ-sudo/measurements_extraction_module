"""
Pose Detection Module using Google Mediapipe

This script detects pose landmarks in an image using Google's Mediapipe Pose solution.
It takes an input image, processes it to detect pose landmarks, draws the landmarks
and connections on the image, and returns the annotated image.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


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

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """
        Detect pose landmarks in an image and draw them.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Tuple of (annotated_image, pose_landmarks)
            - annotated_image: Image with pose landmarks drawn
            - pose_landmarks: Detected pose landmarks object (or None if not detected)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Process the image
        results = self.landmarker.detect(mp_image)

        # Create a copy of the image for annotation
        annotated_image = image.copy()

        # Draw pose landmarks if detected
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:
                # Convert to landmark_pb2 format for drawing
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in pose_landmarks
                ])

                # Draw landmarks
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )

        return annotated_image, results.pose_landmarks

    def close(self):
        """Release resources."""
        self.landmarker.close()


def detect_pose(image_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to detect pose in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the annotated image.

    Returns:
        Annotated image as numpy array.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = PoseDetector()
    annotated_image, landmarks = detector.detect(image)
    detector.close()

    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")

    return annotated_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect pose landmarks in an image")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save output image")
    parser.add_argument("--show", action="store_true", help="Display the result")

    args = parser.parse_args()

    # Detect pose
    result_image = detect_pose(args.input_image, args.output)

    # Display if requested
    if args.show:
        cv2.imshow("Pose Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
