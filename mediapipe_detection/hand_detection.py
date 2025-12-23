"""
Hand Landmarks Detection Module using Google Mediapipe

This script detects hand landmarks in an image using Google's Mediapipe Hands solution.
It takes an input image, processes it to detect hand landmarks (21 landmarks per hand),
draws the landmarks and connections on the image, and returns the annotated image.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class HandDetector:
    """
    A class to detect and visualize hand landmarks using Mediapipe.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the Hand Detector.

        Args:
            model_path: Path to hand landmarker model file. If None, downloads default model.
            num_hands: Maximum number of hands to detect.
            min_hand_detection_confidence: Minimum confidence for hand detection.
            min_hand_presence_confidence: Minimum confidence for hand presence.
            min_tracking_confidence: Minimum confidence for hand tracking.
        """
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Download default model if not provided
        if model_path is None:
            model_path = self._download_default_model()

        # Create hand landmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _download_default_model(self) -> str:
        """Get the default hand landmarker model path."""
        import os

        # Check in mediapipe_task_files directory first
        task_files_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mediapipe_task_files", "hand_landmarker.task")

        if os.path.exists(task_files_path):
            return task_files_path

        # Fall back to current directory
        local_path = "hand_landmarker.task"
        if os.path.exists(local_path):
            return local_path

        # If not found, download it
        import urllib.request
        model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

        print(f"Downloading hand landmarker model...")
        urllib.request.urlretrieve(model_url, local_path)
        print(f"Model downloaded to {local_path}")

        return local_path

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, Optional[List], Optional[List]]:
        """
        Detect hand landmarks in an image and draw them.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Tuple of (annotated_image, hand_landmarks_list, handedness_list)
            - annotated_image: Image with hand landmarks drawn
            - hand_landmarks_list: List of detected hand landmarks (or None if not detected)
            - handedness_list: List indicating left/right hand (or None if not detected)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Process the image
        results = self.landmarker.detect(mp_image)

        # Create a copy of the image for annotation
        annotated_image = image.copy()

        # Draw hand landmarks if detected
        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Convert to landmark_pb2 format for drawing
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in hand_landmarks
                ])

                # Draw landmarks
                solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )

                # Add handedness labels (Left/Right)
                if results.handedness:
                    # Get the hand label (Left or Right)
                    handedness = results.handedness[idx][0].category_name
                    score = results.handedness[idx][0].score

                    # Get hand landmarks for positioning the text
                    h, w, _ = annotated_image.shape

                    # Position text at wrist landmark (index 0)
                    wrist = hand_landmarks[0]
                    text_x = int(wrist.x * w)
                    text_y = int(wrist.y * h) - 20

                    # Draw handedness label
                    cv2.putText(
                        annotated_image,
                        f"{handedness} ({score:.2f})",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        return (
            annotated_image,
            results.hand_landmarks if results.hand_landmarks else None,
            results.handedness if results.handedness else None
        )

    def close(self):
        """Release resources."""
        self.landmarker.close()


def detect_hands(image_path: str, output_path: Optional[str] = None) -> np.ndarray:
    """
    Convenience function to detect hands in an image file.

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
    detector = HandDetector()
    annotated_image, landmarks, handedness = detector.detect(image)
    detector.close()

    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")

    return annotated_image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect hand landmarks in an image")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save output image")
    parser.add_argument("--show", action="store_true", help="Display the result")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect")

    args = parser.parse_args()

    # Detect hands
    result_image = detect_hands(args.input_image, args.output)

    # Display if requested
    if args.show:
        cv2.imshow("Hand Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
