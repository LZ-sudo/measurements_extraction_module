"""
Hand Landmarks Detection Module using Google Mediapipe

This script detects hand landmarks in an image using Google's Mediapipe Hands solution.
It extracts specific landmark pairs needed for hand measurements and returns them as JSON.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from typing import Dict, Optional


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

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect hand landmarks in an image and extract measurement landmark pairs.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary containing normalized landmark coordinates for hand measurements:
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
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Process the image
        results = self.landmarker.detect(mp_image)

        # Initialize result dictionary
        hand_data = {"hands": []}

        # Extract landmark pairs if detected
        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Helper function to extract landmark coordinates
                def get_landmark_coords(lm_idx: int) -> Dict:
                    lm = hand_landmarks[lm_idx]
                    return {"x": lm.x, "y": lm.y, "z": lm.z}

                # Get handedness info
                handedness = results.handedness[idx][0].category_name if results.handedness else "Unknown"
                confidence = results.handedness[idx][0].score if results.handedness else 0.0

                # Hand length: landmarks 0 (wrist) and 12 (base of ring finger)
                hand_info = {
                    "handedness": handedness,
                    "confidence": float(confidence),
                    "hand_length": {
                        "landmark_0": get_landmark_coords(0),   # Wrist
                        "landmark_12": get_landmark_coords(12)  # Base of ring finger (middle metacarpal)
                    }
                }

                hand_data["hands"].append(hand_info)

        return hand_data

    def close(self):
        """Release resources."""
        self.landmarker.close()


def detect_hands(image_path: str, output_path: Optional[str] = None, num_hands: int = 2) -> Dict:
    """
    Convenience function to detect hand landmarks in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        num_hands: Maximum number of hands to detect (default: 2).

    Returns:
        Dictionary containing landmark pairs for hand measurements.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = HandDetector(num_hands=num_hands)
    hand_data = detector.detect(image)
    detector.close()

    # Save output if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(hand_data, f, indent=2)
        print(f"Hand landmark coordinates saved to {output_path}")

    return hand_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect hand landmarks in an image and extract measurement pairs")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect")

    args = parser.parse_args()

    # Detect hands and extract landmark pairs
    hand_data = detect_hands(args.input_image, args.output, num_hands=args.max_hands)

    # Print the results
    print("\nHand Landmark Pairs:")
    print(json.dumps(hand_data, indent=2))
