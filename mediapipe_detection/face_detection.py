"""
Face Mesh Detection Module using Google Mediapipe

This script detects face mesh landmarks in an image using Google's Mediapipe Face Mesh solution.
It takes an input image, processes it to detect face landmarks (478 3D landmarks),
draws the landmarks and connections on the image, and returns the annotated image.
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from typing import Dict, Optional, List, Tuple
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class FaceDetector:
    """
    A class to detect and visualize face mesh landmarks using Mediapipe.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the Face Landmarker.

        Args:
            model_path: Path to face landmarker model file. If None, downloads default model.
            num_faces: Maximum number of faces to detect.
            min_face_detection_confidence: Minimum confidence for face detection.
            min_face_presence_confidence: Minimum confidence for face presence.
            min_tracking_confidence: Minimum confidence for face tracking.
        """
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Download default model if not provided
        if model_path is None:
            model_path = self._download_default_model()

        # Create face landmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def _download_default_model(self) -> str:
        """Get the default face landmarker model path."""
        import os

        # Check in mediapipe_task_files directory first
        task_files_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mediapipe_task_files", "face_landmarker.task")

        if os.path.exists(task_files_path):
            return task_files_path

        # Fall back to current directory
        local_path = "face_landmarker.task"
        if os.path.exists(local_path):
            return local_path

        # If not found, download it
        import urllib.request
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

        print(f"Downloading face landmarker model...")
        urllib.request.urlretrieve(model_url, local_path)
        print(f"Model downloaded to {local_path}")

        return local_path

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect face mesh landmarks in an image and extract all landmarks as JSON.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary containing all face landmarks (478 per face):
            {
                "faces": [
                    {
                        "landmarks": [
                            {"x": ..., "y": ..., "z": ...},
                            ... (478 total landmarks)
                        ]
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
        face_data = {"faces": []}

        # Extract all landmarks if detected
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                landmarks_list = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in face_landmarks
                ]
                face_data["faces"].append({"landmarks": landmarks_list})

        return face_data

    def visualize(self, image: np.ndarray) -> np.ndarray:
        """
        Draw face mesh landmarks on the image for visualization.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Image with face landmarks drawn on it.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Process the image
        results = self.landmarker.detect(mp_image)

        # Create a copy of the image for annotation
        annotated_image = image.copy()

        # Draw face mesh landmarks if detected
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                # Convert to landmark_pb2 format for drawing
                face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                face_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                    for landmark in face_landmarks
                ])

                # Draw tesselation
                solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Draw contours
                solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style()
                )
                # Draw irises
                solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        return annotated_image

    def close(self):
        """Release resources."""
        self.landmarker.close()


def detect_face(
    image_path: str,
    output_path: Optional[str] = None,
    num_faces: int = 1,
    visualize_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to detect face mesh in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        num_faces: Maximum number of faces to detect (default: 1).
        visualize_path: Optional path to save visualization image.

    Returns:
        Dictionary containing face landmarks.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = FaceDetector(num_faces=num_faces)
    face_data = detector.detect(image)

    # Generate visualization if requested
    if visualize_path:
        annotated_image = detector.visualize(image)
        cv2.imwrite(visualize_path, annotated_image)
        print(f"Visualization saved to {visualize_path}")

    detector.close()

    # Save JSON output if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(face_data, f, indent=2)
        print(f"Face landmark coordinates saved to {output_path}")

    return face_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect face mesh landmarks in an image")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument("-v", "--visualize", type=str, help="Path to save visualization image with landmarks")
    parser.add_argument("--max-faces", type=int, default=1, help="Maximum number of faces to detect")

    args = parser.parse_args()

    # Detect face mesh
    face_data = detect_face(args.input_image, args.output, args.max_faces, args.visualize)

    # Print the results
    print(f"\nDetected {len(face_data.get('faces', []))} face(s)")
    if face_data.get('faces'):
        for idx, face in enumerate(face_data['faces']):
            print(f"Face {idx + 1}: {len(face['landmarks'])} landmarks")
