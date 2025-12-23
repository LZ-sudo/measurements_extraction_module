"""
Hair Segmentation Module using MediaPipe Image Segmenter

This script performs hair segmentation on an image using MediaPipe's Image Segmenter
with a custom hair segmentation model, combined with face detection to improve accuracy.
Extracts normalized hair length coordinates for measurements.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
from typing import Dict, Optional, Tuple
import os


class HairSegmenter:
    """
    A class to perform hair segmentation on images using MediaPipe.
    Uses face detection to locate the head region for better segmentation accuracy.
    """

    def __init__(self, model_path: Optional[str] = None, use_face_detection: bool = True):
        """
        Initialize the Hair Segmenter.

        Args:
            model_path: Path to hair segmentation model (.tflite file).
                       If None, looks for hair_segmenter.tflite in mediapipe_task_files.
            use_face_detection: Whether to use face detection to locate head region first.
                              This significantly improves accuracy for full-body images.
        """
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.use_face_detection = use_face_detection

        # Initialize face detector if needed
        if use_face_detection:
            face_model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "mediapipe_task_files",
                "face_landmarker.task"
            )
            if os.path.exists(face_model_path):
                face_base_options = python.BaseOptions(model_asset_path=face_model_path)
                face_options = vision.FaceLandmarkerOptions(
                    base_options=face_base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1
                )
                self.face_detector = vision.FaceLandmarker.create_from_options(face_options)
            else:
                print(f"Warning: Face detector not found. Proceeding without face detection.")
                self.use_face_detection = False
                self.face_detector = None
        else:
            self.face_detector = None

        # Get hair segmentation model path
        if model_path is None:
            task_files_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "mediapipe_task_files",
                "hair_segmenter.tflite"
            )
            if os.path.exists(task_files_path):
                model_path = task_files_path
            else:
                raise FileNotFoundError(
                    f"Hair segmentation model not found at {task_files_path}. "
                    "Please provide a valid model_path."
                )

        # Create the options for ImageSegmenter
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )

        # Create the image segmenter
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def _get_head_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Detect face and estimate head region including hair.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Tuple of (top, bottom, left, right) coordinates of head region.
        """
        h, w = image.shape[:2]

        # Convert to RGB for face detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect face
        results = self.face_detector.detect(mp_image)

        if results.face_landmarks:
            # Get face landmarks
            face_landmarks = results.face_landmarks[0]

            # Calculate face bounding box
            x_coords = [int(lm.x * w) for lm in face_landmarks]
            y_coords = [int(lm.y * h) for lm in face_landmarks]

            face_left = min(x_coords)
            face_right = max(x_coords)
            face_top = min(y_coords)
            face_bottom = max(y_coords)

            face_width = face_right - face_left
            face_height = face_bottom - face_top

            # Estimate head region (expand to include hair)
            # Hair can extend significantly, especially for long hair on female subjects
            head_top = max(0, face_top - int(face_height * 1.2))  # Expand up more for hair volume
            head_bottom = min(h, face_bottom + int(face_height * 2.5))  # Expand down for long hair
            head_left = max(0, face_left - int(face_width * 0.6))  # Expand sides more
            head_right = min(w, face_right + int(face_width * 0.6))

            return head_top, head_bottom, head_left, head_right
        else:
            # No face detected, use top portion of image
            return 0, int(h * 0.5), 0, w

    def segment(self, image: np.ndarray) -> Dict:
        """
        Perform hair segmentation on an image and extract hair length measurements.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Dictionary containing normalized hair length coordinates:
            {
                "hair_length": {
                    "top": {"y": ...},     # Normalized y-coordinate of topmost hair pixel
                    "bottom": {"y": ...}   # Normalized y-coordinate of bottommost hair pixel
                }
            }
        """
        h, w = image.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        # Get head region if using face detection
        if self.use_face_detection and self.face_detector is not None:
            head_top, head_bottom, head_left, head_right = self._get_head_region(image)

            # Extract head region
            head_region = image[head_top:head_bottom, head_left:head_right]

            # Segment the head region
            head_rgb = cv2.cvtColor(head_region, cv2.COLOR_BGR2RGB)
            mp_head = mp.Image(image_format=mp.ImageFormat.SRGB, data=head_rgb)

            segmentation_result = self.segmenter.segment(mp_head)
            category_mask = segmentation_result.category_mask
            mask_array = category_mask.numpy_view()

            # Create binary mask for head region
            head_mask = (mask_array > 0.5).astype(np.uint8)

            # Place head mask into full image mask
            full_mask[head_top:head_bottom, head_left:head_right] = head_mask

        else:
            # No face detection, segment full image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            segmentation_result = self.segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask
            mask_array = category_mask.numpy_view()

            # Create binary mask
            full_mask = (mask_array > 0.5).astype(np.uint8)

        # Initialize result dictionary
        hair_data = {}

        # Find hair pixels
        hair_pixels = np.where(full_mask > 0)

        if len(hair_pixels[0]) > 0:
            # Get topmost and bottommost y-coordinates
            top_y = int(np.min(hair_pixels[0]))
            bottom_y = int(np.max(hair_pixels[0]))

            # Normalize coordinates (divide by image height)
            hair_data["hair_length"] = {
                "top": {"y": top_y / h},
                "bottom": {"y": bottom_y / h}
            }

        return hair_data

    def close(self):
        """Release resources."""
        self.segmenter.close()
        if self.face_detector is not None:
            self.face_detector.close()


def segment_hair(
    image_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    use_face_detection: bool = True
) -> Dict:
    """
    Convenience function to segment hair in an image file and extract hair length.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        model_path: Optional path to hair segmentation model.
        use_face_detection: Whether to use face detection (recommended for full-body images).

    Returns:
        Dictionary containing normalized hair length coordinates.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create segmenter and process image
    segmenter = HairSegmenter(model_path=model_path, use_face_detection=use_face_detection)
    hair_data = segmenter.segment(image)
    segmenter.close()

    # Save output if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(hair_data, f, indent=2)
        print(f"Hair length coordinates saved to {output_path}")

    return hair_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform hair segmentation on an image and extract hair length")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument("--model", type=str, help="Path to hair segmentation model (.tflite)")
    parser.add_argument("--no-face-detection", action="store_true",
                       help="Disable face detection (use for close-up head shots)")

    args = parser.parse_args()

    # Segment hair and extract length
    hair_data = segment_hair(
        args.input_image,
        args.output,
        args.model,
        use_face_detection=not args.no_face_detection
    )

    # Print the results
    print("\nHair Length:")
    print(json.dumps(hair_data, indent=2))
