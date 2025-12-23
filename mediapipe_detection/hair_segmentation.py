"""
Hair Segmentation Module using MediaPipe Image Segmenter

This script performs hair segmentation on an image using MediaPipe's Image Segmenter
with a custom hair segmentation model, combined with face detection to improve accuracy.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional
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
            # Hair typically extends above and around the face
            head_top = max(0, face_top - int(face_height * 1.0))  # Expand up significantly
            head_bottom = min(h, face_bottom + int(face_height * 0.3))  # Include chin
            head_left = max(0, face_left - int(face_width * 0.4))  # Expand sides
            head_right = min(w, face_right + int(face_width * 0.4))

            return head_top, head_bottom, head_left, head_right
        else:
            # No face detected, use top portion of image
            return 0, int(h * 0.5), 0, w

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform hair segmentation on an image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Tuple of (annotated_image, mask)
            - annotated_image: Original image with hair region highlighted
            - mask: Binary mask where hair is white (255) and non-hair is black (0)
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
            head_mask = (mask_array > 0.5).astype(np.uint8) * 255

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
            full_mask = (mask_array > 0.5).astype(np.uint8) * 255

        # Create annotated image with colored overlay
        annotated_image = image.copy()

        # Create cyan overlay for hair regions
        overlay = np.zeros_like(image)
        overlay[full_mask > 0] = [255, 255, 0]  # Cyan in BGR

        # Blend overlay with original image
        alpha = 0.5
        annotated_image = cv2.addWeighted(annotated_image, 1, overlay, alpha, 0)

        # Draw contours around hair regions
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated_image, contours, -1, (0, 255, 255), 2)

        return annotated_image, full_mask

    def close(self):
        """Release resources."""
        self.segmenter.close()
        if self.face_detector is not None:
            self.face_detector.close()


def segment_hair(
    image_path: str,
    output_path: Optional[str] = None,
    mask_output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    use_face_detection: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to segment hair in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the annotated image.
        mask_output_path: Optional path to save the segmentation mask.
        model_path: Optional path to hair segmentation model.
        use_face_detection: Whether to use face detection (recommended for full-body images).

    Returns:
        Tuple of (annotated_image, mask).
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create segmenter and process image
    segmenter = HairSegmenter(model_path=model_path, use_face_detection=use_face_detection)
    annotated_image, mask = segmenter.segment(image)
    segmenter.close()

    # Save outputs if paths provided
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved to {output_path}")

    if mask_output_path:
        cv2.imwrite(mask_output_path, mask)
        print(f"Segmentation mask saved to {mask_output_path}")

    return annotated_image, mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform hair segmentation on an image using MediaPipe")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save output image")
    parser.add_argument("-m", "--mask", type=str, help="Path to save segmentation mask")
    parser.add_argument("--model", type=str, help="Path to hair segmentation model (.tflite)")
    parser.add_argument("--no-face-detection", action="store_true",
                       help="Disable face detection (use for close-up head shots)")
    parser.add_argument("--show", action="store_true", help="Display the result")

    args = parser.parse_args()

    # Segment hair
    result_image, result_mask = segment_hair(
        args.input_image,
        args.output,
        args.mask,
        args.model,
        use_face_detection=not args.no_face_detection
    )

    # Display if requested
    if args.show:
        cv2.imshow("Hair Segmentation", result_image)
        cv2.imshow("Hair Mask", result_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
