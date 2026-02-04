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
from PIL import Image


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
                       If None, looks for hair_segmenter.tflite in weight_files.
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
                "weight_files",
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
                "weight_files",
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

    def segment(self, image: np.ndarray, return_mask: bool = False, return_isolated: bool = False) -> Dict:
        """
        Perform hair segmentation on an image and extract hair length measurements.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).
            return_mask: If True, includes the segmentation mask in the returned dictionary.
            return_isolated: If True, includes isolated hair image (BGRA with transparent background).

        Returns:
            Dictionary containing normalized hair length coordinates:
            {
                "hair_length": {
                    "top": {"y": ...},     # Normalized y-coordinate of topmost hair pixel
                    "bottom": {"y": ...}   # Normalized y-coordinate of bottommost hair pixel
                },
                "mask": np.ndarray (optional),      # Binary mask if return_mask=True
                "isolated": np.ndarray (optional)   # BGRA image if return_isolated=True
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

            # Squeeze extra dimensions if present (some models return (H, W, 1))
            if mask_array.ndim == 3:
                mask_array = np.squeeze(mask_array, axis=-1)

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

            # Squeeze extra dimensions if present (some models return (H, W, 1))
            if mask_array.ndim == 3:
                mask_array = np.squeeze(mask_array, axis=-1)

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

        # Include mask if requested
        if return_mask:
            hair_data["mask"] = full_mask

        # Include isolated hair image if requested
        if return_isolated:
            # Create BGRA image (BGR + Alpha channel)
            bgra_image = np.zeros((h, w, 4), dtype=np.uint8)
            # Expand mask to 3 channels for RGB masking
            mask_3channel = np.stack([full_mask, full_mask, full_mask], axis=2)
            # Copy RGB values only where hair is detected (multiply by mask)
            bgra_image[:, :, 0:3] = image * mask_3channel
            # Set alpha channel: 255 (opaque) where hair is detected, 0 (transparent) elsewhere
            bgra_image[:, :, 3] = full_mask * 255
            hair_data["isolated"] = bgra_image

        return hair_data

    def visualize(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize hair segmentation with overlay and length markers.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Image with hair segmentation overlay and length markers.
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

            # Squeeze extra dimensions if present (some models return (H, W, 1))
            if mask_array.ndim == 3:
                mask_array = np.squeeze(mask_array, axis=-1)

            # Create binary mask for head region
            head_mask = (mask_array > 0.5).astype(np.uint8)

            # Place head mask into full image mask
            full_mask[head_top:head_bottom, head_left:head_right] = head_mask

            # Create visualization
            annotated_image = image.copy()

            # Draw head region bounding box (for debugging)
            cv2.rectangle(annotated_image, (head_left, head_top), (head_right, head_bottom),
                         (255, 255, 0), 2)
            cv2.putText(annotated_image, "Head Region", (head_left, head_top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        else:
            # No face detection, segment full image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            segmentation_result = self.segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask
            mask_array = category_mask.numpy_view()

            # Squeeze extra dimensions if present (some models return (H, W, 1))
            if mask_array.ndim == 3:
                mask_array = np.squeeze(mask_array, axis=-1)

            # Create binary mask
            full_mask = (mask_array > 0.5).astype(np.uint8)

            # Create visualization
            annotated_image = image.copy()

        # Create colored overlay (semi-transparent red for hair)
        overlay = np.zeros_like(image)
        overlay[full_mask > 0] = [0, 0, 255]  # Red for hair
        annotated_image = cv2.addWeighted(annotated_image, 0.7, overlay, 0.3, 0)

        # Find and draw hair length markers
        hair_pixels = np.where(full_mask > 0)
        if len(hair_pixels[0]) > 0:
            top_y = int(np.min(hair_pixels[0]))
            bottom_y = int(np.max(hair_pixels[0]))

            # Draw horizontal lines at top and bottom
            cv2.line(annotated_image, (0, top_y), (w, top_y), (0, 255, 0), 2)
            cv2.line(annotated_image, (0, bottom_y), (w, bottom_y), (0, 255, 0), 2)

            # Add labels
            cv2.putText(annotated_image, f"Top: y={top_y/h:.3f}", (10, top_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_image, f"Bottom: y={bottom_y/h:.3f}", (10, bottom_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw hair length span line on the side
            mid_x = w - 50
            cv2.line(annotated_image, (mid_x, top_y), (mid_x, bottom_y), (255, 0, 255), 3)
            cv2.circle(annotated_image, (mid_x, top_y), 5, (255, 0, 255), -1)
            cv2.circle(annotated_image, (mid_x, bottom_y), 5, (255, 0, 255), -1)

            # Add hair length measurement
            hair_length_normalized = (bottom_y - top_y) / h
            cv2.putText(annotated_image, f"Hair Length: {hair_length_normalized:.3f}",
                       (mid_x - 200, (top_y + bottom_y) // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return annotated_image

    def close(self):
        """Release resources."""
        self.segmenter.close()
        if self.face_detector is not None:
            self.face_detector.close()


def segment_hair(
    image_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    use_face_detection: bool = True,
    visualize_path: Optional[str] = None,
    isolated_hair_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to segment hair in an image file and extract hair length.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        model_path: Optional path to hair segmentation model.
        use_face_detection: Whether to use face detection (recommended for full-body images).
        visualize_path: Optional path to save visualization image.
        isolated_hair_path: Optional path to save isolated hair as PNG with transparent background.

    Returns:
        Dictionary containing normalized hair length coordinates.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create segmenter and process image
    segmenter = HairSegmenter(model_path=model_path, use_face_detection=use_face_detection)
    hair_data = segmenter.segment(image, return_isolated=bool(isolated_hair_path))

    # Generate visualization if requested
    if visualize_path:
        annotated_image = segmenter.visualize(image)
        cv2.imwrite(visualize_path, annotated_image)
        print(f"Visualization saved to {visualize_path}")

    # Generate isolated hair PNG if requested
    if isolated_hair_path and "isolated" in hair_data:
        # Convert BGRA to RGBA for PIL (OpenCV uses BGR, PIL uses RGB)
        bgra_image = hair_data["isolated"]
        # Split channels: B, G, R, A
        b, g, r, a = cv2.split(bgra_image)
        # Merge as R, G, B, A for PIL
        rgba_image = cv2.merge([r, g, b, a])
        # Convert to PIL Image and save with transparency
        pil_image = Image.fromarray(rgba_image, mode='RGBA')
        pil_image.save(isolated_hair_path, 'PNG')
        print(f"Isolated hair image saved to {isolated_hair_path}")
        # Remove the image array from the returned data to keep it clean
        del hair_data["isolated"]

    segmenter.close()

    # Save output if path provided
    if output_path:
        # Create a clean copy for JSON output (without numpy arrays)
        json_data = {k: v for k, v in hair_data.items() if not isinstance(v, np.ndarray)}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"Hair length coordinates saved to {output_path}")

    return hair_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform hair segmentation on an image and extract hair length")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument("-v", "--visualize", type=str, help="Path to save visualization image with segmentation overlay")
    parser.add_argument("-i", "--isolated", type=str, help="Path to save isolated hair as PNG with transparent background")
    parser.add_argument("--model", type=str, help="Path to hair segmentation model (.tflite)")
    parser.add_argument("--no-face-detection", action="store_true",
                       help="Disable face detection (use for close-up head shots)")

    args = parser.parse_args()

    # Segment hair and extract length
    hair_data = segment_hair(
        args.input_image,
        args.output,
        args.model,
        use_face_detection=not args.no_face_detection,
        visualize_path=args.visualize,
        isolated_hair_path=args.isolated
    )

    # Print the results
    print("\nHair Length:")
    print(json.dumps(hair_data, indent=2))
