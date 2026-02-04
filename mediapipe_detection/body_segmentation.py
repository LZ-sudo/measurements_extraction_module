"""
Body Segmentation Module using MediaPipe Image Segmenter

This script performs body segmentation on an image using MediaPipe's Image Segmenter
with the selfie segmentation model to isolate the person from the background.
Extracts normalized height coordinates for measurements.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
from typing import Dict, Optional
import os


class BodySegmenter:
    """
    A class to perform body segmentation on images using MediaPipe.
    Uses the selfie segmenter model to separate person from background.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Body Segmenter.

        Args:
            model_path: Path to selfie segmentation model (.tflite file).
                       If None, looks for selfie_segmenter.tflite in weight_files.
        """
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        # Get body segmentation model path
        if model_path is None:
            task_files_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "weight_files",
                "selfie_segmenter.tflite"
            )
            if os.path.exists(task_files_path):
                model_path = task_files_path
            else:
                raise FileNotFoundError(
                    f"Body segmentation model not found at {task_files_path}. "
                    "Please provide a valid model_path or download selfie_segmenter.tflite "
                    "from MediaPipe models."
                )

        # Create the options for ImageSegmenter
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True
        )

        # Create the image segmenter
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def segment(self, image: np.ndarray) -> Dict:
        """
        Perform body segmentation on an image and extract height measurements.

        Args:
            image: Input image as numpy array (BGR format from OpenCV).

        Returns:
            Dictionary containing normalized height coordinates:
            {
                "height": {
                    "top": {"y": ...},     # Normalized y-coordinate of topmost body pixel
                    "bottom": {"y": ...}   # Normalized y-coordinate of bottommost body pixel
                }
            }
        """
        h, w = image.shape[:2]

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Perform segmentation
        segmentation_result = self.segmenter.segment(mp_image)

        # Check if we have confidence masks or category mask
        if hasattr(segmentation_result, 'confidence_masks') and segmentation_result.confidence_masks:
            # Use confidence mask (selfie segmenter uses this)
            confidence_mask = segmentation_result.confidence_masks[0]
            mask_array = confidence_mask.numpy_view()
            # Threshold the confidence mask (values range from 0.0 to 1.0)
            body_mask = (mask_array > 0.5).astype(np.uint8)
        else:
            # Fall back to category mask
            category_mask = segmentation_result.category_mask
            mask_array = category_mask.numpy_view()
            # For category mask: category 0 = background, category 1 = person
            body_mask = (mask_array == 1).astype(np.uint8)

        # Initialize result dictionary
        height_data = {}

        # Find body pixels
        body_pixels = np.where(body_mask > 0)

        if len(body_pixels[0]) > 0:
            # Get topmost and bottommost y-coordinates
            top_y = int(np.min(body_pixels[0]))
            bottom_y = int(np.max(body_pixels[0]))

            # Normalize coordinates (divide by image height)
            height_data["height"] = {
                "top": {"y": top_y / h},
                "bottom": {"y": bottom_y / h}
            }

        return height_data

    def visualize(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize body segmentation with overlay and height markers.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Image with segmentation overlay and height markers.
        """
        h, w = image.shape[:2]

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Perform segmentation
        segmentation_result = self.segmenter.segment(mp_image)

        # Get the mask
        if hasattr(segmentation_result, 'confidence_masks') and segmentation_result.confidence_masks:
            confidence_mask = segmentation_result.confidence_masks[0]
            mask_array = confidence_mask.numpy_view()
            body_mask = (mask_array > 0.5).astype(np.uint8)
        else:
            category_mask = segmentation_result.category_mask
            mask_array = category_mask.numpy_view()
            body_mask = (mask_array == 1).astype(np.uint8)

        # Create visualization
        annotated_image = image.copy()

        # Create colored overlay (semi-transparent green for body)
        overlay = np.zeros_like(image)
        overlay[body_mask > 0] = [0, 255, 0]  # Green for body
        annotated_image = cv2.addWeighted(annotated_image, 0.7, overlay, 0.3, 0)

        # Find and draw height markers
        body_pixels = np.where(body_mask > 0)
        if len(body_pixels[0]) > 0:
            top_y = int(np.min(body_pixels[0]))
            bottom_y = int(np.max(body_pixels[0]))

            # Draw horizontal lines at top and bottom
            cv2.line(annotated_image, (0, top_y), (w, top_y), (0, 0, 255), 2)
            cv2.line(annotated_image, (0, bottom_y), (w, bottom_y), (0, 0, 255), 2)

            # Add labels
            cv2.putText(annotated_image, f"Top: y={top_y/h:.3f}", (10, top_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated_image, f"Bottom: y={bottom_y/h:.3f}", (10, bottom_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw height span line on the side
            mid_x = w - 50
            cv2.line(annotated_image, (mid_x, top_y), (mid_x, bottom_y), (255, 0, 255), 3)
            cv2.circle(annotated_image, (mid_x, top_y), 5, (255, 0, 255), -1)
            cv2.circle(annotated_image, (mid_x, bottom_y), 5, (255, 0, 255), -1)

            # Add height measurement
            height_normalized = (bottom_y - top_y) / h
            cv2.putText(annotated_image, f"Height: {height_normalized:.3f}", (mid_x - 150, (top_y + bottom_y) // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        return annotated_image

    def close(self):
        """Release resources."""
        self.segmenter.close()


def segment_body(
    image_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    visualize_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to segment body in an image file and extract height.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        model_path: Optional path to body segmentation model.
        visualize_path: Optional path to save visualization image.

    Returns:
        Dictionary containing normalized height coordinates.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create segmenter and process image
    segmenter = BodySegmenter(model_path=model_path)
    height_data = segmenter.segment(image)

    # Generate visualization if requested
    if visualize_path:
        annotated_image = segmenter.visualize(image)
        cv2.imwrite(visualize_path, annotated_image)
        print(f"Visualization saved to {visualize_path}")

    segmenter.close()

    # Save output if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(height_data, f, indent=2)
        print(f"Body height coordinates saved to {output_path}")

    return height_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform body segmentation on an image and extract height")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument("-v", "--visualize", type=str, help="Path to save visualization image with segmentation overlay")
    parser.add_argument("--model", type=str, help="Path to body segmentation model (.tflite)")

    args = parser.parse_args()

    # Segment body and extract height
    height_data = segment_body(
        args.input_image,
        args.output,
        args.model,
        args.visualize
    )

    # Print the results
    print("\nBody Height:")
    print(json.dumps(height_data, indent=2))
