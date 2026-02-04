"""
Head Detection Module using YOLOv8

This module provides head detection using a YOLOv8 model trained on CrowdHuman dataset.
The primary use case is to detect the top of a subject's head for accurate height measurement,
replacing the less reliable body segmentation approach.

Model source: https://github.com/Owen718/Head-Detection-Yolov8
The head bounding box top edge provides a consistent reference point for height calculation.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional

from ultralytics import YOLO


# ============================================================================
# Model Configuration
# ============================================================================

# Get the weight files directory (relative to this module)
MODULE_DIR = Path(__file__).parent.parent
WEIGHT_FILES_DIR = MODULE_DIR / "weight_files"

# Model file name (from Owen718/Head-Detection-Yolov8)
HEAD_MODEL_FILENAME = "best.pt"

# Google Drive file ID for the pre-trained model
# From: https://drive.google.com/file/d/1qlBmiEU4GBV13fxPhLZqjhjBbREvs8-m/view?usp=sharing
GDRIVE_FILE_ID = "1qlBmiEU4GBV13fxPhLZqjhjBbREvs8-m"


def download_model_from_gdrive(output_path: Path) -> None:
    """Download the head detection model from Google Drive.

    Args:
        output_path: Path where the model should be saved.

    Raises:
        ImportError: If gdown is not installed.
        Exception: If download fails.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for automatic model download. "
            "Install it with: pip install gdown"
        )

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download from Google Drive
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading head detection model from Google Drive...")
    print(f"  URL: https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
    print(f"  Destination: {output_path}")

    gdown.download(url, str(output_path), quiet=False)

    if not output_path.exists():
        raise Exception("Download failed - model file not found after download")

    print(f"Model downloaded successfully: {output_path}")


def get_model_path() -> Path:
    """Get the path to the head detection model, downloading if necessary.

    Returns:
        Path to the model file.

    Raises:
        Exception: If the model cannot be found or downloaded.
    """
    model_path = WEIGHT_FILES_DIR / HEAD_MODEL_FILENAME

    if not model_path.exists():
        print(f"Head detection model not found at: {model_path}")
        print("Attempting to download from Google Drive...")
        download_model_from_gdrive(model_path)

    return model_path


class HeadDetector:
    """
    A class to detect human heads using YOLOv8.
    Returns head bounding boxes for height measurement.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        conf_threshold: float = 0.5
    ):
        """
        Initialize the Head Detector using YOLOv8.

        Args:
            device: Device to run on - 'cpu', 'cuda', or 'mps'. None for auto-detect.
            conf_threshold: Minimum confidence threshold for detections.
        """
        model_path = get_model_path()

        # Initialize YOLO model
        self.model = YOLO(str(model_path))
        self.device = device
        self.conf_threshold = conf_threshold

    def _select_main_head(
        self,
        boxes: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Select the main head from detected heads.

        Strategy: Select the head with the largest bounding box area,
        which typically corresponds to the person closest to the camera.

        Args:
            boxes: Array of bounding boxes (N, 4) as (x1, y1, x2, y2) in pixels.

        Returns:
            Bounding box for the main head (x1, y1, x2, y2), or None if no head detected.
        """
        if len(boxes) == 0:
            return None

        best_box = None
        best_area = 0

        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = box

        return best_box

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect heads in an image and return head bounding box data.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary containing head detection results with normalized coordinates.
        """
        h, w = image.shape[:2]

        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )

        # Initialize result dictionary
        detection_result = {
            'head_detected': False,
            'head_bbox': None,
            'head_top_y': None,
            'all_heads': []
        }

        if not results or len(results[0].boxes) == 0:
            return detection_result

        # Get all boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # Store all detected heads
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = box
            detection_result['all_heads'].append({
                'bbox': {
                    'x1': float(x1 / w),
                    'y1': float(y1 / h),
                    'x2': float(x2 / w),
                    'y2': float(y2 / h)
                },
                'bbox_pixels': {
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2)
                },
                'confidence': float(conf),
                'head_top_y': float(y1 / h)
            })

        # Select main head
        main_box = self._select_main_head(boxes)

        if main_box is not None:
            x1, y1, x2, y2 = main_box[:4]
            detection_result['head_detected'] = True
            detection_result['head_bbox'] = {
                'x1': float(x1 / w),
                'y1': float(y1 / h),
                'x2': float(x2 / w),
                'y2': float(y2 / h)
            }
            detection_result['head_bbox_pixels'] = {
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2)
            }
            # Head top y-coordinate (normalized) - this is the key output for height
            detection_result['head_top_y'] = float(y1 / h)
            detection_result['head_top_y_pixels'] = float(y1)

        return detection_result

    def visualize(self, image: np.ndarray, detection_result: Optional[Dict] = None) -> np.ndarray:
        """
        Draw head detection results on the image for visualization.

        Args:
            image: Input image as numpy array (BGR format).
            detection_result: Optional pre-computed detection result.
                            If None, detection will be run.

        Returns:
            Image with head bounding boxes drawn on it.
        """
        annotated_image = image.copy()

        if detection_result is None:
            detection_result = self.detect(image)

        # Draw all detected heads
        for head_idx, head in enumerate(detection_result.get('all_heads', [])):
            bbox_px = head['bbox_pixels']
            x1, y1 = int(bbox_px['x1']), int(bbox_px['y1'])
            x2, y2 = int(bbox_px['x2']), int(bbox_px['y2'])
            conf = head['confidence']

            # Draw bounding box (green for secondary heads)
            color = (0, 255, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

            # Draw confidence label
            label = f"Head {head_idx+1}: {conf:.2f}"
            cv2.putText(
                annotated_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Highlight main head with different color
        if detection_result.get('head_detected'):
            bbox_px = detection_result['head_bbox_pixels']
            x1, y1 = int(bbox_px['x1']), int(bbox_px['y1'])
            x2, y2 = int(bbox_px['x2']), int(bbox_px['y2'])

            # Draw main head box in blue (thicker)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Draw head top line in red
            cv2.line(annotated_image, (x1, y1), (x2, y1), (0, 0, 255), 2)
            cv2.putText(
                annotated_image, "HEAD TOP", (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

        return annotated_image

    def close(self):
        """Release resources."""
        pass


def detect_head(
    image_path: str,
    output_path: Optional[str] = None,
    visualize_path: Optional[str] = None,
    device: Optional[str] = None,
    conf_threshold: float = 0.5
) -> Dict:
    """
    Convenience function to detect heads in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        visualize_path: Optional path to save visualization image.
        device: Device to run on - 'cpu', 'cuda', 'mps', or None for auto-detect.
        conf_threshold: Minimum confidence threshold for detections.

    Returns:
        Dictionary containing head detection results.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Create detector and process image
    detector = HeadDetector(
        device=device,
        conf_threshold=conf_threshold
    )
    detection_result = detector.detect(image)

    # Generate visualization if requested
    if visualize_path:
        annotated_image = detector.visualize(image, detection_result)
        cv2.imwrite(visualize_path, annotated_image)
        print(f"Visualization saved to {visualize_path}")

    detector.close()

    # Save output if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detection_result, f, indent=2)
        print(f"Head detection results saved to {output_path}")

    return detection_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect heads using YOLOv8 model trained on CrowdHuman"
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument(
        "-v", "--visualize", type=str,
        help="Path to save visualization image with head bounding boxes"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto-detect)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold for detections (default: 0.5)"
    )

    args = parser.parse_args()

    # Detect heads
    result = detect_head(
        args.input_image,
        args.output,
        args.visualize,
        args.device,
        args.conf
    )

    # Print the results
    print("\nHead Detection Results (YOLOv8):")
    print(json.dumps(result, indent=2))
