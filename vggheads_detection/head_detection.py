"""
Head Detection Module using VGGHeads TorchScript model

Runs inference directly against the VGGHeads TorchScript model downloaded
from HuggingFace, without importing the head_detector package. This avoids
the package's strict Python 3.10 / torch~=2.0.1 / numpy==1.23.5 dependency
pins, which are incompatible with Python 3.13 and numpy 2.x.

Model source: https://github.com/KupynOrest/head_detector (MIT License)
Weights: HuggingFace repo "okupyn/vgg_heads" (auto-downloaded on first use)
"""

import json
import cv2
import numpy as np
import torch
import torchvision.ops
from pathlib import Path
from typing import Dict, Optional, Tuple

from huggingface_hub import hf_hub_download


_REPO_ID = "okupyn/vgg_heads"
_DEFAULT_MODEL = "vgg_heads_l"
_IMAGE_SIZE = 640


def _load_model(model_name: str, device: torch.device) -> torch.jit.ScriptModule:
    """Download and load the VGGHeads TorchScript model from HuggingFace."""
    model_path = hf_hub_download(_REPO_ID, f"{model_name}.trcd")
    loaded = torch.jit.load(model_path, map_location=device)
    loaded.eval()
    return loaded


def _preprocess(image_rgb: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int], float]:
    """
    Resize with aspect ratio, pad to IMAGE_SIZE x IMAGE_SIZE, normalise to [0, 1].

    Returns:
        tensor: (1, 3, H, W) float tensor on device
        padding: (pad_x, pad_y) pixels added to left/top respectively
        scale: scale factor applied during resize
    """
    h, w = image_rgb.shape[:2]
    if h > w:
        new_h, new_w = _IMAGE_SIZE, int(w * _IMAGE_SIZE / h)
    else:
        new_h, new_w = int(h * _IMAGE_SIZE / w), _IMAGE_SIZE
    scale = _IMAGE_SIZE / max(h, w)

    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pad_w = _IMAGE_SIZE - resized.shape[1]
    pad_h = _IMAGE_SIZE - resized.shape[0]
    padded = cv2.copyMakeBorder(
        resized,
        pad_h // 2, pad_h - pad_h // 2,
        pad_w // 2, pad_w - pad_w // 2,
        cv2.BORDER_CONSTANT, value=127,
    )
    tensor = (
        torch.from_numpy(padded)
        .to(device)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )
    return tensor, (pad_w // 2, pad_h // 2), scale


def _nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    top_k: int = 1000,
    keep_top_k: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply confidence filtering and NMS, return pixel boxes and scores as numpy arrays.
    Mirrors the logic in head_detector/utils.py without needing the package.
    """
    boxes_f = boxes.detach().float().squeeze(0)
    scores_f = scores.detach().float().squeeze(0).squeeze(-1)

    conf_mask = scores_f >= confidence_threshold
    boxes_f = boxes_f[conf_mask]
    scores_f = scores_f[conf_mask]

    if scores_f.size(0) > top_k:
        top = torch.topk(scores_f, k=top_k, largest=True, sorted=True)
        boxes_f = boxes_f[top.indices]
        scores_f = scores_f[top.indices]

    if scores_f.size(0) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    keep = torchvision.ops.nms(boxes_f, scores_f, iou_threshold)
    keep = keep[:keep_top_k]
    return boxes_f[keep].cpu().numpy(), scores_f[keep].cpu().numpy()


def _postprocess_boxes(
    boxes_xyxy: np.ndarray,
    padding: Tuple[int, int],
    scale: float,
) -> np.ndarray:
    """Reverse the padding and scaling to recover pixel coordinates in the original image."""
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    boxes = boxes_xyxy.copy().clip(0, _IMAGE_SIZE)
    boxes[:, [0, 2]] -= padding[0]  # remove left pad
    boxes[:, [1, 3]] -= padding[1]  # remove top pad
    boxes /= scale
    return np.rint(boxes).astype(int)


class HeadDetector:
    """
    Detects human heads using the VGGHeads TorchScript model.
    Returns bounding box data.
    Model weights are downloaded from HuggingFace automatically on first use.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        conf_threshold: float = 0.5,
    ):
        self.conf_threshold = conf_threshold
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _load_model(model, self._device)

    def _select_main_head_idx(self, boxes: np.ndarray) -> Optional[int]:
        """Return index of the head with the largest bounding box area."""
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return int(np.argmax(areas))

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect heads in a BGR image and return normalised bounding box data.

        Returns:
            head_detected (bool)
            head_top_y (float | None)         -- normalised [0, 1]
            head_top_y_pixels (float | None)
            head_bbox (dict | None)            -- normalised x1/y1/x2/y2
            head_bbox_pixels (dict | None)     -- pixel x1/y1/x2/y2
            all_heads (list)
        """
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor, padding, scale = _preprocess(rgb, self._device)
        with torch.no_grad():
            raw_output = self._model(tensor)

        # Model returns (bboxes_xyxy, scores, flame_params); we only need the first two
        boxes_raw, scores_raw = raw_output[0], raw_output[1]
        boxes_px, scores_np = _nms(boxes_raw, scores_raw, self.conf_threshold)
        boxes_px = _postprocess_boxes(boxes_px, padding, scale)

        result: Dict = {
            "head_detected": False,
            "head_top_y": None,
            "head_top_y_pixels": None,
            "head_bbox": None,
            "head_bbox_pixels": None,
            "all_heads": [],
        }

        for box, score in zip(boxes_px, scores_np):
            x1, y1, x2, y2 = box
            result["all_heads"].append({
                "bbox": {
                    "x1": float(x1 / w),
                    "y1": float(y1 / h),
                    "x2": float(x2 / w),
                    "y2": float(y2 / h),
                },
                "bbox_pixels": {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                },
                "confidence": float(score),
                "head_top_y": float(y1 / h),
            })

        idx = self._select_main_head_idx(boxes_px)
        if idx is not None:
            x1, y1, x2, y2 = boxes_px[idx]
            result["head_detected"] = True
            result["head_top_y"] = float(y1 / h)
            result["head_top_y_pixels"] = float(y1)
            result["head_bbox"] = {
                "x1": float(x1 / w), "y1": float(y1 / h),
                "x2": float(x2 / w), "y2": float(y2 / h),
            }
            result["head_bbox_pixels"] = {
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
            }

        return result

    def visualize(self, image: np.ndarray, detection_result: Optional[Dict] = None) -> np.ndarray:
        """Draw head bounding boxes on a BGR image."""
        if detection_result is None:
            detection_result = self.detect(image)

        annotated = image.copy()
        for head in detection_result.get("all_heads", []):
            px = head["bbox_pixels"]
            x1, y1 = int(px["x1"]), int(px["y1"])
            x2, y2 = int(px["x2"]), int(px["y2"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{head['confidence']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if detection_result.get("head_detected"):
            px = detection_result["head_bbox_pixels"]
            x1, y1 = int(px["x1"]), int(px["y1"])
            x2, y2 = int(px["x2"]), int(px["y2"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.line(annotated, (x1, y1), (x2, y1), (0, 0, 255), 2)
            cv2.putText(annotated, "HEAD TOP", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return annotated


def detect_head(
    image_path: str,
    output_path: Optional[str] = None,
    visualize_path: Optional[str] = None,
    device: Optional[str] = None,
    conf_threshold: float = 0.5,
) -> Dict:
    """
    Detect heads in an image file and return normalised bounding box data.

    The device parameter is accepted for API compatibility; the detector
    automatically uses CUDA when available.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        visualize_path: Optional path to save a visualisation image.
        device: Accepted for API compatibility. Device is selected automatically.
        conf_threshold: Minimum confidence threshold for detections.

    Returns:
        Dictionary with head_detected, head_top_y, head_bbox, etc.
    """
    del device  # auto-selected by HeadDetector
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    detector = HeadDetector(conf_threshold=conf_threshold)
    detection_result = detector.detect(image)

    if visualize_path:
        annotated = detector.visualize(image, detection_result)
        cv2.imwrite(visualize_path, annotated)
        print(f"Visualisation saved to {visualize_path}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detection_result, f, indent=2)
        print(f"Head detection results saved to {output_path}")

    return detection_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect heads using VGGHeads TorchScript model (MIT license)"
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument("-v", "--visualize", type=str,
                        help="Path to save visualisation image")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    args = parser.parse_args()

    result = detect_head(args.input_image, args.output, args.visualize,
                         conf_threshold=args.conf)
    print("\nHead Detection Results (VGGHeads):")
    print(json.dumps(result, indent=2))
