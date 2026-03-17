"""
Movement Accuracy Utility Functions

Provides keypoint extraction from video or image, torso-relative pose normalization,
and joint interior-angle error computation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from easy_ViTPose import VitInference

# Add the measurements_extraction_module directory to sys.path so that
# vitpose_detection.config_model and body_region_config can be imported
# regardless of whether this module is run directly or as part of a package.
_MEASUREMENTS_DIR = Path(__file__).resolve().parent.parent
if str(_MEASUREMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_MEASUREMENTS_DIR))

from vitpose_detection.config_model import get_model_paths
from body_region_config import AngleTriplet


# Minimum confidence score for a keypoint to be considered valid.
CONFIDENCE_THRESHOLD = 0.2

# Recognised image file extensions.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image_path(path: Path) -> bool:
    """Return True if the path points to a recognised image file type."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS

# COCO-WholeBody body joint indices used as normalization anchors.
_LEFT_SHOULDER_IDX  = 5
_RIGHT_SHOULDER_IDX = 6
_LEFT_HIP_IDX       = 11
_RIGHT_HIP_IDX      = 12


def _select_main_person(keypoints_dict: dict) -> Optional[np.ndarray]:
    """
    Select the person with the largest bounding box from detected persons.

    Args:
        keypoints_dict: Dict of {person_id: np.ndarray shape (133, 3)} as (y, x, score).

    Returns:
        Keypoints array for the main person, or None if no valid person found.
    """
    if not keypoints_dict:
        return None

    best_person = None
    best_area   = 0.0

    for kpts in keypoints_dict.values():
        valid = kpts[kpts[:, 2] > CONFIDENCE_THRESHOLD]
        if len(valid) < 5:
            continue
        area = (valid[:, 1].max() - valid[:, 1].min()) * (valid[:, 0].max() - valid[:, 0].min())
        if area > best_area:
            best_area   = area
            best_person = kpts

    return best_person


def extract_keypoints_from_video(
    video_path: str,
    device: Optional[str] = None,
    yolo_size: int = 320,
) -> List[Optional[np.ndarray]]:
    """
    Extract per-frame raw keypoints from a video using ViTPose-H (wholebody).

    Args:
        video_path: Path to the input video file.
        device: Device to run on ('cpu', 'cuda', 'mps', or None for auto-detect).
        yolo_size: YOLO input image size.

    Returns:
        List with one entry per frame. Each entry is either:
        - np.ndarray of shape (133, 3) as (y, x, score) if a person was detected, or
        - None if no person was detected in that frame.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    vitpose_path, yolo_path = get_model_paths("h", "yolov8s")

    model = VitInference(
        vitpose_path,
        yolo_path,
        model_name="h",
        yolo_size=yolo_size,
        is_video=True,
        device=device,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_keypoints: List[Optional[np.ndarray]] = []
    frame_idx = 0

    print(f"Extracting keypoints: {video_path.name} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        keypoints_dict = model.inference(img_rgb)
        frame_keypoints.append(_select_main_person(keypoints_dict))

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{total_frames} frames processed")

    cap.release()

    if hasattr(model, "reset"):
        model.reset()

    valid_count = sum(1 for k in frame_keypoints if k is not None)
    print(f"  Done. Valid frames: {valid_count}/{len(frame_keypoints)}")

    return frame_keypoints


def extract_keypoints_from_image(
    image_path: str,
    device: Optional[str] = None,
) -> List[Optional[np.ndarray]]:
    """
    Extract keypoints from a single image using ViTPose-H (wholebody).

    Returns a single-element list for compatibility with the video pipeline.
    When paired with a video source, DTW trivially aligns the single frame
    against every video frame.

    Args:
        image_path: Path to the input image file.
        device: Device to run on ('cpu', 'cuda', 'mps', or None for auto-detect).

    Returns:
        List containing one entry:
        - np.ndarray of shape (133, 3) as (y, x, score) if a person was detected, or
        - None if no person was detected.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    vitpose_path, yolo_path = get_model_paths("h", "yolov8s")

    model = VitInference(
        vitpose_path,
        yolo_path,
        model_name="h",
        yolo_size=320,
        is_video=False,
        device=device,
    )

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints_dict = model.inference(img_rgb)
    kpts = _select_main_person(keypoints_dict)

    valid = 1 if kpts is not None else 0
    print(f"Extracting keypoints: {image_path.name} (single image)")
    print(f"  Done. Valid frames: {valid}/1")

    return [kpts]


def normalize_pose(frame_kpts: np.ndarray) -> Optional[Dict[int, np.ndarray]]:
    """
    Normalize keypoint positions to torso-relative coordinates.

    Anchors all positions to the hip midpoint and scales by the average
    shoulder-to-hip distance. This makes the metric invariant to subject
    scale and camera distance differences between the two videos.

    Uses whichever anchor joints are above CONFIDENCE_THRESHOLD rather than
    requiring all four. Requires at least one valid hip and at least one valid
    ipsilateral shoulder-hip pair for a meaningful reference frame.

    Args:
        frame_kpts: Array of shape (133, 3) as (y, x, score).

    Returns:
        Dict mapping COCO keypoint index to normalized np.ndarray([x, y]),
        containing only keypoints that exceed CONFIDENCE_THRESHOLD.
        Returns None if anchor keypoints are missing or degenerate.
    """
    l_shoulder = frame_kpts[_LEFT_SHOULDER_IDX]
    r_shoulder = frame_kpts[_RIGHT_SHOULDER_IDX]
    l_hip      = frame_kpts[_LEFT_HIP_IDX]
    r_hip      = frame_kpts[_RIGHT_HIP_IDX]

    # Hip midpoint from whichever hips are above threshold
    valid_hips = [a for a in [l_hip, r_hip] if a[2] >= CONFIDENCE_THRESHOLD]
    if not valid_hips:
        return None
    hip_mid_x = sum(a[1] for a in valid_hips) / len(valid_hips)
    hip_mid_y = sum(a[0] for a in valid_hips) / len(valid_hips)

    # Torso scale from available ipsilateral shoulder-hip pairs
    torso_dists = []
    for shoulder, hip in [(l_shoulder, l_hip), (r_shoulder, r_hip)]:
        if shoulder[2] >= CONFIDENCE_THRESHOLD and hip[2] >= CONFIDENCE_THRESHOLD:
            torso_dists.append(
                float(np.linalg.norm([shoulder[1] - hip[1], shoulder[0] - hip[0]]))
            )

    if not torso_dists:
        return None

    torso_scale = sum(torso_dists) / len(torso_dists)
    if torso_scale < 1e-6:
        return None

    normalized: Dict[int, np.ndarray] = {}
    for idx in range(len(frame_kpts)):
        y_px, x_px, score = frame_kpts[idx]
        if score >= CONFIDENCE_THRESHOLD:
            normalized[idx] = np.array([
                (x_px - hip_mid_x) / torso_scale,
                (y_px - hip_mid_y) / torso_scale,
            ])

    return normalized


def build_frame_vector(
    normalized_kpts: Dict[int, np.ndarray],
    joint_indices: List[int],
) -> Optional[np.ndarray]:
    """
    Flatten tracked joint (x, y) coordinates into a 1D vector for DTW input.

    Returns None if any of the requested joint indices are missing
    (i.e., below confidence threshold for that frame).
    """
    coords = []
    for idx in joint_indices:
        if idx not in normalized_kpts:
            return None
        coords.append(normalized_kpts[idx])
    return np.concatenate(coords)




def _angle_between(
    a: np.ndarray,
    vertex: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute the interior angle (degrees) at vertex between vectors (vertex->a)
    and (vertex->b).

    Returns 0.0 if either vector is degenerate (near-zero length).
    """
    v1 = a - vertex
    v2 = b - vertex
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-9:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def compute_angle_error(
    gt_kpts: Dict[int, np.ndarray],
    pred_kpts: Dict[int, np.ndarray],
    angle_triplets: List[AngleTriplet],
) -> Dict[str, float]:
    """
    Compute the absolute angular error (degrees) at each triplet vertex
    between GT and predicted poses.

    Args:
        gt_kpts:        Ground truth normalised keypoints.
        pred_kpts:      Predicted normalised keypoints.
        angle_triplets: AngleTriplet list from the BodyRegion.

    Returns:
        Dict mapping triplet name to angular error in degrees.
        Triplets where any of the three keypoints are missing are omitted.
    """
    errors: Dict[str, float] = {}
    for triplet in angle_triplets:
        idxs = (triplet.vertex_index, triplet.a_index, triplet.b_index)
        if any(i not in gt_kpts or i not in pred_kpts for i in idxs):
            continue
        gt_angle   = _angle_between(gt_kpts[triplet.a_index],   gt_kpts[triplet.vertex_index],   gt_kpts[triplet.b_index])
        pred_angle = _angle_between(pred_kpts[triplet.a_index], pred_kpts[triplet.vertex_index], pred_kpts[triplet.b_index])
        errors[triplet.name] = abs(gt_angle - pred_angle)
    return errors


def compute_angle_accuracy(
    gt_kpts: Dict[int, np.ndarray],
    pred_kpts: Dict[int, np.ndarray],
    angle_triplets: List[AngleTriplet],
) -> Dict[str, float]:
    """
    Compute Gaussian-kernel accuracy for each triplet's angular error.

    Formula: accuracy = exp(-error_deg^2 / (2 * sigma_deg^2))
    At 0 deg error -> 1.0; at sigma_deg error -> ~0.61; at 2*sigma_deg -> ~0.14.

    Args:
        gt_kpts:        Ground truth normalised keypoints.
        pred_kpts:      Predicted normalised keypoints.
        angle_triplets: AngleTriplet list from the BodyRegion.

    Returns:
        Dict mapping triplet name to accuracy in [0, 1].
    """
    scores: Dict[str, float] = {}
    for triplet in angle_triplets:
        idxs = (triplet.vertex_index, triplet.a_index, triplet.b_index)
        if any(i not in gt_kpts or i not in pred_kpts for i in idxs):
            continue
        gt_angle   = _angle_between(gt_kpts[triplet.a_index],   gt_kpts[triplet.vertex_index],   gt_kpts[triplet.b_index])
        pred_angle = _angle_between(pred_kpts[triplet.a_index], pred_kpts[triplet.vertex_index], pred_kpts[triplet.b_index])
        err = abs(gt_angle - pred_angle)
        scores[triplet.name] = float(np.exp(-err ** 2 / (2.0 * triplet.sigma_deg ** 2)))
    return scores
