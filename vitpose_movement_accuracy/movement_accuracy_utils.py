"""
Movement Accuracy Utility Functions

Provides keypoint extraction from video or image, torso-relative pose normalization,
DTW temporal alignment, and pose accuracy metrics (OKS, MPJPE, joint angle error).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from dtaidistance import dtw_ndim
from easy_ViTPose import VitInference

# Add the measurements_extraction_module directory to sys.path so that
# vitpose_detection.config_model and body_region_config can be imported
# regardless of whether this module is run directly or as part of a package.
_MEASUREMENTS_DIR = Path(__file__).resolve().parent.parent
if str(_MEASUREMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_MEASUREMENTS_DIR))

from vitpose_detection.config_model import get_model_paths
from body_region_config import AngleTriplet, JointInfo


# Minimum confidence score for a keypoint to be considered valid.
CONFIDENCE_THRESHOLD = 0.3

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

    if any(a[2] < CONFIDENCE_THRESHOLD for a in [l_shoulder, r_shoulder, l_hip, r_hip]):
        return None

    # Hip midpoint in (x, y) order (kpts are stored as y, x, score)
    hip_mid_x = (l_hip[1] + r_hip[1]) / 2.0
    hip_mid_y = (l_hip[0] + r_hip[0]) / 2.0

    # Torso scale: average of left and right shoulder-to-hip distances
    left_torso  = float(np.linalg.norm([l_shoulder[1] - l_hip[1], l_shoulder[0] - l_hip[0]]))
    right_torso = float(np.linalg.norm([r_shoulder[1] - r_hip[1], r_shoulder[0] - r_hip[0]]))
    torso_scale = (left_torso + right_torso) / 2.0

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


def align_with_dtw(
    gt_vectors: List[np.ndarray],
    pred_vectors: List[np.ndarray],
) -> List[Tuple[int, int]]:
    """
    Align two pose sequences temporally using Dynamic Time Warping.

    Args:
        gt_vectors: Per-frame flattened joint vectors for the ground truth sequence.
        pred_vectors: Per-frame flattened joint vectors for the predicted sequence.

    Returns:
        List of (gt_frame_idx, pred_frame_idx) pairs representing the optimal alignment.
    """
    gt_array   = np.array(gt_vectors,   dtype=np.float64)
    pred_array = np.array(pred_vectors, dtype=np.float64)
    return dtw_ndim.warping_path(gt_array, pred_array)


def compute_oks(
    gt_kpts: Dict[int, np.ndarray],
    pred_kpts: Dict[int, np.ndarray],
    joints: List[JointInfo],
) -> float:
    """
    Compute Object Keypoint Similarity (OKS) for a single aligned frame pair.

    Formula: OKS_i = exp(-d_i^2 / (2 * sigma_i^2))
    Scale is 1.0 since keypoints are already in torso-normalized space.
    Only joints visible in both sequences contribute to the score.

    Args:
        gt_kpts:   Ground truth normalized keypoints {coco_index: np.array([x, y])}.
        pred_kpts: Predicted normalized keypoints.
        joints:    JointInfo list defining which joints to include.

    Returns:
        Mean OKS in [0, 1]. Returns 0.0 if no valid joints are available.
    """
    scores = []
    for joint in joints:
        idx = joint.coco_index
        if idx not in gt_kpts or idx not in pred_kpts:
            continue
        dist_sq = float(np.sum((gt_kpts[idx] - pred_kpts[idx]) ** 2))
        scores.append(np.exp(-dist_sq / (2.0 * joint.sigma ** 2)))

    return float(np.mean(scores)) if scores else 0.0


def compute_mpjpe(
    gt_kpts: Dict[int, np.ndarray],
    pred_kpts: Dict[int, np.ndarray],
    joints: List[JointInfo],
) -> Dict[str, float]:
    """
    Compute per-joint Euclidean distance (MPJPE) in torso-normalized units.

    A value of 0.10 means the joint position differs by 10% of the torso height.

    Args:
        gt_kpts:   Ground truth normalized keypoints.
        pred_kpts: Predicted normalized keypoints.
        joints:    JointInfo list defining which joints to evaluate.

    Returns:
        Dict mapping joint name to Euclidean distance in torso units.
    """
    errors: Dict[str, float] = {}
    for joint in joints:
        idx = joint.coco_index
        if idx not in gt_kpts or idx not in pred_kpts:
            continue
        errors[joint.name] = float(np.linalg.norm(gt_kpts[idx] - pred_kpts[idx]))
    return errors


def compute_joint_angle(
    kpts: Dict[int, np.ndarray],
    proximal_idx: int,
    vertex_idx: int,
    distal_idx: int,
) -> Optional[float]:
    """
    Compute the interior angle in degrees at the vertex joint.

    Args:
        kpts:          Normalized keypoints dict.
        proximal_idx:  COCO index of the proximal joint (e.g. shoulder).
        vertex_idx:    COCO index of the vertex joint (e.g. elbow).
        distal_idx:    COCO index of the distal joint (e.g. wrist).

    Returns:
        Angle in degrees [0, 180], or None if any joint is missing or degenerate.
    """
    if proximal_idx not in kpts or vertex_idx not in kpts or distal_idx not in kpts:
        return None

    v1 = kpts[proximal_idx] - kpts[vertex_idx]
    v2 = kpts[distal_idx]   - kpts[vertex_idx]

    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < 1e-6 or norm2 < 1e-6:
        return None

    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_angle_error(
    gt_kpts: Dict[int, np.ndarray],
    pred_kpts: Dict[int, np.ndarray],
    angle_triplets: List[AngleTriplet],
) -> Dict[str, float]:
    """
    Compute absolute angle error in degrees for each joint angle triplet.

    Args:
        gt_kpts:        Ground truth normalized keypoints.
        pred_kpts:      Predicted normalized keypoints.
        angle_triplets: AngleTriplet list from the BodyRegion.

    Returns:
        Dict mapping triplet name to absolute angle error in degrees.
    """
    errors: Dict[str, float] = {}
    for triplet in angle_triplets:
        gt_angle   = compute_joint_angle(gt_kpts,   triplet.proximal_index, triplet.vertex_index, triplet.distal_index)
        pred_angle = compute_joint_angle(pred_kpts, triplet.proximal_index, triplet.vertex_index, triplet.distal_index)
        if gt_angle is not None and pred_angle is not None:
            errors[triplet.name] = abs(gt_angle - pred_angle)
    return errors
