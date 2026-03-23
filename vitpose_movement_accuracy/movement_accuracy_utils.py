"""
Movement Accuracy Utility Functions

Provides keypoint extraction from images, torso-relative pose normalisation,
joint interior-angle error computation, skeleton visualisation, and batch
output helpers (directory pairing, slideshow generation).

This module operates on single images only.

Known limitations
-----------------
Camera perspective mismatch
    The metric compares 2D keypoint positions extracted from a real camera
    photograph and a Blender render.  Real cameras have lens distortion and
    varying focal lengths and cannot match Blender's camera exactly.  The
    same physical pose may therefore project to different pixel positions in
    each image, introducing systematic errors unrelated to pose quality.

ViTPose domain gap
    ViTPose was trained on photographs of real people.  Rendered avatars have
    a visually different appearance (flat lighting, uniform skin, no texture
    variation) which can degrade keypoint localisation on the model images and
    produce spuriously low or high confidence scores.

Unsigned interior angles are direction-blind
    The shoulder angle measures elevation (deviation from hanging straight
    down) but cannot distinguish whether the arm is pointing forward, sideways,
    or inward at the same elevation.  Two poses with the same elevation but
    opposite lateral directions will receive the same shoulder score.

Gaussian scoring inflates accuracy
    The formula exp(-e^2 / 2*sigma^2) maps to ~0.61 at sigma deg error.
    With sigma=15 deg the score for a 15-degree error is already 61%, which
    produces high overall accuracy even for poses with moderate deviations.

Single-frame snapshot dependency
    Images capture a moment in time.  Minor timing differences between when
    the GT photograph and the model render were taken can shift the apparent
    pose and affect scores independently of overall pose quality.

IMU sensor drift propagation
    The ground-truth pose is driven by IMU sensor data which accumulates drift
    over time.  The GT image may therefore not perfectly reflect the subject's
    actual joint angles at that moment, making the reference itself noisy.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from body_region_config import AngleTriplet, BodyRegion


# Minimum confidence score for a keypoint to be considered valid.
CONFIDENCE_THRESHOLD = 0.2

# Recognised image file extensions.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image_path(path: Path) -> bool:
    """Return True if the path points to a recognised image file type."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS


# COCO-WholeBody body joint indices used as normalisation anchors.
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


def extract_keypoints_from_image(
    image_path: str,
    device: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Extract keypoints from a single image using ViTPose-H (wholebody).

    Args:
        image_path: Path to the input image file.
        device: Device to run on ('cpu', 'cuda', 'mps', or None for auto-detect).

    Returns:
        np.ndarray of shape (133, 3) as (y, x, score) if a person was detected,
        or None if no person was detected.
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

    return kpts


def normalize_pose(frame_kpts: np.ndarray) -> Optional[Dict[int, np.ndarray]]:
    """
    Normalize keypoint positions to torso-relative coordinates.

    Anchors all positions to the hip midpoint and scales by the average
    shoulder-to-hip distance. This makes the metric invariant to subject
    scale and camera distance differences between the two images.

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


def _angle_between(
    a: np.ndarray,
    vertex: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute the unsigned interior angle (degrees) at vertex between vectors
    (vertex->a) and (vertex->b).

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


# ---------------------------------------------------------------------------
# Visualization constants
# ---------------------------------------------------------------------------

# BGR display colors per arm region, exported for use in comparison overlays.
REGION_COLORS: Dict[str, tuple] = {
    "left_arm":  (50, 200, 50),   # green
    "right_arm": (50, 130, 255),  # orange
}

# Fixed colors for shared torso landmarks so they are not overwritten by
# whichever arm region is drawn last.
_TORSO_JOINT_COLORS: Dict[int, tuple] = {
    5:  (50, 200, 50),    # left shoulder  -> green (matches left_arm)
    6:  (50, 130, 255),   # right shoulder -> orange (matches right_arm)
    11: (180, 180, 180),  # left hip       -> gray
    12: (180, 180, 180),  # right hip      -> gray
}

# Arm skeleton connections (shoulder->elbow->wrist->finger bases).
# Thumb uses CMC (92 left, 113 right); other fingers use MCP.
_REGION_CONNECTIONS: Dict[str, List[tuple]] = {
    "left_arm": [
        (5, 7), (7, 9),
        (9, 92), (9, 96), (9, 100), (9, 104), (9, 108),
    ],
    "right_arm": [
        (6, 8), (8, 10),
        (10, 113), (10, 117), (10, 121), (10, 125), (10, 129),
    ],
}

# Torso reference frame drawn in gray beneath arm overlays.
_TORSO_CONNECTIONS: List[tuple] = [
    (11, 12),  # hip bar
    (5, 6),    # shoulder bar
    (11, 5),   # left torso side
    (12, 6),   # right torso side
]
_TORSO_COLOR: tuple = (180, 180, 180)  # gray BGR


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------

def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Scale an image to the given height, preserving aspect ratio."""
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    return cv2.resize(image, (int(image.shape[1] * scale), height))


# ---------------------------------------------------------------------------
# Skeleton drawing
# ---------------------------------------------------------------------------

def draw_keypoints_on_image(
    image: np.ndarray,
    kpts: Optional[np.ndarray],
    regions: List[BodyRegion],
) -> np.ndarray:
    """
    Draw torso reference frame and arm skeleton onto a copy of image.

    Drawing order (back to front):
      1. Torso box (hip bar, shoulder bar, side lines) in gray.
      2. Arm connections (shoulder->elbow->wrist->fingers) in arm colors.
      3. Joint circles: shoulders and hips use fixed colors; arm joints use
         their region color.

    Args:
        image:   Source BGR image.
        kpts:    Raw keypoints array of shape (133, 3) as (y, x, score),
                 or None to return the image unchanged.
        regions: BodyRegion list used to determine which joints to draw.
    """
    img = image.copy()
    if kpts is None:
        return img

    def _pt(idx: int) -> tuple:
        return (int(kpts[idx][1]), int(kpts[idx][0]))

    def _visible(idx: int) -> bool:
        return idx < len(kpts) and kpts[idx][2] >= CONFIDENCE_THRESHOLD

    for i, j in _TORSO_CONNECTIONS:
        if _visible(i) and _visible(j):
            cv2.line(img, _pt(i), _pt(j), _TORSO_COLOR, 2)

    for region in regions:
        color = REGION_COLORS.get(region.name, (180, 180, 180))
        for i, j in _REGION_CONNECTIONS.get(region.name, []):
            if _visible(i) and _visible(j):
                cv2.line(img, _pt(i), _pt(j), color, 2)

    joint_color_map: Dict[int, tuple] = {}
    for region in regions:
        color = REGION_COLORS.get(region.name, (180, 180, 180))
        for joint in region.joints:
            if joint.coco_index not in joint_color_map:
                joint_color_map[joint.coco_index] = color
    joint_color_map.update(_TORSO_JOINT_COLORS)

    for idx, color in joint_color_map.items():
        if _visible(idx):
            cx, cy = _pt(idx)
            cv2.circle(img, (cx, cy), 6, color, -1)
            cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1)

    return img


# ---------------------------------------------------------------------------
# Batch output helpers
# ---------------------------------------------------------------------------

def pair_image_directories(
    gt_dir: Path,
    pred_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """
    Pair image files from two directories by the first numeric sequence in
    each filename. Files are matched when their numeric key is the same in
    both directories (e.g. pose_1_gt.jpg pairs with pose_1_model.jpg via "1").

    Returns a list of (key, gt_path, pred_path) tuples sorted by ascending
    numeric key value.
    """
    def _key(path: Path) -> Optional[str]:
        m = re.search(r'\d+', path.stem)
        return m.group(0) if m else None

    gt_map   = {_key(p): p for p in sorted(gt_dir.iterdir())   if is_image_path(p) and _key(p)}
    pred_map = {_key(p): p for p in sorted(pred_dir.iterdir()) if is_image_path(p) and _key(p)}
    common   = sorted(set(gt_map) & set(pred_map), key=lambda k: int(k))
    return [(k, gt_map[k], pred_map[k]) for k in common]


def generate_slideshow_video(
    image_paths: List[Path],
    output_path: Path,
    fps: int = 30,
    hold_seconds: int = 2,
) -> None:
    """
    Compile a list of comparison images into a slideshow MP4, holding each
    image for hold_seconds before advancing to the next.

    Images that differ in size from the first are resized to match.
    """
    if not image_paths:
        return

    first = cv2.imread(str(image_paths[0]))
    if first is None:
        print("Failed to read comparison images for slideshow.")
        return

    h, w             = first.shape[:2]
    frames_per_image = fps * hold_seconds
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        for _ in range(frames_per_image):
            writer.write(img)

    writer.release()
    print(f"Slideshow saved to: {output_path}")
