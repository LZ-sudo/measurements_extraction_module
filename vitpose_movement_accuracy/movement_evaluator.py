"""
Movement Accuracy Evaluator

Evaluates how accurately an animated model replicates the movement of a subject
using ViTPose pose estimation and DTW temporal alignment.
Accepts videos or images as input for both the ground truth and model sources.

Usage (CLI):
    python -m measurements_extraction_module.vitpose_movement_accuracy.movement_evaluator \
        --gt path/to/ground_truth.mp4 \
        --pred path/to/model_render.mp4 \
        [--regions left_arm right_arm] \
        [--device cuda]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from body_region_config import REGION_REGISTRY, DEFAULT_REGIONS, BodyRegion
from movement_accuracy_utils import (
    extract_keypoints_from_video,
    extract_keypoints_from_image,
    is_image_path,
    normalize_pose,
    build_frame_vector,
    compute_angle_error,
    compute_angle_accuracy,
    CONFIDENCE_THRESHOLD,
)

# BGR display colors per arm region
_REGION_COLORS: Dict[str, tuple] = {
    "left_arm":  (50, 200, 50),   # green
    "right_arm": (50, 130, 255),  # orange
}

# Fixed color assignments for shared torso landmarks so they are never
# overwritten by whichever arm region happens to be drawn last.
_TORSO_JOINT_COLORS: Dict[int, tuple] = {
    5:  (50, 200, 50),    # left shoulder  → green (matches left_arm)
    6:  (50, 130, 255),   # right shoulder → orange (matches right_arm)
    11: (180, 180, 180),  # left hip       → gray
    12: (180, 180, 180),  # right hip      → gray
}

# Arm-only skeleton connections (shoulder→elbow→wrist→finger bases).
# Thumb uses CMC (92 left, 113 right); other fingers use MCP.
# Hip-to-shoulder and cross-torso lines are drawn separately via _TORSO_CONNECTIONS.
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

# Torso reference frame connections drawn in gray beneath arm overlays.
_TORSO_CONNECTIONS: List[tuple] = [
    (11, 12),  # hip bar
    (5, 6),    # shoulder bar
    (11, 5),   # left torso side
    (12, 6),   # right torso side
]
_TORSO_COLOR: tuple = (180, 180, 180)  # gray BGR


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Scale an image to the given height, preserving aspect ratio."""
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    return cv2.resize(image, (int(image.shape[1] * scale), height))


class MovementEvaluator:
    """
    Evaluates movement replication accuracy between a ground truth source
    and a model source using pose-based metrics.
    Both sources can be videos or images (mixed inputs are supported).

    Pipeline:
        1. Extract ViTPose keypoints from both sources (per-frame for videos,
           single frame for images).
        2. Normalize poses to torso-relative coordinates.
        3. Align sequences temporally with DTW.
        4. Compute joint angle error over aligned frame pairs.
        5. Aggregate into per-region and per-angle summaries.
    """

    def __init__(
        self,
        gt_source: str,
        pred_source: str,
        regions: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            gt_source:   Path to the ground truth video or image of the subject.
            pred_source: Path to the animated model video or image.
            regions:     Body region names to evaluate. Defaults to
                         ["left_arm", "right_arm"].
                         Available: left_arm, right_arm.
            device:      Device to run on ('cpu', 'cuda', 'mps', or None for auto).
        """
        self.gt_source    = Path(gt_source)
        self.pred_source  = Path(pred_source)
        self.region_names = regions if regions is not None else DEFAULT_REGIONS
        self.device       = device

        invalid = [r for r in self.region_names if r not in REGION_REGISTRY]
        if invalid:
            raise ValueError(
                f"Unknown region(s): {invalid}. "
                f"Available: {list(REGION_REGISTRY.keys())}"
            )

        self.regions: List[BodyRegion] = [REGION_REGISTRY[r] for r in self.region_names]

        # Set by evaluate(); used by generate_comparison_image() / generate_comparison_video()
        self._gt_raw:            Optional[List]      = None
        self._pred_raw:          Optional[List]      = None
        self._warping_path:      Optional[List]      = None
        self._gt_frame_indices:  Optional[List[int]] = None
        self._pred_frame_indices: Optional[List[int]] = None

    def _collect_joint_indices(self) -> List[int]:
        """
        Return a deduplicated ordered list of joint indices required for frame
        filtering and angle computation.

        Only indices referenced in angle_triplets are included. Visualization-only
        joints (e.g. finger base chains) are intentionally excluded so that low-
        confidence finger detections do not disqualify an otherwise valid frame.
        Torso anchors (5, 6, 11, 12) are included implicitly via the shoulder
        angle triplets but are also added explicitly as a safety net for
        normalize_pose.
        """
        seen = set()
        indices = []
        for region in self.regions:
            for triplet in region.angle_triplets:
                for idx in (triplet.vertex_index, triplet.a_index, triplet.b_index):
                    if idx not in seen:
                        indices.append(idx)
                        seen.add(idx)
        # Explicit torso anchors required by normalize_pose
        for idx in (5, 6, 11, 12):
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
        return indices

    def _build_valid_sequence(
        self,
        frame_keypoints,
        joint_indices: List[int],
    ) -> tuple[List[np.ndarray], List[int]]:
        """
        Build a normalized frame-vector sequence, skipping frames where any
        tracked joint is missing or below the confidence threshold.

        Returns:
            Tuple of (vectors, original_frame_indices).
        """
        vectors: List[np.ndarray] = []
        valid_frame_indices: List[int] = []

        for frame_idx, kpts in enumerate(frame_keypoints):
            if kpts is None:
                continue
            normalized = normalize_pose(kpts)
            if normalized is None:
                continue
            vec = build_frame_vector(normalized, joint_indices)
            if vec is not None:
                vectors.append(vec)
                valid_frame_indices.append(frame_idx)

        return vectors, valid_frame_indices

    def evaluate(self) -> Dict:
        """
        Run the full evaluation pipeline.

        Returns:
            Dict with the following structure:
            {
                "overall_mean_accuracy": float,        # Mean accuracy across all regions [0, 1]
                "overall_mean_angle_error_deg": float, # Mean angle error across all regions
                "regions": {
                    "<region_name>": {
                        "mean_accuracy": float,
                        "mean_angle_error_deg": float,
                        "angles": {
                            "<triplet_name>": {
                                "mean_accuracy": float,        # Gaussian-kernel accuracy [0, 1]
                                "mean_angle_error_deg": float, # Mean absolute error in degrees
                            },
                            ...
                        }
                    },
                    ...
                }
            }
        """
        joint_indices = self._collect_joint_indices()

        # Step 1: Keypoint extraction (dispatch based on source file type)
        print("=== Ground Truth Source ===")
        if is_image_path(self.gt_source):
            gt_raw = extract_keypoints_from_image(str(self.gt_source), self.device)
        else:
            gt_raw = extract_keypoints_from_video(str(self.gt_source), self.device)

        print("\n=== Model Source ===")
        if is_image_path(self.pred_source):
            pred_raw = extract_keypoints_from_image(str(self.pred_source), self.device)
        else:
            pred_raw = extract_keypoints_from_video(str(self.pred_source), self.device)

        self._gt_raw   = gt_raw
        self._pred_raw = pred_raw

        # Step 2: Build frame pairing
        # For image inputs, bypass frame vector building entirely — the metric
        # functions already skip individual bones whose endpoints are missing,
        # so there is no need for the all-or-nothing joint gate used by
        # build_frame_vector (which was designed for DTW alignment).
        # For video inputs, filter to frames where all tracked joints are visible.
        if is_image_path(self.gt_source) and is_image_path(self.pred_source):
            gt_frame_indices         = [0]
            pred_frame_indices       = [0]
            self._gt_frame_indices   = gt_frame_indices
            self._pred_frame_indices = pred_frame_indices
            warping_path = [(0, 0)]
        else:
            gt_vectors,   gt_frame_indices   = self._build_valid_sequence(gt_raw,   joint_indices)
            pred_vectors, pred_frame_indices = self._build_valid_sequence(pred_raw, joint_indices)

            self._gt_frame_indices   = gt_frame_indices
            self._pred_frame_indices = pred_frame_indices

            if not gt_vectors or not pred_vectors:
                raise RuntimeError(
                    "Insufficient valid frames for evaluation. "
                    "Check that the tracked joints are visible throughout the videos."
                )

            print(f"\nValid frames  GT: {len(gt_vectors)}  Model: {len(pred_vectors)}")
            n = min(len(gt_vectors), len(pred_vectors))
            warping_path = list(zip(range(n), range(n)))

        self._warping_path = warping_path

        # Step 4: Accumulate per-frame accuracy and error per angle triplet
        region_acc_list: Dict[str, Dict[str, List[float]]] = {
            r.name: {triplet.name: [] for triplet in r.angle_triplets}
            for r in self.regions
        }
        region_err_list: Dict[str, Dict[str, List[float]]] = {
            r.name: {triplet.name: [] for triplet in r.angle_triplets}
            for r in self.regions
        }

        for gt_vec_idx, pred_vec_idx in warping_path:
            gt_kpts_raw   = gt_raw[gt_frame_indices[gt_vec_idx]]
            pred_kpts_raw = pred_raw[pred_frame_indices[pred_vec_idx]]

            if gt_kpts_raw is None or pred_kpts_raw is None:
                continue

            gt_norm   = normalize_pose(gt_kpts_raw)
            pred_norm = normalize_pose(pred_kpts_raw)

            if gt_norm is None or pred_norm is None:
                continue

            for region in self.regions:
                if region.angle_triplets:
                    acc_scores = compute_angle_accuracy(gt_norm, pred_norm, region.angle_triplets)
                    err_scores = compute_angle_error(gt_norm, pred_norm, region.angle_triplets)
                    for name, acc in acc_scores.items():
                        region_acc_list[region.name][name].append(acc)
                    for name, err in err_scores.items():
                        region_err_list[region.name][name].append(err)

        # Guard: if no metrics were accumulated at all raise so the caller
        # can mark this pair as UNKNOWN rather than show 0%.
        any_metrics = any(
            len(accs) > 0
            for r_accs in region_acc_list.values()
            for accs in r_accs.values()
        )
        if not any_metrics:
            raise RuntimeError(
                "No angle metrics could be computed. "
                "Check that the torso and arm joints are visible in both images."
            )

        # Step 5: Aggregate
        results: Dict = {
            "overall_mean_accuracy":        0.0,
            "overall_mean_angle_error_deg": 0.0,
            "regions": {},
        }
        region_acc_means: List[float] = []
        region_err_means: List[float] = []

        for region in self.regions:
            angle_entries: Dict[str, Dict] = {}
            all_accs: List[float] = []
            all_errs: List[float] = []

            for triplet_name in region_acc_list[region.name]:
                accs = region_acc_list[region.name][triplet_name]
                errs = region_err_list[region.name][triplet_name]
                mean_acc = float(np.mean(accs)) if accs else 0.0
                mean_err = float(np.mean(errs)) if errs else 0.0
                angle_entries[triplet_name] = {
                    "mean_accuracy":        mean_acc,
                    "mean_angle_error_deg": mean_err,
                }
                all_accs.extend(accs)
                all_errs.extend(errs)

            region_mean_acc = float(np.mean(all_accs)) if all_accs else 0.0
            region_mean_err = float(np.mean(all_errs)) if all_errs else 0.0
            region_acc_means.append(region_mean_acc)
            region_err_means.append(region_mean_err)

            results["regions"][region.name] = {
                "mean_accuracy":        region_mean_acc,
                "mean_angle_error_deg": region_mean_err,
                "angles":               angle_entries,
            }

        results["overall_mean_accuracy"] = (
            float(np.mean(region_acc_means)) if region_acc_means else 0.0
        )
        results["overall_mean_angle_error_deg"] = (
            float(np.mean(region_err_means)) if region_err_means else 0.0
        )

        return results

    def print_results(self, results: Dict) -> None:
        """Print evaluation results in a formatted layout."""
        separator = "=" * 62
        print(f"\n{separator}")
        print("  MOVEMENT ACCURACY EVALUATION RESULTS")
        print(separator)
        print(f"  Ground Truth : {self.gt_source.name}")
        print(f"  Model Source : {self.pred_source.name}")
        print(f"  Regions      : {', '.join(self.region_names)}")
        print("-" * 62)
        print(
            f"  Overall Accuracy          : "
            f"{results['overall_mean_accuracy'] * 100:.1f}%"
        )
        print(
            f"  Overall Mean Angle Error  : "
            f"{results['overall_mean_angle_error_deg']:.1f} deg"
        )
        print()
        for region_name, rdata in results["regions"].items():
            print(
                f"  {region_name:<20} "
                f"accuracy: {rdata['mean_accuracy'] * 100:.1f}%"
                f"  |  mean error: {rdata['mean_angle_error_deg']:.1f} deg"
            )
            for triplet_name, adata in rdata["angles"].items():
                print(
                    f"    {triplet_name:<34} "
                    f"{adata['mean_accuracy'] * 100:.1f}%"
                    f"  |  {adata['mean_angle_error_deg']:.1f} deg"
                )
            print()
        print(separator)

    def save_results(self, results: Dict, output_path: Path) -> None:
        """Serialize evaluation results and source metadata to a JSON file."""
        payload = {
            "gt_source":  str(self.gt_source),
            "pred_source": str(self.pred_source),
            "regions": self.region_names,
            **results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to: {output_path}")

    def _draw_keypoints_on_image(
        self,
        image: np.ndarray,
        kpts: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Draw torso reference frame and arm skeleton onto a copy of image.

        Drawing order (back to front):
          1. Torso box (hip bar, shoulder bar, side lines) in gray.
          2. Arm connections (shoulder→elbow→wrist→fingers) in arm colors.
          3. Joint circles with fixed colors: shoulders match their arm,
             hips are gray, all other arm joints match their arm.
        """
        img = image.copy()
        if kpts is None:
            return img

        def _pt(idx: int):
            return (int(kpts[idx][1]), int(kpts[idx][0]))  # (x, y) from (y, x, score)

        def _visible(idx: int) -> bool:
            return idx < len(kpts) and kpts[idx][2] >= CONFIDENCE_THRESHOLD

        # Step 1: Draw torso reference frame in gray
        for i, j in _TORSO_CONNECTIONS:
            if _visible(i) and _visible(j):
                cv2.line(img, _pt(i), _pt(j), _TORSO_COLOR, 2)

        # Step 2: Draw arm-specific connections in arm colors
        for region in self.regions:
            color = _REGION_COLORS.get(region.name, (180, 180, 180))
            for i, j in _REGION_CONNECTIONS.get(region.name, []):
                if _visible(i) and _visible(j):
                    cv2.line(img, _pt(i), _pt(j), color, 2)

        # Step 3: Build a per-index color map so torso landmarks keep their
        # fixed colors regardless of which region is iterated last.
        joint_color_map: Dict[int, tuple] = {}
        for region in self.regions:
            color = _REGION_COLORS.get(region.name, (180, 180, 180))
            for joint in region.joints:
                idx = joint.coco_index
                if idx not in joint_color_map:
                    joint_color_map[idx] = color
        # Overwrite with fixed torso colors (shoulders + hips)
        joint_color_map.update(_TORSO_JOINT_COLORS)

        # Step 4: Draw joint circles
        for idx, color in joint_color_map.items():
            if _visible(idx):
                cx, cy = _pt(idx)
                cv2.circle(img, (cx, cy), 6, color, -1)
                cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1)

        return img

    def generate_comparison_image(
        self,
        results: Optional[Dict],
        output_path: Path,
    ) -> None:
        """
        Generate a side-by-side comparison image with arm keypoints overlaid.

        results may be None when evaluation failed (e.g. insufficient landmark
        confidence). In that case keypoints are still drawn where detectable
        and the accuracy indicator shows UNKNOWN.
        """
        if not (is_image_path(self.gt_source) and is_image_path(self.pred_source)):
            raise RuntimeError(
                "Comparison visualization is only supported for image inputs."
            )

        gt_img   = cv2.imread(str(self.gt_source))
        pred_img = cv2.imread(str(self.pred_source))
        if gt_img is None or pred_img is None:
            raise RuntimeError("Failed to read source images for visualization.")

        gt_kpts   = self._gt_raw[0]   if self._gt_raw   is not None else None
        pred_kpts = self._pred_raw[0] if self._pred_raw is not None else None

        gt_vis   = self._draw_keypoints_on_image(gt_img,   gt_kpts)
        pred_vis = self._draw_keypoints_on_image(pred_img, pred_kpts)

        target_h = 720
        gt_vis   = _resize_to_height(gt_vis,   target_h)
        pred_vis = _resize_to_height(pred_vis, target_h)

        header_h = 50
        gap      = 10
        total_w  = gt_vis.shape[1] + gap + pred_vis.shape[1]
        canvas   = np.zeros((target_h + header_h, total_w, 3), dtype=np.uint8)

        canvas[header_h:, :gt_vis.shape[1]]       = gt_vis
        canvas[header_h:, gt_vis.shape[1] + gap:] = pred_vis

        font      = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        cv2.putText(canvas, "Ground Truth", (10, 34),
                    font, 0.8, (220, 220, 220), thickness)
        cv2.putText(canvas, "Model", (gt_vis.shape[1] + gap + 10, 34),
                    font, 0.8, (220, 220, 220), thickness)

        if results is not None:
            overall_pct = results["overall_mean_accuracy"] * 100
            overall_err = results["overall_mean_angle_error_deg"]
            acc_text    = f"Accuracy: {overall_pct:.1f}%"
            sub_text    = f"Angle error: {overall_err:.1f} deg"
            acc_color   = (50, 220, 50)
        else:
            acc_text  = "Accuracy: UNKNOWN"
            sub_text  = ""
            acc_color = (80, 80, 220)

        (tw, _), _ = cv2.getTextSize(acc_text, font, 0.8, thickness)
        cv2.putText(canvas, acc_text, (total_w - tw - 10, 34),
                    font, 0.8, acc_color, thickness)
        if sub_text:
            (stw, _), _ = cv2.getTextSize(sub_text, font, 0.55, 1)
            cv2.putText(canvas, sub_text, (total_w - stw - 10, header_h + 18),
                        font, 0.55, (160, 220, 160), 1)

        y_legend = header_h + 22
        for region_name in self.region_names:
            color = _REGION_COLORS.get(region_name, (180, 180, 180))
            if results is not None:
                rdata = results["regions"].get(region_name, {})
                pct   = rdata.get("mean_accuracy", 0.0) * 100
                err   = rdata.get("mean_angle_error_deg", 0.0)
                label = f"{region_name}: {pct:.1f}%  ({err:.1f} deg)"
            else:
                label = f"{region_name}: UNKNOWN"
            cv2.rectangle(canvas, (10, y_legend - 12), (24, y_legend + 2), color, -1)
            cv2.putText(canvas, label, (30, y_legend), font, 0.5, (220, 220, 220), 1)
            y_legend += 22

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)
        print(f"Comparison image saved to: {output_path}")


    def generate_comparison_video(self, results: Dict, output_path: Path) -> None:
        """
        Generate a side-by-side comparison video with arm keypoints overlaid,
        using DTW-aligned frame pairs so corresponding poses are shown together.

        Raises RuntimeError if evaluate() has not been called first.
        """
        if (
            self._gt_raw is None
            or self._pred_raw is None
            or self._warping_path is None
            or self._gt_frame_indices is None
            or self._pred_frame_indices is None
        ):
            raise RuntimeError(
                "evaluate() must be called before generate_comparison_video()."
            )

        def _read_all_frames(path: Path) -> List[np.ndarray]:
            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            return frames

        def _get_fps(path: Path) -> float:
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else 30.0

        print("Reading GT video frames for comparison video...")
        gt_frames   = _read_all_frames(self.gt_source)   if not is_image_path(self.gt_source)   else [cv2.imread(str(self.gt_source))]
        print("Reading model video frames for comparison video...")
        pred_frames = _read_all_frames(self.pred_source) if not is_image_path(self.pred_source) else [cv2.imread(str(self.pred_source))]

        if not gt_frames or not pred_frames or gt_frames[0] is None or pred_frames[0] is None:
            raise RuntimeError("Failed to read frames from source files.")

        fps = _get_fps(self.gt_source) if not is_image_path(self.gt_source) else _get_fps(self.pred_source)

        target_h = 720
        header_h = 50
        gap      = 10

        gt_sample   = _resize_to_height(gt_frames[0],   target_h)
        pred_sample = _resize_to_height(pred_frames[0], target_h)
        total_w = gt_sample.shape[1] + gap + pred_sample.shape[1]
        total_h = target_h + header_h

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (total_w, total_h))

        font      = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        overall_pct = results["overall_mean_accuracy"] * 100
        acc_text    = f"Accuracy: {overall_pct:.1f}%"
        (tw, _), _  = cv2.getTextSize(acc_text, font, 0.8, thickness)

        print(f"Writing {len(self._warping_path)} aligned frames to comparison video...")

        for gt_vec_idx, pred_vec_idx in self._warping_path:
            gt_frame_idx   = self._gt_frame_indices[gt_vec_idx]
            pred_frame_idx = self._pred_frame_indices[pred_vec_idx]

            gt_frame   = gt_frames[gt_frame_idx]   if gt_frame_idx   < len(gt_frames)   else gt_frames[-1]
            pred_frame = pred_frames[pred_frame_idx] if pred_frame_idx < len(pred_frames) else pred_frames[-1]

            gt_kpts   = self._gt_raw[gt_frame_idx]
            pred_kpts = self._pred_raw[pred_frame_idx]

            gt_vis   = _resize_to_height(self._draw_keypoints_on_image(gt_frame,   gt_kpts),   target_h)
            pred_vis = _resize_to_height(self._draw_keypoints_on_image(pred_frame, pred_kpts), target_h)

            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            canvas[header_h:, :gt_vis.shape[1]]       = gt_vis
            canvas[header_h:, gt_vis.shape[1] + gap:] = pred_vis

            cv2.putText(canvas, "Ground Truth", (10, 34),
                        font, 0.8, (220, 220, 220), thickness)
            cv2.putText(canvas, "Model", (gt_vis.shape[1] + gap + 10, 34),
                        font, 0.8, (220, 220, 220), thickness)
            cv2.putText(canvas, acc_text, (total_w - tw - 10, 34),
                        font, 0.8, (50, 220, 50), thickness)

            y_legend = header_h + 22
            for region_name in self.region_names:
                color = _REGION_COLORS.get(region_name, (180, 180, 180))
                rdata = results["regions"].get(region_name, {})
                pct   = rdata.get("mean_accuracy", 0.0) * 100
                err   = rdata.get("mean_angle_error_deg", 0.0)
                label = f"{region_name}: {pct:.1f}%  ({err:.1f} deg)"
                cv2.rectangle(canvas, (10, y_legend - 12), (24, y_legend + 2), color, -1)
                cv2.putText(canvas, label, (30, y_legend), font, 0.55, (220, 220, 220), 1)
                y_legend += 22

            writer.write(canvas)

        writer.release()
        print(f"Comparison video saved to: {output_path}")


def _pair_image_directories(
    gt_dir: Path,
    pred_dir: Path,
) -> List[Tuple[str, Path, Path]]:
    """
    Pair image files from two directories by the first numeric sequence found
    in each filename. Files are matched when their numeric key is the same in
    both directories (e.g. pose_1_gt.jpg pairs with pose_1_model.jpg via key "1").

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


def _generate_slideshow_video(
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

    h, w              = first.shape[:2]
    frames_per_image  = fps * hold_seconds
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate movement replication accuracy between two sources using ViTPose."
    )
    parser.add_argument(
        "--gt", required=True,
        help="Path to the ground truth source (video or image) of the subject.",
    )
    parser.add_argument(
        "--pred", required=True,
        help="Path to the animated model source (video or image).",
    )
    parser.add_argument(
        "--regions", nargs="+", default=None,
        metavar="REGION",
        help=(
            "Body regions to evaluate. "
            "Choices: left_arm right_arm. "
            "Defaults to: left_arm right_arm."
        ),
    )
    parser.add_argument(
        "--device", default=None, choices=["cpu", "cuda"],
        help="Inference device (default: auto-detect).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        metavar="DIR",
        help=(
            "Directory to save outputs. "
            "Always writes a JSON report; also writes a comparison PNG for image inputs."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    gt_path   = Path(args.gt)
    pred_path = Path(args.pred)

    if gt_path.is_dir() and pred_path.is_dir():
        # --- Directory mode: batch process paired images ---
        pairs = _pair_image_directories(gt_path, pred_path)
        if not pairs:
            print("No matching image pairs found in the provided directories.")
            sys.exit(1)

        print(f"Found {len(pairs)} image pair(s): {[k for k, _, _ in pairs]}")

        output_dir = Path(args.output_dir) if args.output_dir else gt_path.parent / "comparison_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_image_paths: List[Path] = []

        for key, gt_file, pred_file in pairs:
            print(f"\n{'=' * 62}")
            print(f"  Pair {key}:  {gt_file.name}  vs  {pred_file.name}")
            print(f"{'=' * 62}")

            evaluator = MovementEvaluator(
                gt_source=str(gt_file),
                pred_source=str(pred_file),
                regions=args.regions,
                device=args.device,
            )

            results = None
            try:
                results = evaluator.evaluate()
                evaluator.print_results(results)
                evaluator.save_results(results, output_dir / f"{key}_results.json")
            except RuntimeError as exc:
                print(f"  Evaluation failed: {exc}")

            comp_path = output_dir / f"{key}_comparison.png"
            evaluator.generate_comparison_image(results, comp_path)
            comparison_image_paths.append(comp_path)

        _generate_slideshow_video(
            comparison_image_paths,
            output_dir / "comparison_slideshow.mp4",
        )

    else:
        # --- Single file mode ---
        evaluator = MovementEvaluator(
            gt_source=args.gt,
            pred_source=args.pred,
            regions=args.regions,
            device=args.device,
        )

        results = evaluator.evaluate()
        evaluator.print_results(results)

        if args.output_dir:
            output_dir = Path(args.output_dir)
            stem = f"{evaluator.gt_source.stem}_vs_{evaluator.pred_source.stem}"
            evaluator.save_results(results, output_dir / f"{stem}_results.json")
            if is_image_path(evaluator.gt_source) and is_image_path(evaluator.pred_source):
                evaluator.generate_comparison_image(
                    results, output_dir / f"{stem}_comparison.png"
                )
            else:
                evaluator.generate_comparison_video(
                    results, output_dir / f"{stem}_comparison.mp4"
                )
