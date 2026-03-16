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
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from body_region_config import REGION_REGISTRY, DEFAULT_REGIONS, BodyRegion
from movement_accuracy_utils import (
    extract_keypoints_from_video,
    extract_keypoints_from_image,
    is_image_path,
    normalize_pose,
    build_frame_vector,
    align_with_dtw,
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

        # Set by evaluate(); used by generate_comparison_image()
        self._gt_raw:   Optional[List] = None
        self._pred_raw: Optional[List] = None

    def _collect_joint_indices(self) -> List[int]:
        """Return a deduplicated ordered list of joint indices across all active regions."""
        seen = set()
        indices = []
        for region in self.regions:
            for joint in region.joints:
                if joint.coco_index not in seen:
                    indices.append(joint.coco_index)
                    seen.add(joint.coco_index)
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

        # Step 2: Build DTW-compatible sequences (normalized, valid frames only)
        gt_vectors,   gt_frame_indices   = self._build_valid_sequence(gt_raw,   joint_indices)
        pred_vectors, pred_frame_indices = self._build_valid_sequence(pred_raw, joint_indices)

        if not gt_vectors or not pred_vectors:
            raise RuntimeError(
                "Insufficient valid frames for evaluation. "
                "Check that the tracked joints are visible throughout the videos."
            )

        print(
            f"\nValid frames for DTW  GT: {len(gt_vectors)}"
            f"  Model: {len(pred_vectors)}"
        )

        # Step 3: DTW alignment
        print("Running DTW alignment...")
        warping_path = align_with_dtw(gt_vectors, pred_vectors)
        print(f"Alignment complete. Aligned frame pairs: {len(warping_path)}")

        # Step 4: Accumulate per-frame accuracy and error per triplet
        region_acc_list: Dict[str, Dict[str, List[float]]] = {
            r.name: {t.name: [] for t in r.angle_triplets} for r in self.regions
        }
        region_err_list: Dict[str, Dict[str, List[float]]] = {
            r.name: {t.name: [] for t in r.angle_triplets} for r in self.regions
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
                    for triplet_name, acc in acc_scores.items():
                        region_acc_list[region.name][triplet_name].append(acc)
                    for triplet_name, err in err_scores.items():
                        region_err_list[region.name][triplet_name].append(err)

        # Step 5: Aggregate
        results: Dict = {
            "overall_mean_accuracy": 0.0,
            "overall_mean_angle_error_deg": 0.0,
            "regions": {},
        }
        region_acc_means: List[float] = []
        region_err_means: List[float] = []

        for region in self.regions:
            triplet_entries: Dict[str, Dict] = {}
            all_accs: List[float] = []
            all_errs: List[float] = []

            for triplet_name in region_acc_list[region.name]:
                accs = region_acc_list[region.name][triplet_name]
                errs = region_err_list[region.name][triplet_name]
                mean_acc = float(np.mean(accs)) if accs else 0.0
                mean_err = float(np.mean(errs)) if errs else 0.0
                triplet_entries[triplet_name] = {
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
                "angles":               triplet_entries,
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
        overall_pct = results["overall_mean_accuracy"] * 100
        print(
            f"  Overall Accuracy         : "
            f"{results['overall_mean_accuracy']:.4f}  ({overall_pct:.1f}%)"
        )
        print(
            f"  Overall Mean Angle Error : "
            f"{results['overall_mean_angle_error_deg']:.2f} deg"
        )
        print()
        for region_name, rdata in results["regions"].items():
            region_pct = rdata["mean_accuracy"] * 100
            print(
                f"  {region_name:<20} "
                f"accuracy: {rdata['mean_accuracy']:.4f}  ({region_pct:.1f}%)"
                f"  |  mean error: {rdata['mean_angle_error_deg']:.2f} deg"
            )
            for angle_name, adata in rdata["angles"].items():
                angle_pct = adata["mean_accuracy"] * 100
                print(
                    f"    {angle_name:<30} "
                    f"{adata['mean_accuracy']:.4f}  ({angle_pct:.1f}%)"
                    f"  |  {adata['mean_angle_error_deg']:.2f} deg"
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

    def generate_comparison_image(self, results: Dict, output_path: Path) -> None:
        """
        Generate a side-by-side comparison image with arm keypoints overlaid.

        Only supported when both sources are image files. Raises RuntimeError
        for video inputs or if evaluate() has not been called.
        """
        if not (is_image_path(self.gt_source) and is_image_path(self.pred_source)):
            raise RuntimeError(
                "Comparison visualization is only supported for image inputs."
            )
        if self._gt_raw is None or self._pred_raw is None:
            raise RuntimeError(
                "evaluate() must be called before generate_comparison_image()."
            )

        gt_img   = cv2.imread(str(self.gt_source))
        pred_img = cv2.imread(str(self.pred_source))
        if gt_img is None or pred_img is None:
            raise RuntimeError("Failed to read source images for visualization.")

        gt_vis   = self._draw_keypoints_on_image(gt_img,   self._gt_raw[0])
        pred_vis = self._draw_keypoints_on_image(pred_img, self._pred_raw[0])

        target_h = 720
        gt_vis   = _resize_to_height(gt_vis,   target_h)
        pred_vis = _resize_to_height(pred_vis, target_h)

        header_h = 50
        gap      = 10
        total_w  = gt_vis.shape[1] + gap + pred_vis.shape[1]
        canvas   = np.zeros((target_h + header_h, total_w, 3), dtype=np.uint8)

        canvas[header_h:, :gt_vis.shape[1]]          = gt_vis
        canvas[header_h:, gt_vis.shape[1] + gap:]    = pred_vis

        font      = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        cv2.putText(canvas, "Ground Truth", (10, 34),
                    font, 0.8, (220, 220, 220), thickness)
        cv2.putText(canvas, "Model", (gt_vis.shape[1] + gap + 10, 34),
                    font, 0.8, (220, 220, 220), thickness)

        overall_pct = results["overall_mean_accuracy"] * 100
        acc_text    = f"Overall: {overall_pct:.1f}%"
        (tw, _), _  = cv2.getTextSize(acc_text, font, 0.8, thickness)
        cv2.putText(canvas, acc_text, (total_w - tw - 10, 34),
                    font, 0.8, (50, 220, 50), thickness)

        y_legend = header_h + 22
        for region_name in self.region_names:
            color = _REGION_COLORS.get(region_name, (180, 180, 180))
            rdata = results["regions"].get(region_name, {})
            pct   = rdata.get("mean_accuracy", 0.0) * 100
            label = f"{region_name}: {pct:.1f}%"
            cv2.rectangle(canvas, (10, y_legend - 12), (24, y_legend + 2), color, -1)
            cv2.putText(canvas, label, (30, y_legend), font, 0.55, (220, 220, 220), 1)
            y_legend += 22

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)
        print(f"Comparison image saved to: {output_path}")


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
