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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from body_region_config import REGION_REGISTRY, DEFAULT_REGIONS, BodyRegion
from movement_accuracy_utils import (
    extract_keypoints_from_video,
    extract_keypoints_from_image,
    is_image_path,
    normalize_pose,
    build_frame_vector,
    align_with_dtw,
    compute_oks,
    compute_mpjpe,
    compute_angle_error,
)


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
        4. Compute OKS, MPJPE, and joint angle error over aligned frame pairs.
        5. Aggregate into per-region and per-joint summaries.
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
    ) -> Tuple[List[np.ndarray], List[int]]:
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
                "overall_oks": float,        # Mean OKS across all regions [0, 1]
                "regions": {
                    "<region_name>": {
                        "oks": float,
                        "mean_angle_error_deg": float   # only for regions with angle triplets
                    },
                    ...
                },
                "joints": {
                    "<joint_name>": {
                        "oks":        float,   # Mean OKS for this joint [0, 1]
                        "mean_mpjpe": float,   # Mean distance in torso units
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

        # Step 4: Compute metrics over aligned frame pairs
        region_oks_list:    Dict[str, List[float]] = {r.name: [] for r in self.regions}
        region_angle_list:  Dict[str, List[float]] = {r.name: [] for r in self.regions}
        joint_oks_list:     Dict[str, List[float]] = {}
        joint_mpjpe_list:   Dict[str, List[float]] = {}

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
                # Region-level OKS
                region_oks_list[region.name].append(
                    compute_oks(gt_norm, pred_norm, region.joints)
                )

                # Angle error (only for regions with defined angle triplets, e.g. arms/legs)
                if region.angle_triplets:
                    angle_errs = compute_angle_error(gt_norm, pred_norm, region.angle_triplets)
                    region_angle_list[region.name].extend(angle_errs.values())

                # Per-joint OKS and MPJPE
                mpjpe = compute_mpjpe(gt_norm, pred_norm, region.joints)
                for joint in region.joints:
                    joint_oks_list.setdefault(joint.name, []).append(
                        compute_oks(gt_norm, pred_norm, [joint])
                    )
                    if joint.name in mpjpe:
                        joint_mpjpe_list.setdefault(joint.name, []).append(mpjpe[joint.name])

        # Step 5: Aggregate
        results: Dict = {"overall_oks": 0.0, "regions": {}, "joints": {}}
        region_mean_oks: List[float] = []

        for region in self.regions:
            oks_vals   = region_oks_list[region.name]
            mean_oks   = float(np.mean(oks_vals)) if oks_vals else 0.0
            region_mean_oks.append(mean_oks)

            region_entry: Dict = {"oks": mean_oks}
            angle_vals = region_angle_list[region.name]
            if angle_vals:
                region_entry["mean_angle_error_deg"] = float(np.mean(angle_vals))

            results["regions"][region.name] = region_entry

        results["overall_oks"] = float(np.mean(region_mean_oks)) if region_mean_oks else 0.0

        for joint_name in joint_oks_list:
            results["joints"][joint_name] = {
                "oks":        float(np.mean(joint_oks_list[joint_name])),
                "mean_mpjpe": float(np.mean(joint_mpjpe_list.get(joint_name, [0.0]))),
            }

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
        overall_pct = results["overall_oks"] * 100
        print(f"  Overall OKS  : {results['overall_oks']:.4f}  ({overall_pct:.1f}%)")
        print()
        print("  Per-Region Breakdown:")
        for region_name, rdata in results["regions"].items():
            pct  = rdata["oks"] * 100
            line = f"    {region_name:<20} OKS: {rdata['oks']:.4f}  ({pct:.1f}%)"
            if "mean_angle_error_deg" in rdata:
                line += f"  |  angle error: {rdata['mean_angle_error_deg']:.2f} deg"
            print(line)
        print()
        print("  Per-Joint Detail:")
        for joint_name, jdata in results["joints"].items():
            pct = jdata["oks"] * 100
            print(
                f"    {joint_name:<26} OKS: {jdata['oks']:.4f}  ({pct:.1f}%)"
                f"  |  MPJPE: {jdata['mean_mpjpe']:.4f} torso units"
            )
        print(separator)


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
        "--device", default=None, choices=["cpu", "cuda", "mps"],
        help="Inference device (default: auto-detect).",
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
