"""
Movement Accuracy Evaluator

Evaluates how accurately an animated model replicates the movement of a subject
using ViTPose pose estimation and joint interior-angle comparison.
Accepts single images as input for both ground truth and model sources.

Accuracy is measured via the unsigned interior angle at each joint (shoulder
elevation, elbow flexion) rather than absolute keypoint position. This makes
the metric invariant to the subject's global position, scale, and camera
distance.

Known limitations
-----------------
Camera perspective mismatch
    The metric compares 2D keypoints from a real camera photograph and a
    Blender render.  Because the two viewpoints cannot be perfectly matched,
    the same physical pose may project to different 2D positions in each
    image, introducing errors unrelated to actual pose quality.  This is the
    primary reason scores may not rank models intuitively.

ViTPose domain gap
    ViTPose was trained on photographs of real people.  Blender renders have
    a visually different appearance (flat lighting, uniform skin) which can
    degrade keypoint localisation on model images and produce incorrect or
    low-confidence detections.

Unsigned interior angles are direction-blind
    The shoulder angle measures elevation only.  Two poses where the arm is
    at the same elevation but pointing in opposite lateral directions receive
    the same shoulder score, because the unsigned angle cannot distinguish
    direction.

Single-frame snapshot dependency
    Each evaluation compares one GT photograph to one model render.  Minor
    timing differences between captures can shift the apparent pose
    independently of model quality.

IMU sensor drift propagation
    Ground-truth poses are driven by IMU data which accumulates drift over time.
    The GT image may therefore not perfectly reflect the subject's actual joint
    angles, making the reference itself noisy.

Usage (CLI):
    # Single pair
    python -m measurements_extraction_module.vitpose_movement_accuracy.movement_evaluator \
        --gt path/to/ground_truth.jpg \
        --pred path/to/model_render.jpg \
        [--regions left_arm right_arm] \
        [--device cuda]

    # Directory batch
    python -m measurements_extraction_module.vitpose_movement_accuracy.movement_evaluator \
        --gt path/to/gt_dir/ \
        --pred path/to/pred_dir/ \
        [--regions left_arm right_arm] \
        [--output-dir path/to/output/]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from body_region_config import REGION_REGISTRY, DEFAULT_REGIONS, BodyRegion
from movement_accuracy_utils import (
    extract_keypoints_from_image,
    is_image_path,
    normalize_pose,
    compute_angle_error,
    compute_angle_accuracy,
    draw_keypoints_on_image,
    resize_to_height,
    pair_image_directories,
    generate_slideshow_video,
    REGION_COLORS,
)


class MovementEvaluator:
    """
    Evaluates movement replication accuracy between a ground truth image
    and a model image using joint interior-angle metrics.

    Pipeline:
        1. Extract ViTPose keypoints from both images.
        2. Normalize poses to torso-relative coordinates (hip midpoint origin,
           shoulder-to-hip scale).
        3. Compute unsigned interior-angle error (degrees) at each tracked
           joint, skipping triplets where any keypoint is below the confidence
           threshold.
        4. Convert errors to Gaussian-kernel accuracy scores.
        5. Aggregate into per-region and overall summaries.
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
            gt_source:   Path to the ground truth image of the subject.
            pred_source: Path to the animated model image.
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
        self._gt_raw:   Optional[np.ndarray] = None
        self._pred_raw: Optional[np.ndarray] = None

    def evaluate(self) -> Dict:
        """
        Run the full evaluation pipeline on the configured image pair.

        Returns:
            {
                "overall_mean_accuracy":        float,
                "overall_mean_angle_error_deg": float,
                "regions": {
                    "<region_name>": {
                        "mean_accuracy":        float,
                        "mean_angle_error_deg": float,
                        "angles": {
                            "<triplet_name>": {
                                "mean_accuracy":        float,
                                "mean_angle_error_deg": float,
                            },
                            ...
                        }
                    },
                    ...
                }
            }

        Raises:
            RuntimeError: If either source is not an image, no person is
                detected, torso anchors are missing, or no angle metrics
                could be computed.
        """
        if not is_image_path(self.gt_source):
            raise RuntimeError(
                f"GT source is not a supported image file: {self.gt_source}."
            )
        if not is_image_path(self.pred_source):
            raise RuntimeError(
                f"Predicted source is not a supported image file: {self.pred_source}."
            )

        # Step 1: Keypoint extraction
        print("=== Ground Truth Source ===")
        gt_raw = extract_keypoints_from_image(str(self.gt_source), self.device)

        print("\n=== Model Source ===")
        pred_raw = extract_keypoints_from_image(str(self.pred_source), self.device)

        self._gt_raw   = gt_raw
        self._pred_raw = pred_raw

        if gt_raw is None or pred_raw is None:
            raise RuntimeError(
                "No person detected in one or both source images. "
                "Check that the subject is clearly visible."
            )

        # Step 2: Normalize poses
        gt_norm   = normalize_pose(gt_raw)
        pred_norm = normalize_pose(pred_raw)

        if gt_norm is None or pred_norm is None:
            raise RuntimeError(
                "Could not normalize pose: torso anchor keypoints (hips, shoulders) "
                "are missing or below confidence threshold in one or both images."
            )

        # Step 3: Compute per-triplet accuracy and error
        region_acc: Dict[str, Dict[str, float]] = {}
        region_err: Dict[str, Dict[str, float]] = {}

        for region in self.regions:
            if not region.angle_triplets:
                continue
            region_acc[region.name] = compute_angle_accuracy(gt_norm, pred_norm, region.angle_triplets)
            region_err[region.name] = compute_angle_error(gt_norm, pred_norm, region.angle_triplets)

        if not any(region_acc.values()):
            raise RuntimeError(
                "No angle metrics could be computed. "
                "Check that the torso and arm joints are visible in both images."
            )

        # Step 4: Aggregate
        region_acc_means: List[float] = []
        region_err_means: List[float] = []
        results: Dict = {
            "overall_mean_accuracy":        0.0,
            "overall_mean_angle_error_deg": 0.0,
            "regions":                      {},
        }

        for region in self.regions:
            computed_acc = region_acc.get(region.name, {})
            computed_err = region_err.get(region.name, {})

            region_mean_acc = float(np.mean(list(computed_acc.values()))) if computed_acc else 0.0
            region_mean_err = float(np.mean(list(computed_err.values()))) if computed_err else 0.0
            region_acc_means.append(region_mean_acc)
            region_err_means.append(region_mean_err)

            angle_entries: Dict[str, Dict] = {}
            for triplet in region.angle_triplets:
                t_name = triplet.name
                if t_name in computed_acc:
                    angle_entries[t_name] = {
                        "mean_accuracy":        computed_acc[t_name],
                        "mean_angle_error_deg": computed_err[t_name],
                    }
                else:
                    angle_entries[t_name] = {
                        "mean_accuracy":        None,
                        "mean_angle_error_deg": None,
                        "note":                 "skipped: keypoint below confidence threshold",
                    }

            results["regions"][region.name] = {
                "mean_accuracy":        region_mean_acc,
                "mean_angle_error_deg": region_mean_err,
                "angles":               angle_entries,
            }

        results["overall_mean_accuracy"]        = float(np.mean(region_acc_means)) if region_acc_means else 0.0
        results["overall_mean_angle_error_deg"] = float(np.mean(region_err_means)) if region_err_means else 0.0

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
        print(f"  Overall Accuracy         : {results['overall_mean_accuracy'] * 100:.1f}%")
        print(f"  Overall Mean Angle Error : {results['overall_mean_angle_error_deg']:.1f} deg")
        print()
        for region_name, rdata in results["regions"].items():
            print(
                f"  {region_name:<20} "
                f"accuracy: {rdata['mean_accuracy'] * 100:.1f}%"
                f"  |  mean error: {rdata['mean_angle_error_deg']:.1f} deg"
            )
            for triplet_name, adata in rdata["angles"].items():
                if adata.get("mean_accuracy") is None:
                    print(f"    {triplet_name:<34} skipped (below confidence threshold)")
                else:
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
            "gt_source":   str(self.gt_source),
            "pred_source": str(self.pred_source),
            "regions":     self.region_names,
            **results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Results saved to: {output_path}")

    def generate_comparison_image(
        self,
        results: Optional[Dict],
        output_path: Path,
    ) -> None:
        """
        Generate a side-by-side comparison image with arm keypoints overlaid.

        results may be None when evaluation failed (e.g. insufficient landmark
        confidence). Keypoints are still drawn where detectable and the accuracy
        indicator shows UNKNOWN.
        """
        gt_img   = cv2.imread(str(self.gt_source))
        pred_img = cv2.imread(str(self.pred_source))
        if gt_img is None or pred_img is None:
            raise RuntimeError("Failed to read source images for visualization.")

        target_h = 720
        header_h = 50
        gap      = 10

        gt_vis   = resize_to_height(draw_keypoints_on_image(gt_img,   self._gt_raw,   self.regions), target_h)
        pred_vis = resize_to_height(draw_keypoints_on_image(pred_img, self._pred_raw, self.regions), target_h)

        total_w = gt_vis.shape[1] + gap + pred_vis.shape[1]
        canvas  = np.zeros((target_h + header_h, total_w, 3), dtype=np.uint8)
        canvas[header_h:, :gt_vis.shape[1]]       = gt_vis
        canvas[header_h:, gt_vis.shape[1] + gap:] = pred_vis

        font      = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        cv2.putText(canvas, "Ground Truth", (10, 34), font, 0.8, (220, 220, 220), thickness)
        cv2.putText(canvas, "Model", (gt_vis.shape[1] + gap + 10, 34), font, 0.8, (220, 220, 220), thickness)

        if results is not None:
            acc_text  = f"Accuracy: {results['overall_mean_accuracy'] * 100:.1f}%"
            sub_text  = f"Angle error: {results['overall_mean_angle_error_deg']:.1f} deg"
            acc_color = (50, 220, 50)
        else:
            acc_text  = "Accuracy: UNKNOWN"
            sub_text  = ""
            acc_color = (80, 80, 220)

        (tw, _), _ = cv2.getTextSize(acc_text, font, 0.8, thickness)
        cv2.putText(canvas, acc_text, (total_w - tw - 10, 34), font, 0.8, acc_color, thickness)
        if sub_text:
            (stw, _), _ = cv2.getTextSize(sub_text, font, 0.55, 1)
            cv2.putText(canvas, sub_text, (total_w - stw - 10, header_h + 18), font, 0.55, (160, 220, 160), 1)

        y_legend = header_h + 22
        for region_name in self.region_names:
            color = REGION_COLORS.get(region_name, (180, 180, 180))
            if results is not None:
                rdata = results["regions"].get(region_name, {})
                label = f"{region_name}: {rdata.get('mean_accuracy', 0.0) * 100:.1f}%  ({rdata.get('mean_angle_error_deg', 0.0):.1f} deg)"
            else:
                label = f"{region_name}: UNKNOWN"
            cv2.rectangle(canvas, (10, y_legend - 12), (24, y_legend + 2), color, -1)
            cv2.putText(canvas, label, (30, y_legend), font, 0.5, (220, 220, 220), 1)
            y_legend += 22

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)
        print(f"Comparison image saved to: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate movement replication accuracy between two images using ViTPose."
    )
    parser.add_argument("--gt",   required=True, help="Path to the ground truth image or directory.")
    parser.add_argument("--pred", required=True, help="Path to the animated model image or directory.")
    parser.add_argument(
        "--regions", nargs="+", default=None, metavar="REGION",
        help="Body regions to evaluate. Choices: left_arm right_arm. Defaults to both.",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Inference device (default: auto-detect).")
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Directory to save outputs (JSON report + comparison PNG).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    gt_path   = Path(args.gt)
    pred_path = Path(args.pred)

    if gt_path.is_dir() and pred_path.is_dir():
        pairs = pair_image_directories(gt_path, pred_path)
        if not pairs:
            print("No matching image pairs found in the provided directories.")
            sys.exit(1)

        print(f"Found {len(pairs)} image pair(s): {[k for k, _, _ in pairs]}")

        output_dir = Path(args.output_dir) if args.output_dir else gt_path.parent / "comparison_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_image_paths: List[Path] = []
        all_results: List[Dict] = []

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
                all_results.append(results)
            except RuntimeError as exc:
                print(f"  Evaluation failed: {exc}")

            comp_path = output_dir / f"{key}_comparison.png"
            evaluator.generate_comparison_image(results, comp_path)
            comparison_image_paths.append(comp_path)

        generate_slideshow_video(comparison_image_paths, output_dir / "comparison_slideshow.mp4")

        # Batch summary across all successfully evaluated pairs
        if all_results:
            separator = "=" * 62
            print(f"\n{separator}")
            print(f"  BATCH SUMMARY  ({len(all_results)} of {len(pairs)} pairs evaluated)")
            print(separator)

            overall_accs = [r["overall_mean_accuracy"]        for r in all_results]
            overall_errs = [r["overall_mean_angle_error_deg"] for r in all_results]
            print(f"  Overall Accuracy (mean)    : {np.mean(overall_accs) * 100:.1f}%")
            print(f"  Overall Accuracy (range)   : {min(overall_accs) * 100:.1f}% - {max(overall_accs) * 100:.1f}%")
            print(f"  Overall Angle Error (mean) : {np.mean(overall_errs):.1f} deg")
            print()

            region_names = args.regions if args.regions is not None else DEFAULT_REGIONS
            for region_name in region_names:
                region_accs = [
                    r["regions"][region_name]["mean_accuracy"]
                    for r in all_results if region_name in r["regions"]
                ]
                region_errs = [
                    r["regions"][region_name]["mean_angle_error_deg"]
                    for r in all_results if region_name in r["regions"]
                ]
                if not region_accs:
                    continue
                print(
                    f"  {region_name:<20} "
                    f"accuracy: {np.mean(region_accs) * 100:.1f}%"
                    f"  |  mean error: {np.mean(region_errs):.1f} deg"
                )

                sample_region = next(
                    (r["regions"][region_name] for r in all_results if region_name in r["regions"]),
                    None,
                )
                if sample_region:
                    for triplet_name in sample_region["angles"]:
                        triplet_accs = [
                            r["regions"][region_name]["angles"][triplet_name]["mean_accuracy"]
                            for r in all_results
                            if region_name in r["regions"]
                            and triplet_name in r["regions"][region_name]["angles"]
                            and r["regions"][region_name]["angles"][triplet_name]["mean_accuracy"] is not None
                        ]
                        triplet_errs = [
                            r["regions"][region_name]["angles"][triplet_name]["mean_angle_error_deg"]
                            for r in all_results
                            if region_name in r["regions"]
                            and triplet_name in r["regions"][region_name]["angles"]
                            and r["regions"][region_name]["angles"][triplet_name]["mean_angle_error_deg"] is not None
                        ]
                        if triplet_accs:
                            print(
                                f"    {triplet_name:<34} "
                                f"{np.mean(triplet_accs) * 100:.1f}%"
                                f"  |  {np.mean(triplet_errs):.1f} deg"
                                f"  ({len(triplet_accs)}/{len(all_results)} pairs)"
                            )
                        else:
                            print(f"    {triplet_name:<34} no valid data")
                print()
            print(separator)

    else:
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
            evaluator.generate_comparison_image(results, output_dir / f"{stem}_comparison.png")
