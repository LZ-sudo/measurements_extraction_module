"""
Movement Accuracy Evaluator

Evaluates how accurately an animated model replicates the movement of a subject
using ViTPose pose estimation and joint interior-angle comparison.
Accepts videos or images as input for both ground truth and model sources.

Accuracy is measured via the interior angle at each joint (shoulder elevation,
elbow flexion) rather than absolute keypoint position. This makes the metric
invariant to the subject's global position, scale, and camera distance.

Usage (CLI):
    python -m measurements_extraction_module.vitpose_movement_accuracy.movement_evaluator \
        --gt path/to/ground_truth.mp4 \
        --pred path/to/model_render.mp4 \
        [--regions left_arm right_arm] \
        [--device cuda]
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
    extract_keypoints_from_video,
    extract_keypoints_from_image,
    is_image_path,
    normalize_pose,
    collect_joint_indices,
    build_valid_sequence,
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
    Evaluates movement replication accuracy between a ground truth source
    and a model source using joint interior-angle metrics.
    Both sources can be videos or images.

    Pipeline:
        1. Extract ViTPose keypoints from both sources (per-frame for videos,
           single frame for images).
        2. Normalize poses to torso-relative coordinates (hip midpoint origin,
           shoulder-to-hip scale).
        3. Pair frames 1:1 (image inputs bypass this step entirely).
        4. Compute interior-angle error (degrees) at each tracked joint over
           paired frames, then convert to a Gaussian-kernel accuracy score.
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
        self._gt_raw:             Optional[List] = None
        self._pred_raw:           Optional[List] = None
        self._warping_path:       Optional[List] = None
        self._gt_frame_indices:   Optional[List[int]] = None
        self._pred_frame_indices: Optional[List[int]] = None

    def evaluate(self) -> Dict:
        """
        Run the full evaluation pipeline.

        Returns:
            {
                "overall_mean_accuracy":        float,  # Gaussian-kernel accuracy [0, 1]
                "overall_mean_angle_error_deg": float,  # Mean absolute angle error in degrees
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
        """
        joint_indices = collect_joint_indices(self.regions)

        # Step 1: Keypoint extraction
        print("=== Ground Truth Source ===")
        gt_raw = (
            extract_keypoints_from_image(str(self.gt_source), self.device)
            if is_image_path(self.gt_source)
            else extract_keypoints_from_video(str(self.gt_source), self.device)
        )

        print("\n=== Model Source ===")
        pred_raw = (
            extract_keypoints_from_image(str(self.pred_source), self.device)
            if is_image_path(self.pred_source)
            else extract_keypoints_from_video(str(self.pred_source), self.device)
        )

        self._gt_raw   = gt_raw
        self._pred_raw = pred_raw

        # Step 2: Build frame pairing.
        # Image inputs bypass frame-vector filtering entirely — metric functions
        # skip individual joints gracefully, so the all-or-nothing gate used
        # by build_valid_sequence is not needed.
        if is_image_path(self.gt_source) and is_image_path(self.pred_source):
            gt_frame_indices   = [0]
            pred_frame_indices = [0]
            warping_path       = [(0, 0)]
        else:
            gt_vectors,   gt_frame_indices   = build_valid_sequence(gt_raw,   joint_indices)
            pred_vectors, pred_frame_indices = build_valid_sequence(pred_raw, joint_indices)

            if not gt_vectors or not pred_vectors:
                raise RuntimeError(
                    "Insufficient valid frames for evaluation. "
                    "Check that the tracked joints are visible throughout the videos."
                )

            print(f"\nValid frames  GT: {len(gt_vectors)}  Model: {len(pred_vectors)}")
            n            = min(len(gt_vectors), len(pred_vectors))
            warping_path = list(zip(range(n), range(n)))

        self._gt_frame_indices   = gt_frame_indices
        self._pred_frame_indices = pred_frame_indices
        self._warping_path       = warping_path

        # Step 3: Accumulate per-frame accuracy and error per angle triplet
        region_acc_list: Dict[str, Dict[str, List[float]]] = {
            r.name: {t.name: [] for t in r.angle_triplets} for r in self.regions
        }
        region_err_list: Dict[str, Dict[str, List[float]]] = {
            r.name: {t.name: [] for t in r.angle_triplets} for r in self.regions
        }

        for gt_idx, pred_idx in warping_path:
            gt_raw_frame   = gt_raw[gt_frame_indices[gt_idx]]
            pred_raw_frame = pred_raw[pred_frame_indices[pred_idx]]
            if gt_raw_frame is None or pred_raw_frame is None:
                continue

            gt_norm   = normalize_pose(gt_raw_frame)
            pred_norm = normalize_pose(pred_raw_frame)
            if gt_norm is None or pred_norm is None:
                continue

            for region in self.regions:
                if not region.angle_triplets:
                    continue
                for name, acc in compute_angle_accuracy(gt_norm, pred_norm, region.angle_triplets).items():
                    region_acc_list[region.name][name].append(acc)
                for name, err in compute_angle_error(gt_norm, pred_norm, region.angle_triplets).items():
                    region_err_list[region.name][name].append(err)

        if not any(
            accs
            for r_accs in region_acc_list.values()
            for accs in r_accs.values()
        ):
            raise RuntimeError(
                "No angle metrics could be computed. "
                "Check that the torso and arm joints are visible in both images."
            )

        # Step 4: Aggregate
        region_acc_means: List[float] = []
        region_err_means: List[float] = []
        results: Dict = {"overall_mean_accuracy": 0.0, "overall_mean_angle_error_deg": 0.0, "regions": {}}

        for region in self.regions:
            all_accs: List[float] = []
            all_errs: List[float] = []
            angle_entries: Dict[str, Dict] = {}

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
        if not (is_image_path(self.gt_source) and is_image_path(self.pred_source)):
            raise RuntimeError("Comparison visualization is only supported for image inputs.")

        gt_img   = cv2.imread(str(self.gt_source))
        pred_img = cv2.imread(str(self.pred_source))
        if gt_img is None or pred_img is None:
            raise RuntimeError("Failed to read source images for visualization.")

        gt_kpts   = self._gt_raw[0]   if self._gt_raw   is not None else None
        pred_kpts = self._pred_raw[0] if self._pred_raw is not None else None

        target_h = 720
        header_h = 50
        gap      = 10

        gt_vis   = resize_to_height(draw_keypoints_on_image(gt_img,   gt_kpts,   self.regions), target_h)
        pred_vis = resize_to_height(draw_keypoints_on_image(pred_img, pred_kpts, self.regions), target_h)

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

    def generate_comparison_video(self, results: Dict, output_path: Path) -> None:
        """
        Generate a side-by-side comparison video with arm keypoints overlaid,
        using 1:1 paired frames.

        Raises RuntimeError if evaluate() has not been called first.
        """
        if any(v is None for v in (
            self._gt_raw, self._pred_raw, self._warping_path,
            self._gt_frame_indices, self._pred_frame_indices,
        )):
            raise RuntimeError("evaluate() must be called before generate_comparison_video().")

        def _read_frames(path: Path) -> List[np.ndarray]:
            cap    = cv2.VideoCapture(str(path))
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

        print("Reading GT video frames...")
        gt_frames   = _read_frames(self.gt_source)   if not is_image_path(self.gt_source)   else [cv2.imread(str(self.gt_source))]
        print("Reading model video frames...")
        pred_frames = _read_frames(self.pred_source) if not is_image_path(self.pred_source) else [cv2.imread(str(self.pred_source))]

        if not gt_frames or not pred_frames or gt_frames[0] is None or pred_frames[0] is None:
            raise RuntimeError("Failed to read frames from source files.")

        fps = _get_fps(self.gt_source) if not is_image_path(self.gt_source) else _get_fps(self.pred_source)

        target_h = 720
        header_h = 50
        gap      = 10

        gt_sample   = resize_to_height(gt_frames[0],   target_h)
        pred_sample = resize_to_height(pred_frames[0], target_h)
        total_w = gt_sample.shape[1] + gap + pred_sample.shape[1]
        total_h = target_h + header_h

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, total_h))

        font        = cv2.FONT_HERSHEY_SIMPLEX
        thickness   = 2
        acc_text    = f"Accuracy: {results['overall_mean_accuracy'] * 100:.1f}%"
        (tw, _), _  = cv2.getTextSize(acc_text, font, 0.8, thickness)

        print(f"Writing {len(self._warping_path)} paired frames to comparison video...")

        for gt_vec_idx, pred_vec_idx in self._warping_path:
            gt_fi   = self._gt_frame_indices[gt_vec_idx]
            pred_fi = self._pred_frame_indices[pred_vec_idx]

            gt_frame   = gt_frames[gt_fi]     if gt_fi   < len(gt_frames)   else gt_frames[-1]
            pred_frame = pred_frames[pred_fi]  if pred_fi < len(pred_frames) else pred_frames[-1]

            gt_vis   = resize_to_height(draw_keypoints_on_image(gt_frame,   self._gt_raw[gt_fi],   self.regions), target_h)
            pred_vis = resize_to_height(draw_keypoints_on_image(pred_frame, self._pred_raw[pred_fi], self.regions), target_h)

            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
            canvas[header_h:, :gt_vis.shape[1]]       = gt_vis
            canvas[header_h:, gt_vis.shape[1] + gap:] = pred_vis

            cv2.putText(canvas, "Ground Truth", (10, 34), font, 0.8, (220, 220, 220), thickness)
            cv2.putText(canvas, "Model", (gt_vis.shape[1] + gap + 10, 34), font, 0.8, (220, 220, 220), thickness)
            cv2.putText(canvas, acc_text, (total_w - tw - 10, 34), font, 0.8, (50, 220, 50), thickness)

            y_legend = header_h + 22
            for region_name in self.region_names:
                color = REGION_COLORS.get(region_name, (180, 180, 180))
                rdata = results["regions"].get(region_name, {})
                label = f"{region_name}: {rdata.get('mean_accuracy', 0.0) * 100:.1f}%  ({rdata.get('mean_angle_error_deg', 0.0):.1f} deg)"
                cv2.rectangle(canvas, (10, y_legend - 12), (24, y_legend + 2), color, -1)
                cv2.putText(canvas, label, (30, y_legend), font, 0.55, (220, 220, 220), 1)
                y_legend += 22

            writer.write(canvas)

        writer.release()
        print(f"Comparison video saved to: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate movement replication accuracy between two sources using ViTPose."
    )
    parser.add_argument("--gt",   required=True, help="Path to the ground truth source (video or image).")
    parser.add_argument("--pred", required=True, help="Path to the animated model source (video or image).")
    parser.add_argument(
        "--regions", nargs="+", default=None, metavar="REGION",
        help="Body regions to evaluate. Choices: left_arm right_arm. Defaults to both.",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Inference device (default: auto-detect).")
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Directory to save outputs (JSON report + comparison PNG/MP4).",
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
                        ]
                        triplet_errs = [
                            r["regions"][region_name]["angles"][triplet_name]["mean_angle_error_deg"]
                            for r in all_results
                            if region_name in r["regions"]
                            and triplet_name in r["regions"][region_name]["angles"]
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
            if is_image_path(evaluator.gt_source) and is_image_path(evaluator.pred_source):
                evaluator.generate_comparison_image(results, output_dir / f"{stem}_comparison.png")
            else:
                evaluator.generate_comparison_video(results, output_dir / f"{stem}_comparison.mp4")
