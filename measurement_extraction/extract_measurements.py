"""
Measurement Extraction Orchestrator

This script orchestrates the complete measurement extraction process:
1. Runs detection scripts to extract landmark coordinates:
   - VGGHeads for head detection (accurate head top for height measurement)
   - ViTPose for pose detection (highest accuracy)
   - MediaPipe for hair segmentation, head width, and hand detection
2. Loads calibration data from backdrop calibration
3. Converts normalized coordinates to real-world measurements in cm
4. Outputs separate JSON files for hair and body measurements

Usage:
    python extract_measurements.py <input_image> <calibration_json> [options]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import MediaPipe detection modules (for segmentation and head width)
from mediapipe_detection.hair_segmentation import segment_hair
from mediapipe_detection.head_detection import detect_head_width

# Import ViTPose detection modules (for pose - highest accuracy)
from vitpose_detection.pose_detection import detect_pose

# Import MediaPipe hand detection (better T-pose accuracy than ViTPose wholebody hands)
from mediapipe_detection.hand_detection import detect_hands

# Import VGGHeads head detection (MIT license, replaces YOLOv8)
from vggheads_detection.head_detection import detect_head

# Import measurement utilities
from measurement_extraction.measurement_extraction_utils import (
    extract_height,
    extract_hair_length,
    extract_pose_measurements,
    extract_hand_measurements,
    extract_neck_length,
    prepare_image_for_measurement,
    cleanup_temporary_image,
    measure_height_in_pixels,
    calculate_shift_factor,
    apply_calibration_shift,
    get_image_dimensions
)

# Import camera calibration
from measurement_calibration.camera_calibration import CameraCalibrator


class MeasurementExtractor:
    """
    Orchestrates the complete measurement extraction pipeline.
    """

    def __init__(
        self,
        image_path: str,
        calibration_path: str,
        output_dir: Optional[str] = None,
        save_intermediates: bool = False,
        camera_calibration_path: Optional[str] = None,
        known_height_cm: Optional[float] = None,
        visualization_dir: Optional[str] = None,
    ):
        """
        Initialize the measurement extractor.

        Args:
            image_path: Path to input image
            calibration_path: Path to calibration JSON file
            output_dir: Directory to save output files (default: same as image)
            save_intermediates: Whether to save intermediate landmark JSON files
            camera_calibration_path: Optional path to camera calibration JSON for undistortion
            known_height_cm: Optional subject's known height in cm for depth correction
            visualization_dir: Optional directory to save measurement visualization image
        """
        self.image_path = image_path
        self.calibration_path = calibration_path
        self.camera_calibration_path = camera_calibration_path
        self.known_height_cm = known_height_cm
        self.visualization_dir = visualization_dir

        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.dirname(image_path)
        else:
            self.output_dir = output_dir
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.save_intermediates = save_intermediates

        # Load calibration data
        with open(calibration_path, 'r', encoding='utf-8') as f:
            self.calibration_data = json.load(f)

        # Validate calibration data
        if 'calibration_data' not in self.calibration_data:
            raise ValueError("Invalid calibration data: missing 'calibration_data' key")

        if self.calibration_data['calibration_data'].get('cm_per_normalized_unit') is None:
            raise ValueError("Invalid calibration data: cm_per_normalized_unit is None")

        # Load camera calibration if provided
        self.camera_calibrator = None
        self.temp_image_path = None
        if camera_calibration_path:
            self.camera_calibrator = CameraCalibrator()
            self.camera_calibrator.load_calibration(camera_calibration_path)
            print(f"Loaded camera calibration from: {camera_calibration_path}")
            print("Image will be undistorted before measurement extraction")

    def _get_intermediate_path(self, name: str) -> Optional[str]:
        """Get path for intermediate JSON file if saving intermediates."""
        if self.save_intermediates:
            return os.path.join(self.output_dir, f"{name}.json")
        return None

    def run_detections(self, image_path: str) -> Dict:
        """
        Run all detection scripts and collect landmark data.
        Uses ViTPose for pose/hand detection and MediaPipe for segmentation.

        Args:
            image_path: Path to the image to process (original or undistorted)

        Returns:
            Dictionary containing all detection results
        """
        print("Running detections (ViTPose + MediaPipe + VGGHeads)...")

        detections = {}

        # 1. Head detection using VGGHeads (for accurate head top position in height)
        print("  - Head detection (VGGHeads)...")
        try:
            head_output = self._get_intermediate_path("head_detection")
            detections['head'] = detect_head(
                image_path,
                output_path=head_output
            )
            if detections['head'].get('head_detected'):
                print(f"     Head detected (top_y={detections['head']['head_top_y']:.4f})")
            else:
                print(f"     WARNING: No head detected - height measurement will fail")
        except Exception as e:
            print(f"     Head detection failed: {e}")
            detections['head'] = {}

        # 2. Hair segmentation (for hair length)
        print("  - Hair segmentation...")
        try:
            hair_output = self._get_intermediate_path("hair_segmentation")
            detections['hair'] = segment_hair(
                image_path,
                output_path=hair_output,
                use_face_detection=True
            )
            print(f"     Hair segmentation complete")
        except Exception as e:
            print(f"     Hair segmentation failed: {e}")
            detections['hair'] = {}

        # 3. Pose detection using ViTPose (for body measurements)
        print("  - Pose detection (ViTPose)...")
        try:
            pose_output = self._get_intermediate_path("pose_detection")
            detections['pose'] = detect_pose(
                image_path,
                output_path=pose_output
            )
            print(f"     Pose detection complete")
        except Exception as e:
            print(f"     Pose detection failed: {e}")
            detections['pose'] = {}

        # 4. Head width detection using MediaPipe (better ear landmark accuracy)
        print("  - Head width detection (MediaPipe)...")
        try:
            head_width_output = self._get_intermediate_path("head_width_detection")
            head_width_data = detect_head_width(
                image_path,
                output_path=head_width_output
            )
            # Merge head_width into pose data
            if head_width_data.get('head_width'):
                detections['pose']['head_width'] = head_width_data['head_width']
            print(f"     Head width detection complete")
        except Exception as e:
            print(f"     Head width detection failed: {e}")

        # 5. Hand detection using MediaPipe (better T-pose accuracy)
        print("  - Hand detection (MediaPipe)...")
        try:
            hand_output = self._get_intermediate_path("hand_detection")
            detections['hands'] = detect_hands(
                image_path,
                output_path=hand_output
            )
            print(f"     Hand detection complete")
        except Exception as e:
            print(f"     Hand detection failed: {e}")
            detections['hands'] = {}

        return detections

    def extract_measurements(self, detections: Dict) -> Dict:
        """
        Extract measurements from detection results using calibration data.

        Args:
            detections: Dictionary containing detection results

        Returns:
            Dictionary with 'body_measurements' and 'hair_measurements'
        """
        print("\nExtracting measurements...")

        measurements = {
            'body_measurements': {},
            'hair_measurements': {}
        }

        # Determine which calibration to use (original or shifted)
        calibration_to_use = self.calibration_data

        # Apply plane shift if known height is provided
        if self.known_height_cm is not None:
            print(f"\n   Applying depth correction using known height: {self.known_height_cm} cm")

            # Get image dimensions
            _, image_height = get_image_dimensions(self.image_path)

            # Measure subject height in pixels using head detection
            measured_height_px = measure_height_in_pixels(
                detections.get('head', {}),
                detections.get('pose', {}),
                image_height
            )

            if measured_height_px is not None:
                # Calculate initial shift factor
                pixels_per_cm_backdrop = self.calibration_data['calibration_data']['pixels_per_cm']
                initial_shift_factor = calculate_shift_factor(
                    self.known_height_cm,
                    measured_height_px,
                    pixels_per_cm_backdrop
                )

                print(f"   Initial shift factor: {initial_shift_factor:.4f}")

                # Apply shift to calibration with iterative refinement
                calibration_to_use = apply_calibration_shift(
                    self.calibration_data,
                    initial_shift_factor,
                    self.known_height_cm,
                    head_data=detections.get('head', {}),
                    pose_data=detections.get('pose', {}),
                    image_path=self.image_path,
                    iterate=True,
                    tolerance_cm=0.1,
                    max_iterations=10
                )

                # Get final shift factor after iteration
                shift_factor = calibration_to_use['plane_shift']['shift_factor']
                print(f"   Final shift factor: {shift_factor:.4f}")
                print(f"   Original pixels_per_cm: {pixels_per_cm_backdrop:.4f}")
                print(f"   Shifted pixels_per_cm: {calibration_to_use['calibration_data']['pixels_per_cm']:.4f}")

                # Store shift info in measurements for reference
                measurements['plane_shift_applied'] = True
                measurements['shift_factor'] = shift_factor
            else:
                print(f"   WARNING: Could not measure height in pixels, using original calibration")
                measurements['plane_shift_applied'] = False

        # Extract height using VGGHeads head detection
        if detections.get('head') and detections['head'].get('head_detected'):
            height_cm = extract_height(
                detections['head'],
                calibration_to_use,
                self.image_path,
                pose_data=detections.get('pose')  # Pass pose data for heel landmarks
            )
            if height_cm is not None:
                # Apply plane shift correction for perspective-corrected height
                # The perspective method uses homography which maps to backdrop plane,
                # so we need to divide by shift_factor to get subject plane measurement
                if 'plane_shift' in calibration_to_use:
                    height_cm = height_cm / calibration_to_use['plane_shift']['shift_factor']
                    print(f"   Height (after depth correction): {height_cm:.2f} cm")
                else:
                    print(f"   Height: {height_cm:.2f} cm")
                measurements['body_measurements']['height_cm'] = height_cm
            else:
                print(f"   Height extraction failed")

        # Extract hair length from hair detection
        if detections.get('hair'):
            hair_length_cm = extract_hair_length(
                detections['hair'],
                calibration_to_use,
                self.image_path
            )
            if hair_length_cm is not None:
                measurements['hair_measurements']['height_length_cm'] = hair_length_cm
                print(f"   Hair length: {hair_length_cm:.2f} cm")
            else:
                print(f"   Hair length extraction failed")

        # Extract pose measurements
        if detections.get('pose'):
            pose_measurements = extract_pose_measurements(
                detections['pose'],
                calibration_to_use,
                self.image_path
            )

            # Add valid measurements
            for key, value in pose_measurements.items():
                if value is not None:
                    measurements['body_measurements'][f"{key}_cm"] = value
                    print(f"   {key.replace('_', ' ').title()}: {value:.2f} cm")
                else:
                    print(f"   {key.replace('_', ' ').title()} extraction failed")

            # Extract neck length if available
            neck_length_cm = extract_neck_length(
                detections['pose'],
                calibration_to_use,
                self.image_path
            )
            if neck_length_cm is not None:
                measurements['body_measurements']['neck_length_cm'] = neck_length_cm
                print(f"   Neck Length: {neck_length_cm:.2f} cm")

        # Extract hand measurements
        if detections.get('hands'):
            hand_length_cm = extract_hand_measurements(
                detections['hands'],
                calibration_to_use,
                self.image_path
            )
            if hand_length_cm is not None:
                measurements['body_measurements']['hand_length_cm'] = hand_length_cm
                print(f"   Hand Length: {hand_length_cm:.2f} cm")
            else:
                print(f"   Hand length extraction failed")

        return measurements

    def _create_measurement_visualization(self, detections: Dict) -> Optional[str]:
        """
        Draw body measurement landmarks and lines on the ArUco visualization.

        Loads aruco_backdrop_detection.jpg (or the original image as fallback) as the
        base, overlays a semi-transparent filled polygon on the backdrop region, then
        draws coloured line segments and landmark points for each of the 9 measurements
        (head width, shoulder width, hip width, upper arm, forearm, upper leg, lower leg,
        shoulder-to-waist, hand length). A colour legend is added in the top-right corner.
        The result is saved as measurement_visualization.jpg in visualization_dir.

        Args:
            detections: Dictionary of all detection results from run_detections()

        Returns:
            Path to the saved visualization, or None on failure
        """
        try:
            import cv2
            from pathlib import Path as _Path

            pose_data = detections.get('pose', {})
            hands_data = detections.get('hands', {})

            # Load original image as the clean base
            original = cv2.imread(self.image_path)
            if original is None:
                print(f"Warning: Could not load original image: {self.image_path}")
                return None

            # Blend ArUco visualization at 50% opacity over the original so its
            # marker lines are visible but subdued, keeping measurement lines prominent
            aruco_vis_path = _Path(self.visualization_dir) / 'aruco_backdrop_detection.jpg'
            if aruco_vis_path.exists():
                aruco_vis = cv2.imread(str(aruco_vis_path))
                if aruco_vis is not None and aruco_vis.shape == original.shape:
                    image = cv2.addWeighted(aruco_vis, 0.50, original, 0.50, 0)
                else:
                    image = original.copy()
            else:
                image = original.copy()

            h, w = image.shape[:2]

            def to_px(lm):
                return (int(lm['x'] * w), int(lm['y'] * h))

            def draw_segment(img, pt1, pt2, color, label=None):
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
                for pt in (pt1, pt2):
                    cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
                    cv2.circle(img, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)
                if label:
                    mid = ((pt1[0] + pt2[0]) // 2 + 6, (pt1[1] + pt2[1]) // 2 - 4)
                    cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                color, 1, cv2.LINE_AA)

            # Distinct BGR colours per measurement
            C = {
                'head_width':     (30,  180, 255),
                'shoulder_width': (0,   220,   0),
                'hip_width':      (220,   0, 200),
                'upper_arm':      (0,   140, 255),
                'forearm':        (0,   255, 150),
                'upper_leg':      (180,   0, 255),
                'lower_leg':      (255, 160,   0),
                'shoulder_waist': (0,   200, 255),
                'hand_length':    (200,  80, 200),
            }

            # 1. Head width  (landmark_7 <-> landmark_8)
            hw = pose_data.get('head_width', {})
            lm7, lm8 = hw.get('landmark_7'), hw.get('landmark_8')
            if lm7 and lm8:
                draw_segment(image, to_px(lm7), to_px(lm8), C['head_width'], 'Head W')

            # 2. Shoulder width  (landmark_11 <-> landmark_12)
            sw = pose_data.get('shoulder_width', {})
            lm11, lm12 = sw.get('landmark_11'), sw.get('landmark_12')
            if lm11 and lm12:
                draw_segment(image, to_px(lm11), to_px(lm12), C['shoulder_width'], 'Shoulder W')

            # 3. Hip width  (landmark_23 <-> landmark_24)
            hiw = pose_data.get('hip_width', {})
            lm23, lm24 = hiw.get('landmark_23'), hiw.get('landmark_24')
            if lm23 and lm24:
                draw_segment(image, to_px(lm23), to_px(lm24), C['hip_width'], 'Hip W')

            # 4–8. Bilateral measurements — draw both sides, label on left only
            bilateral = [
                ('upper_arm_length',  [('left',  'landmark_11', 'landmark_13'),
                                       ('right', 'landmark_12', 'landmark_14')],
                 C['upper_arm'],     'Upper Arm'),
                ('forearm_length',    [('left',  'landmark_13', 'landmark_15'),
                                       ('right', 'landmark_14', 'landmark_16')],
                 C['forearm'],       'Forearm'),
                ('upper_leg_length',  [('left',  'landmark_23', 'landmark_25'),
                                       ('right', 'landmark_24', 'landmark_26')],
                 C['upper_leg'],     'Upper Leg'),
                ('lower_leg_length',  [('left',  'landmark_25', 'landmark_27'),
                                       ('right', 'landmark_26', 'landmark_28')],
                 C['lower_leg'],     'Lower Leg'),
                ('shoulder_to_waist', [('left',  'landmark_11', 'landmark_23'),
                                       ('right', 'landmark_12', 'landmark_24')],
                 C['shoulder_waist'], 'Shld-Waist'),
            ]
            for data_key, sides, color, base_label in bilateral:
                seg_data = pose_data.get(data_key, {})
                for i, (side_key, key_a, key_b) in enumerate(sides):
                    side = seg_data.get(side_key, {})
                    lm_a, lm_b = side.get(key_a), side.get(key_b)
                    if lm_a and lm_b:
                        label = base_label if i == 0 else None
                        draw_segment(image, to_px(lm_a), to_px(lm_b), color, label)

            # 9. Hand length  (wrist landmark_0 -> middle finger tip landmark_12)
            for hand in hands_data.get('hands', []):
                hl = hand.get('hand_length', {})
                lm0, lm12h = hl.get('landmark_0'), hl.get('landmark_12')
                if lm0 and lm12h:
                    draw_segment(image, to_px(lm0), to_px(lm12h), C['hand_length'], 'Hand')

            # Colour legend (top-right corner)
            legend_items = [
                ('Head W',      C['head_width']),
                ('Shoulder W',  C['shoulder_width']),
                ('Hip W',       C['hip_width']),
                ('Upper Arm',   C['upper_arm']),
                ('Forearm',     C['forearm']),
                ('Upper Leg',   C['upper_leg']),
                ('Lower Leg',   C['lower_leg']),
                ('Shld-Waist',  C['shoulder_waist']),
                ('Hand',        C['hand_length']),
            ]
            line_h = 22
            legend_x = w - 160
            legend_y0 = 40
            leg_overlay = image.copy()
            cv2.rectangle(
                leg_overlay,
                (legend_x - 8, legend_y0 - 16),
                (w - 5, legend_y0 + len(legend_items) * line_h + 4),
                (20, 20, 20), -1,
            )
            image = cv2.addWeighted(leg_overlay, 0.60, image, 0.40, 0)
            for j, (name, color) in enumerate(legend_items):
                y = legend_y0 + j * line_h
                cv2.line(image, (legend_x, y), (legend_x + 20, y), color, 3, cv2.LINE_AA)
                cv2.putText(image, name, (legend_x + 26, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

            output_path = _Path(self.visualization_dir) / 'measurement_visualization.jpg'
            cv2.imwrite(str(output_path), image)
            return str(output_path)

        except Exception as e:
            print(f"Warning: Could not create measurement visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_outputs(self, measurements: Dict, output_prefix: Optional[str] = None):
        """
        Save measurement outputs to separate JSON files.

        Args:
            measurements: Dictionary with body and hair measurements
            output_prefix: Optional prefix for output filenames
        """
        print("\nSaving outputs...")

        # Determine output prefix
        if output_prefix is None:
            image_name = Path(self.image_path).stem
            output_prefix = image_name

        # Save body measurements
        body_output_path = os.path.join(
            self.output_dir,
            f"{output_prefix}_body_measurements.json"
        )
        body_output = {
            "body_measurements": measurements['body_measurements']
        }
        with open(body_output_path, 'w', encoding='utf-8') as f:
            json.dump(body_output, f, indent=2)
        print(f"   Body measurements saved to: {body_output_path}")

        # Save hair measurements
        hair_output_path = os.path.join(
            self.output_dir,
            f"{output_prefix}_hair_measurements.json"
        )
        hair_output = {
            "hair_measurement": measurements['hair_measurements']
        }
        with open(hair_output_path, 'w', encoding='utf-8') as f:
            json.dump(hair_output, f, indent=2)
        print(f"   Hair measurements saved to: {hair_output_path}")

        return body_output_path, hair_output_path

    def run(self, output_prefix: Optional[str] = None):
        """
        Run the complete measurement extraction pipeline.

        Args:
            output_prefix: Optional prefix for output filenames

        Returns:
            Tuple of (body_measurements_path, hair_measurements_path)
        """
        print("="*70)
        print("MEASUREMENT EXTRACTION PIPELINE")
        print("="*70)
        print(f"Input image: {self.image_path}")
        print(f"Calibration: {self.calibration_path}")
        if self.camera_calibration_path:
            print(f"Camera calibration: {self.camera_calibration_path}")
        if self.known_height_cm:
            print(f"Known height: {self.known_height_cm} cm (depth correction enabled)")
        print(f"Output directory: {self.output_dir}")
        print()

        try:
            # Step 0: Prepare image (undistort if camera calibration provided)
            image_to_process, self.temp_image_path = prepare_image_for_measurement(
                self.image_path,
                self.camera_calibrator
            )

            # Step 1: Run detections
            detections = self.run_detections(image_to_process)

            # Step 2: Extract measurements
            measurements = self.extract_measurements(detections)

            # Step 3: Save outputs
            output_paths = self.save_outputs(measurements, output_prefix)

            # Step 4: Create measurement visualization if requested
            if self.visualization_dir:
                vis_path = self._create_measurement_visualization(detections)
                if vis_path:
                    print(f"   Measurement visualization saved to: {vis_path}")

            print()
            print("="*70)
            print("EXTRACTION COMPLETE")
            print("="*70)

            return output_paths

        finally:
            # Always clean up temporary files
            cleanup_temporary_image(self.temp_image_path)


def extract_measurements(
    image_path: str,
    calibration_path: str,
    output_dir: Optional[str] = None,
    output_prefix: Optional[str] = None,
    save_intermediates: bool = False,
    camera_calibration_path: Optional[str] = None,
    known_height_cm: Optional[float] = None,
    visualization_dir: Optional[str] = None,
) -> tuple:
    """
    Convenience function to run measurement extraction pipeline.

    Args:
        image_path: Path to input image
        calibration_path: Path to calibration JSON file
        output_dir: Directory to save outputs (default: same as image)
        output_prefix: Prefix for output filenames (default: image filename)
        save_intermediates: Whether to save intermediate landmark JSONs
        camera_calibration_path: Optional path to camera calibration JSON for undistortion
        known_height_cm: Optional subject's known height in cm for depth correction

    Returns:
        Tuple of (body_measurements_path, hair_measurements_path)
    """
    extractor = MeasurementExtractor(
        image_path,
        calibration_path,
        output_dir,
        save_intermediates,
        camera_calibration_path,
        known_height_cm,
        visualization_dir,
    )
    return extractor.run(output_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract body and hair measurements from an image using calibration data"
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "calibration",
        type=str,
        help="Path to calibration JSON file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        help="Directory to save output files (default: same as input image)"
    )
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        help="Prefix for output filenames (default: input image filename)"
    )
    parser.add_argument(
        "--camera-calibration",
        type=str,
        help="Path to camera calibration JSON file (for lens distortion correction)"
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate landmark detection JSON files"
    )
    parser.add_argument(
        "--height",
        type=float,
        help="Subject's known height in cm (for depth correction)"
    )

    args = parser.parse_args()

    # Run extraction
    try:
        body_path, hair_path = extract_measurements(
            args.input_image,
            args.calibration,
            args.output_dir,
            args.prefix,
            args.save_intermediates,
            args.camera_calibration,
            args.height
        )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
