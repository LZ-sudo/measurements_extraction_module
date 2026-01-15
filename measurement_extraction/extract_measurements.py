"""
Measurement Extraction Orchestrator

This script orchestrates the complete measurement extraction process:
1. Runs MediaPipe detection scripts to extract landmark coordinates
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

# Import MediaPipe detection modules
from mediapipe_detection.body_segmentation import segment_body
from mediapipe_detection.hair_segmentation import segment_hair
from mediapipe_detection.pose_detection import detect_pose
from mediapipe_detection.hand_detection import detect_hands

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
        known_height_cm: Optional[float] = None
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
        """
        self.image_path = image_path
        self.calibration_path = calibration_path
        self.camera_calibration_path = camera_calibration_path
        self.known_height_cm = known_height_cm

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
        Run all MediaPipe detection scripts and collect landmark data.

        Args:
            image_path: Path to the image to process (original or undistorted)

        Returns:
            Dictionary containing all detection results
        """
        print("Running MediaPipe detections...")

        detections = {}

        # 1. Body segmentation (for height)
        print("  - Body segmentation...")
        try:
            body_output = self._get_intermediate_path("body_detection")
            detections['body'] = segment_body(
                image_path,
                output_path=body_output
            )
            print(f"     Body segmentation complete")
        except Exception as e:
            print(f"     Body segmentation failed: {e}")
            detections['body'] = {}

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

        # 3. Pose detection (for body measurements)
        print("  - Pose detection...")
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

        # 4. Hand detection (for hand length)
        print("  - Hand detection...")
        try:
            hand_output = self._get_intermediate_path("hand_detection")
            detections['hands'] = detect_hands(
                image_path,
                output_path=hand_output,
                num_hands=2
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
            image_width, image_height = get_image_dimensions(self.image_path)

            # Measure subject height in pixels
            measured_height_px = measure_height_in_pixels(
                detections.get('body', {}),
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
                    body_data=detections.get('body', {}),
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

        # Extract height from body detection
        if detections.get('body'):
            height_cm = extract_height(
                detections['body'],
                calibration_to_use,
                self.image_path,
                pose_data=detections.get('pose')  # Pass pose data for ankle landmarks
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
            "measurements": measurements['body_measurements']
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
    known_height_cm: Optional[float] = None
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
        known_height_cm
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
