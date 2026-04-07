"""
Complete Measurement Extraction Orchestrator

This script orchestrates the complete end-to-end measurement extraction pipeline:
1. ArUco marker-based backdrop calibration
2. Body and pose detection
3. Measurement extraction with calibration

Usage:
    python complete_measurements.py <image> --marker-details <marker_details_json> [options]

Example:
    python complete_measurements.py subject.jpg --marker-details marker_details.json --camera-calibration camera_cal.json -o measurements.json
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Optional

# Import calibration and measurement extraction functions
from measurement_calibration.calibrate_backdrop import calibrate_backdrop
from measurement_extraction.extract_measurements import extract_measurements


class CompleteMeasurementPipeline:
    """
    Orchestrates the complete measurement extraction pipeline.

    This class handles the full workflow from calibration to final measurements,
    managing intermediate files and providing clean outputs.
    """

    def __init__(
        self,
        image_path: str,
        marker_details_path: str,
        camera_calibration_path: Optional[str] = None,
        output_measurements_path: Optional[str] = None,
        output_calibration_path: Optional[str] = None,
        output_visualization_dir: Optional[str] = None,
        known_height_cm: Optional[float] = None,
        debug: bool = False
    ):
        """
        Initialize the complete measurement pipeline.

        Args:
            image_path: Path to input image with ArUco markers
            marker_details_path: Path to marker details JSON containing marker_size_cm and marker_positions_cm
            camera_calibration_path: Optional path to camera calibration JSON
            output_measurements_path: Path to save final measurements JSON
            output_calibration_path: Optional path to save calibration JSON
            output_visualization_dir: Optional directory to save calibration visualization
            known_height_cm: Optional subject's known height in cm for depth correction
            debug: Enable debug output
        """
        self.image_path = image_path
        self.marker_details_path = marker_details_path
        self.camera_calibration_path = camera_calibration_path
        self.output_measurements_path = output_measurements_path
        self.output_calibration_path = output_calibration_path
        self.output_visualization_dir = output_visualization_dir
        self.known_height_cm = known_height_cm
        self.debug = debug

        # Load marker details for debug output
        with open(marker_details_path, 'r', encoding='utf-8') as f:
            self.marker_details = json.load(f)

        # Temporary files to clean up
        self.temp_files = []

    def run(self) -> Dict:
        """
        Run the complete measurement extraction pipeline.

        Returns:
            Dictionary containing body and hair measurements
        """
        if self.debug:
            print("=" * 70)
            print("COMPLETE MEASUREMENT EXTRACTION PIPELINE")
            print("=" * 70)
            print(f"Input image: {self.image_path}")
            print(f"Marker details: {self.marker_details_path}")
            print(f"  Marker size: {self.marker_details.get('marker_size_cm')} cm")
            print(f"  Marker positions: {list(self.marker_details.get('marker_positions_cm', {}).keys())}")
            if self.camera_calibration_path:
                print(f"Camera calibration: {self.camera_calibration_path}")
            if self.known_height_cm:
                print(f"Known height: {self.known_height_cm} cm (depth correction enabled)")
            print()

        try:
            # Step 1: Run backdrop calibration
            if self.debug:
                print("STEP 1: Backdrop Calibration")
                print("-" * 70)

            calibration_result = self._run_calibration()

            if self.debug:
                print()

            # Step 2: Extract measurements using calibration
            if self.debug:
                print("STEP 2: Measurement Extraction")
                print("-" * 70)

            measurements = self._run_measurement_extraction(calibration_result)

            if self.debug:
                print()
                print("=" * 70)
                print("PIPELINE COMPLETE")
                print("=" * 70)

            return measurements

        finally:
            # Clean up temporary files
            self._cleanup_temp_files()

    def _run_calibration(self) -> Dict:
        """
        Run backdrop calibration step.

        Returns:
            Calibration result dictionary
        """
        # Determine calibration output path
        if self.output_calibration_path:
            calibration_path = self.output_calibration_path
        else:
            # Create temporary calibration file
            temp_fd, calibration_path = tempfile.mkstemp(suffix='_calibration.json', text=True)
            os.close(temp_fd)
            self.temp_files.append(calibration_path)

        # Run calibration
        calibration_result = calibrate_backdrop(
            image_path=self.image_path,
            output_calibration_path=calibration_path,
            visualization_dir=self.output_visualization_dir,
            camera_calibration_path=self.camera_calibration_path,
            marker_details_path=self.marker_details_path,
            debug=self.debug
        )

        # Store calibration path for measurement extraction
        self.calibration_path = calibration_path

        return calibration_result

    def _run_measurement_extraction(self, calibration_result: Dict) -> Dict:
        """
        Run measurement extraction step.

        Args:
            calibration_result: Calibration data from backdrop calibration

        Returns:
            Dictionary containing measurements
        """
        # Determine output directory for measurements
        if self.output_measurements_path:
            output_dir = os.path.dirname(self.output_measurements_path)
            output_prefix = Path(self.output_measurements_path).stem

            # Handle case where output is just a filename without directory
            if not output_dir:
                output_dir = os.path.dirname(self.image_path)
        else:
            output_dir = os.path.dirname(self.image_path)
            output_prefix = Path(self.image_path).stem

        # Run measurement extraction
        body_path, hair_path = extract_measurements(
            image_path=self.image_path,
            calibration_path=self.calibration_path,
            output_dir=output_dir,
            output_prefix=output_prefix,
            save_intermediates=False,  # Don't save intermediate detection JSONs
            camera_calibration_path=self.camera_calibration_path,
            known_height_cm=self.known_height_cm,
            visualization_dir=self.output_visualization_dir,
        )

        # Load and combine measurements
        with open(body_path, 'r', encoding='utf-8') as f:
            body_data = json.load(f)

        with open(hair_path, 'r', encoding='utf-8') as f:
            hair_data = json.load(f)

        # Store output paths
        self.body_measurements_path = body_path
        self.hair_measurements_path = hair_path

        # If user specified custom output path, save combined measurements there
        if self.output_measurements_path and not self.output_measurements_path.endswith('_body_measurements.json'):
            combined_measurements = {
                "body_measurements": body_data.get("body_measurements", {}),
                "hair_measurements": hair_data.get("hair_measurement", {})
            }

            with open(self.output_measurements_path, 'w', encoding='utf-8') as f:
                json.dump(combined_measurements, f, indent=2)

            if self.debug:
                print(f"\nCombined measurements saved to: {self.output_measurements_path}")

        return {
            "body_measurements": body_data.get("body_measurements", {}),
            "hair_measurements": hair_data.get("hair_measurement", {})
        }

    def _cleanup_temp_files(self):
        """Clean up temporary files created during pipeline execution."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    if self.debug:
                        print(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not remove temporary file {temp_file}: {e}")


def main():
    """Main entry point for the complete measurement extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete end-to-end measurement extraction from image with ArUco markers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with marker details JSON
  python complete_measurements.py subject.jpg --marker-details marker_details.json

  # With camera calibration and custom output
  python complete_measurements.py subject.jpg \\
      --marker-details marker_details.json \\
      --camera-calibration camera_cal.json \\
      -o measurements.json

  # With depth correction using known height
  python complete_measurements.py subject.jpg \\
      --marker-details marker_details.json \\
      --camera-calibration camera_cal.json \\
      --height 168.5 \\
      -o measurements.json

  # With all outputs saved
  python complete_measurements.py subject.jpg \\
      --marker-details marker_details.json \\
      --camera-calibration camera_cal.json \\
      --height 168.5 \\
      -o measurements.json \\
      --save-calibration calibration.json \\
      --save-visualization ./visualizations

  # With debug output
  python complete_measurements.py subject.jpg \\
      --marker-details marker_details.json --debug
        """
    )

    # Required arguments
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to input image with ArUco markers and subject"
    )
    parser.add_argument(
        "--marker-details",
        type=str,
        required=True,
        help="Path to JSON file containing marker_size_cm and marker_positions_cm"
    )

    # Optional calibration inputs
    parser.add_argument(
        "--camera-calibration",
        type=str,
        help="Path to camera calibration JSON file (for lens distortion correction)"
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to save measurements JSON (default: <image_name>_measurements.json)"
    )
    parser.add_argument(
        "--save-calibration",
        type=str,
        help="Path to save calibration JSON (default: not saved)"
    )
    parser.add_argument(
        "--save-visualization",
        type=str,
        help="Directory to save calibration visualization image (default: not saved)"
    )

    # Measurement options
    parser.add_argument(
        "--height",
        type=float,
        help="Subject's known height in cm (for depth correction)"
    )

    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    # Validate input image exists
    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found: {args.input_image}")
        sys.exit(1)

    # Validate marker details file exists
    if not os.path.exists(args.marker_details):
        print(f"Error: Marker details file not found: {args.marker_details}")
        sys.exit(1)

    # Validate camera calibration exists if provided
    if args.camera_calibration and not os.path.exists(args.camera_calibration):
        print(f"Error: Camera calibration file not found: {args.camera_calibration}")
        sys.exit(1)

    # Create pipeline and run
    pipeline = CompleteMeasurementPipeline(
        image_path=args.input_image,
        marker_details_path=args.marker_details,
        camera_calibration_path=args.camera_calibration,
        output_measurements_path=args.output,
        output_calibration_path=args.save_calibration,
        output_visualization_dir=args.save_visualization,
        known_height_cm=args.height,
        debug=args.debug
    )

    try:
        measurements = pipeline.run()

        # Print summary if not in debug mode
        if not args.debug:
            print("\n" + "=" * 70)
            print("MEASUREMENT EXTRACTION SUMMARY")
            print("=" * 70)

            body_measurements = measurements.get("body_measurements", {})
            hair_measurements = measurements.get("hair_measurements", {})

            if body_measurements:
                print("\nBody Measurements:")
                for key, value in body_measurements.items():
                    print(f"  {key}: {value:.2f} cm" if value is not None else f"  {key}: N/A")

            if hair_measurements:
                print("\nHair Measurements:")
                for key, value in hair_measurements.items():
                    print(f"  {key}: {value:.2f} cm" if value is not None else f"  {key}: N/A")

            print("\nOutput files:")
            print(f"  Body measurements: {pipeline.body_measurements_path}")
            print(f"  Hair measurements: {pipeline.hair_measurements_path}")

            if args.output:
                print(f"  Combined measurements: {args.output}")

            if args.save_calibration:
                print(f"  Calibration data: {args.save_calibration}")

            if args.save_visualization:
                print(f"  Visualization: {args.save_visualization}/aruco_backdrop_detection.jpg")

            print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
