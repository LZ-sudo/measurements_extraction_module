"""
Camera Calibration Script
=========================

Performs camera calibration using checkerboard images.
"""

import cv2
import numpy as np
import json
import glob
import argparse
from pathlib import Path


def calibrate_camera_from_images(
    image_dir,
    checkerboard_size=(9, 7),
    square_size_mm=40.0,
    output_path=None,
    visualize=False
):
    """
    Calibrate camera using checkerboard pattern images.

    Args:
        image_dir: Directory containing checkerboard images
        checkerboard_size: Inner corner count (columns, rows)
        square_size_mm: Physical size of each square in mm
        output_path: Path to save calibration JSON
        visualize: Whether to show detected corners

    Returns:
        Dictionary with calibration results
    """
    # Prepare object points (3D points in real world space)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    successful_images = []
    failed_images = []

    # Find all images
    image_paths = sorted(glob.glob(str(Path(image_dir) / "*.jpg")))
    image_paths += sorted(glob.glob(str(Path(image_dir) / "*.png")))

    if not image_paths:
        return {
            "success": False,
            "error": f"No images found in {image_dir}"
        }

    print(f"Processing {len(image_paths)} images...")
    print()

    for i, img_path in enumerate(image_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            failed_images.append(img_path)
            print(f"[{i:2d}/{len(image_paths)}] FAIL {Path(img_path).name} - Could not read")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if ret:
            obj_points.append(objp)

            # Refine corner positions for sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners_refined)

            successful_images.append(img_path)
            print(f"[{i:2d}/{len(image_paths)}] OK {Path(img_path).name} - Corners detected")

            if visualize:
                # Draw and display corners
                img_viz = img.copy()
                cv2.drawChessboardCorners(img_viz, checkerboard_size, corners_refined, ret)
                cv2.imshow(f'Corners: {Path(img_path).name}', img_viz)
                cv2.waitKey(500)
        else:
            failed_images.append(img_path)
            print(f"[{i:2d}/{len(image_paths)}] FAIL {Path(img_path).name} - No corners found")

    if visualize:
        cv2.destroyAllWindows()

    print()

    if len(obj_points) < 3:
        return {
            "success": False,
            "error": f"Insufficient valid images. Found {len(obj_points)}, need at least 3",
            "num_successful_images": len(successful_images),
            "num_failed_images": len(failed_images),
            "successful_images": successful_images,
            "failed_images": failed_images
        }

    # Calibrate camera
    print("Running calibration optimization...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        gray.shape[::-1],
        None,
        None
    )

    # Calculate reprojection error
    total_error = 0
    for i in range(len(obj_points)):
        img_points_reprojected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], img_points_reprojected, cv2.NORM_L2) / len(img_points_reprojected)
        total_error += error

    mean_error = total_error / len(obj_points)

    results = {
        "success": True,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.tolist(),
        "reprojection_error": mean_error,
        "num_successful_images": len(successful_images),
        "num_failed_images": len(failed_images),
        "successful_images": successful_images,
        "failed_images": failed_images
    }

    # Save calibration if output path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Calibration saved to: {output_path}")
        print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera using checkerboard pattern images"
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        default="camera_calibration_inputs",
        help="Directory containing checkerboard calibration images (default: camera_calibration_inputs)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="measurement_calibration/camera_calibration_outputs/camera_calibration.json",
        help="Path to save calibration JSON (default: measurement_calibration/outputs/camera_calibration.json)"
    )
    parser.add_argument(
        "--checkerboard-size",
        type=str,
        default="8x6",
        help="Checkerboard inner corners in format COLSxROWS (default: 8x6 for 9x7 squares)"
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=40.0,
        help="Physical size of each checkerboard square in mm (default: 40.0)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show detected corners for each image"
    )

    args = parser.parse_args()

    # Parse checkerboard size
    try:
        cols, rows = map(int, args.checkerboard_size.split('x'))
        CHECKERBOARD_SIZE = (cols, rows)
    except:
        print(f"Error: Invalid checkerboard size '{args.checkerboard_size}'")
        print("Use format: COLSxROWS (e.g., 8x6)")
        return

    CALIBRATION_IMAGES_DIR = args.input_dir
    OUTPUT_PATH = args.output
    SQUARE_SIZE_MM = args.square_size
    VISUALIZE = args.visualize

    print("=" * 70)
    print("CAMERA CALIBRATION - CHECKERBOARD METHOD")
    print("=" * 70)
    print()
    print(f"Calibration images: {CALIBRATION_IMAGES_DIR}")
    print(f"Checkerboard: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} inner corners")
    print(f"Square size: {SQUARE_SIZE_MM} mm")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Run calibration
    results = calibrate_camera_from_images(
        image_dir=CALIBRATION_IMAGES_DIR,
        checkerboard_size=CHECKERBOARD_SIZE,
        square_size_mm=SQUARE_SIZE_MM,
        output_path=OUTPUT_PATH,
        visualize=VISUALIZE
    )

    # Display results
    print("=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print()

    if results["success"]:
        print(f"SUCCESS: Calibration successful!")
        print()
        print(f"Images used: {results['num_successful_images']}/{results['num_successful_images'] + results['num_failed_images']}")
        print(f"Images failed: {results['num_failed_images']}/{results['num_successful_images'] + results['num_failed_images']}")
        print(f"Reprojection error: {results['reprojection_error']:.4f} pixels")
        print()

        # Quality assessment
        if results['reprojection_error'] < 0.5:
            quality = "Excellent (< 0.5 px)"
        elif results['reprojection_error'] < 1.0:
            quality = "Good (< 1.0 px)"
        elif results['reprojection_error'] < 2.0:
            quality = "Acceptable (< 2.0 px)"
        else:
            quality = "Poor - consider retaking calibration images"

        print(f"Calibration quality: {quality}")
        print()

        # Display camera matrix
        print("Camera Matrix:")
        cm = results['camera_matrix']
        print(f"  fx = {cm[0][0]:.2f} px")
        print(f"  fy = {cm[1][1]:.2f} px")
        print(f"  cx = {cm[0][2]:.2f} px")
        print(f"  cy = {cm[1][2]:.2f} px")
        print()

        # Display distortion coefficients
        print("Distortion Coefficients:")
        dc = results['distortion_coefficients'][0]
        print(f"  k1 = {dc[0]:.6f} (radial)")
        print(f"  k2 = {dc[1]:.6f} (radial)")
        print(f"  p1 = {dc[2]:.6f} (tangential)")
        print(f"  p2 = {dc[3]:.6f} (tangential)")
        print(f"  k3 = {dc[4]:.6f} (radial)")
        print()

        print("=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print()
        print("The calibration file is ready to use!")
        print()
        print("To use with backdrop calibration:")
        print("  python measurement_calibration/calibrate_backdrop.py \\")
        print("    <image> \\")
        print(f"    --camera-calibration {OUTPUT_PATH}")
        print()

        if results['num_failed_images'] > 0:
            print("Note: Some images failed corner detection:")
            for img_path in results['failed_images']:
                print(f"  - {Path(img_path).name}")
            print()
    else:
        print(f"FAILED: Calibration failed: {results['error']}")
        print()

        if results.get('failed_images'):
            print(f"Failed images ({len(results['failed_images'])}):")
            for img_path in results['failed_images'][:10]:
                print(f"  - {Path(img_path).name}")
            if len(results['failed_images']) > 10:
                print(f"  ... and {len(results['failed_images']) - 10} more")
        print()
        print("Troubleshooting:")
        print("  - Ensure checkerboard pattern is clearly visible")
        print("  - Verify CHECKERBOARD_SIZE = (9, 7) matches your pattern")
        print("  - Check that images are in focus and well-lit")
        print("  - Ensure checkerboard is flat (not bent or curved)")

    print("=" * 70)


if __name__ == "__main__":
    main()
