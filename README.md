# Measurements Extraction Module

This module provides tools for extracting body measurements from images using easyViTPose (for high-accuracy pose detection), Google's Mediapipe library (for segmentation), ArUco marker-based calibration, and computer vision techniques. It includes landmark detection, segmentation, camera calibration, and robust measurement extraction capabilities for anthropometric analysis.

## Module Structure

```
measurements_extraction_module/
├── .git/                          # Git repository
├── .gitignore                     # Git ignore patterns
├── .gitmodules                    # Git submodule configuration
├── mediapipe_detection/           # Segmentation and head width detection
│   ├── head_detection.py          # Head width via MediaPipe Pose (ear landmarks)
│   └── hair_segmentation.py       # Hair segmentation with YOLOv8 head detection
├── vitpose_detection/             # High-accuracy pose detection (easyViTPose)
│   ├── config_model.py            # Model configuration and downloading
│   ├── pose_detection.py          # Body pose landmarks (133 wholebody keypoints)
│   └── hand_detection.py          # Hand landmarks (21 points per hand)
├── vitpose_movement_accuracy/     # Movement replication accuracy evaluation
│   ├── __init__.py                # Package exports
│   ├── body_region_config.py      # Body region and angle triplet definitions
│   ├── movement_accuracy_utils.py # Keypoint extraction, normalization, DTW, angle computation
│   └── movement_evaluator.py      # Main evaluator class and CLI
├── yolov8_detection/              # YOLOv8-based head detection for height
│   └── head_detection.py          # Head bounding box detection (CrowdHuman)
├── measurement_calibration/       # ArUco-based calibration system
│   ├── calibrate_backdrop.py      # ArUco marker backdrop calibration
│   ├── calibration_utils.py       # Calibration helper functions
│   ├── calibration_config.py      # Calibration configuration
│   ├── camera_calibration.py      # Camera calibration utilities
│   └── undistort_image.py         # Image undistortion using calibration
├── measurement_extraction/        # Measurement extraction tools
│   ├── extract_measurements.py    # Main measurement extraction orchestrator
│   └── measurement_extraction_utils.py  # Measurement calculation utilities
├── weight_files/                  # Model weight files
├── input_images/                  # Sample input images
├── outputs/                       # Generated outputs (JSON, images)
├── hair_classification_model/     # Hair classification submodule
├── calibrate_camera.py            # Camera lens calibration script
├── classify_hair.py               # Hair classification script
├── complete_measurements.py       # End-to-end measurement pipeline
├── generate_aruco_markers.py      # Generate printable ArUco markers PDF
├── marker_positions_example.json  # Example marker position configuration
├── requirements.txt               # Python dependencies
└── LICENSE.md                     # License information
```

## Features Overview

### Pose Detection (easyViTPose)
- **High-Accuracy Pose Detection**: 133 wholebody keypoints using ViTPose-Huge model
- **Hand Detection**: 21 landmarks per hand extracted from wholebody keypoints
- **Automatic Model Download**: Models downloaded from Hugging Face Hub and cached locally
- **Better Occlusion Handling**: Superior accuracy for challenging poses and clothing occlusion

### Head Detection for Height (YOLOv8)
- **Accurate Head Top Detection**: YOLOv8 model trained on CrowdHuman dataset for reliable head bounding box detection
- **Hair-Invariant Height Measurement**: Uses head bounding box top edge instead of body segmentation, providing consistent results regardless of hair volume
- **Automatic Model Download**: Model downloaded from Google Drive on first run and cached in `weight_files/`
- Model source: [Owen718/Head-Detection-Yolov8](https://github.com/Owen718/Head-Detection-Yolov8)

### Segmentation (MediaPipe)
- **Hair Segmentation**: Combined with YOLOv8 head detection for accurate hair boundary detection in full-body images
- **Head Width Detection**: Via MediaPipe Pose ear landmarks (7 & 8)

### ArUco Marker-Based Calibration
- **Robust Backdrop Calibration**: Sub-pixel accurate ArUco marker detection for scale extraction
- **Multi-Axis Calibration**: Averages horizontal, vertical, and marker-size-based measurements for robustness
- **Perspective Correction**: Homography-based transformation for accurate height measurement
- **Camera Calibration**: Optional lens distortion correction using chessboard calibration
- **Flexible Marker Positioning**: Configurable marker positions for different backdrop setups

### Movement Accuracy Evaluation (vitpose_movement_accuracy)
- **Pose-Based Accuracy Metric**: Evaluates how accurately an animated 3D model replicates the arm movements of a subject using joint angle comparison
- **IMU-Sensor Aligned Metrics**: Angle triplets are designed to match the sensor pairs present in the capture rig (chest, upper arm, forearm, back of hand), making the metric meaningful for IMU-driven animation
- **DTW Temporal Alignment**: Dynamic Time Warping aligns sequences of different lengths or speeds before frame-level comparison, supporting both video-to-video and image-to-image evaluation
- **Gaussian-Kernel Accuracy**: Converts angle errors to a [0, 1] accuracy score via `exp(-error² / (2σ²))`, where σ is a per-joint tolerance in degrees
- **Per-Joint Breakdown**: Reports accuracy and mean angle error per region (left arm, right arm) and per angle triplet (arm elevation, elbow flexion, wrist deviation)
- **Comparison Visualization**: Generates a side-by-side PNG with arm skeleton and torso reference frame overlaid for image inputs

### Measurements Extracted
- **Height**: Full body height from heel landmarks to head top (using YOLOv8 head detection)
- **Hair Length**: Vertical hair measurement
- **Width Measurements**: Head width, shoulder width, hip width
- **Bilateral Measurements**: Upper/lower arm, upper/lower leg, shoulder-to-waist (averaged left/right)
- **Hand Length**: Wrist to middle fingertip distance

## Installation

### 0. Create virtual environment

```bash
# Create python virtual environment
python -m venv venv

# Activate python virtual environment
venv/Scripts/activate
```

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Models

**MediaPipe Models** (included in `weight_files/`):
- `pose_landmarker_heavy.task` (~30MB) - for head width detection
- `hair_segmenter.tflite` - for hair segmentation

**YOLOv8 Head Detection Model** (downloaded automatically on first run to `weight_files/`):
- `best.pt` (~130MB) - YOLOv8 model trained on CrowdHuman for head detection
- Used for: Height measurement (head top detection) and hair segmentation (head region localization)

**ViTPose Models** (downloaded automatically on first run):
- `vitpose-h-wholebody.pth` (~350MB) - ViTPose-Huge for pose detection
- `yolov8s.pt` (~22MB) - YOLOv8 for person detection

ViTPose models are cached at `~/.cache/vitpose/` after download.

## Quick Start

### 1. Generate ArUco Markers

Generate printable ArUco markers for your backdrop:

```bash
# Generate markers sized 16.4cm (default: 12cm)
python generate_aruco_markers.py -s 16.4 -o aruco_markers.pdf
```

Print the PDF at 100% scale (no scaling) and attach the 4 markers to your backdrop corners.

### 2. Create Marker Position Configuration

Create a JSON file specifying where you placed the markers (in cm from floor):

```json
{
  "top_left": {"x": 0, "y": 200},
  "top_right": {"x": 100, "y": 200},
  "bottom_left": {"x": 0, "y": 10},
  "bottom_right": {"x": 100, "y": 10}
}
```

**Important:** Heights (y values) are measured from the floor where the person stands.

### 3. Run Complete Measurement Pipeline

```bash
# Basic usage
python complete_measurements.py subject.jpg -s 16.4 --marker-positions marker_positions.json

# With camera calibration for lens distortion correction
python complete_measurements.py subject.jpg -s 16.4 --marker-positions marker_positions.json --camera-calibration camera_cal.json

# Save all outputs
python complete_measurements.py subject.jpg -s 16.4 --marker-positions marker_positions.json --camera-calibration camera_cal.json -o measurements.json --save-calibration calibration.json --save-visualization ./visualizations
```

## Movement Accuracy Evaluation

### Quick Start (CLI)

Compare a ground truth subject video or image against an animated model render:

```bash
# Video-to-video comparison
python -m vitpose_movement_accuracy.movement_evaluator \
    --gt path/to/subject.mp4 \
    --pred path/to/model_render.mp4 \
    --output-dir ./outputs

# Image-to-image comparison (also generates a comparison PNG)
python -m vitpose_movement_accuracy.movement_evaluator \
    --gt path/to/subject.jpg \
    --pred path/to/model_render.jpg \
    --output-dir ./outputs

# Evaluate a specific region only, on a specific device
python -m vitpose_movement_accuracy.movement_evaluator \
    --gt subject.mp4 \
    --pred model.mp4 \
    --regions left_arm \
    --device cuda \
    --output-dir ./outputs
```

### Quick Start (Python API)

```python
from vitpose_movement_accuracy import MovementEvaluator

evaluator = MovementEvaluator(
    gt_source="path/to/subject.mp4",
    pred_source="path/to/model_render.mp4",
    regions=["left_arm", "right_arm"],  # default
    device=None,                         # auto-detect
)

results = evaluator.evaluate()
evaluator.print_results(results)

# Save JSON report and (for image inputs) comparison PNG
evaluator.save_results(results, output_path="outputs/results.json")
evaluator.generate_comparison_image(results, output_path="outputs/comparison.png")
```

### Angle Triplets and Sensor Mapping

Each angle triplet corresponds to a pair of IMU sensors in the capture rig:

| Triplet | Joints (proximal → vertex → distal) | Sensor pair |
|---|---|---|
| `left_arm_elevation` | right shoulder → left shoulder → left elbow | Chest + left upper arm |
| `left_elbow_angle` | left shoulder → left elbow → left wrist | Left upper arm + left forearm |
| `left_wrist_angle` | left elbow → left wrist → left index MCP | Left forearm + left hand |
| `right_arm_elevation` | left shoulder → right shoulder → right elbow | Chest + right upper arm |
| `right_elbow_angle` | right shoulder → right elbow → right wrist | Right upper arm + right forearm |
| `right_wrist_angle` | right elbow → right wrist → right index MCP | Right forearm + right hand |

The torso reference frame (both shoulders and hips) is included in the DTW alignment vector so that arm position is evaluated relative to the full body context.

## Advanced Usage

### Individual Pipeline Steps

If you need more control, run the pipeline steps individually:

#### Step 1: Backdrop Calibration

```bash
# Calibrate using ArUco markers
python measurement_calibration/calibrate_backdrop.py input.jpg -s 16.4 --marker-positions marker_positions.json -c calibration.json -v ./visualizations
```

#### Step 2: Extract Measurements

```bash
# Extract measurements using calibration
python measurement_extraction/extract_measurements.py input.jpg calibration.json -o ./outputs --camera-calibration camera_cal.json
```

### Camera Calibration (Optional but Recommended)

For best accuracy, calibrate your camera to correct lens distortion:

```bash
# Capture multiple images of a chessboard pattern (9x6 inner corners)
python calibrate_camera.py calibrate \
    ./calibration_images/*.jpg \
    -o camera_calibration.json \
    --pattern-size 9x6

# Test the calibration
python calibrate_camera.py test \
    camera_calibration.json \
    test_image.jpg \
    -o undistorted.jpg
```

## Configuration

### Marker Positions

The `marker_positions.json` file is **required** and defines the physical layout of your ArUco markers:

```json
{
  "top_left": {"x": 0, "y": 210},      // Marker ID 0 (x, y in cm from floor)
  "top_right": {"x": 100, "y": 210},   // Marker ID 1
  "bottom_left": {"x": 0, "y": 20},    // Marker ID 2
  "bottom_right": {"x": 100, "y": 20}  // Marker ID 3
}
```

**Key Points:**
- `x`: Horizontal position in cm (0 = left edge)
- `y`: Vertical height in cm from floor (0 = floor level)
- Heights must account for backdrop elevation (e.g., if backdrop is 10cm off floor, add 10cm to all y values)
- Use actual positions where marker **centers** are located

See [marker_positions_example.json](marker_positions_example.json) for a detailed example.

### Default Configuration

Edit [measurement_calibration/calibration_config.py](measurement_calibration/calibration_config.py) to change:
- `DEFAULT_MARKER_SIZE_CM` - Default marker size
- `DEFAULT_MARKER_POSITIONS` - Default marker layout
- `ARUCO_DICT_TYPE` - ArUco dictionary type
- Confidence thresholds and visualization settings

## Output Files

### Measurements JSON

Body measurements (`*_body_measurements.json`):
```json
{
  "measurements": {
    "height_cm": 170.21,
    "head_width_cm": 13.23,
    "shoulder_width_cm": 35.96,
    "hip_width_cm": 21.60,
    "upper_arm_length_cm": 28.71,
    "forearm_length_cm": 27.19,
    "upper_leg_length_cm": 41.04,
    "lower_leg_length_cm": 34.42,
    "shoulder_to_waist_cm": 57.59,
    "hand_length_cm": 14.79
  }
}
```

### Movement Accuracy Report

Movement accuracy report (`*_results.json`):
```json
{
  "gt_source": "path/to/subject.mp4",
  "pred_source": "path/to/model_render.mp4",
  "regions": ["left_arm", "right_arm"],
  "overall_mean_accuracy": 0.8731,
  "overall_mean_angle_error_deg": 6.42,
  "regions": {
    "left_arm": {
      "mean_accuracy": 0.8914,
      "mean_angle_error_deg": 5.81,
      "angles": {
        "left_arm_elevation":  { "mean_accuracy": 0.9102, "mean_angle_error_deg": 4.33 },
        "left_elbow_angle":    { "mean_accuracy": 0.8843, "mean_angle_error_deg": 6.12 },
        "left_wrist_angle":    { "mean_accuracy": 0.8797, "mean_angle_error_deg": 6.98 }
      }
    },
    "right_arm": { "..." : "..." }
  }
}
```

For image inputs, a side-by-side comparison PNG (`*_comparison.png`) is also generated, showing the arm skeleton and torso reference frame overlaid on both images with overall accuracy annotated.

Hair measurements (`*_hair_measurements.json`):
```json
{
  "hair_measurement": {
    "height_length_cm": 25.3
  }
}
```

### Calibration JSON

Contains detailed calibration data including:
- `pixels_per_cm` - Average scale from 3 methods (marker size, horizontal distance, vertical distance)
- `backdrop_corners` - Detected marker center positions
- `marker_positions` - Physical marker layout used
- Individual calibration method breakdowns for validation


## Troubleshooting

### Height measurements differ between marker sizes
- Ensure `marker_positions.json` accurately reflects where marker **centers** are located
- Larger markers have centers further from edges: (marker_size / 2) offset
- Account for backdrop elevation if mounted off the floor

### ArUco markers not detected
- Ensure markers are flat, well-lit, and in focus
- Check that marker size matches what you specified (`-s` parameter)
- Verify markers are from DICT_4X4_50 dictionary
- Use `--debug` flag to see detailed detection information

### Measurements seem inaccurate
- Use camera calibration to correct lens distortion (especially for wide-angle cameras)
- Verify marker positions match physical setup
- Check calibration confidence (should be "high")
- With `--debug`, verify all 3 calibration methods give similar `pixels_per_cm` values

## References

- [easyViTPose](https://github.com/JunkyByte/easy_ViTPose) - High-accuracy pose estimation used by both pose detection and movement accuracy evaluation
- [ViTPose Paper](https://arxiv.org/abs/2204.12484) - Vision Transformer for Generic Body Pose Estimation
- [Head-Detection-Yolov8](https://github.com/Owen718/Head-Detection-Yolov8) - YOLOv8 head detection model trained on CrowdHuman dataset
- [Mediapipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [OpenCV ArUco Marker Detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [Camera Calibration with OpenCV](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Perspective Transformation (Homography)](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
- [dtaidistance](https://github.com/wannesm/dtaidistance) - Dynamic Time Warping for temporal sequence alignment
- [COCO-WholeBody Keypoint Format](https://github.com/jin-s13/COCO-WholeBody) - 133-keypoint layout used for pose indexing