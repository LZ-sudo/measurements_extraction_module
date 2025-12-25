# Measurements Extraction Module

This module provides tools for extracting body measurements from images using Google's Mediapipe library, computer vision techniques, and OCR. It includes landmark detection, segmentation, and calibration capabilities for anthropometric measurement extraction.

## Module Structure

```
measurements_extraction_module/
├── mediapipe_detection/          # Landmark detection and segmentation
│   ├── pose_detection.py         # Body pose landmarks (33 points)
│   ├── face_detection.py         # Facial landmarks (478 points)
│   ├── hand_detection.py         # Hand landmarks (21 points per hand)
│   ├── hair_segmentation.py      # Hair segmentation with face detection
│   └── body_segmentation.py      # Full-body segmentation
├── measurement_extraction/        # Calibration and measurement tools
│   └── extract_norm_per_cm.py    # Backdrop calibration using OCR
├── mediapipe_task_files/         # Downloaded Mediapipe models
├── input_images/                 # Sample input images
├── outputs/                      # Generated outputs (JSON, images)
└── requirements.txt              # Python dependencies
```

## Features Overview

### Landmark Detection
- **Pose Detection**: 33 body landmarks for measurement extraction
- **Face Detection**: 478 facial landmarks with high precision
- **Hand Detection**: 21 landmarks per hand with handedness classification

### Segmentation
- **Hair Segmentation**: Combined with face detection for accurate hair boundary detection
- **Body Segmentation**: Full-body isolation using selfie segmenter

### Calibration
- **Measurement Backdrop Calibration**: Automatic extraction of cm-to-pixel ratio using Tesseract OCR and Hough line detection

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `opencv-python>=4.8.0` - Computer vision operations
- `opencv-contrib-python>=4.8.0` - Extended OpenCV features
- `numpy>=1.24.0` - Numerical operations
- `Pillow>=10.0.0` - Image processing
- `mediapipe==0.10.9` - Google Mediapipe for landmark detection
- `pytesseract>=0.3.10` - Python wrapper for Tesseract OCR
- `matplotlib>=3.7.0` - Visualization
- `scipy>=1.10.0` - Scientific computing
- `seaborn>=0.12.0` - Statistical visualization

### 2. Install Tesseract OCR (Required for Calibration)

**Windows:**
Download and install Tesseract OCR 5.5.0:
```
https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe
```
Default installation path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. Mediapipe Models

Models are automatically downloaded on first use:
- `pose_landmarker_heavy.task` (~30MB)
- `face_landmarker.task` (~10MB)
- `hand_landmarker.task` (~10MB)

## Usage

### Pose Detection

Extracts body landmark coordinates for measurement calculations.

**Python API:**
```python
from mediapipe_detection import PoseDetector
import cv2

image = cv2.imread('input.jpg')
detector = PoseDetector()
landmark_data = detector.detect(image)
detector.close()

# landmark_data contains:
# - head_width: landmarks 7, 8 (left/right ear)
# - shoulder_width: landmarks 11, 12
# - hip_width: landmarks 23, 24
# - upper_arm_length: landmarks 11-13 (left), 12-14 (right)
# - forearm_length: landmarks 13-15 (left), 14-16 (right)
```

**Command Line:**
```bash
python mediapipe_detection/pose_detection.py input.jpg -o output.json
```

### Face Detection

Detects 478 facial landmarks for detailed facial measurements.

**Python API:**
```python
from mediapipe_detection import FaceDetector
import cv2

image = cv2.imread('input.jpg')
detector = FaceDetector(num_faces=1)
face_data = detector.detect(image)
detector.close()
```

**Command Line:**
```bash
python mediapipe_detection/face_detection.py input.jpg -o output.json --max-faces 1
```

### Hand Detection

Detects hand landmarks with left/right classification.

**Python API:**
```python
from mediapipe_detection import HandDetector
import cv2

image = cv2.imread('input.jpg')
detector = HandDetector(num_hands=2)
hand_data = detector.detect(image)
detector.close()

# hand_data contains:
# - hands[].handedness: "Left" or "Right"
# - hands[].confidence: detection confidence
# - hands[].hand_length: landmarks 0 (wrist) and 12 (ring finger base)
```

**Command Line:**
```bash
python mediapipe_detection/hand_detection.py input.jpg -o output.json --max-hands 2
```

### Hair Segmentation

Segments hair region using face detection for improved accuracy.

**Python API:**
```python
from mediapipe_detection import HairSegmenter
import cv2

image = cv2.imread('input.jpg')
segmenter = HairSegmenter(use_face_detection=True)
hair_data = segmenter.segment(image)
segmenter.close()

# hair_data contains:
# - hair_length.top.y: normalized y-coordinate of hair top
# - hair_length.bottom.y: normalized y-coordinate of hair bottom
```

**Command Line:**
```bash
python mediapipe_detection/hair_segmentation.py input.jpg -o output.json
# Use --no-face-detection for close-up head shots
```

### Body Segmentation

Segments entire body from background for height measurement.

**Python API:**
```python
from mediapipe_detection import BodySegmenter
import cv2

image = cv2.imread('input.jpg')
segmenter = BodySegmenter()
height_data = segmenter.segment(image)
segmenter.close()

# height_data contains:
# - height.top.y: normalized y-coordinate of body top
# - height.bottom.y: normalized y-coordinate of body bottom
```

**Command Line:**
```bash
python mediapipe_detection/body_segmentation.py input.jpg -o output.json
```

### Measurement Backdrop Calibration

Extracts calibration factor from measurement backdrop images to convert normalized coordinates to real-world centimeters.

**Features:**
- Detects bold horizontal lines (10cm demarcations) using Hough Line Transform
- Reads measurement numbers (20-220cm) using Tesseract OCR
- Calculates `cm_per_normalized_unit` conversion factor
- Provides confidence levels (high/medium/low/failed)
- Debug mode for troubleshooting detection

**Python API:**
```python
from measurement_extraction.extract_norm_per_cm import extract_calibration

calibration_data = extract_calibration(
    image_path="backdrop.jpg",
    output_path="calibration.json",
    tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    debug=False
)

# calibration_data contains:
# - cm_per_normalized_unit: conversion factor
# - pixels_per_cm: pixel density
# - detected_lines: number of lines detected
# - detected_numbers: number of OCR readings
# - confidence: "high", "medium", "low", or "failed"
```

**Command Line:**
```bash
# Basic usage
python measurement_extraction/extract_norm_per_cm.py backdrop.jpg -o calibration.json

# With debug output
python measurement_extraction/extract_norm_per_cm.py backdrop.jpg -o calibration.json --debug

# Custom Tesseract path
python measurement_extraction/extract_norm_per_cm.py backdrop.jpg -o calibration.json --tesseract-cmd "path/to/tesseract.exe"
```

**Output Format:**
```json
{
  "cm_per_normalized_unit": 195.42,
  "pixels_per_cm": 6.55,
  "detected_lines": 19,
  "detected_numbers": 17,
  "line_spacing_pixels": 65.5,
  "cm_interval": 10,
  "confidence": "high"
}
```

**Calibration Parameters:**

The line detection parameters can be tuned for different backdrop setups in `extract_norm_per_cm.py:47-59`:

```python
# Canny edge detection thresholds
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Hough Line Transform parameters
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi/180,
    threshold=106,                    # Higher = only bold lines
    minLineLength=image.shape[1] * 0.47,  # Minimum line length
    maxLineGap=25                     # Maximum gap in line
)
```

## Complete Measurement Pipeline Example

```python
import cv2
import json
from mediapipe_detection import (
    PoseDetector, FaceDetector, HandDetector,
    HairSegmenter, BodySegmenter
)
from measurement_extraction.extract_norm_per_cm import extract_calibration

# Step 1: Load calibration from backdrop image
calibration = extract_calibration("backdrop.jpg", "calibration.json")
cm_per_unit = calibration["cm_per_normalized_unit"]

# Step 2: Load subject image
image = cv2.imread('subject.jpg')
h, w = image.shape[:2]

# Step 3: Extract all measurements
pose_detector = PoseDetector()
face_detector = FaceDetector()
hand_detector = HandDetector()
hair_segmenter = HairSegmenter()
body_segmenter = BodySegmenter()

pose_data = pose_detector.detect(image)
face_data = face_detector.detect(image)
hand_data = hand_detector.detect(image)
hair_data = hair_segmenter.segment(image)
body_data = body_segmenter.segment(image)

# Step 4: Convert normalized coordinates to centimeters
def normalized_to_cm(normalized_value):
    return normalized_value * cm_per_unit

# Example: Calculate shoulder width in cm
if pose_data.get("shoulder_width"):
    left_shoulder = pose_data["shoulder_width"]["landmark_11"]
    right_shoulder = pose_data["shoulder_width"]["landmark_12"]
    shoulder_width_normalized = abs(left_shoulder["x"] - right_shoulder["x"])
    shoulder_width_cm = shoulder_width_normalized * cm_per_unit
    print(f"Shoulder width: {shoulder_width_cm:.2f} cm")

# Example: Calculate body height in cm
if body_data.get("height"):
    height_normalized = body_data["height"]["bottom"]["y"] - body_data["height"]["top"]["y"]
    height_cm = height_normalized * cm_per_unit
    print(f"Body height: {height_cm:.2f} cm")

# Clean up
pose_detector.close()
face_detector.close()
hand_detector.close()
hair_segmenter.close()
body_segmenter.close()
```

## Coordinate System

All landmark and segmentation outputs use **normalized coordinates** (0.0 to 1.0):

- **X-coordinates**: Normalized by image width (0 = left, 1 = right)
- **Y-coordinates**: Normalized by image height (0 = top, 1 = bottom)
- **Z-coordinates** (landmarks only): Depth relative to landmark 0, scaled by image width

To convert to real-world measurements:
```python
# For distances
distance_cm = normalized_distance * calibration["cm_per_normalized_unit"]

# For pixel coordinates
x_pixels = x_normalized * image_width
y_pixels = y_normalized * image_height
```

## API Reference

### Common Parameters

All detector classes support:
- `model_path` (Optional[str]): Path to model file. If None, downloads automatically.
- `min_detection_confidence` (float): Minimum confidence threshold (0.0-1.0)
- `min_presence_confidence` (float): Minimum presence confidence (0.0-1.0)
- `min_tracking_confidence` (float): Minimum tracking confidence (0.0-1.0)

### Return Formats

**Pose Detection:**
```python
{
  "head_width": {
    "landmark_7": {"x": float, "y": float, "z": float},
    "landmark_8": {"x": float, "y": float, "z": float}
  },
  "shoulder_width": {...},
  "hip_width": {...},
  "upper_arm_length": {
    "left": {"landmark_11": {...}, "landmark_13": {...}},
    "right": {"landmark_12": {...}, "landmark_14": {...}}
  },
  "forearm_length": {...}
}
```

**Hand Detection:**
```python
{
  "hands": [
    {
      "handedness": "Left" or "Right",
      "confidence": float,
      "hand_length": {
        "landmark_0": {"x": float, "y": float, "z": float},
        "landmark_12": {"x": float, "y": float, "z": float}
      }
    }
  ]
}
```

**Segmentation (Hair/Body):**
```python
{
  "hair_length" or "height": {
    "top": {"y": float},
    "bottom": {"y": float}
  }
}
```

## Best Practices

1. **Image Quality**:
   - Use well-lit images with clear subject visibility
   - Avoid shadows on measurement backdrop
   - Ensure subject stands straight against backdrop

2. **Calibration**:
   - Capture backdrop calibration image at the same distance as subject photos
   - Ensure bold measurement lines are clearly visible
   - Use debug mode to verify line and number detection

3. **Pose Detection**:
   - Subject should face camera directly for width measurements
   - Keep arms slightly away from body for better detection
   - Use T-pose or A-pose for most reliable results

4. **Hair Segmentation**:
   - Enable face detection for full-body images (default)
   - Disable face detection only for close-up head shots
   - Ensure hair contrasts with background

5. **Error Handling**:
   - Check for None/empty returns from detection methods
   - Validate calibration confidence before using measurements
   - Use try-except blocks for file I/O operations

## Troubleshooting

### Tesseract OCR Not Found
```
Error: Tesseract executable not found
```
**Solution**: Install Tesseract OCR and specify path:
```python
calibration = extract_calibration(
    "backdrop.jpg",
    tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
```

### Low Line Detection
```
Calibration failed: Not enough lines detected
```
**Solution**:
- Increase edge detection sensitivity by lowering Canny thresholds
- Reduce `threshold` parameter in HoughLinesP
- Use debug mode to visualize detected lines

### Poor Number Recognition
```
detected_numbers: 0 or very few
```
**Solution**:
- Ensure numbers are clearly visible and high contrast
- Check Tesseract installation
- Use `--debug` flag to see OCR output
- Verify image quality and lighting

### No Landmarks Detected
```
pose_data: {}
```
**Solution**:
- Verify subject is fully visible in frame
- Check image format (should be BGR from OpenCV)
- Lower confidence thresholds
- Ensure adequate lighting

## Technical Details

### Mediapipe API Version
This module uses the Mediapipe Tasks API (mediapipe>=0.10.0):
- `mediapipe.tasks.python.vision` for vision tasks
- Automatic model management and downloading
- Compatible with latest Mediapipe releases

### Image Format
- **Input**: BGR format (OpenCV default)
- **Internal processing**: Converted to RGB for Mediapipe
- **Output**: JSON format with normalized coordinates

### Performance
- **Pose Detection**: ~100-200ms per image (CPU)
- **Face Detection**: ~50-100ms per image (CPU)
- **Hand Detection**: ~50-100ms per image (CPU)
- **Segmentation**: ~200-400ms per image (CPU)
- **Calibration**: ~1-2s per image (CPU)

## References

- [Mediapipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [Mediapipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Mediapipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [Mediapipe Image Segmentation](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)
- [OpenCV Hough Line Transform](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## License

This module is part of the Hair Classification Capstone Project.
