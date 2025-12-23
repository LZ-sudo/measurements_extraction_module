# Mediapipe Detection Module

This module contains scripts for detecting landmarks and segmenting features in images using Google's Mediapipe library and computer vision techniques.

## Scripts Overview

### 1. pose_detection.py
Detects 33 body pose landmarks using Mediapipe Pose Landmarker.

**Features:**
- Full body pose detection
- Automatic model download
- Configurable confidence thresholds
- Visual overlay of landmarks and connections

**Usage:**
```python
from mediapipe_detection import PoseDetector

detector = PoseDetector()
annotated_image, landmarks = detector.detect(image)
detector.close()
```

**Command Line:**
```bash
python pose_detection.py input.jpg -o output.jpg --show
```

### 2. face_detection.py
Detects 478 facial landmarks using Mediapipe Face Landmarker.

**Features:**
- Detailed face mesh with 478 3D landmarks
- Face tesselation, contours, and iris detection
- Multiple face support
- Automatic model download

**Usage:**
```python
from mediapipe_detection import FaceDetector

detector = FaceDetector(num_faces=1)
annotated_image, landmarks = detector.detect(image)
detector.close()
```

**Command Line:**
```bash
python face_detection.py input.jpg -o output.jpg --show --max-faces 1
```

### 3. hand_detection.py
Detects 21 hand landmarks per hand using Mediapipe Hand Landmarker.

**Features:**
- Detects up to 2 hands (configurable)
- Left/Right handedness classification
- Confidence scores for each detection
- Automatic model download

**Usage:**
```python
from mediapipe_detection import HandDetector

detector = HandDetector(num_hands=2)
annotated_image, landmarks, handedness = detector.detect(image)
detector.close()
```

**Command Line:**
```bash
python hand_detection.py input.jpg -o output.jpg --show --max-hands 2
```

### 4. hair_segmentation.py
Performs hair segmentation using computer vision techniques.

**Features:**
- Two segmentation methods:
  - `color_based`: HSV color space segmentation
  - `grabcut`: GrabCut algorithm for region segmentation
- Outputs both annotated image and binary mask
- Configurable morphological operations

**Usage:**
```python
from mediapipe_detection import HairSegmenter

segmenter = HairSegmenter(method='color_based')
annotated_image, mask = segmenter.segment(image)
```

**Command Line:**
```bash
python hair_segmentation.py input.jpg -o output.jpg -m mask.jpg --method color_based --show
```

## Installation

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. The Mediapipe models will be automatically downloaded on first use:
   - `pose_landmarker_heavy.task` (~30MB)
   - `face_landmarker.task` (~10MB)
   - `hand_landmarker.task` (~10MB)

## API Reference

### Common Parameters

All detector classes support these common parameters:
- `model_path` (Optional[str]): Path to model file. If None, downloads automatically.
- `min_*_detection_confidence` (float): Minimum confidence threshold for detection (0.0-1.0)
- `min_*_presence_confidence` (float): Minimum confidence for presence (0.0-1.0)
- `min_tracking_confidence` (float): Minimum confidence for tracking (0.0-1.0)

### Return Values

All `detect()` methods return:
- `annotated_image`: NumPy array with landmarks drawn
- `landmarks`: Detected landmarks data (or None if not detected)

## Integration Example

```python
import cv2
from mediapipe_detection import PoseDetector, FaceDetector, HandDetector, HairSegmenter

# Load image
image = cv2.imread('input.jpg')

# Initialize detectors
pose_detector = PoseDetector()
face_detector = FaceDetector()
hand_detector = HandDetector()
hair_segmenter = HairSegmenter()

# Run all detections
pose_img, pose_landmarks = pose_detector.detect(image)
face_img, face_landmarks = face_detector.detect(image)
hand_img, hand_landmarks, handedness = hand_detector.detect(image)
hair_img, hair_mask = hair_segmenter.segment(image)

# Clean up
pose_detector.close()
face_detector.close()
hand_detector.close()

# Use the results for measurements...
```

### 5. body_segmentation.py
Performs full-body segmentation using MediaPipe's selfie segmenter.

**Features:**
- Segments entire body from background
- Outputs normalized height coordinates (top/bottom y-coordinates)
- Uses confidence mask thresholding
- Returns JSON format

**Usage:**
```python
from mediapipe_detection import BodySegmenter

segmenter = BodySegmenter()
height_data = segmenter.segment(image)
segmenter.close()
```

**Command Line:**
```bash
python body_segmentation.py input.jpg -o output.json
```

### 6. extract_norm_per_cm.py
Extracts calibration factor from measurement backdrop images using OpenCV line detection and Tesseract OCR.

**Features:**
- Detects horizontal lines using Hough Line Transform
- Extracts measurement numbers using Tesseract OCR (high accuracy for printed text)
- Calculates cm_per_normalized_unit for converting measurements
- Returns calibration confidence level
- Debug mode to troubleshoot OCR detection

**Prerequisites:**
- Tesseract OCR must be installed locally
- Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe` (Windows)

**Usage:**
```python
from measurement_extraction import extract_calibration

calibration_data = extract_calibration("backdrop.jpg", "calibration.json")
```

**Command Line:**
```bash
python measurement_extraction/extract_norm_per_cm.py input.jpg -o calibration.json
```

Specify custom Tesseract path:
```bash
python measurement_extraction/extract_norm_per_cm.py input.jpg -o calibration.json --tesseract-cmd "path/to/tesseract.exe"
```

## Notes

- All images are expected in BGR format (OpenCV default)
- Internally converted to RGB for Mediapipe processing
- Models are downloaded automatically on first use
- For best results, ensure good lighting and clear subject visibility
- Hair segmentation may require tuning for different hair colors/types
- Tesseract OCR 5.5.0 requires local installation (https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)

## Mediapipe API Version

This module uses the new Mediapipe Tasks API (mediapipe>=0.10.0):
- `mediapipe.tasks.python.vision` for vision tasks
- Landmark detection with automatic model management
- Compatible with the latest Mediapipe releases

## References

- [Mediapipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [Mediapipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)
- [Mediapipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [Mediapipe Image Segmentation](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)
