"""
Hand Landmark Detection Module using MediaPipe Hand Landmarker (Tasks API)

Detects 21 hand landmarks per hand using MediaPipe's HandLandmarker.
The model file (hand_landmarker.task, ~29 MB) is downloaded automatically
from Google's model storage on first use and cached in weight_files/.

Output schema is identical to vitpose_detection/hand_detection.py so this
module can serve as a drop-in replacement once validated.

Handedness note: MediaPipe labels hands from the image's perspective (not the
subject's). In a front-facing photo the person's left hand sits on the LEFT
side of the image, which MediaPipe classifies as "Right" (mirror convention).
detect() corrects for this so "Left"/"Right" refer to the subject's own hands.

Usage (standalone):
    python hand_detection.py subject.jpg -v output_hands.jpg -o output_hands.json
"""

import cv2
import json
import urllib.request
import mediapipe as mp
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


# Model details
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MODEL_FILENAME = "hand_landmarker.task"


# Hand keypoint names (MediaPipe / COCO-WholeBody share the same 21-point layout)
HAND_KEYPOINT_NAMES = [
    "wrist",        # 0  -- used for hand length
    "thumb_cmc",    # 1
    "thumb_mcp",    # 2
    "thumb_ip",     # 3
    "thumb_tip",    # 4
    "index_mcp",    # 5
    "index_pip",    # 6
    "index_dip",    # 7
    "index_tip",    # 8
    "middle_mcp",   # 9
    "middle_pip",   # 10
    "middle_dip",   # 11
    "middle_tip",   # 12  -- used for hand length
    "ring_mcp",     # 13
    "ring_pip",     # 14
    "ring_dip",     # 15
    "ring_tip",     # 16
    "pinky_mcp",    # 17
    "pinky_pip",    # 18
    "pinky_dip",    # 19
    "pinky_tip",    # 20
]

# Finger bone connections for visualization
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (5, 9), (9, 13), (13, 17),               # Palm
]


def _get_model_path() -> str:
    """
    Return the path to hand_landmarker.task, downloading it if not present.
    The file is cached in weight_files/ next to the module root.
    """
    weight_dir = Path(__file__).resolve().parent.parent / "weight_files"
    weight_dir.mkdir(parents=True, exist_ok=True)
    model_path = weight_dir / _MODEL_FILENAME

    if not model_path.exists():
        print(f"Downloading hand landmarker model to {model_path} ...")
        urllib.request.urlretrieve(_MODEL_URL, str(model_path))
        print("Download complete.")

    return str(model_path)


class HandLandmarkDetector:
    """
    Detects 21 hand landmarks per hand using MediaPipe HandLandmarker (Tasks API).
    Output format is compatible with the existing pipeline's detect_hands() schema.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the MediaPipe HandLandmarker detector.

        Args:
            model_path: Path to hand_landmarker.task. Downloaded automatically if None.
            num_hands: Maximum number of hands to detect (default 2).
            min_hand_detection_confidence: Minimum confidence for initial detection.
            min_hand_presence_confidence: Minimum confidence for presence check.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        if model_path is None:
            model_path = _get_model_path()

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def _landmarks_to_list(self, hand_landmarks) -> List[Dict]:
        """Convert a MediaPipe NormalizedLandmark list to coordinate dicts."""
        return [
            {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
            for lm in hand_landmarks
        ]

    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect hand landmarks in a BGR image.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            Dictionary matching the pipeline's hand detection schema:
            {
                "hands": [
                    {
                        "handedness": "Left" or "Right"  (subject's perspective),
                        "confidence": float,
                        "hand_length": {
                            "landmark_0":  {"x": ..., "y": ..., "z": ...},
                            "landmark_12": {"x": ..., "y": ..., "z": ...}
                        },
                        "all_landmarks": { "landmark_0": {...}, ..., "landmark_20": {...} }
                    }
                ]
            }
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        hand_data: Dict = {"hands": []}

        if not results.hand_landmarks:
            return hand_data

        for hand_landmarks, handedness_list in zip(
            results.hand_landmarks, results.handedness
        ):
            landmarks = self._landmarks_to_list(hand_landmarks)

            # MediaPipe uses mirror convention; flip to get subject's own hand label
            raw_label = handedness_list[0].display_name   # "Left" or "Right"
            subject_label = "Right" if raw_label == "Left" else "Left"
            confidence = float(handedness_list[0].score)

            all_landmarks = {
                f"landmark_{i}": lm for i, lm in enumerate(landmarks)
            }

            hand_info = {
                "handedness": subject_label,
                "confidence": confidence,
                "hand_length": {
                    "landmark_0":  landmarks[0],   # Wrist
                    "landmark_12": landmarks[12],  # Middle finger tip
                },
                "all_landmarks": all_landmarks,
            }
            hand_data["hands"].append(hand_info)

        return hand_data

    def visualize(
        self,
        image: np.ndarray,
        detection_result: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Draw hand landmarks and the wrist-to-middle-tip measurement line.

        Args:
            image: Input BGR image.
            detection_result: Optional pre-computed result from detect().

        Returns:
            Annotated BGR image.
        """
        if detection_result is None:
            detection_result = self.detect(image)

        h, w = image.shape[:2]
        annotated = image.copy()

        hand_colors = {
            "Left":  (255, 128, 0),   # Orange for subject's left hand
            "Right": (0, 128, 255),   # Blue for subject's right hand
        }

        for hand in detection_result.get("hands", []):
            label = hand["handedness"]
            color = hand_colors.get(label, (0, 255, 0))
            all_lm = hand.get("all_landmarks", {})

            def px(key):
                lm = all_lm[key]
                return (int(lm["x"] * w), int(lm["y"] * h))

            # Draw connections
            for a, b in _HAND_CONNECTIONS:
                ka, kb = f"landmark_{a}", f"landmark_{b}"
                if ka in all_lm and kb in all_lm:
                    cv2.line(annotated, px(ka), px(kb), color, 2, cv2.LINE_AA)

            # Draw all keypoints
            for i in range(len(HAND_KEYPOINT_NAMES)):
                key = f"landmark_{i}"
                if key not in all_lm:
                    continue
                pt = px(key)
                radius = 6 if i in (0, 4, 8, 12, 16, 20) else 4
                cv2.circle(annotated, pt, radius, color, -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, radius, (255, 255, 255), 1, cv2.LINE_AA)

            # Highlight wrist and middle-tip (the two measurement landmarks)
            wrist_pt = px("landmark_0")
            tip_pt   = px("landmark_12")
            cv2.circle(annotated, wrist_pt, 9, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(annotated, tip_pt,   9, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.line(annotated, wrist_pt, tip_pt, (0, 255, 0), 2, cv2.LINE_AA)

            # Confidence label at the wrist
            lx, ly = wrist_pt
            cv2.putText(
                annotated, f"{hand['confidence']:.2f}",
                (lx - 20, ly - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

        return annotated

    def close(self):
        """Release resources."""
        self._landmarker.close()


def detect_hands(
    image_path: str,
    output_path: Optional[str] = None,
    visualize_path: Optional[str] = None,
    min_hand_detection_confidence: float = 0.5,
) -> Dict:
    """
    Detect hand landmarks in an image file.

    Args:
        image_path: Path to input image file.
        output_path: Optional path to save the JSON output.
        visualize_path: Optional path to save a visualisation image.
        min_hand_detection_confidence: Minimum confidence for hand detection.

    Returns:
        Dictionary containing hand landmarks matching the pipeline schema.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")

    detector = HandLandmarkDetector(
        min_hand_detection_confidence=min_hand_detection_confidence,
    )
    hand_data = detector.detect(image)

    if visualize_path:
        annotated = detector.visualize(image, hand_data)
        cv2.imwrite(visualize_path, annotated)
        print(f"Visualisation saved to {visualize_path}")

    detector.close()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hand_data, f, indent=2)
        print(f"Hand landmark coordinates saved to {output_path}")

    return hand_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect hand landmarks using MediaPipe HandLandmarker"
    )
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, help="Path to save JSON output")
    parser.add_argument(
        "-v", "--visualize", type=str,
        help="Path to save visualisation image with landmarks",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Minimum hand detection confidence (default: 0.5)",
    )
    args = parser.parse_args()

    result = detect_hands(
        args.input_image,
        args.output,
        args.visualize,
        min_hand_detection_confidence=args.conf,
    )

    # Print a compact summary (not the full all_landmarks block)
    print("\nHand Landmarks (MediaPipe):")
    summary = {
        "hands_detected": len(result["hands"]),
        "hands": [
            {
                "handedness":  h["handedness"],
                "confidence":  round(h["confidence"], 4),
                "wrist":       h["hand_length"]["landmark_0"],
                "middle_tip":  h["hand_length"]["landmark_12"],
            }
            for h in result["hands"]
        ],
    }
    print(json.dumps(summary, indent=2))
