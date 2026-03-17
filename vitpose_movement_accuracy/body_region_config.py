"""
Body Region Configuration for Movement Accuracy Evaluation

Defines configurable body regions with their COCO-WholeBody keypoint indices
and angle triplets for interior-angle-based accuracy computation.

Accuracy is measured by computing the interior angle at each joint vertex
from three keypoint indices (vertex, arm-A, arm-B) in the torso-normalised
coordinate frame. Using angles rather than absolute positions makes the metric
invariant to subject scale, camera distance, and the avatar's global position
in frame.

sigma_deg controls the Gaussian accuracy falloff: accuracy ~= 0.61 when the
angular error equals sigma_deg.
"""

from typing import Dict, List

from pydantic import BaseModel


class JointInfo(BaseModel):
    name: str
    coco_index: int


class AngleTriplet(BaseModel):
    name: str
    vertex_index: int  # joint where the angle is measured
    a_index: int       # proximal landmark (first arm of the angle)
    b_index: int       # distal landmark (second arm of the angle)
    sigma_deg: float   # angular tolerance in degrees


class BodyRegion(BaseModel):
    name: str
    joints: List[JointInfo]
    angle_triplets: List[AngleTriplet] = []


LEFT_ARM = BodyRegion(
    name="left_arm",
    joints=[
        # Ipsilateral arm chain
        JointInfo(name="left_shoulder",    coco_index=5),
        JointInfo(name="left_elbow",       coco_index=7),
        JointInfo(name="left_wrist",       coco_index=9),
        JointInfo(name="left_thumb_cmc",   coco_index=92),
        JointInfo(name="left_index_mcp",   coco_index=96),
        JointInfo(name="left_middle_mcp",  coco_index=100),
        JointInfo(name="left_ring_mcp",    coco_index=104),
        JointInfo(name="left_pinky_mcp",   coco_index=108),
        # Torso reference landmarks
        JointInfo(name="right_shoulder",   coco_index=6),
        JointInfo(name="left_hip",         coco_index=11),
        JointInfo(name="right_hip",        coco_index=12),
    ],
    angle_triplets=[
        # Arm elevation: angle at the shoulder between the torso side and upper arm
        AngleTriplet(name="left_shoulder_angle", vertex_index=5,  a_index=11, b_index=7,  sigma_deg=15.0),
        # Elbow flexion: angle at the elbow between the upper arm and forearm
        AngleTriplet(name="left_elbow_angle",    vertex_index=7,  a_index=5,  b_index=9,  sigma_deg=15.0),
    ],
)

RIGHT_ARM = BodyRegion(
    name="right_arm",
    joints=[
        # Ipsilateral arm chain
        JointInfo(name="right_shoulder",   coco_index=6),
        JointInfo(name="right_elbow",      coco_index=8),
        JointInfo(name="right_wrist",      coco_index=10),
        JointInfo(name="right_thumb_cmc",  coco_index=113),
        JointInfo(name="right_index_mcp",  coco_index=117),
        JointInfo(name="right_middle_mcp", coco_index=121),
        JointInfo(name="right_ring_mcp",   coco_index=125),
        JointInfo(name="right_pinky_mcp",  coco_index=129),
        # Torso reference landmarks
        JointInfo(name="left_shoulder",    coco_index=5),
        JointInfo(name="left_hip",         coco_index=11),
        JointInfo(name="right_hip",        coco_index=12),
    ],
    angle_triplets=[
        # Arm elevation: angle at the shoulder between the torso side and upper arm
        AngleTriplet(name="right_shoulder_angle", vertex_index=6,  a_index=12, b_index=8,  sigma_deg=15.0),
        # Elbow flexion: angle at the elbow between the upper arm and forearm
        AngleTriplet(name="right_elbow_angle",    vertex_index=8,  a_index=6,  b_index=10, sigma_deg=15.0),
    ],
)

REGION_REGISTRY: Dict[str, BodyRegion] = {
    "left_arm":  LEFT_ARM,
    "right_arm": RIGHT_ARM,
}

DEFAULT_REGIONS: List[str] = ["left_arm", "right_arm"]
