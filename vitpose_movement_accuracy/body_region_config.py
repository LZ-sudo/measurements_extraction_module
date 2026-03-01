"""
Body Region Configuration for Movement Accuracy Evaluation

Defines configurable body regions with their COCO-WholeBody keypoint indices
and angle triplets for joint angle error computation.

Each arm region includes the ipsilateral arm chain, the contralateral shoulder,
and both hips as reference landmarks so that DTW alignment and visualisation
always carry the full torso frame of reference.
"""

from typing import Dict, List

from pydantic import BaseModel


class JointInfo(BaseModel):
    name: str
    coco_index: int


class AngleTriplet(BaseModel):
    name: str
    proximal_index: int
    vertex_index: int
    distal_index: int
    sigma_deg: float  # angle tolerance in degrees; accuracy ~= 0.61 at this error


class BodyRegion(BaseModel):
    name: str
    joints: List[JointInfo]
    angle_triplets: List[AngleTriplet] = []


# COCO-WholeBody hand keypoint layout (21 points per hand, offsets from hand base):
#   +0:  wrist
#   +1:  thumb CMC    +2:  thumb MCP    +3:  thumb IP     +4:  thumb TIP
#   +5:  index MCP    +6:  index PIP    +7:  index DIP    +8:  index TIP
#   +9:  middle MCP   +10: middle PIP   +11: middle DIP   +12: middle TIP
#   +13: ring MCP     +14: ring PIP     +15: ring DIP     +16: ring TIP
#   +17: pinky MCP    +18: pinky PIP    +19: pinky DIP    +20: pinky TIP
#
# Left hand base index: 91   Right hand base index: 112
# Base knuckle indices (left / right) — CMC for thumb, MCP for all other fingers:
#   thumb CMC: 92 / 113    index MCP:  96 / 117    middle MCP: 100 / 121
#   ring MCP: 104 / 125    pinky MCP: 108 / 129

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
        AngleTriplet(
            name="left_arm_elevation",
            proximal_index=6,
            vertex_index=5,
            distal_index=7,
            sigma_deg=15.0,
        ),
        AngleTriplet(
            name="left_elbow_angle",
            proximal_index=5,
            vertex_index=7,
            distal_index=9,
            sigma_deg=10.0,
        ),
        AngleTriplet(
            name="left_wrist_angle",
            proximal_index=7,
            vertex_index=9,
            distal_index=96,
            sigma_deg=12.0,
        ),
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
        AngleTriplet(
            name="right_arm_elevation",
            proximal_index=5,
            vertex_index=6,
            distal_index=8,
            sigma_deg=15.0,
        ),
        AngleTriplet(
            name="right_elbow_angle",
            proximal_index=6,
            vertex_index=8,
            distal_index=10,
            sigma_deg=10.0,
        ),
        AngleTriplet(
            name="right_wrist_angle",
            proximal_index=8,
            vertex_index=10,
            distal_index=117,
            sigma_deg=12.0,
        ),
    ],
)

REGION_REGISTRY: Dict[str, BodyRegion] = {
    "left_arm":  LEFT_ARM,
    "right_arm": RIGHT_ARM,
}

DEFAULT_REGIONS: List[str] = ["left_arm", "right_arm"]
