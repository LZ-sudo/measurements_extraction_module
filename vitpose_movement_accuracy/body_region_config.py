"""
Body Region Configuration for Movement Accuracy Evaluation

Defines configurable body regions with their COCO-WholeBody keypoint indices
and angle triplets for joint angle error computation.

Sigma values are calibrated for torso-normalized coordinate space:
- A sigma of 0.10 means deviations within 10% of torso height are considered accurate.
- Smaller joints (wrist, hand MCP) use tighter sigmas.
"""

from typing import Dict, List

from pydantic import BaseModel


class JointInfo(BaseModel):
    name: str
    coco_index: int
    sigma: float


class AngleTriplet(BaseModel):
    name: str
    proximal_index: int
    vertex_index: int
    distal_index: int


class BodyRegion(BaseModel):
    name: str
    joints: List[JointInfo]
    angle_triplets: List[AngleTriplet] = []


# Sigma values in torso-normalized coordinate space.
# exp(-d^2 / (2 * sigma^2)): at d=sigma, OKS ~= 0.61
_SIGMA_SHOULDER = 0.10
_SIGMA_ELBOW    = 0.08
_SIGMA_WRIST    = 0.06
_SIGMA_MCP      = 0.05


# COCO-WholeBody hand keypoint layout (21 points per hand, offsets from hand base):
#   +0:  wrist
#   +1:  thumb CMC    +2:  thumb MCP    +3:  thumb IP     +4:  thumb TIP
#   +5:  index MCP    +6:  index PIP    +7:  index DIP    +8:  index TIP
#   +9:  middle MCP   +10: middle PIP   +11: middle DIP   +12: middle TIP
#   +13: ring MCP     +14: ring PIP     +15: ring DIP     +16: ring TIP
#   +17: pinky MCP    +18: pinky PIP    +19: pinky DIP    +20: pinky TIP
#
# Left hand base index: 91   Right hand base index: 112
# MCP knuckle indices (left / right):
#   thumb:  93 / 114    index:  96 / 117    middle: 100 / 121
#   ring:  104 / 125    pinky: 108 / 129

LEFT_ARM = BodyRegion(
    name="left_arm",
    joints=[
        JointInfo(name="left_shoulder",    coco_index=5,   sigma=_SIGMA_SHOULDER),
        JointInfo(name="left_elbow",       coco_index=7,   sigma=_SIGMA_ELBOW),
        JointInfo(name="left_wrist",       coco_index=9,   sigma=_SIGMA_WRIST),
        JointInfo(name="left_thumb_mcp",   coco_index=93,  sigma=_SIGMA_MCP),
        JointInfo(name="left_index_mcp",   coco_index=96,  sigma=_SIGMA_MCP),
        JointInfo(name="left_middle_mcp",  coco_index=100, sigma=_SIGMA_MCP),
        JointInfo(name="left_ring_mcp",    coco_index=104, sigma=_SIGMA_MCP),
        JointInfo(name="left_pinky_mcp",   coco_index=108, sigma=_SIGMA_MCP),
    ],
    angle_triplets=[
        AngleTriplet(
            name="left_elbow_angle",
            proximal_index=5,
            vertex_index=7,
            distal_index=9,
        ),
    ],
)

RIGHT_ARM = BodyRegion(
    name="right_arm",
    joints=[
        JointInfo(name="right_shoulder",   coco_index=6,   sigma=_SIGMA_SHOULDER),
        JointInfo(name="right_elbow",      coco_index=8,   sigma=_SIGMA_ELBOW),
        JointInfo(name="right_wrist",      coco_index=10,  sigma=_SIGMA_WRIST),
        JointInfo(name="right_thumb_mcp",  coco_index=114, sigma=_SIGMA_MCP),
        JointInfo(name="right_index_mcp",  coco_index=117, sigma=_SIGMA_MCP),
        JointInfo(name="right_middle_mcp", coco_index=121, sigma=_SIGMA_MCP),
        JointInfo(name="right_ring_mcp",   coco_index=125, sigma=_SIGMA_MCP),
        JointInfo(name="right_pinky_mcp",  coco_index=129, sigma=_SIGMA_MCP),
    ],
    angle_triplets=[
        AngleTriplet(
            name="right_elbow_angle",
            proximal_index=6,
            vertex_index=8,
            distal_index=10,
        ),
    ],
)

REGION_REGISTRY: Dict[str, BodyRegion] = {
    "left_arm":  LEFT_ARM,
    "right_arm": RIGHT_ARM,
}

DEFAULT_REGIONS: List[str] = ["left_arm", "right_arm"]
