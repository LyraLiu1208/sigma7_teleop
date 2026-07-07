from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    width: int
    unit: str
    description: str


OBS_FIELDS = (
    Field("ee_position", 3, "m", "peg_tip site position in world frame"),
    Field("ee_rotation_matrix", 9, "unitless", "peg_tip site rotation matrix, row-major"),
    Field("ee_linear_velocity", 3, "m/s", "peg_tip translational velocity in world frame"),
    Field("ee_angular_velocity", 3, "rad/s", "peg_tip angular velocity in world frame"),
    Field("peg_position", 3, "m", "peg body position in world frame"),
    Field("peg_rotation_matrix", 9, "unitless", "peg body rotation matrix, row-major"),
    Field("hole_position", 3, "m", "hole_center site position in world frame"),
    Field("relative_peg_hole_position", 3, "m", "peg_tip minus hole_center in world frame"),
    Field("insertion_depth", 1, "m", "positive depth along the hole insertion axis"),
    Field("lateral_error", 1, "m", "radial peg_tip error around the hole axis"),
    Field("axis_alignment", 1, "unitless", "dot product between peg local axis and hole axis"),
    Field("orientation_error", 1, "rad", "norm of task-space orientation error vector"),
    Field("contact_state", 1, "id", "0 no peg contact, 1 peg non-hole contact, 2 peg-hole contact"),
    Field("normal_force_proxy", 1, "N", "sum of MuJoCo normal contact force magnitudes on peg contacts"),
)

U_REF_FIELDS = (
    Field("target_position", 3, "m", "desired peg_tip position in world frame"),
    Field("target_rotation_matrix", 9, "unitless", "desired peg_tip rotation, row-major"),
    Field("delta_position", 3, "m", "target_position minus current ee_position"),
    Field("delta_orientation_error", 3, "rad", "task-space orientation error vector"),
    Field("phase_id", 1, "id", "controller or validation phase id"),
)

INFO_FIELDS = (
    Field("insertion_depth", 1, "m", "positive depth along the hole insertion axis"),
    Field("lateral_error", 1, "m", "radial peg_tip error around the hole axis"),
    Field("axis_alignment", 1, "unitless", "dot product between peg local axis and hole axis"),
    Field("orientation_error", 1, "rad", "norm of task-space orientation error vector"),
    Field("peg_contact_count", 1, "count", "number of active peg contacts"),
    Field("peg_hole_contact_count", 1, "count", "number of active peg-hole contacts"),
    Field("normal_force_proxy", 1, "N", "sum of MuJoCo normal contact force magnitudes on peg contacts"),
    Field("min_contact_distance", 1, "m", "minimum peg contact distance, NaN if no peg contact"),
    Field("contact_state", 1, "id", "0 no peg contact, 1 peg non-hole contact, 2 peg-hole contact"),
)


def field_dim(fields: tuple[Field, ...]) -> int:
    return sum(field.width for field in fields)


OBS_DIM = field_dim(OBS_FIELDS)
U_REF_DIM = field_dim(U_REF_FIELDS)
INFO_DIM = field_dim(INFO_FIELDS)


def require_2d(name: str, value: np.ndarray, width: int) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {value.shape}.")
    if value.shape[1] != width:
        raise ValueError(f"{name} width must be {width}, got shape {value.shape}.")
