from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.trajectory_schema import INFO_DIM, OBS_DIM, U_REF_DIM, require_2d


@dataclass(frozen=True)
class Phase4Field:
    name: str
    width: int
    unit: str
    description: str


VALIDATION_PHASES = ("approach_hold", "descend", "insert", "final_hold", "done")
VALIDATION_PHASE_TO_ID = {phase: idx for idx, phase in enumerate(VALIDATION_PHASES)}

IMPEDANCE_FIELDS = (
    Phase4Field("desired_position", 3, "m", "commanded peg_tip position in world frame"),
    Phase4Field("desired_rotation_matrix", 9, "unitless", "commanded peg_tip rotation matrix, row-major"),
    Phase4Field("position_error", 3, "m", "desired minus current peg_tip position"),
    Phase4Field("orientation_error", 3, "rad", "task-space orientation error vector"),
    Phase4Field("task_linear_velocity", 3, "m/s", "peg_tip translational velocity from task Jacobian"),
    Phase4Field("task_angular_velocity", 3, "rad/s", "peg_tip angular velocity from task Jacobian"),
    Phase4Field("stiffness_diag", 6, "N/m, Nm/rad", "fixed task-space stiffness diagonal"),
    Phase4Field("damping_diag", 6, "Ns/m, Nms/rad", "fixed task-space damping diagonal"),
    Phase4Field("task_wrench", 6, "N, Nm", "operational-space wrench before J^T mapping"),
    Phase4Field("raw_joint_torque", 7, "Nm", "unclipped arm torque requested by OSC"),
    Phase4Field("commanded_joint_torque", 7, "Nm", "arm torque after motor ctrlrange clipping"),
    Phase4Field("max_abs_commanded_torque", 1, "Nm", "maximum absolute clipped arm torque this step"),
    Phase4Field("torque_saturation_flag", 1, "bool", "1 if raw torque exceeded a motor ctrlrange"),
)


def field_dim(fields: tuple[Phase4Field, ...]) -> int:
    return sum(field.width for field in fields)


IMPEDANCE_DIM = field_dim(IMPEDANCE_FIELDS)
REQUIRED_PHASE4_ARRAY_KEYS = ("obs", "u_ref", "impedance", "info", "phase", "time")
REQUIRED_PHASE4_SUMMARY_KEYS = (
    "case_name",
    "steps",
    "termination_reason",
    "final_depth",
    "final_lateral_error",
    "final_axis_alignment",
    "final_orientation_error",
    "contact_detected",
    "hole_contact_detected",
    "contact_onset_step",
    "max_normal_force",
    "max_abs_commanded_torque",
    "torque_saturation_count",
    "jam_detected",
)


def validate_phase4_trace_arrays(arrays: dict[str, np.ndarray], summary: dict[str, Any] | None = None) -> None:
    missing = [key for key in REQUIRED_PHASE4_ARRAY_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"Phase 4 trace arrays missing keys: {missing}.")

    obs = np.asarray(arrays["obs"])
    u_ref = np.asarray(arrays["u_ref"])
    impedance = np.asarray(arrays["impedance"])
    info = np.asarray(arrays["info"])
    phase = np.asarray(arrays["phase"])
    time = np.asarray(arrays["time"])

    require_2d("obs", obs, OBS_DIM)
    require_2d("u_ref", u_ref, U_REF_DIM)
    require_2d("impedance", impedance, IMPEDANCE_DIM)
    require_2d("info", info, INFO_DIM)

    length = obs.shape[0]
    if length == 0:
        raise ValueError("Phase 4 trace arrays must contain at least one timestep.")
    for name, value in (("u_ref", u_ref), ("impedance", impedance), ("info", info), ("phase", phase), ("time", time)):
        if value.shape[0] != length:
            raise ValueError(f"{name} length {value.shape[0]} does not match obs length {length}.")
    if phase.ndim != 1:
        raise ValueError(f"phase must be a 1D array, got shape {phase.shape}.")
    if time.ndim != 1:
        raise ValueError(f"time must be a 1D array, got shape {time.shape}.")
    if not np.all(np.isfinite(impedance)):
        raise ValueError("impedance array contains non-finite values.")

    if summary is not None:
        missing_summary = [key for key in REQUIRED_PHASE4_SUMMARY_KEYS if key not in summary]
        if missing_summary:
            raise ValueError(f"Phase 4 summary missing keys: {missing_summary}.")
