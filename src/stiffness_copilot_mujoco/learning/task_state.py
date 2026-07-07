from __future__ import annotations

from stiffness_copilot_mujoco.learning.privileged_task_state import (
    PRIVILEGED_TASK_STATE_ANGLE_WRAPPING,
    PRIVILEGED_TASK_STATE_FRAME,
    PRIVILEGED_TASK_STATE_NAMES,
    PRIVILEGED_TASK_STATE_SCHEMA,
    PRIVILEGED_TASK_STATE_UNITS,
    PrivilegedTaskStateMetadata,
    compute_peg_in_hole_task_state,
    compute_peg_in_hole_task_state_with_metadata,
    describe_privileged_task_state_schema,
    peg_hole_task_state,
)


TASK_STATE_DIM = 6


__all__ = [
    "PRIVILEGED_TASK_STATE_ANGLE_WRAPPING",
    "PRIVILEGED_TASK_STATE_FRAME",
    "PRIVILEGED_TASK_STATE_NAMES",
    "PRIVILEGED_TASK_STATE_SCHEMA",
    "PRIVILEGED_TASK_STATE_UNITS",
    "PrivilegedTaskStateMetadata",
    "TASK_STATE_DIM",
    "compute_peg_in_hole_task_state",
    "compute_peg_in_hole_task_state_with_metadata",
    "describe_privileged_task_state_schema",
    "peg_hole_task_state",
]
