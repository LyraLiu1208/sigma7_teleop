from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np

from stiffness_copilot_mujoco.metrics.task_metrics import (
    hole_center_position,
    hole_insertion_axis_world,
    insertion_depth,
    peg_axis_error,
    peg_tip_position,
)
from stiffness_copilot_mujoco.sim.ids import PegHoleIds


PRIVILEGED_TASK_STATE_NAMES = ["dx", "dy", "dz_insert", "yaw_error", "roll_error", "pitch_error"]
PRIVILEGED_TASK_STATE_UNITS = ["m", "m", "m", "rad", "rad", "rad"]
PRIVILEGED_TASK_STATE_FRAME = "hole/world task frame"
PRIVILEGED_TASK_STATE_ANGLE_WRAPPING = "[-pi, pi]"
PRIVILEGED_TASK_STATE_SCHEMA = "peg_hole_relative_6d_v1"


@dataclass(frozen=True)
class PrivilegedTaskStateMetadata:
    state_schema: str
    state_names: tuple[str, ...]
    state_units: tuple[str, ...]
    state_frame: str
    angle_wrapping: str
    yaw_source: str
    roll_pitch_source: str
    roll_pitch_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_schema": self.state_schema,
            "state_names": list(self.state_names),
            "state_units": list(self.state_units),
            "state_frame": self.state_frame,
            "angle_wrapping": self.angle_wrapping,
            "yaw_source": self.yaw_source,
            "roll_pitch_source": self.roll_pitch_source,
            "roll_pitch_available": bool(self.roll_pitch_available),
        }


def describe_privileged_task_state_schema() -> dict[str, Any]:
    return PrivilegedTaskStateMetadata(
        state_schema=PRIVILEGED_TASK_STATE_SCHEMA,
        state_names=tuple(PRIVILEGED_TASK_STATE_NAMES),
        state_units=tuple(PRIVILEGED_TASK_STATE_UNITS),
        state_frame=PRIVILEGED_TASK_STATE_FRAME,
        angle_wrapping=PRIVILEGED_TASK_STATE_ANGLE_WRAPPING,
        yaw_source="relative_yaw_from_body_rotations",
        roll_pitch_source="relative_roll_pitch_from_body_rotations",
        roll_pitch_available=True,
    ).to_dict()


def _wrap_to_pi(value: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(value, dtype=float)
    wrapped = np.arctan2(np.sin(arr), np.cos(arr))
    if np.isscalar(value):
        return float(wrapped)
    return wrapped


def _ensure_finite_state(state: np.ndarray) -> np.ndarray:
    if state.shape != (6,):
        raise ValueError(f"Privileged task state must have shape (6,), got {state.shape}.")
    if not np.all(np.isfinite(state)):
        raise ValueError("Privileged task state contains non-finite values.")
    state = np.asarray(state, dtype=float)
    state[3:] = _wrap_to_pi(state[3:])
    if np.any(state[3:] < -np.pi - 1e-9) or np.any(state[3:] > np.pi + 1e-9):
        raise ValueError("Angular components must be wrapped to [-pi, pi].")
    return state


def _relative_rotation_to_ypr(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    pitch = float(np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)))
    roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
    return _wrap_to_pi(yaw), _wrap_to_pi(roll), _wrap_to_pi(pitch)


def _body_rotation(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)


def _legacy_state_to_unified(
    legacy_task_state: np.ndarray,
    *,
    hole_yaw: float = 0.0,
) -> tuple[np.ndarray, PrivilegedTaskStateMetadata]:
    legacy = np.asarray(legacy_task_state, dtype=float).reshape(-1)
    if legacy.shape != (8,):
        raise ValueError(f"Legacy task_state must have shape (8,), got {legacy.shape}.")
    unified = np.array(
        [
            legacy[0],
            legacy[1],
            legacy[6],
            _wrap_to_pi(float(legacy[5]) - hole_yaw),
            0.0,
            0.0,
        ],
        dtype=float,
    )
    return _ensure_finite_state(unified), PrivilegedTaskStateMetadata(
        state_schema=PRIVILEGED_TASK_STATE_SCHEMA,
        state_names=tuple(PRIVILEGED_TASK_STATE_NAMES),
        state_units=tuple(PRIVILEGED_TASK_STATE_UNITS),
        state_frame=PRIVILEGED_TASK_STATE_FRAME,
        angle_wrapping=PRIVILEGED_TASK_STATE_ANGLE_WRAPPING,
        yaw_source="legacy_task_state_world_yaw_minus_hole_yaw",
        roll_pitch_source="zero_filled_from_legacy_task_state",
        roll_pitch_available=False,
    )


def _from_runtime_data(
    data: mujoco.MjData,
    ids: PegHoleIds,
) -> tuple[np.ndarray, PrivilegedTaskStateMetadata]:
    peg_tip = peg_tip_position(data, ids)
    hole_center = hole_center_position(data, ids)
    delta_world = np.asarray(peg_tip, dtype=float) - np.asarray(hole_center, dtype=float)

    hole_rotation = _body_rotation(data, ids.hole_body)
    peg_rotation = _body_rotation(data, ids.peg_body)
    relative_rotation = hole_rotation.T @ peg_rotation
    yaw_error, roll_error, pitch_error = _relative_rotation_to_ypr(relative_rotation)

    # The active peg-hole scenes keep the hole frame aligned with the world frame,
    # so the relative position is expressed in the hole/world task frame directly.
    state = np.array(
        [
            float(delta_world[0]),
            float(delta_world[1]),
            float(insertion_depth(data, ids)),
            float(yaw_error),
            float(roll_error),
            float(pitch_error),
        ],
        dtype=float,
    )
    return _ensure_finite_state(state), PrivilegedTaskStateMetadata(
        state_schema=PRIVILEGED_TASK_STATE_SCHEMA,
        state_names=tuple(PRIVILEGED_TASK_STATE_NAMES),
        state_units=tuple(PRIVILEGED_TASK_STATE_UNITS),
        state_frame=PRIVILEGED_TASK_STATE_FRAME,
        angle_wrapping=PRIVILEGED_TASK_STATE_ANGLE_WRAPPING,
        yaw_source="relative_rotation_matrix",
        roll_pitch_source="relative_rotation_matrix",
        roll_pitch_available=True,
    )


def compute_peg_in_hole_task_state_with_metadata(
    sample_or_trace_row: Any,
    scene_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    del scene_config
    if isinstance(sample_or_trace_row, np.ndarray):
        row = np.asarray(sample_or_trace_row, dtype=float).reshape(-1)
        if row.shape == (6,):
            state = _ensure_finite_state(row)
            metadata = describe_privileged_task_state_schema()
            metadata.update(
                {
                    "yaw_source": "precomputed_unified_state",
                    "roll_pitch_source": "precomputed_unified_state",
                    "roll_pitch_available": True,
                }
            )
            return state, metadata
        if row.shape == (8,):
            state, info = _legacy_state_to_unified(row)
            return state, info.to_dict()
        raise ValueError(f"Unsupported ndarray row shape {row.shape}; expected (6,) or (8,).")

    if isinstance(sample_or_trace_row, Mapping):
        mapping = sample_or_trace_row
        if "task_state" in mapping:
            task_state = np.asarray(mapping["task_state"], dtype=float).reshape(-1)
            if task_state.shape == (6,):
                state = _ensure_finite_state(task_state)
                metadata = describe_privileged_task_state_schema()
                metadata.update(
                    {
                        "yaw_source": "precomputed_unified_state",
                        "roll_pitch_source": "precomputed_unified_state",
                        "roll_pitch_available": True,
                    }
                )
                return state, metadata
            if task_state.shape == (8,):
                state, info = _legacy_state_to_unified(task_state)
                return state, info.to_dict()

        if "data" in mapping and "task_ids" in mapping:
            data = mapping["data"]
            ids = mapping["task_ids"]
            if isinstance(data, mujoco.MjData) and isinstance(ids, PegHoleIds):
                return _from_runtime_data(data, ids)

    if isinstance(sample_or_trace_row, mujoco.MjData) and scene_config is not None:
        raise ValueError("A PegHoleIds instance is required when computing privileged task state from MjData.")

    raise ValueError(
        "Unsupported input for compute_peg_in_hole_task_state_with_metadata; expected a 6D/8D array, "
        "a mapping with task_state, or a mapping containing Mujoco data and task ids."
    )


def compute_peg_in_hole_task_state(
    sample_or_trace_row: Any,
    scene_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    state, _ = compute_peg_in_hole_task_state_with_metadata(sample_or_trace_row, scene_config=scene_config)
    return state


def peg_hole_task_state(data: mujoco.MjData, ids: PegHoleIds, *, hole_clearance_delta: float = 0.0) -> np.ndarray:
    del hole_clearance_delta
    state, _ = _from_runtime_data(data, ids)
    return state


__all__ = [
    "PRIVILEGED_TASK_STATE_ANGLE_WRAPPING",
    "PRIVILEGED_TASK_STATE_FRAME",
    "PRIVILEGED_TASK_STATE_NAMES",
    "PRIVILEGED_TASK_STATE_SCHEMA",
    "PRIVILEGED_TASK_STATE_UNITS",
    "PrivilegedTaskStateMetadata",
    "compute_peg_in_hole_task_state",
    "compute_peg_in_hole_task_state_with_metadata",
    "describe_privileged_task_state_schema",
    "peg_hole_task_state",
]
