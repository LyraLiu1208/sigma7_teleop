from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from stiffness_copilot_mujoco.panda_control import PandaArmIds, arm_qpos, arm_qvel, site_jacobians
from stiffness_copilot_mujoco.pose_math import orientation_error, site_rotation


TRACK_A_BASELINE_CONTROLLER_PROFILE = "mid_high_baseline"
TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE = "track_a_c600"
TRACK_A_CONTROLLER_PROFILE_ALIASES = {
    "mid_high": TRACK_A_BASELINE_CONTROLLER_PROFILE,
    "mid_high_base": TRACK_A_BASELINE_CONTROLLER_PROFILE,
}


@dataclass(frozen=True)
class TaskSpaceImpedanceGains:
    """Fixed OSC gains for Panda peg_tip control.

    The controller structure is adapted from kevinzakka/mjctrl `opspace.py`
    (Apache-2.0): task-space inertia shaping, nullspace posture control,
    and MuJoCo bias-force compensation.
    """

    position_stiffness: tuple[float, float, float] = (100.0, 100.0, 100.0)
    orientation_stiffness: tuple[float, float, float] = (50.0, 50.0, 50.0)
    nullspace_stiffness: tuple[float, float, float, float, float, float, float] = (
        75.0,
        75.0,
        50.0,
        50.0,
        40.0,
        25.0,
        25.0,
    )
    damping_ratio: float = 1.0
    position_twist_gain: float = 0.95
    orientation_twist_gain: float = 0.95
    integration_dt: float = 1.0
    task_inertia_pinv_rcond: float = 1e-2


@dataclass(frozen=True)
class TaskSpaceImpedanceCommand:
    torque: np.ndarray
    task_torque: np.ndarray
    nullspace_torque: np.ndarray
    wrench: np.ndarray
    desired_task_accel: np.ndarray
    twist_error: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    task_inertia: np.ndarray
    position_stiffness_matrix: np.ndarray
    position_damping_matrix: np.ndarray


def _tuple_floats(value: Any, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple.")
    result = tuple(float(item) for item in value)
    if len(result) != length:
        raise ValueError(f"{name} must have length {length}, got {len(result)}.")
    return result


def gains_from_mapping(mapping: dict[str, Any]) -> TaskSpaceImpedanceGains:
    return TaskSpaceImpedanceGains(
        position_stiffness=_tuple_floats(mapping["position_stiffness"], length=3, name="position_stiffness"),
        orientation_stiffness=_tuple_floats(mapping["orientation_stiffness"], length=3, name="orientation_stiffness"),
        nullspace_stiffness=_tuple_floats(mapping["nullspace_stiffness"], length=7, name="nullspace_stiffness"),
        damping_ratio=float(mapping["damping_ratio"]),
        position_twist_gain=float(mapping["position_twist_gain"]),
        orientation_twist_gain=float(mapping["orientation_twist_gain"]),
        integration_dt=float(mapping["integration_dt"]),
        task_inertia_pinv_rcond=float(mapping["task_inertia_pinv_rcond"]),
    )


def load_gain_profiles(config_path: Path) -> dict[str, dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"No gain profiles found in {config_path}.")
    return profiles


def load_task_space_impedance_gains(config_path: Path, profile: str | None = None) -> tuple[str, TaskSpaceImpedanceGains]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"No gain profiles found in {config_path}.")
    requested = profile or str(config.get("default_profile", "nominal"))
    selected = TRACK_A_CONTROLLER_PROFILE_ALIASES.get(requested, requested)
    if selected in profiles:
        return selected, gains_from_mapping(profiles[selected])
    if requested in profiles:
        return requested, gains_from_mapping(profiles[requested])
    raise KeyError(f"Unknown impedance gain profile {requested!r}. Available: {', '.join(sorted(profiles))}")


def _arm_inverse_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData, ids: PandaArmIds) -> np.ndarray:
    full_inverse_mass = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_solveM(model, data, full_inverse_mass, np.eye(model.nv))
    return full_inverse_mass[np.ix_(ids.qvel_addrs, ids.qvel_addrs)]


def _task_space_inertia(
    jacobian: np.ndarray,
    inverse_mass: np.ndarray,
    *,
    pinv_rcond: float,
) -> np.ndarray:
    inverse_task_inertia = jacobian @ inverse_mass @ jacobian.T
    if abs(float(np.linalg.det(inverse_task_inertia))) >= 1e-2:
        return np.linalg.inv(inverse_task_inertia)
    return np.linalg.pinv(inverse_task_inertia, rcond=pinv_rcond)


def _clip_arm_torque(model: mujoco.MjModel, ids: PandaArmIds, torque: np.ndarray) -> np.ndarray:
    clipped = np.asarray(torque, dtype=float).copy()
    for idx, actuator_id in enumerate(ids.actuator_ids):
        low, high = model.actuator_ctrlrange[actuator_id]
        clipped[idx] = np.clip(clipped[idx], low, high)
    return clipped


def _as_spd_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}.")
    matrix = 0.5 * (matrix + matrix.T)
    eigvals = np.linalg.eigvalsh(matrix)
    if np.any(eigvals <= 0.0):
        raise ValueError(f"{name} must be positive definite, got eigenvalues {eigvals}.")
    return matrix


def _spd_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(_as_spd_matrix(matrix, name="matrix"))
    sqrt_matrix = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return 0.5 * (sqrt_matrix + sqrt_matrix.T)


def _position_stiffness_matrix(
    gains: TaskSpaceImpedanceGains,
    position_stiffness_matrix: np.ndarray | None,
) -> np.ndarray:
    if position_stiffness_matrix is not None:
        return _as_spd_matrix(position_stiffness_matrix, name="position_stiffness_matrix")
    stiffness = np.asarray(gains.position_stiffness, dtype=float)
    if stiffness.shape != (3,) or np.any(stiffness <= 0.0):
        raise ValueError(f"gains.position_stiffness must contain three positive values, got {stiffness}.")
    return np.diag(stiffness)


def task_space_impedance_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    site_name: str,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    arm_ids: PandaArmIds,
    gains: TaskSpaceImpedanceGains = TaskSpaceImpedanceGains(),
    position_stiffness_matrix: np.ndarray | None = None,
    nullspace_target_qpos: np.ndarray | None = None,
    compensate_bias: bool = True,
    clip_to_ctrlrange: bool = True,
) -> TaskSpaceImpedanceCommand:
    site_id = model.site(site_name).id
    current_position = np.array(data.site_xpos[site_id], dtype=float)
    current_rotation = site_rotation(data, site_id)

    jacp, jacr = site_jacobians(model, data, site_name, arm_ids)
    jacobian = np.vstack([jacp, jacr])
    qpos = arm_qpos(data, arm_ids)
    qvel = arm_qvel(data, arm_ids)
    site_velocity = jacobian @ qvel

    position_error = np.asarray(target_position, dtype=float) - current_position
    rotation_error = orientation_error(current_rotation, np.asarray(target_rotation, dtype=float))
    twist_error = np.concatenate(
        [
            gains.position_twist_gain * position_error / gains.integration_dt,
            gains.orientation_twist_gain * rotation_error / gains.integration_dt,
        ]
    )

    k_pos = _position_stiffness_matrix(gains, position_stiffness_matrix)
    d_pos = gains.damping_ratio * 2.0 * _spd_matrix_sqrt(k_pos)
    k_rot = np.asarray(gains.orientation_stiffness, dtype=float)
    if k_rot.shape != (3,) or np.any(k_rot <= 0.0):
        raise ValueError(f"gains.orientation_stiffness must contain three positive values, got {k_rot}.")
    d_rot = gains.damping_ratio * 2.0 * np.sqrt(k_rot)
    inverse_mass = _arm_inverse_mass_matrix(model, data, arm_ids)
    task_inertia = _task_space_inertia(
        jacobian,
        inverse_mass,
        pinv_rcond=gains.task_inertia_pinv_rcond,
    )

    desired_linear = k_pos @ twist_error[:3] - d_pos @ site_velocity[:3]
    desired_angular = k_rot * twist_error[3:] - d_rot * site_velocity[3:]
    desired_task_accel = np.concatenate([desired_linear, desired_angular])
    wrench = task_inertia @ desired_task_accel
    task_torque = jacobian.T @ wrench

    nullspace_torque = np.zeros(7, dtype=float)
    if nullspace_target_qpos is not None:
        q0 = np.asarray(nullspace_target_qpos, dtype=float)
        if q0.shape != (7,):
            raise ValueError(f"Expected nullspace_target_qpos shape (7,), got {q0.shape}.")
        null_stiffness = np.asarray(gains.nullspace_stiffness, dtype=float)
        null_damping = gains.damping_ratio * 2.0 * np.sqrt(null_stiffness)
        jbar = inverse_mass @ jacobian.T @ task_inertia
        null_projection = np.eye(7) - jacobian.T @ jbar.T
        null_accel = null_stiffness * (q0 - qpos) - null_damping * qvel
        nullspace_torque = null_projection @ null_accel

    torque = task_torque + nullspace_torque
    if compensate_bias:
        torque = torque + np.asarray(data.qfrc_bias[list(arm_ids.qvel_addrs)], dtype=float)
    if clip_to_ctrlrange:
        torque = _clip_arm_torque(model, arm_ids, torque)

    return TaskSpaceImpedanceCommand(
        torque=torque,
        task_torque=task_torque,
        nullspace_torque=nullspace_torque,
        wrench=wrench,
        desired_task_accel=desired_task_accel,
        twist_error=twist_error,
        position_error=position_error,
        orientation_error=rotation_error,
        linear_velocity=site_velocity[:3],
        angular_velocity=site_velocity[3:],
        task_inertia=task_inertia,
        position_stiffness_matrix=k_pos,
        position_damping_matrix=d_pos,
    )
