from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import (
    ContactQuery,
    contact_state_vector,
    extract_contact_state,
    extract_net_peg_hole_contact_force_world,
)
from stiffness_copilot_mujoco.controllers.impedance import (
    load_task_space_impedance_gains,
    task_space_impedance_torque,
)
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.learning.dataset_schema import validate_learning_dataset
from stiffness_copilot_mujoco.learning.stiffness_labels import StiffnessLabelConfig, build_stiffness_labels_with_diagnostics
from stiffness_copilot_mujoco.learning.task_state import peg_hole_task_state
from stiffness_copilot_mujoco.metrics.task_metrics import geometry_from_config, hole_center_position, load_scene_config
from stiffness_copilot_mujoco.panda_control import arm_qpos, arm_qvel, panda_arm_ids, set_arm_torque_ctrl
from stiffness_copilot_mujoco.pose_math import site_rotation
from stiffness_copilot_mujoco.rollout_observation import collect_step, reset_from_config
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
from stiffness_copilot_mujoco.sim.scene import ROOT as PROJECT_ROOT
from stiffness_copilot_mujoco.sim.scene import cleanup_runtime_scene, render_config_file, render_runtime_config


DEFAULT_TORQUE_CONFIG = PROJECT_ROOT / "configs" / "scenes" / "panda_peg_in_hole_torque.yaml"
DEFAULT_GAIN_CONFIG = PROJECT_ROOT / "configs" / "controllers" / "fixed_impedance.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "datasets" / "learning"

MODE_SUCCESS_INSERTION = 0
MODE_RIM_PROBE = 1
MODE_SPIRAL_SEARCH = 2
MODE_YAW_PROBE = 3
MODE_JAM_RECOVERY = 4
MODE_PERTURBED_INSERTION = 5
MODE_NAMES = {
    MODE_SUCCESS_INSERTION: "success_insertion",
    MODE_RIM_PROBE: "rim_probe",
    MODE_SPIRAL_SEARCH: "spiral_search",
    MODE_YAW_PROBE: "yaw_probe",
    MODE_JAM_RECOVERY: "jam_recovery",
    MODE_PERTURBED_INSERTION: "perturbed_insertion",
}


def _phase_for_step(step: int, approach_steps: int, descend_steps: int, insert_steps: int, hold_steps: int) -> tuple[int, int, int]:
    lengths = (approach_steps, descend_steps, insert_steps, hold_steps)
    cursor = 0
    for phase_id, length in enumerate(lengths):
        if step < cursor + length:
            return phase_id, step - cursor, length
        cursor += length
    return 3, max(step - cursor, 0), 1


def _target_position(
    hole_center: np.ndarray,
    xy_offset: np.ndarray,
    phase_id: int,
    phase_step: int,
    phase_length: int,
    *,
    approach_height: float,
    descend_height: float,
    insert_depth: float,
) -> np.ndarray:
    if phase_id == 0:
        z_offset = approach_height
    elif phase_id == 1:
        progress = min(max(phase_step / max(phase_length - 1, 1), 0.0), 1.0)
        z_offset = approach_height + progress * (descend_height - approach_height)
    elif phase_id == 2:
        progress = min(max(phase_step / max(phase_length - 1, 1), 0.0), 1.0)
        z_offset = descend_height + progress * (-insert_depth - descend_height)
    else:
        z_offset = -insert_depth
    return hole_center + np.array([xy_offset[0], xy_offset[1], z_offset], dtype=float)


def _rotation_about_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _search_offset(
    *,
    phase_id: int,
    phase_step: int,
    phase_length: int,
    radius: float,
    angle: float,
) -> np.ndarray:
    if phase_id not in (1, 2) or radius <= 0.0:
        return np.zeros(2, dtype=float)
    progress = min(max(phase_step / max(phase_length - 1, 1), 0.0), 1.0)
    theta = angle + 2.0 * np.pi * progress
    envelope = np.sin(np.pi * progress)
    return radius * envelope * np.array([np.cos(theta), np.sin(theta)], dtype=float)


def _choose_mode(
    rng: np.random.Generator,
    *,
    dataset_mode: str,
    mode_prob_success_insertion: float,
    mode_prob_rim_probe: float,
    mode_prob_spiral_search: float,
    mode_prob_yaw_probe: float,
    mode_prob_jam_recovery: float,
    mode_prob_perturbed_insertion: float,
) -> int:
    if dataset_mode in ("insertion", "success_insertion"):
        return MODE_SUCCESS_INSERTION
    if dataset_mode == "perturbed_insertion":
        return MODE_PERTURBED_INSERTION
    if dataset_mode == "rim_probe":
        return MODE_RIM_PROBE
    if dataset_mode == "spiral_search":
        return MODE_SPIRAL_SEARCH
    if dataset_mode == "yaw_probe":
        return MODE_YAW_PROBE
    if dataset_mode == "jam_recovery":
        return MODE_JAM_RECOVERY
    if dataset_mode != "mixed":
        raise ValueError(f"Unsupported dataset_mode {dataset_mode!r}.")
    weights = np.array(
        [
            mode_prob_success_insertion,
            mode_prob_rim_probe,
            mode_prob_spiral_search,
            mode_prob_yaw_probe,
            mode_prob_jam_recovery,
            mode_prob_perturbed_insertion,
        ],
        dtype=float,
    )
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("Mode probabilities must be non-negative and have positive sum.")
    weights = weights / np.sum(weights)
    return int(
        rng.choice(
            [
                MODE_SUCCESS_INSERTION,
                MODE_RIM_PROBE,
                MODE_SPIRAL_SEARCH,
                MODE_YAW_PROBE,
                MODE_JAM_RECOVERY,
                MODE_PERTURBED_INSERTION,
            ],
            p=weights,
        )
    )


def _rim_probe_target_position(
    hole_center: np.ndarray,
    xy_offset: np.ndarray,
    step: int,
    *,
    approach_steps: int,
    descend_steps: int,
    total_steps: int,
    approach_height: float,
    rim_probe_z: float,
    probe_radius: float,
    probe_angle: float,
    probe_hold_steps: int,
    release_steps: int,
) -> tuple[np.ndarray, int]:
    if step < approach_steps:
        return hole_center + np.array([xy_offset[0], xy_offset[1], approach_height], dtype=float), 0
    if step < approach_steps + descend_steps:
        phase_step = step - approach_steps
        progress = min(max(phase_step / max(descend_steps - 1, 1), 0.0), 1.0)
        z_offset = approach_height + progress * (rim_probe_z - approach_height)
        return hole_center + np.array([xy_offset[0], xy_offset[1], z_offset], dtype=float), 1

    directions = np.array(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    rotation = np.array(
        [
            [np.cos(probe_angle), -np.sin(probe_angle)],
            [np.sin(probe_angle), np.cos(probe_angle)],
        ],
        dtype=float,
    )
    directions = directions @ rotation.T
    cycle_steps = max(probe_hold_steps + release_steps, 1)
    probe_step = step - approach_steps - descend_steps
    cycle_id = min(probe_step // cycle_steps, len(directions) - 1)
    cycle_phase = probe_step % cycle_steps
    if probe_step >= len(directions) * cycle_steps:
        lateral = xy_offset
        phase_id = 3
    elif cycle_phase < probe_hold_steps:
        lateral = xy_offset + probe_radius * directions[cycle_id]
        phase_id = 2
    else:
        lateral = xy_offset
        phase_id = 3
    if step >= total_steps:
        phase_id = 3
    return hole_center + np.array([lateral[0], lateral[1], rim_probe_z], dtype=float), phase_id


def _spiral_search_target_position(
    hole_center: np.ndarray,
    xy_offset: np.ndarray,
    step: int,
    *,
    approach_steps: int,
    descend_steps: int,
    total_steps: int,
    approach_height: float,
    spiral_z: float,
    spiral_radius: float,
    spiral_angle: float,
    spiral_turns: float,
) -> tuple[np.ndarray, int]:
    if step < approach_steps:
        return hole_center + np.array([xy_offset[0], xy_offset[1], approach_height], dtype=float), 0
    if step < approach_steps + descend_steps:
        phase_step = step - approach_steps
        progress = min(max(phase_step / max(descend_steps - 1, 1), 0.0), 1.0)
        z_offset = approach_height + progress * (spiral_z - approach_height)
        return hole_center + np.array([xy_offset[0], xy_offset[1], z_offset], dtype=float), 1

    spiral_step = step - approach_steps - descend_steps
    spiral_steps = max(total_steps - approach_steps - descend_steps, 1)
    progress = min(max(spiral_step / max(spiral_steps - 1, 1), 0.0), 1.0)
    radius = spiral_radius * (0.35 + 0.65 * progress)
    theta = spiral_angle + 2.0 * np.pi * spiral_turns * progress
    lateral = xy_offset + radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)
    return hole_center + np.array([lateral[0], lateral[1], spiral_z], dtype=float), 2 if step < total_steps else 3


def _yaw_probe_target_position(
    hole_center: np.ndarray,
    xy_offset: np.ndarray,
    step: int,
    *,
    approach_steps: int,
    descend_steps: int,
    total_steps: int,
    approach_height: float,
    yaw_probe_z: float,
    yaw_probe_radius: float,
    yaw_probe_angle: float,
) -> tuple[np.ndarray, int]:
    preload = yaw_probe_radius * np.array([np.cos(yaw_probe_angle), np.sin(yaw_probe_angle)], dtype=float)
    lateral = xy_offset + preload
    if step < approach_steps:
        return hole_center + np.array([xy_offset[0], xy_offset[1], approach_height], dtype=float), 0
    if step < approach_steps + descend_steps:
        phase_step = step - approach_steps
        progress = min(max(phase_step / max(descend_steps - 1, 1), 0.0), 1.0)
        z_offset = approach_height + progress * (yaw_probe_z - approach_height)
        lateral_progress = min(max(progress * 1.5, 0.0), 1.0)
        current_lateral = xy_offset + lateral_progress * preload
        return hole_center + np.array([current_lateral[0], current_lateral[1], z_offset], dtype=float), 1
    return hole_center + np.array([lateral[0], lateral[1], yaw_probe_z], dtype=float), 2 if step < total_steps else 3


def _yaw_probe_offset(
    step: int,
    *,
    approach_steps: int,
    descend_steps: int,
    total_steps: int,
    yaw_amplitude: float,
    yaw_cycles: float,
    yaw_phase: float,
) -> float:
    if step < approach_steps + descend_steps:
        return 0.0
    probe_step = step - approach_steps - descend_steps
    probe_steps = max(total_steps - approach_steps - descend_steps, 1)
    progress = min(max(probe_step / max(probe_steps - 1, 1), 0.0), 1.0)
    envelope = np.sin(np.pi * progress)
    return float(yaw_amplitude * envelope * np.sin(yaw_phase + 2.0 * np.pi * yaw_cycles * progress))


def _jam_recovery_target_position(
    hole_center: np.ndarray,
    xy_offset: np.ndarray,
    step: int,
    *,
    approach_steps: int,
    descend_steps: int,
    total_steps: int,
    approach_height: float,
    jam_z: float,
    jam_radius: float,
    jam_angle: float,
    recovery_radius: float,
    recovery_cycles: float,
) -> tuple[np.ndarray, int]:
    jam_direction = np.array([np.cos(jam_angle), np.sin(jam_angle)], dtype=float)
    jam_lateral = xy_offset + jam_radius * jam_direction
    if step < approach_steps:
        return hole_center + np.array([xy_offset[0], xy_offset[1], approach_height], dtype=float), 0
    if step < approach_steps + descend_steps:
        phase_step = step - approach_steps
        progress = min(max(phase_step / max(descend_steps - 1, 1), 0.0), 1.0)
        z_offset = approach_height + progress * (jam_z - approach_height)
        lateral_progress = min(max(progress * 1.4, 0.0), 1.0)
        lateral = xy_offset + lateral_progress * (jam_lateral - xy_offset)
        return hole_center + np.array([lateral[0], lateral[1], z_offset], dtype=float), 1

    recovery_step = step - approach_steps - descend_steps
    recovery_steps = max(total_steps - approach_steps - descend_steps, 1)
    progress = min(max(recovery_step / max(recovery_steps - 1, 1), 0.0), 1.0)
    theta = jam_angle + 2.0 * np.pi * recovery_cycles * progress
    rotating_probe = recovery_radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)
    preload = (0.45 + 0.35 * np.cos(2.0 * np.pi * recovery_cycles * progress)) * jam_radius * jam_direction
    lateral = xy_offset + preload + rotating_probe
    z_offset = jam_z - 0.003 * progress + 0.002 * np.sin(2.0 * np.pi * recovery_cycles * progress)
    return hole_center + np.array([lateral[0], lateral[1], z_offset], dtype=float), 2 if step < total_steps else 3


def _mode_target_position(
    *,
    mode_id: int,
    hole_center: np.ndarray,
    xy_offset: np.ndarray,
    step: int,
    approach_steps: int,
    descend_steps: int,
    insert_steps: int,
    hold_steps: int,
    search_radius: float,
    search_angle: float,
    rim_probe_z: float,
    probe_radius: float,
    probe_angle: float,
    probe_hold_steps: int,
    release_steps: int,
    rim_probe_approach_steps: int,
    rim_probe_descend_steps: int,
    spiral_z: float,
    spiral_radius: float,
    spiral_angle: float,
    spiral_turns: float,
    spiral_approach_steps: int,
    spiral_descend_steps: int,
    yaw_probe_z: float,
    yaw_probe_radius: float,
    yaw_probe_angle: float,
    yaw_probe_approach_steps: int,
    yaw_probe_descend_steps: int,
    jam_z: float,
    jam_radius: float,
    jam_angle: float,
    recovery_radius: float,
    recovery_cycles: float,
    jam_approach_steps: int,
    jam_descend_steps: int,
) -> tuple[np.ndarray, int]:
    if mode_id in (MODE_SUCCESS_INSERTION, MODE_PERTURBED_INSERTION):
        phase_id, phase_step, phase_length = _phase_for_step(step, approach_steps, descend_steps, insert_steps, hold_steps)
        phase_xy_offset = xy_offset + _search_offset(
            phase_id=phase_id,
            phase_step=phase_step,
            phase_length=phase_length,
            radius=search_radius,
            angle=search_angle,
        )
        return (
            _target_position(
                hole_center,
                phase_xy_offset,
                phase_id,
                phase_step,
                phase_length,
                approach_height=0.18,
                descend_height=0.012,
                insert_depth=0.03,
            ),
            phase_id,
        )
    if mode_id == MODE_RIM_PROBE:
        return _rim_probe_target_position(
            hole_center,
            xy_offset,
            step,
            approach_steps=min(approach_steps, rim_probe_approach_steps),
            descend_steps=min(descend_steps, rim_probe_descend_steps),
            total_steps=approach_steps + descend_steps + insert_steps + hold_steps,
            approach_height=0.18,
            rim_probe_z=rim_probe_z,
            probe_radius=probe_radius,
            probe_angle=probe_angle,
            probe_hold_steps=probe_hold_steps,
            release_steps=release_steps,
        )
    if mode_id == MODE_SPIRAL_SEARCH:
        return _spiral_search_target_position(
            hole_center,
            xy_offset,
            step,
            approach_steps=min(approach_steps, spiral_approach_steps),
            descend_steps=min(descend_steps, spiral_descend_steps),
            total_steps=approach_steps + descend_steps + insert_steps + hold_steps,
            approach_height=0.18,
            spiral_z=spiral_z,
            spiral_radius=spiral_radius,
            spiral_angle=spiral_angle,
            spiral_turns=spiral_turns,
        )
    if mode_id == MODE_YAW_PROBE:
        return _yaw_probe_target_position(
            hole_center,
            xy_offset,
            step,
            approach_steps=min(approach_steps, yaw_probe_approach_steps),
            descend_steps=min(descend_steps, yaw_probe_descend_steps),
            total_steps=approach_steps + descend_steps + insert_steps + hold_steps,
            approach_height=0.18,
            yaw_probe_z=yaw_probe_z,
            yaw_probe_radius=yaw_probe_radius,
            yaw_probe_angle=yaw_probe_angle,
        )
    if mode_id == MODE_JAM_RECOVERY:
        return _jam_recovery_target_position(
            hole_center,
            xy_offset,
            step,
            approach_steps=min(approach_steps, jam_approach_steps),
            descend_steps=min(descend_steps, jam_descend_steps),
            total_steps=approach_steps + descend_steps + insert_steps + hold_steps,
            approach_height=0.18,
            jam_z=jam_z,
            jam_radius=jam_radius,
            jam_angle=jam_angle,
            recovery_radius=recovery_radius,
            recovery_cycles=recovery_cycles,
        )
    raise ValueError(f"Unsupported mode_id {mode_id}.")


def _scene_with_clearance_delta(scene_config: dict, clearance_delta: float) -> dict:
    config = copy.deepcopy(scene_config)
    hole = config["hole"]
    hole["outer_radius"] = float(hole["outer_radius"]) + clearance_delta
    hole["inner_radius"] = float(hole["inner_radius"]) + clearance_delta
    hole["completion_lateral_tolerance"] = max(
        0.0005,
        float(hole.get("completion_lateral_tolerance", 0.002)) + clearance_delta,
    )
    if hole["inner_radius"] <= 0.0 or hole["outer_radius"] <= hole["inner_radius"]:
        raise ValueError(
            "Invalid hole clearance delta: "
            f"delta={clearance_delta:g}, inner_radius={hole['inner_radius']:g}, outer_radius={hole['outer_radius']:g}."
        )
    return config


def _clip_torque(model: mujoco.MjModel, arm_ids, torque: np.ndarray) -> np.ndarray:
    clipped = np.asarray(torque, dtype=float).copy()
    for idx, actuator_id in enumerate(arm_ids.actuator_ids):
        low, high = model.actuator_ctrlrange[actuator_id]
        clipped[idx] = np.clip(clipped[idx], low, high)
    return clipped


def _state_row(obs: np.ndarray, data: mujoco.MjData, arm_ids) -> np.ndarray:
    return np.concatenate([arm_qpos(data, arm_ids), arm_qvel(data, arm_ids), obs[:40]])


def _action_row(u_ref: np.ndarray, torque: np.ndarray) -> np.ndarray:
    return np.concatenate([u_ref, torque])


def collect_dataset(
    *,
    output_path: Path,
    episodes: int,
    seed: int,
    sample_stride: int,
    config_path: Path,
    gain_config: Path,
    gain_profile: str,
    offset_radius_min: float,
    offset_radius_max: float,
    yaw_error_max_deg: float,
    search_radius_max: float,
    hole_clearance_delta_min: float,
    hole_clearance_delta_max: float,
    approach_steps: int,
    descend_steps: int,
    insert_steps: int,
    hold_steps: int,
    label_neighbors: int,
    dataset_mode: str = "success_insertion",
    mode_prob_success_insertion: float = 0.4,
    mode_prob_rim_probe: float = 0.6,
    mode_prob_spiral_search: float = 0.0,
    mode_prob_yaw_probe: float = 0.0,
    mode_prob_jam_recovery: float = 0.0,
    mode_prob_perturbed_insertion: float = 0.0,
    success_offset_radius_max: float = 0.0005,
    success_yaw_error_max_deg: float = 0.25,
    success_search_radius_max: float = 0.0,
    probe_radius_min: float = 0.003,
    probe_radius_max: float = 0.008,
    probe_hold_steps: int = 120,
    release_steps: int = 20,
    rim_probe_z_min: float = 0.0,
    rim_probe_z_max: float = 0.006,
    rim_probe_approach_steps: int = 100,
    rim_probe_descend_steps: int = 300,
    spiral_radius_min: float = 0.003,
    spiral_radius_max: float = 0.008,
    spiral_z_min: float = 0.0,
    spiral_z_max: float = 0.006,
    spiral_turns_min: float = 3.0,
    spiral_turns_max: float = 6.0,
    spiral_approach_steps: int = 100,
    spiral_descend_steps: int = 300,
    yaw_probe_radius_min: float = 0.003,
    yaw_probe_radius_max: float = 0.008,
    yaw_probe_z_min: float = 0.0,
    yaw_probe_z_max: float = 0.006,
    yaw_probe_amplitude_deg_min: float = 2.0,
    yaw_probe_amplitude_deg_max: float = 6.0,
    yaw_probe_cycles_min: float = 4.0,
    yaw_probe_cycles_max: float = 8.0,
    yaw_probe_approach_steps: int = 100,
    yaw_probe_descend_steps: int = 300,
    jam_radius_min: float = 0.004,
    jam_radius_max: float = 0.010,
    jam_z_min: float = -0.002,
    jam_z_max: float = 0.004,
    recovery_radius_min: float = 0.002,
    recovery_radius_max: float = 0.006,
    recovery_cycles_min: float = 2.0,
    recovery_cycles_max: float = 5.0,
    jam_approach_steps: int = 100,
    jam_descend_steps: int = 300,
) -> Path:
    rng = np.random.default_rng(seed)
    scene_path = render_config_file(config_path)
    scene_config = load_scene_config(config_path)
    _, gains = load_task_space_impedance_gains(gain_config, gain_profile)

    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    contact_rows: list[np.ndarray] = []
    reward_rows: list[float] = []
    time_rows: list[float] = []
    episode_rows: list[int] = []
    task_state_rows: list[np.ndarray] = []
    contact_force_rows: list[np.ndarray] = []
    clearance_delta_rows: list[float] = []
    mode_rows: list[int] = []
    episode_mode_ids: list[int] = []
    episode_final_depth: list[float] = []
    episode_final_lateral: list[float] = []
    episode_success: list[bool] = []

    total_steps = approach_steps + descend_steps + insert_steps + hold_steps
    for episode_id in range(episodes):
        clearance_delta = rng.uniform(hole_clearance_delta_min, hole_clearance_delta_max)
        mode_id = _choose_mode(
            rng,
            dataset_mode=dataset_mode,
            mode_prob_success_insertion=mode_prob_success_insertion,
            mode_prob_rim_probe=mode_prob_rim_probe,
            mode_prob_spiral_search=mode_prob_spiral_search,
            mode_prob_yaw_probe=mode_prob_yaw_probe,
            mode_prob_jam_recovery=mode_prob_jam_recovery,
            mode_prob_perturbed_insertion=mode_prob_perturbed_insertion,
        )
        episode_mode_ids.append(mode_id)
        episode_scene_config = _scene_with_clearance_delta(scene_config, clearance_delta)
        episode_scene_path = (
            render_runtime_config(episode_scene_config, prefix="runtime_learning_")
            if clearance_delta != 0.0
            else scene_path
        )
        try:
            model = load_model(episode_scene_path)
        finally:
            cleanup_runtime_scene(episode_scene_path)
        data = mujoco.MjData(model)
        reset_from_config(model, data, episode_scene_config)

        geometry = geometry_from_config(episode_scene_config)
        task_ids = peg_hole_ids(model, segments=geometry.segments)
        arm_ids = panda_arm_ids(model)
        nullspace_target_qpos = arm_qpos(data, arm_ids)
        hole_center = hole_center_position(data, task_ids)
        target_rotation = site_rotation(data, model.site("peg_tip").id)
        active_yaw_error_max = success_yaw_error_max_deg if mode_id == MODE_SUCCESS_INSERTION else yaw_error_max_deg
        yaw_error = rng.uniform(-np.deg2rad(active_yaw_error_max), np.deg2rad(active_yaw_error_max))
        target_rotation = _rotation_about_z(yaw_error) @ target_rotation
        if mode_id == MODE_SUCCESS_INSERTION:
            offset_radius = rng.uniform(0.0, success_offset_radius_max)
        else:
            offset_radius = rng.uniform(offset_radius_min, offset_radius_max)
        offset_angle = rng.uniform(-np.pi, np.pi)
        xy_offset = offset_radius * np.array([np.cos(offset_angle), np.sin(offset_angle)], dtype=float)
        active_search_radius_max = success_search_radius_max if mode_id == MODE_SUCCESS_INSERTION else search_radius_max
        search_radius = rng.uniform(0.0, active_search_radius_max)
        search_angle = rng.uniform(-np.pi, np.pi)
        probe_radius = rng.uniform(probe_radius_min, probe_radius_max)
        probe_angle = rng.uniform(-np.pi, np.pi)
        rim_probe_z = rng.uniform(rim_probe_z_min, rim_probe_z_max)
        spiral_radius = rng.uniform(spiral_radius_min, spiral_radius_max)
        spiral_angle = rng.uniform(-np.pi, np.pi)
        spiral_z = rng.uniform(spiral_z_min, spiral_z_max)
        spiral_turns = rng.uniform(spiral_turns_min, spiral_turns_max)
        yaw_probe_radius = rng.uniform(yaw_probe_radius_min, yaw_probe_radius_max)
        yaw_probe_angle = rng.uniform(-np.pi, np.pi)
        yaw_probe_z = rng.uniform(yaw_probe_z_min, yaw_probe_z_max)
        yaw_probe_amplitude = np.deg2rad(rng.uniform(yaw_probe_amplitude_deg_min, yaw_probe_amplitude_deg_max))
        yaw_probe_cycles = rng.uniform(yaw_probe_cycles_min, yaw_probe_cycles_max)
        yaw_probe_phase = rng.uniform(-np.pi, np.pi)
        jam_radius = rng.uniform(jam_radius_min, jam_radius_max)
        jam_angle = rng.uniform(-np.pi, np.pi)
        jam_z = rng.uniform(jam_z_min, jam_z_max)
        recovery_radius = rng.uniform(recovery_radius_min, recovery_radius_max)
        recovery_cycles = rng.uniform(recovery_cycles_min, recovery_cycles_max)
        last_depth = 0.0

        for step in range(total_steps + 1):
            target_position, phase_id = _mode_target_position(
                mode_id=mode_id,
                hole_center=hole_center,
                xy_offset=xy_offset,
                step=step,
                approach_steps=approach_steps,
                descend_steps=descend_steps,
                insert_steps=insert_steps,
                hold_steps=hold_steps,
                search_radius=search_radius,
                search_angle=search_angle,
                rim_probe_z=rim_probe_z,
                probe_radius=probe_radius,
                probe_angle=probe_angle,
                probe_hold_steps=probe_hold_steps,
                release_steps=release_steps,
                rim_probe_approach_steps=rim_probe_approach_steps,
                rim_probe_descend_steps=rim_probe_descend_steps,
                spiral_z=spiral_z,
                spiral_radius=spiral_radius,
                spiral_angle=spiral_angle,
                spiral_turns=spiral_turns,
                spiral_approach_steps=spiral_approach_steps,
                spiral_descend_steps=spiral_descend_steps,
                yaw_probe_z=yaw_probe_z,
                yaw_probe_radius=yaw_probe_radius,
                yaw_probe_angle=yaw_probe_angle,
                yaw_probe_approach_steps=yaw_probe_approach_steps,
                yaw_probe_descend_steps=yaw_probe_descend_steps,
                jam_z=jam_z,
                jam_radius=jam_radius,
                jam_angle=jam_angle,
                recovery_radius=recovery_radius,
                recovery_cycles=recovery_cycles,
                jam_approach_steps=jam_approach_steps,
                jam_descend_steps=jam_descend_steps,
            )
            dynamic_target_rotation = target_rotation
            if mode_id == MODE_YAW_PROBE:
                yaw_offset = _yaw_probe_offset(
                    step,
                    approach_steps=min(approach_steps, yaw_probe_approach_steps),
                    descend_steps=min(descend_steps, yaw_probe_descend_steps),
                    total_steps=total_steps,
                    yaw_amplitude=yaw_probe_amplitude,
                    yaw_cycles=yaw_probe_cycles,
                    yaw_phase=yaw_probe_phase,
                )
                dynamic_target_rotation = _rotation_about_z(yaw_offset) @ target_rotation
            command = task_space_impedance_torque(
                model,
                data,
                site_name="peg_tip",
                target_position=target_position,
                target_rotation=dynamic_target_rotation,
                arm_ids=arm_ids,
                gains=gains,
                nullspace_target_qpos=nullspace_target_qpos,
                clip_to_ctrlrange=False,
            )
            torque = _clip_torque(model, arm_ids, command.torque)
            obs, u_ref, info = collect_step(
                model,
                data,
                arm_ids=arm_ids,
                task_ids=task_ids,
                target_position=target_position,
                target_rotation=dynamic_target_rotation,
                phase_id=phase_id,
            )
            query = ContactQuery(model=model, data=data, task_ids=task_ids)
            contact = extract_contact_state(query)
            depth = float(info[0])
            normal_force = float(contact.normal_force)
            reward_proxy = (depth - last_depth) - 0.0005 * normal_force
            last_depth = depth

            if step % sample_stride == 0:
                state_rows.append(_state_row(obs, data, arm_ids))
                action_rows.append(_action_row(u_ref, torque))
                contact_rows.append(contact_state_vector(contact))
                reward_rows.append(reward_proxy)
                time_rows.append(float(data.time))
                episode_rows.append(episode_id)
                task_state_rows.append(peg_hole_task_state(data, task_ids, hole_clearance_delta=clearance_delta))
                contact_force_rows.append(extract_net_peg_hole_contact_force_world(query))
                clearance_delta_rows.append(clearance_delta)
                mode_rows.append(mode_id)

            set_arm_torque_ctrl(model, data, arm_ids, torque)
            mujoco.mj_step(model, data)
        final_task_state = peg_hole_task_state(data, task_ids, hole_clearance_delta=clearance_delta)
        final_lateral = float(np.linalg.norm(final_task_state[:2]))
        final_depth = float(final_task_state[2])
        episode_final_depth.append(final_depth)
        episode_final_lateral.append(final_lateral)
        episode_success.append(bool(final_depth >= 0.95 * 0.03 and final_lateral <= geometry.radial_clearance))

    task_state = np.vstack(task_state_rows)
    contact_force_world = np.vstack(contact_force_rows)
    stiffness_matrix_target, stiffness_cholesky_target, label_diagnostics = build_stiffness_labels_with_diagnostics(
        task_state,
        contact_force_world,
        config=StiffnessLabelConfig(neighbors=label_neighbors),
    )
    arrays = {
        "state": np.vstack(state_rows),
        "action": np.vstack(action_rows),
        "contact_state": np.vstack(contact_rows),
        "reward_proxy": np.asarray(reward_rows, dtype=float),
        "timestamp": np.asarray(time_rows, dtype=float),
        "episode_id": np.asarray(episode_rows, dtype=np.int32),
        "task_state": task_state,
        "task_state_6d": task_state,
        "privileged_task_state_6d": task_state,
        "hole_clearance_delta": np.asarray(clearance_delta_rows, dtype=float),
        "mode_id": np.asarray(mode_rows, dtype=np.int32),
        "contact_force_world": contact_force_world,
        "stiffness_matrix_target": stiffness_matrix_target,
        "stiffness_cholesky_target": stiffness_cholesky_target,
        **label_diagnostics,
    }
    metadata = {
        "schema_version": "learning",
        "num_episodes": episodes,
        "num_samples": int(arrays["state"].shape[0]),
        "sample_stride": sample_stride,
        "seed": seed,
        "profile": gain_profile,
        "offset_radius_min": offset_radius_min,
        "offset_radius_max": offset_radius_max,
        "yaw_error_max_deg": yaw_error_max_deg,
        "search_radius_max": search_radius_max,
        "hole_clearance_delta_min": hole_clearance_delta_min,
        "hole_clearance_delta_max": hole_clearance_delta_max,
        "label_neighbors": label_neighbors,
        "dataset_mode": dataset_mode,
        "dataset_modes": {str(key): value for key, value in MODE_NAMES.items()},
        "mode_probabilities": {
            "success_insertion": mode_prob_success_insertion,
            "rim_probe": mode_prob_rim_probe,
            "spiral_search": mode_prob_spiral_search,
            "yaw_probe": mode_prob_yaw_probe,
            "jam_recovery": mode_prob_jam_recovery,
            "perturbed_insertion": mode_prob_perturbed_insertion,
        },
        "episode_mode_counts": {
            MODE_NAMES[key]: int(np.sum(np.asarray(episode_mode_ids, dtype=np.int32) == key))
            for key in sorted(MODE_NAMES)
        },
        "episode_success_counts": {
            MODE_NAMES[key]: int(
                np.sum(
                    (np.asarray(episode_mode_ids, dtype=np.int32) == key)
                    & np.asarray(episode_success, dtype=bool)
                )
            )
            for key in sorted(MODE_NAMES)
        },
        "episode_success_ratio_by_mode": {
            MODE_NAMES[key]: (
                float(
                    np.mean(
                        np.asarray(episode_success, dtype=bool)[
                            np.asarray(episode_mode_ids, dtype=np.int32) == key
                        ]
                    )
                )
                if np.any(np.asarray(episode_mode_ids, dtype=np.int32) == key)
                else 0.0
            )
            for key in sorted(MODE_NAMES)
        },
        "episode_final_depth_mean_by_mode": {
            MODE_NAMES[key]: (
                float(
                    np.mean(
                        np.asarray(episode_final_depth, dtype=float)[
                            np.asarray(episode_mode_ids, dtype=np.int32) == key
                        ]
                    )
                )
                if np.any(np.asarray(episode_mode_ids, dtype=np.int32) == key)
                else 0.0
            )
            for key in sorted(MODE_NAMES)
        },
        "episode_final_lateral_mean_by_mode": {
            MODE_NAMES[key]: (
                float(
                    np.mean(
                        np.asarray(episode_final_lateral, dtype=float)[
                            np.asarray(episode_mode_ids, dtype=np.int32) == key
                        ]
                    )
                )
                if np.any(np.asarray(episode_mode_ids, dtype=np.int32) == key)
                else 0.0
            )
            for key in sorted(MODE_NAMES)
        },
        "success_offset_radius_max": success_offset_radius_max,
        "success_yaw_error_max_deg": success_yaw_error_max_deg,
        "success_search_radius_max": success_search_radius_max,
        "probe_radius_min": probe_radius_min,
        "probe_radius_max": probe_radius_max,
        "probe_hold_steps": probe_hold_steps,
        "release_steps": release_steps,
        "rim_probe_z_min": rim_probe_z_min,
        "rim_probe_z_max": rim_probe_z_max,
        "rim_probe_approach_steps": rim_probe_approach_steps,
        "rim_probe_descend_steps": rim_probe_descend_steps,
        "spiral_radius_min": spiral_radius_min,
        "spiral_radius_max": spiral_radius_max,
        "spiral_z_min": spiral_z_min,
        "spiral_z_max": spiral_z_max,
        "spiral_turns_min": spiral_turns_min,
        "spiral_turns_max": spiral_turns_max,
        "spiral_approach_steps": spiral_approach_steps,
        "spiral_descend_steps": spiral_descend_steps,
        "yaw_probe_radius_min": yaw_probe_radius_min,
        "yaw_probe_radius_max": yaw_probe_radius_max,
        "yaw_probe_z_min": yaw_probe_z_min,
        "yaw_probe_z_max": yaw_probe_z_max,
        "yaw_probe_amplitude_deg_min": yaw_probe_amplitude_deg_min,
        "yaw_probe_amplitude_deg_max": yaw_probe_amplitude_deg_max,
        "yaw_probe_cycles_min": yaw_probe_cycles_min,
        "yaw_probe_cycles_max": yaw_probe_cycles_max,
        "yaw_probe_approach_steps": yaw_probe_approach_steps,
        "yaw_probe_descend_steps": yaw_probe_descend_steps,
        "jam_radius_min": jam_radius_min,
        "jam_radius_max": jam_radius_max,
        "jam_z_min": jam_z_min,
        "jam_z_max": jam_z_max,
        "recovery_radius_min": recovery_radius_min,
        "recovery_radius_max": recovery_radius_max,
        "recovery_cycles_min": recovery_cycles_min,
        "recovery_cycles_max": recovery_cycles_max,
        "jam_approach_steps": jam_approach_steps,
        "jam_descend_steps": jam_descend_steps,
        "complement_regularization": StiffnessLabelConfig().complement_regularization,
        "state_dim": int(arrays["state"].shape[1]),
        "action_dim": int(arrays["action"].shape[1]),
        "contact_dim": int(arrays["contact_state"].shape[1]),
        "task_state_dim": int(arrays["task_state"].shape[1]),
        "target": "normalized_spd_stiffness_cholesky",
    }
    validate_learning_dataset(arrays, metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays, metadata=json.dumps(metadata, sort_keys=True))
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect non-vision Stiffness Copilot learning dataset.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=20)
    parser.add_argument("--config", type=Path, default=DEFAULT_TORQUE_CONFIG)
    parser.add_argument("--gain-config", type=Path, default=DEFAULT_GAIN_CONFIG)
    parser.add_argument("--gain-profile", type=str, default="stiff")
    parser.add_argument("--offset-radius-min", type=float, default=0.001)
    parser.add_argument("--offset-radius-max", type=float, default=0.006)
    parser.add_argument("--yaw-error-max-deg", type=float, default=3.0)
    parser.add_argument("--search-radius-max", type=float, default=0.0015)
    parser.add_argument("--hole-clearance-delta-min", type=float, default=0.0)
    parser.add_argument("--hole-clearance-delta-max", type=float, default=0.0)
    parser.add_argument("--approach-steps", type=int, default=600)
    parser.add_argument("--descend-steps", type=int, default=1200)
    parser.add_argument("--insert-steps", type=int, default=1800)
    parser.add_argument("--hold-steps", type=int, default=400)
    parser.add_argument("--label-neighbors", type=int, default=32)
    parser.add_argument(
        "--dataset-mode",
        choices=("insertion", "success_insertion", "perturbed_insertion", "rim_probe", "spiral_search", "yaw_probe", "jam_recovery", "mixed"),
        default="success_insertion",
    )
    parser.add_argument("--mode-prob-success-insertion", type=float, default=None)
    parser.add_argument("--mode-prob-insertion", type=float, default=None)
    parser.add_argument("--mode-prob-rim-probe", type=float, default=0.6)
    parser.add_argument("--mode-prob-spiral-search", type=float, default=0.0)
    parser.add_argument("--mode-prob-yaw-probe", type=float, default=0.0)
    parser.add_argument("--mode-prob-jam-recovery", type=float, default=0.0)
    parser.add_argument("--mode-prob-perturbed-insertion", type=float, default=0.0)
    parser.add_argument("--success-offset-radius-max", type=float, default=0.0005)
    parser.add_argument("--success-yaw-error-max-deg", type=float, default=0.25)
    parser.add_argument("--success-search-radius-max", type=float, default=0.0)
    parser.add_argument("--probe-radius-min", type=float, default=0.003)
    parser.add_argument("--probe-radius-max", type=float, default=0.008)
    parser.add_argument("--probe-hold-steps", type=int, default=120)
    parser.add_argument("--release-steps", type=int, default=20)
    parser.add_argument("--rim-probe-z-min", type=float, default=0.0)
    parser.add_argument("--rim-probe-z-max", type=float, default=0.006)
    parser.add_argument("--rim-probe-approach-steps", type=int, default=100)
    parser.add_argument("--rim-probe-descend-steps", type=int, default=300)
    parser.add_argument("--spiral-radius-min", type=float, default=0.003)
    parser.add_argument("--spiral-radius-max", type=float, default=0.008)
    parser.add_argument("--spiral-z-min", type=float, default=0.0)
    parser.add_argument("--spiral-z-max", type=float, default=0.006)
    parser.add_argument("--spiral-turns-min", type=float, default=3.0)
    parser.add_argument("--spiral-turns-max", type=float, default=6.0)
    parser.add_argument("--spiral-approach-steps", type=int, default=100)
    parser.add_argument("--spiral-descend-steps", type=int, default=300)
    parser.add_argument("--yaw-probe-radius-min", type=float, default=0.003)
    parser.add_argument("--yaw-probe-radius-max", type=float, default=0.008)
    parser.add_argument("--yaw-probe-z-min", type=float, default=0.0)
    parser.add_argument("--yaw-probe-z-max", type=float, default=0.006)
    parser.add_argument("--yaw-probe-amplitude-deg-min", type=float, default=2.0)
    parser.add_argument("--yaw-probe-amplitude-deg-max", type=float, default=6.0)
    parser.add_argument("--yaw-probe-cycles-min", type=float, default=4.0)
    parser.add_argument("--yaw-probe-cycles-max", type=float, default=8.0)
    parser.add_argument("--yaw-probe-approach-steps", type=int, default=100)
    parser.add_argument("--yaw-probe-descend-steps", type=int, default=300)
    parser.add_argument("--jam-radius-min", type=float, default=0.004)
    parser.add_argument("--jam-radius-max", type=float, default=0.010)
    parser.add_argument("--jam-z-min", type=float, default=-0.002)
    parser.add_argument("--jam-z-max", type=float, default=0.004)
    parser.add_argument("--recovery-radius-min", type=float, default=0.002)
    parser.add_argument("--recovery-radius-max", type=float, default=0.006)
    parser.add_argument("--recovery-cycles-min", type=float, default=2.0)
    parser.add_argument("--recovery-cycles-max", type=float, default=5.0)
    parser.add_argument("--jam-approach-steps", type=int, default=100)
    parser.add_argument("--jam-descend-steps", type=int, default=300)
    args = parser.parse_args(argv)
    mode_prob_success_insertion = (
        args.mode_prob_success_insertion
        if args.mode_prob_success_insertion is not None
        else 0.4
        if args.mode_prob_insertion is None
        else args.mode_prob_insertion
    )

    output = args.output or DEFAULT_OUTPUT_ROOT / f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    output_path = collect_dataset(
        output_path=output,
        episodes=args.episodes,
        seed=args.seed,
        sample_stride=args.sample_stride,
        config_path=args.config,
        gain_config=args.gain_config,
        gain_profile=args.gain_profile,
        offset_radius_min=args.offset_radius_min,
        offset_radius_max=args.offset_radius_max,
        yaw_error_max_deg=args.yaw_error_max_deg,
        search_radius_max=args.search_radius_max,
        hole_clearance_delta_min=args.hole_clearance_delta_min,
        hole_clearance_delta_max=args.hole_clearance_delta_max,
        approach_steps=args.approach_steps,
        descend_steps=args.descend_steps,
        insert_steps=args.insert_steps,
        hold_steps=args.hold_steps,
        label_neighbors=args.label_neighbors,
        dataset_mode=args.dataset_mode,
        mode_prob_success_insertion=mode_prob_success_insertion,
        mode_prob_rim_probe=args.mode_prob_rim_probe,
        mode_prob_spiral_search=args.mode_prob_spiral_search,
        mode_prob_yaw_probe=args.mode_prob_yaw_probe,
        mode_prob_jam_recovery=args.mode_prob_jam_recovery,
        mode_prob_perturbed_insertion=args.mode_prob_perturbed_insertion,
        success_offset_radius_max=args.success_offset_radius_max,
        success_yaw_error_max_deg=args.success_yaw_error_max_deg,
        success_search_radius_max=args.success_search_radius_max,
        probe_radius_min=args.probe_radius_min,
        probe_radius_max=args.probe_radius_max,
        probe_hold_steps=args.probe_hold_steps,
        release_steps=args.release_steps,
        rim_probe_z_min=args.rim_probe_z_min,
        rim_probe_z_max=args.rim_probe_z_max,
        rim_probe_approach_steps=args.rim_probe_approach_steps,
        rim_probe_descend_steps=args.rim_probe_descend_steps,
        spiral_radius_min=args.spiral_radius_min,
        spiral_radius_max=args.spiral_radius_max,
        spiral_z_min=args.spiral_z_min,
        spiral_z_max=args.spiral_z_max,
        spiral_turns_min=args.spiral_turns_min,
        spiral_turns_max=args.spiral_turns_max,
        spiral_approach_steps=args.spiral_approach_steps,
        spiral_descend_steps=args.spiral_descend_steps,
        yaw_probe_radius_min=args.yaw_probe_radius_min,
        yaw_probe_radius_max=args.yaw_probe_radius_max,
        yaw_probe_z_min=args.yaw_probe_z_min,
        yaw_probe_z_max=args.yaw_probe_z_max,
        yaw_probe_amplitude_deg_min=args.yaw_probe_amplitude_deg_min,
        yaw_probe_amplitude_deg_max=args.yaw_probe_amplitude_deg_max,
        yaw_probe_cycles_min=args.yaw_probe_cycles_min,
        yaw_probe_cycles_max=args.yaw_probe_cycles_max,
        yaw_probe_approach_steps=args.yaw_probe_approach_steps,
        yaw_probe_descend_steps=args.yaw_probe_descend_steps,
        jam_radius_min=args.jam_radius_min,
        jam_radius_max=args.jam_radius_max,
        jam_z_min=args.jam_z_min,
        jam_z_max=args.jam_z_max,
        recovery_radius_min=args.recovery_radius_min,
        recovery_radius_max=args.recovery_radius_max,
        recovery_cycles_min=args.recovery_cycles_min,
        recovery_cycles_max=args.recovery_cycles_max,
        jam_approach_steps=args.jam_approach_steps,
        jam_descend_steps=args.jam_descend_steps,
    )
    with np.load(output_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
    print(f"wrote: {output_path}")
    print(
        "episodes={episodes} samples={samples} task_state_dim={task_state_dim} target={target}".format(
            episodes=metadata["num_episodes"],
            samples=metadata["num_samples"],
            task_state_dim=metadata["task_state_dim"],
            target=metadata["target"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
