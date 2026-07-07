from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import ContactQuery, extract_contact_state
from stiffness_copilot_mujoco.controllers.gains import load_baseline_gains
from stiffness_copilot_mujoco.controllers.impedance import (
    TRACK_A_BASELINE_CONTROLLER_PROFILE,
    TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE,
    TaskSpaceImpedanceGains,
    load_task_space_impedance_gains,
    task_space_impedance_torque,
)
from stiffness_copilot_mujoco.controllers.stiffness_command_smoothing import (
    StiffnessCommandSmoother,
    StiffnessCommandSmoothingConfig,
    resolve_deployment_stiffness_smoothing_config,
)
from stiffness_copilot_mujoco.episodes.episode_spec import (
    EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
    EPISODE_TRAJECTORY_SOURCE_OPEN_LOOP_FAMILY,
    EpisodeSpec,
)
from stiffness_copilot_mujoco.episodes.teleop_proxy import (
    TELEOP_MODE_POSITION_ORIENTATION,
    validate_teleop_mode,
    nominal_hole_rotation,
)
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.evaluation.force_metrics import ForceMetricThresholds
from stiffness_copilot_mujoco.metrics.task_metrics import (
    geometry_from_config,
    hole_center_position,
    hole_insertion_axis_world,
    insertion_depth,
    lateral_error,
    load_scene_config,
    peg_axis_alignment,
)
from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl
from stiffness_copilot_mujoco.pose_math import site_rotation
from stiffness_copilot_mujoco.rollout_observation import reset_from_config
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
from stiffness_copilot_mujoco.sim.scene import (
    ROOT,
    cleanup_runtime_scene,
    render_config_file,
    render_runtime_config,
    validate_canonical_eye_in_hand_camera_config,
)


BaselineName = Literal["base", "low", "high", "mid_high_baseline", "track_a_c600"]
StepCallback = Callable[[int, str, mujoco.MjModel, mujoco.MjData, dict[str, float | bool | np.ndarray]], None]


DEFAULT_TORQUE_CONFIG = ROOT / "configs" / "scenes" / "panda_peg_in_hole_torque.yaml"
DEFAULT_GAIN_CONFIG = ROOT / "configs" / "controllers" / "fixed_impedance.yaml"
DEFAULT_LOW_FORCE_SUCCESS_THRESHOLD = ForceMetricThresholds().low_force_success_threshold

@dataclass(frozen=True)
class RolloutConfig:
    config_path: Path = DEFAULT_TORQUE_CONFIG
    gain_config_path: Path = DEFAULT_GAIN_CONFIG
    approach_hold_steps: int = 600
    descend_steps: int = 1200
    insert_steps: int = 1800
    final_hold_steps: int = 400
    approach_height: float = 0.18
    descend_height: float = 0.012
    insert_depth: float = 0.03
    site_name: str = "peg_tip"
    print_every: int = 0
    max_steps: int | None = None

    @property
    def total_steps(self) -> int:
        scheduled = self.approach_hold_steps + self.descend_steps + self.insert_steps + self.final_hold_steps
        return scheduled if self.max_steps is None else min(scheduled, self.max_steps)


@dataclass(frozen=True)
class RolloutPerturbation:
    hole_xy_offset: tuple[float, float] = (0.0, 0.0)
    hole_yaw_offset: float = 0.0
    teleop_noise_xy_amplitude: float = 0.0
    teleop_noise_cycles: float = 1.0
    teleop_noise_phase_x: float = 0.0
    teleop_noise_phase_y: float = 0.0
    clearance_delta: float = 0.0
    friction_scale: float = 1.0
    peg_tilt_x: float = 0.0
    peg_tilt_y: float = 0.0

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "hole_xy_offset": [float(self.hole_xy_offset[0]), float(self.hole_xy_offset[1])],
            "hole_yaw_offset": float(self.hole_yaw_offset),
            "teleop_noise_xy_amplitude": float(self.teleop_noise_xy_amplitude),
            "teleop_noise_cycles": float(self.teleop_noise_cycles),
            "teleop_noise_phase_x": float(self.teleop_noise_phase_x),
            "teleop_noise_phase_y": float(self.teleop_noise_phase_y),
            "clearance_delta": float(self.clearance_delta),
            "friction_scale": float(self.friction_scale),
            "peg_tilt_x": float(self.peg_tilt_x),
            "peg_tilt_y": float(self.peg_tilt_y),
        }


@dataclass(frozen=True)
class EpisodeSummary:
    baseline: str
    profile: str
    seed: int
    episode_id: int
    xy_offset: tuple[float, float]
    steps: int
    final_depth: float
    final_lateral_error: float
    final_axis_alignment: float
    final_orientation_error: float
    contact_detected: bool
    hole_contact_detected: bool
    contact_onset_step: int
    max_normal_force: float
    mean_normal_force_contact: float
    max_tangential_force: float
    max_penetration_depth: float
    max_abs_commanded_torque: float
    torque_saturation_count: int
    depth_progress_after_contact: float
    depth_reached: bool
    low_force_success: bool
    completion_like: bool
    smoothing_enabled: bool
    smoothing_method: str
    smoothing_alpha: float
    policy_update_period_steps: int
    hold_between_updates: bool
    smoothing_target_kind: str
    stiffness_before_smoothing_summary: dict[str, object]
    stiffness_after_smoothing_summary: dict[str, object]
    perturbation: dict[str, float | list[float]] | None
    teleop_mode: str = "position_only"
    target_rotation: np.ndarray | None = None
    actual_ee_rotation: np.ndarray | None = None
    orientation_error_vector: np.ndarray | None = None
    axis_alignment_error: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "profile": self.profile,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "xy_offset": list(self.xy_offset),
            "steps": self.steps,
            "final_depth": self.final_depth,
            "final_lateral_error": self.final_lateral_error,
            "final_axis_alignment": self.final_axis_alignment,
            "final_orientation_error": self.final_orientation_error,
            "contact_detected": self.contact_detected,
            "hole_contact_detected": self.hole_contact_detected,
            "contact_onset_step": self.contact_onset_step,
            "max_normal_force": self.max_normal_force,
            "mean_normal_force_contact": self.mean_normal_force_contact,
            "max_tangential_force": self.max_tangential_force,
            "max_penetration_depth": self.max_penetration_depth,
            "max_abs_commanded_torque": self.max_abs_commanded_torque,
            "torque_saturation_count": self.torque_saturation_count,
            "depth_progress_after_contact": self.depth_progress_after_contact,
            "depth_reached": self.depth_reached,
            "low_force_success": self.low_force_success,
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_method": self.smoothing_method,
            "smoothing_alpha": self.smoothing_alpha,
            "policy_update_period_steps": self.policy_update_period_steps,
            "hold_between_updates": self.hold_between_updates,
            "smoothing_target_kind": self.smoothing_target_kind,
            "stiffness_before_smoothing_summary": self.stiffness_before_smoothing_summary,
            "stiffness_after_smoothing_summary": self.stiffness_after_smoothing_summary,
            "perturbation": self.perturbation,
            "teleop_mode": self.teleop_mode,
            "target_rotation": None if self.target_rotation is None else self.target_rotation.tolist(),
            "actual_ee_rotation": None if self.actual_ee_rotation is None else self.actual_ee_rotation.tolist(),
            "orientation_error_vector": None
            if self.orientation_error_vector is None
            else self.orientation_error_vector.tolist(),
            "axis_alignment_error": self.axis_alignment_error,
        }


def phase_for_step(step: int, config: RolloutConfig) -> tuple[str, int, int]:
    boundaries = (
        ("approach_hold", config.approach_hold_steps),
        ("descend", config.descend_steps),
        ("insert", config.insert_steps),
        ("final_hold", config.final_hold_steps),
    )
    cursor = 0
    for phase, length in boundaries:
        if step < cursor + length:
            return phase, step - cursor, length
        cursor += length
    return "done", max(step - cursor, 0), 1


def target_position_for_phase(
    *,
    phase: str,
    phase_step: int,
    phase_length: int,
    hole_center: np.ndarray,
    xy_offset: np.ndarray,
    config: RolloutConfig,
) -> np.ndarray:
    if phase == "approach_hold":
        z_offset = config.approach_height
    elif phase == "descend":
        progress = min(max(phase_step / max(phase_length - 1, 1), 0.0), 1.0)
        z_offset = config.approach_height + progress * (config.descend_height - config.approach_height)
    elif phase == "insert":
        progress = min(max(phase_step / max(phase_length - 1, 1), 0.0), 1.0)
        z_offset = config.descend_height + progress * (-config.insert_depth - config.descend_height)
    elif phase in ("final_hold", "done"):
        z_offset = -config.insert_depth
    else:
        raise ValueError(f"Unsupported rollout phase {phase!r}.")
    return hole_center + np.array([xy_offset[0], xy_offset[1], z_offset], dtype=float)


def target_position_with_teleop_noise(
    target_position: np.ndarray,
    *,
    step: int,
    config: RolloutConfig,
    perturbation: RolloutPerturbation | None,
) -> np.ndarray:
    if perturbation is None or perturbation.teleop_noise_xy_amplitude <= 0.0:
        return target_position
    progress = min(max(step / max(config.total_steps, 1), 0.0), 1.0)
    envelope = np.sin(np.pi * progress)
    if envelope <= 0.0:
        return target_position
    angle = 2.0 * np.pi * perturbation.teleop_noise_cycles * progress
    noise = perturbation.teleop_noise_xy_amplitude * envelope * np.array(
        [
            np.sin(angle + perturbation.teleop_noise_phase_x),
            np.sin(angle + perturbation.teleop_noise_phase_y),
            0.0,
        ],
        dtype=float,
    )
    return target_position + noise


def _rotation_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rotation_y(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def target_rotation_with_peg_tilt(target_rotation: np.ndarray, perturbation: RolloutPerturbation | None) -> np.ndarray:
    if perturbation is None:
        return target_rotation
    if abs(perturbation.peg_tilt_x) <= 0.0 and abs(perturbation.peg_tilt_y) <= 0.0:
        return target_rotation
    return np.asarray(target_rotation, dtype=float).reshape(3, 3) @ _rotation_x(perturbation.peg_tilt_x) @ _rotation_y(perturbation.peg_tilt_y)


def scene_for_rollout(config_path: Path, perturbation: RolloutPerturbation | None) -> tuple[Path, dict]:
    from stiffness_copilot_mujoco.robustness import apply_perturbation_to_scene_config

    scene_config = load_scene_config(config_path)
    validate_canonical_eye_in_hand_camera_config(scene_config)
    if perturbation is None:
        return render_config_file(config_path), scene_config

    randomized = apply_perturbation_to_scene_config(scene_config, perturbation)
    validate_canonical_eye_in_hand_camera_config(randomized)
    return render_runtime_config(randomized, prefix=f"runtime_{Path(config_path).stem}_"), randomized


def perturbation_from_episode_spec(episode_spec: EpisodeSpec) -> RolloutPerturbation:
    return RolloutPerturbation(**episode_spec.to_perturbation_kwargs())


def target_position_stiffness_matrix(
    gains: TaskSpaceImpedanceGains,
    position_stiffness_matrix: np.ndarray | None,
) -> np.ndarray:
    if position_stiffness_matrix is not None:
        return np.asarray(position_stiffness_matrix, dtype=float).reshape(3, 3)
    stiffness = np.asarray(gains.position_stiffness, dtype=float)
    if stiffness.shape != (3,):
        raise ValueError(f"gains.position_stiffness must have shape (3,), got {stiffness.shape}.")
    return np.diag(stiffness)


def clip_torque(model: mujoco.MjModel, arm_ids, torque: np.ndarray) -> tuple[np.ndarray, bool]:
    clipped = np.asarray(torque, dtype=float).copy()
    saturated = False
    for idx, actuator_id in enumerate(arm_ids.actuator_ids):
        low, high = model.actuator_ctrlrange[actuator_id]
        value = float(np.clip(clipped[idx], low, high))
        saturated = saturated or abs(value - clipped[idx]) > 1e-9
        clipped[idx] = value
    return clipped, saturated


def run_fixed_stiffness_episode(
    *,
    baseline: BaselineName,
    seed: int,
    xy_offset: np.ndarray,
    config: RolloutConfig = RolloutConfig(),
    episode_id: int = 0,
    gains: TaskSpaceImpedanceGains | None = None,
    profile: str | None = None,
    position_stiffness_matrix: np.ndarray | None = None,
    stiffness_smoothing: StiffnessCommandSmoothingConfig | None = None,
    step_callback: StepCallback | None = None,
    perturbation: RolloutPerturbation | None = None,
    episode_spec: EpisodeSpec | None = None,
    teleop_mode: str | None = None,
    target_rotation_override: np.ndarray | None = None,
) -> EpisodeSummary:
    if episode_spec is not None and perturbation is None:
        perturbation = perturbation_from_episode_spec(episode_spec)
    nominal_scene_config = load_scene_config(config.config_path)
    scene_path, scene_config = scene_for_rollout(config.config_path, perturbation)
    try:
        model = load_model(scene_path)
    finally:
        cleanup_runtime_scene(scene_path)
    data = mujoco.MjData(model)
    reset_from_config(model, data, scene_config)

    geometry = geometry_from_config(scene_config)
    simulation_dt_seconds = float(scene_config.get("physics", {}).get("timestep", 0.002))
    task_ids = peg_hole_ids(model, segments=geometry.segments)
    arm_ids = panda_arm_ids(model)
    nullspace_target_qpos = arm_qpos(data, arm_ids)
    hole_center = hole_center_position(data, task_ids)
    initial_site_rotation = site_rotation(data, model.site(config.site_name).id)
    effective_teleop_mode = (
        validate_teleop_mode(teleop_mode)
        if teleop_mode is not None
        else (episode_spec.teleop_mode if episode_spec is not None else "position_only")
    )
    if episode_spec is not None and episode_spec.teleop_mode == TELEOP_MODE_POSITION_ORIENTATION:
        target_rotation = episode_spec.target_rotation_at_step(0)
    elif effective_teleop_mode == TELEOP_MODE_POSITION_ORIENTATION:
        target_rotation = (
            np.asarray(target_rotation_override, dtype=float).reshape(3, 3)
            if target_rotation_override is not None
            else nominal_hole_rotation(nominal_scene_config)
        )
    else:
        target_rotation = target_rotation_with_peg_tilt(initial_site_rotation, perturbation)
    if gains is None or profile is None:
        if baseline == "mid_high_baseline":
            profile, gains = load_task_space_impedance_gains(config.gain_config_path, TRACK_A_BASELINE_CONTROLLER_PROFILE)
        elif baseline == TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE:
            profile, gains = load_task_space_impedance_gains(
                config.gain_config_path,
                TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE,
            )
        else:
            _, loaded_profile, loaded_gains = load_baseline_gains(config.gain_config_path, baseline)
            profile = loaded_profile
            gains = loaded_gains

    xy = np.asarray(xy_offset, dtype=float)
    if xy.shape != (2,):
        raise ValueError(f"xy_offset must have shape (2,), got {xy.shape}.")
    if episode_spec is not None:
        if episode_spec.trajectory_source not in {
            EPISODE_TRAJECTORY_SOURCE_OPEN_LOOP_FAMILY,
            EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
        }:
            raise ValueError(f"Unsupported episode_spec trajectory_source {episode_spec.trajectory_source!r}.")
        if config.max_steps is not None and config.max_steps < episode_spec.total_steps:
            raise ValueError(
                f"RolloutConfig.max_steps {config.max_steps} is shorter than episode_spec.total_steps {episode_spec.total_steps}."
            )
    smoothing_config = resolve_deployment_stiffness_smoothing_config(stiffness_smoothing)
    smoother = StiffnessCommandSmoother(
        smoothing_config,
        simulation_dt_seconds=simulation_dt_seconds,
    )

    contact_detected = False
    hole_contact_detected = False
    contact_onset_step = -1
    depth_at_contact = 0.0
    max_normal_force = 0.0
    normal_force_sum = 0.0
    normal_force_count = 0
    max_tangential_force = 0.0
    max_penetration_depth = 0.0
    max_abs_torque = 0.0
    torque_saturation_count = 0
    final_orientation_error = 0.0
    final_orientation_error_vector = np.zeros(3, dtype=float)
    final_target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
    final_actual_ee_rotation = np.asarray(initial_site_rotation, dtype=float).reshape(3, 3)
    final_axis_alignment_error = 0.0
    final_step = 0

    effective_total_steps = episode_spec.total_steps if episode_spec is not None else config.total_steps
    for step in range(effective_total_steps + 1):
        if episode_spec is None:
            phase, phase_step, phase_length = phase_for_step(step, config)
            target_position = target_position_for_phase(
                phase=phase,
                phase_step=phase_step,
                phase_length=phase_length,
                hole_center=hole_center,
                xy_offset=xy,
                config=config,
            )
            target_position = target_position_with_teleop_noise(
                target_position,
                step=step,
                config=config,
                perturbation=perturbation,
            )
        else:
            phase = episode_spec.phase_name_at_step(step)
            target_position = episode_spec.target_position_at_step(step, reference_position=hole_center)
            if episode_spec.teleop_mode == "position_orientation":
                target_rotation = episode_spec.target_rotation_at_step(step)
        target_stiffness = target_position_stiffness_matrix(gains, position_stiffness_matrix)
        smoothing_step = smoother.apply(step=step, target_matrix=target_stiffness)
        command = task_space_impedance_torque(
            model,
            data,
            site_name=config.site_name,
            target_position=target_position,
            target_rotation=target_rotation,
            arm_ids=arm_ids,
            gains=gains,
            position_stiffness_matrix=smoothing_step.command_matrix,
            nullspace_target_qpos=nullspace_target_qpos,
            clip_to_ctrlrange=False,
        )
        commanded_torque, saturated = clip_torque(model, arm_ids, command.torque)
        max_abs_torque = max(max_abs_torque, float(np.max(np.abs(commanded_torque))))
        torque_saturation_count += int(saturated)

        contact = extract_contact_state(ContactQuery(model=model, data=data, task_ids=task_ids))
        if contact.in_contact:
            if not contact_detected:
                contact_detected = True
                hole_contact_detected = True
                contact_onset_step = step
                depth_at_contact = insertion_depth(data, task_ids)
            normal_force_sum += contact.normal_force
            normal_force_count += 1
        max_normal_force = max(max_normal_force, contact.normal_force)
        max_tangential_force = max(max_tangential_force, contact.tangential_force)
        max_penetration_depth = max(max_penetration_depth, contact.penetration_depth)
        final_orientation_error = float(np.linalg.norm(command.orientation_error))
        final_orientation_error_vector = np.asarray(command.orientation_error, dtype=float)
        final_target_rotation = np.asarray(target_rotation, dtype=float).reshape(3, 3)
        final_actual_ee_rotation = site_rotation(data, model.site(config.site_name).id)
        final_axis_alignment_error = float(1.0 - peg_axis_alignment(data, task_ids, hole_insertion_axis_world()))

        metrics = {
            "target_position": target_position,
            "position_error": float(np.linalg.norm(command.position_error)),
            "orientation_error": final_orientation_error,
            "lateral_error": lateral_error(data, task_ids),
            "depth": insertion_depth(data, task_ids),
            "normal_force": contact.normal_force,
            "tangential_force": contact.tangential_force,
            "penetration_depth": contact.penetration_depth,
            "max_abs_commanded_torque": max_abs_torque,
            "torque_saturated": saturated,
            "in_contact": contact.in_contact,
            "position_stiffness_target": target_stiffness,
            "position_stiffness_command": smoothing_step.command_matrix,
            "smoothing_update_applied": smoothing_step.update_applied,
            "smoothing_hold_applied": smoothing_step.hold_applied,
            "teleop_mode": effective_teleop_mode,
            "target_rotation": np.asarray(target_rotation, dtype=float).reshape(3, 3),
            "actual_ee_rotation": final_actual_ee_rotation,
            "orientation_error_vector": final_orientation_error_vector,
            "axis_alignment_error": final_axis_alignment_error,
        }
        if config.print_every and step % config.print_every == 0:
            print(
                f"baseline={baseline} step={step} phase={phase} "
                f"lat={metrics['lateral_error']:.6f} depth={metrics['depth']:.6f} "
                f"normal={metrics['normal_force']:.3f} max_tau={max_abs_torque:.3f}"
            )
        if step_callback is not None:
            step_callback(step, phase, model, data, metrics)

        final_step = step
        if phase == "done":
            break
        set_arm_torque_ctrl(model, data, arm_ids, commanded_torque)
        mujoco.mj_step(model, data)

    final_depth = insertion_depth(data, task_ids)
    final_lateral = lateral_error(data, task_ids)
    mean_normal_force_contact = normal_force_sum / normal_force_count if normal_force_count else 0.0
    depth_reached = bool(final_depth >= 0.95 * config.insert_depth)
    low_force_success = bool(depth_reached and max_normal_force <= DEFAULT_LOW_FORCE_SUCCESS_THRESHOLD)
    smoothing_summary = smoother.summary_dict()
    smoothing_config_summary = smoothing_summary["config"]
    return EpisodeSummary(
        baseline=baseline,
        profile=profile,
        seed=seed,
        episode_id=episode_id,
        xy_offset=(float(xy[0]), float(xy[1])),
        steps=final_step,
        final_depth=final_depth,
        final_lateral_error=final_lateral,
        final_axis_alignment=peg_axis_alignment(data, task_ids, hole_insertion_axis_world()),
        final_orientation_error=final_orientation_error,
        contact_detected=contact_detected,
        hole_contact_detected=hole_contact_detected,
        contact_onset_step=contact_onset_step,
        max_normal_force=max_normal_force,
        mean_normal_force_contact=mean_normal_force_contact,
        max_tangential_force=max_tangential_force,
        max_penetration_depth=max_penetration_depth,
        max_abs_commanded_torque=max_abs_torque,
        torque_saturation_count=torque_saturation_count,
        depth_progress_after_contact=final_depth - depth_at_contact if contact_detected else 0.0,
        depth_reached=depth_reached,
        low_force_success=low_force_success,
        completion_like=depth_reached,
        smoothing_enabled=bool(smoothing_config_summary["enabled"]),
        smoothing_method=str(smoothing_config_summary["method"]),
        smoothing_alpha=float(smoothing_config_summary["alpha"]),
        policy_update_period_steps=int(smoothing_config_summary["policy_update_period_steps"]),
        hold_between_updates=bool(smoothing_config_summary["hold_between_updates"]),
        smoothing_target_kind=str(smoothing_config_summary["target_kind"]),
        stiffness_before_smoothing_summary=dict(smoothing_summary["stiffness_before_smoothing_summary"]),
        stiffness_after_smoothing_summary=dict(smoothing_summary["stiffness_after_smoothing_summary"]),
        perturbation=perturbation.to_dict() if perturbation is not None else None,
        teleop_mode=effective_teleop_mode,
        target_rotation=final_target_rotation,
        actual_ee_rotation=final_actual_ee_rotation,
        orientation_error_vector=final_orientation_error_vector,
        axis_alignment_error=final_axis_alignment_error,
    )
