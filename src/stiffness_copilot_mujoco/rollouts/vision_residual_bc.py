from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import ContactQuery, extract_contact_state, extract_net_peg_hole_contact_force_world
from stiffness_copilot_mujoco.controllers.impedance import (
    TRACK_A_BASELINE_CONTROLLER_PROFILE,
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
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.learning.vision_residual_stiffness import (
    IMAGE_ONLY_RESIDUAL_BC_METHOD_NAME,
    IMAGE_ONLY_RESIDUAL_BC_OUTPUT_SPACE,
    IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND,
    VisionResidualBCPolicy,
    load_image_only_residual_bc_policy,
)
from stiffness_copilot_mujoco.evaluation.force_metrics import ForceMetricThresholds
from stiffness_copilot_mujoco.metrics.task_metrics import (
    geometry_from_config,
    hole_center_position,
    hole_insertion_axis_world,
    insertion_depth,
    lateral_error,
    peg_axis_alignment,
)
from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl
from stiffness_copilot_mujoco.pose_math import site_rotation
from stiffness_copilot_mujoco.rollout_observation import reset_from_config
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    DEFAULT_GAIN_CONFIG,
    EpisodeSummary,
    RolloutConfig,
    RolloutPerturbation,
    cleanup_runtime_scene,
    clip_torque,
    phase_for_step,
    perturbation_from_episode_spec,
    scene_for_rollout,
    target_position_for_phase,
    target_position_with_teleop_noise,
    target_rotation_with_peg_tilt,
)
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
from stiffness_copilot_mujoco.sim.scene import (
    apply_eye_in_hand_camera_pose,
    eye_in_hand_camera_pose_from_config,
    validate_canonical_eye_in_hand_camera_config,
)
from stiffness_copilot_mujoco.vision.rendering import MujocoRgbRenderer


@dataclass(frozen=True, kw_only=True)
class VisionImageOnlyResidualBCEpisodeSummary(EpisodeSummary):
    policy_path: str
    camera_name: str
    image_width: int
    image_height: int
    renderer_mode: str
    residual_scale: float
    fallback_used: bool
    policy_input_type: str
    uses_privileged_state_for_policy: bool
    mean_residual_pred: float
    max_abs_residual_pred: float
    mean_residual_after_bound: float
    max_abs_residual_after_bound: float
    mean_policy_theta: tuple[float, ...]
    mean_stiffness_eig: tuple[float, float, float]
    min_stiffness_eig: float
    max_stiffness_eig: float
    update_rate_hz: float | None
    stiffness_update_hz_target: float | None
    smoothing_scheduler: str
    stiffness_smoothing_summary: dict[str, object]
    full_trace: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result.update(
            {
                "policy_path": self.policy_path,
                "camera_name": self.camera_name,
                "image_width": self.image_width,
                "image_height": self.image_height,
                "renderer_mode": self.renderer_mode,
                "residual_scale": self.residual_scale,
                "fallback_used": self.fallback_used,
                "policy_input_type": self.policy_input_type,
                "uses_privileged_state_for_policy": self.uses_privileged_state_for_policy,
                "mean_residual_pred": self.mean_residual_pred,
                "max_abs_residual_pred": self.max_abs_residual_pred,
                "mean_residual_after_bound": self.mean_residual_after_bound,
                "max_abs_residual_after_bound": self.max_abs_residual_after_bound,
                "mean_policy_theta": list(self.mean_policy_theta),
                "mean_stiffness_eig": list(self.mean_stiffness_eig),
                "min_stiffness_eig": self.min_stiffness_eig,
                "max_stiffness_eig": self.max_stiffness_eig,
                "update_rate_hz": self.update_rate_hz,
                "stiffness_update_hz_target": self.stiffness_update_hz_target,
                "smoothing_scheduler": self.smoothing_scheduler,
                "stiffness_smoothing_summary": self.stiffness_smoothing_summary,
            }
        )
        return result


def _assert_rgb_frame(frame: np.ndarray, *, expected_height: int, expected_width: int) -> np.ndarray:
    image = np.asarray(frame, dtype=np.uint8)
    if image.shape != (expected_height, expected_width, 3):
        raise ValueError(f"RGB frame must have shape {(expected_height, expected_width, 3)}, observed {image.shape}.")
    return image


def run_image_only_residual_bc_episode(
    *,
    policy: VisionResidualBCPolicy,
    policy_path: Path,
    seed: int,
    xy_offset: np.ndarray,
    config: RolloutConfig = RolloutConfig(),
    episode_id: int = 0,
    reference_controller_id: str = TRACK_A_BASELINE_CONTROLLER_PROFILE,
    reference_stiffness_matrix: np.ndarray | None = None,
    perturbation: RolloutPerturbation | None = None,
    camera_name: str = "eye_in_hand_rgb",
    image_width: int = 128,
    image_height: int = 128,
    renderer_mode: str = "native",
    residual_scale: float = 1.0,
    stiffness_smoothing: StiffnessCommandSmoothingConfig | None = None,
    step_callback: Callable[[int, str, mujoco.MjModel, mujoco.MjData, dict[str, float | bool | np.ndarray]], None] | None = None,
    episode_spec: EpisodeSpec | None = None,
) -> VisionImageOnlyResidualBCEpisodeSummary:
    if renderer_mode not in {"native", "legacy_debug_only"}:
        raise ValueError("renderer_mode must be one of {'native', 'legacy_debug_only'}.")
    residual_scale = float(residual_scale)
    if not np.isfinite(residual_scale) or residual_scale < 0.0:
        raise ValueError("residual_scale must be a finite, non-negative scalar.")
    if episode_spec is not None and perturbation is None:
        perturbation = perturbation_from_episode_spec(episode_spec)
    scene_path, scene_config = scene_for_rollout(config.config_path, perturbation)
    try:
        model = load_model(scene_path)
    finally:
        cleanup_runtime_scene(scene_path)

    data = mujoco.MjData(model)
    reset_from_config(model, data, scene_config)
    validate_canonical_eye_in_hand_camera_config(scene_config, camera_name=camera_name)
    camera_local_position, camera_rotation, camera_fovy = eye_in_hand_camera_pose_from_config(
        scene_config,
        camera_name=camera_name,
    )
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Unknown camera {camera_name!r}.")
    apply_eye_in_hand_camera_pose(
        model,
        data,
        camera_id=camera_id,
        local_position=camera_local_position,
        rotation=camera_rotation,
        fovy=camera_fovy,
    )
    geometry = geometry_from_config(scene_config)
    simulation_dt_seconds = float(scene_config.get("physics", {}).get("timestep", 0.002))
    task_ids = peg_hole_ids(model, segments=geometry.segments)
    arm_ids = panda_arm_ids(model)
    nullspace_target_qpos = arm_qpos(data, arm_ids)
    hole_center = hole_center_position(data, task_ids)
    target_rotation = target_rotation_with_peg_tilt(site_rotation(data, model.site(config.site_name).id), perturbation)
    loaded_profile, gains = load_task_space_impedance_gains(config.gain_config_path or DEFAULT_GAIN_CONFIG, reference_controller_id)
    reference_controller_id = loaded_profile
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
    if reference_stiffness_matrix is None:
        reference_stiffness_matrix = np.diag(np.asarray(gains.position_stiffness, dtype=float))
    reference_stiffness_matrix = np.asarray(reference_stiffness_matrix, dtype=float)
    if reference_stiffness_matrix.shape != (3, 3):
        raise ValueError(f"reference_stiffness_matrix must have shape (3, 3), got {reference_stiffness_matrix.shape}.")

    contact_detected = False
    hole_contact_detected = False
    contact_onset_step = -1
    depth_at_contact = 0.0
    normal_force_sum = 0.0
    normal_force_count = 0
    max_normal_force = 0.0
    max_tangential_force = 0.0
    max_penetration_depth = 0.0
    max_abs_torque = 0.0
    torque_saturation_count = 0
    final_orientation_error = 0.0
    final_step = 0
    residual_preds: list[float] = []
    residual_scaled_preds: list[float] = []
    residual_after_bounds: list[float] = []
    theta_values: list[np.ndarray] = []
    stiffness_eigs: list[np.ndarray] = []
    trace_rows: list[dict[str, object]] = []

    effective_total_steps = episode_spec.total_steps if episode_spec is not None else config.total_steps

    with MujocoRgbRenderer(
        model,
        camera_name=camera_name,
        width=image_width,
        height=image_height,
        renderer_mode=renderer_mode,
    ) as renderer:
        renderer_mode = renderer.mode
        fallback_used = bool(renderer.fallback_used)
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
                target_position = target_position_with_teleop_noise(target_position, step=step, config=config, perturbation=perturbation)
            else:
                phase = episode_spec.phase_name_at_step(step)
                target_position = episode_spec.target_position_at_step(step, reference_position=hole_center)
            frame = _assert_rgb_frame(renderer.render(data), expected_height=image_height, expected_width=image_width)
            raw, bounded, position_stiffness, theta, theta_delta = policy.predict_image_only(
                frame,
                residual_scale=residual_scale,
            )
            raw_vector = np.asarray(raw, dtype=float).reshape(-1)
            bounded_vector = np.asarray(bounded, dtype=float).reshape(-1)
            raw_scalar = float(raw_vector[0]) if raw_vector.size else 0.0
            scaled_vector = raw_vector * residual_scale
            scaled_scalar = float(scaled_vector[0]) if scaled_vector.size else 0.0
            bounded_scalar = float(bounded_vector[0]) if bounded_vector.size else 0.0
            if not np.allclose(policy.base_spec.base_matrix, reference_stiffness_matrix, atol=1e-9, rtol=0.0):
                raise RuntimeError("Residual policy base stiffness does not match the selected reference controller matrix.")
            if not np.isfinite(raw_scalar) or not np.isfinite(bounded_scalar):
                raise RuntimeError("Vision policy produced a non-finite residual.")
            if abs(bounded_scalar) > IMAGE_ONLY_RESIDUAL_BC_RESIDUAL_BOUND + 1e-9:
                raise RuntimeError("Vision policy residual exceeded the configured residual bound.")
            if not np.all(np.isfinite(position_stiffness)) or not np.all(np.isfinite(theta)) or not np.all(np.isfinite(theta_delta)):
                raise RuntimeError("Vision policy produced non-finite stiffness reconstruction.")
            eig = np.linalg.eigvalsh(position_stiffness)
            if not np.all(np.isfinite(eig)) or np.min(eig) <= 0.0:
                raise RuntimeError("Vision policy stiffness reconstruction is not SPD-safe.")
            command_matrix = reference_stiffness_matrix + (position_stiffness - policy.base_spec.base_matrix)
            smoothing_step = smoother.apply(step=step, target_matrix=command_matrix)

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
            contact_force_world = extract_net_peg_hole_contact_force_world(ContactQuery(model=model, data=data, task_ids=task_ids))
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
            if step_callback is not None:
                step_callback(
                    step,
                    phase,
                    model,
                    data,
                    {
                        "target_position": target_position,
                        "position_error": float(np.linalg.norm(command.position_error)),
                        "orientation_error": final_orientation_error,
                        "lateral_error": float(lateral_error(data, task_ids)),
                        "depth": float(insertion_depth(data, task_ids)),
                        "normal_force": float(contact.normal_force),
                        "tangential_force": float(contact.tangential_force),
                        "penetration_depth": float(contact.penetration_depth),
                        "max_abs_commanded_torque": float(max_abs_torque),
                        "torque_saturated": bool(saturated),
                        "in_contact": bool(contact.in_contact),
                        "predicted_residual_raw": raw_scalar,
                        "predicted_residual_scaled": scaled_scalar,
                        "residual_pred": raw_scalar,
                        "residual_after_bound": bounded_scalar,
                        "residual_scale": residual_scale,
                        "position_stiffness_before_residual": reference_stiffness_matrix,
                        "position_stiffness_after_residual": position_stiffness,
                        "position_stiffness_target": position_stiffness,
                        "position_stiffness_command": smoothing_step.command_matrix,
                        "stiffness_matrix_raw_before_smoothing": smoothing_step.raw_matrix_before_smoothing,
                        "stiffness_matrix_after_smoothing": smoothing_step.smoothed_matrix,
                        "stiffness_update_hz_target": smoothing_step.stiffness_update_hz_target,
                        "stiffness_update_index": smoothing_step.stiffness_update_index,
                        "stiffness_update_interval_steps": smoothing_step.stiffness_update_interval_steps,
                        "stiffness_update_interval_seconds": smoothing_step.stiffness_update_interval_seconds,
                        "steps_since_last_stiffness_refresh": smoothing_step.steps_since_last_stiffness_refresh,
                        "stiffness_smoothing_scheduler": smoothing_step.scheduler,
                        "stiffness_update_scheduler": smoothing_step.scheduler,
                        "stiffness_refreshed_this_step": bool(smoothing_step.update_applied),
                        "smoothing_update_applied": smoothing_step.update_applied,
                        "smoothing_hold_applied": smoothing_step.hold_applied,
                        "offdiag_xy_before_smoothing": float(np.asarray(smoothing_step.raw_matrix_before_smoothing, dtype=float)[0, 1])
                        if smoothing_step.raw_matrix_before_smoothing is not None
                        else None,
                        "offdiag_xy_after_smoothing": float(np.asarray(smoothing_step.smoothed_matrix, dtype=float)[0, 1])
                        if smoothing_step.smoothed_matrix is not None
                        else None,
                    },
                )
            residual_preds.append(raw_scalar)
            residual_scaled_preds.append(scaled_scalar)
            residual_after_bounds.append(bounded_scalar)
            theta_values.append(theta)
            stiffness_eigs.append(eig)
            trace_rows.append(
                {
                    "episode_id": int(episode_id),
                    "episode_spec_id": episode_spec.episode_spec_id if episode_spec is not None else None,
                    "trajectory_family": episode_spec.trajectory_family if episode_spec is not None else None,
                    "trajectory_family_id": int(episode_spec.trajectory_family_id) if episode_spec is not None else None,
                    "controller_id": reference_controller_id,
                    "seed": int(seed),
                    "episode": int(episode_id),
                    "step": int(step),
                    "time": float(data.time),
                    "phase": phase,
                    "depth": float(insertion_depth(data, task_ids)),
                    "lateral_error": float(lateral_error(data, task_ids)),
                    "residual_scale": residual_scale,
                    "predicted_residual_raw": raw_scalar,
                    "predicted_residual_scaled": scaled_scalar,
                    "residual_pred": raw_scalar,
                    "residual_pred_vector": raw_vector.tolist(),
                    "residual_action": raw_scalar,
                    "residual_action_vector": raw_vector.tolist(),
                    "residual_after_bound": bounded_scalar,
                    "residual_after_bound_vector": bounded_vector.tolist(),
                    "residual_action_after_bound": bounded_scalar,
                    "residual_action_after_bound_vector": bounded_vector.tolist(),
                    "group_delta": bounded_vector.tolist(),
                    "theta_delta": [float(v) for v in np.asarray(theta_delta, dtype=float).reshape(-1)],
                    "theta": [float(v) for v in np.asarray(theta, dtype=float).reshape(-1)],
                    "stiffness_matrix_before_residual": np.asarray(reference_stiffness_matrix, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_matrix_after_residual": np.asarray(position_stiffness, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_x": float(position_stiffness[0, 0]),
                    "stiffness_y": float(position_stiffness[1, 1]),
                    "stiffness_z": float(position_stiffness[2, 2]),
                    "stiffness_matrix_target": np.asarray(position_stiffness, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_matrix_raw_before_smoothing": np.asarray(smoothing_step.raw_matrix_before_smoothing, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_matrix_after_smoothing": np.asarray(smoothing_step.smoothed_matrix, dtype=float).reshape(3, 3).tolist(),
                    "stiffness_matrix_command": np.asarray(smoothing_step.command_matrix, dtype=float).reshape(3, 3).tolist(),
                    "reference_stiffness_x": float(reference_stiffness_matrix[0, 0]),
                    "reference_stiffness_y": float(reference_stiffness_matrix[1, 1]),
                    "reference_stiffness_z": float(reference_stiffness_matrix[2, 2]),
                    "stiffness_command_x": float(smoothing_step.command_matrix[0, 0]),
                    "stiffness_command_y": float(smoothing_step.command_matrix[1, 1]),
                    "stiffness_command_z": float(smoothing_step.command_matrix[2, 2]),
                    "stiffness_update_hz_target": smoothing_step.stiffness_update_hz_target,
                    "stiffness_update_index": smoothing_step.stiffness_update_index,
                    "stiffness_update_interval_steps": smoothing_step.stiffness_update_interval_steps,
                    "stiffness_update_interval_seconds": smoothing_step.stiffness_update_interval_seconds,
                    "steps_since_last_stiffness_refresh": smoothing_step.steps_since_last_stiffness_refresh,
                    "stiffness_smoothing_scheduler": smoothing_step.scheduler,
                    "stiffness_update_scheduler": smoothing_step.scheduler,
                    "stiffness_refreshed_this_step": bool(smoothing_step.update_applied),
                    "normal_force": float(contact.normal_force),
                    "contact_state": bool(contact.in_contact),
                    "contact_force_world": [float(v) for v in np.asarray(contact_force_world, dtype=float).reshape(-1)],
                    "smoothing_update_applied": bool(smoothing_step.update_applied),
                    "smoothing_hold_applied": bool(smoothing_step.hold_applied),
                    "offdiag_xy_before_smoothing": float(np.asarray(smoothing_step.raw_matrix_before_smoothing, dtype=float)[0, 1])
                    if smoothing_step.raw_matrix_before_smoothing is not None
                    else None,
                    "offdiag_xy_after_smoothing": float(np.asarray(smoothing_step.smoothed_matrix, dtype=float)[0, 1])
                    if smoothing_step.smoothed_matrix is not None
                    else None,
                    "renderer_mode": renderer_mode,
                    "fallback_used": fallback_used,
                    "policy_input_type": "rgb_only",
                    "uses_privileged_state_for_policy": False,
                    "policy_camera_name": camera_name,
                }
            )
            final_step = step
            if phase == "done":
                break
            set_arm_torque_ctrl(model, data, arm_ids, commanded_torque)
            mujoco.mj_step(model, data)

    final_depth = insertion_depth(data, task_ids)
    final_lateral = lateral_error(data, task_ids)
    mean_normal_force_contact = normal_force_sum / normal_force_count if normal_force_count else 0.0
    completion_like = bool(final_depth >= 0.95 * config.insert_depth and final_lateral <= geometry.radial_clearance)
    low_force_success = bool(final_depth >= 0.95 * config.insert_depth and max_normal_force <= ForceMetricThresholds().low_force_success_threshold)
    smoothing_summary = smoother.summary_dict()
    smoothing_config_summary = smoothing_summary["config"]
    theta_array = np.vstack(theta_values) if theta_values else np.zeros((0, 6), dtype=float)
    eig_array = np.vstack(stiffness_eigs) if stiffness_eigs else np.zeros((0, 3), dtype=float)
    return VisionImageOnlyResidualBCEpisodeSummary(
        baseline="image_only_residual_bc",
        profile=reference_controller_id,
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
        depth_reached=bool(final_depth >= 0.95 * config.insert_depth),
        low_force_success=low_force_success,
        completion_like=completion_like,
        smoothing_enabled=bool(smoothing_config_summary["enabled"]),
        smoothing_method=str(smoothing_config_summary["method"]),
        smoothing_alpha=float(smoothing_config_summary["alpha"]),
        policy_update_period_steps=int(smoothing_config_summary["policy_update_period_steps"]),
        update_rate_hz=None
        if smoothing_config_summary.get("update_rate_hz") is None
        else float(smoothing_config_summary["update_rate_hz"]),
        stiffness_update_hz_target=None
        if smoothing_config_summary.get("stiffness_update_hz_target") is None
        else float(smoothing_config_summary["stiffness_update_hz_target"]),
        smoothing_scheduler=str(smoothing_summary.get("scheduler", smoothing_config_summary.get("scheduler", "fixed_period"))),
        hold_between_updates=bool(smoothing_config_summary["hold_between_updates"]),
        smoothing_target_kind=str(smoothing_config_summary["target_kind"]),
        stiffness_before_smoothing_summary=dict(smoothing_summary["stiffness_before_smoothing_summary"]),
        stiffness_after_smoothing_summary=dict(smoothing_summary["stiffness_after_smoothing_summary"]),
        stiffness_smoothing_summary=dict(smoothing_summary),
        perturbation=perturbation.to_dict() if perturbation is not None else None,
        policy_path=str(policy_path),
        camera_name=camera_name,
        image_width=image_width,
        image_height=image_height,
        renderer_mode=renderer_mode,
        residual_scale=residual_scale,
        fallback_used=fallback_used,
        policy_input_type="rgb_only",
        uses_privileged_state_for_policy=False,
        mean_residual_pred=float(np.mean(residual_preds)) if residual_preds else 0.0,
        max_abs_residual_pred=float(np.max(np.abs(residual_preds))) if residual_preds else 0.0,
        mean_residual_after_bound=float(np.mean(residual_after_bounds)) if residual_after_bounds else 0.0,
        max_abs_residual_after_bound=float(np.max(np.abs(residual_after_bounds))) if residual_after_bounds else 0.0,
        mean_policy_theta=tuple(float(v) for v in np.mean(theta_array, axis=0)) if theta_array.size else tuple(),
        mean_stiffness_eig=tuple(float(v) for v in np.mean(eig_array, axis=0)) if eig_array.size else (0.0, 0.0, 0.0),
        min_stiffness_eig=float(np.min(eig_array)) if eig_array.size else 0.0,
        max_stiffness_eig=float(np.max(eig_array)) if eig_array.size else 0.0,
        full_trace=trace_rows,
    )


__all__ = [
    "VisionImageOnlyResidualBCEpisodeSummary",
    "load_image_only_residual_bc_policy",
    "run_image_only_residual_bc_episode",
]
