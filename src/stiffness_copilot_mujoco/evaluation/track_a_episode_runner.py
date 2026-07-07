from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from stiffness_copilot_mujoco.controllers.controller_spec import ControllerSpec
from stiffness_copilot_mujoco.controllers.impedance import load_task_space_impedance_gains
from stiffness_copilot_mujoco.controllers.track_a_controllers import (
    DEFAULT_TRACK_A_CONTROLLERS_YAML,
    get_track_a_controller,
)
from stiffness_copilot_mujoco.controllers.stiffness_command_smoothing import (
    StiffnessCommandSmoothingConfig,
    resolve_deployment_stiffness_smoothing_config,
)
from stiffness_copilot_mujoco.evaluation.force_metrics import ForceMetricThresholds
from stiffness_copilot_mujoco.episodes.episode_spec import EpisodeSpec, EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.learning.vision_residual_stiffness import describe_residual_policy_contract
from stiffness_copilot_mujoco.metrics.task_metrics import geometry_from_config, hole_center_position, load_scene_config, site_position
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    DEFAULT_GAIN_CONFIG,
    EpisodeSummary,
    RolloutConfig,
    RolloutPerturbation,
    clip_torque,
    perturbation_from_episode_spec,
    run_fixed_stiffness_episode,
    scene_for_rollout,
)
from stiffness_copilot_mujoco.rollouts.vision_residual_bc import (
    VisionImageOnlyResidualBCEpisodeSummary,
    load_image_only_residual_bc_policy,
    run_image_only_residual_bc_episode,
)
from stiffness_copilot_mujoco.controllers.impedance import task_space_impedance_torque
from stiffness_copilot_mujoco.contact.state import ContactQuery, extract_contact_state, extract_net_peg_hole_contact_force_world
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl
from stiffness_copilot_mujoco.pose_math import site_rotation
from stiffness_copilot_mujoco.rollout_observation import reset_from_config
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
from stiffness_copilot_mujoco.sim.scene import (
    CANONICAL_EYE_IN_HAND_CAMERA_ATTACHMENT_PARENT,
    CANONICAL_EYE_IN_HAND_CAMERA_MOUNT_TYPE,
    CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION,
    canonical_eye_in_hand_camera_pose,
    cleanup_runtime_scene,
)
from stiffness_copilot_mujoco.metrics.task_metrics import lateral_error, insertion_depth, hole_insertion_axis_world, peg_axis_alignment


TRACK_A_COLLECTION_CONTROLLER_PROFILE = "track_a_c600"
TRACK_A_BASELINE_CONTROLLER_PROFILE = "track_a_baseline_b325"
TRACK_A_PAIRING_REFERENCE_POSITION_MODE = "hole_center_position"
TRACK_A_CONTACT_ACCOUNTING_MODE = "rollout_trace_contact_mask"
TRACK_A_RUNNER_VERSION = "track_a_paired_screening_v2_objective_protocol"
TRACK_A_STIFFNESS_REPRESENTATION = "full_spd_matrix"
TRACK_A_RESIDUAL_PARAMETERIZATION = "constrained_shared_lateral_scalar"
TRACK_A_RESIDUAL_AFFECTS = ["K_xx", "K_yy"]
TRACK_A_RESIDUAL_UNAFFECTED = ["K_zz", "off_diagonal_terms"]
TRACK_A_SMOOTHING_REQUIRED = True


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class TrackATraceStats:
    max_position_error: float = 0.0
    max_lateral_tracking_error: float = 0.0
    max_vertical_tracking_error: float = 0.0
    final_position_error: float = 0.0
    final_actual_x: float = 0.0
    final_actual_y: float = 0.0
    final_actual_z: float = 0.0
    final_target_x: float = 0.0
    final_target_y: float = 0.0
    final_target_z: float = 0.0
    steps: int = 0


def _step_trace_callback(
    *,
    role: str,
    rows: list[dict[str, object]],
    normal_forces: list[float],
    stats: TrackATraceStats,
) -> Callable[[int, str, mujoco.MjModel, mujoco.MjData, dict[str, float | bool | np.ndarray]], None]:
    def callback(step: int, phase: str, model: mujoco.MjModel, data: mujoco.MjData, metrics: dict[str, float | bool | np.ndarray]) -> None:
        target_position = np.asarray(metrics["target_position"], dtype=float)
        actual_position = site_position(data, model.site("peg_tip").id)
        normal_force = float(metrics["normal_force"])
        normal_forces.append(normal_force)
        stats.max_position_error = max(stats.max_position_error, float(metrics["position_error"]))
        stats.max_lateral_tracking_error = max(
            stats.max_lateral_tracking_error,
            float(np.linalg.norm(actual_position[:2] - target_position[:2])),
        )
        stats.max_vertical_tracking_error = max(stats.max_vertical_tracking_error, abs(float(actual_position[2] - target_position[2])))
        stats.final_position_error = float(metrics["position_error"])
        stats.final_actual_x = float(actual_position[0])
        stats.final_actual_y = float(actual_position[1])
        stats.final_actual_z = float(actual_position[2])
        stats.final_target_x = float(target_position[0])
        stats.final_target_y = float(target_position[1])
        stats.final_target_z = float(target_position[2])
        stats.steps = int(step)
        rows.append(
            {
                "controller_role": role,
                "step": int(step),
                "phase": phase,
                "target_x": float(target_position[0]),
                "target_y": float(target_position[1]),
                "target_z": float(target_position[2]),
                "actual_x": float(actual_position[0]),
                "actual_y": float(actual_position[1]),
                "actual_z": float(actual_position[2]),
                "position_error": float(metrics["position_error"]),
                "lateral_error": float(metrics["lateral_error"]),
                "depth": float(metrics["depth"]),
                "normal_force": normal_force,
                "tangential_force": float(metrics["tangential_force"]),
                "penetration_depth": float(metrics["penetration_depth"]),
                "torque_saturated": bool(metrics["torque_saturated"]),
                "in_contact": bool(metrics["in_contact"]),
                "stiffness_matrix_raw_before_smoothing": None,
                "stiffness_matrix_after_smoothing": None,
                "stiffness_update_hz_target": None,
                "stiffness_update_index": None,
                "stiffness_update_interval_steps": None,
                "stiffness_update_interval_seconds": None,
                "steps_since_last_stiffness_refresh": None,
                "stiffness_smoothing_scheduler": None,
                "stiffness_update_scheduler": None,
                "stiffness_refreshed_this_step": None,
                "offdiag_xy_before_smoothing": None,
                "offdiag_xy_after_smoothing": None,
            }
        )

    return callback


def _trace_contact_count(rows: list[dict[str, object]]) -> int:
    return int(sum(1 for row in rows if bool(row.get("in_contact"))))


def _force_quantiles(forces: list[float]) -> dict[str, float]:
    if not forces:
        return {"p95_normal_force": 0.0, "p99_normal_force": 0.0}
    array = np.asarray(forces, dtype=float)
    return {
        "p95_normal_force": float(np.percentile(array, 95)),
        "p99_normal_force": float(np.percentile(array, 99)),
    }


def _controller_spec_to_dict(controller_spec: ControllerSpec, *, simulation_dt_seconds: float) -> dict[str, Any]:
    return controller_spec.to_dict(simulation_dt_seconds=simulation_dt_seconds)


def _strip_full_trace(summary_dict: dict[str, Any]) -> dict[str, Any]:
    payload = dict(summary_dict)
    payload.pop("full_trace", None)
    return payload


def _episode_termination_reason(summary: EpisodeSummary | VisionImageOnlyResidualBCEpisodeSummary) -> str:
    if summary.depth_reached and summary.low_force_success:
        return "success_low_force"
    if summary.depth_reached:
        return "success_depth_reached"
    if summary.contact_detected and summary.torque_saturation_count > 0:
        return "torque_saturation"
    if summary.contact_detected:
        return "contact_failure"
    return "no_contact"


def _make_controller_spec(
    config_path: Path,
    *,
    requested_profile: str,
    controller_role: str,
    position_stiffness_matrix: np.ndarray | None = None,
    position_stiffness_matrix_source: str | None = None,
) -> ControllerSpec:
    selected_profile, gains = load_task_space_impedance_gains(config_path, requested_profile)
    return ControllerSpec(
        controller_role=controller_role,
        controller_kind="task_space_impedance",
        requested_profile=requested_profile,
        selected_profile=selected_profile,
        gain_config_path=str(config_path),
        gains=gains,
        position_stiffness_matrix=None if position_stiffness_matrix is None else np.asarray(position_stiffness_matrix, dtype=float),
        position_stiffness_matrix_source=position_stiffness_matrix_source,
    )


def _shared_episode_metadata(
    *,
    controller_profile: str,
    controller_id: str | None = None,
    controller_stiffness_matrix: np.ndarray | None = None,
    controllers_yaml: Path | None = None,
    episode_spec: EpisodeSpec,
    simulation_dt_seconds: float,
    controller_spec: ControllerSpec | None = None,
    policy_path: Path | None = None,
    policy_metadata: dict[str, Any] | None = None,
    smoothing_config: StiffnessCommandSmoothingConfig | None = None,
    summary: EpisodeSummary | VisionImageOnlyResidualBCEpisodeSummary,
    normal_forces: list[float],
    trace_rows: list[dict[str, object]],
    trace_stats: TrackATraceStats,
    runtime_mode: str,
    full_trace: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    force_quantiles = _force_quantiles(normal_forces)
    update_steps = [int(row["step"]) for row in trace_rows if bool(row.get("stiffness_refreshed_this_step", row.get("smoothing_update_applied", False)))]
    update_interval_steps = [int(b - a) for a, b in zip(update_steps[:-1], update_steps[1:])] if len(update_steps) > 1 else []
    update_interval_distribution: dict[str, int] = {}
    for interval in update_interval_steps:
        key = str(int(interval))
        update_interval_distribution[key] = update_interval_distribution.get(key, 0) + 1
    duration_seconds = (
        float(trace_rows[-1]["time"]) - float(trace_rows[0]["time"])
        if len(trace_rows) > 1 and "time" in trace_rows[0] and "time" in trace_rows[-1]
        else 0.0
    )
    achieved_stiffness_update_hz = float(len(update_steps) / duration_seconds) if duration_seconds > 0.0 else float(len(update_steps))
    raw_offdiag_values = [float(np.asarray(row.get("stiffness_matrix_raw_before_smoothing"), dtype=float)[0, 1]) for row in trace_rows if row.get("stiffness_matrix_raw_before_smoothing") is not None]
    after_offdiag_values = [float(np.asarray(row.get("stiffness_matrix_after_smoothing"), dtype=float)[0, 1]) for row in trace_rows if row.get("stiffness_matrix_after_smoothing") is not None]
    preserves_offdiag_coupling = any(
        abs(raw) > 1e-12 and abs(after) > 1e-12 for raw, after in zip(raw_offdiag_values, after_offdiag_values)
    )
    deployment_alignment = "stiffness_copilot_style" if (smoothing_config is not None and smoothing_config.update_rate_hz == 90.0 and smoothing_config.method == "log_spd_ema") else "legacy_or_debug"
    row: dict[str, Any] = {
        "controller_profile": controller_profile,
        "controller_id": controller_id or controller_profile,
        "episode_spec_id": episode_spec.episode_spec_id,
        "episode_id": int(episode_spec.episode_id),
        "episode_spec_schema_version": episode_spec.to_dict().get("episode_spec_schema_version"),
        "scene": episode_spec.scene,
        "setting_id": episode_spec.setting_id,
        "profile_name": episode_spec.profile_name,
        "contact_condition_name": episode_spec.contact_condition_name,
        "trajectory_source": EPISODE_TRAJECTORY_SOURCE_EPISODE_SPEC_REPLAY,
        "episode_spec_trajectory_source": episode_spec.trajectory_source,
        "trajectory_family": episode_spec.trajectory_family,
        "trajectory_family_id": int(episode_spec.trajectory_family_id),
        "trajectory_total_steps": int(episode_spec.total_steps),
        "actual_hole_xy": episode_spec.actual_hole_xy.tolist(),
        "actual_hole_position": episode_spec.actual_hole_position.tolist(),
        "hole_xy_offset": episode_spec.hole_xy_offset.tolist(),
        "hole_xy_offset_semantics": episode_spec.hole_xy_offset_semantics,
        "hole_xy_offset_distribution": episode_spec.hole_xy_offset_distribution,
        "trajectory_follows_randomized_hole": bool(episode_spec.trajectory_follows_randomized_hole),
        "contact_generation_parameters_fixed": bool(episode_spec.contact_generation_parameters_fixed),
        "fixed_contact_condition": _json_ready(episode_spec.fixed_contact_condition),
        "controller_spec": _controller_spec_to_dict(controller_spec, simulation_dt_seconds=simulation_dt_seconds) if controller_spec is not None else None,
        "controller_stiffness_matrix": None if controller_stiffness_matrix is None else np.asarray(controller_stiffness_matrix, dtype=float).tolist(),
        "controllers_yaml": None if controllers_yaml is None else str(controllers_yaml),
        "summary": _strip_full_trace(summary.to_dict()),
        "contact_detected": bool(summary.contact_detected),
        "hole_contact_detected": bool(summary.hole_contact_detected),
        "contact_onset_step": int(summary.contact_onset_step),
        "contact_count": _trace_contact_count(trace_rows),
        "contact_fraction": float(_trace_contact_count(trace_rows) / max(len(trace_rows), 1)),
        "max_normal_force": float(summary.max_normal_force),
        "max_tangential_force": float(summary.max_tangential_force),
        "p95_normal_force": force_quantiles["p95_normal_force"],
        "p99_normal_force": force_quantiles["p99_normal_force"],
        "final_depth": float(summary.final_depth),
        "final_lateral_error": float(summary.final_lateral_error),
        "depth_reached": bool(summary.depth_reached),
        "low_force_success": bool(summary.low_force_success),
        "torque_saturation_count": int(summary.torque_saturation_count),
        "termination_step": int(summary.steps),
        "total_steps_requested": int(episode_spec.total_steps),
        "reference_position_mode": TRACK_A_PAIRING_REFERENCE_POSITION_MODE,
        "contact_accounting_mode": TRACK_A_CONTACT_ACCOUNTING_MODE,
        "runtime_mode": runtime_mode,
        "trace_row_count": int(len(trace_rows)),
        "termination_reason": _episode_termination_reason(summary),
        "deployment_alignment": deployment_alignment,
        "target_stiffness_update_hz": None if smoothing_config is None or smoothing_config.update_rate_hz is None else float(smoothing_config.update_rate_hz),
        "achieved_stiffness_update_hz": achieved_stiffness_update_hz,
        "update_interval_step_distribution": update_interval_distribution,
        "stiffness_smoothing_method": None if smoothing_config is None else str(smoothing_config.method),
        "stiffness_smoothing_alpha": None if smoothing_config is None else float(smoothing_config.alpha),
        "stiffness_update_scheduler": "time_accumulator" if smoothing_config is not None and smoothing_config.update_rate_hz is not None else "fixed_period",
        "uses_full_matrix_smoothing": bool(smoothing_config is not None and smoothing_config.method == "log_spd_ema"),
        "preserves_offdiag_coupling": bool(preserves_offdiag_coupling),
        "trace_stats": {
            "max_position_error": float(trace_stats.max_position_error),
            "max_lateral_tracking_error": float(trace_stats.max_lateral_tracking_error),
            "max_vertical_tracking_error": float(trace_stats.max_vertical_tracking_error),
            "final_position_error": float(trace_stats.final_position_error),
            "final_actual_position": [trace_stats.final_actual_x, trace_stats.final_actual_y, trace_stats.final_actual_z],
            "final_target_position": [trace_stats.final_target_x, trace_stats.final_target_y, trace_stats.final_target_z],
        },
    }
    if full_trace is not None:
        row["full_trace"] = full_trace
    if policy_path is not None:
        row["policy_path"] = str(policy_path)
    if policy_metadata is not None:
        row["policy_metadata_summary"] = policy_metadata
    if smoothing_config is not None:
        row["stiffness_smoothing_config"] = {
            "enabled": bool(smoothing_config.enabled),
            "method": str(smoothing_config.method),
            "alpha": float(smoothing_config.alpha),
            "policy_update_period_steps": int(smoothing_config.policy_update_period_steps),
            "policy_update_period_steps_requested": None
            if smoothing_config.policy_update_period_steps is None
            else int(smoothing_config.policy_update_period_steps),
            "effective_policy_update_period_steps": int(
                smoothing_config.resolved_policy_update_period_steps(simulation_dt_seconds=simulation_dt_seconds)
            ),
            "effective_policy_update_period_seconds": float(1.0 / float(smoothing_config.update_rate_hz))
            if smoothing_config.update_rate_hz is not None
            else float(
                simulation_dt_seconds
                * smoothing_config.resolved_policy_update_period_steps(simulation_dt_seconds=simulation_dt_seconds)
            ),
            "update_rate_hz": None if smoothing_config.update_rate_hz is None else float(smoothing_config.update_rate_hz),
            "stiffness_update_hz_target": None
            if smoothing_config.update_rate_hz is None
            else float(smoothing_config.update_rate_hz),
            "scheduler": "time_accumulator" if smoothing_config.update_rate_hz is not None else "fixed_period",
            "hold_between_updates": bool(smoothing_config.hold_between_updates),
            "target_kind": str(smoothing_config.target_kind),
        }
    return row


def run_track_a_fixed_controller_episode(
    *,
    episode_spec: EpisodeSpec,
    controller_profile: str,
    controller_role: str,
    gain_config_path: Path,
    config: RolloutConfig,
    seed: int,
    position_stiffness_matrix: np.ndarray | None = None,
    position_stiffness_matrix_source: str | None = None,
    stiffness_smoothing: StiffnessCommandSmoothingConfig | None = None,
    include_full_trace: bool = False,
) -> dict[str, Any]:
    scene_path, scene_config = scene_for_rollout(config.config_path, perturbation_from_episode_spec(episode_spec))
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
    target_rotation = site_rotation(data, model.site(config.site_name).id)
    controller_spec = _make_controller_spec(
        gain_config_path,
        requested_profile=controller_profile,
        controller_role=controller_role,
        position_stiffness_matrix=position_stiffness_matrix,
        position_stiffness_matrix_source=position_stiffness_matrix_source,
    )
    stiffness_smoothing = resolve_deployment_stiffness_smoothing_config(stiffness_smoothing)

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
    final_step = 0
    rows: list[dict[str, object]] = []
    normal_forces: list[float] = []
    trace_stats = TrackATraceStats()

    from stiffness_copilot_mujoco.controllers.stiffness_command_smoothing import StiffnessCommandSmoother

    smoother = StiffnessCommandSmoother(stiffness_smoothing, simulation_dt_seconds=simulation_dt_seconds)
    target_stiffness = np.asarray(controller_spec.position_stiffness_matrix, dtype=float) if controller_spec.position_stiffness_matrix is not None else np.diag(np.asarray(controller_spec.gains.position_stiffness, dtype=float))

    for step in range(episode_spec.total_steps + 1):
        phase = episode_spec.phase_name_at_step(step)
        target_position = episode_spec.target_position_at_step(step, reference_position=hole_center)
        smoothing_step = smoother.apply(step=step, target_matrix=target_stiffness)
        command = task_space_impedance_torque(
            model,
            data,
            site_name=config.site_name,
            target_position=target_position,
            target_rotation=target_rotation,
            arm_ids=arm_ids,
            gains=controller_spec.gains,
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
        normal_forces.append(float(contact.normal_force))
        contact_force_world = extract_net_peg_hole_contact_force_world(ContactQuery(model=model, data=data, task_ids=task_ids))
        position_error = float(np.linalg.norm(command.position_error))
        rows.append(
            {
                "episode_id": int(episode_spec.episode_id),
                "episode_spec_id": episode_spec.episode_spec_id,
                "trajectory_family": episode_spec.trajectory_family,
                "trajectory_family_id": int(episode_spec.trajectory_family_id),
                "controller_id": controller_profile,
                "controller_role": controller_role,
                "seed": int(seed),
                "step": int(step),
                "time": float(data.time),
                "phase": phase,
                "target_x": float(target_position[0]),
                "target_y": float(target_position[1]),
                "target_z": float(target_position[2]),
                "position_error": position_error,
                "lateral_error": float(lateral_error(data, task_ids)),
                "depth": float(insertion_depth(data, task_ids)),
                "normal_force": float(contact.normal_force),
                "tangential_force": float(contact.tangential_force),
                "contact_force_world": [float(v) for v in np.asarray(contact_force_world, dtype=float).reshape(-1)],
                "penetration_depth": float(contact.penetration_depth),
                "torque_saturated": bool(saturated),
                "in_contact": bool(contact.in_contact),
                "contact_state": bool(contact.in_contact),
                "stiffness_matrix_raw_before_smoothing": np.asarray(
                    smoothing_step.raw_matrix_before_smoothing if smoothing_step.raw_matrix_before_smoothing is not None else target_stiffness,
                    dtype=float,
                ).reshape(3, 3).tolist(),
                "stiffness_matrix_after_smoothing": np.asarray(
                    smoothing_step.smoothed_matrix if smoothing_step.smoothed_matrix is not None else smoothing_step.command_matrix,
                    dtype=float,
                ).reshape(3, 3).tolist(),
                "stiffness_update_hz_target": smoothing_step.stiffness_update_hz_target,
                "stiffness_update_index": smoothing_step.stiffness_update_index,
                "stiffness_update_interval_steps": smoothing_step.stiffness_update_interval_steps,
                "stiffness_update_interval_seconds": smoothing_step.stiffness_update_interval_seconds,
                "steps_since_last_stiffness_refresh": smoothing_step.steps_since_last_stiffness_refresh,
                "stiffness_smoothing_scheduler": smoothing_step.scheduler,
                "stiffness_update_scheduler": smoothing_step.scheduler,
                "stiffness_refreshed_this_step": bool(smoothing_step.update_applied),
                "offdiag_xy_before_smoothing": None
                if smoothing_step.raw_matrix_before_smoothing is None
                else float(np.asarray(smoothing_step.raw_matrix_before_smoothing, dtype=float)[0, 1]),
                "offdiag_xy_after_smoothing": None
                if smoothing_step.smoothed_matrix is None
                else float(np.asarray(smoothing_step.smoothed_matrix, dtype=float)[0, 1]),
                "smoothing_update_applied": bool(smoothing_step.update_applied),
                "smoothing_hold_applied": bool(smoothing_step.hold_applied),
                "stiffness_matrix_command": np.asarray(smoothing_step.command_matrix, dtype=float).reshape(3, 3).tolist(),
                "stiffness_matrix_target": np.asarray(target_stiffness, dtype=float).reshape(3, 3).tolist(),
            }
        )
        actual_position = site_position(data, model.site(config.site_name).id)
        trace_stats.max_position_error = max(trace_stats.max_position_error, position_error)
        trace_stats.max_lateral_tracking_error = max(
            trace_stats.max_lateral_tracking_error,
            float(np.linalg.norm(actual_position[:2] - target_position[:2])),
        )
        trace_stats.max_vertical_tracking_error = max(
            trace_stats.max_vertical_tracking_error,
            abs(float(actual_position[2] - target_position[2])),
        )
        trace_stats.final_position_error = position_error
        trace_stats.final_target_x = float(target_position[0])
        trace_stats.final_target_y = float(target_position[1])
        trace_stats.final_target_z = float(target_position[2])
        trace_stats.steps = int(step)
        final_step = step
        if phase == "done":
            break
        set_arm_torque_ctrl(model, data, arm_ids, commanded_torque)
        mujoco.mj_step(model, data)

    final_depth = insertion_depth(data, task_ids)
    final_lateral = lateral_error(data, task_ids)
    mean_normal_force_contact = normal_force_sum / normal_force_count if normal_force_count else 0.0
    depth_reached = bool(final_depth >= 0.95 * config.insert_depth)
    low_force_success = bool(depth_reached and max_normal_force <= ForceMetricThresholds().low_force_success_threshold)
    summary = EpisodeSummary(
        baseline=controller_role,
        profile=controller_profile,
        seed=seed,
        episode_id=episode_spec.episode_id,
        xy_offset=(0.0, 0.0),
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
        smoothing_enabled=bool(stiffness_smoothing.enabled),
        smoothing_method=str(stiffness_smoothing.method),
        smoothing_alpha=float(stiffness_smoothing.alpha),
        policy_update_period_steps=int(stiffness_smoothing.policy_update_period_steps),
        hold_between_updates=bool(stiffness_smoothing.hold_between_updates),
        smoothing_target_kind=str(stiffness_smoothing.target_kind),
        stiffness_before_smoothing_summary=dict(smoother.summary_dict()["stiffness_before_smoothing_summary"]),
        stiffness_after_smoothing_summary=dict(smoother.summary_dict()["stiffness_after_smoothing_summary"]),
        perturbation=perturbation_from_episode_spec(episode_spec).to_dict(),
    )
    return _shared_episode_metadata(
        controller_profile=controller_profile,
        controller_id=controller_profile,
        controller_stiffness_matrix=controller_spec.position_stiffness_matrix if controller_spec.position_stiffness_matrix is not None else np.diag(np.asarray(controller_spec.gains.position_stiffness, dtype=float)),
        controllers_yaml=Path(position_stiffness_matrix_source) if position_stiffness_matrix_source else None,
        episode_spec=episode_spec,
        simulation_dt_seconds=simulation_dt_seconds,
        controller_spec=controller_spec,
        summary=summary,
        normal_forces=normal_forces,
        trace_rows=rows,
        trace_stats=trace_stats,
        runtime_mode="fixed_controller_replay",
        full_trace=rows if include_full_trace else None,
    )


def run_track_a_residual_episode(
    *,
    episode_spec: EpisodeSpec,
    policy_path: Path,
    reference_controller_id: str,
    reference_stiffness_matrix: np.ndarray,
    controllers_yaml: Path = DEFAULT_TRACK_A_CONTROLLERS_YAML,
    config: RolloutConfig,
    seed: int,
    camera_name: str = "eye_in_hand_rgb",
    image_width: int = 128,
    image_height: int = 128,
    renderer_mode: str = "native",
    residual_scale: float = 1.0,
    stiffness_smoothing: StiffnessCommandSmoothingConfig | None = None,
    include_full_trace: bool = False,
) -> dict[str, Any]:
    policy = load_image_only_residual_bc_policy(policy_path)
    reference_controller = get_track_a_controller(reference_controller_id, controllers_yaml=controllers_yaml)
    reference_matrix = np.asarray(reference_stiffness_matrix, dtype=float)
    if reference_matrix.shape != (3, 3):
        raise ValueError(f"reference_stiffness_matrix must have shape (3, 3), got {reference_matrix.shape}.")
    if not np.allclose(reference_matrix, reference_controller.position_stiffness_matrix, atol=1e-9, rtol=0.0):
        raise ValueError(
            "reference_stiffness_matrix does not match the registry entry for reference_controller_id "
            f"{reference_controller_id!r}."
        )
    result = run_image_only_residual_bc_episode(
        policy=policy,
        policy_path=policy_path,
        seed=seed,
        xy_offset=np.zeros(2, dtype=float),
        config=config,
        episode_id=int(episode_spec.episode_id),
        reference_controller_id=reference_controller_id,
        reference_stiffness_matrix=reference_matrix,
        perturbation=None,
        camera_name=camera_name,
        image_width=image_width,
        image_height=image_height,
        renderer_mode=renderer_mode,
        residual_scale=residual_scale,
        stiffness_smoothing=stiffness_smoothing,
        episode_spec=episode_spec,
    )
    rows = [dict(row) for row in result.full_trace]
    normal_forces = [float(row.get("normal_force", 0.0)) for row in rows]
    simulation_dt_seconds = float(load_scene_config(config.config_path).get("physics", {}).get("timestep", 0.002))
    controller_spec = _make_controller_spec(
        config.gain_config_path,
        requested_profile=reference_controller_id,
        controller_role="residual_policy",
        position_stiffness_matrix=reference_matrix,
        position_stiffness_matrix_source=str(controllers_yaml),
    )
    trace_stats = TrackATraceStats(
        max_position_error=float(max((float(row.get("position_error", 0.0)) for row in rows), default=0.0)),
        max_lateral_tracking_error=float(max((abs(float(row.get("lateral_error", 0.0))) for row in rows), default=0.0)),
        max_vertical_tracking_error=float(max((abs(float(row.get("target_z", 0.0)) - float(row.get("actual_z", 0.0))) for row in rows), default=0.0)),
        final_position_error=float(rows[-1].get("position_error", 0.0)) if rows else 0.0,
        final_actual_x=float(rows[-1].get("actual_x", 0.0)) if rows else 0.0,
        final_actual_y=float(rows[-1].get("actual_y", 0.0)) if rows else 0.0,
        final_actual_z=float(rows[-1].get("actual_z", 0.0)) if rows else 0.0,
        final_target_x=float(rows[-1].get("target_x", 0.0)) if rows else 0.0,
        final_target_y=float(rows[-1].get("target_y", 0.0)) if rows else 0.0,
        final_target_z=float(rows[-1].get("target_z", 0.0)) if rows else 0.0,
        steps=int(result.steps),
    )
    payload = _shared_episode_metadata(
        controller_profile=reference_controller_id,
        controller_id=reference_controller_id,
        controller_stiffness_matrix=reference_matrix,
        controllers_yaml=controllers_yaml,
        episode_spec=episode_spec,
        simulation_dt_seconds=simulation_dt_seconds,
        controller_spec=controller_spec,
        policy_path=policy_path,
        policy_metadata=policy.metadata,
        smoothing_config=stiffness_smoothing,
        summary=result,
        normal_forces=normal_forces,
        trace_rows=rows,
        trace_stats=trace_stats,
        runtime_mode="residual_policy_replay",
        full_trace=rows if include_full_trace else None,
    )
    payload["residual_scale"] = float(residual_scale)
    if isinstance(payload.get("policy_metadata"), dict):
        payload["policy_metadata"]["residual_scale"] = float(residual_scale)
    return payload


def summarize_policy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    base_spec = metadata.get("base_stiffness_spec") or {}
    return {
        "schema_version": metadata.get("schema_version"),
        "method_name": metadata.get("method_name"),
        "input_mode": metadata.get("input_mode"),
        "output_space": metadata.get("output_space"),
        "output_dim": metadata.get("output_dim"),
        "is_residual_policy": metadata.get("is_residual_policy"),
        "is_full_stiffness_policy": metadata.get("is_full_stiffness_policy"),
        "stiffness_representation": metadata.get("stiffness_representation"),
        "residual_parameterization": metadata.get("residual_parameterization"),
        "residual_affects": metadata.get("residual_affects"),
        "residual_unaffected": metadata.get("residual_unaffected"),
        "smoothing_required": metadata.get("smoothing_required"),
        "renderer_mode": metadata.get("renderer_mode"),
        "fallback_used": metadata.get("fallback_used"),
        "residual_bound": metadata.get("residual_bound"),
        "eye_in_hand_camera_pose_version": metadata.get("eye_in_hand_camera_pose_version"),
        "eye_in_hand_camera_canonical": metadata.get("eye_in_hand_camera_canonical"),
        "eye_in_hand_camera_name": metadata.get("eye_in_hand_camera_name"),
        "eye_in_hand_camera_attachment_parent": metadata.get("eye_in_hand_camera_attachment_parent"),
        "eye_in_hand_camera_mount_type": metadata.get("eye_in_hand_camera_mount_type"),
        "contact_condition_name": metadata.get("contact_condition_name"),
        "fixed_clearance_delta": metadata.get("fixed_clearance_delta"),
        "fixed_friction_scale": metadata.get("fixed_friction_scale"),
        "fixed_peg_tilt_x": metadata.get("fixed_peg_tilt_x"),
        "fixed_peg_tilt_y": metadata.get("fixed_peg_tilt_y"),
        "fixed_teleop_noise_xy_amplitude": metadata.get("fixed_teleop_noise_xy_amplitude"),
        "fixed_teleop_noise_cycles": metadata.get("fixed_teleop_noise_cycles"),
        "fixed_teleop_noise_phase_x": metadata.get("fixed_teleop_noise_phase_x"),
        "fixed_teleop_noise_phase_y": metadata.get("fixed_teleop_noise_phase_y"),
        "dataset_path": metadata.get("dataset_path") or metadata.get("dataset"),
        "controllers_yaml": metadata.get("controllers_yaml"),
        "reference_controller_id": metadata.get("reference_controller_id"),
        "reference_stiffness_matrix": metadata.get("reference_stiffness_matrix"),
        "collection_controller_id": metadata.get("collection_controller_id"),
        "collection_stiffness_matrix": metadata.get("collection_stiffness_matrix"),
        "base_stiffness_spec": base_spec,
        "base_stiffness_base_matrix": base_spec.get("base_matrix"),
        "base_stiffness_active_group_names": base_spec.get("active_group_names"),
        "base_stiffness_active_groups": base_spec.get("active_groups"),
        "base_stiffness_residual_bounds": base_spec.get("residual_bounds"),
        "encoder_seed": metadata.get("encoder_seed"),
        "encoder_conv_channels": metadata.get("encoder_conv_channels"),
    }


def validate_track_a_v2_policy_metadata(metadata: dict[str, Any]) -> list[str]:
    hard_failures: list[str] = []
    expected_contract: dict[str, Any] | None = None
    required_exact_values = {
        "schema_version": "vision_residual_bc_policy_v2",
        "method_name": "image_only_residual_bc",
        "input_mode": "image_only",
        "uses_task_state_input": False,
        "uses_contact_force_input": False,
        "uses_clearance_input": False,
        "uses_trajectory_phase_input": False,
        "is_residual_policy": True,
        "is_full_stiffness_policy": False,
        "renderer_mode": "mujoco_native",
        "fallback_used": False,
        "eye_in_hand_camera_pose_version": CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION,
        "eye_in_hand_camera_canonical": True,
        "eye_in_hand_camera_name": "eye_in_hand_rgb",
        "eye_in_hand_camera_attachment_parent": CANONICAL_EYE_IN_HAND_CAMERA_ATTACHMENT_PARENT,
        "eye_in_hand_camera_mount_type": CANONICAL_EYE_IN_HAND_CAMERA_MOUNT_TYPE,
        "stiffness_representation": TRACK_A_STIFFNESS_REPRESENTATION,
        "smoothing_required": TRACK_A_SMOOTHING_REQUIRED,
    }
    for key in ("controllers_yaml", "reference_controller_id", "reference_stiffness_matrix", "collection_controller_id", "collection_stiffness_matrix"):
        if key not in metadata:
            hard_failures.append(f"policy metadata field {key!r} is required")
    base_spec = metadata.get("base_stiffness_spec")
    if not isinstance(base_spec, dict):
        hard_failures.append("policy metadata field 'base_stiffness_spec' must be a mapping")
    else:
        try:
            base_spec_obj = BaseStiffnessSpec.from_metadata(base_spec)
            expected_contract = describe_residual_policy_contract(base_spec_obj)
        except Exception as exc:
            hard_failures.append(f"policy metadata field 'base_stiffness_spec' is invalid: {exc}")
        else:
            base_matrix = np.asarray(base_spec_obj.base_matrix, dtype=float)
            if base_matrix.shape != (3, 3):
                hard_failures.append(
                    f"policy metadata field 'base_stiffness_spec.base_matrix' must have shape (3, 3), observed {base_matrix.shape}"
                )
            residual_bounds = np.asarray(base_spec_obj.residual_bounds, dtype=float)
            if residual_bounds.shape != (int(expected_contract["output_dim"]),):
                hard_failures.append(
                    "policy metadata field 'base_stiffness_spec.residual_bounds' must have shape "
                    f"({int(expected_contract['output_dim'])},), observed {residual_bounds.shape}"
                )
            if expected_contract is not None:
                required_exact_values.update(
                    {
                        "output_space": expected_contract["output_space"],
                        "output_dim": expected_contract["output_dim"],
                        "residual_parameterization": expected_contract["residual_parameterization"],
                        "residual_affects": expected_contract["residual_affects"],
                        "residual_unaffected": expected_contract["residual_unaffected"],
                    }
                )
    for key, expected in required_exact_values.items():
        observed = metadata.get(key)
        if observed != expected:
            hard_failures.append(f"policy metadata field {key!r} must be {expected!r}, observed {observed!r}")
    reference_matrix = np.asarray(metadata.get("reference_stiffness_matrix"), dtype=float)
    collection_matrix = np.asarray(metadata.get("collection_stiffness_matrix"), dtype=float)
    if reference_matrix.shape != (3, 3):
        hard_failures.append(
            f"policy metadata field 'reference_stiffness_matrix' must have shape (3, 3), observed {reference_matrix.shape}"
        )
    if collection_matrix.shape != (3, 3):
        hard_failures.append(
            f"policy metadata field 'collection_stiffness_matrix' must have shape (3, 3), observed {collection_matrix.shape}"
        )
    if base_spec and reference_matrix.shape == (3, 3):
        base_matrix = np.asarray(base_spec.get("base_matrix"), dtype=float)
        if base_matrix.shape == (3, 3) and not np.allclose(base_matrix, reference_matrix, atol=1e-9, rtol=0.0):
            hard_failures.append("reference_stiffness_matrix must match base_stiffness_spec.base_matrix")
    if reference_matrix.shape == (3, 3) and collection_matrix.shape == (3, 3):
        if not np.allclose(reference_matrix, collection_matrix, atol=1e-9, rtol=0.0):
            hard_failures.append("collection_stiffness_matrix must match reference_stiffness_matrix")
    if float(metadata.get("residual_bound", 0.0) or 0.0) != 0.35:
        hard_failures.append(
            f"policy metadata field 'residual_bound' must be 0.35, observed {metadata.get('residual_bound')!r}"
        )
    pose = metadata.get("eye_in_hand_camera_pose")
    if not isinstance(pose, dict):
        hard_failures.append("policy metadata field 'eye_in_hand_camera_pose' must be a mapping")
    else:
        expected_pose = canonical_eye_in_hand_camera_pose(str(metadata.get("eye_in_hand_camera_name") or "eye_in_hand_rgb"))
        for key in ("camera_name", "attachment_parent", "mount_type"):
            if pose.get(key) != expected_pose[key]:
                hard_failures.append(
                    f"policy metadata field 'eye_in_hand_camera_pose.{key}' must be {expected_pose[key]!r}, observed {pose.get(key)!r}"
                )
        for key in ("pos", "forward", "up"):
            observed = np.asarray(pose.get(key), dtype=float)
            expected = np.asarray(expected_pose[key], dtype=float)
            if observed.shape != expected.shape or not np.allclose(observed, expected, atol=1e-12, rtol=0.0):
                hard_failures.append(
                    f"policy metadata field 'eye_in_hand_camera_pose.{key}' must be {expected.tolist()!r}, observed {observed.tolist()!r}"
                )
        if not np.isclose(float(pose.get("fovy", 0.0)), float(expected_pose["fovy"]), atol=1e-12, rtol=0.0):
            hard_failures.append(
                f"policy metadata field 'eye_in_hand_camera_pose.fovy' must be {expected_pose['fovy']!r}, observed {pose.get('fovy')!r}"
            )
    return hard_failures


__all__ = [
    "TRACK_A_BASELINE_CONTROLLER_PROFILE",
    "TRACK_A_COLLECTION_CONTROLLER_PROFILE",
    "TRACK_A_CONTACT_ACCOUNTING_MODE",
    "TRACK_A_PAIRING_REFERENCE_POSITION_MODE",
    "TRACK_A_RUNNER_VERSION",
    "TRACK_A_STIFFNESS_REPRESENTATION",
    "TRACK_A_RESIDUAL_PARAMETERIZATION",
    "TRACK_A_RESIDUAL_AFFECTS",
    "TRACK_A_RESIDUAL_UNAFFECTED",
    "TRACK_A_SMOOTHING_REQUIRED",
    "TrackATraceStats",
    "run_track_a_fixed_controller_episode",
    "run_track_a_residual_episode",
    "summarize_policy_metadata",
    "validate_track_a_v2_policy_metadata",
]
