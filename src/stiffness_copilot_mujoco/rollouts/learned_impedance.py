from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import ContactQuery, extract_contact_state
from stiffness_copilot_mujoco.controllers.impedance import TaskSpaceImpedanceGains, load_task_space_impedance_gains, task_space_impedance_torque
from stiffness_copilot_mujoco.controllers.stiffness_command_smoothing import (
    StiffnessCommandSmoothingConfig,
    summary_fields_from_smoothing_summary,
)
from stiffness_copilot_mujoco.evaluation.force_metrics import ForceMetricThresholds
from stiffness_copilot_mujoco.learning.supervised_policy import (
    SupervisedStiffnessPolicy,
    log_euclidean_ema_spd,
    scale_normalized_stiffness,
)
from stiffness_copilot_mujoco.learning.task_state import peg_hole_task_state
from stiffness_copilot_mujoco.franka_viewer import load_model
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
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    DEFAULT_GAIN_CONFIG,
    DEFAULT_TORQUE_CONFIG,
    EpisodeSummary,
    RolloutConfig,
    RolloutPerturbation,
    cleanup_runtime_scene,
    clip_torque,
    phase_for_step,
    scene_for_rollout,
    target_position_for_phase,
    target_position_with_teleop_noise,
)
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids


@dataclass(frozen=True, kw_only=True)
class LearnedEpisodeSummary(EpisodeSummary):
    policy_path: str
    k_min: float
    k_max: float
    mean_predicted_stiffness_eig_min: float
    mean_predicted_stiffness_eig_mid: float
    mean_predicted_stiffness_eig_max: float
    min_predicted_stiffness_eig: float
    max_predicted_stiffness_eig: float
    contact_onset_trace: list[dict[str, float | int | str | bool]]

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result.update(
            {
                "policy_path": self.policy_path,
                "k_min": self.k_min,
                "k_max": self.k_max,
                "mean_predicted_stiffness_eig_min": self.mean_predicted_stiffness_eig_min,
                "mean_predicted_stiffness_eig_mid": self.mean_predicted_stiffness_eig_mid,
                "mean_predicted_stiffness_eig_max": self.mean_predicted_stiffness_eig_max,
                "min_predicted_stiffness_eig": self.min_predicted_stiffness_eig,
                "max_predicted_stiffness_eig": self.max_predicted_stiffness_eig,
                "contact_onset_trace": self.contact_onset_trace,
            }
        )
        return result


def _contact_onset_trace(rows: list[dict[str, float | int | str | bool]], onset_step: int) -> list[dict[str, float | int | str | bool]]:
    if onset_step < 0:
        return []
    by_step = {int(row["step"]): row for row in rows}
    trace = []
    for offset in (-200, -100, -50, -10, 0, 10, 50, 100):
        step = onset_step + offset
        row = by_step.get(step)
        if row is not None:
            trace.append(dict(row, contact_onset_offset=offset))
    return trace


def run_learned_stiffness_episode(
    *,
    policy: SupervisedStiffnessPolicy,
    policy_path: Path,
    seed: int,
    xy_offset: np.ndarray,
    config: RolloutConfig = RolloutConfig(),
    episode_id: int = 0,
    base_gains: TaskSpaceImpedanceGains | None = None,
    base_profile: str = "stiff",
    k_min: float = 50.0,
    k_max: float = 200.0,
    perturbation: RolloutPerturbation | None = None,
    stiffness_smoothing_alpha: float = 0.2,
    stiffness_smoothing_eps: float = 1e-8,
    stiffness_smoothing_enabled: bool = True,
) -> LearnedEpisodeSummary:
    scene_path, scene_config = scene_for_rollout(config.config_path, perturbation)
    try:
        model = load_model(scene_path)
    finally:
        cleanup_runtime_scene(scene_path)
    data = mujoco.MjData(model)
    reset_from_config(model, data, scene_config)

    geometry = geometry_from_config(scene_config)
    task_ids = peg_hole_ids(model, segments=geometry.segments)
    arm_ids = panda_arm_ids(model)
    nullspace_target_qpos = arm_qpos(data, arm_ids)
    hole_center = hole_center_position(data, task_ids)
    target_rotation = site_rotation(data, model.site(config.site_name).id)
    if base_gains is None:
        loaded_profile, base_gains = load_task_space_impedance_gains(config.gain_config_path, base_profile)
        base_profile = loaded_profile

    xy = np.asarray(xy_offset, dtype=float)
    if xy.shape != (2,):
        raise ValueError(f"xy_offset must have shape (2,), got {xy.shape}.")

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
    predicted_eigvals: list[np.ndarray] = []
    high_stiffness = np.diag(np.asarray(base_gains.position_stiffness, dtype=float))
    smoothed_stiffness = high_stiffness.copy()
    trace_rows: list[dict[str, float | int | str | bool]] = []

    for step in range(config.total_steps + 1):
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
        task_state = peg_hole_task_state(data, task_ids, hole_clearance_delta=0.0)
        normalized_stiffness = policy.predict_normalized_matrix(task_state)
        raw_stiffness = scale_normalized_stiffness(normalized_stiffness, k_min=k_min, k_max=k_max)
        if stiffness_smoothing_enabled:
            smoothed_stiffness = log_euclidean_ema_spd(
                smoothed_stiffness,
                raw_stiffness,
                alpha=stiffness_smoothing_alpha,
                eps=stiffness_smoothing_eps,
            )
            position_stiffness = smoothed_stiffness
        else:
            position_stiffness = raw_stiffness
        contact = extract_contact_state(ContactQuery(model=model, data=data, task_ids=task_ids))
        predicted_eigvals.append(np.linalg.eigvalsh(position_stiffness))
        command = task_space_impedance_torque(
            model,
            data,
            site_name=config.site_name,
            target_position=target_position,
            target_rotation=target_rotation,
            arm_ids=arm_ids,
            gains=base_gains,
            position_stiffness_matrix=position_stiffness,
            nullspace_target_qpos=nullspace_target_qpos,
            clip_to_ctrlrange=False,
        )
        commanded_torque, saturated = clip_torque(model, arm_ids, command.torque)
        max_abs_torque = max(max_abs_torque, float(np.max(np.abs(commanded_torque))))
        torque_saturation_count += int(saturated)

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
        eigvals_step = predicted_eigvals[-1]
        trace_rows.append(
            {
                "step": step,
                "phase": phase,
                "depth": insertion_depth(data, task_ids),
                "lateral_error": lateral_error(data, task_ids),
                "contact": bool(contact.in_contact),
                "normal_force": float(contact.normal_force),
                "k_eig_min": float(eigvals_step[0]),
                "k_eig_mid": float(eigvals_step[1]),
                "k_eig_max": float(eigvals_step[2]),
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
    eigvals = np.vstack(predicted_eigvals)
    mean_eigvals = np.mean(eigvals, axis=0)
    smoothing_fields = summary_fields_from_smoothing_summary(
        {
            "config": StiffnessCommandSmoothingConfig(
                enabled=bool(stiffness_smoothing_enabled),
                method="log_spd_ema",
                alpha=float(stiffness_smoothing_alpha),
                policy_update_period_steps=1,
                hold_between_updates=False,
            ).to_dict(),
            "stiffness_before_smoothing_summary": {
                "count": 0,
                "diag_mean": [0.0, 0.0, 0.0],
                "diag_min": [0.0, 0.0, 0.0],
                "diag_max": [0.0, 0.0, 0.0],
            },
            "stiffness_after_smoothing_summary": {
                "count": 0,
                "diag_mean": [0.0, 0.0, 0.0],
                "diag_min": [0.0, 0.0, 0.0],
                "diag_max": [0.0, 0.0, 0.0],
            },
        }
    )
    return LearnedEpisodeSummary(
        baseline="learned",
        profile=base_profile,
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
        **smoothing_fields,
        perturbation=perturbation.to_dict() if perturbation is not None else None,
        policy_path=str(policy_path),
        k_min=k_min,
        k_max=k_max,
        mean_predicted_stiffness_eig_min=float(mean_eigvals[0]),
        mean_predicted_stiffness_eig_mid=float(mean_eigvals[1]),
        mean_predicted_stiffness_eig_max=float(mean_eigvals[2]),
        min_predicted_stiffness_eig=float(np.min(eigvals)),
        max_predicted_stiffness_eig=float(np.max(eigvals)),
        contact_onset_trace=_contact_onset_trace(trace_rows, contact_onset_step),
    )


__all__ = ["LearnedEpisodeSummary", "run_learned_stiffness_episode", "DEFAULT_GAIN_CONFIG", "DEFAULT_TORQUE_CONFIG"]
