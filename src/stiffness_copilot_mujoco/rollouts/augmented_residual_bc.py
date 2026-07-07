from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import ContactQuery, extract_contact_state
from stiffness_copilot_mujoco.controllers.impedance import (
    TRACK_A_BASELINE_CONTROLLER_PROFILE,
    load_task_space_impedance_gains,
    task_space_impedance_torque,
)
from stiffness_copilot_mujoco.controllers.stiffness_command_smoothing import disabled_smoothing_summary_fields
from stiffness_copilot_mujoco.evaluation.force_metrics import ForceMetricThresholds
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.learning.augmented_residual_stiffness import (
    AugmentedObservationBuilder,
    AugmentedResidualSPDStiffnessPolicy,
)
from stiffness_copilot_mujoco.learning.residual_stiffness import ResidualSPDStiffnessPolicy
from stiffness_copilot_mujoco.learning.task_state import peg_hole_task_state
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
    scene_for_rollout,
    target_position_for_phase,
    target_position_with_teleop_noise,
    target_rotation_with_peg_tilt,
)
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids


@dataclass(frozen=True, kw_only=True)
class AugmentedResidualBCEpisodeSummary(EpisodeSummary):
    bc_policy_path: str
    augmented_policy_path: str
    mean_augmented_delta_norm: float
    max_augmented_delta_norm: float
    mean_bc_deviation_from_v1: float
    max_bc_deviation_from_v1: float
    mean_abs_group_delta: tuple[float, ...]
    mean_abs_theta_delta: tuple[float, ...]
    mean_policy_theta: tuple[float, ...]
    mean_stiffness_eig: tuple[float, float, float]
    min_stiffness_eig: float
    max_stiffness_eig: float
    full_trace: list[dict]

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result.update(
            {
                "bc_policy_path": self.bc_policy_path,
                "augmented_policy_path": self.augmented_policy_path,
                "mean_augmented_delta_norm": self.mean_augmented_delta_norm,
                "max_augmented_delta_norm": self.max_augmented_delta_norm,
                "mean_bc_deviation_from_v1": self.mean_bc_deviation_from_v1,
                "max_bc_deviation_from_v1": self.max_bc_deviation_from_v1,
                "mean_abs_group_delta": list(self.mean_abs_group_delta),
                "mean_abs_theta_delta": list(self.mean_abs_theta_delta),
                "mean_policy_theta": list(self.mean_policy_theta),
                "mean_stiffness_eig": list(self.mean_stiffness_eig),
                "min_stiffness_eig": self.min_stiffness_eig,
                "max_stiffness_eig": self.max_stiffness_eig,
                "full_trace": self.full_trace,
            }
        )
        return result


def run_augmented_residual_bc_episode(
    *,
    policy: AugmentedResidualSPDStiffnessPolicy,
    policy_path: Path,
    bc_policy: ResidualSPDStiffnessPolicy,
    bc_policy_path: Path,
    seed: int,
    xy_offset: np.ndarray,
    config: RolloutConfig = RolloutConfig(),
    episode_id: int = 0,
    base_profile: str = TRACK_A_BASELINE_CONTROLLER_PROFILE,
    perturbation: RolloutPerturbation | None = None,
) -> AugmentedResidualBCEpisodeSummary:
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
    target_rotation = target_rotation_with_peg_tilt(site_rotation(data, model.site(config.site_name).id), perturbation)
    loaded_profile, gains = load_task_space_impedance_gains(config.gain_config_path or DEFAULT_GAIN_CONFIG, base_profile)
    base_profile = loaded_profile
    xy = np.asarray(xy_offset, dtype=float)
    builder = AugmentedObservationBuilder(
        task_state_dim=int(policy.metadata.get("source_task_state_dim", 6)),
        residual_dim=len(policy.base_spec.active_groups),
        history_steps=policy.history_steps,
    )

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
    group_deltas: list[np.ndarray] = []
    theta_deltas: list[np.ndarray] = []
    theta_values: list[np.ndarray] = []
    stiffness_eigs: list[np.ndarray] = []
    bc_deviations: list[float] = []
    trace_rows: list[dict] = []

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
        target_position = target_position_with_teleop_noise(target_position, step=step, config=config, perturbation=perturbation)
        task_state = peg_hole_task_state(data, task_ids, hole_clearance_delta=0.0)
        contact = extract_contact_state(ContactQuery(model=model, data=data, task_ids=task_ids))
        augmented_obs = builder.observe(
            task_state=task_state,
            normal_force=contact.normal_force,
            contact=contact.in_contact,
        )
        position_stiffness, theta, theta_delta, group_delta = policy.predict(augmented_obs)
        bc_group_delta = bc_policy.predict(task_state)[3]
        bc_deviation = float(np.linalg.norm(group_delta - bc_group_delta))
        command = task_space_impedance_torque(
            model,
            data,
            site_name=config.site_name,
            target_position=target_position,
            target_rotation=target_rotation,
            arm_ids=arm_ids,
            gains=gains,
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
        group_deltas.append(group_delta)
        theta_deltas.append(theta_delta)
        theta_values.append(theta)
        bc_deviations.append(bc_deviation)
        eig = np.linalg.eigvalsh(position_stiffness)
        stiffness_eigs.append(eig)
        trace_rows.append(
            {
                "step": step,
                "phase": phase,
                "depth": insertion_depth(data, task_ids),
                "lateral_error": lateral_error(data, task_ids),
                "contact": bool(contact.in_contact),
                "normal_force": float(contact.normal_force),
                "delta_norm": float(np.linalg.norm(theta_delta)),
                "group_delta": [float(v) for v in group_delta],
                "bc_group_delta": [float(v) for v in bc_group_delta],
                "bc_deviation_from_v1": bc_deviation,
                "theta_delta": [float(v) for v in theta_delta],
                "theta": [float(v) for v in theta],
                "k_eig_min": float(eig[0]),
                "k_eig_mid": float(eig[1]),
                "k_eig_max": float(eig[2]),
                "torque_saturated": bool(saturated),
            }
        )
        builder.update(task_state=task_state, normal_force=contact.normal_force)
        builder.set_previous_residual(group_delta)
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
    group_array = np.vstack(group_deltas)
    theta_delta_array = np.vstack(theta_deltas)
    theta_array = np.vstack(theta_values)
    eig_array = np.vstack(stiffness_eigs)
    delta_norms = np.linalg.norm(theta_delta_array, axis=1)
    bc_dev_array = np.asarray(bc_deviations, dtype=float)
    return AugmentedResidualBCEpisodeSummary(
        baseline="augmented_residual_bc",
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
        **disabled_smoothing_summary_fields(),
        perturbation=perturbation.to_dict() if perturbation is not None else None,
        bc_policy_path=str(bc_policy_path),
        augmented_policy_path=str(policy_path),
        mean_augmented_delta_norm=float(np.mean(delta_norms)),
        max_augmented_delta_norm=float(np.max(delta_norms)),
        mean_bc_deviation_from_v1=float(np.mean(bc_dev_array)),
        max_bc_deviation_from_v1=float(np.max(bc_dev_array)),
        mean_abs_group_delta=tuple(float(v) for v in np.mean(np.abs(group_array), axis=0)),
        mean_abs_theta_delta=tuple(float(v) for v in np.mean(np.abs(theta_delta_array), axis=0)),
        mean_policy_theta=tuple(float(v) for v in np.mean(theta_array, axis=0)),
        mean_stiffness_eig=tuple(float(v) for v in np.mean(eig_array, axis=0)),
        min_stiffness_eig=float(np.min(eig_array)),
        max_stiffness_eig=float(np.max(eig_array)),
        full_trace=trace_rows,
    )


__all__ = ["AugmentedResidualBCEpisodeSummary", "run_augmented_residual_bc_episode"]
