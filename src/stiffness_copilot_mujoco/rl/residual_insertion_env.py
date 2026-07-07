from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import ContactQuery, extract_contact_state
from stiffness_copilot_mujoco.controllers.impedance import (
    TRACK_A_BASELINE_CONTROLLER_PROFILE,
    load_task_space_impedance_gains,
    task_space_impedance_torque,
)
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.learning.residual_rrl import ResidualRRLPolicy
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec, ResidualSPDStiffnessPolicy
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
from stiffness_copilot_mujoco.robustness import apply_perturbation_to_scene_config
from stiffness_copilot_mujoco.rollout_observation import reset_from_config
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    DEFAULT_GAIN_CONFIG,
    RolloutConfig,
    RolloutPerturbation,
    clip_torque,
    phase_for_step,
    target_position_for_phase,
    target_position_with_teleop_noise,
    target_rotation_with_peg_tilt,
)
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
from stiffness_copilot_mujoco.sim.scene import ROOT, render_config_to_file
from stiffness_copilot_mujoco.metrics.task_metrics import load_scene_config


@dataclass(frozen=True)
class ResidualRRLRewardConfig:
    progress_weight: float = 8.0
    force_weight: float = 0.002
    force_clip: float = 500.0
    force_tail_weight: float = 0.0
    safe_force_threshold: float = 200.0
    tail_force_threshold: float = 500.0
    tail_quadratic_weight: float = 1e-5
    catastrophic_force_threshold: float = 1000.0
    catastrophic_termination_threshold: float = 30000.0
    catastrophic_penalty: float = 500.0
    lateral_weight: float = 15.0
    torque_saturation_weight: float = 0.05
    bc_deviation_weight: float = 5.0
    action_rate_weight: float = 0.02
    success_bonus: float = 50.0
    failure_penalty: float = 50.0
    low_force_success_bonus: float = 20.0
    low_force_success_threshold: float = 200.0
    progress_clip: float = 0.001
    max_torque_saturation_count: int = 500

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ResidualInsertionEnv:
    def __init__(
        self,
        *,
        scene_config: Path,
        gain_config: Path = DEFAULT_GAIN_CONFIG,
        base_profile: str = TRACK_A_BASELINE_CONTROLLER_PROFILE,
        bc_policy: ResidualSPDStiffnessPolicy,
        base_spec: BaseStiffnessSpec | None = None,
        rollout_config: RolloutConfig | None = None,
        control_stride: int = 10,
        reward_config: ResidualRRLRewardConfig = ResidualRRLRewardConfig(),
    ) -> None:
        if control_stride <= 0:
            raise ValueError("control_stride must be positive.")
        self.scene_config = scene_config
        self.gain_config = gain_config
        self.base_profile = base_profile
        self.bc_policy = bc_policy
        self.base_spec = base_spec or bc_policy.base_spec
        self.rollout_config = rollout_config or RolloutConfig(config_path=scene_config, gain_config_path=gain_config)
        self.control_stride = int(control_stride)
        self.reward_config = reward_config
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.scene_runtime_config: dict[str, Any] | None = None
        self.geometry = None
        self.task_ids = None
        self.arm_ids = None
        self.nullspace_target_qpos = None
        self.hole_center = None
        self.target_rotation = None
        self.gains = None
        self.profile = base_profile
        self.perturbation: RolloutPerturbation | None = None
        self.step_index = 0
        self.episode_id = 0
        self.prev_depth = 0.0
        self.prev_action = np.zeros(len(self.base_spec.active_groups), dtype=float)
        self.max_episode_force = 0.0
        self.max_tangential_force = 0.0
        self.max_penetration_depth = 0.0
        self.max_abs_torque = 0.0
        self.torque_saturation_count = 0
        self.contact_detected = False
        self.hole_contact_detected = False
        self.contact_onset_step = -1
        self.depth_at_contact = 0.0
        self.normal_force_sum = 0.0
        self.normal_force_count = 0
        self.reward_sum = 0.0
        self.early_terminated = False
        self.early_termination_reason = ""

    @property
    def action_dim(self) -> int:
        return len(self.base_spec.active_groups)

    def reset(
        self,
        *,
        seed: int = 0,
        perturbation: RolloutPerturbation | None = None,
        episode_id: int = 0,
    ) -> np.ndarray:
        self.perturbation = perturbation
        self.episode_id = int(episode_id)
        base_scene_config = load_scene_config(self.rollout_config.config_path)
        scene_config = apply_perturbation_to_scene_config(base_scene_config, perturbation)
        scene_path = render_config_to_file(
            scene_config,
            ROOT / "models" / "scenes" / f"rrl_runtime_{os.getpid()}.xml",
        )
        self.scene_runtime_config = scene_config
        self.model = load_model(scene_path)
        try:
            scene_path.unlink()
        except OSError:
            pass
        self.data = mujoco.MjData(self.model)
        reset_from_config(self.model, self.data, scene_config)
        self.geometry = geometry_from_config(scene_config)
        self.task_ids = peg_hole_ids(self.model, segments=self.geometry.segments)
        self.arm_ids = panda_arm_ids(self.model)
        self.nullspace_target_qpos = arm_qpos(self.data, self.arm_ids)
        self.hole_center = hole_center_position(self.data, self.task_ids)
        self.target_rotation = target_rotation_with_peg_tilt(
            site_rotation(self.data, self.model.site(self.rollout_config.site_name).id),
            perturbation,
        )
        self.profile, self.gains = load_task_space_impedance_gains(self.rollout_config.gain_config_path, self.base_profile)
        self.step_index = 0
        self.prev_depth = insertion_depth(self.data, self.task_ids)
        self.prev_action = self.bc_policy.predict(peg_hole_task_state(self.data, self.task_ids, hole_clearance_delta=0.0))[3]
        self.max_episode_force = 0.0
        self.max_tangential_force = 0.0
        self.max_penetration_depth = 0.0
        self.max_abs_torque = 0.0
        self.torque_saturation_count = 0
        self.contact_detected = False
        self.hole_contact_detected = False
        self.contact_onset_step = -1
        self.depth_at_contact = 0.0
        self.normal_force_sum = 0.0
        self.normal_force_count = 0
        self.reward_sum = 0.0
        self.early_terminated = False
        self.early_termination_reason = ""
        _ = seed
        return self.observation()

    def observation(self) -> np.ndarray:
        self._require_state()
        return peg_hole_task_state(self.data, self.task_ids, hole_clearance_delta=0.0)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self._require_state()
        action = np.asarray(action, dtype=float)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},), got {action.shape}.")
        action = np.clip(action, -self.base_spec.residual_bounds, self.base_spec.residual_bounds)

        stride_max_force = 0.0
        stride_torque_saturation_count = 0
        done = False
        terminal_reason = ""
        final_orientation_error = 0.0

        for _ in range(self.control_stride):
            phase, phase_step, phase_length = phase_for_step(self.step_index, self.rollout_config)
            if phase == "done":
                done = True
                terminal_reason = "schedule_done"
                break

            target_position = target_position_for_phase(
                phase=phase,
                phase_step=phase_step,
                phase_length=phase_length,
                hole_center=self.hole_center,
                xy_offset=np.zeros(2, dtype=float),
                config=self.rollout_config,
            )
            target_position = target_position_with_teleop_noise(
                target_position,
                step=self.step_index,
                config=self.rollout_config,
                perturbation=self.perturbation,
            )
            position_stiffness, theta, theta_delta = self.base_spec.matrix_from_group_delta(action, clip=True)
            command = task_space_impedance_torque(
                self.model,
                self.data,
                site_name=self.rollout_config.site_name,
                target_position=target_position,
                target_rotation=self.target_rotation,
                arm_ids=self.arm_ids,
                gains=self.gains,
                position_stiffness_matrix=position_stiffness,
                nullspace_target_qpos=self.nullspace_target_qpos,
                clip_to_ctrlrange=False,
            )
            commanded_torque, saturated = clip_torque(self.model, self.arm_ids, command.torque)
            final_orientation_error = float(np.linalg.norm(command.orientation_error))
            self.max_abs_torque = max(self.max_abs_torque, float(np.max(np.abs(commanded_torque))))
            self.torque_saturation_count += int(saturated)
            stride_torque_saturation_count += int(saturated)

            contact = extract_contact_state(ContactQuery(model=self.model, data=self.data, task_ids=self.task_ids))
            if contact.in_contact:
                if not self.contact_detected:
                    self.contact_detected = True
                    self.hole_contact_detected = True
                    self.contact_onset_step = self.step_index
                    self.depth_at_contact = insertion_depth(self.data, self.task_ids)
                self.normal_force_sum += contact.normal_force
                self.normal_force_count += 1
            stride_max_force = max(stride_max_force, contact.normal_force)
            self.max_episode_force = max(self.max_episode_force, contact.normal_force)
            self.max_tangential_force = max(self.max_tangential_force, contact.tangential_force)
            self.max_penetration_depth = max(self.max_penetration_depth, contact.penetration_depth)

            if not np.all(np.isfinite(commanded_torque)):
                done = True
                terminal_reason = "nonfinite_torque"
                self.early_terminated = True
                self.early_termination_reason = terminal_reason
                break
            if self.max_episode_force >= self.reward_config.catastrophic_termination_threshold:
                done = True
                terminal_reason = "catastrophic_force"
                self.early_terminated = True
                self.early_termination_reason = terminal_reason
                break
            if self.torque_saturation_count >= self.reward_config.max_torque_saturation_count:
                done = True
                terminal_reason = "torque_saturation_limit"
                self.early_terminated = True
                self.early_termination_reason = terminal_reason
                break

            set_arm_torque_ctrl(self.model, self.data, self.arm_ids, commanded_torque)
            mujoco.mj_step(self.model, self.data)
            self.step_index += 1

        obs = self.observation()
        depth = insertion_depth(self.data, self.task_ids)
        depth_progress = float(np.clip(depth - self.prev_depth, -self.reward_config.progress_clip, self.reward_config.progress_clip))
        lateral = lateral_error(self.data, self.task_ids)
        bc_action = self.bc_policy.predict(obs)[3]
        bc_deviation = float(np.linalg.norm(action - bc_action))
        action_rate = float(np.linalg.norm(action - self.prev_action))
        reward_terms = self._reward_terms(
            depth_progress=depth_progress,
            normal_force=stride_max_force,
            lateral=lateral,
            torque_saturation_count=stride_torque_saturation_count,
            bc_deviation=bc_deviation,
            action_rate=action_rate,
        )
        reward = float(sum(reward_terms.values()))

        if not done and self.step_index >= self.rollout_config.total_steps:
            done = True
            terminal_reason = "schedule_done"
        depth_reached = self.depth_reached()
        low_force_success = bool(depth_reached and self.max_episode_force <= self.reward_config.low_force_success_threshold)
        completion_like = self.completion_like()
        if done:
            if depth_reached:
                reward += self.reward_config.success_bonus
                reward_terms["terminal_success_bonus"] = self.reward_config.success_bonus
                if low_force_success:
                    reward += self.reward_config.low_force_success_bonus
                    reward_terms["terminal_low_force_success_bonus"] = self.reward_config.low_force_success_bonus
            else:
                reward -= self.reward_config.failure_penalty
                reward_terms["terminal_failure_penalty"] = -self.reward_config.failure_penalty

        self.reward_sum += reward
        self.prev_depth = depth
        self.prev_action = action.copy()

        info = {
            "step": int(self.step_index),
            "episode_id": int(self.episode_id),
            "done": bool(done),
            "terminal_reason": terminal_reason,
            "depth_reached": bool(depth_reached),
            "low_force_success": bool(low_force_success),
            "completion_like": bool(completion_like),
            "depth": float(depth),
            "depth_progress": float(depth_progress),
            "lateral_error": float(lateral),
            "normal_force": float(stride_max_force),
            "max_episode_force": float(self.max_episode_force),
            "torque_saturation_count": int(stride_torque_saturation_count),
            "episode_torque_saturation_count": int(self.torque_saturation_count),
            "bc_deviation": float(bc_deviation),
            "action_rate": float(action_rate),
            "residual_norm": float(np.linalg.norm(action)),
            "action": [float(v) for v in action],
            "bc_action": [float(v) for v in bc_action],
            "reward": float(reward),
            "reward_terms": reward_terms,
            "final_orientation_error": final_orientation_error,
            "early_terminated": bool(self.early_terminated),
        }
        return obs, reward, done, info

    def _reward_terms(
        self,
        *,
        depth_progress: float,
        normal_force: float,
        lateral: float,
        torque_saturation_count: int,
        bc_deviation: float,
        action_rate: float,
    ) -> dict[str, float]:
        clipped_force = min(float(normal_force), self.reward_config.force_clip)
        tail_excess = max(float(normal_force) - self.reward_config.tail_force_threshold, 0.0)
        force_excess = max(float(normal_force) - self.reward_config.safe_force_threshold, 0.0)
        catastrophic = float(normal_force) >= self.reward_config.catastrophic_force_threshold
        return {
            "progress": self.reward_config.progress_weight * depth_progress,
            "force": -self.reward_config.force_weight * clipped_force,
            "force_tail": -self.reward_config.force_tail_weight * tail_excess,
            "force_tail_quadratic": -self.reward_config.tail_quadratic_weight * float(force_excess**2),
            "catastrophic": -self.reward_config.catastrophic_penalty if catastrophic else 0.0,
            "lateral": -self.reward_config.lateral_weight * float(lateral),
            "torque_saturation": -self.reward_config.torque_saturation_weight * float(torque_saturation_count),
            "bc_deviation": -self.reward_config.bc_deviation_weight * float(bc_deviation**2),
            "action_rate": -self.reward_config.action_rate_weight * float(action_rate**2),
        }

    def completion_like(self) -> bool:
        self._require_state()
        return bool(
            insertion_depth(self.data, self.task_ids) >= 0.95 * self.rollout_config.insert_depth
            and lateral_error(self.data, self.task_ids) <= self.geometry.radial_clearance
        )

    def depth_reached(self) -> bool:
        self._require_state()
        return bool(insertion_depth(self.data, self.task_ids) >= 0.95 * self.rollout_config.insert_depth)

    def episode_summary_info(self) -> dict[str, Any]:
        self._require_state()
        final_depth = insertion_depth(self.data, self.task_ids)
        depth_reached = self.depth_reached()
        low_force_success = bool(depth_reached and self.max_episode_force <= self.reward_config.low_force_success_threshold)
        completion_like = self.completion_like()
        return {
            "steps": int(self.step_index),
            "final_depth": float(final_depth),
            "final_lateral_error": float(lateral_error(self.data, self.task_ids)),
            "final_axis_alignment": float(peg_axis_alignment(self.data, self.task_ids, hole_insertion_axis_world())),
            "contact_detected": bool(self.contact_detected),
            "hole_contact_detected": bool(self.hole_contact_detected),
            "contact_onset_step": int(self.contact_onset_step),
            "max_normal_force": float(self.max_episode_force),
            "mean_normal_force_contact": float(self.normal_force_sum / self.normal_force_count) if self.normal_force_count else 0.0,
            "max_tangential_force": float(self.max_tangential_force),
            "max_penetration_depth": float(self.max_penetration_depth),
            "max_abs_commanded_torque": float(self.max_abs_torque),
            "torque_saturation_count": int(self.torque_saturation_count),
            "depth_progress_after_contact": float(final_depth - self.depth_at_contact) if self.contact_detected else 0.0,
            "depth_reached": bool(depth_reached),
            "low_force_success": bool(low_force_success),
            "completion_like": bool(completion_like),
            "perturbation": self.perturbation.to_dict() if self.perturbation is not None else None,
            "reward_sum": float(self.reward_sum),
            "early_terminated": bool(self.early_terminated),
            "early_termination_reason": self.early_termination_reason,
        }

    def close(self) -> None:
        self.model = None
        self.data = None

    def _require_state(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("Environment must be reset before use.")


def load_rrl_policy_for_env(path: Path) -> ResidualRRLPolicy:
    return ResidualRRLPolicy.load(path)


__all__ = ["ResidualInsertionEnv", "ResidualRRLRewardConfig", "load_rrl_policy_for_env"]
