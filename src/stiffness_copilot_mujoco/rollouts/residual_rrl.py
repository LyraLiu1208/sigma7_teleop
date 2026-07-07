from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.controllers.stiffness_command_smoothing import disabled_smoothing_summary_fields
from stiffness_copilot_mujoco.learning.residual_rrl import ResidualRRLPolicy
from stiffness_copilot_mujoco.learning.residual_stiffness import ResidualSPDStiffnessPolicy
from stiffness_copilot_mujoco.rl.residual_insertion_env import ResidualInsertionEnv, ResidualRRLRewardConfig
from stiffness_copilot_mujoco.rollouts.fixed_impedance import EpisodeSummary, RolloutConfig, RolloutPerturbation


@dataclass(frozen=True, kw_only=True)
class ResidualRRLEpisodeSummary(EpisodeSummary):
    bc_policy_path: str
    rrl_policy_path: str
    mean_delta_norm: float
    max_delta_norm: float
    mean_bc_deviation: float
    max_bc_deviation: float
    mean_action_rate: float
    max_action_rate: float
    reward_sum: float
    reward_mean: float
    reward_p05: float
    early_terminated: bool
    early_termination_reason: str
    full_trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result.update(
            {
                "bc_policy_path": self.bc_policy_path,
                "rrl_policy_path": self.rrl_policy_path,
                "mean_delta_norm": self.mean_delta_norm,
                "max_delta_norm": self.max_delta_norm,
                "mean_bc_deviation": self.mean_bc_deviation,
                "max_bc_deviation": self.max_bc_deviation,
                "mean_action_rate": self.mean_action_rate,
                "max_action_rate": self.max_action_rate,
                "reward_sum": self.reward_sum,
                "reward_mean": self.reward_mean,
                "reward_p05": self.reward_p05,
                "early_terminated": self.early_terminated,
                "early_termination_reason": self.early_termination_reason,
                "full_trace": self.full_trace,
            }
        )
        return result


def run_residual_rrl_episode(
    *,
    policy: ResidualRRLPolicy,
    policy_path: Path,
    bc_policy: ResidualSPDStiffnessPolicy,
    bc_policy_path: Path,
    seed: int,
    xy_offset: np.ndarray,
    config: RolloutConfig,
    episode_id: int = 0,
    base_profile: str = "mid_high_baseline",
    perturbation: RolloutPerturbation | None = None,
    control_stride: int = 1,
    reward_config: ResidualRRLRewardConfig = ResidualRRLRewardConfig(),
) -> ResidualRRLEpisodeSummary:
    if np.linalg.norm(np.asarray(xy_offset, dtype=float)) > 1e-12:
        raise ValueError("Residual RRL rollouts use scene perturbations; nonzero xy_offset is not supported.")
    env = ResidualInsertionEnv(
        scene_config=config.config_path,
        gain_config=config.gain_config_path,
        base_profile=base_profile,
        bc_policy=bc_policy,
        base_spec=policy.base_spec,
        rollout_config=config,
        control_stride=control_stride,
        reward_config=reward_config,
    )
    obs = env.reset(seed=seed, perturbation=perturbation, episode_id=episode_id)
    trace_rows: list[dict[str, Any]] = []
    done = False
    while not done:
        action = policy.predict_group_delta(obs, bc_policy=bc_policy)
        obs, reward, done, info = env.step(action)
        row = dict(info)
        row["reward"] = float(reward)
        trace_rows.append(row)

    summary = env.episode_summary_info()
    rewards = np.array([float(row["reward"]) for row in trace_rows], dtype=float)
    delta_norms = np.array([float(row["residual_norm"]) for row in trace_rows], dtype=float)
    bc_deviation = np.array([float(row["bc_deviation"]) for row in trace_rows], dtype=float)
    action_rate = np.array([float(row["action_rate"]) for row in trace_rows], dtype=float)
    env.close()
    return ResidualRRLEpisodeSummary(
        baseline="residual_rrl",
        profile=base_profile,
        seed=seed,
        episode_id=episode_id,
        xy_offset=(0.0, 0.0),
        steps=int(summary["steps"]),
        final_depth=float(summary["final_depth"]),
        final_lateral_error=float(summary["final_lateral_error"]),
        final_axis_alignment=float(summary["final_axis_alignment"]),
        final_orientation_error=float(trace_rows[-1].get("final_orientation_error", 0.0)) if trace_rows else 0.0,
        contact_detected=bool(summary["contact_detected"]),
        hole_contact_detected=bool(summary["hole_contact_detected"]),
        contact_onset_step=int(summary["contact_onset_step"]),
        max_normal_force=float(summary["max_normal_force"]),
        mean_normal_force_contact=float(summary["mean_normal_force_contact"]),
        max_tangential_force=float(summary["max_tangential_force"]),
        max_penetration_depth=float(summary["max_penetration_depth"]),
        max_abs_commanded_torque=float(summary["max_abs_commanded_torque"]),
        torque_saturation_count=int(summary["torque_saturation_count"]),
        depth_progress_after_contact=float(summary["depth_progress_after_contact"]),
        depth_reached=bool(float(summary["final_depth"]) >= 0.95 * config.insert_depth),
        low_force_success=bool(
            float(summary["final_depth"]) >= 0.95 * config.insert_depth
            and float(summary["max_normal_force"]) <= reward_config.low_force_success_threshold
        ),
        completion_like=bool(summary["completion_like"]),
        **disabled_smoothing_summary_fields(),
        perturbation=summary["perturbation"],
        bc_policy_path=str(bc_policy_path),
        rrl_policy_path=str(policy_path),
        mean_delta_norm=float(np.mean(delta_norms)) if delta_norms.size else 0.0,
        max_delta_norm=float(np.max(delta_norms)) if delta_norms.size else 0.0,
        mean_bc_deviation=float(np.mean(bc_deviation)) if bc_deviation.size else 0.0,
        max_bc_deviation=float(np.max(bc_deviation)) if bc_deviation.size else 0.0,
        mean_action_rate=float(np.mean(action_rate)) if action_rate.size else 0.0,
        max_action_rate=float(np.max(action_rate)) if action_rate.size else 0.0,
        reward_sum=float(summary["reward_sum"]),
        reward_mean=float(np.mean(rewards)) if rewards.size else 0.0,
        reward_p05=float(np.percentile(rewards, 5.0)) if rewards.size else 0.0,
        early_terminated=bool(summary["early_terminated"]),
        early_termination_reason=str(summary["early_termination_reason"]),
        full_trace=trace_rows,
    )


__all__ = ["ResidualRRLEpisodeSummary", "run_residual_rrl_episode"]
