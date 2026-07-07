from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from stiffness_copilot_mujoco.evaluation.fixed_stiffness_baselines import aggregate_summaries, sample_rollout_perturbations
from stiffness_copilot_mujoco.learning.supervised_policy import SupervisedStiffnessPolicy
from stiffness_copilot_mujoco.rollouts.fixed_impedance import EpisodeSummary, RolloutConfig, run_fixed_stiffness_episode
from stiffness_copilot_mujoco.rollouts.learned_impedance import LearnedEpisodeSummary, run_learned_stiffness_episode
from stiffness_copilot_mujoco.sim.scene import ROOT


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "evaluations" / "learned_stiffness_baseline"


@dataclass(frozen=True)
class LearnedBaselineEvaluationResult:
    output_dir: Path
    summaries: tuple[EpisodeSummary, ...]
    aggregate: dict[str, dict[str, float | int]]


def aggregate_all_summaries(summaries: list[EpisodeSummary]) -> dict[str, dict[str, float | int]]:
    fixed = aggregate_summaries(summaries)
    learned_rows = [summary for summary in summaries if summary.baseline == "learned"]
    if learned_rows:
        numeric_fields = (
            "final_depth",
            "final_lateral_error",
            "max_normal_force",
            "mean_normal_force_contact",
            "max_tangential_force",
            "max_penetration_depth",
            "max_abs_commanded_torque",
            "torque_saturation_count",
            "depth_progress_after_contact",
        )
        learned: dict[str, float | int] = {
            "episodes": len(learned_rows),
            "contact_count": sum(int(row.contact_detected) for row in learned_rows),
            "hole_contact_count": sum(int(row.hole_contact_detected) for row in learned_rows),
            "completion_like_count": sum(int(row.completion_like) for row in learned_rows),
        }
        for field in numeric_fields:
            values = np.array([float(getattr(row, field)) for row in learned_rows], dtype=float)
            learned[f"mean_{field}"] = float(np.mean(values))
            learned[f"max_{field}"] = float(np.max(values))
        learned_eig = np.array(
            [
                [
                    row.mean_predicted_stiffness_eig_min,
                    row.mean_predicted_stiffness_eig_mid,
                    row.mean_predicted_stiffness_eig_max,
                ]
                for row in learned_rows
                if isinstance(row, LearnedEpisodeSummary)
            ],
            dtype=float,
        )
        if learned_eig.size:
            learned["mean_predicted_stiffness_eig_min"] = float(np.mean(learned_eig[:, 0]))
            learned["mean_predicted_stiffness_eig_mid"] = float(np.mean(learned_eig[:, 1]))
            learned["mean_predicted_stiffness_eig_max"] = float(np.mean(learned_eig[:, 2]))
        trace_offsets: dict[int, list[dict[str, float | int | str | bool]]] = {}
        for row in learned_rows:
            if not isinstance(row, LearnedEpisodeSummary):
                continue
            for trace_row in row.contact_onset_trace:
                offset = int(trace_row["contact_onset_offset"])
                trace_offsets.setdefault(offset, []).append(trace_row)
        for offset, offset_rows in sorted(trace_offsets.items()):
            prefix = f"contact_onset_offset_{offset:+d}"
            learned[f"{prefix}_count"] = len(offset_rows)
            for field in ("depth", "lateral_error", "normal_force", "k_eig_min", "k_eig_mid", "k_eig_max"):
                values = np.array([float(item[field]) for item in offset_rows], dtype=float)
                learned[f"{prefix}_mean_{field}"] = float(np.mean(values))
        fixed["learned"] = learned
    return fixed


def evaluate_learned_stiffness_baseline(
    *,
    policy_path: Path,
    episodes: int = 20,
    seed: int = 0,
    config: RolloutConfig | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    include_fixed: bool = True,
    k_min: float = 50.0,
    k_max: float = 200.0,
    base_profile: str = "stiff",
    hole_xy_radius: float = 0.001,
    hole_yaw_max_deg: float = 1.0,
    teleop_noise_xy_amplitude: float = 0.0002,
    stiffness_smoothing_alpha: float = 0.2,
    stiffness_smoothing_eps: float = 1e-8,
    stiffness_smoothing_enabled: bool = True,
) -> LearnedBaselineEvaluationResult:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    rollout_config = config or RolloutConfig()
    policy = SupervisedStiffnessPolicy.load(policy_path)
    run_name = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    perturbations = sample_rollout_perturbations(
        episodes=episodes,
        seed=seed,
        hole_xy_radius=hole_xy_radius,
        hole_yaw_max_deg=hole_yaw_max_deg,
        teleop_noise_xy_amplitude=teleop_noise_xy_amplitude,
    )
    centered_command = np.zeros(2, dtype=float)
    summaries: list[EpisodeSummary] = []
    for episode_id, perturbation in enumerate(perturbations):
        if include_fixed:
            for baseline in ("low", "high"):
                summary = run_fixed_stiffness_episode(
                    baseline=baseline,
                    seed=seed,
                    xy_offset=centered_command,
                    config=rollout_config,
                    episode_id=episode_id,
                    perturbation=perturbation,
                )
                summaries.append(summary)
                print(
                    f"baseline={summary.baseline} episode={episode_id} "
                    f"depth={summary.final_depth:.6f} lat={summary.final_lateral_error:.6f} "
                    f"contact={summary.contact_detected} max_normal={summary.max_normal_force:.3f}"
                )
        learned = run_learned_stiffness_episode(
            policy=policy,
            policy_path=policy_path,
            seed=seed,
            xy_offset=centered_command,
            config=rollout_config,
            episode_id=episode_id,
            k_min=k_min,
            k_max=k_max,
            base_profile=base_profile,
            perturbation=perturbation,
            stiffness_smoothing_alpha=stiffness_smoothing_alpha,
            stiffness_smoothing_eps=stiffness_smoothing_eps,
            stiffness_smoothing_enabled=stiffness_smoothing_enabled,
        )
        summaries.append(learned)
        print(
            f"baseline=learned episode={episode_id} "
            f"depth={learned.final_depth:.6f} lat={learned.final_lateral_error:.6f} "
            f"contact={learned.contact_detected} max_normal={learned.max_normal_force:.3f} "
            f"mean_k=({learned.mean_predicted_stiffness_eig_min:.1f},"
            f"{learned.mean_predicted_stiffness_eig_mid:.1f},"
            f"{learned.mean_predicted_stiffness_eig_max:.1f})"
        )

    aggregate = aggregate_all_summaries(summaries)
    with (output_dir / "summary.jsonl").open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary.to_dict(), sort_keys=True) + "\n")
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_config = {
        "episodes": episodes,
        "seed": seed,
        "policy_path": str(policy_path),
        "include_fixed": include_fixed,
        "k_min": k_min,
        "k_max": k_max,
        "base_profile": base_profile,
        "stiffness_smoothing": {
            "enabled": stiffness_smoothing_enabled,
            "alpha": stiffness_smoothing_alpha,
            "eps": stiffness_smoothing_eps,
            "initial_stiffness": "base_profile translational stiffness",
        },
        "command": "centered hole target, xy_offset = [0, 0]",
        "randomization": {
            "hole_xy_radius": hole_xy_radius,
            "hole_yaw_max_deg": hole_yaw_max_deg,
            "teleop_noise_xy_amplitude": teleop_noise_xy_amplitude,
        },
        "config_path": str(rollout_config.config_path),
        "gain_config_path": str(rollout_config.gain_config_path),
        "approach_hold_steps": rollout_config.approach_hold_steps,
        "descend_steps": rollout_config.descend_steps,
        "insert_steps": rollout_config.insert_steps,
        "final_hold_steps": rollout_config.final_hold_steps,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return LearnedBaselineEvaluationResult(output_dir=output_dir, summaries=tuple(summaries), aggregate=aggregate)


__all__ = ["DEFAULT_OUTPUT_ROOT", "LearnedBaselineEvaluationResult", "evaluate_learned_stiffness_baseline"]
