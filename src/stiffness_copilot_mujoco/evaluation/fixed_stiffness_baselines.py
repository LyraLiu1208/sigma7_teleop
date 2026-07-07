from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    DEFAULT_GAIN_CONFIG,
    DEFAULT_TORQUE_CONFIG,
    BaselineName,
    EpisodeSummary,
    RolloutConfig,
    RolloutPerturbation,
    run_fixed_stiffness_episode,
)
from stiffness_copilot_mujoco.sim.scene import ROOT


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "evaluations" / "fixed_stiffness_baselines"


@dataclass(frozen=True)
class BaselineEvaluationResult:
    output_dir: Path
    summaries: tuple[EpisodeSummary, ...]
    aggregate: dict[str, dict[str, float | int]]


def sample_rollout_perturbations(
    *,
    episodes: int,
    seed: int,
    hole_xy_radius: float = 0.001,
    hole_yaw_max_deg: float = 1.0,
    teleop_noise_xy_amplitude: float = 0.0002,
    teleop_noise_cycles_min: float = 0.5,
    teleop_noise_cycles_max: float = 1.5,
) -> tuple[RolloutPerturbation, ...]:
    rng = np.random.default_rng(seed)
    perturbations: list[RolloutPerturbation] = []
    for _ in range(episodes):
        radius = rng.uniform(0.0, hole_xy_radius)
        angle = rng.uniform(-np.pi, np.pi)
        perturbations.append(
            RolloutPerturbation(
                hole_xy_offset=(float(radius * np.cos(angle)), float(radius * np.sin(angle))),
                hole_yaw_offset=float(rng.uniform(-np.deg2rad(hole_yaw_max_deg), np.deg2rad(hole_yaw_max_deg))),
                teleop_noise_xy_amplitude=float(teleop_noise_xy_amplitude),
                teleop_noise_cycles=float(rng.uniform(teleop_noise_cycles_min, teleop_noise_cycles_max)),
                teleop_noise_phase_x=float(rng.uniform(-np.pi, np.pi)),
                teleop_noise_phase_y=float(rng.uniform(-np.pi, np.pi)),
            )
        )
    return tuple(perturbations)


def aggregate_summaries(summaries: list[EpisodeSummary]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
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
    for baseline in ("low", "high"):
        rows = [summary for summary in summaries if summary.baseline == baseline]
        if not rows:
            continue
        baseline_result: dict[str, float | int] = {
            "episodes": len(rows),
            "contact_count": sum(int(row.contact_detected) for row in rows),
            "hole_contact_count": sum(int(row.hole_contact_detected) for row in rows),
            "completion_like_count": sum(int(row.completion_like) for row in rows),
        }
        for field in numeric_fields:
            values = np.array([float(getattr(row, field)) for row in rows], dtype=float)
            baseline_result[f"mean_{field}"] = float(np.mean(values))
            baseline_result[f"max_{field}"] = float(np.max(values))
        result[baseline] = baseline_result
    return result


def evaluate_fixed_stiffness_baselines(
    *,
    episodes: int = 50,
    seed: int = 0,
    config: RolloutConfig | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    print_every: int = 0,
    hole_xy_radius: float = 0.001,
    hole_yaw_max_deg: float = 1.0,
    teleop_noise_xy_amplitude: float = 0.0002,
) -> BaselineEvaluationResult:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    rollout_config = config or RolloutConfig(config_path=DEFAULT_TORQUE_CONFIG, gain_config_path=DEFAULT_GAIN_CONFIG)
    if print_every != rollout_config.print_every:
        rollout_config = RolloutConfig(
            config_path=rollout_config.config_path,
            gain_config_path=rollout_config.gain_config_path,
            approach_hold_steps=rollout_config.approach_hold_steps,
            descend_steps=rollout_config.descend_steps,
            insert_steps=rollout_config.insert_steps,
            final_hold_steps=rollout_config.final_hold_steps,
            approach_height=rollout_config.approach_height,
            descend_height=rollout_config.descend_height,
            insert_depth=rollout_config.insert_depth,
            site_name=rollout_config.site_name,
            print_every=print_every,
            max_steps=rollout_config.max_steps,
        )

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
                f"contact={summary.contact_detected} max_normal={summary.max_normal_force:.3f} "
                f"saturation={summary.torque_saturation_count}"
            )

    aggregate = aggregate_summaries(summaries)
    with (output_dir / "summary.jsonl").open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary.to_dict(), sort_keys=True) + "\n")
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_config = {
        "episodes": episodes,
        "seed": seed,
        "command": "centered hole target, xy_offset = [0, 0]",
        "randomization": {
            "hole_xy_radius": hole_xy_radius,
            "hole_yaw_max_deg": hole_yaw_max_deg,
            "teleop_noise_xy_amplitude": teleop_noise_xy_amplitude,
        },
        "baselines": {"low": "soft", "high": "stiff"},
        "config_path": str(rollout_config.config_path),
        "gain_config_path": str(rollout_config.gain_config_path),
        "approach_hold_steps": rollout_config.approach_hold_steps,
        "descend_steps": rollout_config.descend_steps,
        "insert_steps": rollout_config.insert_steps,
        "final_hold_steps": rollout_config.final_hold_steps,
        "approach_height": rollout_config.approach_height,
        "descend_height": rollout_config.descend_height,
        "insert_depth": rollout_config.insert_depth,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BaselineEvaluationResult(output_dir=output_dir, summaries=tuple(summaries), aggregate=aggregate)


__all__ = [
    "BaselineEvaluationResult",
    "aggregate_summaries",
    "evaluate_fixed_stiffness_baselines",
    "sample_rollout_perturbations",
]
