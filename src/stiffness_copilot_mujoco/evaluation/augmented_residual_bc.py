from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.controllers.impedance import load_task_space_impedance_gains
from stiffness_copilot_mujoco.learning.augmented_residual_stiffness import AugmentedResidualSPDStiffnessPolicy
from stiffness_copilot_mujoco.learning.residual_stiffness import ResidualSPDStiffnessPolicy
from stiffness_copilot_mujoco.robustness import RobustnessPreset, get_robustness_preset, sample_robustness_perturbations
from stiffness_copilot_mujoco.rollouts.augmented_residual_bc import AugmentedResidualBCEpisodeSummary, run_augmented_residual_bc_episode
from stiffness_copilot_mujoco.rollouts.fixed_impedance import RolloutConfig, run_fixed_stiffness_episode
from stiffness_copilot_mujoco.rollouts.residual_bc import run_residual_bc_episode
from stiffness_copilot_mujoco.scenes import get_scene_spec
from stiffness_copilot_mujoco.sim.scene import ROOT


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "evaluations" / "augmented_residual_bc"


@dataclass(frozen=True)
class AugmentedResidualBCEvaluationResult:
    output_dir: Path
    aggregate: dict[str, dict[str, float | int | list[int]]]


def aggregate_augmented_pareto(
    rows: list,
    *,
    catastrophic_force_threshold: float,
    low_force_success_threshold: float,
) -> dict[str, dict[str, float | int | list[int]]]:
    result: dict[str, dict[str, float | int | list[int]]] = {}
    for name in ("low", "high", "mid_high_baseline", "residual_bc", "augmented_residual_bc"):
        selected = [row for row in rows if row.baseline == name]
        if not selected:
            continue
        max_forces = np.array([row.max_normal_force for row in selected], dtype=float)
        success = np.array([row.completion_like for row in selected], dtype=bool)
        aggregate: dict[str, float | int | list[int]] = {
            "episodes": len(selected),
            "success_rate": float(np.mean(success)),
            "failure_count": int(np.sum(~success)),
            "contact_count": int(sum(int(row.contact_detected) for row in selected)),
            "mean_final_depth": float(np.mean([row.final_depth for row in selected])),
            "mean_final_lateral_error": float(np.mean([row.final_lateral_error for row in selected])),
            "mean_max_normal_force": float(np.mean(max_forces)),
            "p95_max_normal_force": float(np.percentile(max_forces, 95.0)),
            "p99_max_normal_force": float(np.percentile(max_forces, 99.0)),
            "max_max_normal_force": float(np.max(max_forces)),
            "catastrophic_force_count": int(np.sum(max_forces >= catastrophic_force_threshold)),
            "success_with_low_force_rate": float(np.mean(success & (max_forces <= low_force_success_threshold))),
            "mean_contact_force_contact_only": float(np.mean([row.mean_normal_force_contact for row in selected])),
            "mean_torque_saturation_count": float(np.mean([row.torque_saturation_count for row in selected])),
            "worst_episode_ids": [
                int(row.episode_id)
                for row in sorted(selected, key=lambda item: float(item.max_normal_force), reverse=True)[:5]
            ],
        }
        if name == "residual_bc":
            aggregate["mean_delta_norm"] = float(np.mean([row.mean_delta_norm for row in selected]))
            aggregate["max_delta_norm"] = float(np.max([row.max_delta_norm for row in selected]))
        if name == "augmented_residual_bc":
            augmented = [row for row in selected if isinstance(row, AugmentedResidualBCEpisodeSummary)]
            aggregate["mean_augmented_delta_norm"] = float(np.mean([row.mean_augmented_delta_norm for row in augmented]))
            aggregate["max_augmented_delta_norm"] = float(np.max([row.max_augmented_delta_norm for row in augmented]))
            aggregate["mean_bc_deviation_from_v1"] = float(np.mean([row.mean_bc_deviation_from_v1 for row in augmented]))
            aggregate["max_bc_deviation_from_v1"] = float(np.max([row.max_bc_deviation_from_v1 for row in augmented]))
        result[name] = aggregate
    return result


def worst_episode_rows(rows: list, *, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for name in ("low", "high", "mid_high_baseline", "residual_bc", "augmented_residual_bc"):
        selected = [row for row in rows if row.baseline == name]
        selected = sorted(selected, key=lambda item: float(item.max_normal_force), reverse=True)[:top_k]
        result[name] = [
            {
                "episode_id": int(row.episode_id),
                "success": bool(row.completion_like),
                "max_normal_force": float(row.max_normal_force),
                "mean_normal_force_contact": float(row.mean_normal_force_contact),
                "final_depth": float(row.final_depth),
                "final_lateral_error": float(row.final_lateral_error),
                "torque_saturation_count": int(row.torque_saturation_count),
                "perturbation": row.perturbation,
            }
            for row in selected
        ]
    return result


def write_pareto_csv(path: Path, agg: dict[str, dict[str, float | int | list[int]]]) -> None:
    fields = [
        "baseline",
        "success_rate",
        "failure_count",
        "mean_max_normal_force",
        "p95_max_normal_force",
        "p99_max_normal_force",
        "max_max_normal_force",
        "catastrophic_force_count",
        "success_with_low_force_rate",
        "mean_contact_force_contact_only",
        "mean_torque_saturation_count",
        "mean_delta_norm",
        "max_delta_norm",
        "mean_augmented_delta_norm",
        "max_augmented_delta_norm",
        "mean_bc_deviation_from_v1",
        "max_bc_deviation_from_v1",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for baseline, values in agg.items():
            row = {"baseline": baseline}
            row.update({field: values.get(field, "") for field in fields if field != "baseline"})
            writer.writerow(row)


def evaluate_augmented_residual_bc(
    *,
    scene: str,
    bc_policy_path: Path,
    augmented_policy_path: Path,
    episodes: int = 50,
    seed: int = 0,
    gain_config: Path = ROOT / "configs" / "controllers" / "fixed_impedance.yaml",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    catastrophic_force_threshold: float = 1000.0,
    low_force_success_threshold: float = 200.0,
    include_fixed: bool = True,
    max_steps: int | None = None,
    robustness_preset: RobustnessPreset | None = None,
) -> AugmentedResidualBCEvaluationResult:
    spec = get_scene_spec(scene)
    profile = spec.base_profile
    _, mid_high_gains = load_task_space_impedance_gains(gain_config, profile)
    bc_policy = ResidualSPDStiffnessPolicy.load(bc_policy_path)
    augmented_policy = AugmentedResidualSPDStiffnessPolicy.load(augmented_policy_path)
    config = RolloutConfig(config_path=spec.config_path, gain_config_path=gain_config, max_steps=max_steps)
    preset = robustness_preset or get_robustness_preset(scene)
    perturbations = sample_robustness_perturbations(
        scene=scene,
        episodes=episodes,
        seed=seed,
        preset=preset,
    )
    output_dir = output_root / (
        run_id or f"{preset.setting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    xy = np.zeros(2, dtype=float)
    rows = []
    for episode_id, perturbation in enumerate(perturbations):
        if include_fixed:
            for baseline in ("low", "high"):
                rows.append(
                    run_fixed_stiffness_episode(
                        baseline=baseline,
                        seed=seed,
                        xy_offset=xy,
                        config=config,
                        episode_id=episode_id,
                        perturbation=perturbation,
                    )
                )
            rows.append(
                run_fixed_stiffness_episode(
                    baseline="mid_high_baseline",
                    seed=seed,
                    xy_offset=xy,
                    config=config,
                    episode_id=episode_id,
                    gains=mid_high_gains,
                    profile=profile,
                    perturbation=perturbation,
                )
            )
        residual = run_residual_bc_episode(
            policy=bc_policy,
            policy_path=bc_policy_path,
            seed=seed,
            xy_offset=xy,
            config=config,
            episode_id=episode_id,
            base_profile=profile,
            perturbation=perturbation,
        )
        rows.append(residual)
        augmented = run_augmented_residual_bc_episode(
            policy=augmented_policy,
            policy_path=augmented_policy_path,
            bc_policy=bc_policy,
            bc_policy_path=bc_policy_path,
            seed=seed,
            xy_offset=xy,
            config=config,
            episode_id=episode_id,
            base_profile=profile,
            perturbation=perturbation,
        )
        rows.append(augmented)
        print(
            f"episode={episode_id} residual_bc={residual.completion_like} augmented_residual_bc={augmented.completion_like} "
            f"bc_force={residual.max_normal_force:.3f} augmented_force={augmented.max_normal_force:.3f}"
        )

    agg = aggregate_augmented_pareto(
        rows,
        catastrophic_force_threshold=catastrophic_force_threshold,
        low_force_success_threshold=low_force_success_threshold,
    )
    with (output_dir / "summary.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    (output_dir / "aggregate.json").write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "worst_episodes.json").write_text(json.dumps(worst_episode_rows(rows), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_pareto_csv(output_dir / "pareto.csv", agg)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "setting_id": preset.setting_id,
                "robustness_preset": preset.to_metadata(),
                "episodes": episodes,
                "seed": seed,
                "bc_policy": str(bc_policy_path),
                "augmented_policy": str(augmented_policy_path),
                "gain_config": str(gain_config),
                "catastrophic_force_threshold": catastrophic_force_threshold,
                "low_force_success_threshold": low_force_success_threshold,
                "max_steps": max_steps,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return AugmentedResidualBCEvaluationResult(output_dir=output_dir, aggregate=agg)


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "AugmentedResidualBCEvaluationResult",
    "aggregate_augmented_pareto",
    "evaluate_augmented_residual_bc",
]
