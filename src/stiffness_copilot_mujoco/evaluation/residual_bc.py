from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from stiffness_copilot_mujoco.controllers.impedance import load_task_space_impedance_gains
from stiffness_copilot_mujoco.learning.residual_stiffness import ResidualSPDStiffnessPolicy
from stiffness_copilot_mujoco.robustness import RobustnessPreset, get_robustness_preset, sample_robustness_perturbations
from stiffness_copilot_mujoco.rollouts.fixed_impedance import EpisodeSummary, RolloutConfig, run_fixed_stiffness_episode
from stiffness_copilot_mujoco.rollouts.residual_bc import ResidualBCEpisodeSummary, run_residual_bc_episode
from stiffness_copilot_mujoco.scenes import get_scene_spec
from stiffness_copilot_mujoco.sim.scene import ROOT


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "evaluations" / "residual_bc"
DEFAULT_GAIN_CONFIG = ROOT / "configs" / "controllers" / "fixed_impedance.yaml"


@dataclass(frozen=True)
class ResidualBCEvaluationResult:
    output_dir: Path
    summaries: tuple[EpisodeSummary, ...]
    aggregate: dict[str, dict[str, float | int]]


@dataclass(frozen=True)
class TracedFixedEpisodeSummary(EpisodeSummary):
    full_trace: tuple[dict[str, float | int | str | bool], ...]

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result["full_trace"] = list(self.full_trace)
        return result


def _passive_trace_row(step: int, phase: str, metrics: dict[str, object]) -> dict[str, float | int | str | bool]:
    return {
        "step": int(step),
        "phase": str(phase),
        "depth": float(metrics.get("depth", 0.0)),
        "lateral_error": float(metrics.get("lateral_error", 0.0)),
        "contact": bool(metrics.get("in_contact", False)),
        "normal_force": float(metrics.get("normal_force", 0.0)),
        "torque_saturated": bool(metrics.get("torque_saturated", False)),
    }


def _run_fixed_stiffness_episode_with_trace(**kwargs: object) -> TracedFixedEpisodeSummary:
    trace: list[dict[str, float | int | str | bool]] = []

    def callback(step: int, phase: str, _model: object, _data: object, metrics: dict[str, object]) -> None:
        # Strictly passive: only copy already-computed scalar metrics for shared-protocol reporting.
        trace.append(_passive_trace_row(step, phase, metrics))

    summary = run_fixed_stiffness_episode(step_callback=callback, **kwargs)
    return TracedFixedEpisodeSummary(**summary.__dict__, full_trace=tuple(trace))


def aggregate_residual_summaries(summaries: list[EpisodeSummary]) -> dict[str, dict[str, float | int]]:
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
    for name in ("base", "residual_bc"):
        rows = [row for row in summaries if row.baseline == name]
        if not rows:
            continue
        aggregate: dict[str, float | int] = {
            "episodes": len(rows),
            "contact_count": sum(int(row.contact_detected) for row in rows),
            "completion_like_count": sum(int(row.completion_like) for row in rows),
            "success_rate": float(np.mean([row.completion_like for row in rows])),
        }
        for field in numeric_fields:
            values = np.array([float(getattr(row, field)) for row in rows], dtype=float)
            aggregate[f"mean_{field}"] = float(np.mean(values))
            aggregate[f"max_{field}"] = float(np.max(values))
        if name == "residual_bc":
            residual_rows = [row for row in rows if isinstance(row, ResidualBCEpisodeSummary)]
            aggregate["mean_delta_norm"] = float(np.mean([row.mean_delta_norm for row in residual_rows]))
            aggregate["max_delta_norm"] = float(np.max([row.max_delta_norm for row in residual_rows]))
            aggregate["mean_min_stiffness_eig"] = float(np.mean([row.mean_stiffness_eig[0] for row in residual_rows]))
            aggregate["mean_max_stiffness_eig"] = float(np.mean([row.mean_stiffness_eig[2] for row in residual_rows]))
        result[name] = aggregate
    if "base" in result and "residual_bc" in result:
        result["comparison"] = {
            "success_rate_delta": float(result["residual_bc"]["success_rate"] - result["base"]["success_rate"]),
            "mean_max_normal_force_delta": float(
                result["residual_bc"]["mean_max_normal_force"] - result["base"]["mean_max_normal_force"]
            ),
            "mean_contact_force_delta": float(
                result["residual_bc"]["mean_mean_normal_force_contact"] - result["base"]["mean_mean_normal_force_contact"]
            ),
            "mean_depth_delta": float(result["residual_bc"]["mean_final_depth"] - result["base"]["mean_final_depth"]),
        }
    return result


def evaluate_residual_bc(
    *,
    scene: str,
    policy_path: Path,
    episodes: int = 20,
    seed: int = 0,
    gain_config: Path = DEFAULT_GAIN_CONFIG,
    base_profile: str | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    include_base: bool = True,
    robustness_preset: RobustnessPreset | None = None,
) -> ResidualBCEvaluationResult:
    spec = get_scene_spec(scene)
    profile = base_profile or spec.base_profile
    _, gains = load_task_space_impedance_gains(gain_config, profile)
    config = RolloutConfig(config_path=spec.config_path, gain_config_path=gain_config)
    policy = ResidualSPDStiffnessPolicy.load(policy_path)
    preset = robustness_preset or get_robustness_preset(scene)
    output_dir = output_root / (
        run_id or f"{preset.setting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    perturbations = sample_robustness_perturbations(
        scene=scene,
        episodes=episodes,
        seed=seed,
        preset=preset,
    )
    xy = np.zeros(2, dtype=float)
    summaries: list[EpisodeSummary] = []
    for episode_id, perturbation in enumerate(perturbations):
        if include_base:
            base = _run_fixed_stiffness_episode_with_trace(
                baseline="base",
                seed=seed,
                xy_offset=xy,
                config=config,
                episode_id=episode_id,
                gains=gains,
                profile=profile,
                perturbation=perturbation,
            )
            summaries.append(base)
            print(
                f"baseline=base episode={episode_id} success={base.completion_like} "
                f"depth={base.final_depth:.6f} lat={base.final_lateral_error:.6f} "
                f"max_normal={base.max_normal_force:.3f}"
            )
        residual = run_residual_bc_episode(
            policy=policy,
            policy_path=policy_path,
            seed=seed,
            xy_offset=xy,
            config=config,
            episode_id=episode_id,
            base_profile=profile,
            perturbation=perturbation,
        )
        summaries.append(residual)
        print(
            f"baseline=residual_bc episode={episode_id} success={residual.completion_like} "
            f"depth={residual.final_depth:.6f} lat={residual.final_lateral_error:.6f} "
            f"max_normal={residual.max_normal_force:.3f} mean_delta_norm={residual.mean_delta_norm:.4f}"
        )

    aggregate = aggregate_residual_summaries(summaries)
    with (output_dir / "summary.jsonl").open("w", encoding="utf-8") as handle:
        for summary in summaries:
            handle.write(json.dumps(summary.to_dict(), sort_keys=True) + "\n")
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "scene": scene,
                "scene_config": str(spec.config_path),
                "policy_path": str(policy_path),
                "episodes": episodes,
                "seed": seed,
                "base_profile": profile,
                "gain_config": str(gain_config),
                "include_base": include_base,
                "setting_id": preset.setting_id,
                "robustness_preset": preset.to_metadata(),
                "randomization": "robustness_preset",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ResidualBCEvaluationResult(output_dir=output_dir, summaries=tuple(summaries), aggregate=aggregate)


__all__ = ["DEFAULT_OUTPUT_ROOT", "ResidualBCEvaluationResult", "aggregate_residual_summaries", "evaluate_residual_bc"]
