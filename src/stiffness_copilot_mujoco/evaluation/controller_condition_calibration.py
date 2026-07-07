from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from stiffness_copilot_mujoco.controllers.controller_spec import (
    ControllerSpec,
    TRACK_A_BASELINE_CONTROLLER_ROLE,
    TRACK_A_COLLECTION_CONTROLLER_ROLE,
    TRACK_A_CONTROLLER_FORCE_ACCOUNTING,
    TRACK_A_CONTROLLER_TERMINATION_CONDITION,
    TRACK_A_CONTROLLER_UPDATE_MODE,
    TRACK_A_CONTROLLER_UPDATE_PERIOD_STEPS,
    TRACK_A_TASK_SPACE_CONTROLLER_KIND,
)
from stiffness_copilot_mujoco.controllers.gains import (
    load_track_a_baseline_controller_gains,
    load_track_a_data_collection_controller_gains,
)
from stiffness_copilot_mujoco.controllers.stiffness_command_smoothing import (
    StiffnessCommandSmoother,
    StiffnessCommandSmoothingConfig,
)
from stiffness_copilot_mujoco.evaluation.bad_case_registry import (
    BadCaseThresholds,
    make_bad_case_record,
    write_bad_case_registry,
)
from stiffness_copilot_mujoco.learning.open_loop_residual_dataset import (
    TRAJECTORY_FAMILIES,
    classify_episode_admission,
    generate_open_loop_trajectory,
)
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.metrics.task_metrics import load_scene_config
from stiffness_copilot_mujoco.robustness import ControlledContactProfile, make_controlled_contact_profile, sample_controlled_contact_perturbations
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    EpisodeSummary,
    RolloutConfig,
    RolloutPerturbation,
    run_fixed_stiffness_episode,
)
from stiffness_copilot_mujoco.scenes import get_scene_spec


CALIBRATION_OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "artifacts" / "evaluations" / "track_a_controller_condition_calibration"

CALIBRATION_COLLECTION_GAIN_CANDIDATES: dict[str, tuple[float, float, float]] = {
    "C375": (375.0, 375.0, 280.0),
    "C400": (400.0, 400.0, 300.0),
    "C425": (425.0, 425.0, 315.0),
    "C450": (450.0, 450.0, 330.0),
}

CALIBRATION_BASELINE_GAIN_CANDIDATES: dict[str, tuple[float, float, float]] = {
    "B300": (300.0, 300.0, 220.0),
    "B325": (325.0, 325.0, 240.0),
    "B350": (350.0, 350.0, 260.0),
}

CALIBRATION_CONTROLLER_CANDIDATE_SETS = {
    "current": ("C450", "B300"),
    "full": ("C375", "C400", "C425", "C450", "B300", "B325", "B350"),
}

CALIBRATION_TOGGLE_STATES = (False, True)
CALIBRATION_LOW_FORCE_THRESHOLDS = (100.0, 150.0, 200.0)
CALIBRATION_DEFAULT_PROFILE_NAME = "circle_calibrated_v1_global_hole_fixed_contact"
CALIBRATION_DEFAULT_CONTACT_CONDITION = "light"
CALIBRATION_DEFAULT_CONTROLLED_CONTACT_PROFILE = make_controlled_contact_profile(
    profile_name=CALIBRATION_DEFAULT_PROFILE_NAME,
    contact_condition_name=CALIBRATION_DEFAULT_CONTACT_CONDITION,
    hole_xy_radius=0.02,
    teleop_noise_xy_amplitude=0.0010,
    teleop_noise_cycles=1.0,
    teleop_noise_phase_x=0.0,
    teleop_noise_phase_y=1.5707963267948966,
    clearance_delta=-0.0004,
    friction_scale=1.15,
    peg_tilt_x=0.0087,
    peg_tilt_y=-0.0087,
)


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_float(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _format_tilt_name(value: float) -> str:
    return _format_float(abs(float(value)))


def _candidate_score_label(score: float) -> str:
    if score >= 3.0:
        return "strong"
    if score >= 2.0:
        return "moderate"
    return "weak"


@dataclass
class TraceAccumulator:
    step_count: int = 0
    contact_step_count: int = 0
    max_position_error: float = 0.0
    max_lateral_tracking_error: float = 0.0
    max_vertical_tracking_error: float = 0.0
    smoothing_update_count: int = 0
    smoothing_hold_count: int = 0
    normal_forces: list[float] | None = None
    tangential_forces: list[float] | None = None
    depths: list[float] | None = None
    contacts: list[bool] | None = None
    times: list[float] | None = None
    trace_rows: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        self.normal_forces = []
        self.tangential_forces = []
        self.depths = []
        self.contacts = []
        self.times = []
        self.trace_rows = []


@dataclass(frozen=True)
class CalibrationControllerCandidate:
    candidate_name: str
    candidate_group: Literal["collection", "baseline"]
    controller_spec: ControllerSpec
    position_stiffness_matrix: np.ndarray | None
    position_stiffness_matrix_source: str | None
    candidate_score_label: str = ""


@dataclass(frozen=True)
class CalibrationConditionCandidate:
    condition_name: str
    controlled_contact_profile: ControlledContactProfile
    condition_axis: str
    condition_value: float


@dataclass(frozen=True)
class CalibrationRunPaths:
    output_dir: Path
    summary_path: Path
    metadata_path: Path
    candidate_metrics_path: Path
    bad_case_registry_path: Path
    report_path: Path
    recommended_roles_path: Path
    manual_commands_path: Path


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=float)
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _trace_callback(acc: TraceAccumulator):
    def callback(step: int, phase: str, model, data, metrics: dict[str, float | bool | np.ndarray]) -> None:
        target_position = np.asarray(metrics["target_position"], dtype=float)
        actual_position = np.asarray(data.site_xpos[model.site("peg_tip").id], dtype=float)
        acc.step_count = int(step) + 1
        in_contact = bool(metrics["in_contact"])
        acc.contact_step_count += int(in_contact)
        acc.max_position_error = max(acc.max_position_error, float(metrics["position_error"]))
        acc.max_lateral_tracking_error = max(
            acc.max_lateral_tracking_error,
            float(np.linalg.norm(actual_position[:2] - target_position[:2])),
        )
        acc.max_vertical_tracking_error = max(
            acc.max_vertical_tracking_error,
            abs(float(actual_position[2] - target_position[2])),
        )
        acc.smoothing_update_count += int(bool(metrics.get("smoothing_update_applied", False)))
        acc.smoothing_hold_count += int(bool(metrics.get("smoothing_hold_applied", False)))
        normal_force = float(metrics["normal_force"])
        tangential_force = float(metrics["tangential_force"])
        depth = float(metrics["depth"])
        acc.normal_forces.append(normal_force)
        acc.tangential_forces.append(tangential_force)
        acc.depths.append(depth)
        acc.contacts.append(in_contact)
        acc.times.append(float(data.time))
        acc.trace_rows.append(
            {
                "step": int(step),
                "phase": phase,
                "time": float(data.time),
                "normal_force": normal_force,
                "tangential_force": tangential_force,
                "depth": depth,
                "lateral_error": float(metrics["lateral_error"]),
                "position_error": float(metrics["position_error"]),
                "in_contact": in_contact,
                "smoothing_update_applied": bool(metrics.get("smoothing_update_applied", False)),
                "smoothing_hold_applied": bool(metrics.get("smoothing_hold_applied", False)),
            }
        )

    return callback


def _candidate_profile_name(group: str) -> str:
    return "track_a_c600" if group == "collection" else "mid_high_baseline"


def _make_controller_spec(
    *,
    gain_config: Path,
    group: Literal["collection", "baseline"],
    candidate_name: str,
    gains: Any,
    position_stiffness_matrix: np.ndarray | None,
) -> CalibrationControllerCandidate:
    selected_profile = _candidate_profile_name(group)
    controller_spec = ControllerSpec(
        controller_role=TRACK_A_COLLECTION_CONTROLLER_ROLE if group == "collection" else TRACK_A_BASELINE_CONTROLLER_ROLE,
        controller_kind=TRACK_A_TASK_SPACE_CONTROLLER_KIND,
        requested_profile=selected_profile,
        selected_profile=selected_profile,
        gain_config_path=str(gain_config),
        gains=gains,
        position_stiffness_matrix=None if position_stiffness_matrix is None else np.asarray(position_stiffness_matrix, dtype=float),
        position_stiffness_matrix_source=None
        if position_stiffness_matrix is None
        else f"calibration.{candidate_name}.explicit_matrix",
        control_update_mode=TRACK_A_CONTROLLER_UPDATE_MODE,
        control_update_period_steps=TRACK_A_CONTROLLER_UPDATE_PERIOD_STEPS,
        force_accounting_mode=TRACK_A_CONTROLLER_FORCE_ACCOUNTING,
        termination_condition=TRACK_A_CONTROLLER_TERMINATION_CONDITION,
    )
    return CalibrationControllerCandidate(
        candidate_name=candidate_name,
        candidate_group=group,
        controller_spec=controller_spec,
        position_stiffness_matrix=None if position_stiffness_matrix is None else np.asarray(position_stiffness_matrix, dtype=float),
        position_stiffness_matrix_source=None
        if position_stiffness_matrix is None
        else f"calibration.{candidate_name}.explicit_matrix",
    )


def build_controller_candidates(gain_config: Path, *, controller_candidate_set: str = "full") -> list[CalibrationControllerCandidate]:
    if controller_candidate_set not in CALIBRATION_CONTROLLER_CANDIDATE_SETS:
        raise ValueError(f"Unknown controller candidate set {controller_candidate_set!r}.")
    collection_selected_profile, collection_gains = load_track_a_data_collection_controller_gains(gain_config)
    baseline_selected_profile, baseline_gains = load_track_a_baseline_controller_gains(gain_config)
    if collection_selected_profile != "track_a_c600":
        raise ValueError(f"Expected track_a_c600 profile, observed {collection_selected_profile!r}.")
    if baseline_selected_profile != "mid_high_baseline":
        raise ValueError(f"Expected mid_high_baseline profile, observed {baseline_selected_profile!r}.")

    candidates: list[CalibrationControllerCandidate] = []
    for candidate_name in CALIBRATION_CONTROLLER_CANDIDATE_SETS[controller_candidate_set]:
        if candidate_name.startswith("C"):
            vector = CALIBRATION_COLLECTION_GAIN_CANDIDATES[candidate_name]
            gains = replace(collection_gains, position_stiffness=vector)
            matrix = np.diag(np.asarray(vector, dtype=float))
            candidates.append(
                _make_controller_spec(
                    gain_config=gain_config,
                    group="collection",
                    candidate_name=candidate_name,
                    gains=gains,
                    position_stiffness_matrix=matrix,
                )
            )
            continue
        vector = CALIBRATION_BASELINE_GAIN_CANDIDATES[candidate_name]
        gains = replace(baseline_gains, position_stiffness=vector)
        candidates.append(
            _make_controller_spec(
                gain_config=gain_config,
                group="baseline",
                candidate_name=candidate_name,
                gains=gains,
                position_stiffness_matrix=None,
            )
        )
    return candidates


def build_default_contact_condition(*, hole_xy_radius: float = 0.02) -> CalibrationConditionCandidate:
    profile = replace(CALIBRATION_DEFAULT_CONTROLLED_CONTACT_PROFILE, hole_xy_radius=float(hole_xy_radius))
    return CalibrationConditionCandidate(
        condition_name=CALIBRATION_DEFAULT_CONTACT_CONDITION,
        controlled_contact_profile=profile,
        condition_axis="default",
        condition_value=float(hole_xy_radius),
    )


def _format_condition_name(axis: str, value: float) -> str:
    if axis == "teleop":
        return f"light_teleop_{_format_float(value)}"
    if axis == "clearance":
        return f"light_clearance_{_format_float(value)}"
    if axis == "tilt":
        return f"light_tilt_{_format_tilt_name(value)}"
    if axis == "friction":
        return f"light_friction_{_format_float(value)}"
    return f"light_{axis}_{_format_float(value)}"


def build_condition_candidates(*, include_condition_sweep: bool, hole_xy_radius: float = 0.02) -> list[CalibrationConditionCandidate]:
    candidates = [build_default_contact_condition(hole_xy_radius=hole_xy_radius)]
    if not include_condition_sweep:
        return candidates

    base = replace(CALIBRATION_DEFAULT_CONTROLLED_CONTACT_PROFILE, hole_xy_radius=float(hole_xy_radius))
    teleop_candidates = (0.0008, 0.0012)
    clearance_candidates = (-0.0003, -0.0005)
    tilt_candidates = (0.0060, 0.0120)
    friction_candidates = (1.0, 1.3)

    for value in teleop_candidates:
        candidates.append(
            CalibrationConditionCandidate(
                condition_name=_format_condition_name("teleop", value),
                controlled_contact_profile=replace(base, teleop_noise_xy_amplitude=float(value), contact_condition_name=_format_condition_name("teleop", value)),
                condition_axis="teleop",
                condition_value=float(value),
            )
        )
    for value in clearance_candidates:
        candidates.append(
            CalibrationConditionCandidate(
                condition_name=_format_condition_name("clearance", value),
                controlled_contact_profile=replace(base, clearance_delta=float(value), contact_condition_name=_format_condition_name("clearance", value)),
                condition_axis="clearance",
                condition_value=float(value),
            )
        )
    for value in tilt_candidates:
        candidates.append(
            CalibrationConditionCandidate(
                condition_name=_format_condition_name("tilt", value),
                controlled_contact_profile=replace(
                    base,
                    peg_tilt_x=float(value),
                    peg_tilt_y=float(-value),
                    contact_condition_name=_format_condition_name("tilt", value),
                ),
                condition_axis="tilt",
                condition_value=float(value),
            )
        )
    for value in friction_candidates:
        candidates.append(
            CalibrationConditionCandidate(
                condition_name=_format_condition_name("friction", value),
                controlled_contact_profile=replace(base, friction_scale=float(value), contact_condition_name=_format_condition_name("friction", value)),
                condition_axis="friction",
                condition_value=float(value),
            )
        )
    return candidates


def _episode_specs_for_condition(
    *,
    scene: str,
    setting: str,
    episodes: int,
    seed: int,
    rollout_config: RolloutConfig,
    profile_name: str,
    controlled_contact_profile: ControlledContactProfile,
) -> list:
    preset_seed = np.random.SeedSequence(seed)
    perturbation_seq, trajectory_seq = preset_seed.spawn(2)
    perturbation_seed = int(perturbation_seq.generate_state(1, dtype=np.uint32)[0])
    trajectory_seed = int(trajectory_seq.generate_state(1, dtype=np.uint32)[0])
    perturbations = sample_controlled_contact_perturbations(
        episodes=episodes,
        seed=perturbation_seed,
        profile=controlled_contact_profile,
    )
    trajectory_rng = np.random.default_rng(trajectory_seed)
    scene_spec = get_scene_spec(scene)
    nominal_scene_config = load_scene_config(scene_spec.config_path)
    nominal_hole_position = np.asarray(nominal_scene_config["hole"]["pos"], dtype=float)
    nominal_hole_xy = np.asarray(nominal_hole_position[:2], dtype=float)
    profile_metadata = controlled_contact_profile.to_metadata()
    specs = []
    for episode_id, perturbation in enumerate(perturbations):
        family = TRAJECTORY_FAMILIES[episode_id % len(TRAJECTORY_FAMILIES)]
        plan = generate_open_loop_trajectory(
            family=family,
            rng=trajectory_rng,
            perturbation=perturbation,
            base_config=rollout_config,
        )
        from stiffness_copilot_mujoco.episodes.episode_spec import EpisodeSpec  # local import avoids cycle in typing

        specs.append(
            EpisodeSpec.create(
                episode_id=episode_id,
                seed=seed,
                scene=scene,
                setting_id=setting,
                profile_name=profile_name,
                contact_condition_name=controlled_contact_profile.contact_condition_name,
                nominal_hole_position=nominal_hole_position,
                nominal_hole_xy=nominal_hole_xy,
                hole_xy_offset=np.asarray(perturbation.hole_xy_offset, dtype=float),
                hole_yaw_offset=float(perturbation.hole_yaw_offset),
                hole_xy_radius=float(controlled_contact_profile.hole_xy_radius),
                hole_xy_offset_semantics=str(profile_metadata["hole_xy_offset_semantics"]),
                hole_xy_offset_distribution=str(profile_metadata["hole_xy_offset_distribution"]),
                trajectory_follows_randomized_hole=bool(profile_metadata["trajectory_follows_randomized_hole"]),
                contact_generation_parameters_fixed=bool(profile_metadata["contact_generation_parameters_fixed"]),
                fixed_contact_condition={
                    "teleop_noise_xy_amplitude": float(perturbation.teleop_noise_xy_amplitude),
                    "teleop_noise_cycles": float(perturbation.teleop_noise_cycles),
                    "teleop_noise_phase_x": float(perturbation.teleop_noise_phase_x),
                    "teleop_noise_phase_y": float(perturbation.teleop_noise_phase_y),
                    "clearance_delta": float(perturbation.clearance_delta),
                    "friction_scale": float(perturbation.friction_scale),
                    "peg_tilt_x": float(perturbation.peg_tilt_x),
                    "peg_tilt_y": float(perturbation.peg_tilt_y),
                    "fixed_hole_yaw_offset": float(perturbation.hole_yaw_offset),
                },
                trajectory_source="open_loop_family_episode_spec",
                trajectory_family=plan.family,
                trajectory_family_id=plan.family_id,
                trajectory_parameters=plan.parameters,
                target_offsets=plan.target_offsets,
                phase_ids=plan.phase_ids,
                total_steps=plan.total_steps,
            )
        )
    return specs


def _trace_summary(values: list[float]) -> dict[str, object]:
    if not values:
        zeros = [0.0, 0.0, 0.0]
        return {"count": 0, "mean": zeros, "min": zeros, "max": zeros}
    stacked = np.asarray(values, dtype=float)
    return {
        "count": int(stacked.shape[0]),
        "mean": float(np.mean(stacked)),
        "min": float(np.min(stacked)),
        "max": float(np.max(stacked)),
    }


def _episode_row(
    *,
    run_id: str,
    condition_name: str,
    controller_candidate: CalibrationControllerCandidate,
    summary: EpisodeSummary,
    trace: TraceAccumulator,
    thresholds: BadCaseThresholds,
    smoothing_config: StiffnessCommandSmoothingConfig,
) -> tuple[dict[str, object], dict[str, object], str]:
    normal_forces = np.asarray(trace.normal_forces or [], dtype=float)
    tangential_forces = np.asarray(trace.tangential_forces or [], dtype=float)
    depths = np.asarray(trace.depths or [], dtype=float)
    contacts = np.asarray(trace.contacts or [], dtype=bool)
    contact_forces = normal_forces[contacts] if contacts.size else np.zeros(0, dtype=float)

    raw_max_force = float(summary.max_normal_force)
    sampled_max_force = float(np.max(normal_forces)) if normal_forces.size else 0.0
    arrays_finite = bool(np.all(np.isfinite(normal_forces))) and bool(np.all(np.isfinite(tangential_forces)))
    _, solver_spike_suspicious, admission_reason = classify_episode_admission(
        episode_complete=bool(summary.depth_reached),
        arrays_finite=arrays_finite,
        full_step_max_force=raw_max_force,
        sampled_max_force=sampled_max_force,
    )

    low_force_success = {
        f"low_force_success_rate_{int(threshold)}N": bool(summary.depth_reached and raw_max_force <= float(threshold))
        for threshold in CALIBRATION_LOW_FORCE_THRESHOLDS
    }
    raw_max_force_step = int(np.argmax(normal_forces)) if normal_forces.size else -1
    peak_depth = float(depths[raw_max_force_step]) if depths.size and raw_max_force_step >= 0 else float(summary.final_depth)
    short_filtered = _moving_average(normal_forces, 5)
    long_filtered = _moving_average(normal_forces, 25)
    all_p95 = float(np.percentile(normal_forces, 95)) if normal_forces.size else 0.0
    all_p99 = float(np.percentile(normal_forces, 99)) if normal_forces.size else 0.0
    tangential_p95 = float(np.percentile(tangential_forces, 95)) if tangential_forces.size else 0.0
    tangential_p99 = float(np.percentile(tangential_forces, 99)) if tangential_forces.size else 0.0
    contact_p95 = float(np.percentile(contact_forces, 95)) if contact_forces.size else 0.0
    contact_p99 = float(np.percentile(contact_forces, 99)) if contact_forces.size else 0.0
    contact_fraction = float(trace.contact_step_count / max(trace.step_count, 1))
    contact_episode = bool(summary.hole_contact_detected)
    catastrophic = bool(raw_max_force >= thresholds.catastrophic_force_threshold)
    torque_saturation_episode = bool(summary.torque_saturation_count > 0)
    admission_label = "eligible"
    if catastrophic:
        admission_label = "catastrophic_quarantine"
    elif solver_spike_suspicious:
        admission_label = "solver_spike_quarantine"
    elif torque_saturation_episode:
        admission_label = "torque_saturation_warning"
    elif raw_max_force >= thresholds.low_force_success_threshold:
        admission_label = "high_force_warning"

    metrics = {
        "raw_max_force": raw_max_force,
        "raw_max_force_step": raw_max_force_step,
        "raw_max_force_time": float(trace.times[raw_max_force_step]) if trace.times and raw_max_force_step >= 0 else None,
        "raw_max_force_contact_state": bool(contacts[raw_max_force_step]) if contacts.size and raw_max_force_step >= 0 else False,
        "contact_only_p95_force": contact_p95,
        "contact_only_p99_force": contact_p99,
        "all_step_p95_force": all_p95,
        "all_step_p99_force": all_p99,
        "all_step_tangential_p95_force": tangential_p95,
        "all_step_tangential_p99_force": tangential_p99,
        "filtered_max_force_short_window": float(np.max(short_filtered)) if short_filtered.size else 0.0,
        "filtered_max_force_long_window": float(np.max(long_filtered)) if long_filtered.size else 0.0,
        "high_force_duration_steps": int(np.sum(normal_forces >= thresholds.low_force_success_threshold)),
        "high_force_duration": float(np.sum(normal_forces >= thresholds.low_force_success_threshold) * 0.002),
        "high_force_fraction": float(np.mean(normal_forces >= thresholds.low_force_success_threshold)) if normal_forces.size else 0.0,
        "raw_to_p99_ratio": float(raw_max_force / max(all_p99, 1e-9)),
        "raw_to_filtered_ratio": float(raw_max_force / max(float(np.max(long_filtered)) if long_filtered.size else 1e-9, 1e-9)),
    }
    context = {
        "success": bool(summary.depth_reached),
        "depth_reached": bool(summary.depth_reached),
        "low_force_success": bool(summary.low_force_success),
        "final_depth": float(summary.final_depth),
        "depth_progress_around_peak": float(summary.final_depth - peak_depth),
        "contact_duration_steps": int(trace.contact_step_count),
        "contact_duration": float(trace.contact_step_count * 0.002),
        "action_jump_around_peak": 0.0,
        "residual_rate_around_peak": 0.0,
        "rebound": False,
        "trace_incomplete": False,
    }
    bad_case_record = make_bad_case_record(
        run_id=run_id,
        method=condition_name,
        track="track_a",
        policy_id=None,
        geometry="circle",
        setting_id="circle_calibrated_v1",
        seed=int(summary.seed),
        episode_id=int(summary.episode_id),
        metrics=metrics,
        trace_context=context,
        initial_perturbation=summary.perturbation,
        automatic_classification=admission_reason,
        notes=f"controller={controller_candidate.candidate_name}",
    )
    row = {
        "run_id": run_id,
        "condition_name": condition_name,
        "controller_group": controller_candidate.candidate_group,
        "candidate_name": controller_candidate.candidate_name,
        "controller_role": controller_candidate.controller_spec.controller_role,
        "smoothing_enabled": bool(summary.smoothing_enabled),
        "smoothing_method": summary.smoothing_method,
        "smoothing_alpha": float(summary.smoothing_alpha),
        "policy_update_period_steps": int(summary.policy_update_period_steps),
        "hold_between_updates": bool(summary.hold_between_updates),
        "episode_id": int(summary.episode_id),
        "seed": int(summary.seed),
        "depth_reached": bool(summary.depth_reached),
        "low_force_success": bool(summary.low_force_success),
        "contact_detected": bool(summary.contact_detected),
        "hole_contact_detected": bool(summary.hole_contact_detected),
        "contact_fraction": contact_fraction,
        "contact_onset_step": int(summary.contact_onset_step),
        "contact_onset_time": float(summary.contact_onset_step * 0.002) if summary.contact_onset_step >= 0 else None,
        "max_normal_force": raw_max_force,
        "p95_normal_force": all_p95,
        "p99_normal_force": all_p99,
        "max_tangential_force": float(summary.max_tangential_force),
        "p95_tangential_force": tangential_p95,
        "p99_tangential_force": tangential_p99,
        "catastrophic": catastrophic,
        "solver_spike_suspicious": bool(solver_spike_suspicious),
        "torque_saturation_count": int(summary.torque_saturation_count),
        "torque_saturation_episode": torque_saturation_episode,
        "final_depth": float(summary.final_depth),
        "final_lateral_error": float(summary.final_lateral_error),
        "max_position_error": float(trace.max_position_error),
        "max_lateral_tracking_error": float(trace.max_lateral_tracking_error),
        "max_vertical_tracking_error": float(trace.max_vertical_tracking_error),
        "mean_normal_force_contact": float(summary.mean_normal_force_contact),
        "mean_max_abs_commanded_torque": float(summary.max_abs_commanded_torque),
        "admission_label": admission_label,
        "low_force_success_rate_100N": low_force_success["low_force_success_rate_100N"],
        "low_force_success_rate_150N": low_force_success["low_force_success_rate_150N"],
        "low_force_success_rate_200N": low_force_success["low_force_success_rate_200N"],
        "automatic_bad_case_classification": bad_case_record["automatic_classification"],
    }
    return row, bad_case_record, admission_label


def _aggregate_rows(
    rows: list[dict[str, object]],
    *,
    thresholds: BadCaseThresholds,
    condition_name: str,
    controller_candidate: CalibrationControllerCandidate,
    smoothing_config: StiffnessCommandSmoothingConfig,
) -> dict[str, object]:
    if not rows:
        return {"episode_count": 0}
    max_forces = np.asarray([float(row["max_normal_force"]) for row in rows], dtype=float)
    tangential_forces = np.asarray([float(row["max_tangential_force"]) for row in rows], dtype=float)
    depth_reached = np.asarray([bool(row["depth_reached"]) for row in rows], dtype=bool)
    contact_detected = np.asarray([bool(row["contact_detected"]) for row in rows], dtype=bool)
    hole_contact = np.asarray([bool(row["hole_contact_detected"]) for row in rows], dtype=bool)
    contact_fraction = np.asarray([float(row["contact_fraction"]) for row in rows], dtype=float)
    contact_onset = np.asarray([int(row["contact_onset_step"]) for row in rows if int(row["contact_onset_step"]) >= 0], dtype=int)
    torque_saturation = np.asarray([int(row["torque_saturation_count"]) for row in rows], dtype=int)
    bad_cases = np.asarray([str(row["admission_label"]) != "eligible" for row in rows], dtype=bool)
    return {
        "condition_name": condition_name,
        "controller_group": controller_candidate.candidate_group,
        "candidate_name": controller_candidate.candidate_name,
        "episode_count": int(len(rows)),
        "depth_reached_rate": float(np.mean(depth_reached)),
        "hole_contact_episode_rate": float(np.mean(hole_contact)),
        "contact_detected_rate": float(np.mean(contact_detected)),
        "contact_fraction_mean": float(np.mean(contact_fraction)),
        "contact_fraction_median": float(np.median(contact_fraction)),
        "contact_onset_mean": float(np.mean(contact_onset)) if contact_onset.size else None,
        "contact_onset_median": float(np.median(contact_onset)) if contact_onset.size else None,
        "mean_max_normal_force": float(np.mean(max_forces)),
        "p95_max_normal_force": float(np.percentile(max_forces, 95)),
        "p99_max_normal_force": float(np.percentile(max_forces, 99)),
        "mean_max_tangential_force": float(np.mean(tangential_forces)),
        "p95_max_tangential_force": float(np.percentile(tangential_forces, 95)),
        "p99_max_tangential_force": float(np.percentile(tangential_forces, 99)),
        "mean_normal_force_contact": float(np.mean([float(row["mean_normal_force_contact"]) for row in rows])),
        "low_force_success_rate_100N": float(np.mean([bool(row["low_force_success_rate_100N"]) for row in rows])),
        "low_force_success_rate_150N": float(np.mean([bool(row["low_force_success_rate_150N"]) for row in rows])),
        "low_force_success_rate_200N": float(np.mean([bool(row["low_force_success_rate_200N"]) for row in rows])),
        "catastrophic_count": int(np.sum(max_forces >= thresholds.catastrophic_force_threshold)),
        "solver_spike_count": int(sum(bool(row["solver_spike_suspicious"]) for row in rows)),
        "torque_saturation_total": int(np.sum(torque_saturation)),
        "torque_saturation_episode_rate": float(np.mean(torque_saturation > 0)),
        "mean_final_depth": float(np.mean([float(row["final_depth"]) for row in rows])),
        "mean_final_lateral_error": float(np.mean([float(row["final_lateral_error"]) for row in rows])),
        "mean_max_position_error": float(np.mean([float(row["max_position_error"]) for row in rows])),
        "mean_max_lateral_tracking_error": float(np.mean([float(row["max_lateral_tracking_error"]) for row in rows])),
        "mean_max_vertical_tracking_error": float(np.mean([float(row["max_vertical_tracking_error"]) for row in rows])),
        "bad_case_count": int(np.sum(bad_cases)),
        "smoothing_enabled": bool(smoothing_config.enabled),
        "smoothing_method": str(smoothing_config.method),
        "smoothing_alpha": float(smoothing_config.alpha),
        "policy_update_period_steps": int(smoothing_config.resolved_policy_update_period_steps()),
        "hold_between_updates": bool(smoothing_config.hold_between_updates),
        "admission_recommendation": _admission_recommendation(rows, thresholds=thresholds),
    }


def _admission_recommendation(rows: list[dict[str, object]], *, thresholds: BadCaseThresholds) -> str:
    if any(bool(row["catastrophic"]) for row in rows):
        return "catastrophic_quarantine"
    if any(bool(row["solver_spike_suspicious"]) for row in rows):
        return "solver_spike_quarantine"
    torque_rate = float(np.mean([bool(row["torque_saturation_episode"]) for row in rows])) if rows else 0.0
    max_force = float(np.mean([float(row["max_normal_force"]) for row in rows])) if rows else 0.0
    if torque_rate > 0.25:
        return "torque_saturation_warning"
    if max_force >= thresholds.low_force_success_threshold:
        return "high_force_warning"
    return "eligible"


def _collection_score(aggregate: dict[str, object]) -> float:
    contact = float(aggregate["hole_contact_episode_rate"])
    frac = float(aggregate["contact_fraction_mean"])
    depth_reached = float(aggregate["depth_reached_rate"])
    low_force = float(aggregate["low_force_success_rate_150N"])
    catastrophic = float(aggregate["catastrophic_count"]) / max(float(aggregate["episode_count"]), 1.0)
    spike = float(aggregate["solver_spike_count"]) / max(float(aggregate["episode_count"]), 1.0)
    torque = float(aggregate["torque_saturation_episode_rate"])
    p99 = float(aggregate["p99_max_normal_force"])
    mean_contact_force = float(aggregate["mean_normal_force_contact"])
    return (
        3.0 * contact
        + 2.0 * frac
        + 1.0 * depth_reached
        + 0.5 * low_force
        + 0.002 * mean_contact_force
        - 3.0 * catastrophic
        - 2.0 * spike
        - 1.0 * torque
        - 0.0005 * p99
    )


def _baseline_score(aggregate: dict[str, object]) -> float:
    depth_reached = float(aggregate["depth_reached_rate"])
    contact = float(aggregate["hole_contact_episode_rate"])
    frac = float(aggregate["contact_fraction_mean"])
    low_force = float(aggregate["low_force_success_rate_150N"])
    catastrophic = float(aggregate["catastrophic_count"]) / max(float(aggregate["episode_count"]), 1.0)
    spike = float(aggregate["solver_spike_count"]) / max(float(aggregate["episode_count"]), 1.0)
    torque = float(aggregate["torque_saturation_episode_rate"])
    p99 = float(aggregate["p99_max_normal_force"])
    target_band = 0.65
    return (
        2.5 * (1.0 - min(abs(depth_reached - target_band), 1.0))
        + 1.0 * contact
        + 0.75 * frac
        + 0.5 * (1.0 - low_force)
        - 3.0 * catastrophic
        - 2.0 * spike
        - 1.5 * torque
        - 0.0004 * p99
    )


def _smoothing_delta_summary(no_smoothing: dict[str, object], with_smoothing: dict[str, object]) -> dict[str, object]:
    keys = (
        "depth_reached_rate",
        "hole_contact_episode_rate",
        "contact_fraction_mean",
        "mean_max_normal_force",
        "p95_max_normal_force",
        "p99_max_normal_force",
        "mean_final_depth",
        "mean_final_lateral_error",
        "torque_saturation_total",
    )
    delta = {}
    for key in keys:
        if key in no_smoothing and key in with_smoothing:
            try:
                delta[key] = float(with_smoothing[key]) - float(no_smoothing[key])
            except Exception:
                delta[key] = None
    delta["stability_ok"] = bool(
        abs(float(delta.get("depth_reached_rate", 0.0))) < 1e-9
        and abs(float(delta.get("mean_max_normal_force", 0.0))) < 1e-9
        and abs(float(delta.get("mean_final_depth", 0.0))) < 1e-9
        and abs(float(delta.get("mean_final_lateral_error", 0.0))) < 1e-9
        and abs(float(delta.get("torque_saturation_total", 0.0))) < 1e-9
    )
    return delta


def _best_candidate_by_group(
    aggregates: list[dict[str, object]],
    *,
    group: Literal["collection", "baseline"],
) -> dict[str, object] | None:
    group_rows = [row for row in aggregates if row["controller_group"] == group and not bool(row["smoothing_enabled"])]
    if not group_rows:
        return None
    score_fn = _collection_score if group == "collection" else _baseline_score
    for row in group_rows:
        row["selection_score"] = float(score_fn(row))
        row["selection_score_label"] = _candidate_score_label(float(row["selection_score"]))
    return max(group_rows, key=lambda row: float(row["selection_score"]))


def _candidate_rows_by_name(aggregates: list[dict[str, object]], name: str) -> list[dict[str, object]]:
    return [row for row in aggregates if row["candidate_name"] == name]


def _recommend_contact_condition(condition_rows: list[dict[str, object]]) -> dict[str, object]:
    if not condition_rows:
        return {}
    ranked = sorted(
        condition_rows,
        key=lambda row: (
            float(row["selection_score"]),
            float(row["hole_contact_episode_rate"]),
            float(row["contact_fraction_mean"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    return {
        "recommended_contact_condition": {
            "condition_name": selected["condition_name"],
            "condition_axis": selected.get("condition_axis"),
            "condition_value": selected.get("condition_value"),
            "contact_condition_name": CALIBRATION_DEFAULT_CONTACT_CONDITION if selected["condition_name"] == CALIBRATION_DEFAULT_CONTACT_CONDITION else selected["condition_name"],
            "teleop_noise_xy_amplitude": selected.get("teleop_noise_xy_amplitude"),
            "teleop_noise_cycles": selected.get("teleop_noise_cycles"),
            "teleop_noise_phase_x": selected.get("teleop_noise_phase_x"),
            "teleop_noise_phase_y": selected.get("teleop_noise_phase_y"),
            "clearance_delta": selected.get("clearance_delta"),
            "friction_scale": selected.get("friction_scale"),
            "peg_tilt_x": selected.get("peg_tilt_x"),
            "peg_tilt_y": selected.get("peg_tilt_y"),
            "hole_yaw_offset": selected.get("hole_yaw_offset"),
        }
    }


def _condition_metadata(profile: ControlledContactProfile) -> dict[str, object]:
    payload = dict(profile.to_metadata())
    payload.update(
        {
            "teleop_noise_xy_amplitude": float(profile.teleop_noise_xy_amplitude),
            "teleop_noise_cycles": float(profile.teleop_noise_cycles),
            "teleop_noise_phase_x": float(profile.teleop_noise_phase_x),
            "teleop_noise_phase_y": float(profile.teleop_noise_phase_y),
            "clearance_delta": float(profile.clearance_delta),
            "friction_scale": float(profile.friction_scale),
            "peg_tilt_x": float(profile.peg_tilt_x),
            "peg_tilt_y": float(profile.peg_tilt_y),
            "hole_yaw_offset": float(profile.hole_yaw_offset),
        }
    )
    return payload


def evaluate_controller_condition_calibration(
    *,
    scene: str,
    setting: str,
    profile_name: str,
    contact_condition_name: str,
    episodes: int,
    seed: int,
    renderer_mode: str,
    output_root: Path,
    gain_config: Path,
    controller_candidate_set: str = "full",
    include_condition_sweep: bool = False,
    trace_stride: int = 0,
    smoothing_alpha: float = 0.2,
    policy_update_period_steps: int = 6,
    hold_between_updates: bool = True,
    dry_run: bool = False,
    hole_xy_radius: float = 0.02,
    fixed_teleop_noise_xy_amplitude: float = 0.0010,
    fixed_teleop_noise_cycles: float = 1.0,
    fixed_teleop_noise_phase_x: float = 0.0,
    fixed_teleop_noise_phase_y: float = 1.5707963267948966,
    fixed_clearance_delta: float = -0.0004,
    fixed_friction_scale: float = 1.15,
    fixed_peg_tilt_x: float = 0.0087,
    fixed_peg_tilt_y: float = -0.0087,
    fixed_hole_yaw_offset: float = 0.0,
) -> dict[str, object]:
    if scene != "circle" or setting != "circle_calibrated_v1":
        raise ValueError("Track A controller/condition calibration currently supports only circle / circle_calibrated_v1.")
    if renderer_mode != "native":
        raise ValueError("Calibration requires --renderer-mode native.")
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if trace_stride < 0:
        raise ValueError("trace_stride must be non-negative.")

    thresholds = BadCaseThresholds()
    rollout_config = RolloutConfig(config_path=get_scene_spec(scene).config_path, gain_config_path=gain_config)
    controller_candidates = build_controller_candidates(gain_config, controller_candidate_set=controller_candidate_set)
    condition_candidates = build_condition_candidates(include_condition_sweep=include_condition_sweep, hole_xy_radius=hole_xy_radius)

    if dry_run:
        return {
            "status": "dry_run",
            "scene": scene,
            "setting": setting,
            "profile_name": profile_name,
            "contact_condition_name": contact_condition_name,
            "episodes": episodes,
            "seed": seed,
            "controller_candidate_set": controller_candidate_set,
            "controller_candidates": [
                {
                    "candidate_name": candidate.candidate_name,
                    "candidate_group": candidate.candidate_group,
                    "position_stiffness_matrix": None
                    if candidate.position_stiffness_matrix is None
                    else np.asarray(candidate.position_stiffness_matrix, dtype=float).tolist(),
                    "requested_profile": candidate.controller_spec.requested_profile,
                }
                for candidate in controller_candidates
            ],
            "condition_candidates": [
                {
                    "condition_name": condition.condition_name,
                    "condition_axis": condition.condition_axis,
                    "condition_value": condition.condition_value,
                }
                for condition in condition_candidates
            ],
        }

    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = f"{scene}_{setting}_{controller_candidate_set}_{seed}_{episodes}ep"
    output_dir = output_root / run_stamp
    rerun = 0
    while output_dir.exists():
        rerun += 1
        output_dir = output_root / f"{run_stamp}_rerun{rerun}"
    output_dir.mkdir(parents=True)

    candidate_rows: list[dict[str, object]] = []
    bad_case_records: list[dict[str, object]] = []
    episode_spec_paths: list[str] = []
    condition_summaries: list[dict[str, object]] = []
    run_id = f"track_a_controller_condition_calibration_{seed}_{episodes}ep"
    controlled_condition_override = make_controlled_contact_profile(
        profile_name=profile_name,
        contact_condition_name=contact_condition_name,
        hole_xy_radius=float(hole_xy_radius),
        teleop_noise_xy_amplitude=fixed_teleop_noise_xy_amplitude,
        teleop_noise_cycles=fixed_teleop_noise_cycles,
        teleop_noise_phase_x=fixed_teleop_noise_phase_x,
        teleop_noise_phase_y=fixed_teleop_noise_phase_y,
        clearance_delta=fixed_clearance_delta,
        friction_scale=fixed_friction_scale,
        peg_tilt_x=fixed_peg_tilt_x,
        peg_tilt_y=fixed_peg_tilt_y,
        hole_yaw_offset=fixed_hole_yaw_offset,
    )

    for condition_index, condition in enumerate(condition_candidates):
        condition_seed = seed + condition_index * 1000
        episode_specs = _episode_specs_for_condition(
            scene=scene,
            setting=setting,
            episodes=episodes,
            seed=condition_seed,
            rollout_config=rollout_config,
            profile_name=profile_name,
            controlled_contact_profile=condition.controlled_contact_profile
            if include_condition_sweep
            else controlled_condition_override,
        )
        episode_spec_path = output_dir / "episode_specs" / f"{condition.condition_name}.jsonl"
        from stiffness_copilot_mujoco.episodes.episode_spec import write_episode_specs_jsonl  # local import to keep module scope compact

        write_episode_specs_jsonl(episode_spec_path, episode_specs)
        episode_spec_paths.append(str(episode_spec_path))

        condition_rows: list[dict[str, object]] = []
        for controller_candidate in controller_candidates:
            for smoothing_enabled in CALIBRATION_TOGGLE_STATES:
                smoothing_config = StiffnessCommandSmoothingConfig(
                    enabled=bool(smoothing_enabled),
                    method="diagonal_ema",
                    alpha=float(smoothing_alpha),
                    policy_update_period_steps=int(policy_update_period_steps),
                    hold_between_updates=bool(hold_between_updates),
                )
                if controller_candidate.candidate_group == "collection":
                    baseline_name = "track_a_c600"
                    gains = controller_candidate.controller_spec.gains
                else:
                    baseline_name = "mid_high_baseline"
                    gains = controller_candidate.controller_spec.gains
                rows_for_run: list[dict[str, object]] = []
                for spec in episode_specs:
                    trace = TraceAccumulator()
                    summary = run_fixed_stiffness_episode(
                        baseline=baseline_name,  # type: ignore[arg-type]
                        seed=int(spec.seed),
                        xy_offset=np.zeros(2, dtype=float),
                        config=rollout_config,
                        episode_id=int(spec.episode_id),
                        gains=gains,
                        profile=controller_candidate.controller_spec.selected_profile,
                        position_stiffness_matrix=controller_candidate.position_stiffness_matrix,
                        stiffness_smoothing=smoothing_config,
                        perturbation=None,
                        episode_spec=spec,
                        step_callback=_trace_callback(trace),
                    )
                    row, bad_case_record, admission_label = _episode_row(
                        run_id=run_id,
                        condition_name=f"{condition.condition_name}:{controller_candidate.candidate_name}:{'smooth' if smoothing_enabled else 'raw'}",
                        controller_candidate=controller_candidate,
                        summary=summary,
                        trace=trace,
                        thresholds=thresholds,
                        smoothing_config=smoothing_config,
                    )
                    row["episode_spec_id"] = spec.episode_spec_id
                    row["trajectory_family"] = spec.trajectory_family
                    row["trajectory_family_id"] = int(spec.trajectory_family_id)
                    row["trajectory_total_steps"] = int(spec.total_steps)
                    row["hole_xy_offset"] = _json_ready(spec.hole_xy_offset)
                    row["actual_hole_xy"] = _json_ready(spec.actual_hole_xy)
                    row["contact_condition_name"] = condition.condition_name
                    row["condition_axis"] = condition.condition_axis
                    row["condition_value"] = condition.condition_value
                    row["teleop_noise_xy_amplitude"] = float(condition.controlled_contact_profile.teleop_noise_xy_amplitude)
                    row["teleop_noise_cycles"] = float(condition.controlled_contact_profile.teleop_noise_cycles)
                    row["teleop_noise_phase_x"] = float(condition.controlled_contact_profile.teleop_noise_phase_x)
                    row["teleop_noise_phase_y"] = float(condition.controlled_contact_profile.teleop_noise_phase_y)
                    row["clearance_delta"] = float(condition.controlled_contact_profile.clearance_delta)
                    row["friction_scale"] = float(condition.controlled_contact_profile.friction_scale)
                    row["peg_tilt_x"] = float(condition.controlled_contact_profile.peg_tilt_x)
                    row["peg_tilt_y"] = float(condition.controlled_contact_profile.peg_tilt_y)
                    row["hole_yaw_offset"] = float(condition.controlled_contact_profile.hole_yaw_offset)
                    rows_for_run.append(row)
                    if admission_label != "eligible":
                        bad_case_records.append(bad_case_record)
                    if trace_stride > 0:
                        for trace_row in trace.trace_rows or []:
                            if int(trace_row["step"]) % trace_stride == 0:
                                trace_row = dict(trace_row)
                                trace_row.update(
                                    {
                                        "condition_name": condition.condition_name,
                                        "controller_name": controller_candidate.candidate_name,
                                        "smoothing_enabled": bool(smoothing_enabled),
                                        "episode_spec_id": spec.episode_spec_id,
                                    }
                                )
                    print(
                        f"condition={condition.condition_name} controller={controller_candidate.candidate_name} "
                        f"smoothing={bool(smoothing_enabled)} episode={spec.episode_id} "
                        f"contact={bool(summary.contact_detected)} depth_reached={bool(summary.depth_reached)} "
                        f"max_normal_force={float(summary.max_normal_force):.6f} final_depth={float(summary.final_depth):.6f}",
                        flush=True,
                    )
                aggregate = _aggregate_rows(
                    rows_for_run,
                    thresholds=thresholds,
                    condition_name=condition.condition_name,
                    controller_candidate=controller_candidate,
                    smoothing_config=smoothing_config,
                )
                aggregate_row = dict(aggregate)
                aggregate_row.update(
                    {
                        "condition_name": condition.condition_name,
                        "condition_axis": condition.condition_axis,
                        "condition_value": condition.condition_value,
                        "contact_condition_name": condition.condition_name,
                        "teleop_noise_xy_amplitude": float(condition.controlled_contact_profile.teleop_noise_xy_amplitude),
                        "teleop_noise_cycles": float(condition.controlled_contact_profile.teleop_noise_cycles),
                        "teleop_noise_phase_x": float(condition.controlled_contact_profile.teleop_noise_phase_x),
                        "teleop_noise_phase_y": float(condition.controlled_contact_profile.teleop_noise_phase_y),
                        "clearance_delta": float(condition.controlled_contact_profile.clearance_delta),
                        "friction_scale": float(condition.controlled_contact_profile.friction_scale),
                        "peg_tilt_x": float(condition.controlled_contact_profile.peg_tilt_x),
                        "peg_tilt_y": float(condition.controlled_contact_profile.peg_tilt_y),
                        "hole_yaw_offset": float(condition.controlled_contact_profile.hole_yaw_offset),
                        "selection_score": float(
                            _collection_score(aggregate)
                            if controller_candidate.candidate_group == "collection"
                            else _baseline_score(aggregate)
                        ),
                        "selection_score_label": "",
                        "smoothing_stability_ok": False,
                    }
                )
                aggregate_row["selection_score_label"] = _candidate_score_label(float(aggregate_row["selection_score"]))
                aggregate_row.update(
                    {
                        "smoothing_enabled": bool(smoothing_config.enabled),
                        "smoothing_method": str(smoothing_config.method),
                        "smoothing_alpha": float(smoothing_config.alpha),
                        "policy_update_period_steps": int(smoothing_config.resolved_policy_update_period_steps()),
                        "hold_between_updates": bool(smoothing_config.hold_between_updates),
                    }
                )
                condition_rows.append(aggregate_row)
                candidate_rows.append(aggregate_row)
        best_collection = _best_candidate_by_group(condition_rows, group="collection")
        best_baseline = _best_candidate_by_group(condition_rows, group="baseline")
        condition_summaries.append(
            {
                "condition_name": condition.condition_name,
                "condition_axis": condition.condition_axis,
                "condition_value": condition.condition_value,
                "condition_profile": _condition_metadata(condition.controlled_contact_profile),
                "best_collection_candidate": None if best_collection is None else best_collection["candidate_name"],
                "best_baseline_candidate": None if best_baseline is None else best_baseline["candidate_name"],
                "collection_score": None if best_collection is None else float(best_collection["selection_score"]),
                "baseline_score": None if best_baseline is None else float(best_baseline["selection_score"]),
                "candidate_metrics": condition_rows,
            }
        )

    candidate_metrics_path = output_dir / "controller_condition_calibration_candidate_metrics.csv"
    smoothing_pairs: list[dict[str, object]] = []
    csv_fields = [
        "condition_name",
        "condition_axis",
        "condition_value",
        "contact_condition_name",
        "controller_group",
        "candidate_name",
        "controller_role",
        "smoothing_enabled",
        "smoothing_method",
        "smoothing_alpha",
        "policy_update_period_steps",
        "hold_between_updates",
        "teleop_noise_xy_amplitude",
        "teleop_noise_cycles",
        "teleop_noise_phase_x",
        "teleop_noise_phase_y",
        "clearance_delta",
        "friction_scale",
        "peg_tilt_x",
        "peg_tilt_y",
        "hole_yaw_offset",
        "episode_count",
        "depth_reached_rate",
        "hole_contact_episode_rate",
        "contact_detected_rate",
        "contact_fraction_mean",
        "contact_fraction_median",
        "contact_onset_mean",
        "contact_onset_median",
        "mean_max_normal_force",
        "p95_max_normal_force",
        "p99_max_normal_force",
        "mean_max_tangential_force",
        "p95_max_tangential_force",
        "p99_max_tangential_force",
        "mean_normal_force_contact",
        "low_force_success_rate_100N",
        "low_force_success_rate_150N",
        "low_force_success_rate_200N",
        "catastrophic_count",
        "solver_spike_count",
        "torque_saturation_total",
        "torque_saturation_episode_rate",
        "mean_final_depth",
        "mean_final_lateral_error",
        "mean_max_position_error",
        "mean_max_lateral_tracking_error",
        "mean_max_vertical_tracking_error",
        "bad_case_count",
        "admission_recommendation",
        "selection_score",
        "selection_score_label",
        "smoothing_stability_ok",
    ]
    for condition_name in sorted({str(row["condition_name"]) for row in candidate_rows}):
        for candidate_name in sorted({str(row["candidate_name"]) for row in candidate_rows if str(row["condition_name"]) == condition_name}):
            pair_rows = [
                row
                for row in candidate_rows
                if str(row["condition_name"]) == condition_name and str(row["candidate_name"]) == candidate_name
            ]
            raw_rows = [row for row in pair_rows if not bool(row["smoothing_enabled"])]
            smooth_rows = [row for row in pair_rows if bool(row["smoothing_enabled"])]
            if raw_rows and smooth_rows:
                delta = _smoothing_delta_summary(raw_rows[0], smooth_rows[0])
                for row in pair_rows:
                    row["smoothing_stability_ok"] = bool(delta["stability_ok"])
                smoothing_pairs.append(
                    {
                        "condition_name": condition_name,
                        "candidate_name": candidate_name,
                        "stability_delta": delta,
                    }
                )

    _write_csv(candidate_metrics_path, candidate_rows, fieldnames=csv_fields)
    bad_case_registry_path = output_dir / "controller_condition_calibration_bad_cases.jsonl"
    write_bad_case_registry(bad_case_records, bad_case_registry_path)

    collection_candidates = [row for row in candidate_rows if row["controller_group"] == "collection" and not bool(row["smoothing_enabled"])]
    baseline_candidates = [row for row in candidate_rows if row["controller_group"] == "baseline" and not bool(row["smoothing_enabled"])]
    recommended_collection = max(collection_candidates, key=lambda row: float(row["selection_score"]), default=None)
    recommended_baseline = max(baseline_candidates, key=lambda row: float(row["selection_score"]), default=None)
    recommended_condition = _recommend_contact_condition(
        [row for row in candidate_rows if row["controller_group"] == "collection" and not bool(row["smoothing_enabled"])]
    )
    more_sweep_needed = bool(
        recommended_collection is None
        or recommended_baseline is None
        or float(recommended_collection.get("selection_score", 0.0)) < 1.5
        or float(recommended_baseline.get("selection_score", 0.0)) < 1.0
        or not include_condition_sweep
    )

    summary_payload = {
        "status": "passed",
        "scene": scene,
        "setting": setting,
        "profile_name": profile_name,
        "contact_condition_name": contact_condition_name,
        "episodes": episodes,
        "seed": seed,
        "controller_candidate_set": controller_candidate_set,
        "condition_sweep_enabled": bool(include_condition_sweep),
        "condition_candidates": [
            {
                "condition_name": condition.condition_name,
                "condition_axis": condition.condition_axis,
                "condition_value": condition.condition_value,
            }
            for condition in condition_candidates
        ],
        "candidate_metrics_path": str(candidate_metrics_path),
        "bad_case_registry_path": str(bad_case_registry_path),
        "episode_spec_paths": episode_spec_paths,
        "condition_summaries": condition_summaries,
        "recommended_collection_controller": recommended_collection,
        "recommended_baseline_controller": recommended_baseline,
        "recommended_contact_condition": recommended_condition.get("recommended_contact_condition"),
        "more_sweep_needed": bool(more_sweep_needed),
        "smoothing_stability": smoothing_pairs,
    }
    metadata_payload = {
        "scene": scene,
        "setting": setting,
        "profile_name": profile_name,
        "contact_condition_name": contact_condition_name,
        "episodes": episodes,
        "seed": seed,
        "controller_candidate_set": controller_candidate_set,
        "controller_candidates": [
            {
                "candidate_name": candidate.candidate_name,
                "candidate_group": candidate.candidate_group,
                "controller_spec": candidate.controller_spec.to_dict(simulation_dt_seconds=0.002),
                "position_stiffness_matrix": None
                if candidate.position_stiffness_matrix is None
                else np.asarray(candidate.position_stiffness_matrix, dtype=float).tolist(),
                "position_stiffness_matrix_source": candidate.position_stiffness_matrix_source,
            }
            for candidate in controller_candidates
        ],
        "condition_candidates": [
            {
                "condition_name": condition.condition_name,
                "condition_axis": condition.condition_axis,
                "condition_value": condition.condition_value,
                "controlled_contact_profile": _condition_metadata(condition.controlled_contact_profile),
            }
            for condition in condition_candidates
        ],
        "thresholds": {
            "catastrophic_force_threshold": thresholds.catastrophic_force_threshold,
            "low_force_success_threshold": thresholds.low_force_success_threshold,
            "raw_to_p99_spike_ratio": thresholds.raw_to_p99_spike_ratio,
            "raw_to_filtered_spike_ratio": thresholds.raw_to_filtered_spike_ratio,
            "solver_spike_max_high_force_steps": thresholds.solver_spike_max_high_force_steps,
            "jamming_min_high_force_steps": thresholds.jamming_min_high_force_steps,
            "jamming_min_high_force_fraction": thresholds.jamming_min_high_force_fraction,
            "depth_stall_epsilon": thresholds.depth_stall_epsilon,
            "persistent_contact_min_steps": thresholds.persistent_contact_min_steps,
        },
        "fixed_contact_condition": _condition_metadata(controlled_condition_override),
        "rolling_output_dir": str(output_dir),
    }

    summary_path = output_dir / "controller_condition_calibration_summary.json"
    metadata_path = output_dir / "controller_condition_calibration_metadata.json"
    report_path = output_dir / "controller_condition_calibration_report.md"
    recommended_roles_path = output_dir / "recommended_track_a_roles.json"
    manual_commands_path = output_dir / "manual_commands.md"
    _write_json(summary_path, summary_payload)
    _write_json(metadata_path, metadata_payload)
    _write_json(recommended_roles_path, {
        "recommended_collection_controller": None if recommended_collection is None else {
            "candidate_name": recommended_collection["candidate_name"],
            "controller_group": recommended_collection["controller_group"],
            "selection_score": recommended_collection["selection_score"],
            "selection_score_label": recommended_collection["selection_score_label"],
            "smoothing_enabled": recommended_collection["smoothing_enabled"],
        },
        "recommended_baseline_controller": None if recommended_baseline is None else {
            "candidate_name": recommended_baseline["candidate_name"],
            "controller_group": recommended_baseline["controller_group"],
            "selection_score": recommended_baseline["selection_score"],
            "selection_score_label": recommended_baseline["selection_score_label"],
            "smoothing_enabled": recommended_baseline["smoothing_enabled"],
        },
        "recommended_contact_condition": recommended_condition.get("recommended_contact_condition"),
        "reasoning": [
            "Collection controller should be chosen by contact richness first, not by balanced success/failure.",
            "Baseline controller should stay inside a working-but-imperfect band with no solver pathology.",
            "Smoothing is a stability sanity check for fixed controllers, not the main calibration axis.",
        ],
        "known_risks": [
            "Condition sweep is sequential and intentionally not factorial.",
            "Observed calibration is MuJoCo-specific and may not transfer to a different contact model.",
            "A single 20-episode batch can still under-sample rare jam tails.",
        ],
        "whether_more_sweep_needed": bool(more_sweep_needed),
    })
    manual_commands = [
        "# Track A controller/condition calibration",
        "",
        "1. Tiny smoke (2 episodes, current collection + current baseline only):",
        "",
        "```bash",
        "cd /Users/lyra/Desktop/MasterThesis/stiffness_copilot_mujoco",
        "",
        "/opt/anaconda3/bin/mjpython scripts/calibrate_track_a_controller_roles.py \\",
        "  --scene circle \\",
        "  --setting circle_calibrated_v1 \\",
        "  --profile-name circle_calibrated_v1_global_hole_fixed_contact \\",
        "  --contact-condition-name light \\",
        "  --renderer-mode native \\",
        "  --episodes 2 \\",
        "  --seed 9100 \\",
        "  --controller-candidate-set current \\",
        "  --output-root /Users/lyra/Desktop/MasterThesis/stiffness_copilot_mujoco/artifacts/evaluations/track_a_controller_condition_calibration/smoke_2ep",
        "```",
        "",
        "2. Controller-only sweep (20 episodes, C375/C400/C425/C450 and B300/B325/B350):",
        "",
        "```bash",
        "cd /Users/lyra/Desktop/MasterThesis/stiffness_copilot_mujoco",
        "",
        "/opt/anaconda3/bin/mjpython scripts/calibrate_track_a_controller_roles.py \\",
        "  --scene circle \\",
        "  --setting circle_calibrated_v1 \\",
        "  --profile-name circle_calibrated_v1_global_hole_fixed_contact \\",
        "  --contact-condition-name light \\",
        "  --renderer-mode native \\",
        "  --episodes 20 \\",
        "  --seed 9200 \\",
        "  --controller-candidate-set full \\",
        "  --output-root /Users/lyra/Desktop/MasterThesis/stiffness_copilot_mujoco/artifacts/evaluations/track_a_controller_condition_calibration/controller_sweep_20ep",
        "```",
        "",
        "3. Optional follow-up condition sweep (run only after inspecting the controller-only sweep):",
        "",
        "```bash",
        "cd /Users/lyra/Desktop/MasterThesis/stiffness_copilot_mujoco",
        "",
        "/opt/anaconda3/bin/mjpython scripts/calibrate_track_a_controller_roles.py \\",
        "  --scene circle \\",
        "  --setting circle_calibrated_v1 \\",
        "  --profile-name circle_calibrated_v1_global_hole_fixed_contact \\",
        "  --contact-condition-name light \\",
        "  --renderer-mode native \\",
        "  --episodes 20 \\",
        "  --seed 9300 \\",
        "  --controller-candidate-set full \\",
        "  --condition-sweep \\",
        "  --output-root /Users/lyra/Desktop/MasterThesis/stiffness_copilot_mujoco/artifacts/evaluations/track_a_controller_condition_calibration/condition_sweep_20ep",
        "```",
    ]
    manual_commands_path.write_text("\n".join(manual_commands) + "\n", encoding="utf-8")
    report_lines = [
        "# Track A Controller / Condition Role Calibration Sweep",
        "",
        "This calibration sweep is implemented but runtime results are pending manual execution.",
        "",
        "## Role recommendation",
        "",
        "- collection controller: pending runtime sweep",
        "- baseline controller: pending runtime sweep",
        "- contact condition: pending runtime sweep",
        "",
        "## Low-force thresholds",
        "",
        "- 100 N",
        "- 150 N",
        "- 200 N",
        "",
        "## Admission / quarantine",
        "",
        "- `eligible`",
        "- `high_force_warning`",
        "- `torque_saturation_warning`",
        "- `solver_spike_quarantine`",
        "- `catastrophic_quarantine`",
        "",
        "## Next experiment",
        "",
        "Run the controller-only sweep first, then decide whether the condition sweep is worth the runtime.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "status": "passed",
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "metadata_path": str(metadata_path),
        "candidate_metrics_path": str(candidate_metrics_path),
        "bad_case_registry_path": str(bad_case_registry_path),
        "report_path": str(report_path),
        "recommended_roles_path": str(recommended_roles_path),
        "manual_commands_path": str(manual_commands_path),
        "recommended_collection_controller": recommended_collection,
        "recommended_baseline_controller": recommended_baseline,
        "recommended_contact_condition": recommended_condition.get("recommended_contact_condition"),
        "more_sweep_needed": bool(more_sweep_needed),
    }


__all__ = [
    "CALIBRATION_BASELINE_GAIN_CANDIDATES",
    "CALIBRATION_COLLECTION_GAIN_CANDIDATES",
    "CALIBRATION_CONTROLLER_CANDIDATE_SETS",
    "CALIBRATION_DEFAULT_CONTACT_CONDITION",
    "CALIBRATION_DEFAULT_CONTROLLED_CONTACT_PROFILE",
    "CALIBRATION_DEFAULT_PROFILE_NAME",
    "CALIBRATION_LOW_FORCE_THRESHOLDS",
    "CALIBRATION_OUTPUT_ROOT",
    "CalibrationConditionCandidate",
    "CalibrationControllerCandidate",
    "CalibrationRunPaths",
    "build_condition_candidates",
    "build_controller_candidates",
    "build_default_contact_condition",
    "evaluate_controller_condition_calibration",
]
