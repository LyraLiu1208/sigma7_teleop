from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import (
    ContactQuery,
    contact_state_vector,
    extract_contact_state,
    extract_net_peg_hole_contact_force_world,
)
from stiffness_copilot_mujoco.controllers.impedance import (
    TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE,
    task_space_impedance_torque,
)
from stiffness_copilot_mujoco.controllers.track_a_controllers import (
    DEFAULT_TRACK_A_CONTROLLERS_YAML,
    load_track_a_controller_runtime,
)
from stiffness_copilot_mujoco.episodes.episode_spec import (
    EPISODE_SPEC_SCHEMA_VERSION,
    EPISODE_TRAJECTORY_SOURCE_OPEN_LOOP_FAMILY,
    EpisodeSpec,
    write_episode_specs_jsonl,
)
from stiffness_copilot_mujoco.episodes.teleop_proxy import (
    TELEOP_MODE_POSITION_ORIENTATION,
    TELEOP_MODE_POSITION_ONLY,
    build_target_orientations,
    nominal_hole_rotation,
    validate_teleop_mode,
)
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.learning.dataset_collection import _action_row, _state_row
from stiffness_copilot_mujoco.learning.residual_dataset import validate_residual_dataset
from stiffness_copilot_mujoco.learning.frozen_train_val_split import FrozenTrainValSplit
from stiffness_copilot_mujoco.learning.residual_label_projection import (
    DEFAULT_BASELINE_K,
    DEFAULT_CONTACT_GATE_HIGH,
    DEFAULT_CONTACT_GATE_LOW,
    DEFAULT_DIAGNOSTIC_K_MAX,
    DEFAULT_DIAGNOSTIC_K_MIN,
    DEFAULT_L21_COUPLING_PERCENTILE,
    DEFAULT_MIN_CALIBRATION_SAMPLES,
    DEFAULT_NEUTRAL_CONTACT_THRESHOLD,
    LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
    LABEL_PROJECTION_LEGACY,
    LABEL_PROJECTION_RESIDUAL_FIRST,
    project_residual_first_labels,
)
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.learning.stiffness_labels import (
    StiffnessLabelConfig,
    build_stiffness_labels_with_diagnostics,
)
from stiffness_copilot_mujoco.learning.privileged_task_state import describe_privileged_task_state_schema
from stiffness_copilot_mujoco.learning.task_state import peg_hole_task_state
from stiffness_copilot_mujoco.metrics.task_metrics import geometry_from_config, hole_center_position, load_scene_config, peg_tip_position
from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl
from stiffness_copilot_mujoco.pose_math import site_rotation
from stiffness_copilot_mujoco.runtime_defaults import default_native_launcher
from stiffness_copilot_mujoco.robustness import (
    ControlledContactProfile,
    get_robustness_preset,
    sample_controlled_contact_perturbations,
    sample_robustness_perturbations,
)
from stiffness_copilot_mujoco.rollout_observation import collect_step, reset_from_config
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    RolloutConfig,
    RolloutPerturbation,
    cleanup_runtime_scene,
    clip_torque,
    scene_for_rollout,
    target_rotation_with_peg_tilt,
)
from stiffness_copilot_mujoco.scenes import get_scene_spec
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
from stiffness_copilot_mujoco.sim.scene import (
    ROOT,
    apply_eye_in_hand_camera_pose,
    canonical_eye_in_hand_camera_pose,
    eye_in_hand_camera_pose_from_config,
    validate_canonical_eye_in_hand_camera_config,
)
from stiffness_copilot_mujoco.vision.rendering import MujocoRgbRenderer
from stiffness_copilot_mujoco.learning.vision_residual_dataset import infer_training_data_valid


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "datasets" / "residual_bc_residual_first_v1"
DEFAULT_GAIN_CONFIG = ROOT / "configs" / "controllers" / "fixed_impedance.yaml"
DEFAULT_COLLECTION_CONTROLLER_ID = "track_a_c600"

TRAJECTORY_FAMILIES = (
    "direct_insertion",
    "diagonal_insertion",
    "lateral_sweep_insertion",
    "arc_approach_insertion",
    "decaying_spiral_insertion",
)
TRAJECTORY_FAMILY_TO_ID = {name: idx for idx, name in enumerate(TRAJECTORY_FAMILIES)}
TRAJECTORY_PARAMETER_FIELDS = (
    "approach_steps",
    "descend_steps",
    "insert_steps",
    "hold_steps",
    "approach_height",
    "descend_height",
    "insert_depth",
    "initial_x",
    "initial_y",
    "terminal_x",
    "terminal_y",
    "curve_radius",
    "curve_angle",
    "curve_turns",
    "curve_phase",
    "duration_scale",
)
RANDOMIZATION_FIELDS = (
    "hole_x",
    "hole_y",
    "hole_yaw",
    "clearance_delta",
    "friction_scale",
    "peg_tilt_x",
    "peg_tilt_y",
    "teleop_noise_xy_amplitude",
    "teleop_noise_cycles",
)


@dataclass(frozen=True)
class OpenLoopTrajectoryPlan:
    family: str
    family_id: int
    parameters: np.ndarray
    target_offsets: np.ndarray
    phase_ids: np.ndarray

    @property
    def total_steps(self) -> int:
        return int(self.target_offsets.shape[0] - 1)


@dataclass(frozen=True)
class OpenLoopCollectionResult:
    output_dir: Path
    raw_dataset: Path
    eligible_dataset: Path | None
    collection_summary: Path
    episode_csv: Path
    episode_specs: Path
    family_summary: Path


def _sample_xy(rng: np.random.Generator, radius_min: float, radius_max: float) -> np.ndarray:
    radius = float(rng.uniform(radius_min, radius_max))
    angle = float(rng.uniform(-np.pi, np.pi))
    return radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _phase_progress(step: int, start: int, length: int) -> float:
    return float(np.clip((step - start) / max(length - 1, 1), 0.0, 1.0))


def _trajectory_xy(
    family: str,
    *,
    step: int,
    approach_steps: int,
    descend_steps: int,
    insert_steps: int,
    initial_xy: np.ndarray,
    terminal_xy: np.ndarray,
    curve_radius: float,
    curve_angle: float,
    curve_turns: float,
    curve_phase: float,
) -> np.ndarray:
    descend_start = approach_steps
    insert_start = approach_steps + descend_steps
    hold_start = insert_start + insert_steps
    if step < descend_start:
        return initial_xy.copy()

    if family == "direct_insertion":
        return terminal_xy.copy()

    if family == "diagonal_insertion":
        if step < insert_start:
            progress = _phase_progress(step, descend_start, descend_steps)
        elif step < hold_start:
            progress = _phase_progress(step, insert_start, insert_steps)
            progress = 0.75 + 0.25 * progress
        else:
            progress = 1.0
        return (1.0 - progress) * initial_xy + progress * terminal_xy

    if family == "lateral_sweep_insertion":
        direction = np.array([np.cos(curve_angle), np.sin(curve_angle)], dtype=float)
        if step < insert_start:
            progress = _phase_progress(step, descend_start, descend_steps)
            base = (1.0 - progress) * initial_xy + progress * terminal_xy
            sweep = curve_radius * np.sin(np.pi * progress) * np.sin(2.0 * np.pi * progress)
            return base + sweep * direction
        return terminal_xy.copy()

    if family == "arc_approach_insertion":
        if step < insert_start:
            progress = _phase_progress(step, descend_start, descend_steps)
            angle = curve_phase + curve_angle * progress
            radius = curve_radius * (1.0 - 0.75 * progress)
            arc = radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
            return terminal_xy + arc
        return terminal_xy.copy()

    if family == "decaying_spiral_insertion":
        if step < hold_start:
            combined_length = descend_steps + insert_steps
            progress = _phase_progress(step, descend_start, combined_length)
            radius = curve_radius * (1.0 - progress)
            angle = curve_phase + 2.0 * np.pi * curve_turns * progress
            spiral = radius * np.array([np.cos(angle), np.sin(angle)], dtype=float)
            return terminal_xy + spiral
        return terminal_xy.copy()

    raise ValueError(f"Unsupported trajectory family {family!r}.")


def _trajectory_z(
    step: int,
    *,
    approach_steps: int,
    descend_steps: int,
    insert_steps: int,
    approach_height: float,
    descend_height: float,
    insert_depth: float,
) -> tuple[float, int]:
    descend_start = approach_steps
    insert_start = approach_steps + descend_steps
    hold_start = insert_start + insert_steps
    if step < descend_start:
        return approach_height, 0
    if step < insert_start:
        progress = _phase_progress(step, descend_start, descend_steps)
        return approach_height + progress * (descend_height - approach_height), 1
    if step < hold_start:
        progress = _phase_progress(step, insert_start, insert_steps)
        return descend_height + progress * (-insert_depth - descend_height), 2
    return -insert_depth, 3


def _teleop_noise(
    *,
    step: int,
    total_steps: int,
    perturbation: RolloutPerturbation,
) -> np.ndarray:
    amplitude = float(perturbation.teleop_noise_xy_amplitude)
    if amplitude <= 0.0:
        return np.zeros(2, dtype=float)
    progress = float(np.clip(step / max(total_steps, 1), 0.0, 1.0))
    envelope = float(np.sin(np.pi * progress))
    angle = 2.0 * np.pi * float(perturbation.teleop_noise_cycles) * progress
    return amplitude * envelope * np.array(
        [
            np.sin(angle + float(perturbation.teleop_noise_phase_x)),
            np.sin(angle + float(perturbation.teleop_noise_phase_y)),
        ],
        dtype=float,
    )


def generate_open_loop_trajectory(
    *,
    family: str,
    rng: np.random.Generator,
    perturbation: RolloutPerturbation,
    base_config: RolloutConfig,
) -> OpenLoopTrajectoryPlan:
    if family not in TRAJECTORY_FAMILY_TO_ID:
        raise KeyError(f"Unknown trajectory family {family!r}.")
    duration_scale = float(rng.uniform(0.8, 1.2))
    approach_steps = max(1, int(round(base_config.approach_hold_steps * duration_scale)))
    descend_steps = max(2, int(round(base_config.descend_steps * float(rng.uniform(0.8, 1.2)))))
    insert_steps = max(2, int(round(base_config.insert_steps * float(rng.uniform(0.8, 1.2)))))
    hold_steps = max(1, int(round(base_config.final_hold_steps * float(rng.uniform(0.8, 1.2)))))

    terminal_xy = _sample_xy(rng, 0.0, 0.0008)
    curve_radius = 0.0
    curve_angle = 0.0
    curve_turns = 0.0
    curve_phase = float(rng.uniform(-np.pi, np.pi))

    if family == "direct_insertion":
        initial_xy = _sample_xy(rng, 0.0, 0.0010)
        terminal_xy = initial_xy.copy()
    elif family == "diagonal_insertion":
        initial_xy = _sample_xy(rng, 0.0010, 0.0025)
    elif family == "lateral_sweep_insertion":
        curve_radius = float(rng.uniform(0.0010, 0.0025))
        curve_angle = float(rng.uniform(-np.pi, np.pi))
        initial_xy = terminal_xy - curve_radius * np.array(
            [np.cos(curve_angle), np.sin(curve_angle)],
            dtype=float,
        )
    elif family == "arc_approach_insertion":
        curve_radius = float(rng.uniform(0.0008, 0.0020))
        curve_angle = float(rng.uniform(np.deg2rad(45.0), np.deg2rad(120.0)))
        if rng.random() < 0.5:
            curve_angle = -curve_angle
        initial_xy = terminal_xy + curve_radius * np.array(
            [np.cos(curve_phase), np.sin(curve_phase)],
            dtype=float,
        )
    elif family == "decaying_spiral_insertion":
        curve_radius = float(rng.uniform(0.0005, 0.0018))
        curve_turns = float(rng.uniform(0.5, 1.5))
        initial_xy = terminal_xy + curve_radius * np.array(
            [np.cos(curve_phase), np.sin(curve_phase)],
            dtype=float,
        )
    else:  # pragma: no cover - guarded above
        raise AssertionError(family)

    total_steps = approach_steps + descend_steps + insert_steps + hold_steps
    offsets = np.empty((total_steps + 1, 3), dtype=float)
    phase_ids = np.empty(total_steps + 1, dtype=np.int8)
    for step in range(total_steps + 1):
        xy = _trajectory_xy(
            family,
            step=step,
            approach_steps=approach_steps,
            descend_steps=descend_steps,
            insert_steps=insert_steps,
            initial_xy=initial_xy,
            terminal_xy=terminal_xy,
            curve_radius=curve_radius,
            curve_angle=curve_angle,
            curve_turns=curve_turns,
            curve_phase=curve_phase,
        )
        xy += _teleop_noise(step=step, total_steps=total_steps, perturbation=perturbation)
        z, phase_id = _trajectory_z(
            step,
            approach_steps=approach_steps,
            descend_steps=descend_steps,
            insert_steps=insert_steps,
            approach_height=base_config.approach_height,
            descend_height=base_config.descend_height,
            insert_depth=base_config.insert_depth,
        )
        offsets[step] = (float(xy[0]), float(xy[1]), z)
        phase_ids[step] = phase_id

    parameters = np.array(
        [
            approach_steps,
            descend_steps,
            insert_steps,
            hold_steps,
            base_config.approach_height,
            base_config.descend_height,
            base_config.insert_depth,
            initial_xy[0],
            initial_xy[1],
            terminal_xy[0],
            terminal_xy[1],
            curve_radius,
            curve_angle,
            curve_turns,
            curve_phase,
            duration_scale,
        ],
        dtype=float,
    )
    return OpenLoopTrajectoryPlan(
        family=family,
        family_id=TRAJECTORY_FAMILY_TO_ID[family],
        parameters=parameters,
        target_offsets=offsets,
        phase_ids=phase_ids,
    )


def classify_episode_admission(
    *,
    episode_complete: bool,
    arrays_finite: bool,
    full_step_max_force: float,
    sampled_max_force: float,
) -> tuple[bool, bool, str]:
    capture_ratio = sampled_max_force / full_step_max_force if full_step_max_force > 0.0 else 1.0
    solver_spike_suspicious = bool(
        full_step_max_force >= 1000.0
        and capture_ratio < 0.05
        and sampled_max_force < 500.0
    )
    if not arrays_finite:
        return False, solver_spike_suspicious, "nonfinite_rollout"
    if not episode_complete:
        return False, solver_spike_suspicious, "incomplete_rollout"
    if solver_spike_suspicious:
        return False, True, "solver_spike_suspicious"
    return True, False, ""


def _stratified_split(
    episode_ids: np.ndarray,
    success: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    train: list[np.ndarray] = []
    val: list[np.ndarray] = []
    for outcome in (False, True):
        group = np.asarray(episode_ids[success == outcome], dtype=np.int32).copy()
        rng.shuffle(group)
        if group.size <= 1:
            count = group.size
        else:
            count = min(max(int(round(0.8 * group.size)), 1), group.size - 1)
        train.append(group[:count])
        val.append(group[count:])
    return (
        np.sort(np.concatenate(train) if train else np.empty(0, dtype=np.int32)),
        np.sort(np.concatenate(val) if val else np.empty(0, dtype=np.int32)),
    )


def _json_ready(value: Any) -> Any:
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _available_output_dir(output_root: Path, run_name: str) -> Path:
    candidate = output_root / run_name
    rerun = 0
    while candidate.exists():
        rerun += 1
        candidate = output_root / f"{run_name}_rerun{rerun}"
    candidate.mkdir(parents=True)
    return candidate


def _resolve_raw_collection_path(raw_dataset: Path) -> Path:
    if raw_dataset.is_file():
        return raw_dataset
    if not raw_dataset.exists():
        raise FileNotFoundError(f"Raw collection path does not exist: {raw_dataset}")
    if (raw_dataset / "raw_collection.npz").exists():
        return raw_dataset / "raw_collection.npz"
    if (raw_dataset / "eligible_residual_bc.npz").exists():
        return raw_dataset / "eligible_residual_bc.npz"
    npz_files = sorted(raw_dataset.glob("*.npz"))
    if len(npz_files) == 1:
        return npz_files[0]
    raise FileNotFoundError(
        f"Could not resolve a raw or eligible dataset under {raw_dataset}. Expected raw_collection.npz."
    )


def build_eligible_residual_dataset_from_raw(
    raw_dataset: Path,
    output_path: Path | None = None,
    *,
    label_neighbors: int = 32,
    knn_block_size: int = 1024,
    label_projection: str = LABEL_PROJECTION_RESIDUAL_FIRST,
    residual_bound: float | None = None,
    baseline_k: float = DEFAULT_BASELINE_K,
    diagnostic_k_min: float = DEFAULT_DIAGNOSTIC_K_MIN,
    diagnostic_k_max: float = DEFAULT_DIAGNOSTIC_K_MAX,
    label_k_min: float | None = None,
    label_k_max: float | None = None,
    l21_coupling_percentile: float = DEFAULT_L21_COUPLING_PERCENTILE,
    contact_gate_low: float = DEFAULT_CONTACT_GATE_LOW,
    contact_gate_high: float = DEFAULT_CONTACT_GATE_HIGH,
    neutral_contact_threshold: float = DEFAULT_NEUTRAL_CONTACT_THRESHOLD,
    min_calibration_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
    train_episode_ids: np.ndarray | None = None,
    val_episode_ids: np.ndarray | None = None,
) -> Path:
    """Build an eligible residual-BC dataset from a raw collection artifact."""
    raw_dataset = _resolve_raw_collection_path(raw_dataset)
    with np.load(raw_dataset, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))

    if "episode_label_eligible" not in arrays:
        raise KeyError(f"{raw_dataset} is missing episode_label_eligible.")
    if "episode_summary_id" not in arrays:
        raise KeyError(f"{raw_dataset} is missing episode_summary_id.")
    if "episode_id" not in arrays:
        raise KeyError(f"{raw_dataset} is missing episode_id.")

    raw_metadata = dict(metadata)
    scene = str(raw_metadata.get("scene", ""))
    scene_spec = get_scene_spec(scene)
    if "base_stiffness_spec" in raw_metadata:
        base_spec = BaseStiffnessSpec.from_metadata(raw_metadata["base_stiffness_spec"])
    else:
        collection_matrix = np.asarray(raw_metadata["collection_stiffness_matrix"], dtype=float)
        base_spec = BaseStiffnessSpec.from_matrix(
            collection_matrix,
            active_groups=scene_spec.active_groups,
            active_group_names=scene_spec.active_group_names,
            residual_bound=scene_spec.residual_bound,
        )

    eligible_episode_mask = np.asarray(arrays["episode_label_eligible"], dtype=bool)
    eligible_episode_ids = np.asarray(arrays["episode_summary_id"], dtype=np.int32)[eligible_episode_mask]
    eligible_sample_mask = np.isin(np.asarray(arrays["episode_id"], dtype=np.int32), eligible_episode_ids)
    if not np.any(eligible_sample_mask):
        raise RuntimeError(f"No label-eligible samples were collected in {raw_dataset}.")

    task_state = np.asarray(arrays["task_state"], dtype=float)[eligible_sample_mask]
    contact_force = np.asarray(arrays["contact_force_world"], dtype=float)[eligible_sample_mask]
    label_config = StiffnessLabelConfig(neighbors=label_neighbors, knn_block_size=knn_block_size)
    normalized_matrix, normalized_cholesky, diagnostics = build_stiffness_labels_with_diagnostics(
        task_state,
        contact_force,
        config=label_config,
    )
    if residual_bound is None:
        residual_bound = float(np.max(base_spec.residual_bounds))
    if label_projection not in (
        LABEL_PROJECTION_RESIDUAL_FIRST,
        LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
        LABEL_PROJECTION_LEGACY,
    ):
        raise ValueError(
            "Open-loop residual dataset generation only supports "
            f"{LABEL_PROJECTION_RESIDUAL_FIRST!r}, {LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2!r}, "
            f"and {LABEL_PROJECTION_LEGACY!r}; observed {label_projection!r}."
        )
    projection = project_residual_first_labels(
        normalized_matrix,
        base_spec=base_spec,
        contact_forces_world=contact_force,
        label_projection=label_projection,
        residual_bound=float(residual_bound),
        baseline_k=baseline_k,
        diagnostic_k_min=diagnostic_k_min,
        diagnostic_k_max=diagnostic_k_max,
        l21_coupling_percentile=l21_coupling_percentile,
        contact_gate_low=contact_gate_low,
        contact_gate_high=contact_gate_high,
        neutral_contact_threshold=neutral_contact_threshold,
        min_calibration_samples=min_calibration_samples,
    )
    residual_group = projection.residual_group_target
    residual_theta = projection.residual_theta_target
    physical_matrix = projection.physical_stiffness_matrix_target
    physical_cholesky = projection.physical_stiffness_cholesky_target

    eligible_rows = np.where(eligible_episode_mask)[0]
    episode_success = np.asarray(arrays["episode_success"], dtype=bool)[eligible_rows]
    split_seed = int(raw_metadata.get("rng_streams", {}).get("split_seed", 0))
    if train_episode_ids is None or val_episode_ids is None:
        split_rng = np.random.default_rng(split_seed)
        train_ids, val_ids = _stratified_split(
            eligible_episode_ids,
            episode_success,
            rng=split_rng,
        )
    else:
        train_ids = np.asarray(train_episode_ids, dtype=np.int64)
        val_ids = np.asarray(val_episode_ids, dtype=np.int64)

    eligible_arrays: dict[str, np.ndarray] = {
        "state": np.asarray(arrays["state"], dtype=float)[eligible_sample_mask],
        "action": np.asarray(arrays["action"], dtype=float)[eligible_sample_mask],
        "contact_state": np.asarray(arrays["contact_state"], dtype=float)[eligible_sample_mask],
        "task_state": task_state,
        "task_state_6d": task_state,
        "privileged_task_state_6d": task_state,
        "contact_force_world": contact_force,
        "stiffness_matrix_target": normalized_matrix,
        "stiffness_cholesky_target": normalized_cholesky,
        "physical_stiffness_matrix_target": physical_matrix,
        "physical_stiffness_cholesky_target": physical_cholesky,
        "physical_stiffness_eigenvalues": projection.physical_stiffness_eigenvalues,
        "residual_group_target": residual_group,
        "residual_theta_target": residual_theta,
        "theta_base": base_spec.theta_base,
        "base_stiffness_matrix": base_spec.base_matrix,
        "episode_id": np.asarray(arrays["episode_id"], dtype=np.int32)[eligible_sample_mask],
        "sample_step": np.asarray(arrays["sample_step"], dtype=np.int32)[eligible_sample_mask],
        "timestamp": np.asarray(arrays["timestamp"], dtype=float)[eligible_sample_mask],
        "randomization": np.asarray(arrays["randomization"], dtype=float)[eligible_sample_mask],
        "planned_target_position": np.asarray(arrays["planned_target_position"], dtype=float)[eligible_sample_mask],
        "planned_target_rotation": np.asarray(arrays["planned_target_rotation"], dtype=float)[eligible_sample_mask],
        "trajectory_family_id": np.asarray(arrays["trajectory_family_id"], dtype=np.int8)[eligible_sample_mask],
        "trajectory_parameters": np.asarray(arrays["trajectory_parameters"], dtype=float)[eligible_sample_mask],
        "phase_id": np.asarray(arrays["phase_id"], dtype=np.int8)[eligible_sample_mask],
        "episode_summary_id": eligible_episode_ids,
        "episode_success": episode_success,
        "episode_final_depth": np.asarray(arrays["episode_final_depth"], dtype=float)[eligible_rows],
        "episode_final_lateral_error": np.asarray(arrays["episode_final_lateral_error"], dtype=float)[eligible_rows],
        "episode_max_normal_force": np.asarray(arrays["episode_max_normal_force"], dtype=float)[eligible_rows],
        "episode_contact_count": np.asarray(arrays["episode_contact_count"], dtype=np.int32)[eligible_rows],
        "episode_contact_onset_step": np.asarray(arrays["episode_contact_onset_step"], dtype=np.int32)[eligible_rows],
        "episode_perturbation": np.asarray(arrays["episode_perturbation"], dtype=float)[eligible_rows],
        "episode_command_xy_offset": np.asarray(arrays["episode_command_xy_offset"], dtype=float)[eligible_rows],
        "episode_trajectory_family_id": np.asarray(arrays["episode_trajectory_family_id"], dtype=np.int8)[eligible_rows],
        "episode_trajectory_parameters": np.asarray(arrays["episode_trajectory_parameters"], dtype=float)[eligible_rows],
        "train_episode_ids": train_ids,
        "val_episode_ids": val_ids,
        **diagnostics,
    }
    if "rgb_images" in arrays:
        eligible_arrays["rgb_images"] = np.asarray(arrays["rgb_images"], dtype=np.uint8)[eligible_sample_mask]

    task_state_metadata = describe_privileged_task_state_schema()
    updated_metadata = {
        "schema_version": (
            "residual_bc_residual_first_v1"
            if label_projection == LABEL_PROJECTION_RESIDUAL_FIRST
            else "residual_bc_contact_gated_centered_v2"
            if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2
            else "residual_bc_v1"
        ),
        "scene": scene,
        "scene_config": raw_metadata.get("scene_config"),
        "eye_in_hand_camera_pose_version": raw_metadata.get("eye_in_hand_camera_pose_version"),
        "eye_in_hand_camera_canonical": raw_metadata.get("eye_in_hand_camera_canonical"),
        "eye_in_hand_camera_name": raw_metadata.get("eye_in_hand_camera_name"),
        "eye_in_hand_camera_attachment_parent": raw_metadata.get("eye_in_hand_camera_attachment_parent"),
        "eye_in_hand_camera_mount_type": raw_metadata.get("eye_in_hand_camera_mount_type"),
        "eye_in_hand_camera_pose": raw_metadata.get("eye_in_hand_camera_pose"),
        "setting_id": raw_metadata.get("setting_id"),
        "profile_name": raw_metadata.get("profile_name"),
        "collection_seed": raw_metadata.get("collection_seed"),
        "robustness_preset": raw_metadata.get("robustness_preset"),
        "controller_id": raw_metadata.get("controller_id"),
        "scenario_id": raw_metadata.get("scenario_id"),
        "collection_controller_id": raw_metadata.get("collection_controller_id"),
        "collection_stiffness_matrix": raw_metadata.get("collection_stiffness_matrix"),
        "dataset_path": str(output_path or raw_dataset.parent / "eligible_residual_bc.npz"),
        "controllers_yaml": raw_metadata.get("controllers_yaml"),
        "base_profile": raw_metadata.get("base_profile"),
        "gain_config": raw_metadata.get("gain_config"),
        "base_stiffness_spec": base_spec.to_metadata(),
        "trajectory_plan": raw_metadata.get("trajectory_plan"),
        "trajectory_source": raw_metadata.get("trajectory_source"),
        "trajectory_families": raw_metadata.get("trajectory_families"),
        "trajectory_parameter_fields": raw_metadata.get("trajectory_parameter_fields"),
        "num_episodes": int(eligible_episode_ids.size),
        "raw_episode_count": int(np.asarray(arrays["episode_summary_id"]).shape[0]),
        "num_samples": int(task_state.shape[0]),
        "seed": raw_metadata.get("seed"),
        "rng_streams": raw_metadata.get("rng_streams"),
        "sample_stride": raw_metadata.get("sample_stride"),
        "rgb_enabled": bool(raw_metadata.get("rgb_enabled", False)),
        "rgb_camera_name": raw_metadata.get("rgb_camera_name"),
        "rgb_image_width": raw_metadata.get("rgb_image_width"),
        "rgb_image_height": raw_metadata.get("rgb_image_height"),
        "rgb_image_stride": raw_metadata.get("rgb_image_stride"),
        "rgb_max_episodes": raw_metadata.get("rgb_max_episodes"),
        "renderer_mode": raw_metadata.get("renderer_mode"),
        "fallback_used": raw_metadata.get("fallback_used"),
        "native_launcher_required": raw_metadata.get("native_launcher_required"),
        "training_data_valid": raw_metadata.get("training_data_valid"),
        "training_data_valid_reason": raw_metadata.get("training_data_valid_reason"),
        "episode_specs_path": raw_metadata.get("episode_specs_path"),
        "frozen_paired_episode_specs_path": raw_metadata.get("frozen_paired_episode_specs_path"),
        "frozen_train_val_split_path": raw_metadata.get("frozen_train_val_split_path"),
        "episode_spec_schema_version": raw_metadata.get("episode_spec_schema_version"),
        "sample_rate_hz": raw_metadata.get("sample_rate_hz"),
        "label_neighbors": int(label_neighbors),
        "knn_block_size": int(knn_block_size),
        "force_sectorization_used": True,
        "sectorization": "polar_azimuth_grid_v1",
        "sector_polar_bins": int(label_config.polar_bins),
        "sector_azimuth_bins": int(label_config.azimuth_bins),
        "num_force_sectors": int(label_config.polar_bins * label_config.azimuth_bins),
        "sector_magnitude_percentile": float(label_config.force_percentile),
        "label_projection": label_projection,
        "target": "baseline_relative_residual",
        "residual_bound": float(residual_bound),
        "baseline_k": float(baseline_k),
        "diagnostic_k_min": float(diagnostic_k_min),
        "diagnostic_k_max": float(diagnostic_k_max),
        "residual_to_physical_semantics": "negative_softens_positive_stiffens",
        "label_k_range_role": "deprecated_legacy_compatibility_only",
        "state_schema": task_state_metadata["state_schema"],
        "privileged_state_schema": task_state_metadata["state_schema"],
        "privileged_state_names": task_state_metadata["state_names"],
        "privileged_state_units": task_state_metadata["state_units"],
        "state_frame": task_state_metadata["state_frame"],
        "angle_wrapping": task_state_metadata["angle_wrapping"],
        "yaw_source": task_state_metadata["yaw_source"],
        "roll_pitch_source": task_state_metadata["roll_pitch_source"],
        "roll_pitch_available": task_state_metadata["roll_pitch_available"],
        "roll_pitch_available_fraction": 1.0,
        "primary_target_space": (
            "baseline_relative_residual"
            if label_projection in (LABEL_PROJECTION_RESIDUAL_FIRST, LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2)
            else "legacy_direct_physical_k"
        ),
        "primary_supervision_target": "residual_group_target",
        "physical_stiffness_matrix_target_role": projection.label_metadata["physical_stiffness_matrix_target_role"],
        "task_state_dim": int(task_state.shape[1]),
        "residual_dim": int(residual_group.shape[1]),
        "randomization_fields": raw_metadata.get("randomization_fields"),
        "legacy_field_mapping": raw_metadata.get("legacy_field_mapping"),
        "episode_outcome_fields": [
            "success",
            "final_depth",
            "final_lateral_error",
            "max_normal_force",
            "contact_count",
            "contact_onset_step",
            "perturbation",
            "command_xy_offset",
            "trajectory_family",
            "trajectory_parameters",
        ],
        "episode_split": {
            "strategy": "success_stratified_80_20",
            "split_seed": split_seed,
            "train_episode_count": int(train_ids.size),
            "val_episode_count": int(val_ids.size),
        },
        "raw_source_dataset": str(raw_dataset),
        "excluded_episode_ids": np.asarray(arrays["episode_summary_id"], dtype=np.int32)[~eligible_episode_mask].tolist(),
        "labels_built": True,
        "relabel_source_dataset": str(raw_dataset),
        "relabel_source_label_k_min": raw_metadata.get("label_k_min"),
        "relabel_source_label_k_max": raw_metadata.get("label_k_max"),
        "relabel_version": (
            "residual_first_baseline_relative_v1"
            if label_projection == LABEL_PROJECTION_RESIDUAL_FIRST
            else "residual_first_contact_gated_centered_v2"
            if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2
            else "residual_bc_relabel_v1"
        ),
        **projection.label_metadata,
    }
    if label_projection == LABEL_PROJECTION_LEGACY:
        updated_metadata["label_k_min"] = float(label_k_min) if label_k_min is not None else None
        updated_metadata["label_k_max"] = float(label_k_max) if label_k_max is not None else None
        updated_metadata["label_k_min_deprecated"] = False
        updated_metadata["label_k_max_deprecated"] = False
    else:
        updated_metadata["label_k_min"] = 300.0
        updated_metadata["label_k_max"] = 600.0
        updated_metadata["label_k_min_deprecated"] = True
        updated_metadata["label_k_max_deprecated"] = True
        updated_metadata["label_k_range_role"] = "deprecated_legacy_compatibility_only"
        if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2:
            updated_metadata.setdefault("contact_gate_applied", True)
            updated_metadata.setdefault("free_space_residual_policy", "baseline_anchor")
            updated_metadata.setdefault("residual_group_target_neutral", 0.0)

    if output_path is None:
        output_path = raw_dataset.parent / "eligible_residual_bc.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **eligible_arrays, metadata=json.dumps(updated_metadata, sort_keys=True))
    validate_residual_dataset(output_path)
    return output_path


def collect_randomized_open_loop_residual_dataset(
    *,
    episodes: int = 80,
    seed: int = 2000,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    sample_stride: int = 50,
    save_rgb: bool = False,
    camera_name: str = "eye_in_hand_rgb",
    image_width: int = 128,
    image_height: int = 128,
    image_stride: int = 1,
    max_rgb_episodes: int | None = None,
    renderer_mode: str = "native",
    allow_debug_fallback_renderer: bool = False,
    controller_id: str | None = None,
    controllers_yaml: Path = DEFAULT_TRACK_A_CONTROLLERS_YAML,
    controller_profile: str = TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE,
    scenario_id: str | None = None,
    profile_name: str | None = None,
    teleop_mode: str = TELEOP_MODE_POSITION_ONLY,
    controlled_contact_profile: ControlledContactProfile | None = None,
    label_neighbors: int = 32,
    knn_block_size: int = 1024,
    label_projection: str = LABEL_PROJECTION_RESIDUAL_FIRST,
    residual_bound: float | None = None,
    label_k_min: float = 300.0,
    label_k_max: float = 600.0,
    baseline_k: float = DEFAULT_BASELINE_K,
    diagnostic_k_min: float = DEFAULT_DIAGNOSTIC_K_MIN,
    diagnostic_k_max: float = DEFAULT_DIAGNOSTIC_K_MAX,
    l21_coupling_percentile: float = DEFAULT_L21_COUPLING_PERCENTILE,
    contact_gate_low: float = DEFAULT_CONTACT_GATE_LOW,
    contact_gate_high: float = DEFAULT_CONTACT_GATE_HIGH,
    neutral_contact_threshold: float = DEFAULT_NEUTRAL_CONTACT_THRESHOLD,
    min_calibration_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
    build_labels: bool = True,
    successful_episodes_only: bool = False,
    max_attempts: int | None = None,
    gain_config: Path = DEFAULT_GAIN_CONFIG,
) -> OpenLoopCollectionResult:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive.")
    if image_stride <= 0:
        raise ValueError("image_stride must be positive.")
    if save_rgb and sample_stride % image_stride != 0:
        raise ValueError("For synchronized RGB capture, image_stride must evenly divide sample_stride.")
    if renderer_mode not in {"native", "legacy_debug_only"}:
        raise ValueError("renderer_mode must be one of {'native', 'legacy_debug_only'}.")
    if save_rgb and renderer_mode != "native" and not allow_debug_fallback_renderer:
        raise ValueError("Active image dataset collection requires native rendering.")
    teleop_mode = validate_teleop_mode(teleop_mode)
    started = time.perf_counter()
    scene = str(scenario_id or "circle")
    spec = get_scene_spec(scene)
    requested_controller_id = controller_id or controller_profile
    if requested_controller_id == TRACK_A_DATA_COLLECTION_CONTROLLER_PROFILE:
        requested_controller_id = DEFAULT_COLLECTION_CONTROLLER_ID
    controller_entry, profile, gains = load_track_a_controller_runtime(
        requested_controller_id,
        controllers_yaml=controllers_yaml,
        gain_config=gain_config,
    )
    base_config = RolloutConfig(config_path=spec.config_path, gain_config_path=gain_config)
    base_spec = BaseStiffnessSpec.from_matrix(
        controller_entry.position_stiffness_matrix,
        active_groups=spec.active_groups,
        active_group_names=spec.active_group_names,
        residual_bound=spec.residual_bound,
    )
    nominal_scene_config = load_scene_config(spec.config_path)
    nominal_hole_position = np.asarray(nominal_scene_config["hole"]["pos"], dtype=float)
    nominal_hole_xy = np.asarray(nominal_hole_position[:2], dtype=float)
    preset = get_robustness_preset(scene)

    root_seed = np.random.SeedSequence(seed)
    perturbation_seq, trajectory_seq, split_seq = root_seed.spawn(3)
    perturbation_seed = int(perturbation_seq.generate_state(1, dtype=np.uint32)[0])
    trajectory_seed = int(trajectory_seq.generate_state(1, dtype=np.uint32)[0])
    split_seed = int(split_seq.generate_state(1, dtype=np.uint32)[0])
    perturbation_rng = np.random.default_rng(perturbation_seed)
    trajectory_rng = np.random.default_rng(trajectory_seed)
    if controlled_contact_profile is not None:
        profile_name = profile_name or controlled_contact_profile.profile_name
        profile_metadata = controlled_contact_profile.to_metadata()
    else:
        profile_name = profile_name or f"{preset.setting_id}_legacy_randomized"
        profile_metadata = {
            "hole_xy_offset_semantics": "legacy_contact_randomization",
            "hole_xy_offset_units": "m",
            "hole_xy_offset_distribution": f"legacy_preset_uniform_disk(radius={preset.hole_xy_radius:g})",
            "trajectory_follows_randomized_hole": True,
            "contact_generation_parameters_fixed": False,
            "fixed_hole_yaw_offset": None,
            "fixed_teleop_noise_xy_amplitude": None,
            "fixed_teleop_noise_cycles": None,
            "fixed_teleop_noise_phase_x": None,
            "fixed_teleop_noise_phase_y": None,
            "fixed_clearance_delta": None,
            "fixed_friction_scale": None,
            "fixed_peg_tilt_x": None,
            "fixed_peg_tilt_y": None,
            "legacy_field_mapping": {
                "randomization_vector": {
                    "0": "hole_xy_offset_x",
                    "1": "hole_xy_offset_y",
                    "2": "hole_yaw_offset",
                    "3": "clearance_delta",
                    "4": "friction_scale",
                    "5": "peg_tilt_x",
                    "6": "peg_tilt_y",
                    "7": "teleop_noise_xy_amplitude",
                    "8": "teleop_noise_cycles",
                }
            },
        }
    split_rng = np.random.default_rng(split_seed)
    if max_attempts is None:
        max_attempts = max(episodes * 50, episodes + 50) if successful_episodes_only else episodes
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")

    if controlled_contact_profile is not None and controlled_contact_profile.contact_condition_name:
        run_profile_suffix = f"{profile_name or controlled_contact_profile.profile_name}_{controlled_contact_profile.contact_condition_name}"
    else:
        run_profile_suffix = profile_name or "legacy_randomized"
    run_name = f"{preset.setting_id}_open_loop_{episodes}ep_seed{seed}_{run_profile_suffix}"
    if successful_episodes_only:
        run_name = f"{run_name}_success_only"
    output_dir = _available_output_dir(output_root, run_name)
    raw_path = output_dir / "raw_collection.npz"
    eligible_path = output_dir / "eligible_residual_bc.npz"
    summary_path = output_dir / "collection_summary.json"
    episode_csv_path = output_dir / "episodes.csv"
    episode_specs_path = output_dir / "episode_specs.jsonl"
    frozen_episode_specs_path = output_dir / "frozen_paired_episode_specs.jsonl"
    frozen_split_path = output_dir / "frozen_train_val_split.json"
    family_summary_path = output_dir / "trajectory_family_summary.json"

    sample_rows: dict[str, list[Any]] = {
        "state": [],
        "action": [],
        "contact_state": [],
        "task_state": [],
        "contact_force_world": [],
        "normal_force": [],
        "episode_id": [],
        "sample_step": [],
        "timestamp": [],
        "randomization": [],
        "planned_target_position": [],
        "planned_target_rotation": [],
        "trajectory_family_id": [],
        "trajectory_parameters": [],
        "phase_id": [],
    }
    rgb_rows: list[np.ndarray] = []
    episode_rows: list[dict[str, Any]] = []
    episode_perturbations: list[np.ndarray] = []
    episode_plan_parameters: list[np.ndarray] = []
    episode_specs: list[EpisodeSpec] = []
    family_attempt_counts = {family: 0 for family in TRAJECTORY_FAMILIES}
    family_success_counts = {family: 0 for family in TRAJECTORY_FAMILIES}
    attempted_episodes = 0
    discarded_failed_attempts = 0
    collection_renderer_mode: str | None = None
    collection_fallback_used: bool | None = None
    canonical_camera_pose = canonical_eye_in_hand_camera_pose(camera_name)

    episode_slot = 0
    while episode_slot < episodes:
        family = TRAJECTORY_FAMILIES[episode_slot % len(TRAJECTORY_FAMILIES)]
        episode_attempt = 0
        while True:
            episode_id = episode_slot
            if attempted_episodes >= max_attempts:
                raise RuntimeError(
                    f"Reached max_attempts={max_attempts} before collecting {episodes} episodes. "
                    f"successful_episodes_only={successful_episodes_only} "
                    f"successful={len(episode_rows)} failed_attempts={discarded_failed_attempts}."
                )
            episode_attempt += 1
            attempted_episodes += 1
            family_attempt_counts[family] += 1
            if controlled_contact_profile is not None:
                perturbation_seed_i = int(perturbation_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
                perturbation = sample_controlled_contact_perturbations(
                    episodes=1,
                    seed=perturbation_seed_i,
                    profile=controlled_contact_profile,
                )[0]
            else:
                perturbation_seed_i = int(perturbation_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
                perturbation = sample_robustness_perturbations(
                    scene=scene,
                    episodes=1,
                    seed=perturbation_seed_i,
                    preset=preset,
                )[0]
            plan = generate_open_loop_trajectory(
                family=family,
                rng=trajectory_rng,
                perturbation=perturbation,
                base_config=base_config,
            )

            sample_length_snapshot = {key: len(values) for key, values in sample_rows.items()}
            rgb_length_snapshot = len(rgb_rows)
            episode_rows_length = len(episode_rows)
            episode_perturbations_length = len(episode_perturbations)
            episode_plan_parameters_length = len(episode_plan_parameters)
            episode_specs_length = len(episode_specs)

            scene_path, scene_config = scene_for_rollout(spec.config_path, perturbation)
            try:
                model = load_model(scene_path)
            finally:
                cleanup_runtime_scene(scene_path)
            data = mujoco.MjData(model)
            reset_from_config(model, data, scene_config)
            validate_canonical_eye_in_hand_camera_config(scene_config, camera_name=camera_name)
            camera_local_position, camera_rotation, camera_fovy = eye_in_hand_camera_pose_from_config(
                scene_config,
                camera_name=camera_name,
            )
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                raise ValueError(f"Unknown camera {camera_name!r}.")
            apply_eye_in_hand_camera_pose(
                model,
                data,
                camera_id=camera_id,
                local_position=camera_local_position,
                rotation=camera_rotation,
                fovy=camera_fovy,
            )
            geometry = geometry_from_config(scene_config)
            task_ids = peg_hole_ids(model, segments=geometry.segments)
            arm_ids = panda_arm_ids(model)
            nullspace_target_qpos = arm_qpos(data, arm_ids)
            hole_center = hole_center_position(data, task_ids)
            target_rotations = build_target_orientations(
                nominal_scene_config,
                total_steps=plan.total_steps,
                teleop_mode=teleop_mode,
            )
            if teleop_mode == TELEOP_MODE_POSITION_ORIENTATION:
                target_rotation = target_rotations[0] if target_rotations is not None else nominal_hole_rotation(nominal_scene_config)
            else:
                target_rotation = target_rotation_with_peg_tilt(
                    site_rotation(data, model.site(base_config.site_name).id),
                    perturbation,
                )
            perturbation_dict = perturbation.to_dict()
            randomization = np.array(
                [
                    perturbation.hole_xy_offset[0],
                    perturbation.hole_xy_offset[1],
                    perturbation.hole_yaw_offset,
                    perturbation.clearance_delta,
                    perturbation.friction_scale,
                    perturbation.peg_tilt_x,
                    perturbation.peg_tilt_y,
                    perturbation.teleop_noise_xy_amplitude,
                    perturbation.teleop_noise_cycles,
                ],
                dtype=float,
            )
            episode_perturbations.append(randomization)
            episode_plan_parameters.append(plan.parameters)
            fixed_contact_condition = {
                "teleop_noise_xy_amplitude": float(perturbation.teleop_noise_xy_amplitude),
                "teleop_noise_cycles": float(perturbation.teleop_noise_cycles),
                "teleop_noise_phase_x": float(perturbation.teleop_noise_phase_x),
                "teleop_noise_phase_y": float(perturbation.teleop_noise_phase_y),
                "clearance_delta": float(perturbation.clearance_delta),
                "friction_scale": float(perturbation.friction_scale),
                "peg_tilt_x": float(perturbation.peg_tilt_x),
                "peg_tilt_y": float(perturbation.peg_tilt_y),
                "fixed_hole_yaw_offset": float(perturbation.hole_yaw_offset),
                "hole_xy_radius": float(controlled_contact_profile.hole_xy_radius if controlled_contact_profile is not None else preset.hole_xy_radius),
            }
            episode_spec = EpisodeSpec.create(
                episode_id=episode_slot,
                seed=seed,
                scene=scene,
                setting_id=preset.setting_id,
                profile_name=profile_name,
                contact_condition_name=controlled_contact_profile.contact_condition_name if controlled_contact_profile is not None else None,
                nominal_hole_position=nominal_hole_position,
                nominal_hole_xy=nominal_hole_xy,
                hole_xy_offset=np.asarray(perturbation.hole_xy_offset, dtype=float),
                hole_yaw_offset=float(perturbation.hole_yaw_offset),
                hole_xy_radius=float(fixed_contact_condition["hole_xy_radius"]),
                hole_xy_offset_semantics=profile_metadata["hole_xy_offset_semantics"],
                hole_xy_offset_distribution=profile_metadata["hole_xy_offset_distribution"],
                trajectory_follows_randomized_hole=bool(profile_metadata["trajectory_follows_randomized_hole"]),
                contact_generation_parameters_fixed=bool(profile_metadata["contact_generation_parameters_fixed"]),
                fixed_contact_condition=fixed_contact_condition,
                trajectory_source=EPISODE_TRAJECTORY_SOURCE_OPEN_LOOP_FAMILY,
                trajectory_family=plan.family,
                trajectory_family_id=plan.family_id,
                trajectory_parameters=plan.parameters,
                teleop_mode=teleop_mode,
                target_rotations=target_rotations,
                target_offsets=plan.target_offsets,
                phase_ids=plan.phase_ids,
                total_steps=plan.total_steps,
                sample_stride=sample_stride,
                image_stride=image_stride if save_rgb else None,
            )
            episode_specs.append(episode_spec)
            rgb_renderer = None
            if save_rgb and episode_slot < (episodes if max_rgb_episodes is None else min(episodes, max_rgb_episodes)):
                rgb_renderer = MujocoRgbRenderer(
                    model,
                    camera_name=camera_name,
                    width=image_width,
                    height=image_height,
                    renderer_mode=renderer_mode,
                )
                if rgb_renderer.fallback_used and not allow_debug_fallback_renderer:
                    raise RuntimeError(
                        "Fallback-rendered RGB capture is legacy debug only and cannot be saved "
                        "for active image-only datasets."
                    )
                collection_renderer_mode = rgb_renderer.mode
                collection_fallback_used = bool(rgb_renderer.fallback_used)

            sample_start = len(sample_rows["episode_id"])
            full_max_force = 0.0
            sampled_max_force = 0.0
            contact_count = 0
            contact_onset = -1
            max_torque = 0.0
            torque_saturation_count = 0
            arrays_finite = True
            completed = True

            try:
                for step in range(plan.total_steps + 1):
                    target_position = hole_center + plan.target_offsets[step]
                    command = task_space_impedance_torque(
                        model,
                        data,
                        site_name=base_config.site_name,
                        target_position=target_position,
                        target_rotation=target_rotation,
                        arm_ids=arm_ids,
                        gains=gains,
                        position_stiffness_matrix=controller_entry.position_stiffness_matrix,
                        nullspace_target_qpos=nullspace_target_qpos,
                        clip_to_ctrlrange=False,
                    )
                    torque, saturated = clip_torque(model, arm_ids, command.torque)
                    contact_query = ContactQuery(model=model, data=data, task_ids=task_ids)
                    contact = extract_contact_state(contact_query)
                    normal_force = float(contact.normal_force)
                    full_max_force = max(full_max_force, normal_force)
                    if contact.in_contact:
                        contact_count += 1
                        if contact_onset < 0:
                            contact_onset = step
                    max_torque = max(max_torque, float(np.max(np.abs(torque))))
                    torque_saturation_count += int(saturated)

                    finite_step = bool(
                        np.all(np.isfinite(data.qpos))
                        and np.all(np.isfinite(data.qvel))
                        and np.all(np.isfinite(torque))
                        and np.isfinite(normal_force)
                    )
                    if not finite_step:
                        arrays_finite = False
                        completed = False
                        break

                    if step % sample_stride == 0:
                        obs, u_ref, _ = collect_step(
                            model,
                            data,
                            arm_ids=arm_ids,
                            task_ids=task_ids,
                            target_position=target_position,
                            target_rotation=target_rotation,
                            phase_id=int(plan.phase_ids[step]),
                        )
                        task_state = peg_hole_task_state(
                            data,
                            task_ids,
                            hole_clearance_delta=float(perturbation.clearance_delta),
                        )
                        force_world = extract_net_peg_hole_contact_force_world(contact_query)
                        sampled_max_force = max(sampled_max_force, normal_force)
                        sample_rows["state"].append(_state_row(obs, data, arm_ids))
                        sample_rows["action"].append(_action_row(u_ref, torque))
                        sample_rows["contact_state"].append(contact_state_vector(contact))
                        sample_rows["task_state"].append(task_state)
                        sample_rows["contact_force_world"].append(force_world)
                        sample_rows["normal_force"].append(normal_force)
                        sample_rows["episode_id"].append(episode_id)
                        sample_rows["sample_step"].append(step)
                        sample_rows["timestamp"].append(float(data.time))
                        sample_rows["randomization"].append(randomization)
                        sample_rows["planned_target_position"].append(target_position)
                        sample_rows["planned_target_rotation"].append(target_rotation.reshape(-1))
                        sample_rows["trajectory_family_id"].append(plan.family_id)
                        sample_rows["trajectory_parameters"].append(plan.parameters)
                        sample_rows["phase_id"].append(int(plan.phase_ids[step]))
                        if rgb_renderer is not None:
                            rgb_rows.append(rgb_renderer.render(data))

                    if step < plan.total_steps:
                        set_arm_torque_ctrl(model, data, arm_ids, torque)
                        mujoco.mj_step(model, data)
            finally:
                if rgb_renderer is not None:
                    rgb_renderer.close()

            final_state = peg_hole_task_state(
                data,
                task_ids,
                hole_clearance_delta=float(perturbation.clearance_delta),
            )
            final_depth = float(final_state[2])
            final_lateral = float(np.linalg.norm(final_state[:2]))
            final_peg_tip = peg_tip_position(data, task_ids)
            final_hole_center = np.asarray(hole_center, dtype=float).copy()
            final_target_position = np.asarray(target_position, dtype=float).copy()
            success_depth_threshold = float(0.95 * base_config.insert_depth)
            success_lateral_threshold = float(geometry.radial_clearance)
            success = bool(
                completed
                and final_depth >= success_depth_threshold
                and final_lateral <= success_lateral_threshold
            )
            if success:
                failure_reason = ""
            elif not completed:
                failure_reason = "incomplete_rollout"
            elif not arrays_finite:
                failure_reason = "nonfinite_rollout"
            elif final_depth < success_depth_threshold and final_lateral > success_lateral_threshold:
                failure_reason = "insufficient_depth_and_lateral_misalignment"
            elif final_depth < success_depth_threshold:
                failure_reason = "insufficient_depth"
            elif final_lateral > success_lateral_threshold:
                failure_reason = "lateral_misalignment"
            else:
                failure_reason = "unknown"
            label_eligible, spike_suspicious, exclusion_reason = classify_episode_admission(
                episode_complete=completed,
                arrays_finite=arrays_finite,
                full_step_max_force=full_max_force,
                sampled_max_force=sampled_max_force,
            )
            sample_stop = len(sample_rows["episode_id"])
            capture_ratio = sampled_max_force / full_max_force if full_max_force > 0.0 else 1.0
            episode_row = {
                "episode_id": episode_id,
                "episode_spec_id": episode_spec.episode_spec_id,
                "episode_spec_schema_version": EPISODE_SPEC_SCHEMA_VERSION,
                "trajectory_source": EPISODE_TRAJECTORY_SOURCE_OPEN_LOOP_FAMILY,
                "trajectory_family": plan.family,
                "trajectory_family_id": plan.family_id,
                "teleop_mode": teleop_mode,
                "sample_start": sample_start,
                "sample_stop": sample_stop,
                "sample_count": sample_stop - sample_start,
                "total_steps": plan.total_steps,
                "nominal_hole_xy_x": float(nominal_hole_xy[0]),
                "nominal_hole_xy_y": float(nominal_hole_xy[1]),
                "actual_hole_xy_x": float(episode_spec.actual_hole_xy[0]),
                "actual_hole_xy_y": float(episode_spec.actual_hole_xy[1]),
                "trajectory_center_xy_x": float(episode_spec.trajectory_center_xy[0]),
                "trajectory_center_xy_y": float(episode_spec.trajectory_center_xy[1]),
                "trajectory_minus_hole_xy_x": float(episode_spec.trajectory_minus_hole_xy[0]),
                "trajectory_minus_hole_xy_y": float(episode_spec.trajectory_minus_hole_xy[1]),
                "success": success,
                "final_depth": final_depth,
                "final_lateral_error": final_lateral,
                "final_peg_tip_x": float(final_peg_tip[0]),
                "final_peg_tip_y": float(final_peg_tip[1]),
                "final_peg_tip_z": float(final_peg_tip[2]),
                "final_hole_center_x": float(final_hole_center[0]),
                "final_hole_center_y": float(final_hole_center[1]),
                "final_hole_center_z": float(final_hole_center[2]),
                "final_target_position_x": float(final_target_position[0]),
                "final_target_position_y": float(final_target_position[1]),
                "final_target_position_z": float(final_target_position[2]),
                "success_depth_threshold": success_depth_threshold,
                "success_lateral_threshold": success_lateral_threshold,
                "hole_yaw": float(scene_config.get("hole", {}).get("rotation", 0.0)) + float(perturbation.hole_yaw_offset),
                "peg_yaw": float(final_state[5]),
                "insertion_axis_x": 0.0,
                "insertion_axis_y": 0.0,
                "insertion_axis_z": -1.0,
                "failure_reason": failure_reason,
                "full_step_max_force": full_max_force,
                "sampled_max_force": sampled_max_force,
                "capture_ratio": capture_ratio,
                "contact_count": contact_count,
                "contact_onset_step": contact_onset,
                "max_abs_commanded_torque": max_torque,
                "torque_saturation_count": torque_saturation_count,
                "episode_complete": completed,
                "arrays_finite": arrays_finite,
                "solver_spike_suspicious": spike_suspicious,
                "label_eligible": label_eligible,
                "exclusion_reason": exclusion_reason,
                "clearance_delta": float(perturbation.clearance_delta),
                "perturbation": perturbation_dict,
            }
            should_commit = success or not successful_episodes_only
            if should_commit:
                episode_rows.append(episode_row)
                family_success_counts[family] += int(success)
                print(
                    f"episode={episode_id:03d} family={plan.family} success={success} "
                    f"eligible={label_eligible} samples={sample_stop - sample_start} "
                    f"depth={final_depth:.6f} lat={final_lateral:.6f} "
                    f"full_max={full_max_force:.3f} sampled_max={sampled_max_force:.3f}"
                )
                episode_slot += 1
                break

            discarded_failed_attempts += 1
            for key, values in sample_rows.items():
                del values[sample_length_snapshot[key]:]
            del rgb_rows[rgb_length_snapshot:]
            del episode_rows[episode_rows_length:]
            del episode_perturbations[episode_perturbations_length:]
            del episode_plan_parameters[episode_plan_parameters_length:]
            del episode_specs[episode_specs_length:]
            print(
                f"episode={episode_id:03d} attempt={episode_attempt:02d} family={plan.family} "
                f"success={success} eligible={label_eligible} discarded_retry=True "
                f"depth={final_depth:.6f} lat={final_lateral:.6f} "
                f"full_max={full_max_force:.3f} sampled_max={sampled_max_force:.3f}"
            )

    raw_arrays = {
        "state": np.vstack(sample_rows["state"]),
        "action": np.vstack(sample_rows["action"]),
        "contact_state": np.vstack(sample_rows["contact_state"]),
        "task_state": np.vstack(sample_rows["task_state"]),
        "contact_force_world": np.vstack(sample_rows["contact_force_world"]),
        "normal_force": np.asarray(sample_rows["normal_force"], dtype=float),
        "episode_id": np.asarray(sample_rows["episode_id"], dtype=np.int32),
        "sample_step": np.asarray(sample_rows["sample_step"], dtype=np.int32),
        "timestamp": np.asarray(sample_rows["timestamp"], dtype=float),
        "randomization": np.vstack(sample_rows["randomization"]),
        "planned_target_position": np.vstack(sample_rows["planned_target_position"]),
        "planned_target_rotation": np.vstack(sample_rows["planned_target_rotation"]),
        "trajectory_family_id": np.asarray(sample_rows["trajectory_family_id"], dtype=np.int8),
        "trajectory_parameters": np.vstack(sample_rows["trajectory_parameters"]),
        "phase_id": np.asarray(sample_rows["phase_id"], dtype=np.int8),
        "episode_summary_id": np.arange(len(episode_rows), dtype=np.int32),
        "episode_success": np.asarray([row["success"] for row in episode_rows], dtype=bool),
        "episode_final_depth": np.asarray([row["final_depth"] for row in episode_rows], dtype=float),
        "episode_final_lateral_error": np.asarray(
            [row["final_lateral_error"] for row in episode_rows],
            dtype=float,
        ),
        "episode_max_normal_force": np.asarray(
            [row["full_step_max_force"] for row in episode_rows],
            dtype=float,
        ),
        "episode_sampled_max_normal_force": np.asarray(
            [row["sampled_max_force"] for row in episode_rows],
            dtype=float,
        ),
        "episode_force_capture_ratio": np.asarray(
            [row["capture_ratio"] for row in episode_rows],
            dtype=float,
        ),
        "episode_contact_count": np.asarray([row["contact_count"] for row in episode_rows], dtype=np.int32),
        "episode_contact_onset_step": np.asarray(
            [row["contact_onset_step"] for row in episode_rows],
            dtype=np.int32,
        ),
        "episode_perturbation": np.vstack(episode_perturbations),
        "episode_command_xy_offset": np.vstack(
            [parameters[7:9] for parameters in episode_plan_parameters]
        ),
        "episode_trajectory_family_id": np.asarray(
            [row["trajectory_family_id"] for row in episode_rows],
            dtype=np.int8,
        ),
        "episode_trajectory_parameters": np.vstack(episode_plan_parameters),
        "episode_complete": np.asarray([row["episode_complete"] for row in episode_rows], dtype=bool),
        "episode_solver_spike_suspicious": np.asarray(
            [row["solver_spike_suspicious"] for row in episode_rows],
            dtype=bool,
        ),
        "episode_label_eligible": np.asarray(
            [row["label_eligible"] for row in episode_rows],
            dtype=bool,
        ),
    }
    if save_rgb:
        if len(rgb_rows) != int(raw_arrays["task_state"].shape[0]):
            raise RuntimeError("RGB capture count does not match sample count.")
        raw_arrays["rgb_images"] = np.stack(rgb_rows, axis=0).astype(np.uint8, copy=False)
    raw_metadata = {
        "schema_version": "residual_bc_open_loop_raw_v1",
        "scene": scene,
        "setting_id": preset.setting_id,
        "profile_name": profile_name,
        "teleop_mode": teleop_mode,
        "eye_in_hand_camera_pose_version": canonical_camera_pose["pose_version"],
        "eye_in_hand_camera_canonical": bool(canonical_camera_pose["canonical"]),
        "eye_in_hand_camera_name": canonical_camera_pose["camera_name"],
        "eye_in_hand_camera_attachment_parent": canonical_camera_pose["attachment_parent"],
        "eye_in_hand_camera_mount_type": canonical_camera_pose["mount_type"],
        "eye_in_hand_camera_pose": canonical_camera_pose,
        "collection_seed": seed,
        **profile_metadata,
        "robustness_preset": preset.to_metadata(),
        "controller_id": controller_entry.controller_id,
        "scenario_id": scenario_id,
        "collection_controller_id": controller_entry.controller_id,
        "collection_stiffness_matrix": controller_entry.position_stiffness_matrix.tolist(),
        "dataset_path": str(raw_path),
        "controllers_yaml": str(controllers_yaml),
        "base_profile": profile,
        "controller_profile": profile,
        "gain_config": str(gain_config),
        "trajectory_plan": "randomized_open_loop_v1",
        "trajectory_source": EPISODE_TRAJECTORY_SOURCE_OPEN_LOOP_FAMILY,
        "trajectory_families": list(TRAJECTORY_FAMILIES),
        "trajectory_parameter_fields": list(TRAJECTORY_PARAMETER_FIELDS),
        "successful_episodes_only": bool(successful_episodes_only),
        "collection_attempts": int(attempted_episodes),
        "discarded_failed_attempts": int(discarded_failed_attempts),
        "num_episodes": int(len(episode_rows)),
        "num_samples": int(raw_arrays["task_state"].shape[0]),
        "sample_stride": sample_stride,
        "rgb_enabled": bool(save_rgb),
        "rgb_camera_name": camera_name if save_rgb else None,
        "rgb_image_width": int(image_width) if save_rgb else None,
        "rgb_image_height": int(image_height) if save_rgb else None,
        "rgb_image_stride": int(image_stride) if save_rgb else None,
        "rgb_max_episodes": None if max_rgb_episodes is None else int(max_rgb_episodes),
        "renderer_mode": collection_renderer_mode if save_rgb else None,
        "fallback_used": collection_fallback_used if save_rgb else None,
        "native_launcher_required": default_native_launcher(),
        "episode_specs_path": str(episode_specs_path),
        "frozen_paired_episode_specs_path": str(frozen_episode_specs_path),
        "frozen_train_val_split_path": str(frozen_split_path),
        "episode_spec_schema_version": EPISODE_SPEC_SCHEMA_VERSION,
        "simulation_timestep": 0.002,
        "sample_rate_hz": 1.0 / (0.002 * sample_stride),
        "seed": seed,
        "rng_streams": {
            "root_seed": seed,
            "perturbation_seed": perturbation_seed,
            "trajectory_seed": trajectory_seed,
            "split_seed": split_seed,
        },
        "randomization_fields": list(RANDOMIZATION_FIELDS),
        "legacy_field_mapping": profile_metadata.get("legacy_field_mapping"),
        "admission_rule": {
            "solver_spike_full_step_threshold": 1000.0,
            "solver_spike_capture_ratio_threshold": 0.05,
            "solver_spike_sampled_force_ceiling": 500.0,
            "whole_episode_quarantine": True,
        },
        "labels_built": False,
    }
    raw_metadata["training_data_valid"], raw_metadata["training_data_valid_reason"] = infer_training_data_valid(
        raw_metadata,
        rgb_images_present=bool(save_rgb) and "rgb_images" in raw_arrays,
    )
    np.savez_compressed(raw_path, **raw_arrays, metadata=json.dumps(raw_metadata, sort_keys=True))

    eligible_episode_ids = np.asarray(
        [row["episode_id"] for row in episode_rows if row["label_eligible"]],
        dtype=np.int32,
    )
    eligible_episode_mask = np.asarray(raw_arrays["episode_label_eligible"], dtype=bool)
    eligible_sample_mask = np.isin(np.asarray(raw_arrays["episode_id"], dtype=np.int32), eligible_episode_ids)
    eligible_success = np.asarray([row["success"] for row in episode_rows if row["label_eligible"]], dtype=bool)
    train_ids, val_ids = _stratified_split(
        eligible_episode_ids,
        eligible_success,
        rng=split_rng,
    )
    eligible_path_default = raw_path.parent / "eligible_residual_bc.npz"
    eligible_path: Path | None = None
    if build_labels:
        if not np.any(eligible_sample_mask):
            raise RuntimeError(f"No label-eligible samples were collected. Raw artifact is available at {raw_path}.")
        eligible_path = build_eligible_residual_dataset_from_raw(
            raw_path,
            output_path=eligible_path_default,
            label_neighbors=label_neighbors,
            knn_block_size=knn_block_size,
            label_projection=label_projection,
            residual_bound=residual_bound,
            label_k_min=label_k_min,
            label_k_max=label_k_max,
            baseline_k=baseline_k,
            diagnostic_k_min=diagnostic_k_min,
            diagnostic_k_max=diagnostic_k_max,
            l21_coupling_percentile=l21_coupling_percentile,
            contact_gate_low=contact_gate_low,
            contact_gate_high=contact_gate_high,
            neutral_contact_threshold=neutral_contact_threshold,
            min_calibration_samples=min_calibration_samples,
            train_episode_ids=train_ids,
            val_episode_ids=val_ids,
        )

    write_episode_specs_jsonl(episode_specs_path, episode_specs)

    with episode_csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv_fields = [
            key for key in episode_rows[0] if key != "perturbation"
        ] + [f"perturbation_{name}" for name in RANDOMIZATION_FIELDS]
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row, values in zip(episode_rows, episode_perturbations, strict=True):
            flat = {key: value for key, value in row.items() if key != "perturbation"}
            flat.update(
                {
                    f"perturbation_{name}": float(value)
                    for name, value in zip(RANDOMIZATION_FIELDS, values, strict=True)
                }
            )
            writer.writerow(flat)

    family_summary: dict[str, dict[str, Any]] = {}
    for family in TRAJECTORY_FAMILIES:
        rows = [row for row in episode_rows if row["trajectory_family"] == family]
        family_failures = (
            int(family_attempt_counts[family] - len(rows))
            if successful_episodes_only
            else int(sum(not row["success"] for row in rows))
        )
        family_summary[family] = {
            "attempts": int(family_attempt_counts[family]),
            "episodes": len(rows),
            "eligible": sum(row["label_eligible"] for row in rows),
            "quarantined": sum(not row["label_eligible"] for row in rows),
            "successes": sum(row["success"] for row in rows),
            "failures": family_failures,
            "solver_spike_suspicious": sum(row["solver_spike_suspicious"] for row in rows),
            "mean_full_step_max_force": float(np.mean([row["full_step_max_force"] for row in rows]))
            if rows
            else None,
        }
    _write_json(family_summary_path, family_summary)

    frozen_split = FrozenTrainValSplit(
        dataset_path=str(eligible_path_default),
        collection_seed=int(seed),
        split_seed=int(split_seed),
        train_episode_ids=np.asarray(train_ids, dtype=np.int64),
        val_episode_ids=np.asarray(val_ids, dtype=np.int64),
        metadata={
            "scene": scene,
            "setting_id": preset.setting_id,
            "profile_name": profile_name,
            "controllers_yaml": str(controllers_yaml),
            "collection_controller_id": controller_entry.controller_id,
        },
    )
    frozen_split.write(frozen_split_path)
    shutil.copy2(episode_specs_path, frozen_episode_specs_path)

    contact_fraction = float(np.mean(np.asarray(raw_arrays["contact_state"], dtype=float)[eligible_sample_mask, 0] > 0.5))
    summary = {
        "status": "passed",
        "runtime_seconds": time.perf_counter() - started,
        "raw_dataset": str(raw_path),
        "eligible_dataset": None if eligible_path is None else str(eligible_path),
        "collection_seed": int(seed),
        "split_seed": int(split_seed),
        "successful_episodes_only": bool(successful_episodes_only),
        "requested_episodes": int(episodes),
        "episodes": int(len(episode_rows)),
        "attempts": int(attempted_episodes),
        "failed_attempts": int(discarded_failed_attempts),
        "eligible_episodes": int(eligible_episode_ids.size),
        "quarantined_episodes": int(episodes - eligible_episode_ids.size),
        "successes": int(sum(row["success"] for row in episode_rows)),
        "failures": int(discarded_failed_attempts if successful_episodes_only else sum(not row["success"] for row in episode_rows)),
        "raw_samples": int(raw_arrays["task_state"].shape[0]),
        "eligible_samples": int(np.count_nonzero(eligible_sample_mask)),
        "eligible_contact_fraction": contact_fraction,
        "catastrophic_episodes": int(
            sum(row["full_step_max_force"] >= 1000.0 for row in episode_rows)
        ),
        "solver_spike_suspicious_episodes": int(
            sum(row["solver_spike_suspicious"] for row in episode_rows)
        ),
        "trajectory_family_counts": {
            family: int(sum(row["trajectory_family"] == family for row in episode_rows))
            for family in TRAJECTORY_FAMILIES
        },
        "rng_streams": raw_metadata["rng_streams"],
        "labels_built": bool(build_labels),
        "episode_selection_mode": "success_only_rejection_sampling" if successful_episodes_only else "attempt_based",
        "validation": {
            "raw_arrays_finite": bool(
                all(
                    np.all(np.isfinite(value))
                    for value in raw_arrays.values()
                    if np.issubdtype(np.asarray(value).dtype, np.number)
                )
            ),
            "sample_stride": sample_stride,
            "sample_rate_hz": raw_metadata["sample_rate_hz"],
        },
        "eligible_validation": {"eligible_residual_dataset": "passed"} if build_labels else None,
        "frozen_paired_episode_specs_path": str(frozen_episode_specs_path),
        "frozen_train_val_split_path": str(frozen_split_path),
    }
    _write_json(summary_path, summary)

    return OpenLoopCollectionResult(
        output_dir=output_dir,
        raw_dataset=raw_path,
        eligible_dataset=eligible_path,
        collection_summary=summary_path,
        episode_csv=episode_csv_path,
        episode_specs=episode_specs_path,
        family_summary=family_summary_path,
    )


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "build_eligible_residual_dataset_from_raw",
    "OpenLoopCollectionResult",
    "OpenLoopTrajectoryPlan",
    "TRAJECTORY_FAMILIES",
    "TRAJECTORY_PARAMETER_FIELDS",
    "classify_episode_admission",
    "collect_randomized_open_loop_residual_dataset",
    "generate_open_loop_trajectory",
]
