from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import ContactQuery, contact_state_vector, extract_contact_state, extract_net_peg_hole_contact_force_world
from stiffness_copilot_mujoco.controllers.impedance import task_space_impedance_torque
from stiffness_copilot_mujoco.controllers.track_a_controllers import (
    DEFAULT_TRACK_A_CONTROLLERS_YAML,
    load_track_a_controller_runtime,
)
from stiffness_copilot_mujoco.franka_viewer import load_model
from stiffness_copilot_mujoco.learning.dataset_collection import _action_row, _clip_torque, _state_row
from stiffness_copilot_mujoco.learning.residual_label_projection import (
    DEFAULT_BASELINE_K,
    DEFAULT_CONTACT_GATE_HIGH,
    DEFAULT_CONTACT_GATE_LOW,
    DEFAULT_DIAGNOSTIC_K_MAX,
    DEFAULT_DIAGNOSTIC_K_MIN,
    DEFAULT_L21_COUPLING_PERCENTILE,
    DEFAULT_MIN_CALIBRATION_SAMPLES,
    DEFAULT_NEUTRAL_CONTACT_THRESHOLD,
    DEFAULT_RESIDUAL_BOUND,
    LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
    LABEL_PROJECTION_LEGACY,
    LABEL_PROJECTION_RESIDUAL_FIRST,
    is_residual_first_projection,
    project_residual_first_labels,
)
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.learning.stiffness_labels import (
    StiffnessLabelConfig,
    build_stiffness_labels_with_diagnostics,
    matrix_to_cholesky_params,
)
from stiffness_copilot_mujoco.learning.privileged_task_state import describe_privileged_task_state_schema
from stiffness_copilot_mujoco.learning.vision_residual_dataset import infer_training_data_valid
from stiffness_copilot_mujoco.learning.supervised_policy import scale_normalized_stiffness
from stiffness_copilot_mujoco.learning.task_state import peg_hole_task_state
from stiffness_copilot_mujoco.metrics.task_metrics import geometry_from_config, hole_center_position
from stiffness_copilot_mujoco.panda_control import arm_qpos, panda_arm_ids, set_arm_torque_ctrl
from stiffness_copilot_mujoco.pose_math import site_rotation
from stiffness_copilot_mujoco.robustness import RobustnessPreset, get_robustness_preset, sample_robustness_perturbations
from stiffness_copilot_mujoco.rollout_observation import collect_step, reset_from_config
from stiffness_copilot_mujoco.rollouts.fixed_impedance import (
    RolloutConfig,
    cleanup_runtime_scene,
    phase_for_step,
    scene_for_rollout,
    target_position_for_phase,
    target_position_with_teleop_noise,
    target_rotation_with_peg_tilt,
)
from stiffness_copilot_mujoco.scenes import get_scene_spec, parse_active_groups
from stiffness_copilot_mujoco.sim.ids import peg_hole_ids
from stiffness_copilot_mujoco.sim.scene import ROOT


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "datasets" / "residual_bc_residual_first_v1"
DEFAULT_GAIN_CONFIG = ROOT / "configs" / "controllers" / "fixed_impedance.yaml"
EPISODE_ARRAY_KEYS = {
    "episode_summary_id",
    "episode_success",
    "episode_final_depth",
    "episode_final_lateral_error",
    "episode_max_normal_force",
    "episode_contact_count",
    "episode_contact_onset_step",
    "episode_perturbation",
    "episode_command_xy_offset",
    "episode_trajectory_family_id",
    "episode_trajectory_parameters",
    "train_episode_ids",
    "val_episode_ids",
}
SAMPLE_ALIGNED_OPTIONAL_KEYS = {
    "sample_step",
    "phase_id",
    "trajectory_family_id",
    "trajectory_parameters",
    "randomization",
    "planned_target_position",
    "planned_target_rotation",
    "rgb_images",
}


def validate_residual_dataset(path: Path) -> dict:
    if path.is_dir():
        candidate = path / "eligible_residual_bc.npz"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"{path} is a directory but does not contain eligible_residual_bc.npz.")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    required = (
        "state",
        "action",
        "contact_state",
        "task_state",
        "contact_force_world",
        "stiffness_matrix_target",
        "stiffness_cholesky_target",
        "residual_group_target",
        "residual_theta_target",
        "theta_base",
        "episode_id",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Residual dataset missing keys: {missing}")
    n = arrays["task_state"].shape[0]
    if n <= 0:
        raise ValueError("Residual dataset must contain at least one sample.")
    for key, value in arrays.items():
        if key in {"theta_base", "base_stiffness_matrix"} or key in EPISODE_ARRAY_KEYS:
            continue
        if np.asarray(value).shape[0] != n:
            raise ValueError(f"{key} length does not match task_state length.")
        if np.issubdtype(np.asarray(value).dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{key} contains non-finite values.")
    for key in SAMPLE_ALIGNED_OPTIONAL_KEYS:
        if key not in arrays:
            continue
        if np.asarray(arrays[key]).shape[0] != n:
            raise ValueError(f"{key} length does not match task_state length.")
    eigvals = np.linalg.eigvalsh(arrays["stiffness_matrix_target"])
    if np.any(eigvals <= 0.0):
        raise ValueError("stiffness_matrix_target must be SPD.")
    spec = metadata.get("base_stiffness_spec", {})
    bounds = np.asarray(spec.get("residual_bounds", []), dtype=float)
    if bounds.size:
        residual = np.asarray(arrays["residual_group_target"], dtype=float)
        if residual.shape[1] != bounds.size:
            raise ValueError(f"residual_group_target dim {residual.shape[1]} does not match residual bounds {bounds.size}.")
        if np.any(np.abs(residual) > bounds[None, :] + 1e-9):
            raise ValueError("residual_group_target exceeds residual bounds.")
        if is_residual_first_projection(metadata.get("label_projection")):
            for key in (
                "residual_bound",
                "baseline_k",
                "diagnostic_k_min",
                "diagnostic_k_max",
                "diagnostic_reconstruction_role",
            ):
                if key not in metadata:
                    raise ValueError(f"Residual-first dataset metadata missing {key!r}.")
            if "physical_stiffness_matrix_target" not in arrays:
                raise ValueError("Residual-first dataset missing diagnostic physical_stiffness_matrix_target.")
            physical = np.asarray(arrays["physical_stiffness_matrix_target"], dtype=float)
            eigvals = np.linalg.eigvalsh(physical)
            if np.any(eigvals <= 0.0):
                raise ValueError("Residual-first diagnostic physical_stiffness_matrix_target must be SPD.")
            if metadata.get("label_projection") == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2:
                for key in (
                    "contact_gate_applied",
                    "free_space_residual_policy",
                    "contact_gate_low",
                    "contact_gate_high",
                    "contact_gate_mean",
                    "contact_gate_zero_fraction",
                    "contact_gate_one_fraction",
                    "neutral_center_source",
                    "neutral_center_values",
                    "neutral_mask_count",
                    "robust_scale_source",
                    "robust_scale_values",
                    "scale_mask_count",
                    "residual_group_target_neutral",
                ):
                    if key not in metadata:
                        raise ValueError(f"Contact-gated residual dataset metadata missing {key!r}.")
    if "episode_summary_id" in arrays:
        episode_count = int(np.asarray(arrays["episode_summary_id"]).shape[0])
        for key in EPISODE_ARRAY_KEYS:
            if key not in arrays or key in {"train_episode_ids", "val_episode_ids"}:
                continue
            if np.asarray(arrays[key]).shape[0] != episode_count:
                raise ValueError(f"{key} length does not match episode_summary_id length.")
        if "train_episode_ids" in arrays and "val_episode_ids" in arrays:
            train_ids = np.asarray(arrays["train_episode_ids"], dtype=int)
            val_ids = np.asarray(arrays["val_episode_ids"], dtype=int)
            if np.intersect1d(train_ids, val_ids).size:
                raise ValueError("Train and validation episode ids overlap.")
    return metadata


def _stratified_episode_split(
    episode_ids: np.ndarray,
    success: np.ndarray,
    *,
    rng: np.random.Generator,
    train_fraction: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(episode_ids, dtype=np.int32)
    outcomes = np.asarray(success, dtype=bool)
    if ids.ndim != 1 or outcomes.shape != ids.shape:
        raise ValueError("Episode ids and success outcomes must be aligned rank-1 arrays.")

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for outcome in (False, True):
        group = ids[outcomes == outcome].copy()
        rng.shuffle(group)
        if group.size <= 1:
            train_count = group.size
        else:
            train_count = int(round(float(train_fraction) * group.size))
            train_count = min(max(train_count, 1), group.size - 1)
        train_parts.append(group[:train_count])
        val_parts.append(group[train_count:])

    train = np.sort(np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int32))
    val = np.sort(np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int32))
    if val.size == 0 and train.size > 1:
        val = train[-1:].copy()
        train = train[:-1]
    if np.intersect1d(train, val).size:
        raise RuntimeError("Train and validation episode splits overlap.")
    return train, val


def _project_label_targets(
    normalized_target: np.ndarray,
    *,
    contact_forces_world: np.ndarray | None,
    base_spec: BaseStiffnessSpec,
    label_projection: str,
    residual_bound: float,
    label_k_min: float | None,
    label_k_max: float | None,
    baseline_k: float,
    diagnostic_k_min: float,
    diagnostic_k_max: float,
    l21_coupling_percentile: float,
    contact_gate_low: float,
    contact_gate_high: float,
    neutral_contact_threshold: float,
    min_calibration_samples: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if label_projection == LABEL_PROJECTION_LEGACY:
        if label_k_min is None or label_k_max is None:
            raise ValueError("Legacy label projection requires label_k_min and label_k_max.")
        physical_matrix_target = np.stack(
            [scale_normalized_stiffness(matrix, k_min=label_k_min, k_max=label_k_max) for matrix in normalized_target],
            axis=0,
        )
        physical_cholesky_target = np.stack(
            [matrix_to_cholesky_params(matrix) for matrix in physical_matrix_target],
            axis=0,
        )
        residual_group = np.vstack([base_spec.target_to_group_delta(theta) for theta in physical_cholesky_target])
        residual_theta = np.vstack([base_spec.expand_group_delta(delta, clip=True) for delta in residual_group])
        physical_eigenvalues = np.linalg.eigvalsh(physical_matrix_target)
        metadata = {
            "label_projection": LABEL_PROJECTION_LEGACY,
            "label_projection_legacy": LABEL_PROJECTION_LEGACY,
            "primary_supervision_target": "residual_target",
            "primary_target_space": "legacy_direct_physical_k_then_residual",
            "residual_to_physical_semantics": "legacy_direct_k_phys_then_residual_projection",
            "label_projection_description": "legacy direct physical stiffness target followed by residual projection",
            "baseline_k": float(baseline_k),
            "diagnostic_k_min": float(label_k_min),
            "diagnostic_k_max": float(label_k_max),
            "residual_bound": float(residual_bound),
            "physical_stiffness_matrix_target_role": "legacy_direct_physical_target",
            "legacy_label_k_min": float(label_k_min),
            "legacy_label_k_max": float(label_k_max),
            "label_k_range_role": "legacy_projection_active_for_relabeling",
        }
        return (
            {
                "residual_group_target": residual_group,
                "residual_theta_target": residual_theta,
                "physical_stiffness_matrix_target": physical_matrix_target,
                "physical_stiffness_cholesky_target": physical_cholesky_target,
                "physical_stiffness_eigenvalues": physical_eigenvalues,
            },
            metadata,
        )

    if label_projection not in (
        LABEL_PROJECTION_RESIDUAL_FIRST,
        LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
        LABEL_PROJECTION_LEGACY,
    ):
        raise ValueError(
            "label_projection must be either "
            f"{LABEL_PROJECTION_RESIDUAL_FIRST!r}, {LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2!r} or {LABEL_PROJECTION_LEGACY!r}, observed {label_projection!r}."
        )
    projected = project_residual_first_labels(
        normalized_target,
        base_spec=base_spec,
        contact_forces_world=contact_forces_world,
        label_projection=label_projection,
        residual_bound=residual_bound,
        baseline_k=baseline_k,
        diagnostic_k_min=diagnostic_k_min,
        diagnostic_k_max=diagnostic_k_max,
        l21_coupling_percentile=l21_coupling_percentile,
        contact_gate_low=contact_gate_low,
        contact_gate_high=contact_gate_high,
        neutral_contact_threshold=neutral_contact_threshold,
        min_calibration_samples=min_calibration_samples,
    )
    metadata = dict(projected.label_metadata)
    metadata.update(
        {
            "label_k_min": 300.0,
            "label_k_max": 600.0,
            "label_k_min_deprecated": True,
            "label_k_max_deprecated": True,
            "label_k_range_role": "deprecated_legacy_compatibility_only",
        }
    )
    return (
        {
            "residual_group_target": projected.residual_group_target,
            "residual_theta_target": projected.residual_theta_target,
            "physical_stiffness_matrix_target": projected.physical_stiffness_matrix_target,
            "physical_stiffness_cholesky_target": projected.physical_stiffness_cholesky_target,
            "physical_stiffness_eigenvalues": projected.physical_stiffness_eigenvalues,
        },
        metadata,
    )


def relabel_residual_dataset(
    input_path: Path,
    output_path: Path,
    *,
    label_projection: str = LABEL_PROJECTION_RESIDUAL_FIRST,
    residual_bound: float | None = None,
    baseline_k: float = DEFAULT_BASELINE_K,
    diagnostic_k_min: float = DEFAULT_DIAGNOSTIC_K_MIN,
    diagnostic_k_max: float = DEFAULT_DIAGNOSTIC_K_MAX,
    l21_coupling_percentile: float = DEFAULT_L21_COUPLING_PERCENTILE,
    contact_gate_low: float = DEFAULT_CONTACT_GATE_LOW,
    contact_gate_high: float = DEFAULT_CONTACT_GATE_HIGH,
    neutral_contact_threshold: float = DEFAULT_NEUTRAL_CONTACT_THRESHOLD,
    min_calibration_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
    label_k_min: float | None = None,
    label_k_max: float | None = None,
) -> Path:
    """Reuse rollout samples while changing physical label scale or residual bounds."""
    if input_path.is_dir():
        candidate = input_path / "eligible_residual_bc.npz"
        if not candidate.exists():
            raise FileNotFoundError(f"{input_path} is a directory but does not contain eligible_residual_bc.npz.")
        input_path = candidate
    validate_residual_dataset(input_path)
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))

    base_spec = BaseStiffnessSpec.from_metadata(metadata["base_stiffness_spec"])
    if residual_bound is not None:
        if residual_bound <= 0.0:
            raise ValueError("residual_bound must be positive.")
        base_spec = BaseStiffnessSpec(
            base_matrix=base_spec.base_matrix,
            theta_base=base_spec.theta_base,
            active_groups=base_spec.active_groups,
            active_group_names=base_spec.active_group_names,
            residual_bounds=np.full(len(base_spec.active_groups), float(residual_bound), dtype=float),
        )

    normalized_target = np.asarray(arrays["stiffness_matrix_target"], dtype=float)
    contact_forces_world = np.asarray(arrays["contact_force_world"], dtype=float)
    projection_arrays, projection_metadata = _project_label_targets(
        normalized_target,
        contact_forces_world=contact_forces_world,
        base_spec=base_spec,
        label_projection=label_projection,
        residual_bound=float(np.max(base_spec.residual_bounds)),
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
    )

    arrays.update(projection_arrays)
    arrays["task_state_6d"] = arrays["task_state"]
    arrays["privileged_task_state_6d"] = arrays["task_state"]
    arrays["theta_base"] = base_spec.theta_base
    arrays["base_stiffness_matrix"] = base_spec.base_matrix

    updated_metadata = dict(metadata)
    training_data_valid, training_data_valid_reason = infer_training_data_valid(
        updated_metadata,
        rgb_images_present="rgb_images" in arrays,
    )
    task_state_metadata = describe_privileged_task_state_schema()
    updated_metadata.update(
        {
            "schema_version": (
                "residual_bc_residual_first_v1"
                if label_projection == LABEL_PROJECTION_RESIDUAL_FIRST
                else "residual_bc_contact_gated_centered_v2"
                if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2
                else "residual_bc_v1"
            ),
            "base_stiffness_spec": base_spec.to_metadata(),
            "label_projection": label_projection,
            "residual_bound": float(np.max(base_spec.residual_bounds)),
            "baseline_k": float(baseline_k),
            "diagnostic_k_min": float(diagnostic_k_min),
            "diagnostic_k_max": float(diagnostic_k_max),
            "residual_to_physical_semantics": "negative_softens_positive_stiffens",
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
            "physical_stiffness_matrix_target_role": projection_metadata["physical_stiffness_matrix_target_role"],
            "residual_dim": int(projection_arrays["residual_group_target"].shape[1]),
            "relabel_version": (
                "residual_first_baseline_relative_v1"
                if label_projection == LABEL_PROJECTION_RESIDUAL_FIRST
                else "residual_first_contact_gated_centered_v2"
                if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2
                else "residual_bc_relabel_v1"
            ),
            "relabel_source_dataset": str(input_path),
            "relabel_source_label_k_min": metadata.get("label_k_min"),
            "relabel_source_label_k_max": metadata.get("label_k_max"),
            "training_data_valid": bool(training_data_valid),
            "training_data_valid_reason": training_data_valid_reason,
            **projection_metadata,
        }
    )
    if residual_bound is not None:
        updated_metadata["relabel_residual_bound"] = float(residual_bound)
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays, metadata=json.dumps(updated_metadata, sort_keys=True))
    validate_residual_dataset(output_path)
    return output_path


def collect_residual_dataset(
    *,
    scene: str,
    output_path: Path | None = None,
    episodes: int = 200,
    seed: int = 0,
    robustness_preset: RobustnessPreset | None = None,
    sample_stride: int = 20,
    gain_config: Path = DEFAULT_GAIN_CONFIG,
    controllers_yaml: Path = DEFAULT_TRACK_A_CONTROLLERS_YAML,
    controller_id: str | None = None,
    base_profile: str | None = None,
    active_dims: str | None = None,
    residual_bound: float | None = None,
    label_projection: str = LABEL_PROJECTION_RESIDUAL_FIRST,
    baseline_k: float = DEFAULT_BASELINE_K,
    diagnostic_k_min: float = DEFAULT_DIAGNOSTIC_K_MIN,
    diagnostic_k_max: float = DEFAULT_DIAGNOSTIC_K_MAX,
    approach_hold_steps: int = 600,
    descend_steps: int = 1200,
    insert_steps: int = 1800,
    final_hold_steps: int = 400,
    approach_height: float = 0.18,
    descend_height: float = 0.012,
    insert_depth: float = 0.03,
    command_xy_radius: float = 0.0015,
    label_neighbors: int = 32,
    knn_block_size: int = 1024,
    label_k_min: float | None = None,
    label_k_max: float | None = None,
    l21_coupling_percentile: float = DEFAULT_L21_COUPLING_PERCENTILE,
    contact_gate_low: float = DEFAULT_CONTACT_GATE_LOW,
    contact_gate_high: float = DEFAULT_CONTACT_GATE_HIGH,
    neutral_contact_threshold: float = DEFAULT_NEUTRAL_CONTACT_THRESHOLD,
    min_calibration_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
) -> Path:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    spec = get_scene_spec(scene)
    selected_controller_id = controller_id or base_profile or spec.base_profile
    if base_profile is not None and controller_id is not None and base_profile != controller_id:
        raise ValueError("--controller-id and --base-profile must refer to the same controller when both are provided.")
    active_groups = parse_active_groups(active_dims, spec.active_groups)
    controller_entry, selected_profile, gains = load_track_a_controller_runtime(
        selected_controller_id,
        controllers_yaml=controllers_yaml,
        gain_config=gain_config,
    )
    base_spec = BaseStiffnessSpec.from_matrix(
        controller_entry.position_stiffness_matrix,
        active_groups=active_groups,
        active_group_names=spec.active_group_names if active_dims is None else None,
        residual_bound=residual_bound if residual_bound is not None else spec.residual_bound,
    )

    root_seed = np.random.SeedSequence(seed)
    perturbation_seed_seq, command_seed_seq, split_seed_seq = root_seed.spawn(3)
    perturbation_seed = int(perturbation_seed_seq.generate_state(1, dtype=np.uint32)[0])
    command_seed = int(command_seed_seq.generate_state(1, dtype=np.uint32)[0])
    split_seed = int(split_seed_seq.generate_state(1, dtype=np.uint32)[0])
    command_rng = np.random.default_rng(command_seed)
    split_rng = np.random.default_rng(split_seed)
    rollout_config = RolloutConfig(
        config_path=spec.config_path,
        gain_config_path=gain_config,
        approach_hold_steps=approach_hold_steps,
        descend_steps=descend_steps,
        insert_steps=insert_steps,
        final_hold_steps=final_hold_steps,
        approach_height=approach_height,
        descend_height=descend_height,
        insert_depth=insert_depth,
    )
    preset = robustness_preset or get_robustness_preset(scene)
    perturbations = sample_robustness_perturbations(
        scene=scene,
        episodes=episodes,
        seed=perturbation_seed,
        preset=preset,
    )

    state_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    contact_rows: list[np.ndarray] = []
    task_state_rows: list[np.ndarray] = []
    force_rows: list[np.ndarray] = []
    episode_rows: list[int] = []
    timestamp_rows: list[float] = []
    randomization_rows: list[np.ndarray] = []
    episode_success_rows: list[bool] = []
    episode_final_depth_rows: list[float] = []
    episode_final_lateral_rows: list[float] = []
    episode_max_force_rows: list[float] = []
    episode_contact_count_rows: list[int] = []
    episode_contact_onset_rows: list[int] = []
    episode_perturbation_rows: list[np.ndarray] = []
    episode_command_offset_rows: list[np.ndarray] = []

    for episode_id in range(episodes):
        perturbation = perturbations[episode_id]
        scene_path, randomized_config = scene_for_rollout(spec.config_path, perturbation)
        randomization = perturbation.to_dict()
        try:
            model = load_model(scene_path)
        finally:
            cleanup_runtime_scene(scene_path)
        data = mujoco.MjData(model)
        reset_from_config(model, data, randomized_config)
        geometry = geometry_from_config(randomized_config)
        task_ids = peg_hole_ids(model, segments=geometry.segments)
        arm_ids = panda_arm_ids(model)
        nullspace_target_qpos = arm_qpos(data, arm_ids)
        hole_center = hole_center_position(data, task_ids)
        target_rotation = target_rotation_with_peg_tilt(site_rotation(data, model.site("peg_tip").id), perturbation)
        command_radius = float(command_rng.uniform(0.0, command_xy_radius))
        command_angle = float(command_rng.uniform(-np.pi, np.pi))
        xy_offset = command_radius * np.array([np.cos(command_angle), np.sin(command_angle)], dtype=float)
        episode_max_normal_force = 0.0
        episode_contact_count = 0
        episode_contact_onset = -1

        for step in range(rollout_config.total_steps + 1):
            phase, phase_step, phase_length = phase_for_step(step, rollout_config)
            target_position = target_position_for_phase(
                phase=phase,
                phase_step=phase_step,
                phase_length=phase_length,
                hole_center=hole_center,
                xy_offset=xy_offset,
                config=rollout_config,
            )
            target_position = target_position_with_teleop_noise(
                target_position,
                step=step,
                config=rollout_config,
                perturbation=perturbation,
            )
            command = task_space_impedance_torque(
                model,
                data,
                site_name="peg_tip",
                target_position=target_position,
                target_rotation=target_rotation,
                arm_ids=arm_ids,
                gains=gains,
                position_stiffness_matrix=controller_entry.position_stiffness_matrix,
                nullspace_target_qpos=nullspace_target_qpos,
                clip_to_ctrlrange=False,
            )
            torque = _clip_torque(model, arm_ids, command.torque)
            obs, u_ref, _ = collect_step(
                model,
                data,
                arm_ids=arm_ids,
                task_ids=task_ids,
                target_position=target_position,
                target_rotation=target_rotation,
                phase_id={"approach_hold": 0, "descend": 1, "insert": 2, "final_hold": 3}.get(phase, 3),
            )
            query = ContactQuery(model=model, data=data, task_ids=task_ids)
            contact = extract_contact_state(query)
            episode_max_normal_force = max(episode_max_normal_force, float(contact.normal_force))
            if contact.in_contact:
                episode_contact_count += 1
                if episode_contact_onset < 0:
                    episode_contact_onset = step
            if step % sample_stride == 0:
                state_rows.append(_state_row(obs, data, arm_ids))
                action_rows.append(_action_row(u_ref, torque))
                contact_rows.append(contact_state_vector(extract_contact_state(query)))
                task_state_rows.append(peg_hole_task_state(data, task_ids, hole_clearance_delta=float(randomization["clearance_delta"])))
                force_rows.append(extract_net_peg_hole_contact_force_world(query))
                episode_rows.append(episode_id)
                timestamp_rows.append(float(data.time))
                randomization_rows.append(
                    np.array(
                        [
                            randomization["hole_xy_offset"][0],
                            randomization["hole_xy_offset"][1],
                            randomization["hole_yaw_offset"],
                            randomization["clearance_delta"],
                            randomization["friction_scale"],
                            randomization["peg_tilt_x"],
                            randomization["peg_tilt_y"],
                            randomization["teleop_noise_xy_amplitude"],
                            randomization["teleop_noise_cycles"],
                        ],
                        dtype=float,
                    )
                )
            set_arm_torque_ctrl(model, data, arm_ids, torque)
            mujoco.mj_step(model, data)

        final_task_state = peg_hole_task_state(
            data,
            task_ids,
            hole_clearance_delta=float(randomization["clearance_delta"]),
        )
        final_depth = float(final_task_state[2])
        final_lateral = float(np.linalg.norm(final_task_state[:2]))
        episode_success_rows.append(
            bool(final_depth >= 0.95 * rollout_config.insert_depth and final_lateral <= geometry.radial_clearance)
        )
        episode_final_depth_rows.append(final_depth)
        episode_final_lateral_rows.append(final_lateral)
        episode_max_force_rows.append(episode_max_normal_force)
        episode_contact_count_rows.append(episode_contact_count)
        episode_contact_onset_rows.append(episode_contact_onset)
        episode_perturbation_rows.append(randomization_rows[-1].copy())
        episode_command_offset_rows.append(xy_offset.copy())

    task_state = np.vstack(task_state_rows)
    contact_force_world = np.vstack(force_rows)
    stiffness_matrix_target, stiffness_cholesky_target, diagnostics = build_stiffness_labels_with_diagnostics(
        task_state,
        contact_force_world,
        config=StiffnessLabelConfig(neighbors=label_neighbors, knn_block_size=knn_block_size),
    )
    projection_arrays, projection_metadata = _project_label_targets(
        stiffness_matrix_target,
        contact_forces_world=contact_force_world,
        base_spec=base_spec,
        label_projection=label_projection,
        residual_bound=float(np.max(base_spec.residual_bounds)),
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
    )
    episode_summary_id = np.arange(episodes, dtype=np.int32)
    episode_success = np.asarray(episode_success_rows, dtype=bool)
    train_episode_ids, val_episode_ids = _stratified_episode_split(
        episode_summary_id,
        episode_success,
        rng=split_rng,
    )
    arrays = {
        "state": np.vstack(state_rows),
        "action": np.vstack(action_rows),
        "contact_state": np.vstack(contact_rows),
        "task_state": task_state,
        "contact_force_world": contact_force_world,
        "stiffness_matrix_target": stiffness_matrix_target,
        "stiffness_cholesky_target": stiffness_cholesky_target,
        **projection_arrays,
        "theta_base": base_spec.theta_base,
        "base_stiffness_matrix": base_spec.base_matrix,
        "episode_id": np.asarray(episode_rows, dtype=np.int32),
        "timestamp": np.asarray(timestamp_rows, dtype=float),
        "randomization": np.vstack(randomization_rows),
        "episode_summary_id": episode_summary_id,
        "episode_success": episode_success,
        "episode_final_depth": np.asarray(episode_final_depth_rows, dtype=float),
        "episode_final_lateral_error": np.asarray(episode_final_lateral_rows, dtype=float),
        "episode_max_normal_force": np.asarray(episode_max_force_rows, dtype=float),
        "episode_contact_count": np.asarray(episode_contact_count_rows, dtype=np.int32),
        "episode_contact_onset_step": np.asarray(episode_contact_onset_rows, dtype=np.int32),
        "episode_perturbation": np.vstack(episode_perturbation_rows),
        "episode_command_xy_offset": np.vstack(episode_command_offset_rows),
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        **diagnostics,
    }
    metadata = {
        "schema_version": (
            "residual_bc_residual_first_v1"
            if label_projection == LABEL_PROJECTION_RESIDUAL_FIRST
            else "residual_bc_contact_gated_centered_v2"
            if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2
            else "residual_bc_v1"
        ),
        "scene": scene,
        "scene_config": str(spec.config_path),
        "num_episodes": episodes,
        "num_samples": int(task_state.shape[0]),
        "seed": seed,
        "rng_streams": {
            "root_seed": seed,
            "perturbation_seed": perturbation_seed,
            "command_offset_seed": command_seed,
            "split_seed": split_seed,
        },
        "setting_id": preset.setting_id,
        "robustness_preset": preset.to_metadata(),
        "sample_stride": sample_stride,
        "base_profile": selected_controller_id,
        "controller_id": selected_controller_id,
        "collection_controller_id": selected_controller_id,
        "collection_stiffness_matrix": controller_entry.position_stiffness_matrix.tolist(),
        "controllers_yaml": str(controllers_yaml),
        "collection_controller_profile": selected_profile,
        "gain_config": str(gain_config),
        "base_stiffness_spec": base_spec.to_metadata(),
        "collection_controller_metadata": controller_entry.to_metadata(),
        "target": "baseline_relative_residual",
        "label_projection": label_projection,
        "residual_bound": float(np.max(base_spec.residual_bounds)),
        "baseline_k": float(baseline_k),
        "diagnostic_k_min": float(diagnostic_k_min),
        "diagnostic_k_max": float(diagnostic_k_max),
        "residual_to_physical_semantics": "negative_softens_positive_stiffens",
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
        "physical_stiffness_matrix_target_role": projection_metadata["physical_stiffness_matrix_target_role"],
        "task_state_dim": int(task_state.shape[1]),
        "residual_dim": int(projection_arrays["residual_group_target"].shape[1]),
        "randomization_fields": [
            "hole_x",
            "hole_y",
            "hole_yaw",
            "clearance_delta",
            "friction_scale",
            "peg_tilt_x",
            "peg_tilt_y",
            "teleop_noise_xy_amplitude",
            "teleop_noise_cycles",
        ],
        "label_neighbors": label_neighbors,
        "knn_block_size": knn_block_size,
        "episode_outcome_fields": [
            "success",
            "final_depth",
            "final_lateral_error",
            "max_normal_force",
            "contact_count",
            "contact_onset_step",
            "perturbation",
            "command_xy_offset",
        ],
        "episode_split": {
            "strategy": "success_stratified_80_20",
            "train_episode_count": int(train_episode_ids.size),
            "val_episode_count": int(val_episode_ids.size),
        },
        **projection_metadata,
    }
    if label_projection == LABEL_PROJECTION_LEGACY:
        metadata["label_k_min"] = float(label_k_min) if label_k_min is not None else None
        metadata["label_k_max"] = float(label_k_max) if label_k_max is not None else None
        metadata["label_k_min_deprecated"] = False
        metadata["label_k_max_deprecated"] = False
    else:
        metadata["label_k_min"] = 300.0
        metadata["label_k_max"] = 600.0
        metadata["label_k_min_deprecated"] = True
        metadata["label_k_max_deprecated"] = True
        metadata["label_k_range_role"] = "deprecated_legacy_compatibility_only"
        if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2:
            metadata.setdefault("contact_gate_applied", True)
            metadata.setdefault("free_space_residual_policy", "baseline_anchor")
            metadata.setdefault("residual_group_target_neutral", 0.0)
    output = output_path or DEFAULT_OUTPUT_ROOT / f"{preset.setting_id}_{episodes}ep_seed{seed}_residual_bc.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays, metadata=json.dumps(metadata, sort_keys=True))
    validate_residual_dataset(output)
    return output


__all__ = ["DEFAULT_OUTPUT_ROOT", "collect_residual_dataset", "relabel_residual_dataset", "validate_residual_dataset"]
