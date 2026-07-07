from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stiffness_copilot_mujoco.learning.frozen_train_val_split import load_frozen_train_val_split  # noqa: E402
from stiffness_copilot_mujoco.learning.frozen_vision_backbone import (  # noqa: E402
    FrozenBackboneLoadConfig,
    FrozenBackbonePreprocessConfig,
    build_frozen_dinov3_backbone,
)
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec  # noqa: E402
from stiffness_copilot_mujoco.learning.vision_residual_dataset import load_vision_residual_dataset  # noqa: E402
from stiffness_copilot_mujoco.learning.vision_residual_stiffness import (  # noqa: E402
    DEFAULT_HEAD_DROPOUT,
    DEFAULT_HEAD_HIDDEN_DIMS,
    LINEAR_OR_MLP_V1_HEAD_TYPE,
    MLP_V2_HEAD_TYPE,
    describe_residual_policy_contract,
    load_image_only_residual_bc_policy,
    save_vision_residual_policy,
    train_head_with_adam,
)
from stiffness_copilot_mujoco.controllers.track_a_controllers import (  # noqa: E402
    DEFAULT_TRACK_A_CONTROLLERS_YAML,
    load_track_a_controller_runtime,
)
from stiffness_copilot_mujoco.learning.conservative_image_augmentation import (  # noqa: E402
    LIGHT_CONSERVATIVE_AUGMENTATION_BLACKLIST,
    LIGHT_CONSERVATIVE_AUGMENTATION_MODE,
    LIGHT_CONSERVATIVE_AUGMENTATION_SCOPE,
    LIGHT_CONSERVATIVE_AUGMENTATION_SPEC_VERSION,
    LIGHT_CONSERVATIVE_AUGMENTATION_WHITELIST,
    NO_AUGMENTATION_MODE,
    augment_light_conservative_rgb_batch,
    describe_light_conservative_augmentation,
)


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "models" / "vision_residual_bc"
DEFAULT_DINOV3_REPO = Path(os.environ.get("SIGMA7_DINOV3_REPO", ROOT / "third_party" / "dinov3"))
DEFAULT_DINOV3_CHECKPOINT = Path(
    os.environ.get("SIGMA7_DINOV3_CHECKPOINT", ROOT / "checkpoints" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
)
DEFAULT_DINOV3_ENTRYPOINT = "dinov3_vits16"
DEFAULT_BACKBONE_TYPE = "dinov3_small_frozen"
METHOD_NAME = "image_only_residual_bc"


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("head hidden dims must be a comma-separated list of positive integers.")
    return dims


def _encode_rgb_batches(
    backbone,
    rgb_images: np.ndarray,
    *,
    batch_size: int = 8,
    sample_ids: np.ndarray | None = None,
    augmentation_mode: str = NO_AUGMENTATION_MODE,
    train_seed: int = 0,
    progress_prefix: str | None = None,
) -> np.ndarray:
    images = np.asarray(rgb_images, dtype=np.uint8)
    if sample_ids is None:
        sample_ids = np.arange(images.shape[0], dtype=np.int64)
    else:
        sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if sample_ids.ndim != 1 or sample_ids.shape[0] != images.shape[0]:
        raise ValueError("sample_ids must be one-dimensional and align with rgb_images.")
    features: list[np.ndarray] = []
    total_batches = max(1, int(np.ceil(images.shape[0] / batch_size)))
    for start in range(0, images.shape[0], batch_size):
        stop = min(start + batch_size, images.shape[0])
        batch = images[start:stop]
        batch_ids = sample_ids[start:stop]
        if progress_prefix:
            print(
                f"{progress_prefix} encode_batch={start // batch_size + 1}/{total_batches} samples={start}:{stop}",
                flush=True,
            )
        if augmentation_mode == LIGHT_CONSERVATIVE_AUGMENTATION_MODE:
            batch = augment_light_conservative_rgb_batch(batch, train_seed=train_seed, sample_ids=batch_ids)
        elif augmentation_mode != NO_AUGMENTATION_MODE:
            raise ValueError(
                f"Unsupported augmentation_mode {augmentation_mode!r}. Expected {NO_AUGMENTATION_MODE!r} or "
                f"{LIGHT_CONSERVATIVE_AUGMENTATION_MODE!r}."
            )
        encoded = backbone.encode(batch)
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().numpy()
        features.append(np.asarray(encoded, dtype=np.float32))
    return np.concatenate(features, axis=0)


def _train_val_episode_ids(dataset, split_file: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    frozen_split = load_frozen_train_val_split(split_file)
    split_dataset_path = Path(frozen_split.dataset_path).expanduser().resolve(strict=False) if frozen_split.dataset_path else None
    loaded_dataset_path_raw = dataset.metadata.get("dataset_path") or dataset.metadata.get("dataset")
    loaded_dataset_path = Path(str(loaded_dataset_path_raw)).expanduser().resolve(strict=False) if loaded_dataset_path_raw else None
    if split_dataset_path is not None and loaded_dataset_path is not None and split_dataset_path != loaded_dataset_path:
        raise ValueError(
            f"Frozen split dataset path {frozen_split.dataset_path!r} does not match the loaded dataset provenance "
            f"({loaded_dataset_path_raw!r})."
        )
    return frozen_split.train_episode_ids, frozen_split.val_episode_ids, frozen_split.to_dict()


def _default_output_for_dataset(
    dataset: Path,
    *,
    split_file: Path,
    augmentation_mode: str = NO_AUGMENTATION_MODE,
    head_type: str = LINEAR_OR_MLP_V1_HEAD_TYPE,
) -> Path:
    try:
        with np.load(dataset, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        setting_id = metadata.get("setting_id", metadata.get("scene", "unknown"))
        episodes = metadata.get("num_episodes", "unknown")
        profile_name = metadata.get("collection_controller_id") or metadata.get("base_profile") or metadata.get("profile_name")
        suffix = f"_{profile_name}" if profile_name else ""
        split_stem = split_file.stem
        head_suffix = "_head_mlp_v2" if head_type == MLP_V2_HEAD_TYPE else ""
        if augmentation_mode == LIGHT_CONSERVATIVE_AUGMENTATION_MODE:
            return DEFAULT_OUTPUT_ROOT / f"{setting_id}{suffix}_{episodes}ep_{split_stem}_dinov3_light_aug_residual_bc_policy{head_suffix}.npz"
        return DEFAULT_OUTPUT_ROOT / f"{setting_id}{suffix}_{episodes}ep_{split_stem}_dinov3_residual_bc_policy{head_suffix}.npz"
    except Exception:
        if augmentation_mode == LIGHT_CONSERVATIVE_AUGMENTATION_MODE:
            return DEFAULT_OUTPUT_ROOT / f"dinov3_light_aug_residual_bc_policy{('_head_mlp_v2' if head_type == MLP_V2_HEAD_TYPE else '')}.npz"
        return DEFAULT_OUTPUT_ROOT / f"dinov3_residual_bc_policy{('_head_mlp_v2' if head_type == MLP_V2_HEAD_TYPE else '')}.npz"


def train_frozen_dinov3_residual_bc(
    dataset: Path,
    output: Path,
    *,
    split_file: Path,
    train_seed: int = 0,
    dinov3_repo: Path = DEFAULT_DINOV3_REPO,
    dinov3_checkpoint: Path = DEFAULT_DINOV3_CHECKPOINT,
    dinov3_entrypoint: str = DEFAULT_DINOV3_ENTRYPOINT,
    head_type: str = LINEAR_OR_MLP_V1_HEAD_TYPE,
    head_hidden_dim: int = 32,
    head_hidden_dims: tuple[int, ...] | list[int] | None = None,
    head_dropout: float = DEFAULT_HEAD_DROPOUT,
    epochs: int = 20,
    validation_patience: int = 5,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    controllers_yaml: Path = DEFAULT_TRACK_A_CONTROLLERS_YAML,
    controller_id: str | None = None,
    feature_batch_size: int = 8,
    augmentation_mode: str = NO_AUGMENTATION_MODE,
) -> dict:
    if augmentation_mode not in {NO_AUGMENTATION_MODE, LIGHT_CONSERVATIVE_AUGMENTATION_MODE}:
        raise ValueError(
            f"Unsupported augmentation_mode {augmentation_mode!r}. Expected {NO_AUGMENTATION_MODE!r} or "
            f"{LIGHT_CONSERVATIVE_AUGMENTATION_MODE!r}."
        )
    if head_type not in {LINEAR_OR_MLP_V1_HEAD_TYPE, MLP_V2_HEAD_TYPE}:
        raise ValueError(f"Unsupported head_type {head_type!r}.")
    if head_type == MLP_V2_HEAD_TYPE:
        resolved_hidden_dims = tuple(head_hidden_dims) if head_hidden_dims is not None else DEFAULT_HEAD_HIDDEN_DIMS
    else:
        resolved_hidden_dims = (int(head_hidden_dim),)
    print(
        f"[vision-train] dataset={dataset} output={output} split_file={split_file} train_seed={train_seed} "
        f"augmentation_mode={augmentation_mode} head_type={head_type} head_hidden_dims={resolved_hidden_dims}",
        flush=True,
    )
    dataset_obj = load_vision_residual_dataset(dataset, input_mode="image_only", require_native_renderer=True)
    if dataset_obj.rgb_images is None:
        raise ValueError("rgb_images are required for frozen DINOv3 training.")
    train_episode_ids, val_episode_ids, split_payload = _train_val_episode_ids(dataset_obj, split_file)
    train_idx = np.flatnonzero(np.isin(dataset_obj.episode_id, train_episode_ids))
    val_idx = np.flatnonzero(np.isin(dataset_obj.episode_id, val_episode_ids))
    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError("Frozen split produced an empty train or validation set.")
    collection_controller_id = str(dataset_obj.metadata.get("collection_controller_id") or "")
    if not collection_controller_id:
        raise ValueError("Frozen DINOv3 training dataset is missing collection_controller_id.")
    selected_controller_id = controller_id or collection_controller_id
    if controller_id is not None and controller_id != collection_controller_id:
        raise ValueError(
            f"Training controller_id {controller_id!r} does not match dataset collection_controller_id {collection_controller_id!r}."
        )
    gain_config = Path(dataset_obj.metadata.get("gain_config") or (ROOT / "configs" / "controllers" / "fixed_impedance.yaml"))
    controller_entry, selected_profile, _ = load_track_a_controller_runtime(
        selected_controller_id,
        controllers_yaml=controllers_yaml,
        gain_config=gain_config,
    )
    collection_controller_matrix = np.asarray(dataset_obj.metadata.get("collection_stiffness_matrix"), dtype=float)
    if collection_controller_matrix.shape != (3, 3):
        raise ValueError("Dataset metadata field 'collection_stiffness_matrix' must have shape (3, 3).")
    if not np.allclose(collection_controller_matrix, controller_entry.position_stiffness_matrix, atol=1e-9, rtol=0.0):
        raise ValueError("Dataset collection stiffness matrix does not match the selected controller registry entry.")
    base_spec = BaseStiffnessSpec.from_metadata(dataset_obj.metadata["base_stiffness_spec"])
    contract = describe_residual_policy_contract(base_spec)
    scene_name = str(dataset_obj.metadata.get("scene") or dataset_obj.metadata.get("setting_id") or "vision")
    print(
        f"[{scene_name}-train] samples={dataset_obj.sample_count} train_samples={train_idx.size} val_samples={val_idx.size} "
        f"train_episodes={train_episode_ids.size} val_episodes={val_episode_ids.size}",
        flush=True,
    )

    backbone, backbone_metadata = build_frozen_dinov3_backbone(
        FrozenBackboneLoadConfig(
            dinov3_repo=dinov3_repo,
            dinov3_checkpoint=dinov3_checkpoint,
            dinov3_entrypoint=dinov3_entrypoint,
            preprocess_config=FrozenBackbonePreprocessConfig(),
            seed=train_seed,
        )
    )

    train_features = _encode_rgb_batches(
        backbone,
        dataset_obj.rgb_images[train_idx],
        batch_size=feature_batch_size,
        sample_ids=train_idx,
        augmentation_mode=augmentation_mode,
        train_seed=train_seed,
        progress_prefix=f"[{scene_name}-train][train-encode]",
    )
    val_features = _encode_rgb_batches(
        backbone,
        dataset_obj.rgb_images[val_idx],
        batch_size=feature_batch_size,
        sample_ids=val_idx,
        augmentation_mode=NO_AUGMENTATION_MODE,
        train_seed=train_seed,
        progress_prefix=f"[{scene_name}-train][val-encode]",
    )
    y_train_raw = dataset_obj.residual_group_target[train_idx].astype(np.float32)
    y_val_raw = dataset_obj.residual_group_target[val_idx].astype(np.float32)
    x_mean = train_features.mean(axis=0)
    x_std = train_features.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = y_train_raw.mean(axis=0)
    y_std = y_train_raw.std(axis=0)
    y_std[y_std < 1e-8] = 1.0
    train_x_norm = (train_features - x_mean) / x_std
    val_x_norm = (val_features - x_mean) / x_std
    train_y_norm = (y_train_raw - y_mean) / y_std
    val_y_norm = (y_val_raw - y_mean) / y_std

    head, history, head_summary = train_head_with_adam(
        train_x_norm,
        train_y_norm,
        head_type=head_type,
        hidden_dim=head_hidden_dim,
        hidden_dims=resolved_hidden_dims,
        head_dropout=head_dropout,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=train_seed,
        validation_patience=validation_patience,
        x_val=val_x_norm,
        y_val=val_y_norm,
        y_mean=y_mean,
        y_std=y_std,
        residual_bound=float(base_spec.residual_bounds.max()),
        progress_prefix=f"[{scene_name}-train][head]",
        progress_every=1,
    )

    architecture = {
        "backbone": {
            "type": DEFAULT_BACKBONE_TYPE,
            "frozen": True,
            "dinov3_repo": str(dinov3_repo),
            "dinov3_checkpoint": str(dinov3_checkpoint),
            "dinov3_entrypoint": dinov3_entrypoint,
            "preprocessing_config": backbone.preprocess_config.to_metadata(),
        },
        "head": {
            "type": head_type,
            "hidden_dims": [int(value) for value in resolved_hidden_dims],
            "output_dim": int(y_train_raw.shape[1]),
            "activation": head_summary["head_activation"],
            "dropout": float(head_dropout if head_type == MLP_V2_HEAD_TYPE else 0.0),
            "output_bounding": head_summary["output_bounding"],
        },
    }

    train_metadata = {
        "method_name": METHOD_NAME,
        "schema_version": "vision_residual_bc_policy_v2",
        "dataset": str(dataset),
        "dataset_path": str(dataset),
        "scene": dataset_obj.metadata.get("scene"),
        "setting_id": dataset_obj.metadata.get("setting_id") or dataset_obj.metadata.get("difficulty", "unknown"),
        "controllers_yaml": str(controllers_yaml.resolve()),
        "scenario_id": dataset_obj.metadata.get("scenario_id"),
        "collection_controller_id": collection_controller_id,
        "collection_stiffness_matrix": collection_controller_matrix.tolist(),
        "reference_controller_id": selected_controller_id,
        "reference_stiffness_matrix": controller_entry.position_stiffness_matrix.tolist(),
        "reference_controller_profile": selected_profile,
        "controller_policy_consistency_passed": True,
        "augmentation_mode": augmentation_mode,
        "augmentation_applied": augmentation_mode == LIGHT_CONSERVATIVE_AUGMENTATION_MODE,
        "augmentation_scope": LIGHT_CONSERVATIVE_AUGMENTATION_SCOPE,
        "augmentation_spec_version": LIGHT_CONSERVATIVE_AUGMENTATION_SPEC_VERSION,
        "augmentation_seed": int(train_seed) if augmentation_mode == LIGHT_CONSERVATIVE_AUGMENTATION_MODE else None,
        "augmentation_spec": describe_light_conservative_augmentation(),
        "augmentation_whitelist": [dict(entry) for entry in LIGHT_CONSERVATIVE_AUGMENTATION_WHITELIST],
        "augmentation_blacklist": list(LIGHT_CONSERVATIVE_AUGMENTATION_BLACKLIST),
        "augmentation_reproducibility": "seeded_per_sample_from_train_seed",
        "augmentation_train_only": True,
        "profile_name": dataset_obj.metadata.get("profile_name"),
        "contact_condition_name": dataset_obj.metadata.get("contact_condition_name"),
        "hole_xy_offset_semantics": dataset_obj.metadata.get("hole_xy_offset_semantics"),
        "hole_xy_offset_units": dataset_obj.metadata.get("hole_xy_offset_units"),
        "hole_xy_offset_distribution": dataset_obj.metadata.get("hole_xy_offset_distribution"),
        "trajectory_follows_randomized_hole": dataset_obj.metadata.get("trajectory_follows_randomized_hole"),
        "contact_generation_parameters_fixed": dataset_obj.metadata.get("contact_generation_parameters_fixed"),
        "fixed_hole_yaw_offset": dataset_obj.metadata.get("fixed_hole_yaw_offset"),
        "fixed_teleop_noise_xy_amplitude": dataset_obj.metadata.get("fixed_teleop_noise_xy_amplitude"),
        "fixed_teleop_noise_cycles": dataset_obj.metadata.get("fixed_teleop_noise_cycles"),
        "fixed_teleop_noise_phase_x": dataset_obj.metadata.get("fixed_teleop_noise_phase_x"),
        "fixed_teleop_noise_phase_y": dataset_obj.metadata.get("fixed_teleop_noise_phase_y"),
        "fixed_clearance_delta": dataset_obj.metadata.get("fixed_clearance_delta"),
        "fixed_friction_scale": dataset_obj.metadata.get("fixed_friction_scale"),
        "fixed_peg_tilt_x": dataset_obj.metadata.get("fixed_peg_tilt_x"),
        "fixed_peg_tilt_y": dataset_obj.metadata.get("fixed_peg_tilt_y"),
        "native_launcher_required": dataset_obj.metadata.get("native_launcher_required"),
        "legacy_field_mapping": dataset_obj.metadata.get("legacy_field_mapping"),
        "input_mode": "image_only",
        "camera_name": dataset_obj.metadata.get("rgb_camera_name") or "eye_in_hand_rgb",
        "eye_in_hand_camera_pose_version": dataset_obj.metadata.get("eye_in_hand_camera_pose_version"),
        "eye_in_hand_camera_canonical": dataset_obj.metadata.get("eye_in_hand_camera_canonical"),
        "eye_in_hand_camera_name": dataset_obj.metadata.get("eye_in_hand_camera_name"),
        "eye_in_hand_camera_attachment_parent": dataset_obj.metadata.get("eye_in_hand_camera_attachment_parent"),
        "eye_in_hand_camera_mount_type": dataset_obj.metadata.get("eye_in_hand_camera_mount_type"),
        "eye_in_hand_camera_pose": dataset_obj.metadata.get("eye_in_hand_camera_pose"),
        "image_width": int(dataset_obj.metadata.get("rgb_image_width") or dataset_obj.image_shape[1]),
        "image_height": int(dataset_obj.metadata.get("rgb_image_height") or dataset_obj.image_shape[0]),
        "image_resolution": [
            int(dataset_obj.metadata.get("rgb_image_width") or dataset_obj.image_shape[1]),
            int(dataset_obj.metadata.get("rgb_image_height") or dataset_obj.image_shape[0]),
        ],
        "renderer_mode": dataset_obj.metadata.get("renderer_mode"),
        "fallback_used": bool(dataset_obj.metadata.get("fallback_used", False)),
        "backbone_type": DEFAULT_BACKBONE_TYPE,
        "backbone_frozen": True,
        "dinov3_repo": str(dinov3_repo),
        "dinov3_checkpoint": str(dinov3_checkpoint),
        "dinov3_entrypoint": dinov3_entrypoint,
        "backbone_seed": int(train_seed),
        "stiffness_representation": "full_spd_matrix",
        "residual_parameterization": contract["residual_parameterization"],
        "residual_affects": list(contract["residual_affects"]),
        "residual_unaffected": list(contract["residual_unaffected"]),
        "smoothing_required": True,
        "samples": int(dataset_obj.sample_count),
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "train_episodes": int(train_episode_ids.size),
        "val_episodes": int(val_episode_ids.size),
        "train_validation_split": {
            "train_episode_ids": train_episode_ids.astype(int).tolist(),
            "val_episode_ids": val_episode_ids.astype(int).tolist(),
            "split_file": str(split_file),
        },
        "feature_dim": int(train_features.shape[1]),
        "target_dim": int(y_train_raw.shape[1]),
        "output_space": contract["output_space"],
        "output_dim": int(y_train_raw.shape[1]),
        "residual_bound": float(base_spec.residual_bounds.max()),
        "is_residual_policy": True,
        "is_full_stiffness_policy": False,
        "uses_task_state_input": False,
        "uses_contact_force_input": False,
        "uses_clearance_input": False,
        "uses_trajectory_phase_input": False,
        "provenance": {
            "paper_specified": [
                "wrist_rgb_observation",
                "simulation_privileged_label_construction",
                "image_conditioned_stiffness_policy_philosophy",
            ],
            "project_adapted": [
                "frozen_dinov3_backbone",
                "medium_residual_mlp_head_v2" if head_type == MLP_V2_HEAD_TYPE else "lightweight_mlp_head_v1",
                "existing_residual_safety_path",
            ],
        },
        "architecture": architecture,
        "seed": int(train_seed),
        "train_seed": int(train_seed),
        "epochs": int(len(history)),
        "requested_epochs": int(epochs),
        "validation_patience": validation_patience,
        "head_type": head_type,
        "head_hidden_dims": [int(value) for value in resolved_hidden_dims],
        "head_dropout": float(head_dropout if head_type == MLP_V2_HEAD_TYPE else 0.0),
        "head_activation": head_summary["head_activation"],
        "output_bounding": head_summary["output_bounding"],
        "feature_dim": int(train_features.shape[1]),
        "output_dim": int(y_train_raw.shape[1]),
        "hidden_dim": int(resolved_hidden_dims[0]),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "history": history,
        "training_summary": head_summary,
        "backbone_metadata": backbone_metadata,
        "train_split_path": str(split_file),
        "diagnostic_fields_available": {
            "sample_step": dataset_obj.sample_step is not None,
            "phase_id": dataset_obj.phase_id is not None,
            "trajectory_family_id": dataset_obj.trajectory_family_id is not None,
            "trajectory_parameters": dataset_obj.trajectory_parameters is not None,
            "contact_force_world": dataset_obj.contact_force_world is not None,
        },
    }

    save_vision_residual_policy(
        output,
        input_mode="image_only",
        encoder=backbone,
        head=head,
        head_type=head_type,
        head_hidden_dims=resolved_hidden_dims,
        head_dropout=head_dropout if head_type == MLP_V2_HEAD_TYPE else 0.0,
        head_activation=head_summary["head_activation"],
        output_bounding=head_summary["output_bounding"],
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        base_spec=base_spec,
        metadata=train_metadata,
    )

    policy = load_image_only_residual_bc_policy(output)
    print(f"[{scene_name}-train] verifying_reload output={output}", flush=True)
    if policy.head_type != head_type:
        raise RuntimeError(f"Reloaded policy head_type {policy.head_type!r} does not match saved head_type {head_type!r}.")
    if tuple(policy.hidden_dims) != tuple(resolved_hidden_dims):
        raise RuntimeError(
            f"Reloaded policy hidden_dims {tuple(policy.hidden_dims)!r} do not match saved hidden_dims {tuple(resolved_hidden_dims)!r}."
        )
    sample_idx = int(train_idx[0])
    sample_image = dataset_obj.rgb_images[sample_idx]
    matrix, theta, theta_delta, bounded = policy.predict(task_state=None, rgb_images=sample_image)
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(theta)) and np.all(np.isfinite(theta_delta)) and np.all(np.isfinite(bounded))):
        raise RuntimeError("Loaded frozen DINOv3 policy produced non-finite predictions.")

    result = {
        "dataset": str(dataset),
        "output": str(output),
        "split_file": str(split_file),
        "train_seed": int(train_seed),
        "samples": int(dataset_obj.sample_count),
        "train_samples": int(train_idx.size),
        "val_samples": int(val_idx.size),
        "train_episodes": int(train_episode_ids.size),
        "val_episodes": int(val_episode_ids.size),
        "feature_dim": int(train_features.shape[1]),
        "target_dim": int(y_train_raw.shape[1]),
        "train_loss": float(head_summary["train_loss"]),
        "val_loss": float(head_summary["val_loss"]),
        "best_epoch": int(head_summary["best_epoch"]),
        "best_train_loss": float(head_summary["best_train_loss"]),
        "best_val_loss": float(head_summary["best_val_loss"]),
        "early_stop_reason": head_summary["early_stop_reason"],
        "per_dim_val_mae": head_summary["val_per_dim_mae"],
        "per_dim_sign_accuracy": head_summary["val_per_dim_sign_accuracy"],
        "label_target_mean": head_summary["val_label_target_mean"],
        "label_target_std": head_summary["val_label_target_std"],
        "prediction_mean": head_summary["val_prediction_mean"],
        "prediction_std": head_summary["val_prediction_std"],
        "prediction_saturation_rate": head_summary["prediction_saturation_rate"],
        "epochs_ran": int(len(history)),
        "backbone_type": DEFAULT_BACKBONE_TYPE,
        "backbone_frozen": True,
        "head_type": head_type,
        "head_hidden_dims": [int(value) for value in resolved_hidden_dims],
        "head_dropout": float(head_dropout if head_type == MLP_V2_HEAD_TYPE else 0.0),
        "augmentation_mode": augmentation_mode,
        "training_summary": head_summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[{scene_name}-train] done output={output}", flush=True)
    if not np.isfinite(history[-1][1]):
        raise RuntimeError("Training loss is not finite.")
    if history[-1][1] > history[0][1]:
        print(
            "WARNING: training loss did not decrease; keeping the saved policy artifact because paired evaluation is the primary metric.",
            file=sys.stderr,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a frozen DINOv3 image-only residual BC policy for a single scene.")
    parser.add_argument("--dataset", type=Path, required=True, help="Scene root or eligible_residual_bc.npz path.")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Legacy alias for --train-seed.")
    parser.add_argument("--head-type", choices=(LINEAR_OR_MLP_V1_HEAD_TYPE, MLP_V2_HEAD_TYPE), default=LINEAR_OR_MLP_V1_HEAD_TYPE)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--head-hidden-dim", type=int, default=32)
    parser.add_argument("--head-hidden-dims", type=parse_hidden_dims, default=None)
    parser.add_argument("--head-dropout", type=float, default=DEFAULT_HEAD_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--dinov3-entrypoint", type=str, default=DEFAULT_DINOV3_ENTRYPOINT)
    parser.add_argument("--controllers-yaml", type=Path, default=DEFAULT_TRACK_A_CONTROLLERS_YAML)
    parser.add_argument("--controller-id", type=str, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--augmentation-mode", choices=(NO_AUGMENTATION_MODE, LIGHT_CONSERVATIVE_AUGMENTATION_MODE), default=NO_AUGMENTATION_MODE)
    args = parser.parse_args(argv)
    train_seed = args.train_seed if args.train_seed is not None else (args.seed if args.seed is not None else 0)
    output = args.output or _default_output_for_dataset(
        args.dataset,
        split_file=args.split_file,
        augmentation_mode=args.augmentation_mode,
        head_type=args.head_type,
    )
    resolved_hidden_dims = tuple(args.head_hidden_dims) if args.head_hidden_dims is not None else (
        DEFAULT_HEAD_HIDDEN_DIMS if args.head_type == MLP_V2_HEAD_TYPE else (args.head_hidden_dim,)
    )
    train_frozen_dinov3_residual_bc(
        args.dataset,
        output,
        split_file=args.split_file,
        train_seed=train_seed,
        dinov3_repo=args.dinov3_repo,
        dinov3_checkpoint=args.dinov3_checkpoint,
        dinov3_entrypoint=args.dinov3_entrypoint,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        head_hidden_dims=resolved_hidden_dims,
        head_dropout=args.head_dropout,
        epochs=args.epochs,
        validation_patience=args.validation_patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        controllers_yaml=args.controllers_yaml,
        controller_id=args.controller_id,
        feature_batch_size=args.feature_batch_size,
        augmentation_mode=args.augmentation_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
