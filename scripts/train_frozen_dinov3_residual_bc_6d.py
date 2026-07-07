from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from stiffness_copilot_mujoco.learning.vision_residual_dataset import load_vision_residual_dataset  # noqa: E402
from train_frozen_dinov3_residual_bc import (  # noqa: E402
    DEFAULT_DINOV3_CHECKPOINT,
    DEFAULT_DINOV3_ENTRYPOINT,
    DEFAULT_DINOV3_REPO,
    DEFAULT_OUTPUT_ROOT,
    LINEAR_OR_MLP_V1_HEAD_TYPE,
    MLP_V2_HEAD_TYPE,
    NO_AUGMENTATION_MODE,
    LIGHT_CONSERVATIVE_AUGMENTATION_MODE,
    parse_hidden_dims,
    train_frozen_dinov3_residual_bc as _train_frozen_dinov3_residual_bc,
)


def _default_output_for_dataset_6d(dataset: Path, *, split_file: Path, augmentation_mode: str, head_type: str) -> Path:
    try:
        candidate = dataset / "eligible_residual_bc.npz" if dataset.is_dir() else dataset
        with np.load(candidate, allow_pickle=False) as data:
            metadata_dict = json.loads(str(data["metadata"]))
        setting_id = metadata_dict.get("setting_id", metadata_dict.get("scene", "unknown"))
        episodes = metadata_dict.get("num_episodes", "unknown")
        profile_name = metadata_dict.get("collection_controller_id") or metadata_dict.get("base_profile") or metadata_dict.get("profile_name")
        suffix = f"_{profile_name}" if profile_name else ""
        split_stem = split_file.stem
        head_suffix = "_head_mlp_v2" if head_type == MLP_V2_HEAD_TYPE else ""
        if augmentation_mode == LIGHT_CONSERVATIVE_AUGMENTATION_MODE:
            return DEFAULT_OUTPUT_ROOT / f"{setting_id}{suffix}_{episodes}ep_{split_stem}_dinov3_light_aug_residual_bc_policy_6d{head_suffix}.npz"
        return DEFAULT_OUTPUT_ROOT / f"{setting_id}{suffix}_{episodes}ep_{split_stem}_dinov3_residual_bc_policy_6d{head_suffix}.npz"
    except Exception:
        if augmentation_mode == LIGHT_CONSERVATIVE_AUGMENTATION_MODE:
            return DEFAULT_OUTPUT_ROOT / f"dinov3_light_aug_residual_bc_policy_6d{('_head_mlp_v2' if head_type == MLP_V2_HEAD_TYPE else '')}.npz"
        return DEFAULT_OUTPUT_ROOT / f"dinov3_residual_bc_policy_6d{('_head_mlp_v2' if head_type == MLP_V2_HEAD_TYPE else '')}.npz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a frozen DINOv3 image-only residual BC policy with fixed 6D output.")
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
    parser.add_argument("--head-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument("--dinov3-entrypoint", type=str, default=DEFAULT_DINOV3_ENTRYPOINT)
    parser.add_argument("--controllers-yaml", type=Path, default=ROOT / "configs" / "track_a_controllers.yaml")
    parser.add_argument("--controller-id", type=str, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--augmentation-mode", choices=(NO_AUGMENTATION_MODE, LIGHT_CONSERVATIVE_AUGMENTATION_MODE), default=NO_AUGMENTATION_MODE)
    args = parser.parse_args(argv)

    train_seed = args.train_seed if args.train_seed is not None else (args.seed if args.seed is not None else 0)
    output = args.output or _default_output_for_dataset_6d(
        args.dataset,
        split_file=args.split_file,
        augmentation_mode=args.augmentation_mode,
        head_type=args.head_type,
    )

    dataset_obj = load_vision_residual_dataset(args.dataset, input_mode="image_only", require_native_renderer=True)
    if dataset_obj.residual_group_target.shape[1] != 6:
        raise ValueError(
            f"6D training requires residual_group_target shape [N, 6], observed {dataset_obj.residual_group_target.shape}."
        )

    _train_frozen_dinov3_residual_bc(
        args.dataset,
        output,
        split_file=args.split_file,
        train_seed=train_seed,
        dinov3_repo=args.dinov3_repo,
        dinov3_checkpoint=args.dinov3_checkpoint,
        dinov3_entrypoint=args.dinov3_entrypoint,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        head_hidden_dims=tuple(args.head_hidden_dims) if args.head_hidden_dims is not None else None,
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
