from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _sigma7_residual_pipeline_common import (
    COLLECTION_MODE,
    DEFAULT_MUJOCO_ROOT,
    DEFAULT_PIPELINE_ROOT,
    scene_dataset_6d_root,
    scene_models_root,
    require_safe_segment,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve scene-level aggregated collection dataset paths, then delegate training to the reference frozen DINOv3 6D entrypoint."
    )
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--dataset", type=Path, default=None, help="Optional explicit dataset root or eligible_residual_bc.npz path.")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Legacy alias for --train-seed.")
    parser.add_argument("--head-type", choices=("linear_or_mlp_v1", "mlp_v2"), default="linear_or_mlp_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--head-hidden-dim", type=int, default=32)
    parser.add_argument("--head-hidden-dims", type=str, default=None)
    parser.add_argument("--head-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dinov3-repo", type=Path, default=None)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=None)
    parser.add_argument("--dinov3-entrypoint", type=str, default=None)
    parser.add_argument(
        "--controllers-yaml",
        type=Path,
        default=DEFAULT_MUJOCO_ROOT / "configs" / "track_a_controllers.yaml",
    )
    parser.add_argument("--controller-id", type=str, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--augmentation-mode", choices=("none", "light_conservative"), default="none")
    args = parser.parse_args(argv)

    scene = require_safe_segment(args.scene, name="scene")
    mode = COLLECTION_MODE

    dataset = args.dataset or scene_dataset_6d_root(args.output_root, scene)
    split_file = args.split_file or (Path(dataset) / "frozen_train_val_split.json")
    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        scene_models_root(args.output_root, scene).mkdir(parents=True, exist_ok=True)

    script_path = DEFAULT_MUJOCO_ROOT / "scripts" / "train_frozen_dinov3_residual_bc_6d.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset",
        str(dataset),
        "--split-file",
        str(split_file),
        "--head-type",
        args.head_type,
        "--epochs",
        str(args.epochs),
        "--validation-patience",
        str(args.validation_patience),
        "--lr",
        str(args.lr),
        "--head-hidden-dim",
        str(args.head_hidden_dim),
        "--head-dropout",
        str(args.head_dropout),
        "--weight-decay",
        str(args.weight_decay),
        "--controllers-yaml",
        str(args.controllers_yaml),
        "--feature-batch-size",
        str(args.feature_batch_size),
        "--augmentation-mode",
        args.augmentation_mode,
    ]
    if args.train_seed is not None:
        cmd.extend(["--train-seed", str(args.train_seed)])
    elif args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if output is not None:
        cmd.extend(["--output", str(output)])
    if args.head_hidden_dims is not None:
        cmd.extend(["--head-hidden-dims", args.head_hidden_dims])
    if args.dinov3_repo is not None:
        cmd.extend(["--dinov3-repo", str(args.dinov3_repo)])
    if args.dinov3_checkpoint is not None:
        cmd.extend(["--dinov3-checkpoint", str(args.dinov3_checkpoint)])
    if args.dinov3_entrypoint is not None:
        cmd.extend(["--dinov3-entrypoint", args.dinov3_entrypoint])
    if args.controller_id is not None:
        cmd.extend(["--controller-id", args.controller_id])

    subprocess.run(cmd, check=True)
    print("")
    print("training command finished")
    print(f"mode: {mode}")
    print(f"scene: {scene}")
    print(f"dataset: {dataset}")
    print(f"split_file: {split_file}")
    if output is not None:
        print(f"output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
