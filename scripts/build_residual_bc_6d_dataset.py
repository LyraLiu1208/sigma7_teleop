from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec, PARAM_NAMES  # noqa: E402
from stiffness_copilot_mujoco.learning.residual_dataset import validate_residual_dataset  # noqa: E402
from stiffness_copilot_mujoco.learning.stiffness_labels import matrix_to_cholesky_params  # noqa: E402


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _load_dataset_arrays(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return arrays, metadata


def _make_full_6d_base_spec(existing: BaseStiffnessSpec) -> BaseStiffnessSpec:
    return BaseStiffnessSpec.from_matrix(
        existing.base_matrix,
        active_groups=((0,), (1,), (2,), (3,), (4,), (5,)),
        active_group_names=tuple(PARAM_NAMES),
        residual_bound=float(existing.residual_bounds.max()),
    )


def _full_axis_contact_gated_projection(
    selected_scores: np.ndarray,
    force_magnitude: np.ndarray,
    *,
    residual_bound: float,
    contact_gate_low: float,
    contact_gate_high: float,
    neutral_contact_threshold: float,
    min_calibration_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.asarray(selected_scores, dtype=float)
    force = np.asarray(force_magnitude, dtype=float).reshape(-1)
    if scores.ndim != 2 or scores.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"selected_scores must have shape [N, {len(PARAM_NAMES)}], observed {scores.shape}."
        )
    if force.shape != (scores.shape[0],):
        raise ValueError(f"force_magnitude must have shape ({scores.shape[0]},), observed {force.shape}.")
    if residual_bound <= 0.0:
        raise ValueError("residual_bound must be positive.")
    if contact_gate_high <= contact_gate_low:
        raise ValueError("contact_gate_high must be larger than contact_gate_low.")

    gate = np.clip((force - contact_gate_low) / (contact_gate_high - contact_gate_low), 0.0, 1.0)

    neutral_mask = force < neutral_contact_threshold
    neutral_source = "weak_contact_median"
    if int(np.count_nonzero(neutral_mask)) < min_calibration_samples:
        neutral_mask = force < contact_gate_high
        neutral_source = "contact_gate_high_median"
    if int(np.count_nonzero(neutral_mask)) < min_calibration_samples:
        neutral_mask = np.ones(force.shape[0], dtype=bool)
        neutral_source = "all_samples_median"

    scale_mask = force >= neutral_contact_threshold
    scale_source = "contact_abs_deviation_p95"
    if int(np.count_nonzero(scale_mask)) < min_calibration_samples:
        scale_mask = np.ones(force.shape[0], dtype=bool)
        scale_source = "all_samples_abs_deviation_p95"

    neutral_center = np.median(scores[neutral_mask], axis=0)
    scale = np.percentile(np.abs(scores[scale_mask] - neutral_center[None, :]), 95.0, axis=0)
    scale = np.asarray(scale, dtype=float)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)

    raw_residual = residual_bound * np.clip((scores - neutral_center[None, :]) / scale[None, :], -1.0, 1.0)
    residual = np.clip(gate[:, None] * raw_residual, -residual_bound, residual_bound)
    metadata = {
        "contact_gate_applied": True,
        "free_space_residual_policy": "baseline_anchor",
        "contact_gate_low": float(contact_gate_low),
        "contact_gate_high": float(contact_gate_high),
        "neutral_contact_threshold": float(neutral_contact_threshold),
        "min_calibration_samples": int(min_calibration_samples),
        "contact_gate_mean": float(np.mean(gate)) if gate.size else None,
        "contact_gate_zero_fraction": float(np.mean(gate <= 1e-12)) if gate.size else None,
        "contact_gate_one_fraction": float(np.mean(gate >= 1.0 - 1e-12)) if gate.size else None,
        "neutral_center_source": neutral_source,
        "neutral_center_values": [float(value) for value in np.asarray(neutral_center, dtype=float).reshape(-1)],
        "neutral_mask_count": int(np.count_nonzero(neutral_mask)),
        "robust_scale_source": scale_source,
        "robust_scale_values": [float(value) for value in np.asarray(scale, dtype=float).reshape(-1)],
        "scale_mask_count": int(np.count_nonzero(scale_mask)),
        "residual_group_target_neutral": 0.0,
        "residual_group_target_space": "full_spd_cholesky_delta_6d",
        "selected_score_names": list(PARAM_NAMES),
        "full_axis_residual_projection": True,
    }
    return residual, metadata


def _transform_dataset(
    input_npz: Path,
    output_npz: Path,
    *,
    source_metadata: dict[str, Any],
    source_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    if "residual_theta_target" not in source_arrays:
        raise KeyError(f"{input_npz} is missing residual_theta_target.")
    if "residual_group_target" not in source_arrays:
        raise KeyError(f"{input_npz} is missing residual_group_target.")
    if "stiffness_cholesky_target" not in source_arrays:
        raise KeyError(f"{input_npz} is missing stiffness_cholesky_target.")
    if "base_stiffness_spec" in source_metadata:
        source_base_spec = BaseStiffnessSpec.from_metadata(source_metadata["base_stiffness_spec"])
    else:
        source_base_spec = BaseStiffnessSpec.from_matrix(
            np.asarray(source_metadata["collection_stiffness_matrix"], dtype=float),
            active_groups=((0, 1),),
            active_group_names=("alpha_lateral_shared",),
            residual_bound=float(source_metadata.get("residual_bound", 0.35)),
        )

    full_base_spec = _make_full_6d_base_spec(source_base_spec)
    full_target_source = np.asarray(source_arrays["stiffness_cholesky_target"], dtype=float)
    if full_target_source.ndim != 2 or full_target_source.shape[1] != 6:
        raise ValueError(
            f"{input_npz} stiffness_cholesky_target must have shape [N, 6], observed {full_target_source.shape}."
        )
    contact_force_world = np.asarray(source_arrays["contact_force_world"], dtype=float)
    force_magnitude = np.linalg.norm(contact_force_world, axis=1)
    full_target, projection_metadata = _full_axis_contact_gated_projection(
        full_target_source,
        force_magnitude,
        residual_bound=float(full_base_spec.residual_bounds.max()),
        contact_gate_low=float(source_metadata.get("contact_gate_low", 1.0)),
        contact_gate_high=float(source_metadata.get("contact_gate_high", 10.0)),
        neutral_contact_threshold=float(source_metadata.get("neutral_contact_threshold", 10.0)),
        min_calibration_samples=int(source_metadata.get("min_calibration_samples", 8)),
    )

    legacy_group_target = np.asarray(source_arrays["residual_group_target"], dtype=float)
    sample_count = int(full_target.shape[0])
    output_arrays: dict[str, np.ndarray] = {}
    for key, value in source_arrays.items():
        if key in {"residual_group_target", "residual_theta_target"}:
            continue
        output_arrays[key] = np.asarray(value)
    output_arrays["residual_group_target"] = full_target.astype(np.float32)
    output_arrays["residual_theta_target"] = full_target.astype(np.float32)
    output_arrays["legacy_residual_group_target"] = legacy_group_target.astype(np.float32)

    physical_matrix = np.stack(
        [full_base_spec.matrix_from_group_delta(delta, clip=True)[0] for delta in full_target],
        axis=0,
    )
    physical_cholesky = np.stack([matrix_to_cholesky_params(item) for item in physical_matrix], axis=0)
    physical_eigenvalues = np.linalg.eigvalsh(physical_matrix)
    output_arrays["physical_stiffness_matrix_target"] = physical_matrix.astype(np.float32)
    output_arrays["physical_stiffness_cholesky_target"] = physical_cholesky.astype(np.float32)
    output_arrays["physical_stiffness_eigenvalues"] = physical_eigenvalues.astype(np.float32)

    new_metadata = dict(source_metadata)
    new_metadata.update(
        {
            "dataset_path": str(output_npz),
            "target_contract": "full_spd_cholesky_residual_6d_full_axis_v1",
            "primary_supervision_target": "residual_group_target",
            "primary_target_space": "full_spd_cholesky_residual_6d",
            "residual_group_target_role": "primary_supervision_target",
            "residual_group_target_space": "full_spd_cholesky_delta_6d",
            "residual_group_target_neutral": 0.0,
            "residual_group_target_dim": 6,
            "residual_theta_target_role": "primary_6d_residual_diagnostic",
            "legacy_residual_group_target_role": "scene_specific_compatibility_only",
            "legacy_residual_group_target_dim": int(legacy_group_target.shape[1]) if legacy_group_target.ndim == 2 else None,
            "output_space": "full_spd_cholesky_residual_6d",
            "output_dim": 6,
            "residual_dim": 6,
            "residual_parameterization": "full_spd_cholesky_delta_6d",
            "residual_affects": list(PARAM_NAMES),
            "residual_unaffected": [],
            "base_stiffness_spec": full_base_spec.to_metadata(),
            "label_projection": source_metadata.get("label_projection", "residual_first_contact_gated_centered_v2"),
            "label_projection_description": "contact-gated full-axis 6D Cholesky residual adaptation",
            "label_k_range_role": "deprecated_legacy_compatibility_only",
            "legacy_label_k_min": source_metadata.get("label_k_min"),
            "legacy_label_k_max": source_metadata.get("label_k_max"),
            "legacy_residual_group_target_shape": list(legacy_group_target.shape),
            "full_6d_target_source": "stiffness_cholesky_target",
            "full_6d_target_sample_count": sample_count,
            "full_axis_residual_projection": True,
            "full_axis_residual_projection_metadata": projection_metadata,
            "physical_stiffness_matrix_target_role": "diagnostic_reconstruction_from_full_axis_residual",
            "physical_stiffness_cholesky_target_role": "diagnostic_reconstruction_from_full_axis_residual",
            "physical_stiffness_eigenvalues_role": "diagnostic_reconstruction_from_full_axis_residual",
        }
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, metadata=json.dumps(new_metadata, sort_keys=True), **output_arrays)
    validate_residual_dataset(output_npz)
    return new_metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a scene residual dataset into a fixed 6D residual target contract."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Scene root containing eligible_residual_bc.npz.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output scene root for the 6D dataset.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    input_root = args.input_root
    output_root = args.output_root
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_npz = input_root / "eligible_residual_bc.npz"
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing eligible dataset: {input_npz}")

    source_arrays, source_metadata = _load_dataset_arrays(input_npz)
    output_npz = output_root / "eligible_residual_bc.npz"
    new_metadata = _transform_dataset(
        input_npz,
        output_npz,
        source_metadata=source_metadata,
        source_arrays=source_arrays,
    )

    for filename in (
        "raw_collection.npz",
        "episodes.csv",
        "episode_specs.jsonl",
        "frozen_paired_episode_specs.jsonl",
        "frozen_train_val_split.json",
        "collection_summary.json",
        "collection_metadata.json",
        "selection_summary.json",
        "trajectory_family_summary.json",
    ):
        _copy_if_exists(input_root / filename, output_root / filename)

    collection_metadata_path = output_root / "collection_metadata.json"
    if collection_metadata_path.exists():
        collection_metadata = _load_json(collection_metadata_path)
        collection_metadata.update(
            {
                "dataset_path": str(output_npz),
                "target_contract": "full_spd_cholesky_residual_6d_v1",
                "output_space": "full_spd_cholesky_residual_6d",
                "output_dim": 6,
                "residual_dim": 6,
                "residual_parameterization": "full_spd_cholesky_delta_6d",
                "base_stiffness_spec": new_metadata.get("base_stiffness_spec"),
            }
        )
        _write_json(collection_metadata_path, collection_metadata)

    collection_summary_path = output_root / "collection_summary.json"
    if collection_summary_path.exists():
        collection_summary = _load_json(collection_summary_path)
        collection_summary.update(
            {
                "target_contract": "full_spd_cholesky_residual_6d_v1",
                "output_dataset": str(output_npz),
                "selection_mode": "fixed_6d_residual_contract",
                "metadata": {
                    "num_episodes": int(new_metadata.get("num_episodes", 0)),
                    "num_samples": int(new_metadata.get("num_samples", 0)),
                    "target_contract": "full_spd_cholesky_residual_6d_v1",
                },
            }
        )
        _write_json(collection_summary_path, collection_summary)

    selection_summary_path = output_root / "selection_summary.json"
    if selection_summary_path.exists():
        selection_summary = _load_json(selection_summary_path)
        selection_summary.update(
            {
                "target_contract": "full_spd_cholesky_residual_6d_v1",
                "output_dataset": str(output_npz),
            }
        )
        _write_json(selection_summary_path, selection_summary)

    split_path = output_root / "frozen_train_val_split.json"
    if split_path.exists():
        split_payload = _load_json(split_path)
        split_payload["dataset_path"] = str(output_npz)
        if isinstance(split_payload.get("metadata"), dict):
            split_payload["metadata"]["dataset_path"] = str(output_npz)
            split_payload["metadata"]["target_contract"] = "full_spd_cholesky_residual_6d_v1"
        _write_json(split_path, split_payload)

    print(
        json.dumps(
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "output_dataset": str(output_npz),
                "target_contract": "full_spd_cholesky_residual_6d_v1",
                "output_dim": 6,
                "residual_dim": 6,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
