from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from stiffness_copilot_mujoco.learning.residual_dataset import validate_residual_dataset
from stiffness_copilot_mujoco.learning.residual_label_projection import is_residual_first_projection
from stiffness_copilot_mujoco.learning.stiffness_labels import cholesky_params_to_matrix, matrix_to_cholesky_params


DEFAULT_NEAR_ZERO_FRACTION = 0.10
DEFAULT_HIGH_FORCE_PERCENTILE = 95.0
DEFAULT_RESIDUAL_SATURATION_FRACTION = 0.99


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    path: Path
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]], fields: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _safe_stats(values: np.ndarray | list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "nonfinite_count": int(array.size),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _safe_eigvalsh(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=float)
    return np.linalg.eigvalsh(0.5 * (matrices + np.swapaxes(matrices, -1, -2)))


def _component_distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return {"value": _safe_stats(array)}
    if array.ndim == 2:
        result = {f"dim_{idx}": _safe_stats(array[:, idx]) for idx in range(array.shape[1])}
        result["norm"] = _safe_stats(np.linalg.norm(array, axis=1))
        return result
    return {"all_values": _safe_stats(array.reshape(-1))}


def _align_samples(legacy: Mapping[str, np.ndarray], new: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    legacy_episode = np.asarray(legacy["episode_id"], dtype=int).reshape(-1)
    new_episode = np.asarray(new["episode_id"], dtype=int).reshape(-1)
    legacy_step = np.asarray(legacy["sample_step"], dtype=int).reshape(-1) if "sample_step" in legacy else np.arange(legacy_episode.size, dtype=int)
    new_step = np.asarray(new["sample_step"], dtype=int).reshape(-1) if "sample_step" in new else np.arange(new_episode.size, dtype=int)

    legacy_keys = {(int(ep), int(step)): idx for idx, (ep, step) in enumerate(zip(legacy_episode, legacy_step, strict=True))}
    new_keys = {(int(ep), int(step)): idx for idx, (ep, step) in enumerate(zip(new_episode, new_step, strict=True))}
    common = [key for key in legacy_keys if key in new_keys]
    if not common:
        raise ValueError("No overlapping sample keys between legacy and new datasets.")
    common.sort()
    legacy_indices = np.asarray([legacy_keys[key] for key in common], dtype=int)
    new_indices = np.asarray([new_keys[key] for key in common], dtype=int)
    return legacy_indices, new_indices


def _flatten_mean(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    array = np.asarray(values, dtype=float)
    if mask is not None:
        array = array[mask]
    if array.size == 0:
        return None
    flat = array.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _flatten_stats(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if mask is not None:
        array = array[mask]
    return _safe_stats(array.reshape(-1))


def _scene_from_path(path: Path, metadata: Mapping[str, Any]) -> str:
    scene = metadata.get("scene")
    if scene:
        return str(scene)
    return path.parent.name if path.suffix == ".npz" else path.name


def load_dataset(path: Path) -> DatasetRecord:
    candidate = path
    if path.is_dir():
        if (path / "eligible_residual_bc.npz").exists():
            candidate = path / "eligible_residual_bc.npz"
        else:
            npz_files = sorted(path.glob("*.npz"))
            if len(npz_files) == 1:
                candidate = npz_files[0]
            else:
                raise FileNotFoundError(f"Could not locate a dataset npz under {path}.")
    validate_residual_dataset(candidate)
    with np.load(candidate, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    return DatasetRecord(
        name=_scene_from_path(candidate, metadata),
        path=candidate,
        arrays=arrays,
        metadata=metadata,
    )


def load_scene_records(root: Path) -> list[DatasetRecord]:
    if root.is_file():
        return [load_dataset(root)]

    if (root / "eligible_residual_bc.npz").exists():
        return [load_dataset(root / "eligible_residual_bc.npz")]

    root_metadata_path = root / "metadata.json"
    if root_metadata_path.exists():
        root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
        scenes = root_metadata.get("scenes", {})
        records: list[DatasetRecord] = []
        for scene_name, scene_metadata in scenes.items():
            scene_path = Path(scene_metadata.get("eligible_dataset") or root / scene_name / "eligible_residual_bc.npz")
            if not scene_path.exists():
                candidate = root / scene_path
                if candidate.exists():
                    scene_path = candidate
                else:
                    scene_path = root / scene_name / "eligible_residual_bc.npz"
            records.append(load_dataset(scene_path))
        if records:
            return records

    records = []
    for scene_dir in sorted([child for child in root.iterdir() if child.is_dir()]):
        candidate = scene_dir / "eligible_residual_bc.npz"
        if candidate.exists():
            records.append(load_dataset(candidate))
    if records:
        return records

    raise FileNotFoundError(f"Could not find residual-first datasets under {root}.")


def compute_scene_audit(record: DatasetRecord) -> dict[str, Any]:
    arrays = record.arrays
    metadata = record.metadata
    residual = np.asarray(arrays["residual_group_target"], dtype=float)
    physical = np.asarray(arrays["physical_stiffness_matrix_target"], dtype=float)
    normalized = np.asarray(arrays["stiffness_matrix_target"], dtype=float)
    force_norm = np.linalg.norm(np.asarray(arrays["contact_force_world"], dtype=float), axis=1)
    contact_mask = np.asarray(arrays["contact_state"], dtype=float)[:, 0] > 0.5
    contact_residual = residual[contact_mask]
    noncontact_residual = residual[~contact_mask]
    high_force_threshold = float(np.percentile(force_norm[np.isfinite(force_norm)], DEFAULT_HIGH_FORCE_PERCENTILE)) if force_norm.size else 0.0
    high_force_mask = force_norm >= high_force_threshold
    high_force_residual = residual[high_force_mask]
    physical_eig = _safe_eigvalsh(physical)
    residual_bound = float(metadata.get("residual_bound", np.max(np.abs(residual)) if residual.size else 0.0))
    near_zero_threshold = DEFAULT_NEAR_ZERO_FRACTION * residual_bound
    flat_residual = residual.reshape(-1)
    flat_residual = flat_residual[np.isfinite(flat_residual)]
    positive_fraction = float(np.mean(flat_residual > 0.0)) if flat_residual.size else None
    negative_fraction = float(np.mean(flat_residual < 0.0)) if flat_residual.size else None
    near_zero_fraction = float(np.mean(np.abs(flat_residual) <= near_zero_threshold)) if flat_residual.size else None
    saturation_rate = float(np.mean(np.abs(flat_residual) >= DEFAULT_RESIDUAL_SATURATION_FRACTION * residual_bound)) if flat_residual.size else None
    spd_valid_fraction = float(np.mean(np.all(physical_eig > 0.0, axis=1))) if physical_eig.size else None
    residual_dim = int(residual.shape[1]) if residual.ndim == 2 else 0
    normalized_xy = np.diagonal(normalized, axis1=1, axis2=2)[:, :2] if normalized.ndim == 3 and normalized.shape[-1] >= 2 else np.zeros((normalized.shape[0], 0))
    diagnostic_xy = np.diagonal(physical, axis1=1, axis2=2)[:, :2] if physical.ndim == 3 and physical.shape[-1] >= 2 else np.zeros((physical.shape[0], 0))

    row = {
        "scene": record.name,
        "dataset_path": str(record.path),
        "renderer_mode": metadata.get("renderer_mode"),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        "training_data_valid": metadata.get("training_data_valid"),
        "training_data_valid_reason": metadata.get("training_data_valid_reason"),
        "force_sectorization_used": metadata.get("force_sectorization_used"),
        "sectorization": metadata.get("sectorization"),
        "sector_polar_bins": metadata.get("sector_polar_bins"),
        "sector_azimuth_bins": metadata.get("sector_azimuth_bins"),
        "num_force_sectors": metadata.get("num_force_sectors"),
        "sector_magnitude_percentile": metadata.get("sector_magnitude_percentile"),
        "label_projection": metadata.get("label_projection"),
        "residual_bound": metadata.get("residual_bound"),
        "baseline_k": metadata.get("baseline_k"),
        "diagnostic_k_min": metadata.get("diagnostic_k_min"),
        "diagnostic_k_max": metadata.get("diagnostic_k_max"),
        "residual_dim": residual_dim,
        "label_projection_matches_residual_first": is_residual_first_projection(metadata.get("label_projection")),
        "primary_supervision_target": metadata.get("primary_supervision_target"),
        "primary_target_space": metadata.get("primary_target_space"),
        "residual_to_physical_semantics": metadata.get("residual_to_physical_semantics"),
        "physical_stiffness_matrix_target_role": metadata.get("physical_stiffness_matrix_target_role"),
        "label_k_range_role": metadata.get("label_k_range_role"),
        "label_k_min": metadata.get("label_k_min"),
        "label_k_max": metadata.get("label_k_max"),
        "label_k_min_deprecated": metadata.get("label_k_min_deprecated"),
        "label_k_max_deprecated": metadata.get("label_k_max_deprecated"),
        "contact_gate_applied": metadata.get("contact_gate_applied"),
        "free_space_residual_policy": metadata.get("free_space_residual_policy"),
        "contact_gate_low": metadata.get("contact_gate_low"),
        "contact_gate_high": metadata.get("contact_gate_high"),
        "contact_gate_mean": metadata.get("contact_gate_mean"),
        "contact_gate_zero_fraction": metadata.get("contact_gate_zero_fraction"),
        "contact_gate_one_fraction": metadata.get("contact_gate_one_fraction"),
        "neutral_center_source": metadata.get("neutral_center_source"),
        "neutral_center_values": metadata.get("neutral_center_values"),
        "neutral_mask_count": metadata.get("neutral_mask_count"),
        "neutral_contact_threshold": metadata.get("neutral_contact_threshold"),
        "min_calibration_samples": metadata.get("min_calibration_samples"),
        "robust_scale_source": metadata.get("robust_scale_source"),
        "robust_scale_values": metadata.get("robust_scale_values"),
        "scale_mask_count": metadata.get("scale_mask_count"),
        "residual_group_target_neutral": metadata.get("residual_group_target_neutral"),
        "selected_score_names": metadata.get("selected_score_names"),
        "selected_K_norm_range": metadata.get("selected_K_norm_range"),
        "selected_K_norm_neutral": metadata.get("selected_K_norm_neutral"),
        "residual_min": float(np.min(flat_residual)) if flat_residual.size else None,
        "residual_max": float(np.max(flat_residual)) if flat_residual.size else None,
        "residual_mean": float(np.mean(flat_residual)) if flat_residual.size else None,
        "residual_std": float(np.std(flat_residual)) if flat_residual.size else None,
        "residual_saturation_rate": saturation_rate,
        "positive_residual_fraction": positive_fraction,
        "negative_residual_fraction": negative_fraction,
        "near_zero_residual_fraction": near_zero_fraction,
        "near_zero_threshold": near_zero_threshold,
        "diagnostic_physical_eig_min": float(np.min(physical_eig)) if physical_eig.size else None,
        "diagnostic_physical_eig_max": float(np.max(physical_eig)) if physical_eig.size else None,
        "diagnostic_physical_eig_mean": float(np.mean(physical_eig)) if physical_eig.size else None,
        "spd_valid_fraction": spd_valid_fraction,
        "contact_residual_mean": _flatten_mean(residual, contact_mask),
        "noncontact_residual_mean": _flatten_mean(residual, ~contact_mask),
        "high_force_residual_mean": _flatten_mean(residual, high_force_mask),
        "high_force_threshold": high_force_threshold,
        "residual_distribution": {
            "flattened": _safe_stats(flat_residual),
            "per_dim": {f"dim_{idx}": _safe_stats(residual[:, idx]) for idx in range(residual.shape[1])},
        },
        "physical_diagnostic_distribution": {
            "eigenvalues": _component_distribution(physical_eig),
            "normalized_xy": _component_distribution(normalized_xy),
            "diagnostic_xy": _component_distribution(diagnostic_xy),
        },
        "semantic_checks": {
            "residual_within_bounds": bool(np.all(np.abs(flat_residual) <= residual_bound + 1e-9)) if flat_residual.size else True,
            "positive_residuals_present": bool(np.any(flat_residual > 0.0)),
            "negative_residuals_present": bool(np.any(flat_residual < 0.0)),
            "residual_first_semantics": is_residual_first_projection(metadata.get("label_projection")),
            "output_dims_ok": residual_dim in {1, 2, 3},
            "diagnostic_eig_can_exceed_600": bool(np.max(physical_eig) > 600.0) if physical_eig.size else False,
            "legacy_full_range_primary_path_used": metadata.get("primary_target_space") != "baseline_relative_residual"
            and metadata.get("label_k_range_role") != "deprecated_legacy_compatibility_only",
        },
    }

    if residual_dim == 3:
        l21 = residual[:, 2]
        row["star_specific"] = {
            "l21_min": float(np.min(l21)),
            "l21_max": float(np.max(l21)),
            "l21_mean": float(np.mean(l21)),
            "l21_std": float(np.std(l21)),
            "l21_saturation_rate": float(np.mean(np.abs(l21) >= DEFAULT_RESIDUAL_SATURATION_FRACTION * residual_bound)),
            "K_robot_norm_xy_distribution": _component_distribution(normalized_xy),
            "diagnostic_K_xy_distribution": _component_distribution(diagnostic_xy),
        }
    else:
        row["star_specific"] = None
    return row


def compute_legacy_comparison(legacy: DatasetRecord, new: DatasetRecord) -> dict[str, Any]:
    legacy_indices, new_indices = _align_samples(legacy.arrays, new.arrays)
    legacy_residual = np.asarray(legacy.arrays["residual_group_target"], dtype=float)[legacy_indices]
    new_residual = np.asarray(new.arrays["residual_group_target"], dtype=float)[new_indices]
    if legacy_residual.shape != new_residual.shape:
        raise ValueError(
            f"Residual shape mismatch for scene {legacy.name!r}: legacy={legacy_residual.shape}, new={new_residual.shape}."
        )
    diff = new_residual - legacy_residual
    diff_flat = diff.reshape(-1)
    legacy_flat = legacy_residual.reshape(-1)
    new_flat = new_residual.reshape(-1)
    force_norm = np.linalg.norm(np.asarray(legacy.arrays["contact_force_world"], dtype=float)[legacy_indices], axis=1)
    contact_mask = np.asarray(legacy.arrays["contact_state"], dtype=float)[legacy_indices, 0] > 0.5
    high_force_threshold = float(np.percentile(force_norm[np.isfinite(force_norm)], DEFAULT_HIGH_FORCE_PERCENTILE)) if force_norm.size else 0.0
    high_force_mask = force_norm >= high_force_threshold
    sample_abs_diff = np.mean(np.abs(diff), axis=1)
    sample_signed_diff = np.mean(diff, axis=1)
    overlap_count = int(legacy_indices.size)
    return {
        "scene": legacy.name,
        "legacy_dataset_path": str(legacy.path),
        "new_dataset_path": str(new.path),
        "overlap_sample_count": overlap_count,
        "alignment_key": "episode_id+sample_step",
        "old_residual_min": float(np.min(legacy_flat)) if legacy_flat.size else None,
        "old_residual_max": float(np.max(legacy_flat)) if legacy_flat.size else None,
        "old_residual_mean": float(np.mean(legacy_flat)) if legacy_flat.size else None,
        "old_residual_std": float(np.std(legacy_flat)) if legacy_flat.size else None,
        "new_residual_min": float(np.min(new_flat)) if new_flat.size else None,
        "new_residual_max": float(np.max(new_flat)) if new_flat.size else None,
        "new_residual_mean": float(np.mean(new_flat)) if new_flat.size else None,
        "new_residual_std": float(np.std(new_flat)) if new_flat.size else None,
        "old_saturation_rate": float(np.mean(np.abs(legacy_flat) >= DEFAULT_RESIDUAL_SATURATION_FRACTION * float(legacy.metadata.get("residual_bound", 1.0)))) if legacy_flat.size else None,
        "new_saturation_rate": float(np.mean(np.abs(new_flat) >= DEFAULT_RESIDUAL_SATURATION_FRACTION * float(new.metadata.get("residual_bound", 1.0)))) if new_flat.size else None,
        "old_positive_fraction": float(np.mean(legacy_flat > 0.0)) if legacy_flat.size else None,
        "old_negative_fraction": float(np.mean(legacy_flat < 0.0)) if legacy_flat.size else None,
        "new_positive_fraction": float(np.mean(new_flat > 0.0)) if new_flat.size else None,
        "new_negative_fraction": float(np.mean(new_flat < 0.0)) if new_flat.size else None,
        "correlation_old_new": float(np.corrcoef(legacy_flat, new_flat)[0, 1]) if legacy_flat.size and np.std(legacy_flat) > 1e-12 and np.std(new_flat) > 1e-12 else None,
        "mean_absolute_difference": float(np.mean(np.abs(diff_flat))) if diff_flat.size else None,
        "contact_phase_difference": _flatten_mean(sample_signed_diff, contact_mask),
        "high_force_phase_difference": _flatten_mean(sample_signed_diff, high_force_mask),
        "contact_phase_mean_absolute_difference": _flatten_mean(sample_abs_diff, contact_mask),
        "high_force_phase_mean_absolute_difference": _flatten_mean(sample_abs_diff, high_force_mask),
        "high_force_threshold": high_force_threshold,
        "semantic_checks": {
            "same_residual_dim": legacy_residual.shape[1] == new_residual.shape[1],
            "new_projection_is_residual_first": is_residual_first_projection(new.metadata.get("label_projection")),
            "old_projection_is_legacy": legacy.metadata.get("label_projection") == "direct_k_phys_then_residual_legacy",
        },
    }


def summarize_root(root: Path) -> tuple[list[DatasetRecord], list[dict[str, Any]]]:
    records = load_scene_records(root)
    rows = [compute_scene_audit(record) for record in records]
    return records, rows


def summarize_root_comparison(legacy_root: Path, new_root: Path) -> dict[str, Any]:
    legacy_records = {record.name: record for record in load_scene_records(legacy_root)}
    new_records = {record.name: record for record in load_scene_records(new_root)}
    common_scenes = sorted(set(legacy_records) & set(new_records))
    scene_rows = []
    for scene in common_scenes:
        scene_rows.append(compute_legacy_comparison(legacy_records[scene], new_records[scene]))
    return {
        "legacy_root": str(legacy_root),
        "new_root": str(new_root),
        "common_scenes": common_scenes,
        "scenes": {row["scene"]: row for row in scene_rows},
        "scene_rows": scene_rows,
    }


__all__ = [
    "DatasetRecord",
    "compute_legacy_comparison",
    "compute_scene_audit",
    "load_dataset",
    "load_scene_records",
    "summarize_root",
    "summarize_root_comparison",
    "write_csv",
    "write_json",
]
