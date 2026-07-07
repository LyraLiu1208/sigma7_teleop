from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.learning.stiffness_labels import matrix_to_cholesky_params, symmetrize


LABEL_PROJECTION_RESIDUAL_FIRST = "residual_first_baseline_relative_v1"
LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2 = "residual_first_contact_gated_centered_v2"
LABEL_PROJECTION_LEGACY = "direct_k_phys_then_residual_legacy"

DEFAULT_BASELINE_K = 600.0
DEFAULT_DIAGNOSTIC_K_MIN = 300.0
DEFAULT_DIAGNOSTIC_K_MAX = 900.0
DEFAULT_RESIDUAL_BOUND = 0.35
DEFAULT_L21_COUPLING_PERCENTILE = 95.0
DEFAULT_CONTACT_GATE_LOW = 1.0
DEFAULT_CONTACT_GATE_HIGH = 10.0
DEFAULT_NEUTRAL_CONTACT_THRESHOLD = DEFAULT_CONTACT_GATE_HIGH
DEFAULT_MIN_CALIBRATION_SAMPLES = 8

RESIDUAL_FIRST_LABEL_PROJECTIONS = {
    LABEL_PROJECTION_RESIDUAL_FIRST,
    LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2,
}


@dataclass(frozen=True)
class ResidualFirstLabelProjectionResult:
    residual_group_target: np.ndarray
    residual_theta_target: np.ndarray
    physical_stiffness_matrix_target: np.ndarray
    physical_stiffness_cholesky_target: np.ndarray
    physical_stiffness_eigenvalues: np.ndarray
    label_metadata: dict[str, Any]


def _scene_kind(base_spec: BaseStiffnessSpec) -> str:
    active_group_names = tuple(str(name) for name in base_spec.active_group_names)
    if active_group_names == ("alpha_lateral_shared",):
        return "circle"
    if active_group_names == ("alpha_x", "alpha_y"):
        return "polygon"
    if active_group_names == ("alpha_x", "alpha_y", "l21"):
        return "star"
    raise ValueError(
        "Unsupported residual-first label projection contract for active_group_names="
        f"{active_group_names!r}."
    )


def is_residual_first_projection(label_projection: str | None) -> bool:
    return bool(label_projection in RESIDUAL_FIRST_LABEL_PROJECTIONS)


def _residual_to_physical_scalar(
    residual: np.ndarray,
    *,
    residual_bound: float,
    diagnostic_k_min: float,
    diagnostic_k_max: float,
) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    if residual_bound <= 0.0:
        raise ValueError("residual_bound must be positive.")
    if diagnostic_k_max <= diagnostic_k_min:
        raise ValueError("diagnostic_k_max must be larger than diagnostic_k_min.")
    residual_scale = np.clip(residual / residual_bound, -1.0, 1.0)
    midpoint = 0.5 * (diagnostic_k_min + diagnostic_k_max)
    half_span = 0.5 * (diagnostic_k_max - diagnostic_k_min)
    return midpoint + residual_scale * half_span


def _physical_matrix_from_targets(
    *,
    kx: float,
    ky: float,
    kz: float,
    l21: float = 0.0,
) -> np.ndarray:
    kx = float(max(kx, 1e-8))
    ky = float(max(ky, 1e-8))
    kz = float(max(kz, 1e-8))
    limit = 0.999 * float(np.sqrt(max(ky, 1e-8)))
    l21 = float(np.clip(l21, -limit, limit))
    l22_sq = max(ky - l21 * l21, 1e-8)
    lower = np.array(
        [
            [np.sqrt(kx), 0.0, 0.0],
            [l21, np.sqrt(l22_sq), 0.0],
            [0.0, 0.0, np.sqrt(kz)],
        ],
        dtype=float,
    )
    return symmetrize(lower @ lower.T)


def _contact_gate(force_magnitude: np.ndarray, *, low: float, high: float) -> np.ndarray:
    magnitude = np.asarray(force_magnitude, dtype=float).reshape(-1)
    if high <= low:
        raise ValueError("contact gate high threshold must be larger than low threshold.")
    return np.clip((magnitude - low) / (high - low), 0.0, 1.0)


def _selected_score_components(
    *,
    scene_kind: str,
    diag: np.ndarray,
    normalized_l21: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    if scene_kind == "circle":
        k_lat_norm = 0.5 * (diag[:, 0] + diag[:, 1])
        return k_lat_norm[:, None], ["k_lat_norm"]
    if scene_kind == "polygon":
        return diag[:, :2], ["K_robot_norm_xx", "K_robot_norm_yy"]
    if scene_kind == "star":
        return np.column_stack([diag[:, :2], normalized_l21]), ["K_robot_norm_xx", "K_robot_norm_yy", "l21"]
    raise AssertionError(scene_kind)


def _centered_residual_projection_v2(
    selected_scores: np.ndarray,
    force_magnitude: np.ndarray,
    *,
    residual_bound: float,
    contact_gate_low: float = DEFAULT_CONTACT_GATE_LOW,
    contact_gate_high: float = DEFAULT_CONTACT_GATE_HIGH,
    neutral_contact_threshold: float = DEFAULT_NEUTRAL_CONTACT_THRESHOLD,
    min_calibration_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
) -> tuple[np.ndarray, dict[str, Any]]:
    scores = np.asarray(selected_scores, dtype=float)
    force = np.asarray(force_magnitude, dtype=float).reshape(-1)
    if scores.ndim != 2:
        raise ValueError(f"selected_scores must have shape [N, D], got {scores.shape}.")
    if force.shape != (scores.shape[0],):
        raise ValueError(f"force_magnitude must have shape ({scores.shape[0]},), got {force.shape}.")
    if residual_bound <= 0.0:
        raise ValueError("residual_bound must be positive.")

    gate = _contact_gate(force, low=contact_gate_low, high=contact_gate_high)
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
        "residual_group_target_space": "centered_bounded_residual",
    }
    return residual, metadata


def project_residual_first_labels(
    normalized_matrix: np.ndarray,
    *,
    base_spec: BaseStiffnessSpec,
    contact_forces_world: np.ndarray | None = None,
    label_projection: str = LABEL_PROJECTION_RESIDUAL_FIRST,
    residual_bound: float = DEFAULT_RESIDUAL_BOUND,
    baseline_k: float = DEFAULT_BASELINE_K,
    diagnostic_k_min: float = DEFAULT_DIAGNOSTIC_K_MIN,
    diagnostic_k_max: float = DEFAULT_DIAGNOSTIC_K_MAX,
    l21_coupling_percentile: float = DEFAULT_L21_COUPLING_PERCENTILE,
    contact_gate_low: float = DEFAULT_CONTACT_GATE_LOW,
    contact_gate_high: float = DEFAULT_CONTACT_GATE_HIGH,
    neutral_contact_threshold: float = DEFAULT_NEUTRAL_CONTACT_THRESHOLD,
    min_calibration_samples: int = DEFAULT_MIN_CALIBRATION_SAMPLES,
) -> ResidualFirstLabelProjectionResult:
    matrix = np.asarray(normalized_matrix, dtype=float)
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError(f"normalized_matrix must have shape [N, 3, 3], got {matrix.shape}.")
    if residual_bound <= 0.0:
        raise ValueError("residual_bound must be positive.")
    if label_projection not in RESIDUAL_FIRST_LABEL_PROJECTIONS:
        raise ValueError(
            "label_projection must be one of "
            f"{sorted(RESIDUAL_FIRST_LABEL_PROJECTIONS)!r}, observed {label_projection!r}."
        )

    scene_kind = _scene_kind(base_spec)
    diag = np.diagonal(matrix, axis1=1, axis2=2)
    cholesky = np.stack([matrix_to_cholesky_params(item) for item in matrix], axis=0)
    normalized_l21 = cholesky[:, 3]
    selected_scores, selected_score_names = _selected_score_components(
        scene_kind=scene_kind,
        diag=diag,
        normalized_l21=normalized_l21,
    )
    if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2:
        if contact_forces_world is None:
            raise ValueError("contact_forces_world is required for contact-gated centered projection v2.")
        force_magnitude = np.linalg.norm(np.asarray(contact_forces_world, dtype=float), axis=1)
        residual_group_target, v2_metadata = _centered_residual_projection_v2(
            selected_scores,
            force_magnitude,
            residual_bound=residual_bound,
            contact_gate_low=contact_gate_low,
            contact_gate_high=contact_gate_high,
            neutral_contact_threshold=neutral_contact_threshold,
            min_calibration_samples=min_calibration_samples,
        )
        if scene_kind == "star":
            # Keep the coupling calibration centered and bounded in the v2 path.
            residual_group_target[:, 2] = np.clip(residual_group_target[:, 2], -residual_bound, residual_bound)
            coupling_scale = float(v2_metadata["robust_scale_values"][2])
        else:
            coupling_scale = None
    else:
        if scene_kind == "star":
            raw_scale = float(np.percentile(np.abs(normalized_l21), l21_coupling_percentile))
            if not np.isfinite(raw_scale) or raw_scale <= 1e-12:
                raw_scale = 1.0
            residual_xy = residual_bound * (2.0 * diag[:, :2] - 1.0)
            residual_l21 = residual_bound * np.clip(normalized_l21 / raw_scale, -1.0, 1.0)
            residual_group_target = np.column_stack([residual_xy, residual_l21])
            coupling_scale = raw_scale
        elif scene_kind == "circle":
            k_lat_norm = 0.5 * (diag[:, 0] + diag[:, 1])
            residual_group_target = residual_bound * (2.0 * k_lat_norm - 1.0)
            residual_group_target = residual_group_target[:, None]
            coupling_scale = None
        elif scene_kind == "polygon":
            residual_group_target = residual_bound * (2.0 * diag[:, :2] - 1.0)
            coupling_scale = None
        else:  # pragma: no cover - guarded by _scene_kind.
            raise AssertionError(scene_kind)
        v2_metadata = {}

    physical_x = _residual_to_physical_scalar(
        residual_group_target[:, 0],
        residual_bound=residual_bound,
        diagnostic_k_min=diagnostic_k_min,
        diagnostic_k_max=diagnostic_k_max,
    )
    physical_y = _residual_to_physical_scalar(
        residual_group_target[:, 0] if scene_kind == "circle" else residual_group_target[:, 1],
        residual_bound=residual_bound,
        diagnostic_k_min=diagnostic_k_min,
        diagnostic_k_max=diagnostic_k_max,
    )
    physical_z = np.full(matrix.shape[0], float(base_spec.base_matrix[2, 2]), dtype=float)

    if scene_kind == "star":
        raw_l21 = (residual_group_target[:, 2] / residual_bound) * float(coupling_scale)
        max_l21 = 0.999 * np.sqrt(np.maximum(physical_y, 1e-8))
        physical_l21 = np.clip(raw_l21, -max_l21, max_l21)
        physical_matrix = np.stack(
            [
                _physical_matrix_from_targets(kx=kx, ky=ky, kz=kz, l21=l21)
                for kx, ky, kz, l21 in zip(physical_x, physical_y, physical_z, physical_l21, strict=True)
            ],
            axis=0,
        )
    else:
        physical_matrix = np.stack(
            [
                _physical_matrix_from_targets(kx=kx, ky=ky, kz=kz)
                for kx, ky, kz in zip(physical_x, physical_y, physical_z, strict=True)
            ],
            axis=0,
        )

    physical_cholesky = np.stack([matrix_to_cholesky_params(item) for item in physical_matrix], axis=0)
    physical_eigenvalues = np.linalg.eigvalsh(physical_matrix)
    residual_theta_target = np.stack([base_spec.expand_group_delta(delta, clip=True) for delta in residual_group_target], axis=0)

    label_metadata: dict[str, Any] = {
        "label_projection": label_projection,
        "label_projection_legacy": LABEL_PROJECTION_LEGACY,
        "primary_supervision_target": "residual_group_target",
        "primary_target_space": "baseline_relative_residual",
        "residual_to_physical_semantics": "negative_softens_positive_stiffens",
        "label_projection_description": (
            "residual-first baseline-relative bounded residual stiffness adaptation"
            if label_projection == LABEL_PROJECTION_RESIDUAL_FIRST
            else "contact-gated baseline-anchored centered residual stiffness adaptation"
        ),
        "baseline_k": float(baseline_k),
        "diagnostic_k_min": float(diagnostic_k_min),
        "diagnostic_k_max": float(diagnostic_k_max),
        "residual_bound": float(residual_bound),
        "diagnostic_reconstruction_role": "diagnostic_only_reconstruction_from_residual_group_target",
        "physical_stiffness_matrix_target_role": "diagnostic_reconstruction_from_residual",
        "label_k_range_role": "deprecated_legacy_compatibility_only",
        "physical_stiffness_eigenvalues_field": "physical_stiffness_eigenvalues",
        "residual_group_target_role": "primary_supervision_target",
        "selected_score_names": selected_score_names,
        "selected_K_norm_range": [0.0, 1.0],
        "selected_K_norm_neutral": 0.5,
        "l21_coupling_normalization": None
        if coupling_scale is None
        else {
            "method": (
                "contact_gated_centered_v2_neutral_centered_robust_scale"
                if label_projection == LABEL_PROJECTION_CONTACT_GATED_CENTERED_V2
                else "robust_abs_percentile_clip"
            ),
            "percentile": float(l21_coupling_percentile),
            "scale": float(coupling_scale),
            "bounded_range": [-float(residual_bound), float(residual_bound)],
        },
    }
    label_metadata.update(v2_metadata)
    return ResidualFirstLabelProjectionResult(
        residual_group_target=residual_group_target,
        residual_theta_target=residual_theta_target,
        physical_stiffness_matrix_target=physical_matrix,
        physical_stiffness_cholesky_target=physical_cholesky,
        physical_stiffness_eigenvalues=physical_eigenvalues,
        label_metadata=label_metadata,
    )
