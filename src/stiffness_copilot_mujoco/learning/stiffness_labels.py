from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StiffnessLabelConfig:
    neighbors: int = 32
    knn_block_size: int = 1024
    polar_bins: int = 4
    azimuth_bins: int = 8
    force_percentile: float = 95.0
    regularization: float = 1e-6
    complement_regularization: float = 0.05
    normalize_percentiles: tuple[float, float] = (5.0, 95.0)
    contact_force_threshold: float = 1e-6
    min_normalization_contact_ratio: float = 0.2
    min_normalization_valid_sectors: int = 2


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    return 0.5 * (value + value.T)


def spd_project(matrix: np.ndarray, *, min_eigenvalue: float = 1e-8, max_eigenvalue: float | None = None) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(symmetrize(matrix))
    eigvals = np.maximum(eigvals, min_eigenvalue)
    if max_eigenvalue is not None:
        eigvals = np.minimum(eigvals, max_eigenvalue)
    return symmetrize(eigvecs @ np.diag(eigvals) @ eigvecs.T)


def spd_log(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(spd_project(matrix))
    return symmetrize(eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T)


def spd_exp(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(symmetrize(matrix))
    return symmetrize(eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T)


def matrix_to_cholesky_params(matrix: np.ndarray) -> np.ndarray:
    factor = np.linalg.cholesky(spd_project(matrix))
    return np.array(
        [
            np.log(factor[0, 0]),
            np.log(factor[1, 1]),
            np.log(factor[2, 2]),
            factor[1, 0],
            factor[2, 0],
            factor[2, 1],
        ],
        dtype=float,
    )


def cholesky_params_to_matrix(params: np.ndarray) -> np.ndarray:
    values = np.asarray(params, dtype=float)
    if values.shape != (6,):
        raise ValueError(f"Expected Cholesky parameter shape (6,), got {values.shape}.")
    factor = np.array(
        [
            [np.exp(values[0]), 0.0, 0.0],
            [values[3], np.exp(values[1]), 0.0],
            [values[4], values[5], np.exp(values[2])],
        ],
        dtype=float,
    )
    return symmetrize(factor @ factor.T)


def normalize_task_states(task_states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(task_states, dtype=float)
    if states.ndim != 2:
        raise ValueError(f"task_states must have shape [N, S], got {states.shape}.")
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (states - mean) / std, mean, std


def nearest_neighbor_indices(normalized_states: np.ndarray, *, neighbors: int, block_size: int = 1024) -> np.ndarray:
    states = np.asarray(normalized_states, dtype=float)
    if states.ndim != 2:
        raise ValueError(f"normalized_states must have shape [N, S], got {states.shape}.")
    if neighbors <= 0:
        raise ValueError("neighbors must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    k = min(neighbors, states.shape[0])
    result = np.empty((states.shape[0], k), dtype=np.int64)
    state_norms = np.einsum("ns,ns->n", states, states)
    for start in range(0, states.shape[0], block_size):
        stop = min(start + block_size, states.shape[0])
        block = states[start:stop]
        block_norms = np.einsum("bs,bs->b", block, block)
        distances = block_norms[:, None] + state_norms[None, :] - 2.0 * (block @ states.T)
        distances = np.maximum(distances, 0.0)
        result[start:stop] = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    return result


def _force_sector_ids(forces: np.ndarray, *, polar_bins: int, azimuth_bins: int) -> np.ndarray:
    norms = np.linalg.norm(forces, axis=1)
    directions = forces / np.maximum(norms[:, None], 1e-12)
    polar = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    azimuth = np.mod(np.arctan2(directions[:, 1], directions[:, 0]), 2.0 * np.pi)
    polar_id = np.minimum((polar / np.pi * polar_bins).astype(int), polar_bins - 1)
    azimuth_id = np.minimum((azimuth / (2.0 * np.pi) * azimuth_bins).astype(int), azimuth_bins - 1)
    return polar_id * azimuth_bins + azimuth_id


def infer_environment_stiffness(
    neighbor_forces: np.ndarray,
    *,
    polar_bins: int = 4,
    azimuth_bins: int = 8,
    force_percentile: float = 95.0,
    regularization: float = 1e-6,
) -> tuple[np.ndarray, int]:
    forces = np.asarray(neighbor_forces, dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError(f"neighbor_forces must have shape [K, 3], got {forces.shape}.")

    magnitudes = np.linalg.norm(forces, axis=1)
    valid = magnitudes > 1e-12
    if not np.any(valid):
        return np.eye(3, dtype=float) * np.sqrt(regularization), 0

    valid_forces = forces[valid]
    valid_magnitudes = magnitudes[valid]
    sector_ids = _force_sector_ids(valid_forces, polar_bins=polar_bins, azimuth_bins=azimuth_bins)
    log_matrices: list[np.ndarray] = []

    for sector_id in np.unique(sector_ids):
        sector_forces = valid_forces[sector_ids == sector_id]
        sector_magnitudes = valid_magnitudes[sector_ids == sector_id]
        directions = sector_forces / np.maximum(sector_magnitudes[:, None], 1e-12)
        direction = directions.mean(axis=0)
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-12:
            continue
        direction = direction / direction_norm
        magnitude = float(np.percentile(sector_magnitudes, force_percentile))
        representative_force = magnitude * direction
        sector_matrix = np.outer(representative_force, representative_force) + regularization * np.eye(3)
        log_matrices.append(spd_log(sector_matrix))

    if not log_matrices:
        return np.eye(3, dtype=float) * np.sqrt(regularization), 0
    mean_log = np.mean(np.stack(log_matrices, axis=0), axis=0)
    return spd_exp(mean_log), len(log_matrices)


def complementary_robot_stiffness(
    environment_stiffness: np.ndarray,
    *,
    complement_regularization: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(spd_project(environment_stiffness))
    raw_values = 1.0 / (eigvals + complement_regularization)
    raw_matrix = symmetrize(eigvecs @ np.diag(raw_values) @ eigvecs.T)
    return raw_matrix, raw_values


def normalize_robot_stiffness_matrices(
    raw_matrices: np.ndarray,
    *,
    percentiles: tuple[float, float] = (5.0, 95.0),
    bounds_mask: np.ndarray | None = None,
) -> np.ndarray:
    matrices = np.asarray(raw_matrices, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"raw_matrices must have shape [N, 3, 3], got {matrices.shape}.")

    if bounds_mask is None:
        matrices_for_bounds = matrices
    else:
        mask = np.asarray(bounds_mask, dtype=bool)
        if mask.shape != (matrices.shape[0],):
            raise ValueError(f"bounds_mask must have shape ({matrices.shape[0]},), got {mask.shape}.")
        matrices_for_bounds = matrices[mask]
        if matrices_for_bounds.shape[0] == 0:
            matrices_for_bounds = matrices

    all_eigvals = np.linalg.eigvalsh(np.stack([spd_project(matrix) for matrix in matrices_for_bounds], axis=0)).reshape(-1)
    lower, upper = np.percentile(all_eigvals, percentiles)
    if upper <= lower:
        upper = lower + 1.0

    normalized = []
    for matrix in matrices:
        eigvals, eigvecs = np.linalg.eigh(spd_project(matrix))
        scaled = np.clip((eigvals - lower) / (upper - lower), 1e-8, 1.0)
        normalized.append(symmetrize(eigvecs @ np.diag(scaled) @ eigvecs.T))
    return np.stack(normalized, axis=0)


def build_stiffness_labels(
    task_states: np.ndarray,
    contact_forces_world: np.ndarray,
    config: StiffnessLabelConfig = StiffnessLabelConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    matrices, cholesky, _ = build_stiffness_labels_with_diagnostics(task_states, contact_forces_world, config=config)
    return matrices, cholesky


def build_stiffness_labels_with_diagnostics(
    task_states: np.ndarray,
    contact_forces_world: np.ndarray,
    config: StiffnessLabelConfig = StiffnessLabelConfig(),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    states_norm, _, _ = normalize_task_states(task_states)
    neighbor_ids = nearest_neighbor_indices(states_norm, neighbors=config.neighbors, block_size=config.knn_block_size)

    raw_robot_matrices = []
    environment_matrices = []
    environment_eigvals = []
    raw_robot_eigvals = []
    neighbor_contact_ratios = []
    valid_sector_counts = []
    for ids in neighbor_ids:
        neighbor_forces = contact_forces_world[ids]
        neighbor_force_magnitudes = np.linalg.norm(neighbor_forces, axis=1)
        neighbor_contact_ratios.append(float(np.mean(neighbor_force_magnitudes > config.contact_force_threshold)))
        environment, valid_sector_count = infer_environment_stiffness(
            neighbor_forces,
            polar_bins=config.polar_bins,
            azimuth_bins=config.azimuth_bins,
            force_percentile=config.force_percentile,
            regularization=config.regularization,
        )
        raw_robot, _ = complementary_robot_stiffness(
            environment,
            complement_regularization=config.complement_regularization,
        )
        environment_matrices.append(environment)
        environment_eigvals.append(np.linalg.eigvalsh(spd_project(environment)))
        raw_robot_eigvals.append(np.linalg.eigvalsh(spd_project(raw_robot)))
        valid_sector_counts.append(valid_sector_count)
        raw_robot_matrices.append(raw_robot)

    neighbor_contact_ratio = np.asarray(neighbor_contact_ratios, dtype=float)
    valid_sector_count = np.asarray(valid_sector_counts, dtype=np.int32)
    bounds_mask = (
        (neighbor_contact_ratio >= config.min_normalization_contact_ratio)
        & (valid_sector_count >= config.min_normalization_valid_sectors)
    )
    normalized = normalize_robot_stiffness_matrices(
        np.stack(raw_robot_matrices, axis=0),
        percentiles=config.normalize_percentiles,
        bounds_mask=bounds_mask,
    )
    cholesky = np.stack([matrix_to_cholesky_params(matrix) for matrix in normalized], axis=0)
    diagnostics = {
        "label_neighbor_contact_ratio": neighbor_contact_ratio,
        "label_valid_sector_count": valid_sector_count,
        "environment_stiffness_matrix": np.stack(environment_matrices, axis=0),
        "raw_robot_stiffness_matrix": np.stack(raw_robot_matrices, axis=0),
        "environment_stiffness_eigvals": np.stack(environment_eigvals, axis=0),
        "raw_robot_stiffness_eigvals": np.stack(raw_robot_eigvals, axis=0),
        "label_normalization_mask": bounds_mask.astype(np.float64),
    }
    return normalized, cholesky, diagnostics
