from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stiffness_copilot_mujoco.learning.stiffness_labels import cholesky_params_to_matrix, spd_exp, spd_log, spd_project, symmetrize


@dataclass(frozen=True)
class SupervisedStiffnessPolicy:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    metadata: dict

    @classmethod
    def load(cls, path: Path) -> "SupervisedStiffnessPolicy":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"])) if "metadata" in data.files else {}
            return cls(
                w1=data["w1"].astype(float),
                b1=data["b1"].astype(float),
                w2=data["w2"].astype(float),
                b2=data["b2"].astype(float),
                x_mean=data["x_mean"].astype(float),
                x_std=data["x_std"].astype(float),
                y_mean=data["y_mean"].astype(float),
                y_std=data["y_std"].astype(float),
                metadata=metadata,
            )

    def predict_cholesky(self, task_state: np.ndarray) -> np.ndarray:
        x = np.asarray(task_state, dtype=float)
        if x.shape != self.x_mean.shape:
            raise ValueError(f"task_state must have shape {self.x_mean.shape}, got {x.shape}.")
        x_norm = (x - self.x_mean) / self.x_std
        hidden = np.maximum(0.0, x_norm @ self.w1 + self.b1)
        y_norm = hidden @ self.w2 + self.b2
        return y_norm * self.y_std + self.y_mean

    def predict_normalized_matrix(self, task_state: np.ndarray) -> np.ndarray:
        matrix = cholesky_params_to_matrix(self.predict_cholesky(task_state))
        return spd_project(matrix, min_eigenvalue=1e-8, max_eigenvalue=1.0)


def scale_normalized_stiffness(matrix: np.ndarray, *, k_min: float, k_max: float) -> np.ndarray:
    if k_min <= 0.0 or k_max <= k_min:
        raise ValueError(f"Expected 0 < k_min < k_max, got k_min={k_min:g}, k_max={k_max:g}.")
    eigvals, eigvecs = np.linalg.eigh(spd_project(matrix, min_eigenvalue=1e-8, max_eigenvalue=1.0))
    physical = k_min + eigvals * (k_max - k_min)
    return symmetrize(eigvecs @ np.diag(physical) @ eigvecs.T)


def log_euclidean_ema_spd(
    previous: np.ndarray,
    raw: np.ndarray,
    *,
    alpha: float,
    eps: float = 1e-8,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Expected 0 <= alpha <= 1, got alpha={alpha:g}.")
    previous_spd = spd_project(previous)
    raw_spd = spd_project(raw)
    eye = np.eye(previous_spd.shape[0], dtype=float)
    smoothed_log = (1.0 - alpha) * spd_log(previous_spd + eps * eye) + alpha * spd_log(raw_spd + eps * eye)
    return spd_project(spd_exp(smoothed_log), min_eigenvalue=eps)


__all__ = ["SupervisedStiffnessPolicy", "log_euclidean_ema_spd", "scale_normalized_stiffness"]
