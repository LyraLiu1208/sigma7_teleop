from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stiffness_copilot_mujoco.learning.stiffness_labels import (
    cholesky_params_to_matrix,
    matrix_to_cholesky_params,
    spd_project,
)


PARAM_NAMES = ("alpha1", "alpha2", "alpha3", "l21", "l31", "l32")


@dataclass(frozen=True)
class BaseStiffnessSpec:
    base_matrix: np.ndarray
    theta_base: np.ndarray
    active_groups: tuple[tuple[int, ...], ...]
    active_group_names: tuple[str, ...]
    residual_bounds: np.ndarray

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        active_groups: tuple[tuple[int, ...], ...],
        active_group_names: tuple[str, ...] | None = None,
        residual_bound: float = 0.35,
    ) -> "BaseStiffnessSpec":
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"matrix must have shape (3, 3), got {matrix.shape}.")
        matrix = 0.5 * (matrix + matrix.T)
        eigvals = np.linalg.eigvalsh(matrix)
        if np.any(eigvals <= 0.0):
            raise ValueError(f"matrix must be positive definite, got eigenvalues {eigvals}.")
        theta = matrix_to_cholesky_params(matrix)
        names = active_group_names or tuple("+".join(PARAM_NAMES[idx] for idx in group) for group in active_groups)
        return cls(
            base_matrix=matrix,
            theta_base=theta,
            active_groups=active_groups,
            active_group_names=names,
            residual_bounds=np.full(len(active_groups), float(residual_bound), dtype=float),
        )

    @classmethod
    def from_stiffness_diag(
        cls,
        stiffness: tuple[float, float, float] | np.ndarray,
        *,
        active_groups: tuple[tuple[int, ...], ...],
        active_group_names: tuple[str, ...] | None = None,
        residual_bound: float = 0.35,
    ) -> "BaseStiffnessSpec":
        return cls.from_matrix(
            np.diag(np.asarray(stiffness, dtype=float)),
            active_groups=active_groups,
            active_group_names=active_group_names,
            residual_bound=residual_bound,
        )

    def target_to_group_delta(self, theta_target: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta_target, dtype=float)
        if theta.shape != (6,):
            raise ValueError(f"theta_target must have shape (6,), got {theta.shape}.")
        delta = theta - self.theta_base
        grouped = np.array([float(np.mean(delta[list(group)])) for group in self.active_groups], dtype=float)
        return np.clip(grouped, -self.residual_bounds, self.residual_bounds)

    def expand_group_delta(self, group_delta: np.ndarray, *, clip: bool = True) -> np.ndarray:
        values = np.asarray(group_delta, dtype=float)
        if values.shape != (len(self.active_groups),):
            raise ValueError(f"group_delta must have shape ({len(self.active_groups)},), got {values.shape}.")
        if clip:
            values = np.clip(values, -self.residual_bounds, self.residual_bounds)
        delta = np.zeros(6, dtype=float)
        for value, group in zip(values, self.active_groups, strict=True):
            for idx in group:
                delta[idx] = value
        return delta

    def matrix_from_group_delta(self, group_delta: np.ndarray, *, clip: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta = self.expand_group_delta(group_delta, clip=clip)
        theta = self.theta_base + delta
        return spd_project(cholesky_params_to_matrix(theta)), theta, delta

    def to_metadata(self) -> dict[str, object]:
        return {
            "theta_base": self.theta_base.tolist(),
            "base_matrix": self.base_matrix.tolist(),
            "active_groups": [list(group) for group in self.active_groups],
            "active_group_names": list(self.active_group_names),
            "residual_bounds": self.residual_bounds.tolist(),
            "param_names": list(PARAM_NAMES),
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, object]) -> "BaseStiffnessSpec":
        return cls(
            base_matrix=np.asarray(metadata["base_matrix"], dtype=float),
            theta_base=np.asarray(metadata["theta_base"], dtype=float),
            active_groups=tuple(tuple(int(idx) for idx in group) for group in metadata["active_groups"]),
            active_group_names=tuple(str(name) for name in metadata["active_group_names"]),
            residual_bounds=np.asarray(metadata["residual_bounds"], dtype=float),
        )


@dataclass(frozen=True)
class ResidualSPDStiffnessPolicy:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    base_spec: BaseStiffnessSpec
    metadata: dict

    @classmethod
    def load(cls, path: Path) -> "ResidualSPDStiffnessPolicy":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if "layer_count" in metadata:
                layer_count = int(metadata["layer_count"])
                weights = tuple(data[f"layer_{idx}_weight"].astype(float) for idx in range(layer_count))
                biases = tuple(data[f"layer_{idx}_bias"].astype(float) for idx in range(layer_count))
            else:
                weights = (data["w1"].astype(float), data["w2"].astype(float))
                biases = (data["b1"].astype(float), data["b2"].astype(float))
            return cls(
                weights=weights,
                biases=biases,
                x_mean=data["x_mean"].astype(float),
                x_std=data["x_std"].astype(float),
                y_mean=data["y_mean"].astype(float),
                y_std=data["y_std"].astype(float),
                base_spec=BaseStiffnessSpec.from_metadata(metadata["base_stiffness_spec"]),
                metadata=metadata,
            )

    def _require_single_hidden_layer(self) -> None:
        if len(self.weights) != 2:
            raise ValueError(
                "This compatibility attribute is available only for one-hidden-layer BC policies; "
                f"the loaded policy has {len(self.weights) - 1} hidden layers."
            )

    @property
    def w1(self) -> np.ndarray:
        self._require_single_hidden_layer()
        return self.weights[0]

    @property
    def b1(self) -> np.ndarray:
        self._require_single_hidden_layer()
        return self.biases[0]

    @property
    def w2(self) -> np.ndarray:
        self._require_single_hidden_layer()
        return self.weights[1]

    @property
    def b2(self) -> np.ndarray:
        self._require_single_hidden_layer()
        return self.biases[1]

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return tuple(int(weight.shape[1]) for weight in self.weights[:-1])

    def predict_group_delta_raw(self, task_state: np.ndarray) -> np.ndarray:
        x = np.asarray(task_state, dtype=float)
        if x.shape != self.x_mean.shape:
            raise ValueError(f"task_state must have shape {self.x_mean.shape}, got {x.shape}.")
        x_norm = (x - self.x_mean) / self.x_std
        hidden = x_norm
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            hidden = np.maximum(0.0, hidden @ weight + bias)
        y_norm = hidden @ self.weights[-1] + self.biases[-1]
        return y_norm * self.y_std + self.y_mean

    def predict(self, task_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = self.predict_group_delta_raw(task_state)
        bounded = self.base_spec.residual_bounds * np.tanh(raw / np.maximum(self.base_spec.residual_bounds, 1e-12))
        matrix, theta, delta = self.base_spec.matrix_from_group_delta(bounded, clip=True)
        return matrix, theta, delta, bounded


def save_residual_policy(
    path: Path,
    *,
    weights: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
    biases: tuple[np.ndarray, ...] | list[np.ndarray] | None = None,
    w1: np.ndarray | None = None,
    b1: np.ndarray | None = None,
    w2: np.ndarray | None = None,
    b2: np.ndarray | None = None,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    history: np.ndarray,
    base_spec: BaseStiffnessSpec,
    metadata: dict,
) -> None:
    if weights is None or biases is None:
        if any(value is None for value in (w1, b1, w2, b2)):
            raise ValueError("Provide either weights/biases or the legacy w1/b1/w2/b2 arrays.")
        weights = (np.asarray(w1), np.asarray(w2))
        biases = (np.asarray(b1), np.asarray(b2))
    weights = tuple(np.asarray(value) for value in weights)
    biases = tuple(np.asarray(value) for value in biases)
    if len(weights) != len(biases) or len(weights) < 2:
        raise ValueError("A residual policy requires aligned weights/biases with at least one hidden layer.")
    path.parent.mkdir(parents=True, exist_ok=True)
    full_metadata = dict(metadata)
    full_metadata["base_stiffness_spec"] = base_spec.to_metadata()
    full_metadata["layer_count"] = len(weights)
    full_metadata["hidden_dims"] = [int(weight.shape[1]) for weight in weights[:-1]]
    arrays = {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "history": history,
        "metadata": json.dumps(full_metadata, sort_keys=True),
    }
    for idx, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        arrays[f"layer_{idx}_weight"] = weight
        arrays[f"layer_{idx}_bias"] = bias
    if len(weights) == 2:
        arrays.update(w1=weights[0], b1=biases[0], w2=weights[1], b2=biases[1])
    np.savez_compressed(path, **arrays)


__all__ = [
    "BaseStiffnessSpec",
    "PARAM_NAMES",
    "ResidualSPDStiffnessPolicy",
    "save_residual_policy",
]
