from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec
from stiffness_copilot_mujoco.learning.stiffness_labels import cholesky_params_to_matrix, spd_project


AUGMENTED_OBS_KEY = "augmented_obs"
SCHEMA_VERSION = "augmented_residual_bc_v1"
POLICY_SCHEMA_VERSION = "augmented_residual_bc_policy_v1"


def augmented_feature_names(*, task_state_dim: int, residual_dim: int) -> list[str]:
    names: list[str] = []
    names.extend(f"task_state_{idx}" for idx in range(task_state_dim))
    names.extend(f"task_state_delta_1_{idx}" for idx in range(task_state_dim))
    names.extend(f"task_state_delta_history_{idx}" for idx in range(task_state_dim))
    names.extend(
        [
            "normal_force",
            "normal_force_delta_1",
            "recent_max_normal_force",
            "contact_flag",
            "contact_duration",
        ]
    )
    names.extend(f"previous_residual_group_{idx}" for idx in range(residual_dim))
    return names


def _episode_start_indices(episode_id: np.ndarray) -> dict[int, int]:
    result: dict[int, int] = {}
    for idx, value in enumerate(episode_id):
        result.setdefault(int(value), idx)
    return result


def build_augmented_observations(
    *,
    task_state: np.ndarray,
    normal_force: np.ndarray,
    contact_flag: np.ndarray,
    residual_group_target: np.ndarray,
    episode_id: np.ndarray,
    history_steps: int = 5,
) -> np.ndarray:
    if history_steps <= 0:
        raise ValueError("history_steps must be positive.")
    task_state = np.asarray(task_state, dtype=float)
    normal_force = np.asarray(normal_force, dtype=float).reshape(-1)
    contact_flag = np.asarray(contact_flag, dtype=bool).reshape(-1)
    residual_group_target = np.asarray(residual_group_target, dtype=float)
    episode_id = np.asarray(episode_id, dtype=int).reshape(-1)
    n = task_state.shape[0]
    if normal_force.shape[0] != n or contact_flag.shape[0] != n or residual_group_target.shape[0] != n or episode_id.shape[0] != n:
        raise ValueError("All augmented observation inputs must have the same sample count.")
    if task_state.ndim != 2 or residual_group_target.ndim != 2:
        raise ValueError("task_state and residual_group_target must be rank-2 arrays.")

    starts = _episode_start_indices(episode_id)
    contact_duration = np.zeros(n, dtype=float)
    delta_1 = np.zeros_like(task_state)
    delta_history = np.zeros_like(task_state)
    force_delta = np.zeros(n, dtype=float)
    recent_max_force = np.zeros(n, dtype=float)
    prev_residual = np.zeros_like(residual_group_target)

    current_episode: int | None = None
    duration = 0.0
    for idx in range(n):
        ep = int(episode_id[idx])
        start = starts[ep]
        if current_episode != ep:
            current_episode = ep
            duration = 0.0
        prev_idx = idx - 1 if idx > start else idx
        hist_idx = max(start, idx - history_steps)
        delta_1[idx] = task_state[idx] - task_state[prev_idx]
        delta_history[idx] = task_state[idx] - task_state[hist_idx]
        force_delta[idx] = normal_force[idx] - normal_force[prev_idx]
        recent_max_force[idx] = float(np.max(normal_force[hist_idx : idx + 1]))
        prev_residual[idx] = residual_group_target[prev_idx] if idx > start else 0.0
        duration = duration + 1.0 if contact_flag[idx] else 0.0
        contact_duration[idx] = duration

    return np.hstack(
        [
            task_state,
            delta_1,
            delta_history,
            normal_force[:, None],
            force_delta[:, None],
            recent_max_force[:, None],
            contact_flag.astype(float)[:, None],
            contact_duration[:, None],
            prev_residual,
        ]
    )


@dataclass
class AugmentedObservationBuilder:
    task_state_dim: int
    residual_dim: int
    history_steps: int = 5

    def __post_init__(self) -> None:
        if self.history_steps <= 0:
            raise ValueError("history_steps must be positive.")
        self.reset()

    def reset(self) -> None:
        self._task_history: list[np.ndarray] = []
        self._force_history: list[float] = []
        self._contact_duration = 0.0
        self._prev_residual = np.zeros(self.residual_dim, dtype=float)

    def observe(
        self,
        *,
        task_state: np.ndarray,
        normal_force: float,
        contact: bool,
        previous_residual: np.ndarray | None = None,
    ) -> np.ndarray:
        task_state = np.asarray(task_state, dtype=float)
        if task_state.shape != (self.task_state_dim,):
            raise ValueError(f"task_state must have shape ({self.task_state_dim},), got {task_state.shape}.")
        if previous_residual is not None:
            prev_residual = np.asarray(previous_residual, dtype=float)
            if prev_residual.shape != (self.residual_dim,):
                raise ValueError(f"previous_residual must have shape ({self.residual_dim},), got {prev_residual.shape}.")
            self._prev_residual = prev_residual.copy()
        prev_task = self._task_history[-1] if self._task_history else task_state
        hist_task = self._task_history[-self.history_steps] if len(self._task_history) >= self.history_steps else (self._task_history[0] if self._task_history else task_state)
        prev_force = self._force_history[-1] if self._force_history else float(normal_force)
        recent_forces = [*self._force_history[-self.history_steps + 1 :], float(normal_force)] if self.history_steps > 1 else [float(normal_force)]
        self._contact_duration = self._contact_duration + 1.0 if contact else 0.0
        return np.concatenate(
            [
                task_state,
                task_state - prev_task,
                task_state - hist_task,
                np.array(
                    [
                        float(normal_force),
                        float(normal_force) - prev_force,
                        float(np.max(recent_forces)),
                        float(contact),
                        self._contact_duration,
                    ],
                    dtype=float,
                ),
                self._prev_residual,
            ]
        )

    def update(self, *, task_state: np.ndarray, normal_force: float) -> None:
        self._task_history.append(np.asarray(task_state, dtype=float).copy())
        self._force_history.append(float(normal_force))
        if len(self._task_history) > self.history_steps:
            self._task_history = self._task_history[-self.history_steps :]
            self._force_history = self._force_history[-self.history_steps :]

    def set_previous_residual(self, residual: np.ndarray) -> None:
        residual = np.asarray(residual, dtype=float)
        if residual.shape != (self.residual_dim,):
            raise ValueError(f"residual must have shape ({self.residual_dim},), got {residual.shape}.")
        self._prev_residual = residual.copy()


@dataclass(frozen=True)
class AugmentedResidualSPDStiffnessPolicy:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    base_spec: BaseStiffnessSpec
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "AugmentedResidualSPDStiffnessPolicy":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            return cls(
                w1=data["w1"].astype(float),
                b1=data["b1"].astype(float),
                w2=data["w2"].astype(float),
                b2=data["b2"].astype(float),
                x_mean=data["x_mean"].astype(float),
                x_std=data["x_std"].astype(float),
                y_mean=data["y_mean"].astype(float),
                y_std=data["y_std"].astype(float),
                base_spec=BaseStiffnessSpec.from_metadata(metadata["base_stiffness_spec"]),
                metadata=metadata,
            )

    @property
    def input_dim(self) -> int:
        return int(self.x_mean.shape[0])

    @property
    def history_steps(self) -> int:
        return int(self.metadata.get("history_steps", 5))

    def predict_group_delta_raw(self, augmented_obs: np.ndarray) -> np.ndarray:
        x = np.asarray(augmented_obs, dtype=float)
        if x.shape != self.x_mean.shape:
            raise ValueError(f"augmented_obs must have shape {self.x_mean.shape}, got {x.shape}.")
        x_norm = (x - self.x_mean) / self.x_std
        hidden = np.maximum(0.0, x_norm @ self.w1 + self.b1)
        y_norm = hidden @ self.w2 + self.b2
        return y_norm * self.y_std + self.y_mean

    def predict(self, augmented_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = self.predict_group_delta_raw(augmented_obs)
        bounded = self.base_spec.residual_bounds * np.tanh(raw / np.maximum(self.base_spec.residual_bounds, 1e-12))
        theta_delta = self.base_spec.expand_group_delta(bounded, clip=True)
        theta = self.base_spec.theta_base + theta_delta
        matrix = spd_project(cholesky_params_to_matrix(theta))
        return matrix, theta, theta_delta, bounded


def save_augmented_residual_policy(
    path: Path,
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    history: np.ndarray,
    base_spec: BaseStiffnessSpec,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full_metadata = dict(metadata)
    full_metadata["schema_version"] = POLICY_SCHEMA_VERSION
    full_metadata["base_stiffness_spec"] = base_spec.to_metadata()
    np.savez_compressed(
        path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        history=history,
        metadata=json.dumps(full_metadata, sort_keys=True),
    )


__all__ = [
    "AUGMENTED_OBS_KEY",
    "POLICY_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "AugmentedObservationBuilder",
    "AugmentedResidualSPDStiffnessPolicy",
    "augmented_feature_names",
    "build_augmented_observations",
    "save_augmented_residual_policy",
]
