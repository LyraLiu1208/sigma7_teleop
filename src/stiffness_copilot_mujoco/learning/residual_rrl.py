from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec, ResidualSPDStiffnessPolicy


POLICY_MODES = ("direct", "bc_residual")


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Residual RRL training requires torch. Install the project with torch>=2.2.") from exc
    return torch, nn


@dataclass(frozen=True)
class ResidualRRLPolicy:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    log_std: np.ndarray
    base_spec: BaseStiffnessSpec
    metadata: dict[str, Any]

    @property
    def policy_mode(self) -> str:
        return str(self.metadata.get("policy_mode", "direct"))

    @property
    def residual_alpha(self) -> float:
        return float(self.metadata.get("residual_alpha", 1.0))

    @property
    def correction_bound_scale(self) -> float:
        return float(self.metadata.get("correction_bound_scale", 1.0))

    @classmethod
    def load(cls, path: Path) -> "ResidualRRLPolicy":
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
                log_std=data["log_std"].astype(float),
                base_spec=BaseStiffnessSpec.from_metadata(metadata["base_stiffness_spec"]),
                metadata=metadata,
            )

    @classmethod
    def from_bc(
        cls,
        bc_policy: ResidualSPDStiffnessPolicy,
        *,
        log_std_init: float = -2.0,
        policy_mode: str = "direct",
        residual_alpha: float = 1.0,
        correction_bound_scale: float = 1.0,
        source_bc_policy: str | None = None,
    ) -> "ResidualRRLPolicy":
        if policy_mode not in POLICY_MODES:
            raise ValueError(f"Unsupported policy_mode {policy_mode!r}.")
        return cls(
            w1=bc_policy.w1.copy(),
            b1=bc_policy.b1.copy(),
            w2=bc_policy.w2.copy() if policy_mode == "direct" else np.zeros_like(bc_policy.w2),
            b2=bc_policy.b2.copy() if policy_mode == "direct" else np.zeros_like(bc_policy.b2),
            x_mean=bc_policy.x_mean.copy(),
            x_std=bc_policy.x_std.copy(),
            y_mean=bc_policy.y_mean.copy(),
            y_std=bc_policy.y_std.copy(),
            log_std=np.full(bc_policy.b2.shape, float(log_std_init), dtype=float),
            base_spec=bc_policy.base_spec,
            metadata={
                "schema_version": "residual_rrl_policy_v1",
                "policy_mode": policy_mode,
                "residual_alpha": float(residual_alpha),
                "correction_bound_scale": float(correction_bound_scale),
                "source_bc_policy": source_bc_policy,
                "source_bc_schema_version": bc_policy.metadata.get("schema_version", "unknown"),
                "source_bc_metadata": bc_policy.metadata,
            },
        )

    @property
    def action_dim(self) -> int:
        return int(self.b2.shape[0])

    def actor_output_raw(self, task_state: np.ndarray) -> np.ndarray:
        x = np.asarray(task_state, dtype=float)
        if x.shape != self.x_mean.shape:
            raise ValueError(f"task_state must have shape {self.x_mean.shape}, got {x.shape}.")
        x_norm = (x - self.x_mean) / self.x_std
        hidden = np.maximum(0.0, x_norm @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2

    def predict_group_delta_raw(self, task_state: np.ndarray) -> np.ndarray:
        raw = self.actor_output_raw(task_state)
        if self.policy_mode == "direct":
            return raw * self.y_std + self.y_mean
        if self.policy_mode == "bc_residual":
            return raw
        raise ValueError(f"Unsupported policy_mode {self.policy_mode!r}.")

    def bound_raw_action(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        bounds = np.maximum(self.base_spec.residual_bounds, 1e-12)
        return self.base_spec.residual_bounds * np.tanh(raw / bounds)

    def bound_correction(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        correction_bound = self.correction_bound_scale * self.base_spec.residual_bounds
        return correction_bound * np.tanh(raw)

    def predict_group_delta(self, task_state: np.ndarray, *, bc_policy: ResidualSPDStiffnessPolicy | None = None) -> np.ndarray:
        raw = self.predict_group_delta_raw(task_state)
        if self.policy_mode == "direct":
            return self.bound_raw_action(raw)
        if self.policy_mode == "bc_residual":
            if bc_policy is None:
                raise ValueError("bc_residual RRL policy prediction requires bc_policy.")
            bc_action = bc_policy.predict(task_state)[3]
            correction = self.bound_correction(raw)
            return np.clip(
                bc_action + self.residual_alpha * correction,
                -self.base_spec.residual_bounds,
                self.base_spec.residual_bounds,
            )
        raise ValueError(f"Unsupported policy_mode {self.policy_mode!r}.")

    def predict(
        self,
        task_state: np.ndarray,
        *,
        bc_policy: ResidualSPDStiffnessPolicy | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bounded = self.predict_group_delta(task_state, bc_policy=bc_policy)
        matrix, theta, delta = self.base_spec.matrix_from_group_delta(bounded, clip=True)
        return matrix, theta, delta, bounded


def save_residual_rrl_policy(
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
    log_std: np.ndarray,
    base_spec: BaseStiffnessSpec,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full_metadata = dict(metadata)
    full_metadata["base_stiffness_spec"] = base_spec.to_metadata()
    np.savez_compressed(
        path,
        w1=np.asarray(w1, dtype=float),
        b1=np.asarray(b1, dtype=float),
        w2=np.asarray(w2, dtype=float),
        b2=np.asarray(b2, dtype=float),
        x_mean=np.asarray(x_mean, dtype=float),
        x_std=np.asarray(x_std, dtype=float),
        y_mean=np.asarray(y_mean, dtype=float),
        y_std=np.asarray(y_std, dtype=float),
        log_std=np.asarray(log_std, dtype=float),
        metadata=json.dumps(full_metadata, sort_keys=True),
    )


class ResidualActorCriticPolicy:
    def __init__(
        self,
        *,
        actor: Any,
        critic: Any,
        log_std: Any,
        x_mean: Any,
        x_std: Any,
        y_mean: Any,
        y_std: Any,
        base_spec: BaseStiffnessSpec,
        policy_mode: str = "direct",
        residual_alpha: float = 1.0,
        correction_bound_scale: float = 1.0,
    ):
        if policy_mode not in POLICY_MODES:
            raise ValueError(f"Unsupported policy_mode {policy_mode!r}.")
        self.actor = actor
        self.critic = critic
        self.log_std = log_std
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.base_spec = base_spec
        self.policy_mode = policy_mode
        self.residual_alpha = float(residual_alpha)
        self.correction_bound_scale = float(correction_bound_scale)

    @classmethod
    def from_bc(
        cls,
        bc_policy: ResidualSPDStiffnessPolicy,
        *,
        hidden_dim: int | None = None,
        log_std_init: float = -2.0,
        seed: int = 0,
        policy_mode: str = "direct",
        residual_alpha: float = 1.0,
        correction_bound_scale: float = 1.0,
    ) -> "ResidualActorCriticPolicy":
        if policy_mode not in POLICY_MODES:
            raise ValueError(f"Unsupported policy_mode {policy_mode!r}.")
        torch, nn = _torch()
        torch.manual_seed(seed)
        hidden = hidden_dim or int(bc_policy.b1.shape[0])
        if hidden != int(bc_policy.b1.shape[0]):
            raise ValueError("RRL actor hidden_dim must match the BC policy hidden_dim for direct initialization.")
        action_dim = int(bc_policy.b2.shape[0])
        actor = nn.Sequential(
            nn.Linear(int(bc_policy.w1.shape[0]), hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, action_dim, dtype=torch.float64),
        )
        critic = nn.Sequential(
            nn.Linear(int(bc_policy.w1.shape[0]), hidden, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(hidden, 1, dtype=torch.float64),
        )
        with torch.no_grad():
            actor[0].weight.copy_(torch.as_tensor(bc_policy.w1.T, dtype=torch.float64))
            actor[0].bias.copy_(torch.as_tensor(bc_policy.b1, dtype=torch.float64))
            if policy_mode == "direct":
                actor[2].weight.copy_(torch.as_tensor(bc_policy.w2.T, dtype=torch.float64))
                actor[2].bias.copy_(torch.as_tensor(bc_policy.b2, dtype=torch.float64))
            else:
                actor[2].weight.zero_()
                actor[2].bias.zero_()
            for module in critic:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                    nn.init.zeros_(module.bias)
            nn.init.orthogonal_(critic[2].weight, gain=1.0)
        log_std = torch.nn.Parameter(torch.full((action_dim,), float(log_std_init), dtype=torch.float64))
        return cls(
            actor=actor,
            critic=critic,
            log_std=log_std,
            x_mean=torch.as_tensor(bc_policy.x_mean, dtype=torch.float64),
            x_std=torch.as_tensor(bc_policy.x_std, dtype=torch.float64),
            y_mean=torch.as_tensor(bc_policy.y_mean, dtype=torch.float64),
            y_std=torch.as_tensor(bc_policy.y_std, dtype=torch.float64),
            base_spec=bc_policy.base_spec,
            policy_mode=policy_mode,
            residual_alpha=residual_alpha,
            correction_bound_scale=correction_bound_scale,
        )

    @property
    def parameters(self):
        return list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters())

    def normalize_obs(self, obs: Any) -> Any:
        return (obs - self.x_mean.to(obs.device)) / self.x_std.to(obs.device)

    def raw_mean(self, obs: Any) -> Any:
        actor_out = self.actor(self.normalize_obs(obs))
        if self.policy_mode == "direct":
            return actor_out * self.y_std.to(obs.device) + self.y_mean.to(obs.device)
        if self.policy_mode == "bc_residual":
            return actor_out
        raise ValueError(f"Unsupported policy_mode {self.policy_mode!r}.")

    def value(self, obs: Any) -> Any:
        return self.critic(self.normalize_obs(obs)).squeeze(-1)

    def bound_raw_action(self, raw: Any) -> Any:
        torch, _ = _torch()
        bounds = torch.as_tensor(np.maximum(self.base_spec.residual_bounds, 1e-12), dtype=raw.dtype, device=raw.device)
        residual_bounds = torch.as_tensor(self.base_spec.residual_bounds, dtype=raw.dtype, device=raw.device)
        return residual_bounds * torch.tanh(raw / bounds)

    def bound_correction(self, raw: Any) -> Any:
        torch, _ = _torch()
        correction_bound = torch.as_tensor(
            self.correction_bound_scale * self.base_spec.residual_bounds,
            dtype=raw.dtype,
            device=raw.device,
        )
        return correction_bound * torch.tanh(raw)

    def final_action_from_raw(self, obs: Any, raw: Any, bc_action: Any | None = None) -> Any:
        torch, _ = _torch()
        if self.policy_mode == "direct":
            return self.bound_raw_action(raw)
        if self.policy_mode == "bc_residual":
            if bc_action is None:
                raise ValueError("bc_residual action composition requires bc_action.")
            residual_bounds = torch.as_tensor(self.base_spec.residual_bounds, dtype=raw.dtype, device=raw.device)
            correction = self.bound_correction(raw)
            return torch.clamp(bc_action + self.residual_alpha * correction, -residual_bounds, residual_bounds)
        raise ValueError(f"Unsupported policy_mode {self.policy_mode!r}.")

    def distribution(self, obs: Any) -> Any:
        torch, _ = _torch()
        mean = self.raw_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: Any, *, deterministic: bool = False, bc_action: Any | None = None) -> tuple[Any, Any, Any, Any]:
        dist = self.distribution(obs)
        raw = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        bounded = self.final_action_from_raw(obs, raw, bc_action=bc_action)
        value = self.value(obs)
        return raw, bounded, log_prob, value

    def evaluate_raw_actions(self, obs: Any, raw_actions: Any) -> tuple[Any, Any, Any]:
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value

    def deterministic_action_np(self, obs: np.ndarray, *, bc_action: np.ndarray | None = None) -> np.ndarray:
        torch, _ = _torch()
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(obs, dtype=float)[None, :], dtype=torch.float64)
            raw = self.raw_mean(tensor)
            bc_tensor = None
            if bc_action is not None:
                bc_tensor = torch.as_tensor(np.asarray(bc_action, dtype=float)[None, :], dtype=torch.float64)
            return self.final_action_from_raw(tensor, raw, bc_action=bc_tensor).cpu().numpy()[0]

    def to_numpy_policy(self, *, metadata: dict[str, Any]) -> ResidualRRLPolicy:
        actor0 = self.actor[0]
        actor2 = self.actor[2]
        return ResidualRRLPolicy(
            w1=actor0.weight.detach().cpu().numpy().T.copy(),
            b1=actor0.bias.detach().cpu().numpy().copy(),
            w2=actor2.weight.detach().cpu().numpy().T.copy(),
            b2=actor2.bias.detach().cpu().numpy().copy(),
            x_mean=self.x_mean.detach().cpu().numpy().copy(),
            x_std=self.x_std.detach().cpu().numpy().copy(),
            y_mean=self.y_mean.detach().cpu().numpy().copy(),
            y_std=self.y_std.detach().cpu().numpy().copy(),
            log_std=self.log_std.detach().cpu().numpy().copy(),
            base_spec=self.base_spec,
            metadata=metadata,
        )


def save_actor_critic_checkpoint(
    path: Path,
    *,
    policy: ResidualActorCriticPolicy,
    optimizer: Any,
    metadata: dict[str, Any],
) -> None:
    torch, _ = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": policy.actor.state_dict(),
            "critic": policy.critic.state_dict(),
            "log_std": policy.log_std.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "x_mean": policy.x_mean.detach().cpu(),
            "x_std": policy.x_std.detach().cpu(),
            "y_mean": policy.y_mean.detach().cpu(),
            "y_std": policy.y_std.detach().cpu(),
            "base_stiffness_spec": policy.base_spec.to_metadata(),
            "policy_mode": policy.policy_mode,
            "residual_alpha": policy.residual_alpha,
            "correction_bound_scale": policy.correction_bound_scale,
            "metadata": metadata,
        },
        path,
    )


__all__ = [
    "ResidualActorCriticPolicy",
    "ResidualRRLPolicy",
    "POLICY_MODES",
    "save_actor_critic_checkpoint",
    "save_residual_rrl_policy",
]
