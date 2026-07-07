from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stiffness_copilot_mujoco.learning.residual_dataset import validate_residual_dataset
from stiffness_copilot_mujoco.learning.residual_stiffness import BaseStiffnessSpec, save_residual_policy
from stiffness_copilot_mujoco.sim.scene import ROOT


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "models" / "residual_bc"


def _load_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"]))


def _standardize(train: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (value - mean) / std, mean, std


def _sample_weights(
    *,
    contact_state: np.ndarray,
    contact_force_world: np.ndarray,
    action: np.ndarray,
    train_idx: np.ndarray,
    value_idx: np.ndarray,
    contact_weight: float,
    force_weight: float,
    insert_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    contact = np.asarray(contact_state[:, 0] > 0.5, dtype=bool)
    force_norm = np.linalg.norm(contact_force_world, axis=1)
    force_scale = float(np.percentile(force_norm[train_idx], 95.0))
    if force_scale < 1e-9:
        force_scale = 1.0
    normalized_force = np.clip(force_norm / force_scale, 0.0, 1.0)
    phase_id = np.rint(action[:, -1]).astype(np.int32)
    insert_phase = (phase_id == 2) | (phase_id == 3)
    weights = (
        1.0
        + float(contact_weight) * contact.astype(float)
        + float(force_weight) * normalized_force
        + float(insert_weight) * insert_phase.astype(float)
    )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("sample weights must be finite and positive.")
    return weights[value_idx].astype(np.float64), contact[value_idx]


def _loss_and_grad(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    sample_weight: np.ndarray,
    loss: str,
    huber_delta: float,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    free_mask: np.ndarray,
    free_delta_penalty: float,
) -> tuple[float, np.ndarray]:
    diff = pred - target
    if loss == "mse":
        element_loss = diff**2
        element_grad = 2.0 * diff
    elif loss == "huber":
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive.")
        abs_diff = np.abs(diff)
        quadratic = abs_diff <= huber_delta
        element_loss = np.where(quadratic, 0.5 * diff**2, huber_delta * (abs_diff - 0.5 * huber_delta))
        element_grad = np.where(quadratic, diff, huber_delta * np.sign(diff))
    else:
        raise ValueError(f"Unsupported loss {loss!r}.")

    denom = float(np.sum(sample_weight) * pred.shape[1])
    objective = float(np.sum(sample_weight[:, None] * element_loss) / denom)
    grad = sample_weight[:, None] * element_grad / denom

    if free_delta_penalty > 0.0 and np.any(free_mask):
        pred_raw = pred * y_std + y_mean
        free_pred = pred_raw[free_mask]
        penalty_denom = float(free_pred.size)
        objective += float(free_delta_penalty) * float(np.sum(free_pred**2) / penalty_denom)
        grad_penalty_raw = np.zeros_like(pred_raw)
        grad_penalty_raw[free_mask] = 2.0 * float(free_delta_penalty) * pred_raw[free_mask] / penalty_denom
        grad += grad_penalty_raw * y_std
    return objective, grad


def train_residual_bc(
    dataset: Path,
    output: Path,
    *,
    epochs: int = 80,
    hidden_dim: int | None = None,
    hidden_dims: tuple[int, ...] | list[int] | None = None,
    lr: float = 0.01,
    seed: int = 0,
    validation_patience: int | None = None,
    loss: str = "mse",
    huber_delta: float = 1.0,
    contact_weight: float = 0.0,
    force_weight: float = 0.0,
    insert_weight: float = 0.0,
    free_delta_penalty: float = 0.0,
    failure_episode_weight: float = 1.0,
    label_gate_path: Path | None = None,
) -> dict:
    metadata = validate_residual_dataset(dataset)
    if hidden_dims is None:
        hidden_dims = (hidden_dim if hidden_dim is not None else 64,)
    hidden_dims = tuple(int(value) for value in hidden_dims)
    if not hidden_dims or any(value <= 0 for value in hidden_dims):
        raise ValueError("hidden_dims must contain positive layer widths.")
    if failure_episode_weight < 1.0:
        raise ValueError("failure_episode_weight must be >= 1.0.")
    if metadata.get("setting_id") == "circle_calibrated_v1":
        if label_gate_path is None:
            raise ValueError("circle_calibrated_v1 training requires a passing --label-gate JSON.")
        gate = json.loads(label_gate_path.read_text(encoding="utf-8"))
        if not gate.get("training_allowed", False):
            raise ValueError(f"Label gate does not allow training: {gate.get('disposition')}")
        if Path(str(gate.get("dataset"))).resolve() != dataset.resolve():
            raise ValueError("Label gate dataset does not match the requested training dataset.")
    rng = np.random.default_rng(seed)
    with np.load(dataset, allow_pickle=False) as data:
        x = data["task_state"].astype(np.float64)
        y = data["residual_group_target"].astype(np.float64)
        episode_id = data["episode_id"].astype(np.int64)
        contact_state = data["contact_state"].astype(np.float64)
        contact_force_world = data["contact_force_world"].astype(np.float64)
        action = data["action"].astype(np.float64)
        if "train_episode_ids" in data and "val_episode_ids" in data:
            train_episodes = data["train_episode_ids"].astype(np.int64)
            val_episodes = data["val_episode_ids"].astype(np.int64)
        else:
            episodes = rng.permutation(np.unique(episode_id))
            split = max(1, int(0.8 * len(episodes)))
            train_episodes = episodes[:split]
            val_episodes = episodes[split:] if split < len(episodes) else episodes[:1]
        episode_outcomes = None
        if "episode_summary_id" in data and "episode_success" in data:
            episode_outcomes = {
                int(ep): bool(success)
                for ep, success in zip(data["episode_summary_id"], data["episode_success"], strict=True)
            }

    train_idx = np.flatnonzero(np.isin(episode_id, train_episodes))
    val_idx = np.flatnonzero(np.isin(episode_id, val_episodes))
    x_train_raw, y_train_raw = x[train_idx], y[train_idx]
    x_val_raw, y_val_raw = x[val_idx], y[val_idx]
    train_weight, train_contact = _sample_weights(
        contact_state=contact_state,
        contact_force_world=contact_force_world,
        action=action,
        train_idx=train_idx,
        value_idx=train_idx,
        contact_weight=contact_weight,
        force_weight=force_weight,
        insert_weight=insert_weight,
    )
    val_weight, val_contact = _sample_weights(
        contact_state=contact_state,
        contact_force_world=contact_force_world,
        action=action,
        train_idx=train_idx,
        value_idx=val_idx,
        contact_weight=contact_weight,
        force_weight=force_weight,
        insert_weight=insert_weight,
    )
    if failure_episode_weight > 1.0:
        if episode_outcomes is None:
            raise ValueError("Failure-aware weighting requires true episode outcomes in the dataset.")
        failure_mask = np.asarray([not episode_outcomes[int(ep)] for ep in episode_id[train_idx]], dtype=bool)
        train_weight = train_weight * np.where(failure_mask, float(failure_episode_weight), 1.0)

    x_train, x_mean, x_std = _standardize(x_train_raw, x_train_raw)
    x_val = (x_val_raw - x_mean) / x_std
    y_train, y_mean, y_std = _standardize(y_train_raw, y_train_raw)
    y_val = (y_val_raw - y_mean) / y_std

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    layer_dims = (input_dim, *hidden_dims, output_dim)
    weights = [
        rng.normal(0.0, np.sqrt(2.0 / layer_dims[idx]), size=(layer_dims[idx], layer_dims[idx + 1]))
        for idx in range(len(layer_dims) - 1)
    ]
    biases = [np.zeros(layer_dims[idx + 1], dtype=float) for idx in range(len(layer_dims) - 1)]
    moments_w = [np.zeros_like(value) for value in weights]
    moments_b = [np.zeros_like(value) for value in biases]
    velocities_w = [np.zeros_like(value) for value in weights]
    velocities_b = [np.zeros_like(value) for value in biases]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def forward(xv: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        activations = [xv]
        value = xv
        for weight, bias in zip(weights[:-1], biases[:-1], strict=True):
            value = np.maximum(0.0, value @ weight + bias)
            activations.append(value)
        return activations, value @ weights[-1] + biases[-1]

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = [value.copy() for value in weights]
    best_biases = [value.copy() for value in biases]
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        activations, pred = forward(x_train)
        train_loss, grad_pred = _loss_and_grad(
            pred,
            y_train,
            sample_weight=train_weight,
            loss=loss,
            huber_delta=huber_delta,
            y_mean=y_mean,
            y_std=y_std,
            free_mask=~train_contact,
            free_delta_penalty=free_delta_penalty,
        )
        grad_weights = [np.zeros_like(value) for value in weights]
        grad_biases = [np.zeros_like(value) for value in biases]
        grad_value = grad_pred
        for layer_idx in range(len(weights) - 1, -1, -1):
            grad_weights[layer_idx] = activations[layer_idx].T @ grad_value
            grad_biases[layer_idx] = grad_value.sum(axis=0)
            if layer_idx > 0:
                grad_value = grad_value @ weights[layer_idx].T
                grad_value[activations[layer_idx] <= 0.0] = 0.0
        for idx in range(len(weights)):
            moments_w[idx][:] = beta1 * moments_w[idx] + (1.0 - beta1) * grad_weights[idx]
            velocities_w[idx][:] = beta2 * velocities_w[idx] + (1.0 - beta2) * (grad_weights[idx] ** 2)
            moments_b[idx][:] = beta1 * moments_b[idx] + (1.0 - beta1) * grad_biases[idx]
            velocities_b[idx][:] = beta2 * velocities_b[idx] + (1.0 - beta2) * (grad_biases[idx] ** 2)
            weights[idx][:] -= lr * (moments_w[idx] / (1.0 - beta1**epoch)) / (
                np.sqrt(velocities_w[idx] / (1.0 - beta2**epoch)) + eps
            )
            biases[idx][:] -= lr * (moments_b[idx] / (1.0 - beta1**epoch)) / (
                np.sqrt(velocities_b[idx] / (1.0 - beta2**epoch)) + eps
            )
        _, val_pred = forward(x_val)
        val_loss, _ = _loss_and_grad(
            val_pred,
            y_val,
            sample_weight=val_weight,
            loss=loss,
            huber_delta=huber_delta,
            y_mean=y_mean,
            y_std=y_std,
            free_mask=~val_contact,
            free_delta_penalty=free_delta_penalty,
        )
        history.append((epoch, train_loss, val_loss))
        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = [value.copy() for value in weights]
            best_biases = [value.copy() for value in biases]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if validation_patience is not None and epochs_without_improvement >= validation_patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}")
            break

    weights = best_weights
    biases = best_biases

    base_spec = BaseStiffnessSpec.from_metadata(metadata["base_stiffness_spec"])
    train_metadata = {
        "schema_version": "residual_bc_policy_v2",
        "architecture_schema_version": 2,
        "dataset": str(dataset),
        "scene": metadata["scene"],
        "setting_id": metadata.get("setting_id") or metadata.get("difficulty", "unknown"),
        "base_profile": metadata["base_profile"],
        "input_dim": int(input_dim),
        "target_dim": int(output_dim),
        "max_epochs": epochs,
        "epochs_completed": len(history),
        "validation_patience": validation_patience,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "hidden_dims": list(hidden_dims),
        "lr": lr,
        "seed": seed,
        "loss": loss,
        "huber_delta": huber_delta,
        "contact_weight": contact_weight,
        "force_weight": force_weight,
        "insert_weight": insert_weight,
        "free_delta_penalty": free_delta_penalty,
        "failure_episode_weight": failure_episode_weight,
        "train_episode_ids": train_episodes.tolist(),
        "val_episode_ids": val_episodes.tolist(),
        "label_gate": str(label_gate_path) if label_gate_path is not None else None,
    }
    save_residual_policy(
        output,
        weights=weights,
        biases=biases,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        history=np.asarray(history, dtype=float),
        base_spec=base_spec,
        metadata=train_metadata,
    )
    result = {
        "dataset": str(dataset),
        "output": str(output),
        "samples": int(x.shape[0]),
        "train_episodes": int(len(train_episodes)),
        "val_episodes": int(len(val_episodes)),
        "input_dim": int(input_dim),
        "target_dim": int(output_dim),
        "initial_train_loss": history[0][1],
        "final_train_loss": history[-1][1],
        "final_val_loss": history[-1][2],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "hidden_dims": list(hidden_dims),
        "loss": loss,
        "contact_weight": contact_weight,
        "force_weight": force_weight,
        "insert_weight": insert_weight,
        "free_delta_penalty": free_delta_penalty,
        "failure_episode_weight": failure_episode_weight,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not np.isfinite(history[-1][1]) or history[-1][1] > history[0][1]:
        raise RuntimeError("Training loss did not decrease.")
    return result


__all__ = ["DEFAULT_OUTPUT_ROOT", "train_residual_bc"]
