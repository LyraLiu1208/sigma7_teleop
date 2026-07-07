from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stiffness_copilot_mujoco.sim.scene import ROOT

from stiffness_copilot_mujoco.learning.dataset_schema import validate_learning_dataset


def _load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "metadata"}
        metadata = json.loads(str(data["metadata"]))
    validate_learning_dataset(arrays, metadata)
    features = arrays["task_state"]
    target = arrays["stiffness_cholesky_target"]
    episode_id = arrays["episode_id"]
    return features.astype(np.float64), target.astype(np.float64), episode_id.astype(np.int64), metadata


def _standardize(train: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (value - mean) / std, mean, std


def train(dataset: Path, output: Path, *, epochs: int, hidden_dim: int, lr: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x, y, episode_id, metadata = _load_dataset(dataset)
    episodes = rng.permutation(np.unique(episode_id))
    split = max(1, int(0.8 * len(episodes)))
    train_episodes = episodes[:split]
    val_episodes = episodes[split:] if split < len(episodes) else episodes[:1]
    train_idx = np.flatnonzero(np.isin(episode_id, train_episodes))
    val_idx = np.flatnonzero(np.isin(episode_id, val_episodes))
    x_train_raw, y_train_raw = x[train_idx], y[train_idx]
    x_val_raw, y_val_raw = x[val_idx], y[val_idx]

    x_train, x_mean, x_std = _standardize(x_train_raw, x_train_raw)
    x_val = (x_val_raw - x_mean) / x_std
    y_train, y_mean, y_std = _standardize(y_train_raw, y_train_raw)
    y_val = (y_val_raw - y_mean) / y_std

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=float)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim))
    b2 = np.zeros(output_dim, dtype=float)

    mw1 = np.zeros_like(w1)
    vw1 = np.zeros_like(w1)
    mb1 = np.zeros_like(b1)
    vb1 = np.zeros_like(b1)
    mw2 = np.zeros_like(w2)
    vw2 = np.zeros_like(w2)
    mb2 = np.zeros_like(b2)
    vb2 = np.zeros_like(b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def forward(xv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.maximum(0.0, xv @ w1 + b1)
        pred = hidden @ w2 + b2
        return hidden, pred

    history = []
    for epoch in range(1, epochs + 1):
        hidden, pred = forward(x_train)
        diff = pred - y_train
        train_loss = float(np.mean(diff**2))
        grad_pred = 2.0 * diff / diff.size
        grad_w2 = hidden.T @ grad_pred
        grad_b2 = grad_pred.sum(axis=0)
        grad_hidden = grad_pred @ w2.T
        grad_hidden[hidden <= 0.0] = 0.0
        grad_w1 = x_train.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)

        grads = ((grad_w1, "w1"), (grad_b1, "b1"), (grad_w2, "w2"), (grad_b2, "b2"))
        params = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
        m = {"w1": mw1, "b1": mb1, "w2": mw2, "b2": mb2}
        v = {"w1": vw1, "b1": vb1, "w2": vw2, "b2": vb2}
        for grad, name in grads:
            m[name][:] = beta1 * m[name] + (1.0 - beta1) * grad
            v[name][:] = beta2 * v[name] + (1.0 - beta2) * (grad**2)
            mhat = m[name] / (1.0 - beta1**epoch)
            vhat = v[name] / (1.0 - beta2**epoch)
            params[name][:] -= lr * mhat / (np.sqrt(vhat) + eps)

        _, val_pred = forward(x_val)
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        history.append((epoch, train_loss, val_loss))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        history=np.asarray(history, dtype=float),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    result = {
        "dataset": str(dataset),
        "output": str(output),
        "samples": int(x.shape[0]),
        "train_episodes": int(len(train_episodes)),
        "val_episodes": int(len(val_episodes)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "input_dim": int(input_dim),
        "target_dim": int(output_dim),
        "initial_train_loss": history[0][1],
        "final_train_loss": history[-1][1],
        "final_val_loss": history[-1][2],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not np.isfinite(history[-1][1]) or history[-1][1] >= history[0][1]:
        raise RuntimeError("Training loss did not decrease.")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train minimal numpy MLP supervised baseline.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "models" / "supervised_baseline" / "supervised_baseline.npz")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    train(args.dataset, args.output, epochs=args.epochs, hidden_dim=args.hidden_dim, lr=args.lr, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
