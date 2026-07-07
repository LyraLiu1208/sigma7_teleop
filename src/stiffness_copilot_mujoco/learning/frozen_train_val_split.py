from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FROZEN_TRAIN_VAL_SPLIT_SCHEMA_VERSION = "track_a_frozen_train_val_split_v1"


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


@dataclass(frozen=True)
class FrozenTrainValSplit:
    dataset_path: str
    collection_seed: int
    split_seed: int
    train_episode_ids: np.ndarray
    val_episode_ids: np.ndarray
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FROZEN_TRAIN_VAL_SPLIT_SCHEMA_VERSION,
            "dataset_path": self.dataset_path,
            "collection_seed": int(self.collection_seed),
            "split_seed": int(self.split_seed),
            "train_episode_ids": self.train_episode_ids.astype(int).tolist(),
            "val_episode_ids": self.val_episode_ids.astype(int).tolist(),
            "train_episode_count": int(self.train_episode_ids.size),
            "val_episode_count": int(self.val_episode_ids.size),
            "metadata": _json_ready(self.metadata),
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_frozen_train_val_split(path: Path) -> FrozenTrainValSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != FROZEN_TRAIN_VAL_SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported frozen train/val split schema {schema_version!r}; expected {FROZEN_TRAIN_VAL_SPLIT_SCHEMA_VERSION!r}."
        )
    train_episode_ids = np.asarray(payload.get("train_episode_ids", []), dtype=np.int64)
    val_episode_ids = np.asarray(payload.get("val_episode_ids", []), dtype=np.int64)
    if train_episode_ids.ndim != 1 or val_episode_ids.ndim != 1:
        raise ValueError("train_episode_ids and val_episode_ids must be one-dimensional arrays.")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("Frozen train/val split metadata must be a mapping.")
    return FrozenTrainValSplit(
        dataset_path=str(payload.get("dataset_path") or ""),
        collection_seed=int(payload.get("collection_seed", 0)),
        split_seed=int(payload.get("split_seed", 0)),
        train_episode_ids=train_episode_ids,
        val_episode_ids=val_episode_ids,
        metadata=metadata,
    )


__all__ = [
    "FROZEN_TRAIN_VAL_SPLIT_SCHEMA_VERSION",
    "FrozenTrainValSplit",
    "load_frozen_train_val_split",
]
