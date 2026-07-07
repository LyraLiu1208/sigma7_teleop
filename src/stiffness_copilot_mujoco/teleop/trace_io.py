from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mujoco
import numpy as np

from stiffness_copilot_mujoco.contact.state import extract_net_peg_hole_contact_force_world
from stiffness_copilot_mujoco.metrics.task_metrics import insertion_depth, lateral_error
from stiffness_copilot_mujoco.sim.ids import PegHoleIds


@dataclass(frozen=True)
class ContactForceSnapshot:
    contact_force_world: np.ndarray
    contact_force_norm: float
    num_contacts: int
    contact_active: bool
    raw_contact_forces_local: np.ndarray
    raw_contact_forces_world: np.ndarray
    contact_geom1: np.ndarray
    contact_geom2: np.ndarray
    contact_body1: np.ndarray
    contact_body2: np.ndarray


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stack_numeric(values: Sequence[object], *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if not values:
        return np.asarray(values)
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack([np.asarray(value) for value in values], axis=0)
    if isinstance(first, (list, tuple)):
        return np.asarray([np.asarray(value) for value in values])
    if dtype is not None:
        return np.asarray(values, dtype=dtype)
    return np.asarray(values)


def compute_contact_force_snapshot(model: mujoco.MjModel, data: mujoco.MjData, task_ids: PegHoleIds) -> ContactForceSnapshot:
    hole_geoms = set(task_ids.hole_wall_geoms)
    peg_geoms = set(task_ids.peg_geoms)
    net_force = extract_net_peg_hole_contact_force_world((model, data, task_ids))

    local_forces: list[np.ndarray] = []
    world_forces: list[np.ndarray] = []
    geom1_rows: list[int] = []
    geom2_rows: list[int] = []
    body1_rows: list[int] = []
    body2_rows: list[int] = []

    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 not in peg_geoms and geom2 not in peg_geoms:
            continue
        if geom1 not in hole_geoms and geom2 not in hole_geoms:
            continue

        force_local = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_idx, force_local)
        frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
        force_world = force_local[:3] @ frame
        if geom2 in peg_geoms:
            force_world = -force_world

        local_forces.append(force_local.copy())
        world_forces.append(force_world.copy())
        geom1_rows.append(geom1)
        geom2_rows.append(geom2)
        body1_rows.append(int(model.geom_bodyid[geom1]))
        body2_rows.append(int(model.geom_bodyid[geom2]))

    raw_contact_forces_local = np.stack(local_forces, axis=0) if local_forces else np.zeros((0, 6), dtype=float)
    raw_contact_forces_world = np.stack(world_forces, axis=0) if world_forces else np.zeros((0, 3), dtype=float)

    return ContactForceSnapshot(
        contact_force_world=np.asarray(net_force, dtype=float),
        contact_force_norm=float(np.linalg.norm(net_force)),
        num_contacts=int(len(world_forces)),
        contact_active=bool(world_forces),
        raw_contact_forces_local=raw_contact_forces_local,
        raw_contact_forces_world=raw_contact_forces_world,
        contact_geom1=np.asarray(geom1_rows, dtype=np.int32),
        contact_geom2=np.asarray(geom2_rows, dtype=np.int32),
        contact_body1=np.asarray(body1_rows, dtype=np.int32),
        contact_body2=np.asarray(body2_rows, dtype=np.int32),
    )


def pad_contact_snapshots(snapshots: Sequence[ContactForceSnapshot]) -> dict[str, np.ndarray]:
    count = int(len(snapshots))
    max_contacts = max((int(snapshot.num_contacts) for snapshot in snapshots), default=0)
    local = np.full((count, max_contacts, 6), np.nan, dtype=float)
    world = np.full((count, max_contacts, 3), np.nan, dtype=float)
    geom1 = np.full((count, max_contacts), -1, dtype=np.int32)
    geom2 = np.full((count, max_contacts), -1, dtype=np.int32)
    body1 = np.full((count, max_contacts), -1, dtype=np.int32)
    body2 = np.full((count, max_contacts), -1, dtype=np.int32)
    counts = np.zeros((count,), dtype=np.int32)
    active = np.zeros((count,), dtype=bool)
    force_norm = np.zeros((count,), dtype=float)
    force_world = np.zeros((count, 3), dtype=float)

    for idx, snapshot in enumerate(snapshots):
        n = int(snapshot.num_contacts)
        counts[idx] = n
        active[idx] = bool(snapshot.contact_active)
        force_norm[idx] = float(snapshot.contact_force_norm)
        force_world[idx] = np.asarray(snapshot.contact_force_world, dtype=float).reshape(3)
        if n <= 0:
            continue
        local[idx, :n] = np.asarray(snapshot.raw_contact_forces_local, dtype=float)
        world[idx, :n] = np.asarray(snapshot.raw_contact_forces_world, dtype=float)
        geom1[idx, :n] = np.asarray(snapshot.contact_geom1, dtype=np.int32)
        geom2[idx, :n] = np.asarray(snapshot.contact_geom2, dtype=np.int32)
        body1[idx, :n] = np.asarray(snapshot.contact_body1, dtype=np.int32)
        body2[idx, :n] = np.asarray(snapshot.contact_body2, dtype=np.int32)

    return {
        "contact_force_world": force_world,
        "contact_force_norm": force_norm,
        "num_contacts": counts,
        "contact_active": active,
        "raw_contact_forces_local": local,
        "raw_contact_forces_world": world,
        "contact_geom1": geom1,
        "contact_geom2": geom2,
        "contact_body1": body1,
        "contact_body2": body2,
    }


def compute_episode_summary(
    *,
    time: np.ndarray,
    contact_force_norm: np.ndarray,
    contact_active: np.ndarray,
    insertion_depth_series: np.ndarray,
    lateral_error_series: np.ndarray,
    final_depth_threshold: float,
    final_lateral_threshold: float,
) -> dict[str, Any]:
    force = np.asarray(contact_force_norm, dtype=float).reshape(-1)
    contact = np.asarray(contact_active, dtype=bool).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    depth = np.asarray(insertion_depth_series, dtype=float).reshape(-1)
    lateral = np.asarray(lateral_error_series, dtype=float).reshape(-1)
    if force.shape[0] != time.shape[0]:
        raise ValueError("time and contact_force_norm must have the same length.")
    if contact.shape[0] != time.shape[0]:
        raise ValueError("contact_active and contact_force_norm must have the same length.")
    if depth.shape[0] != time.shape[0]:
        raise ValueError("insertion_depth_series and contact_force_norm must have the same length.")
    if lateral.shape[0] != time.shape[0]:
        raise ValueError("lateral_error_series and contact_force_norm must have the same length.")

    duration_seconds = float(time[-1]) if time.size else 0.0
    max_force = float(np.max(force)) if force.size else 0.0
    mean_force = float(np.mean(force)) if force.size else 0.0
    median_force = float(np.median(force)) if force.size else 0.0
    p95_force = float(np.percentile(force, 95.0)) if force.size else 0.0
    contact_fraction = float(np.mean(contact)) if contact.size else 0.0
    final_depth = float(depth[-1]) if depth.size else 0.0
    final_lateral_error = float(lateral[-1]) if lateral.size else 0.0
    insertion_success = bool(final_depth >= float(final_depth_threshold) and final_lateral_error <= float(final_lateral_threshold))

    return {
        "insertion_success": insertion_success,
        "max_contact_force": max_force,
        "mean_contact_force": mean_force,
        "median_contact_force": median_force,
        "p95_contact_force": p95_force,
        "contact_fraction": contact_fraction,
        "final_depth": final_depth,
        "final_lateral_error": final_lateral_error,
        "duration_seconds": duration_seconds,
        "num_steps": int(time.shape[0]),
    }


def time_series_force_statistics(
    *,
    time: np.ndarray,
    contact_force_norm: np.ndarray,
    contact_active: np.ndarray,
) -> dict[str, float | None]:
    force = np.asarray(contact_force_norm, dtype=float).reshape(-1)
    active = np.asarray(contact_active, dtype=bool).reshape(-1)
    timestamps = np.asarray(time, dtype=float).reshape(-1)
    if force.size == 0:
        return {
            "mean_contact_force_over_time": 0.0,
            "median_contact_force_over_time": 0.0,
            "p95_contact_force_over_time": 0.0,
            "mean_contact_duration": 0.0,
            "mean_time_to_first_contact": None,
            "mean_force_integral": 0.0,
            "mean_force_impulse_proxy": 0.0,
            "mean_peak_force_time": None,
        }

    if timestamps.size >= 2:
        dt = float(np.median(np.diff(timestamps)))
    else:
        dt = 0.0
    contact_indices = np.flatnonzero(active)
    first_contact_time = float(timestamps[int(contact_indices[0])]) if contact_indices.size else None
    peak_index = int(np.argmax(force))
    force_integral = float(np.sum(force) * dt)
    force_impulse_proxy = float(np.sum(force[active]) * dt if np.any(active) else np.sum(force) * dt)
    contact_duration = float(np.sum(active.astype(float)) * dt if dt > 0.0 else np.sum(active.astype(float)))
    peak_time = float(timestamps[peak_index]) if np.any(active) else None
    return {
        "mean_contact_force_over_time": float(np.mean(force)),
        "median_contact_force_over_time": float(np.median(force)),
        "p95_contact_force_over_time": float(np.percentile(force, 95.0)),
        "mean_contact_duration": contact_duration,
        "mean_time_to_first_contact": first_contact_time,
        "mean_force_integral": force_integral,
        "mean_force_impulse_proxy": force_impulse_proxy,
        "mean_peak_force_time": peak_time,
    }


def phase_force_statistics(
    *,
    phase_id: np.ndarray,
    time: np.ndarray,
    contact_force_norm: np.ndarray,
    contact_active: np.ndarray,
) -> dict[str, dict[str, float | None]]:
    phase_id = np.asarray(phase_id, dtype=np.int32).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    force = np.asarray(contact_force_norm, dtype=float).reshape(-1)
    active = np.asarray(contact_active, dtype=bool).reshape(-1)
    labels = {0: "approach", 1: "descend", 2: "insert", 3: "final_hold"}
    result: dict[str, dict[str, float | None]] = {}
    for value, label in labels.items():
        mask = phase_id == int(value)
        if not np.any(mask):
            continue
        result[label] = time_series_force_statistics(
            time=time[mask],
            contact_force_norm=force[mask],
            contact_active=active[mask],
        )
    return result


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, payload)


def save_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


__all__ = [
    "ContactForceSnapshot",
    "compute_contact_force_snapshot",
    "compute_episode_summary",
    "pad_contact_snapshots",
    "phase_force_statistics",
    "save_json",
    "save_npz",
    "time_series_force_statistics",
]
