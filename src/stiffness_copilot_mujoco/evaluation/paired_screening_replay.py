from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stiffness_copilot_mujoco.episodes.episode_spec import EpisodeSpec, load_episode_specs_jsonl, select_episode_spec


@dataclass(frozen=True)
class PairedScreeningReplaySelection:
    episode_specs: list[EpisodeSpec]
    selected_rows: list[dict[str, Any]]
    eligible_rows: list[dict[str, Any]]
    rejected_rows: list[dict[str, Any]]
    selection_summary: dict[str, Any]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def _as_float(value: object, *, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: object, *, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(value)


def load_collection_episode_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Collection episodes CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No episode rows found in {path}")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                "episode_id": _as_int(row.get("episode_id")),
                "episode_spec_id": str(row.get("episode_spec_id") or ""),
                "trajectory_source": str(row.get("trajectory_source") or ""),
                "trajectory_family": str(row.get("trajectory_family") or ""),
                "trajectory_family_id": _as_int(row.get("trajectory_family_id")),
                "sample_count": _as_int(row.get("sample_count")),
                "total_steps": _as_int(row.get("total_steps")),
                "success": _as_bool(row.get("success")),
                "final_depth": _as_float(row.get("final_depth")),
                "final_lateral_error": _as_float(row.get("final_lateral_error")),
                "full_step_max_force": _as_float(row.get("full_step_max_force")),
                "sampled_max_force": _as_float(row.get("sampled_max_force")),
                "capture_ratio": _as_float(row.get("capture_ratio")),
                "contact_count": _as_int(row.get("contact_count")),
                "contact_onset_step": _as_int(row.get("contact_onset_step")),
                "max_abs_commanded_torque": _as_float(row.get("max_abs_commanded_torque")),
                "torque_saturation_count": _as_int(row.get("torque_saturation_count")),
                "episode_complete": _as_bool(row.get("episode_complete")),
                "arrays_finite": _as_bool(row.get("arrays_finite")),
                "solver_spike_suspicious": _as_bool(row.get("solver_spike_suspicious")),
                "label_eligible": _as_bool(row.get("label_eligible")),
                "exclusion_reason": str(row.get("exclusion_reason") or ""),
                "clearance_delta": _as_float(row.get("clearance_delta")),
                "perturbation_hole_x": _as_float(row.get("perturbation_hole_x")),
                "perturbation_hole_y": _as_float(row.get("perturbation_hole_y")),
                "perturbation_hole_yaw": _as_float(row.get("perturbation_hole_yaw")),
                "perturbation_clearance_delta": _as_float(row.get("perturbation_clearance_delta")),
                "perturbation_friction_scale": _as_float(row.get("perturbation_friction_scale")),
                "perturbation_peg_tilt_x": _as_float(row.get("perturbation_peg_tilt_x")),
                "perturbation_peg_tilt_y": _as_float(row.get("perturbation_peg_tilt_y")),
                "perturbation_teleop_noise_xy_amplitude": _as_float(row.get("perturbation_teleop_noise_xy_amplitude")),
                "perturbation_teleop_noise_cycles": _as_float(row.get("perturbation_teleop_noise_cycles")),
            }
        )
    return normalized


def _row_contact_fraction(row: dict[str, Any]) -> float:
    total_steps = max(int(row["total_steps"]), 1)
    return float(row["contact_count"]) / float(total_steps)


def _row_is_contact_bearing(row: dict[str, Any], *, min_contact_fraction: float) -> bool:
    if not bool(row["label_eligible"]):
        return False
    if bool(row["solver_spike_suspicious"]):
        return False
    if bool(row["full_step_max_force"] >= 1000.0):
        return False
    if str(row["exclusion_reason"]).strip():
        return False
    if int(row["contact_count"]) <= 0:
        return False
    if _row_contact_fraction(row) < float(min_contact_fraction):
        return False
    return True


def select_contact_bearing_episode_specs(
    *,
    episode_specs_path: Path,
    collection_episodes_csv: Path,
    episode_spec_id: str | None = None,
    max_episodes: int | None = None,
    min_contact_fraction: float = 0.10,
) -> PairedScreeningReplaySelection:
    episode_specs = load_episode_specs_jsonl(episode_specs_path)
    if not episode_specs:
        raise ValueError(f"No EpisodeSpec entries found in {episode_specs_path}")
    episode_rows = load_collection_episode_rows(collection_episodes_csv)
    rows_by_episode_id = {int(row["episode_id"]): row for row in episode_rows}
    specs_by_episode_id = {int(spec.episode_id): spec for spec in episode_specs}
    missing_ids = sorted(set(rows_by_episode_id) - set(specs_by_episode_id))
    if missing_ids:
        raise ValueError(f"Collection CSV has episode_ids not present in EpisodeSpec JSONL: {missing_ids[:8]}")

    eligible_pairs: list[tuple[EpisodeSpec, dict[str, Any]]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in sorted(episode_rows, key=lambda item: int(item["episode_id"])):
        spec = specs_by_episode_id[int(row["episode_id"])]
        if spec.episode_spec_id != str(row["episode_spec_id"]):
            raise ValueError(
                "Collection CSV episode_spec_id does not match EpisodeSpec JSONL provenance "
                f"for episode_id {row['episode_id']}: csv={row['episode_spec_id']!r}, spec={spec.episode_spec_id!r}."
            )
        row_copy = dict(row)
        row_copy["contact_fraction"] = _row_contact_fraction(row_copy)
        row_copy["is_contact_bearing"] = _row_is_contact_bearing(row_copy, min_contact_fraction=min_contact_fraction)
        if row_copy["is_contact_bearing"]:
            eligible_pairs.append((spec, row_copy))
        else:
            rejected_rows.append(row_copy)

    if episode_spec_id is not None:
        selected_spec = select_episode_spec(episode_specs, episode_spec_id=episode_spec_id)
        selected_row = rows_by_episode_id.get(int(selected_spec.episode_id))
        if selected_row is None:
            raise KeyError(f"EpisodeSpec id {episode_spec_id!r} was not found in collection CSV provenance.")
        selected_row = dict(selected_row)
        selected_row["contact_fraction"] = _row_contact_fraction(selected_row)
        selected_row["is_contact_bearing"] = _row_is_contact_bearing(selected_row, min_contact_fraction=min_contact_fraction)
        if not selected_row["is_contact_bearing"]:
            raise ValueError(
                f"EpisodeSpec id {episode_spec_id!r} does not satisfy the paired screening contact-bearing filter."
            )
        selected_pairs = [(selected_spec, selected_row)]
    else:
        eligible_pairs.sort(
            key=lambda item: (
                -float(item[1]["contact_fraction"]),
                int(item[1]["episode_id"]),
                str(item[0].episode_spec_id),
            )
        )
        if max_episodes is None:
            selected_pairs = eligible_pairs
        else:
            selected_pairs = eligible_pairs[: int(max_episodes)]

    selected_specs = [spec for spec, _ in selected_pairs]
    selected_rows = [row for _, row in selected_pairs]
    selection_summary = {
        "episode_specs_path": str(episode_specs_path),
        "collection_episodes_csv": str(collection_episodes_csv),
        "episode_count": int(len(selected_specs)),
        "eligible_episode_count": int(len(eligible_pairs)),
        "rejected_episode_count": int(len(rejected_rows)),
        "min_contact_fraction": float(min_contact_fraction),
        "selected_episode_ids": [int(spec.episode_id) for spec in selected_specs],
        "selected_episode_spec_ids": [spec.episode_spec_id for spec in selected_specs],
        "selected_trajectory_families": sorted({spec.trajectory_family for spec in selected_specs}),
        "selected_trajectory_sources": sorted({spec.trajectory_source for spec in selected_specs}),
        "selected_actual_hole_xy": [spec.actual_hole_xy.tolist() for spec in selected_specs],
        "selected_trajectory_source": "episode_spec_replay",
        "selected_contact_condition_names": sorted({spec.contact_condition_name for spec in selected_specs}),
    }
    return PairedScreeningReplaySelection(
        episode_specs=selected_specs,
        selected_rows=selected_rows,
        eligible_rows=eligible_pairs_to_rows(eligible_pairs),
        rejected_rows=rejected_rows,
        selection_summary=selection_summary,
    )


def eligible_pairs_to_rows(pairs: list[tuple[EpisodeSpec, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for _, row in pairs]
