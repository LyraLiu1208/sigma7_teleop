from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from stiffness_copilot_mujoco.evaluation.bad_case_registry import (
    BadCaseThresholds,
    make_bad_case_record,
    write_bad_case_registry,
)
from stiffness_copilot_mujoco.evaluation.force_metrics import ForceMetricThresholds, compute_force_metrics


@dataclass(frozen=True)
class SharedEpisodeMetadata:
    run_id: str
    method: str
    track: str
    policy_id: str | None
    geometry: str
    setting_id: str | None
    seed: int
    episode_id: int
    initial_perturbation: Mapping[str, Any] | None = None
    trace_path: str | Path | None = None
    plot_path: str | Path | None = None
    video_path: str | Path | None = None
    notes: str = ""


def _mapping_from_episode(episode: Any) -> dict[str, Any]:
    if isinstance(episode, Mapping):
        return dict(episode)
    if hasattr(episode, "to_dict"):
        return dict(episode.to_dict())
    return {}


def _trace_from_episode(episode: Any, episode_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trace = episode_data.get("full_trace")
    if trace is None and hasattr(episode, "full_trace"):
        trace = getattr(episode, "full_trace")
    if trace is None:
        return []
    return [row for row in trace if isinstance(row, Mapping)]


def _first_value(values: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in values and values[key] is not None:
            return values[key]
    return default


def resolve_depth_reached(values: Mapping[str, Any]) -> bool | None:
    return _to_bool(_first_value(values, ("depth_reached", "success", "completion_like")))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _vector_value(row: Mapping[str, Any]) -> np.ndarray | None:
    value = _first_value(row, ("action", "group_delta", "theta_delta", "residual_action"))
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        return None
    return array


def _residual_norm_value(row: Mapping[str, Any]) -> float | None:
    value = _to_float(_first_value(row, ("residual_norm", "delta_norm", "residual_action_norm")))
    if value is not None:
        return value
    vector = _vector_value(row)
    if vector is None:
        return None
    return float(np.linalg.norm(vector))


def _sampled_trace(trace: Iterable[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], float]]:
    samples: list[tuple[Mapping[str, Any], float]] = []
    for row in trace:
        force = _to_float(_first_value(row, ("normal_force", "force", "max_episode_force")))
        if force is not None:
            samples.append((row, force))
    return samples


def _trace_steps(samples: Sequence[tuple[Mapping[str, Any], float]]) -> list[int] | None:
    steps: list[int] = []
    for row, _force in samples:
        step = _to_int(row.get("step"))
        if step is None:
            return None
        steps.append(step)
    return steps


def _contact_mask(samples: Sequence[tuple[Mapping[str, Any], float]]) -> list[bool] | None:
    mask: list[bool] = []
    for row, _force in samples:
        contact = _to_bool(_first_value(row, ("contact", "in_contact", "contact_detected")))
        if contact is None:
            return None
        mask.append(contact)
    return mask


def _window_depth_progress(samples: Sequence[tuple[Mapping[str, Any], float]], center_idx: int, radius: int = 10) -> float | None:
    lo = max(0, center_idx - radius)
    hi = min(len(samples), center_idx + radius + 1)
    depths = [
        _to_float(_first_value(row, ("depth", "final_depth", "insertion_depth")))
        for row, _force in samples[lo:hi]
    ]
    depths = [value for value in depths if value is not None]
    if len(depths) < 2:
        return None
    return float(depths[-1] - depths[0])


def _action_jump(samples: Sequence[tuple[Mapping[str, Any], float]], center_idx: int, radius: int = 3) -> float | None:
    lo = max(0, center_idx - radius)
    hi = min(len(samples), center_idx + radius + 1)
    vectors = [_vector_value(row) for row, _force in samples[lo:hi]]
    jumps: list[float] = []
    previous: np.ndarray | None = None
    for vector in vectors:
        if vector is not None and previous is not None and vector.shape == previous.shape:
            jumps.append(float(np.linalg.norm(vector - previous)))
        if vector is not None:
            previous = vector
    if not jumps:
        return None
    return float(max(jumps))


def _residual_rate(samples: Sequence[tuple[Mapping[str, Any], float]], center_idx: int, radius: int = 3) -> float | None:
    center_row = samples[center_idx][0]
    direct = _to_float(_first_value(center_row, ("action_rate", "residual_rate", "residual_action_rate")))
    if direct is not None:
        return direct
    lo = max(0, center_idx - radius)
    hi = min(len(samples), center_idx + radius + 1)
    norms = [_residual_norm_value(row) for row, _force in samples[lo:hi]]
    norms = [value for value in norms if value is not None]
    if len(norms) < 2:
        return None
    return float(max(abs(norms[idx] - norms[idx - 1]) for idx in range(1, len(norms))))


def _last_trace_float(trace: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float | None:
    for row in reversed(trace):
        value = _to_float(_first_value(row, keys))
        if value is not None:
            return value
    return None


def _mean(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(numeric))


def _max(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.max(numeric))


def _force_metrics_for_trace(
    samples: Sequence[tuple[Mapping[str, Any], float]],
    *,
    dt: float | None,
    thresholds: ForceMetricThresholds | None,
) -> dict[str, Any]:
    forces = [force for _row, force in samples]
    contacts = _contact_mask(samples)
    metrics = dict(compute_force_metrics(forces, contact_mask=contacts, dt=dt, thresholds=thresholds))
    steps = _trace_steps(samples)
    if forces and steps is not None:
        peak_idx = int(np.argmax(np.asarray(forces, dtype=float)))
        metrics["raw_max_force_step"] = int(steps[peak_idx])
        metrics["raw_max_force_time"] = float(steps[peak_idx] * dt) if dt is not None else None
        if contacts is not None:
            metrics["raw_max_force_contact_state"] = bool(contacts[peak_idx])
    return metrics


def build_shared_episode_row(
    episode: Any,
    metadata: SharedEpisodeMetadata,
    *,
    dt: float | None = None,
    thresholds: ForceMetricThresholds | None = None,
) -> dict[str, Any]:
    episode_data = _mapping_from_episode(episode)
    trace = _trace_from_episode(episode, episode_data)
    samples = _sampled_trace(trace)
    metrics = _force_metrics_for_trace(samples, dt=dt, thresholds=thresholds)
    cfg = thresholds or ForceMetricThresholds()

    peak_idx: int | None = None
    if samples:
        peak_idx = int(np.argmax(np.asarray([force for _row, force in samples], dtype=float)))

    depth_reached = resolve_depth_reached(episode_data)
    raw_max_force = _to_float(metrics.get("raw_max_force"))
    low_force_success = _to_bool(_first_value(episode_data, ("low_force_success", "success_with_low_force")))
    if low_force_success is None and depth_reached is not None and raw_max_force is not None:
        low_force_success = bool(depth_reached and raw_max_force <= cfg.low_force_success_threshold)

    final_depth = _to_float(_first_value(episode_data, ("final_depth", "final_insertion_depth")))
    if final_depth is None:
        final_depth = _last_trace_float(trace, ("depth", "insertion_depth"))
    final_lateral_error = _to_float(episode_data.get("final_lateral_error"))
    if final_lateral_error is None:
        final_lateral_error = _last_trace_float(trace, ("lateral_error",))

    contact_steps = metrics.get("contact_sample_count") if _contact_mask(samples) is not None else None
    contact_duration = float(contact_steps * dt) if contact_steps is not None and dt is not None else None
    depth_progress_around_peak = _window_depth_progress(samples, peak_idx) if peak_idx is not None else None
    action_jump_around_peak = _action_jump(samples, peak_idx) if peak_idx is not None else None
    residual_rate_around_peak = _residual_rate(samples, peak_idx) if peak_idx is not None else None

    residual_norms = [_residual_norm_value(row) for row, _force in samples]
    residual_rates = [
        _to_float(_first_value(row, ("action_rate", "residual_rate", "residual_action_rate")))
        for row, _force in samples
    ]

    row: dict[str, Any] = {
        "run_id": metadata.run_id,
        "method": metadata.method,
        "track": metadata.track,
        "policy_id": metadata.policy_id,
        "geometry": metadata.geometry,
        "setting_id": metadata.setting_id,
        "seed": int(metadata.seed),
        "episode_id": int(metadata.episode_id),
        "initial_perturbation": dict(metadata.initial_perturbation or episode_data.get("perturbation") or {}),
        "depth_reached": depth_reached,
        "success": depth_reached,
        "low_force_success": low_force_success,
        "final_depth": final_depth,
        "final_lateral_error": final_lateral_error,
        "torque_saturation_count": _to_int(episode_data.get("torque_saturation_count")),
        "residual_action_norm": _mean(residual_norms),
        "max_residual_action_norm": _max(residual_norms),
        "residual_action_rate": _mean(residual_rates),
        "max_residual_action_rate": _max(residual_rates),
        "depth_progress_around_peak": depth_progress_around_peak,
        "contact_duration_steps": contact_steps,
        "contact_duration": contact_duration,
        "action_jump_around_peak": action_jump_around_peak,
        "residual_rate_around_peak": residual_rate_around_peak,
        "trace_sample_count": len(samples),
        "trace_has_contact_mask": _contact_mask(samples) is not None,
        "trace_path": str(metadata.trace_path) if metadata.trace_path is not None else None,
        "plot_path": str(metadata.plot_path) if metadata.plot_path is not None else None,
        "video_path": str(metadata.video_path) if metadata.video_path is not None else None,
        "notes": metadata.notes,
    }
    row.update(metrics)
    return _json_ready(row)


def build_bad_case_record_if_needed(
    episode_row: Mapping[str, Any],
    metadata: SharedEpisodeMetadata,
    *,
    bad_case_thresholds: BadCaseThresholds | None = None,
) -> dict[str, Any] | None:
    cfg = bad_case_thresholds or BadCaseThresholds()
    raw_max = _to_float(episode_row.get("raw_max_force")) or 0.0
    high_steps = _to_int(episode_row.get("high_force_duration_steps")) or 0
    raw_to_p99 = _to_float(episode_row.get("raw_to_p99_ratio"))
    raw_to_filtered = _to_float(episode_row.get("raw_to_filtered_ratio"))
    depth_reached = resolve_depth_reached(episode_row)
    depth_progress = _to_float(episode_row.get("depth_progress_around_peak"))
    contact_steps = _to_int(episode_row.get("contact_duration_steps")) or 0
    has_force_or_contact = raw_max > 0.0 or contact_steps > 0

    spike_like = (
        raw_to_p99 is not None
        and raw_to_filtered is not None
        and raw_to_p99 >= cfg.raw_to_p99_spike_ratio
        and raw_to_filtered >= cfg.raw_to_filtered_spike_ratio
    )
    suspicious = (
        depth_reached is False
        or raw_max >= cfg.catastrophic_force_threshold
        or high_steps > 0
        or spike_like
        or (has_force_or_contact and depth_progress is not None and depth_progress < 0.0)
    )
    if not suspicious:
        return None

    trace_context = {
        "depth_reached": depth_reached,
        "success": depth_reached,
        "low_force_success": _to_bool(episode_row.get("low_force_success")),
        "final_depth": episode_row.get("final_depth"),
        "depth_progress_around_peak": episode_row.get("depth_progress_around_peak"),
        "contact_duration_steps": episode_row.get("contact_duration_steps"),
        "contact_duration": episode_row.get("contact_duration"),
        "action_jump_around_peak": episode_row.get("action_jump_around_peak"),
        "residual_rate_around_peak": episode_row.get("residual_rate_around_peak"),
        "rebound": bool(depth_progress is not None and depth_progress < 0.0),
        "trace_incomplete": not bool(episode_row.get("trace_sample_count")),
    }
    return make_bad_case_record(
        run_id=metadata.run_id,
        method=metadata.method,
        track=metadata.track,
        policy_id=metadata.policy_id,
        geometry=metadata.geometry,
        setting_id=metadata.setting_id,
        seed=metadata.seed,
        episode_id=metadata.episode_id,
        metrics=episode_row,
        trace_context=trace_context,
        initial_perturbation=episode_row.get("initial_perturbation") or {},
        trace_path=metadata.trace_path,
        plot_path=metadata.plot_path,
        video_path=metadata.video_path,
        manual_review_classification=None,
        notes=metadata.notes,
    )


def summarize_shared_episode_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    bad_case_records: Sequence[Mapping[str, Any]],
    *,
    episode_metrics_path: str | Path,
    bad_case_registry_path: str | Path,
    thresholds: ForceMetricThresholds | None = None,
) -> dict[str, Any]:
    cfg = thresholds or ForceMetricThresholds()
    summary: dict[str, Any] = {
        "episode_count": len(episode_rows),
        "episode_metrics_path": str(episode_metrics_path),
        "bad_case_registry_path": str(bad_case_registry_path),
        "bad_case_count": len(bad_case_records),
        "bad_case_counts_by_automatic_classification": {},
        "catastrophic_force_threshold": cfg.catastrophic_force_threshold,
        "catastrophic_count": 0,
    }

    depth_reached_values = [resolve_depth_reached(row) for row in episode_rows]
    depth_reached_values = [value for value in depth_reached_values if value is not None]
    if depth_reached_values:
        summary["depth_reached_rate"] = float(np.mean(depth_reached_values))
        summary["success_rate"] = float(summary["depth_reached_rate"])

    low_force_values = [_to_bool(row.get("low_force_success")) for row in episode_rows]
    low_force_values = [value for value in low_force_values if value is not None]
    if low_force_values:
        summary["low_force_success_rate"] = float(np.mean(low_force_values))

    for record in bad_case_records:
        label = str(record.get("automatic_classification"))
        counts = summary["bad_case_counts_by_automatic_classification"]
        counts[label] = int(counts.get(label, 0)) + 1

    relevant_metrics = (
        "raw_max_force",
        "contact_only_p95_force",
        "contact_only_p99_force",
        "all_step_p95_force",
        "all_step_p99_force",
        "filtered_max_force_short_window",
        "filtered_max_force_long_window",
        "high_force_duration",
        "high_force_fraction",
        "raw_to_p99_ratio",
        "raw_to_filtered_ratio",
        "final_depth",
    )
    for metric in relevant_metrics:
        values = np.asarray([_to_float(row.get(metric)) for row in episode_rows], dtype=object)
        numeric = np.asarray([float(value) for value in values if value is not None], dtype=float)
        if numeric.size == 0:
            continue
        summary[f"{metric}_mean"] = float(np.mean(numeric))
        summary[f"{metric}_median"] = float(np.median(numeric))
        summary[f"{metric}_p95"] = float(np.percentile(numeric, 95.0))
        summary[f"{metric}_p99"] = float(np.percentile(numeric, 99.0))

    summary["catastrophic_count"] = int(
        sum(
            1
            for row in episode_rows
            if (_to_float(row.get("raw_max_force")) or 0.0) >= cfg.catastrophic_force_threshold
        )
    )

    by_method: dict[str, Any] = {}
    for method in sorted({str(row.get("method")) for row in episode_rows}):
        selected_rows = [row for row in episode_rows if str(row.get("method")) == method]
        selected_records = [record for record in bad_case_records if str(record.get("method")) == method]
        method_summary: dict[str, Any] = {
            "episode_count": len(selected_rows),
            "bad_case_count": len(selected_records),
            "bad_case_counts_by_automatic_classification": {},
            "catastrophic_count": int(
                sum(
                    1
                    for row in selected_rows
                    if (_to_float(row.get("raw_max_force")) or 0.0) >= cfg.catastrophic_force_threshold
                )
            ),
        }
        method_depth_reached_values = [resolve_depth_reached(row) for row in selected_rows]
        method_depth_reached_values = [value for value in method_depth_reached_values if value is not None]
        if method_depth_reached_values:
            method_summary["depth_reached_rate"] = float(np.mean(method_depth_reached_values))
            method_summary["success_rate"] = float(method_summary["depth_reached_rate"])

        method_low_force_values = [_to_bool(row.get("low_force_success")) for row in selected_rows]
        method_low_force_values = [value for value in method_low_force_values if value is not None]
        if method_low_force_values:
            method_summary["low_force_success_rate"] = float(np.mean(method_low_force_values))

        for record in selected_records:
            label = str(record.get("automatic_classification"))
            counts = method_summary["bad_case_counts_by_automatic_classification"]
            counts[label] = int(counts.get(label, 0)) + 1

        for metric in relevant_metrics:
            values = np.asarray([_to_float(row.get(metric)) for row in selected_rows], dtype=object)
            numeric = np.asarray([float(value) for value in values if value is not None], dtype=float)
            if numeric.size == 0:
                continue
            method_summary[f"{metric}_mean"] = float(np.mean(numeric))
            method_summary[f"{metric}_median"] = float(np.median(numeric))
            method_summary[f"{metric}_p95"] = float(np.percentile(numeric, 95.0))
            method_summary[f"{metric}_p99"] = float(np.percentile(numeric, 99.0))
        by_method[method] = method_summary
    summary["by_method"] = by_method
    return _json_ready(summary)


def write_shared_protocol_outputs(
    episodes: Iterable[tuple[Any, SharedEpisodeMetadata]],
    output_dir: str | Path,
    *,
    dt: float | None = None,
    thresholds: ForceMetricThresholds | None = None,
    bad_case_thresholds: BadCaseThresholds | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episode_metrics_path = output / "episode_metrics.jsonl"
    bad_case_registry_path = output / "bad_case_registry.jsonl"
    shared_summary_path = output / "shared_summary.json"

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for episode, metadata in episodes:
        row = build_shared_episode_row(episode, metadata, dt=dt, thresholds=thresholds)
        rows.append(row)
        record = build_bad_case_record_if_needed(row, metadata, bad_case_thresholds=bad_case_thresholds)
        if record is not None:
            records.append(_json_ready(record))

    _write_jsonl(rows, episode_metrics_path)
    write_bad_case_registry(records, bad_case_registry_path)
    summary = summarize_shared_episode_rows(
        rows,
        records,
        episode_metrics_path=episode_metrics_path,
        bad_case_registry_path=bad_case_registry_path,
        thresholds=thresholds,
    )
    summary["shared_summary_path"] = str(shared_summary_path)
    shared_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output),
        "episode_metrics_path": str(episode_metrics_path),
        "bad_case_registry_path": str(bad_case_registry_path),
        "shared_summary_path": str(shared_summary_path),
        "episode_count": len(rows),
        "bad_case_count": len(records),
    }


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(dict(row)), sort_keys=True) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "SharedEpisodeMetadata",
    "build_bad_case_record_if_needed",
    "build_shared_episode_row",
    "summarize_shared_episode_rows",
    "write_shared_protocol_outputs",
]
