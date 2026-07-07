from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BadCaseThresholds:
    """Conservative first-pass thresholds for review labels."""

    raw_to_p99_spike_ratio: float = 10.0
    raw_to_filtered_spike_ratio: float = 4.0
    solver_spike_max_high_force_steps: int = 2
    jamming_min_high_force_steps: int = 10
    jamming_min_high_force_fraction: float = 0.01
    depth_stall_epsilon: float = 1e-3
    persistent_contact_min_steps: int = 10
    catastrophic_force_threshold: float = 1000.0
    low_force_success_threshold: float = 200.0


BAD_CASE_REGISTRY_FIELDS = [
    "run_id",
    "method",
    "track",
    "policy_id",
    "geometry",
    "setting_id",
    "seed",
    "episode_id",
    "initial_perturbation",
    "success",
    "low_force_success",
    "raw_max_force",
    "raw_max_force_step",
    "raw_max_force_time",
    "raw_max_force_contact_state",
    "contact_only_p95_force",
    "contact_only_p99_force",
    "all_step_p95_force",
    "all_step_p99_force",
    "filtered_max_force_short_window",
    "filtered_max_force_long_window",
    "high_force_duration_steps",
    "high_force_duration",
    "high_force_fraction",
    "raw_to_p99_ratio",
    "raw_to_filtered_ratio",
    "final_depth",
    "depth_progress_around_peak",
    "contact_duration_steps",
    "contact_duration",
    "action_jump_around_peak",
    "residual_rate_around_peak",
    "trace_path",
    "plot_path",
    "video_path",
    "automatic_classification",
    "manual_review_classification",
    "notes",
]


def _get_number(values: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
    value = values.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_int(values: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = values.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def classify_bad_case(
    metrics: Mapping[str, Any],
    trace_context: Mapping[str, Any] | None = None,
    *,
    thresholds: BadCaseThresholds | None = None,
) -> str:
    """Classify a suspicious episode for review.

    Labels are review aids, not final physical truth. Manual review should keep
    or override the automatic label after inspecting traces, plots, and videos.
    """

    cfg = thresholds or BadCaseThresholds()
    context = trace_context or {}

    raw_to_p99 = _get_number(metrics, "raw_to_p99_ratio")
    raw_to_filtered = _get_number(metrics, "raw_to_filtered_ratio")
    high_steps = _get_int(metrics, "high_force_duration_steps")
    high_fraction = _get_number(metrics, "high_force_fraction", 0.0) or 0.0
    raw_max = _get_number(metrics, "raw_max_force", 0.0) or 0.0

    success = context.get("success")
    depth_progress = _get_number(context, "depth_progress_around_peak")
    contact_steps = _get_int(context, "contact_duration_steps")
    rebound = bool(context.get("rebound", False))
    trace_incomplete = bool(context.get("trace_incomplete", False))

    spike_ratio = (
        raw_to_p99 is not None
        and raw_to_filtered is not None
        and raw_to_p99 >= cfg.raw_to_p99_spike_ratio
        and raw_to_filtered >= cfg.raw_to_filtered_spike_ratio
    )
    short_force_event = high_steps <= cfg.solver_spike_max_high_force_steps
    task_failure = success is False
    depth_stalled = depth_progress is not None and depth_progress <= cfg.depth_stall_epsilon
    persistent_contact = contact_steps >= cfg.persistent_contact_min_steps
    sustained_force = high_steps >= cfg.jamming_min_high_force_steps or high_fraction >= cfg.jamming_min_high_force_fraction
    catastrophic = raw_max >= cfg.catastrophic_force_threshold

    if trace_incomplete:
        return "mixed_unclear"

    if sustained_force and (task_failure or depth_stalled or rebound) and persistent_contact:
        return "physical_jamming_candidate"

    if spike_ratio and short_force_event and not task_failure and not rebound and not depth_stalled:
        return "solver_spike_candidate"

    if spike_ratio and (task_failure or rebound or depth_stalled):
        return "mixed_unclear"

    if catastrophic and (task_failure or depth_stalled or rebound):
        return "mixed_unclear"

    if sustained_force and persistent_contact:
        return "physical_jamming_candidate"

    return "mixed_unclear"


def make_bad_case_record(
    *,
    run_id: str,
    method: str,
    track: str,
    policy_id: str | None,
    geometry: str,
    setting_id: str | None,
    seed: int,
    episode_id: int,
    metrics: Mapping[str, Any],
    trace_context: Mapping[str, Any] | None = None,
    initial_perturbation: Mapping[str, Any] | None = None,
    trace_path: str | Path | None = None,
    plot_path: str | Path | None = None,
    video_path: str | Path | None = None,
    automatic_classification: str | None = None,
    manual_review_classification: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Create one registry row using the shared schema."""

    context = dict(trace_context or {})
    label = automatic_classification or classify_bad_case(metrics, context)
    record = {field: None for field in BAD_CASE_REGISTRY_FIELDS}
    record.update(
        {
            "run_id": run_id,
            "method": method,
            "track": track,
            "policy_id": policy_id,
            "geometry": geometry,
            "setting_id": setting_id,
            "seed": int(seed),
            "episode_id": int(episode_id),
            "initial_perturbation": dict(initial_perturbation or {}),
            "success": context.get("success"),
            "low_force_success": context.get("low_force_success"),
            "final_depth": context.get("final_depth"),
            "depth_progress_around_peak": context.get("depth_progress_around_peak"),
            "contact_duration_steps": context.get("contact_duration_steps"),
            "contact_duration": context.get("contact_duration"),
            "action_jump_around_peak": context.get("action_jump_around_peak"),
            "residual_rate_around_peak": context.get("residual_rate_around_peak"),
            "trace_path": str(trace_path) if trace_path is not None else None,
            "plot_path": str(plot_path) if plot_path is not None else None,
            "video_path": str(video_path) if video_path is not None else None,
            "automatic_classification": label,
            "manual_review_classification": manual_review_classification,
            "notes": notes,
        }
    )
    for key in (
        "raw_max_force",
        "raw_max_force_step",
        "raw_max_force_time",
        "raw_max_force_contact_state",
        "contact_only_p95_force",
        "contact_only_p99_force",
        "all_step_p95_force",
        "all_step_p99_force",
        "filtered_max_force_short_window",
        "filtered_max_force_long_window",
        "high_force_duration_steps",
        "high_force_duration",
        "high_force_fraction",
        "raw_to_p99_ratio",
        "raw_to_filtered_ratio",
    ):
        record[key] = metrics.get(key)
    return record


def write_bad_case_registry(records: list[Mapping[str, Any]], path: str | Path) -> None:
    """Write registry records as JSON, JSONL, or CSV based on extension."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".jsonl":
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
        return
    if suffix == ".csv":
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=BAD_CASE_REGISTRY_FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow({field: record.get(field) for field in BAD_CASE_REGISTRY_FIELDS})
        return
    output.write_text(json.dumps([dict(record) for record in records], indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "BAD_CASE_REGISTRY_FIELDS",
    "BadCaseThresholds",
    "classify_bad_case",
    "make_bad_case_record",
    "write_bad_case_registry",
]
