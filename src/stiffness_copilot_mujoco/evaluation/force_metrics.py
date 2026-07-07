from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ForceMetricThresholds:
    """Shared thresholds for diagnostic force summaries."""

    catastrophic_force_threshold: float = 1000.0
    low_force_success_threshold: float = 200.0
    high_force_threshold: float = 1000.0
    short_window_steps: int = 5
    long_window_steps: int = 25


def _as_force_array(force_trace: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(force_trace), dtype=float)
    if values.ndim != 1:
        raise ValueError("force_trace must be one-dimensional")
    if values.size == 0:
        return np.zeros(0, dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=np.inf, neginf=0.0)


def _as_contact_mask(contact_mask: Iterable[bool] | None, size: int) -> np.ndarray | None:
    if contact_mask is None:
        return None
    mask = np.asarray(list(contact_mask), dtype=bool)
    if mask.ndim != 1:
        raise ValueError("contact_mask must be one-dimensional")
    if mask.size != size:
        raise ValueError("contact_mask length must match force_trace length")
    return mask


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    window = max(1, min(int(window), int(values.size)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _moving_median(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    window = max(1, min(int(window), int(values.size)))
    radius = window // 2
    result = np.empty_like(values, dtype=float)
    for idx in range(values.size):
        lo = max(0, idx - radius)
        hi = min(values.size, idx + radius + 1)
        result[idx] = float(np.median(values[lo:hi]))
    return result


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def compute_force_metrics(
    force_trace: Iterable[float],
    *,
    contact_mask: Iterable[bool] | None = None,
    dt: float | None = None,
    thresholds: ForceMetricThresholds | None = None,
    step_offset: int = 0,
) -> dict[str, float | int | bool | None]:
    """Compute raw and robust episode-level force diagnostics.

    Filtering is for metric computation only. It does not change the controller,
    policy, reward, or simulator behavior.
    """

    cfg = thresholds or ForceMetricThresholds()
    force = _as_force_array(force_trace)
    mask = _as_contact_mask(contact_mask, int(force.size))

    if force.size == 0:
        return {
            "force_sample_count": 0,
            "contact_sample_count": 0,
            "raw_max_force": None,
            "raw_max_force_step": None,
            "raw_max_force_time": None,
            "raw_max_force_contact_state": None,
            "contact_only_p95_force": None,
            "contact_only_p99_force": None,
            "all_step_p95_force": None,
            "all_step_p99_force": None,
            "filtered_max_force_short_window": None,
            "filtered_max_force_long_window": None,
            "filtered_median_max_force_short_window": None,
            "filtered_median_max_force_long_window": None,
            "high_force_duration_steps": 0,
            "high_force_duration": 0.0 if dt is not None else None,
            "high_force_fraction": 0.0,
            "raw_to_p99_ratio": None,
            "raw_to_filtered_ratio": None,
            "catastrophic_force_threshold": cfg.catastrophic_force_threshold,
            "low_force_success_threshold": cfg.low_force_success_threshold,
            "high_force_threshold": cfg.high_force_threshold,
            "short_window_steps": cfg.short_window_steps,
            "long_window_steps": cfg.long_window_steps,
        }

    raw_peak_idx = int(np.argmax(force))
    raw_max = float(force[raw_peak_idx])
    contact_values = force[mask] if mask is not None else force
    contact_count = int(contact_values.size)

    short_avg = _moving_average(force, cfg.short_window_steps)
    long_avg = _moving_average(force, cfg.long_window_steps)
    short_median = _moving_median(force, cfg.short_window_steps)
    long_median = _moving_median(force, cfg.long_window_steps)
    p99_all = _safe_percentile(force, 99.0)
    filtered_short = float(np.max(short_avg))

    high_force = force >= float(cfg.high_force_threshold)
    high_steps = int(np.sum(high_force))

    return {
        "force_sample_count": int(force.size),
        "contact_sample_count": contact_count,
        "raw_max_force": raw_max,
        "raw_max_force_step": int(raw_peak_idx + step_offset),
        "raw_max_force_time": float((raw_peak_idx + step_offset) * dt) if dt is not None else None,
        "raw_max_force_contact_state": bool(mask[raw_peak_idx]) if mask is not None else None,
        "contact_only_p95_force": _safe_percentile(contact_values, 95.0),
        "contact_only_p99_force": _safe_percentile(contact_values, 99.0),
        "all_step_p95_force": _safe_percentile(force, 95.0),
        "all_step_p99_force": p99_all,
        "filtered_max_force_short_window": filtered_short,
        "filtered_max_force_long_window": float(np.max(long_avg)),
        "filtered_median_max_force_short_window": float(np.max(short_median)),
        "filtered_median_max_force_long_window": float(np.max(long_median)),
        "high_force_duration_steps": high_steps,
        "high_force_duration": float(high_steps * dt) if dt is not None else None,
        "high_force_fraction": float(high_steps / force.size),
        "raw_to_p99_ratio": _safe_ratio(raw_max, p99_all),
        "raw_to_filtered_ratio": _safe_ratio(raw_max, filtered_short),
        "catastrophic_force_threshold": cfg.catastrophic_force_threshold,
        "low_force_success_threshold": cfg.low_force_success_threshold,
        "high_force_threshold": cfg.high_force_threshold,
        "short_window_steps": cfg.short_window_steps,
        "long_window_steps": cfg.long_window_steps,
    }


def local_peak_window(
    force_trace: Iterable[float],
    *,
    center_step: int | None = None,
    radius: int = 10,
    step_offset: int = 0,
) -> list[dict[str, float | int]]:
    """Return a compact force window around a peak for registry records."""

    force = _as_force_array(force_trace)
    if force.size == 0:
        return []
    if center_step is None:
        center_idx = int(np.argmax(force))
    else:
        center_idx = int(center_step - step_offset)
    center_idx = max(0, min(center_idx, int(force.size) - 1))
    lo = max(0, center_idx - int(radius))
    hi = min(int(force.size), center_idx + int(radius) + 1)
    return [{"step": int(idx + step_offset), "force": float(force[idx])} for idx in range(lo, hi)]


def thresholds_to_dict(thresholds: ForceMetricThresholds | None = None) -> dict[str, float | int]:
    return asdict(thresholds or ForceMetricThresholds())


__all__ = [
    "ForceMetricThresholds",
    "compute_force_metrics",
    "local_peak_window",
    "thresholds_to_dict",
]
