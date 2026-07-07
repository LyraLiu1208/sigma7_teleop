from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Any

cache_root = Path(gettempdir()) / "stiffness_copilot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BASELINES = ("low", "high")


def load_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "summary.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing baseline summary: {summary_path}")
    rows: list[dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("baseline") not in BASELINES:
                raise ValueError(f"Invalid baseline in {summary_path}:{line_number}: {row.get('baseline')!r}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {summary_path}")
    return rows


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _save_hist(values: np.ndarray, *, title: str, xlabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(values, bins=min(16, max(6, len(values) // 3)), color="#356a8f", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_scatter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(x, y, s=28, color="#356a8f", alpha=0.82, linewidths=0.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_fixed_stiffness_baselines(run_dir: Path) -> list[Path]:
    rows = load_summary_rows(run_dir)
    output_paths: list[Path] = []
    plots_dir = run_dir / "plots"

    for baseline in BASELINES:
        baseline_rows = [row for row in rows if row["baseline"] == baseline]
        if not baseline_rows:
            continue
        baseline_dir = plots_dir / baseline
        baseline_dir.mkdir(parents=True, exist_ok=True)

        final_depth = _values(baseline_rows, "final_depth")
        final_lateral = _values(baseline_rows, "final_lateral_error")
        max_normal = _values(baseline_rows, "max_normal_force")
        max_torque = _values(baseline_rows, "max_abs_commanded_torque")

        specs = (
            (
                "final_depth_hist.png",
                lambda path: _save_hist(
                    final_depth,
                    title=f"{baseline} stiffness final insertion depth",
                    xlabel="Final depth (m)",
                    output_path=path,
                ),
            ),
            (
                "final_lateral_error_hist.png",
                lambda path: _save_hist(
                    final_lateral,
                    title=f"{baseline} stiffness final lateral error",
                    xlabel="Final lateral error (m)",
                    output_path=path,
                ),
            ),
            (
                "max_normal_force_hist.png",
                lambda path: _save_hist(
                    max_normal,
                    title=f"{baseline} stiffness max normal force",
                    xlabel="Max normal force proxy (N)",
                    output_path=path,
                ),
            ),
            (
                "lateral_vs_depth.png",
                lambda path: _save_scatter(
                    final_lateral,
                    final_depth,
                    title=f"{baseline} stiffness lateral error vs depth",
                    xlabel="Final lateral error (m)",
                    ylabel="Final depth (m)",
                    output_path=path,
                ),
            ),
            (
                "normal_force_vs_depth.png",
                lambda path: _save_scatter(
                    max_normal,
                    final_depth,
                    title=f"{baseline} stiffness normal force vs depth",
                    xlabel="Max normal force proxy (N)",
                    ylabel="Final depth (m)",
                    output_path=path,
                ),
            ),
            (
                "torque_hist.png",
                lambda path: _save_hist(
                    max_torque,
                    title=f"{baseline} stiffness commanded torque",
                    xlabel="Max absolute commanded torque (Nm)",
                    output_path=path,
                ),
            ),
        )
        for filename, plotter in specs:
            output_path = baseline_dir / filename
            plotter(output_path)
            output_paths.append(output_path)

    if not output_paths:
        raise ValueError(f"No baseline plots were generated from {run_dir}")
    return output_paths
