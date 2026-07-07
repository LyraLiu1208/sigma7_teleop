from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sigma7_screening_mplconfig"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _sigma7_residual_pipeline_common import (
    DEFAULT_PIPELINE_ROOT,
    ensure_mujoco_src_on_path,
    require_safe_segment,
    scene_screening_root,
)


ensure_mujoco_src_on_path()

from stiffness_copilot_mujoco.rollouts.fixed_impedance import RolloutConfig  # noqa: E402


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_light": "#AFC6E8",
    "teal": "#42949E",
    "teal_light": "#BFE2E4",
    "red_strong": "#B64342",
    "red_light": "#E7B7B4",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

AXIS_SPECS = (
    ("Kx", 0, 0, PALETTE["blue_main"], PALETTE["blue_light"]),
    ("Ky", 1, 1, PALETTE["teal"], PALETTE["teal_light"]),
    ("Kz", 2, 2, PALETTE["red_strong"], PALETTE["red_light"]),
)


def _apply_publication_style() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.linewidth"] = 0.9
    mpl.rcParams["legend.frameon"] = False


def _json_ready(value: Any) -> Any:
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_ready(value) for key, value in row.items()})


@dataclass(frozen=True)
class ScreeningRecord:
    manifest_path: Path
    manifest: dict[str, Any]
    trace_path: Path
    trace_rows: list[dict[str, Any]]
    summary_path: Path
    summary: dict[str, Any]


def _normalized_progress(rows: list[dict[str, Any]]) -> np.ndarray:
    steps = np.asarray([int(row.get("step", 0)) for row in rows], dtype=float)
    if steps.size == 0:
        return np.zeros(0, dtype=float)
    denom = max(float(np.max(steps)), 1.0)
    return steps / denom


def _stiffness_curve(rows: list[dict[str, Any]], *, row_idx: int, col_idx: int, grid: np.ndarray) -> np.ndarray | None:
    if not rows:
        return None
    progress = _normalized_progress(rows)
    matrices: list[np.ndarray] = []
    for row in rows:
        value = row.get("stiffness_matrix_command") or row.get("stiffness_matrix_after_smoothing")
        if value is None:
            return None
        matrices.append(np.asarray(value, dtype=float).reshape(3, 3))
    matrix_array = np.stack(matrices, axis=0)
    values = matrix_array[:, row_idx, col_idx]
    order = np.argsort(progress)
    return np.interp(grid, progress[order], values[order])


def _band_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    q25 = np.percentile(values, 25.0, axis=0)
    q75 = np.percentile(values, 75.0, axis=0)
    return median, q25, q75


def _completion_time_seconds(rows: list[dict[str, Any]], *, success_depth_threshold: float) -> float | None:
    for row in rows:
        depth = row.get("depth")
        time_value = row.get("time")
        if depth is None or time_value is None:
            continue
        if float(depth) >= success_depth_threshold:
            return float(time_value)
    return None


def _force_stats(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {
            "raw_max_force": None,
            "p95_force": None,
            "p99_force": None,
            "contact_fraction": 0.0,
        }
    forces = np.asarray([float(row.get("normal_force", 0.0)) for row in rows], dtype=float)
    contact_mask = np.asarray([bool(row.get("contact_state", row.get("in_contact", False))) for row in rows], dtype=bool)
    contact_forces = forces[contact_mask]
    return {
        "raw_max_force": float(np.max(forces)) if forces.size else None,
        "p95_force": float(np.percentile(contact_forces, 95.0)) if contact_forces.size else None,
        "p99_force": float(np.percentile(contact_forces, 99.0)) if contact_forces.size else None,
        "contact_fraction": float(np.mean(contact_mask.astype(float))) if contact_mask.size else 0.0,
    }


def _load_records(
    root: Path,
    *,
    controller_kind: str,
    participant: str | None,
) -> dict[tuple[str, int], ScreeningRecord]:
    manifests = sorted(root.rglob("episode_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No episode_manifest.json files found under {root}.")
    by_pair_key: dict[tuple[str, int], ScreeningRecord] = {}
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        if str(manifest.get("controller_kind")) != controller_kind:
            continue
        manifest_participant = str(manifest.get("participant") or "unknown")
        if participant is not None and manifest_participant != participant:
            continue
        trace_path = Path(manifest["trace_path"])
        summary_path = Path(manifest["summary_path"])
        record = ScreeningRecord(
            manifest_path=manifest_path,
            manifest=manifest,
            trace_path=trace_path,
            trace_rows=_read_jsonl(trace_path),
            summary_path=summary_path,
            summary=_read_json(summary_path),
        )
        pair_key = (manifest_participant, int(manifest["episode_id"]))
        previous = by_pair_key.get(pair_key)
        if previous is None or manifest_path.stat().st_mtime >= previous.manifest_path.stat().st_mtime:
            by_pair_key[pair_key] = record
    if not by_pair_key:
        scope = f" participant={participant}" if participant is not None else ""
        raise FileNotFoundError(f"No {controller_kind} screening manifests found under {root}.{scope}")
    return by_pair_key


def _plot_force_comparison(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(6.4, 0.55 * len(rows)), 3.8))
    x = np.arange(len(rows), dtype=float)
    baseline = np.asarray([float(row["baseline_raw_max_force"]) for row in rows], dtype=float)
    residual = np.asarray([float(row["residual_raw_max_force"]) for row in rows], dtype=float)
    for idx in range(len(rows)):
        ax.plot([x[idx], x[idx]], [baseline[idx], residual[idx]], color=PALETTE["neutral_dark"], alpha=0.25, linewidth=0.8)
    ax.scatter(x, baseline, color=PALETTE["neutral_dark"], s=18, label="Baseline", zorder=3)
    ax.scatter(x, residual, color=PALETTE["blue_main"], s=18, label="Residual", zorder=3)
    ax.set_xlabel("Paired episode")
    ax.set_ylabel("Max normal force (N)")
    ax.set_title("Paired screening max force comparison", loc="left", fontsize=9.2, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_stiffness_curves_xyz(
    rows: list[dict[str, Any]],
    baseline_records: list[ScreeningRecord],
    residual_records: list[ScreeningRecord],
    output_prefix: Path,
) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=False)
    grid = np.linspace(0.0, 1.0, 120)
    for ax, (label, row_idx, col_idx, color, light) in zip(axes, AXIS_SPECS, strict=False):
        baseline_curves: list[np.ndarray] = []
        residual_curves: list[np.ndarray] = []
        for record in baseline_records:
            curve = _stiffness_curve(record.trace_rows, row_idx=row_idx, col_idx=col_idx, grid=grid)
            if curve is not None:
                baseline_curves.append(curve)
        for record in residual_records:
            curve = _stiffness_curve(record.trace_rows, row_idx=row_idx, col_idx=col_idx, grid=grid)
            if curve is not None:
                residual_curves.append(curve)
        if residual_curves:
            residual_array = np.vstack(residual_curves)
            median, q25, q75 = _band_stats(residual_array)
            ax.fill_between(grid, q25, q75, color=light, alpha=0.45, linewidth=0)
            ax.plot(grid, median, color=color, lw=2.0, label="Residual")
        if baseline_curves:
            baseline_array = np.vstack(baseline_curves)
            median, _, _ = _band_stats(baseline_array)
            ax.plot(grid, median, color=PALETTE["neutral_mid"], lw=1.6, ls="--", label="Baseline")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Normalized episode progress")
        ax.set_ylabel("Stiffness (N/m)")
        ax.set_title(label, loc="left", fontsize=9.2, fontweight="bold")
        ax.grid(True, alpha=0.18)
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("Screening stiffness command curves on xyz axes", y=1.02, fontsize=10.2, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze baseline/residual Sigma7 screening runs with xyz stiffness traces.")
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--participant", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--residual-root", type=Path, default=None)
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--success-depth-threshold", type=float, default=0.95 * RolloutConfig().insert_depth)
    args = parser.parse_args(argv)

    _apply_publication_style()
    scene = require_safe_segment(args.scene, name="scene")
    participant = None if args.participant is None else require_safe_segment(args.participant, name="participant")
    baseline_root = args.baseline_root or scene_screening_root(args.output_root, scene)
    residual_root = args.residual_root or scene_screening_root(args.output_root, scene)
    default_analysis_root = scene_screening_root(args.output_root, scene) / "analysis"
    if participant is not None:
        default_analysis_root = default_analysis_root / participant
    analysis_root = args.analysis_root or default_analysis_root
    analysis_root.mkdir(parents=True, exist_ok=True)

    baseline_by_pair = _load_records(baseline_root, controller_kind="baseline", participant=participant)
    residual_by_pair = _load_records(residual_root, controller_kind="residual", participant=participant)
    shared_pair_keys = sorted(set(baseline_by_pair) & set(residual_by_pair))
    if not shared_pair_keys:
        raise RuntimeError("No shared (participant, episode_id) values were found between baseline and residual screening roots.")

    paired_rows: list[dict[str, Any]] = []
    baseline_records: list[ScreeningRecord] = []
    residual_records: list[ScreeningRecord] = []
    for episode_order, pair_key in enumerate(shared_pair_keys):
        row_participant, episode_id = pair_key
        baseline = baseline_by_pair[pair_key]
        residual = residual_by_pair[pair_key]
        baseline_records.append(baseline)
        residual_records.append(residual)
        baseline_force = _force_stats(baseline.trace_rows)
        residual_force = _force_stats(residual.trace_rows)
        baseline_completion = _completion_time_seconds(baseline.trace_rows, success_depth_threshold=float(args.success_depth_threshold))
        residual_completion = _completion_time_seconds(residual.trace_rows, success_depth_threshold=float(args.success_depth_threshold))

        def _axis_stats(record: ScreeningRecord, row_idx: int, col_idx: int) -> tuple[float, float]:
            matrices = [
                np.asarray(row.get("stiffness_matrix_command") or row.get("stiffness_matrix_after_smoothing"), dtype=float).reshape(3, 3)
                for row in record.trace_rows
                if row.get("stiffness_matrix_command") is not None or row.get("stiffness_matrix_after_smoothing") is not None
            ]
            if not matrices:
                return 0.0, 0.0
            values = np.asarray([matrix[row_idx, col_idx] for matrix in matrices], dtype=float)
            return float(np.mean(values)), float(values[-1])

        baseline_mean_kx, baseline_final_kx = _axis_stats(baseline, 0, 0)
        baseline_mean_ky, baseline_final_ky = _axis_stats(baseline, 1, 1)
        baseline_mean_kz, baseline_final_kz = _axis_stats(baseline, 2, 2)
        residual_mean_kx, residual_final_kx = _axis_stats(residual, 0, 0)
        residual_mean_ky, residual_final_ky = _axis_stats(residual, 1, 1)
        residual_mean_kz, residual_final_kz = _axis_stats(residual, 2, 2)

        baseline_summary = baseline.summary
        residual_summary = residual.summary
        paired_rows.append(
            {
                "episode_order": int(episode_order),
                "scene": scene,
                "participant": row_participant,
                "episode_id": int(episode_id),
                "episode_spec_id": baseline.manifest.get("episode_spec_id"),
                "episode_seed": baseline.manifest.get("episode_seed"),
                "screening_seed": baseline.manifest.get("screening_seed"),
                "trajectory_family": baseline.manifest.get("trajectory_family"),
                "trajectory_family_id": baseline.manifest.get("trajectory_family_id"),
                "baseline_success": bool(baseline_summary.get("depth_reached", False)),
                "residual_success": bool(residual_summary.get("depth_reached", False)),
                "baseline_low_force_success": bool(baseline_summary.get("low_force_success", False)),
                "residual_low_force_success": bool(residual_summary.get("low_force_success", False)),
                "baseline_final_depth": float(baseline_summary.get("final_depth", 0.0)),
                "residual_final_depth": float(residual_summary.get("final_depth", 0.0)),
                "baseline_final_lateral_error": float(baseline_summary.get("final_lateral_error", 0.0)),
                "residual_final_lateral_error": float(residual_summary.get("final_lateral_error", 0.0)),
                "baseline_completion_time_s": baseline_completion,
                "residual_completion_time_s": residual_completion,
                "baseline_raw_max_force": baseline_force["raw_max_force"],
                "residual_raw_max_force": residual_force["raw_max_force"],
                "baseline_p95_force": baseline_force["p95_force"],
                "residual_p95_force": residual_force["p95_force"],
                "baseline_p99_force": baseline_force["p99_force"],
                "residual_p99_force": residual_force["p99_force"],
                "baseline_contact_fraction": baseline_force["contact_fraction"],
                "residual_contact_fraction": residual_force["contact_fraction"],
                "force_delta_residual_minus_baseline": (
                    None
                    if baseline_force["raw_max_force"] is None or residual_force["raw_max_force"] is None
                    else float(residual_force["raw_max_force"]) - float(baseline_force["raw_max_force"])
                ),
                "baseline_mean_kx": baseline_mean_kx,
                "baseline_mean_ky": baseline_mean_ky,
                "baseline_mean_kz": baseline_mean_kz,
                "residual_mean_kx": residual_mean_kx,
                "residual_mean_ky": residual_mean_ky,
                "residual_mean_kz": residual_mean_kz,
                "baseline_final_kx": baseline_final_kx,
                "baseline_final_ky": baseline_final_ky,
                "baseline_final_kz": baseline_final_kz,
                "residual_final_kx": residual_final_kx,
                "residual_final_ky": residual_final_ky,
                "residual_final_kz": residual_final_kz,
                "baseline_trace_path": str(baseline.trace_path),
                "residual_trace_path": str(residual.trace_path),
                "baseline_manifest_path": str(baseline.manifest_path),
                "residual_manifest_path": str(residual.manifest_path),
            }
        )

    baseline_success = np.asarray([bool(row["baseline_success"]) for row in paired_rows], dtype=float)
    residual_success = np.asarray([bool(row["residual_success"]) for row in paired_rows], dtype=float)
    baseline_force_vals = np.asarray([float(row["baseline_raw_max_force"]) for row in paired_rows], dtype=float)
    residual_force_vals = np.asarray([float(row["residual_raw_max_force"]) for row in paired_rows], dtype=float)
    baseline_task_time_vals = np.asarray(
        [float(row["baseline_completion_time_s"]) for row in paired_rows if row["baseline_completion_time_s"] is not None],
        dtype=float,
    )
    residual_task_time_vals = np.asarray(
        [float(row["residual_completion_time_s"]) for row in paired_rows if row["residual_completion_time_s"] is not None],
        dtype=float,
    )
    paired_participants = sorted({str(row["participant"]) for row in paired_rows})
    summary = {
        "scene": scene,
        "participant": participant,
        "participants": paired_participants,
        "baseline_root": str(baseline_root),
        "residual_root": str(residual_root),
        "analysis_root": str(analysis_root),
        "pair_count": int(len(paired_rows)),
        "paired_keys": [{"participant": p, "episode_id": episode_id} for p, episode_id in shared_pair_keys],
        "unmatched_baseline_keys": [
            {"participant": p, "episode_id": episode_id}
            for p, episode_id in sorted(set(baseline_by_pair) - set(residual_by_pair))
        ],
        "unmatched_residual_keys": [
            {"participant": p, "episode_id": episode_id}
            for p, episode_id in sorted(set(residual_by_pair) - set(baseline_by_pair))
        ],
        "baseline_success_rate": float(np.mean(baseline_success)) if baseline_success.size else 0.0,
        "residual_success_rate": float(np.mean(residual_success)) if residual_success.size else 0.0,
        "baseline_raw_max_force_mean": float(np.mean(baseline_force_vals)) if baseline_force_vals.size else 0.0,
        "residual_raw_max_force_mean": float(np.mean(residual_force_vals)) if residual_force_vals.size else 0.0,
        "force_delta_residual_minus_baseline_mean": float(np.mean(residual_force_vals - baseline_force_vals))
        if baseline_force_vals.size and residual_force_vals.size
        else 0.0,
        "baseline_task_time_mean_s": float(np.mean(baseline_task_time_vals)) if baseline_task_time_vals.size else None,
        "residual_task_time_mean_s": float(np.mean(residual_task_time_vals)) if residual_task_time_vals.size else None,
        "task_time_delta_residual_minus_baseline_mean_s": (
            float(np.mean(residual_task_time_vals) - np.mean(baseline_task_time_vals))
            if baseline_task_time_vals.size and residual_task_time_vals.size
            else None
        ),
        "outputs": {
            "paired_metrics_csv": str(analysis_root / "paired_episode_metrics.csv"),
            "analysis_summary_json": str(analysis_root / "analysis_summary.json"),
            "force_compare_prefix": str(analysis_root / "force_compare"),
            "stiffness_curves_xyz_prefix": str(analysis_root / "stiffness_curves_xyz"),
        },
    }

    _write_csv(
        analysis_root / "paired_episode_metrics.csv",
        paired_rows,
        fieldnames=list(paired_rows[0].keys()) if paired_rows else [],
    )
    _write_json(analysis_root / "analysis_summary.json", summary)
    _plot_force_comparison(paired_rows, analysis_root / "force_compare")
    _plot_stiffness_curves_xyz(paired_rows, baseline_records, residual_records, analysis_root / "stiffness_curves_xyz")

    print("")
    print("screening analysis complete")
    print(f"scene: {scene}")
    print(f"participant: {participant if participant is not None else '<all>'}")
    print(f"pair_count: {len(paired_rows)}")
    print(f"baseline_root: {baseline_root}")
    print(f"residual_root: {residual_root}")
    print(f"analysis_root: {analysis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
