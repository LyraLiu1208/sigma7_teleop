#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import select
import sys
import time
from collections import deque
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FORCE_FILTER_ORDER = 4
DEFAULT_FORCE_FILTER_CUTOFF_HZ = 40.0
DEFAULT_FORCE_FILTER_FS_HZ = 500.0

COLORS = {
    "kx": "#3670ff",
    "ky": "#50be6e",
    "kz": "#e6823c",
    "force": "#50dcff",
    "text": "#f2f2f2",
    "muted": "#b8b8b8",
    "grid": "#3d3d3d",
    "background": "#171917",
}


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def _configure_window(fig: Any, *, title: str, x: int | None, y: int | None, width: int, height: int) -> None:
    manager = plt.get_current_fig_manager()
    try:
        manager.set_window_title(title)
    except Exception:
        pass
    try:
        # Qt backends.
        if x is not None and y is not None and hasattr(manager, "window"):
            manager.window.setGeometry(int(x), int(y), int(width), int(height))
            return
    except Exception:
        pass
    try:
        # Tk backends.
        if x is not None and y is not None and hasattr(manager, "window"):
            manager.window.wm_geometry(f"{int(width)}x{int(height)}+{int(x)}+{int(y)}")
    except Exception:
        pass
    try:
        fig.set_size_inches(max(3.0, width / 100.0), max(2.4, height / 100.0), forward=True)
    except Exception:
        pass


def _make_figure(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(max(3.0, args.window_width / 100.0), max(2.4, args.window_height / 100.0)))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["muted"], labelsize=8)
    ax.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.set_title(args.title, color=COLORS["text"], loc="left", fontsize=10, pad=10)
    ax.set_xlabel("time (s)", color=COLORS["muted"], fontsize=8)

    artists: dict[str, Any] = {}
    if args.kind == "stiffness":
        ax.set_ylabel("stiffness (N/m)", color=COLORS["muted"], fontsize=8)
        artists["kx_line"], = ax.plot([], [], color=COLORS["kx"], linewidth=2.0, label="Kx")
        artists["ky_line"], = ax.plot([], [], color=COLORS["ky"], linewidth=2.0, label="Ky")
        artists["kz_line"], = ax.plot([], [], color=COLORS["kz"], linewidth=2.0, label="Kz")
        artists["readout"] = fig.text(
            0.04,
            0.93,
            "Kx --   Ky --   Kz --",
            color=COLORS["text"],
            fontsize=15,
            fontweight="bold",
        )
        artists["status"] = fig.text(0.04, 0.885, "waiting for stiffness samples...", color=COLORS["muted"], fontsize=9)
    else:
        ax.set_ylabel("force (N)", color=COLORS["muted"], fontsize=8)
        artists["force_line"], = ax.plot([], [], color=COLORS["force"], linewidth=2.0, label="LPF |F|")
        artists["readout"] = fig.text(
            0.04,
            0.93,
            "LPF |F| -- N",
            color=COLORS["force"],
            fontsize=16,
            fontweight="bold",
        )
        artists["status"] = fig.text(0.04, 0.885, "waiting for force samples...", color=COLORS["muted"], fontsize=9)

    ax.legend(loc="upper right", facecolor=COLORS["background"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=8)
    fig.subplots_adjust(top=0.80, left=0.11, right=0.97, bottom=0.16)
    _configure_window(
        fig,
        title=args.title,
        x=args.window_x,
        y=args.window_y,
        width=max(240, int(args.window_width)),
        height=max(180, int(args.window_height)),
    )
    plt.show(block=False)
    fig.canvas.draw_idle()
    plt.pause(0.05)
    return fig, ax, artists


def _butterworth_sos(fs_hz: float, cutoff_hz: float, order: int) -> np.ndarray:
    from scipy import signal

    nyquist = 0.5 * float(fs_hz)
    normalized_cutoff = float(cutoff_hz) / nyquist
    return signal.butter(int(order), normalized_cutoff, btype="lowpass", output="sos")


def _filtered_force_norm(rows: deque[dict[str, float]]) -> np.ndarray:
    from scipy import signal

    components = np.asarray([[row["fx"], row["fy"], row["fz"]] for row in rows], dtype=float)
    if components.shape[0] == 0:
        return np.zeros(0, dtype=float)
    if components.shape[0] < 4:
        return np.linalg.norm(components, axis=1)
    sos = _butterworth_sos(
        fs_hz=DEFAULT_FORCE_FILTER_FS_HZ,
        cutoff_hz=DEFAULT_FORCE_FILTER_CUTOFF_HZ,
        order=DEFAULT_FORCE_FILTER_ORDER,
    )
    padlen = min(components.shape[0] - 1, max(1, 3 * (2 * len(sos) + 1)))
    if padlen < 1:
        return np.linalg.norm(components, axis=1)
    filtered_components = signal.sosfiltfilt(sos, components, axis=0, padtype="odd", padlen=int(padlen))
    return np.linalg.norm(filtered_components, axis=1)


def _scale_axis(ax: Any, times: np.ndarray, values: list[np.ndarray]) -> None:
    if times.size == 0:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        return
    x_min = float(np.min(times))
    x_max = float(np.max(times))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0
    finite_values = np.concatenate([array[np.isfinite(array)] for array in values if array.size])
    if finite_values.size == 0:
        y_min, y_max = -1.0, 1.0
    else:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        if abs(y_max - y_min) < 1e-9:
            pad = max(1.0, abs(y_max) * 0.1)
        else:
            pad = (y_max - y_min) * 0.12
        y_min -= pad
        y_max += pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _render(args: argparse.Namespace, rows: deque[dict[str, float]], fig: Any, ax: Any, artists: dict[str, Any]) -> None:
    times = np.asarray([row["time"] for row in rows], dtype=float)
    if args.kind == "stiffness":
        kx = np.asarray([row["kx"] for row in rows], dtype=float)
        ky = np.asarray([row["ky"] for row in rows], dtype=float)
        kz = np.asarray([row["kz"] for row in rows], dtype=float)
        artists["kx_line"].set_data(times, kx)
        artists["ky_line"].set_data(times, ky)
        artists["kz_line"].set_data(times, kz)
        if rows:
            latest = rows[-1]
            artists["readout"].set_text(f"Kx {latest['kx']:.1f}   Ky {latest['ky']:.1f}   Kz {latest['kz']:.1f} N/m")
            artists["status"].set_text(f"samples: {len(rows)}")
        _scale_axis(ax, times, [kx, ky, kz])
    else:
        force = _filtered_force_norm(rows)
        artists["force_line"].set_data(times, force)
        if force.size:
            artists["readout"].set_text(f"LPF |F| {float(force[-1]):.1f} N")
            artists["status"].set_text(f"Butterworth low-pass: order 4, cutoff 40 Hz, fs 500 Hz, samples: {len(rows)}")
        _scale_axis(ax, times, [force])

    fig.canvas.draw_idle()
    plt.pause(0.001)


def _row_from_payload(kind: str, payload: dict[str, Any]) -> dict[str, float]:
    t = _finite(payload.get("time"), 0.0)
    if kind == "stiffness":
        return {
            "time": t,
            "kx": _finite(payload.get("kx")),
            "ky": _finite(payload.get("ky")),
            "kz": _finite(payload.get("kz")),
        }
    return {
        "time": t,
        "fx": _finite(payload.get("fx")),
        "fy": _finite(payload.get("fy")),
        "fz": _finite(payload.get("fz")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Display live Sigma7 screening metrics from JSON lines on stdin.")
    parser.add_argument("--kind", choices=("stiffness", "force"), required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--draw-stride", type=int, default=1)
    parser.add_argument("--window-x", type=int, default=None)
    parser.add_argument("--window-y", type=int, default=None)
    parser.add_argument("--window-width", type=int, default=560)
    parser.add_argument("--window-height", type=int, default=400)
    args = parser.parse_args(argv)

    if args.kind == "force":
        try:
            import scipy  # noqa: F401
        except Exception as exc:
            print(f"force metric window requires scipy for sosfiltfilt: {exc}", file=sys.stderr, flush=True)
            return 2

    rows: deque[dict[str, float]] = deque(maxlen=max(int(args.max_points), 2))
    fig, ax, artists = _make_figure(args)
    row_count = 0
    last_render = time.perf_counter()

    while plt.fignum_exists(fig.number):
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not readable:
            plt.pause(0.001)
            continue
        line = sys.stdin.readline()
        if not line:
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("close"):
            break
        row = _row_from_payload(args.kind, payload)
        rows.append(row)
        row_count += 1

        if args.window_seconds > 0.0 and rows:
            cutoff = row["time"] - float(args.window_seconds)
            while len(rows) > 2 and rows[0]["time"] < cutoff:
                rows.popleft()

        now = time.perf_counter()
        if row_count % max(int(args.draw_stride), 1) == 0 or now - last_render > 0.25:
            _render(args, rows, fig, ax, artists)
            last_render = now

    try:
        plt.close(fig)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
