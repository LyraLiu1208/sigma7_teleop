#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from typing import Iterable

import cv2
import numpy as np


DEFAULT_FORCE_FILTER_ORDER = 4
DEFAULT_FORCE_FILTER_CUTOFF_HZ = 40.0
DEFAULT_FORCE_FILTER_FS_HZ = 500.0


COLORS = {
    "x": (54, 112, 255),
    "y": (80, 190, 110),
    "z": (230, 130, 60),
    "norm": (245, 245, 245),
    "normal": (80, 220, 255),
    "tangent": (210, 110, 255),
}


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def _scale(values: Iterable[float], *, pad_fraction: float = 0.12) -> tuple[float, float]:
    finite_values = [float(v) for v in values if np.isfinite(v)]
    if not finite_values:
        return -1.0, 1.0
    lo = min(finite_values)
    hi = max(finite_values)
    if abs(hi - lo) < 1e-9:
        pad = max(1.0, abs(hi) * 0.1)
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _draw_panel(
    canvas: np.ndarray,
    *,
    rect: tuple[int, int, int, int],
    times: list[float],
    series: dict[str, list[float]],
    colors: dict[str, tuple[int, int, int]],
    title: str,
    unit: str,
) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 70), 1)
    cv2.putText(canvas, title, (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    if len(times) < 2:
        return

    values = [v for values_for_key in series.values() for v in values_for_key]
    y_min, y_max = _scale(values)
    t_min = min(times)
    t_max = max(times)
    if abs(t_max - t_min) < 1e-9:
        t_max = t_min + 1.0

    plot_left = x0 + 54
    plot_right = x1 - 14
    plot_top = y0 + 42
    plot_bottom = y1 - 34
    cv2.line(canvas, (plot_left, plot_bottom), (plot_right, plot_bottom), (90, 90, 90), 1)
    cv2.line(canvas, (plot_left, plot_top), (plot_left, plot_bottom), (90, 90, 90), 1)
    cv2.putText(canvas, f"{y_max:.1f}", (x0 + 7, plot_top + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_min:.1f}", (x0 + 7, plot_bottom + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, unit, (plot_left, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    def to_point(t: float, v: float) -> tuple[int, int]:
        x = int(plot_left + (t - t_min) / (t_max - t_min) * (plot_right - plot_left))
        y = int(plot_bottom - (v - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        return x, y

    legend_x = x0 + 10
    for index, (key, values_for_key) in enumerate(series.items()):
        color = colors[key]
        cv2.putText(
            canvas,
            f"{key}: {values_for_key[-1]:.2f}",
            (legend_x + index * 136, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        points = [to_point(t, v) for t, v in zip(times, values_for_key) if np.isfinite(v)]
        for p0, p1 in zip(points[:-1], points[1:]):
            cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)


def _draw_stiffness(rows: deque[dict[str, float]], title: str) -> np.ndarray:
    canvas = np.full((460, 900, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, title, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 1, cv2.LINE_AA)
    times = [row["time"] for row in rows]
    series = {
        "Kx": [row["kx"] for row in rows],
        "Ky": [row["ky"] for row in rows],
        "Kz": [row["kz"] for row in rows],
    }
    _draw_panel(
        canvas,
        rect=(18, 58, 882, 438),
        times=times,
        series=series,
        colors={"Kx": COLORS["x"], "Ky": COLORS["y"], "Kz": COLORS["z"]},
        title="Commanded translational stiffness",
        unit="N/m",
    )
    return canvas


def _draw_force(rows: deque[dict[str, float]], title: str) -> np.ndarray:
    canvas = np.full((460, 900, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, title, (22, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 1, cv2.LINE_AA)
    times = [row["time"] for row in rows]
    series = {"LPF |F|": _filtered_force_norm(rows).tolist()}
    _draw_panel(
        canvas,
        rect=(18, 58, 882, 438),
        times=times,
        series=series,
        colors={"LPF |F|": COLORS["normal"]},
        title="Low-pass filtered resultant contact force",
        unit="N",
    )
    return canvas


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
    parser.add_argument("--window-height", type=int, default=320)
    args = parser.parse_args(argv)

    if args.kind == "force":
        try:
            import scipy  # noqa: F401
        except Exception as exc:
            print(f"force metric window requires scipy for sosfiltfilt: {exc}", file=sys.stderr, flush=True)
            return 2

    rows: deque[dict[str, float]] = deque(maxlen=max(int(args.max_points), 2))
    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.title, max(240, int(args.window_width)), max(180, int(args.window_height)))
    if args.window_x is not None and args.window_y is not None:
        cv2.moveWindow(args.title, int(args.window_x), int(args.window_y))
    initial_canvas = _draw_stiffness(rows, args.title) if args.kind == "stiffness" else _draw_force(rows, args.title)
    cv2.imshow(args.title, initial_canvas)
    cv2.waitKeyEx(1)
    row_count = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("close"):
            break

        t = _finite(payload.get("time"), 0.0)
        if args.kind == "stiffness":
            row = {
                "time": t,
                "kx": _finite(payload.get("kx")),
                "ky": _finite(payload.get("ky")),
                "kz": _finite(payload.get("kz")),
            }
        else:
            row = {
                "time": t,
                "fx": _finite(payload.get("fx")),
                "fy": _finite(payload.get("fy")),
                "fz": _finite(payload.get("fz")),
            }
        rows.append(row)
        row_count += 1

        if args.window_seconds > 0.0 and rows:
            cutoff = t - float(args.window_seconds)
            while len(rows) > 2 and rows[0]["time"] < cutoff:
                rows.popleft()

        if row_count % max(int(args.draw_stride), 1) != 0:
            continue
        canvas = _draw_stiffness(rows, args.title) if args.kind == "stiffness" else _draw_force(rows, args.title)
        cv2.imshow(args.title, canvas)
        key = cv2.waitKeyEx(1)
        if key in {27, ord("q"), ord("Q")}:
            break

    try:
        cv2.destroyWindow(args.title)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
