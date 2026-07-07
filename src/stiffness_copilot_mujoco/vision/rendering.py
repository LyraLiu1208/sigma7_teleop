from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return value / norm


def _project_points(
    points_world: np.ndarray,
    *,
    camera_pos: np.ndarray,
    camera_mat: np.ndarray,
    width: int,
    height: int,
    fovy_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64)
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    camera_mat = np.asarray(camera_mat, dtype=np.float64).reshape(3, 3)
    points_cam = (points_world - camera_pos[None, :]) @ camera_mat
    depth = points_cam[:, 2]
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2.0)
    x = width * 0.5 + f * (points_cam[:, 0] / np.maximum(depth, 1e-6))
    y = height * 0.5 - f * (points_cam[:, 1] / np.maximum(depth, 1e-6))
    return np.stack([x, y], axis=1), depth


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64)
    inside = np.zeros_like(xs, dtype=bool)
    x0, y0 = polygon[-1]
    for x1, y1 in polygon:
        cond = ((y1 > ys) != (y0 > ys)) & (
            xs < (x0 - x1) * (ys - y1) / np.maximum(y0 - y1, 1e-12) + x1
        )
        inside ^= cond
        x0, y0 = x1, y1
    return inside


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] <= 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull if hull.size else pts[:1]


def _fill_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
    *,
    color: np.ndarray,
    alpha: float,
) -> None:
    h, w, _ = image.shape
    if polygon.shape[0] < 3:
        return
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    min_x = max(int(np.floor(xs.min())), 0)
    max_x = min(int(np.ceil(xs.max())), w - 1)
    min_y = max(int(np.floor(ys.min())), 0)
    max_y = min(int(np.ceil(ys.max())), h - 1)
    if min_x > max_x or min_y > max_y:
        return
    grid_x, grid_y = np.meshgrid(
        np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5,
        np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5,
    )
    inside = _points_in_polygon(grid_x, grid_y, polygon)
    if not np.any(inside):
        return
    region = image[min_y : max_y + 1, min_x : max_x + 1]
    region[inside] = (1.0 - alpha) * region[inside] + alpha * color


def _draw_circle(
    image: np.ndarray,
    *,
    center: tuple[float, float],
    radius: float,
    color: np.ndarray,
) -> None:
    h, w, _ = image.shape
    cx, cy = center
    min_x = max(int(np.floor(cx - radius)), 0)
    max_x = min(int(np.ceil(cx + radius)), w - 1)
    min_y = max(int(np.floor(cy - radius)), 0)
    max_y = min(int(np.ceil(cy + radius)), h - 1)
    if min_x > max_x or min_y > max_y:
        return
    xs, ys = np.meshgrid(
        np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5,
        np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5,
    )
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
    if not np.any(mask):
        return
    region = image[min_y : max_y + 1, min_x : max_x + 1]
    region[mask] = color


@dataclass
class _SoftwareRenderer:
    model: mujoco.MjModel
    camera_name: str
    width: int
    height: int

    def __post_init__(self) -> None:
        self._camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if self._camera_id < 0:
            raise ValueError(f"Unknown camera {self.camera_name!r}.")

    def _background(self) -> np.ndarray:
        top = np.array([178.0, 198.0, 218.0], dtype=np.float32)
        bottom = np.array([40.0, 42.0, 48.0], dtype=np.float32)
        ys = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None, None]
        gradient = (1.0 - ys) * top[None, None, :] + ys * bottom[None, None, :]
        return np.broadcast_to(gradient, (self.height, self.width, 3)).copy()

    def render(self, data: mujoco.MjData) -> np.ndarray:
        image = self._background()
        camera_pos = np.array(data.cam_xpos[self._camera_id], dtype=np.float64)
        camera_mat = np.array(data.cam_xmat[self._camera_id], dtype=np.float64).reshape(3, 3)
        fovy = float(self.model.cam_fovy[self._camera_id])

        drawables: list[tuple[float, np.ndarray, float]] = []
        for geom_id in range(self.model.ngeom):
            rgba = np.array(self.model.geom_rgba[geom_id, :4], dtype=np.float64)
            if rgba[3] <= 0.01:
                continue
            geom_type = int(self.model.geom_type[geom_id])
            size = np.array(self.model.geom_size[geom_id], dtype=np.float64)
            geom_pos = np.array(data.geom_xpos[geom_id], dtype=np.float64)
            geom_mat = np.array(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                corners = np.array(
                    [
                        [sx, sy, sz]
                        for sx in (-size[0], size[0])
                        for sy in (-size[1], size[1])
                        for sz in (-size[2], size[2])
                    ],
                    dtype=np.float64,
                )
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                height = size[1]
                radius = size[0]
                theta = np.linspace(0.0, 2.0 * math.pi, 18, endpoint=False)
                ring = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
                top = np.column_stack([ring, np.full(len(ring), height)])
                bottom = np.column_stack([ring, np.full(len(ring), -height)])
                corners = np.concatenate([top, bottom], axis=0)
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                radius = size[0]
                theta = np.linspace(0.0, 2.0 * math.pi, 18, endpoint=False)
                corners = np.stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)], axis=1)
            else:
                if np.all(size > 0.0):
                    corners = np.array(
                        [
                            [sx, sy, sz]
                            for sx in (-size[0], size[0])
                            for sy in (-size[1], size[1])
                            for sz in (-size[2], size[2])
                        ],
                        dtype=np.float64,
                    )
                else:
                    continue
            world_points = corners @ geom_mat.T + geom_pos[None, :]
            screen, depth = _project_points(
                world_points,
                camera_pos=camera_pos,
                camera_mat=camera_mat,
                width=self.width,
                height=self.height,
                fovy_deg=fovy,
            )
            if not np.any(depth > 1e-6):
                continue
            hull = _convex_hull_2d(screen[np.isfinite(screen).all(axis=1)])
            if hull.shape[0] >= 3:
                drawables.append((float(np.mean(depth)), hull, rgba))

        drawables.sort(key=lambda item: item[0], reverse=True)
        for _depth, polygon, rgba in drawables:
            color = np.clip(np.asarray(rgba[:3], dtype=np.float32) * 255.0, 0.0, 255.0)
            alpha = max(0.20, min(float(rgba[3]), 0.85))
            _fill_polygon(image, polygon, color=color, alpha=alpha)

        # Draw a few key sites as markers so the sanity pass can verify visibility.
        for site_name, color, radius in (
            ("peg_tip", np.array([235.0, 80.0, 50.0], dtype=np.float32), 3.5),
            ("hole_center", np.array([40.0, 190.0, 200.0], dtype=np.float32), 3.5),
        ):
            try:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            except Exception:
                continue
            if site_id < 0:
                continue
            site_pos = np.array(data.site_xpos[site_id], dtype=np.float64)
            screen, depth = _project_points(
                site_pos[None, :],
                camera_pos=camera_pos,
                camera_mat=camera_mat,
                width=self.width,
                height=self.height,
                fovy_deg=fovy,
            )
            if depth[0] > 1e-6 and np.isfinite(screen).all():
                _draw_circle(image, center=(float(screen[0, 0]), float(screen[0, 1])), radius=radius, color=color)

        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    def close(self) -> None:
        return None


@dataclass
class MujocoRgbRenderer:
    model: mujoco.MjModel
    camera_name: str
    width: int = 128
    height: int = 128
    renderer_mode: Literal["native", "legacy_debug_only"] = "native"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive.")
        self._renderer = None
        self._software_renderer = None
        self.fallback_used = False
        self.native_error: Exception | None = None
        try:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            self.mode = "mujoco_native"
        except Exception as exc:
            self.native_error = exc
            if self.renderer_mode == "native":
                raise RuntimeError(
                    f"Native MuJoCo RGB renderer is unavailable for camera {self.camera_name!r}: {exc}. "
                    "Active image-only workflows require native rendering."
                ) from exc
            self.fallback_used = True
            self.mode = "legacy_debug_only_software_fallback"
            self._software_renderer = _SoftwareRenderer(
                self.model,
                camera_name=self.camera_name,
                width=self.width,
                height=self.height,
            )

    def render(self, data: mujoco.MjData) -> np.ndarray:
        if self._renderer is not None:
            self._renderer.update_scene(data, camera=self.camera_name)
            frame = self._renderer.render()
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return np.asarray(frame, dtype=np.uint8)
        assert self._software_renderer is not None
        return self._software_renderer.render(data)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        if self._software_renderer is not None:
            self._software_renderer.close()

    def __enter__(self) -> "MujocoRgbRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def render_mujoco_rgb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_name: str,
    width: int,
    height: int,
    renderer_mode: Literal["native", "legacy_debug_only"] = "native",
) -> np.ndarray:
    with MujocoRgbRenderer(model, camera_name=camera_name, width=width, height=height, renderer_mode=renderer_mode) as renderer:
        return renderer.render(data)


__all__ = ["MujocoRgbRenderer", "render_mujoco_rgb"]
