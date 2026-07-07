from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from stiffness_copilot_mujoco.sim.ids import PegHoleIds


@dataclass(frozen=True)
class PegHoleGeometry:
    peg_attachment: str
    peg_radius: float
    peg_half_height: float
    hole_inner_radius: float
    hole_wall_half_height: float
    radial_clearance: float
    segments: int


@dataclass(frozen=True)
class ContactMetrics:
    peg_contact_count: int
    peg_hole_contact_count: int
    peg_normal_force: float
    min_peg_contact_dist: float | None


def load_scene_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def geometry_from_config(config: dict[str, Any]) -> PegHoleGeometry:
    peg = config["peg"]
    hole = config["hole"]
    peg_radius = float(peg.get("radius", peg.get("outer_radius")))
    wall_center_radius = float(hole.get("wall_center_radius", hole.get("inner_radius")))
    wall_radial_half_thickness = float(hole["wall_radial_half_thickness"])
    hole_inner_radius = float(hole.get("inner_radius", wall_center_radius - wall_radial_half_thickness))
    radial_clearance = float(hole.get("completion_lateral_tolerance", hole_inner_radius - peg_radius))
    return PegHoleGeometry(
        peg_attachment=str(peg.get("attachment", "free")),
        peg_radius=peg_radius,
        peg_half_height=float(peg["half_height"]),
        hole_inner_radius=hole_inner_radius,
        hole_wall_half_height=float(hole["wall_half_height"]),
        radial_clearance=radial_clearance,
        segments=int(hole["segments"]),
    )


def site_position(data: mujoco.MjData, site_id: int) -> np.ndarray:
    return np.array(data.site_xpos[site_id], dtype=float)


def body_position(data: mujoco.MjData, body_id: int) -> np.ndarray:
    return np.array(data.xpos[body_id], dtype=float)


def peg_tip_position(data: mujoco.MjData, ids: PegHoleIds) -> np.ndarray:
    return site_position(data, ids.peg_tip_site)


def hole_center_position(data: mujoco.MjData, ids: PegHoleIds) -> np.ndarray:
    return site_position(data, ids.hole_center_site)


def hole_insertion_axis_world() -> np.ndarray:
    return np.array([0.0, 0.0, -1.0], dtype=float)


def peg_hole_delta(data: mujoco.MjData, ids: PegHoleIds) -> np.ndarray:
    return peg_tip_position(data, ids) - hole_center_position(data, ids)


def peg_axis_world(data: mujoco.MjData, ids: PegHoleIds) -> np.ndarray:
    axis = peg_tip_position(data, ids) - body_position(data, ids.peg_body)
    norm = np.linalg.norm(axis)
    if norm <= 0.0:
        raise RuntimeError("Peg axis has zero norm.")
    return axis / norm


def peg_axis_error(data: mujoco.MjData, ids: PegHoleIds, target_axis: np.ndarray) -> np.ndarray:
    target = np.asarray(target_axis, dtype=float)
    target = target / np.linalg.norm(target)
    current = peg_axis_world(data, ids)
    return np.cross(current, target)


def peg_axis_alignment(data: mujoco.MjData, ids: PegHoleIds, target_axis: np.ndarray) -> float:
    target = np.asarray(target_axis, dtype=float)
    target = target / np.linalg.norm(target)
    return float(np.dot(peg_axis_world(data, ids), target))


def lateral_error(data: mujoco.MjData, ids: PegHoleIds) -> float:
    delta = peg_hole_delta(data, ids)
    return float(np.linalg.norm(delta[:2]))


def insertion_depth(data: mujoco.MjData, ids: PegHoleIds) -> float:
    delta = peg_hole_delta(data, ids)
    return float(-delta[2])


def is_tip_inside_hole_radius(data: mujoco.MjData, ids: PegHoleIds, geometry: PegHoleGeometry) -> bool:
    return lateral_error(data, ids) <= geometry.radial_clearance


def contact_metrics(model: mujoco.MjModel, data: mujoco.MjData, ids: PegHoleIds) -> ContactMetrics:
    peg_contact_count = 0
    peg_hole_contact_count = 0
    peg_normal_force = 0.0
    min_dist: float | None = None
    hole_geoms = set(ids.hole_wall_geoms)
    peg_geoms = set(ids.peg_geoms)

    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 not in peg_geoms and geom2 not in peg_geoms:
            continue

        peg_contact_count += 1
        if geom1 in hole_geoms or geom2 in hole_geoms:
            peg_hole_contact_count += 1

        force = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_idx, force)
        peg_normal_force += abs(float(force[0]))
        dist = float(contact.dist)
        min_dist = dist if min_dist is None else min(min_dist, dist)

    return ContactMetrics(
        peg_contact_count=peg_contact_count,
        peg_hole_contact_count=peg_hole_contact_count,
        peg_normal_force=peg_normal_force,
        min_peg_contact_dist=min_dist,
    )
