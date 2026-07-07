from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class PegHoleIds:
    peg_body: int
    hole_body: int
    peg_tip_site: int
    hole_center_site: int
    peg_geoms: tuple[int, ...]
    hole_wall_geoms: tuple[int, ...]

    @property
    def peg_geom(self) -> int:
        return self.peg_geoms[0]


def required_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(f"MuJoCo object {name!r} of type {obj_type!r} was not found.")
    return int(obj_id)


def peg_hole_ids(model: mujoco.MjModel, *, segments: int) -> PegHoleIds:
    peg_geoms = tuple(
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom(geom_id).name.startswith("peg_body")
    )
    if not peg_geoms:
        peg_geoms = (required_id(model, mujoco.mjtObj.mjOBJ_GEOM, "peg_body"),)
    return PegHoleIds(
        peg_body=required_id(model, mujoco.mjtObj.mjOBJ_BODY, "peg"),
        hole_body=required_id(model, mujoco.mjtObj.mjOBJ_BODY, "hole_fixture"),
        peg_tip_site=required_id(model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip"),
        hole_center_site=required_id(model, mujoco.mjtObj.mjOBJ_SITE, "hole_center"),
        peg_geoms=peg_geoms,
        hole_wall_geoms=tuple(
            required_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"hole_wall_{idx:02d}")
            for idx in range(segments)
        ),
    )
