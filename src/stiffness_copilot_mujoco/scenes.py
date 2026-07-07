from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from stiffness_copilot_mujoco.controllers.impedance import TRACK_A_BASELINE_CONTROLLER_PROFILE
from stiffness_copilot_mujoco.sim.scene import ROOT


SceneName = Literal[
    "circle",
    "polygon_circle_logic_v1",
    "star_circle_logic_v1",
]


@dataclass(frozen=True)
class SceneSpec:
    name: SceneName
    config_path: Path
    base_profile: str
    active_groups: tuple[tuple[int, ...], ...]
    active_group_names: tuple[str, ...]
    residual_bound: float


SCENE_SPECS: dict[str, SceneSpec] = {
    "circle": SceneSpec(
        name="circle",
        config_path=ROOT / "configs" / "scenes" / "residual_bc" / "circle.yaml",
        base_profile=TRACK_A_BASELINE_CONTROLLER_PROFILE,
        active_groups=((0, 1),),
        active_group_names=("alpha_lateral_shared",),
        residual_bound=0.35,
    ),
    "polygon_circle_logic_v1": SceneSpec(
        name="polygon_circle_logic_v1",
        config_path=ROOT / "configs" / "scenes" / "residual_bc" / "polygon_circle_logic_v1.yaml",
        base_profile=TRACK_A_BASELINE_CONTROLLER_PROFILE,
        active_groups=((0,), (1,)),
        active_group_names=("alpha_x", "alpha_y"),
        residual_bound=0.35,
    ),
    "star_circle_logic_v1": SceneSpec(
        name="star_circle_logic_v1",
        config_path=ROOT / "configs" / "scenes" / "residual_bc" / "star_circle_logic_v1.yaml",
        base_profile=TRACK_A_BASELINE_CONTROLLER_PROFILE,
        active_groups=((0,), (1,), (3,)),
        active_group_names=("alpha_x", "alpha_y", "l21"),
        residual_bound=0.35,
    ),
}


def scene_names() -> tuple[str, ...]:
    return tuple(SCENE_SPECS)


def get_scene_spec(name: str) -> SceneSpec:
    try:
        return SCENE_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown scene {name!r}. Available scenes: {', '.join(scene_names())}") from exc


def parse_active_groups(value: str | None, default: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    if value is None or value == "":
        return default
    names = {
        "alpha1": 0,
        "alpha2": 1,
        "alpha3": 2,
        "l21": 3,
        "l31": 4,
        "l32": 5,
        "alpha_lateral_shared": (0, 1),
        "alpha_x": 0,
        "alpha_y": 1,
        "alpha_z": 2,
    }
    groups: list[tuple[int, ...]] = []
    for raw in value.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in names:
            raise KeyError(f"Unknown active dim {key!r}.")
        item = names[key]
        groups.append(tuple(item) if isinstance(item, tuple) else (item,))
    if not groups:
        raise ValueError("At least one active dim is required.")
    return tuple(groups)
