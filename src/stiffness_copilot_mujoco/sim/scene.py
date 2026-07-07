from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "scenes" / "panda_peg_in_hole_torque.yaml"
PANDA_XML = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
GENERATED_DIR = ROOT / "models" / "generated"
GENERATED_PANDA_WITH_PEG = GENERATED_DIR / "panda_with_peg.xml"
GENERATED_PANDA_WITH_PEG_TORQUE = GENERATED_DIR / "panda_with_peg_torque.xml"
DEFAULT_EYE_IN_HAND_CAMERA_NAME = "eye_in_hand_rgb"
DEFAULT_EYE_IN_HAND_CAMERA_POS = (0.075, 0.09, 0.1)
DEFAULT_EYE_IN_HAND_CAMERA_FORWARD = (0.40, 0.00, -0.916515138991168)
DEFAULT_EYE_IN_HAND_CAMERA_UP = (0.916515138991168, 0.0, 0.4)
DEFAULT_EYE_IN_HAND_CAMERA_FOVY = 55.0
CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION = "canonical_eye_in_hand_camera_v1"
CANONICAL_EYE_IN_HAND_CAMERA_ATTACHMENT_PARENT = "hand"
CANONICAL_EYE_IN_HAND_CAMERA_MOUNT_TYPE = "hand_mounted_standoff"


def fmt(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:g}" for value in values)


def _contact_attr(config: dict[str, object], name: str, default: tuple[float, ...]) -> str:
    contact = config.get("contact", {})
    if not isinstance(contact, dict):
        return fmt(default)
    values = contact.get(name, default)
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"contact.{name} must be a list or tuple.")
    return fmt(tuple(float(value) for value in values))


def _regular_polygon_vertices(*, sides: int, radius: float) -> list[tuple[float, float]]:
    if sides < 3:
        raise ValueError(f"Polygon must have at least 3 sides, got {sides}.")
    return [
        (radius * math.cos(2.0 * math.pi * idx / sides), radius * math.sin(2.0 * math.pi * idx / sides))
        for idx in range(sides)
    ]


def _star_vertices(
    *,
    points: int,
    outer_radius: float,
    inner_radius: float,
    rotation: float = 0.0,
) -> list[tuple[float, float]]:
    if points < 3:
        raise ValueError(f"Star must have at least 3 points, got {points}.")
    vertices = []
    for idx in range(2 * points):
        radius = outer_radius if idx % 2 == 0 else inner_radius
        theta = math.pi / 2.0 + rotation + math.pi * idx / points
        vertices.append((radius * math.cos(theta), radius * math.sin(theta)))
    return vertices


def prism_mesh_asset(name: str, *, vertices_2d: list[tuple[float, float]], half_height: float) -> str:
    sides = len(vertices_2d)
    vertices: list[float] = []
    bottom_center = 0
    vertices.extend([0.0, 0.0, -half_height])
    for z in (-half_height, half_height):
        for x, y in vertices_2d:
            vertices.extend([x, y, z])
        if z < 0.0:
            top_center = len(vertices) // 3
            vertices.extend([0.0, 0.0, half_height])

    faces: list[int] = []
    bottom_offset = 1
    top_offset = bottom_offset + sides + 1
    for idx in range(sides):
        nxt = (idx + 1) % sides
        faces.extend([bottom_center, bottom_offset + nxt, bottom_offset + idx])
    for idx in range(sides):
        nxt = (idx + 1) % sides
        faces.extend([top_center, top_offset + idx, top_offset + nxt])
    for idx in range(sides):
        nxt = (idx + 1) % sides
        faces.extend([bottom_offset + idx, bottom_offset + nxt, top_offset + nxt])
        faces.extend([bottom_offset + idx, top_offset + nxt, top_offset + idx])

    return f'    <mesh name="{name}" vertex="{fmt(vertices)}" face="{fmt(faces)}"/>'


def triangular_prism_mesh_asset(
    name: str,
    *,
    triangle_2d: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    half_height: float,
) -> str:
    vertices: list[float] = []
    for z in (-half_height, half_height):
        for x, y in triangle_2d:
            vertices.extend([x, y, z])
    faces = [
        0,
        2,
        1,
        3,
        4,
        5,
        0,
        1,
        4,
        0,
        4,
        3,
        1,
        2,
        5,
        1,
        5,
        4,
        2,
        0,
        3,
        2,
        3,
        5,
    ]
    return f'    <mesh name="{name}" vertex="{fmt(vertices)}" face="{fmt(faces)}"/>'


def star_collision_mesh_assets(peg: dict[str, object]) -> str:
    vertices = _star_vertices(
        points=int(peg["points"]),
        outer_radius=float(peg["outer_radius"]),
        inner_radius=float(peg["inner_radius"]),
    )
    half_height = float(peg["half_height"])
    center = (0.0, 0.0)
    lines = []
    for idx, (start, end) in enumerate(zip(vertices, vertices[1:] + vertices[:1], strict=True)):
        lines.append(
            triangular_prism_mesh_asset(
                f"peg_collision_{idx:02d}",
                triangle_2d=(center, start, end),
                half_height=half_height,
            )
        )
    return "\n".join(lines)


def peg_mesh_asset_xml(peg: dict[str, object]) -> str:
    shape = str(peg.get("shape", "cylinder"))
    if shape == "cylinder":
        return ""
    if shape not in {"polygon_prism", "hexagonal_prism", "star_prism"}:
        raise ValueError(f"Unsupported peg.shape {shape!r}.")
    if shape == "star_prism":
        vertices_2d = _star_vertices(
            points=int(peg["points"]),
            outer_radius=float(peg["outer_radius"]),
            inner_radius=float(peg["inner_radius"]),
        )
    else:
        sides = 6 if shape == "hexagonal_prism" else int(peg["sides"])
        vertices_2d = _regular_polygon_vertices(sides=sides, radius=float(peg["radius"]))
    return prism_mesh_asset(
        "peg_mesh",
        vertices_2d=vertices_2d,
        half_height=float(peg["half_height"]),
    ) + (f"\n{star_collision_mesh_assets(peg)}" if shape == "star_prism" else "")


def peg_geom_xml(peg: dict[str, object], *, indent: str, include_mass: bool) -> str:
    shape = str(peg.get("shape", "cylinder"))
    mass_attr = f' mass="{peg["mass"]:g}"' if include_mass else ""
    if shape == "cylinder":
        return (
            f'{indent}<geom name="peg_body" type="cylinder" size="{peg["radius"]:g} {peg["half_height"]:g}" '
            f'material="peg_mat"{mass_attr} class="task_collision"/>'
        )
    if shape in {"polygon_prism", "hexagonal_prism"}:
        return (
            f'{indent}<geom name="peg_body" type="mesh" mesh="peg_mesh" material="peg_mat"{mass_attr} '
            'class="task_collision"/>'
        )
    if shape == "star_prism":
        points = int(peg["points"])
        lines = [
            f'{indent}<geom name="peg_visual" type="mesh" mesh="peg_mesh" material="peg_mat" contype="0" conaffinity="0"/>'
        ]
        pieces = 2 * points
        for idx in range(pieces):
            part_mass = float(peg["mass"]) / pieces if include_mass else None
            part_mass_attr = f' mass="{part_mass:g}"' if part_mass is not None else ""
            lines.append(
                f'{indent}<geom name="peg_body_{idx:02d}" type="mesh" mesh="peg_collision_{idx:02d}" '
                f'rgba="0.92 0.42 0.24 0"{part_mass_attr} class="task_collision"/>'
            )
        return "\n".join(lines)
    raise ValueError(f"Unsupported peg.shape {shape!r}.")


def wall_geoms(hole: dict[str, object]) -> str:
    shape = str(hole.get("shape", "regular"))
    radial_half_thickness = float(hole["wall_radial_half_thickness"])
    half_height = float(hole["wall_half_height"])
    z = half_height / 2.0

    lines = []
    if shape == "star":
        vertices = _star_vertices(
            points=int(hole["points"]),
            outer_radius=float(hole["outer_radius"]),
            inner_radius=float(hole["inner_radius"]),
            rotation=float(hole.get("rotation", 0.0)),
        )
        for idx, (start, end) in enumerate(zip(vertices, vertices[1:] + vertices[:1], strict=True)):
            sx, sy = start
            ex, ey = end
            x = 0.5 * (sx + ex)
            y = 0.5 * (sy + ey)
            edge_x = ex - sx
            edge_y = ey - sy
            tangent_angle = math.atan2(edge_y, edge_x)
            tangent_half_length = 0.5 * math.hypot(edge_x, edge_y)
            qw = math.cos(tangent_angle / 2.0)
            qz = math.sin(tangent_angle / 2.0)
            lines.append(
                "      "
                f'<geom name="hole_wall_{idx:02d}" type="box" '
                f'pos="{x:.5f} {y:.5f} {z:.5f}" '
                f'quat="{qw:.6f} 0 0 {qz:.6f}" '
                f'size="{tangent_half_length:g} {radial_half_thickness:g} {half_height:g}" '
                'material="hole_mat" class="task_collision"/>'
            )
        return "\n".join(lines)

    segments = int(hole["segments"])
    radius = float(hole["wall_center_radius"])
    tangent_half_length = float(hole["wall_tangent_half_length"])
    rotation = float(hole.get("rotation", 0.0))
    for idx in range(segments):
        theta = rotation + 2.0 * math.pi * idx / segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        tangent_angle = theta + math.pi / 2.0
        qw = math.cos(tangent_angle / 2.0)
        qz = math.sin(tangent_angle / 2.0)
        lines.append(
            "      "
            f'<geom name="hole_wall_{idx:02d}" type="box" '
            f'pos="{x:.5f} {y:.5f} {z:.5f}" '
            f'quat="{qw:.6f} 0 0 {qz:.6f}" '
            f'size="{tangent_half_length:g} {radial_half_thickness:g} {half_height:g}" '
            'material="hole_mat" class="task_collision"/>'
        )
    return "\n".join(lines)


def peg_body_xml(peg: dict[str, object]) -> str:
    attachment = str(peg.get("attachment", "free"))
    if attachment == "hand":
        raise ValueError("Hand-attached peg must be rendered inside the Panda hand body.")
    pos_key = "pos"
    peg_tip_z = -float(peg["half_height"])
    return f"""    <body name="peg" pos="{fmt(peg[pos_key])}">
      <freejoint name="peg_freejoint"/>
      <site name="peg_tip" pos="0 0 {peg_tip_z:g}" size="0.007" rgba="0.92 0.42 0.24 1"/>
{peg_geom_xml(peg, indent="      ", include_mass=True)}
    </body>"""


def peg_equality_xml(peg: dict[str, object]) -> str:
    attachment = str(peg.get("attachment", "free"))
    if attachment == "free":
        return ""
    if attachment != "hand":
        raise ValueError(f"Unsupported peg.attachment {attachment!r}. Use 'free' or 'hand'.")
    return ""


def hand_peg_body_xml(peg: dict[str, object]) -> str:
    peg_tip_z = float(peg["half_height"])
    return f"""                      <body name="peg" pos="{fmt(peg["hand_local_pos"])}">
                        <inertial mass="{peg["mass"]:g}" pos="0 0 0" diaginertia="0.00004 0.00004 0.00002"/>
                        <site name="peg_tip" pos="0 0 {peg_tip_z:g}" size="0.007" rgba="0.92 0.42 0.24 1"/>
{peg_geom_xml(peg, indent="                        ", include_mass=False)}
                      </body>"""


def _normalized(vector: tuple[float, float, float] | list[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero vector.")
    return value / norm


def _camera_xyaxes(*, forward: tuple[float, float, float] | list[float], up: tuple[float, float, float] | list[float]) -> tuple[np.ndarray, np.ndarray]:
    forward_vec = _normalized(forward)
    up_vec = _normalized(up)
    right_vec = np.cross(forward_vec, up_vec)
    right_norm = float(np.linalg.norm(right_vec))
    if right_norm <= 1e-12:
        raise ValueError("Camera forward and up vectors must not be collinear.")
    right_vec /= right_norm
    true_up_vec = np.cross(right_vec, forward_vec)
    true_up_vec /= max(float(np.linalg.norm(true_up_vec)), 1e-12)
    return right_vec, true_up_vec


def canonical_eye_in_hand_camera_pose(camera_name: str = DEFAULT_EYE_IN_HAND_CAMERA_NAME) -> dict[str, object]:
    return {
        "camera_name": camera_name,
        "attachment_parent": CANONICAL_EYE_IN_HAND_CAMERA_ATTACHMENT_PARENT,
        "mount_type": CANONICAL_EYE_IN_HAND_CAMERA_MOUNT_TYPE,
        "pos": [float(value) for value in DEFAULT_EYE_IN_HAND_CAMERA_POS],
        "forward": [float(value) for value in DEFAULT_EYE_IN_HAND_CAMERA_FORWARD],
        "up": [float(value) for value in DEFAULT_EYE_IN_HAND_CAMERA_UP],
        "fovy": float(DEFAULT_EYE_IN_HAND_CAMERA_FOVY),
        "pose_version": CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION,
        "canonical": True,
    }


def validate_canonical_eye_in_hand_camera_config(
    scene_config: dict[str, object],
    *,
    camera_name: str = DEFAULT_EYE_IN_HAND_CAMERA_NAME,
) -> dict[str, object]:
    vision = scene_config.get("vision", {})
    if not isinstance(vision, dict):
        raise TypeError("vision must be a mapping when provided.")
    camera_cfg = vision.get("eye_in_hand_camera", {})
    if not isinstance(camera_cfg, dict):
        raise TypeError("vision.eye_in_hand_camera must be a mapping when provided.")
    observed_name = str(camera_cfg.get("name", DEFAULT_EYE_IN_HAND_CAMERA_NAME))
    if observed_name != camera_name:
        raise ValueError(
            f"Scene config camera name {observed_name!r} does not match requested camera {camera_name!r}."
        )
    expected = canonical_eye_in_hand_camera_pose(camera_name=camera_name)
    observed = {
        "camera_name": observed_name,
        "pos": np.asarray(camera_cfg.get("pos", DEFAULT_EYE_IN_HAND_CAMERA_POS), dtype=np.float64),
        "forward": np.asarray(camera_cfg.get("forward", DEFAULT_EYE_IN_HAND_CAMERA_FORWARD), dtype=np.float64),
        "up": np.asarray(camera_cfg.get("up", DEFAULT_EYE_IN_HAND_CAMERA_UP), dtype=np.float64),
        "fovy": float(camera_cfg.get("fovy", DEFAULT_EYE_IN_HAND_CAMERA_FOVY)),
    }
    errors: list[str] = []
    if not np.allclose(observed["pos"], expected["pos"], atol=1e-12, rtol=0.0):
        errors.append(f"pos={observed['pos'].tolist()!r}")
    if not np.allclose(observed["forward"], expected["forward"], atol=1e-12, rtol=0.0):
        errors.append(f"forward={observed['forward'].tolist()!r}")
    if not np.allclose(observed["up"], expected["up"], atol=1e-12, rtol=0.0):
        errors.append(f"up={observed['up'].tolist()!r}")
    if not np.isclose(observed["fovy"], expected["fovy"], atol=1e-12, rtol=0.0):
        errors.append(f"fovy={observed['fovy']!r}")
    if errors:
        raise ValueError(
            "Scene config eye-in-hand camera does not match the canonical contract "
            f"{CANONICAL_EYE_IN_HAND_CAMERA_POSE_VERSION!r}: " + ", ".join(errors)
        )
    return expected


def eye_in_hand_camera_pose_from_config(
    scene_config: dict[str, object],
    *,
    camera_name: str = DEFAULT_EYE_IN_HAND_CAMERA_NAME,
) -> tuple[np.ndarray, np.ndarray, float]:
    validate_canonical_eye_in_hand_camera_config(scene_config, camera_name=camera_name)
    vision = scene_config.get("vision", {})
    if not isinstance(vision, dict):
        raise TypeError("vision must be a mapping when provided.")
    camera_cfg = vision.get("eye_in_hand_camera", {})
    if not isinstance(camera_cfg, dict):
        raise TypeError("vision.eye_in_hand_camera must be a mapping when provided.")
    local_position = np.asarray(camera_cfg.get("pos", DEFAULT_EYE_IN_HAND_CAMERA_POS), dtype=np.float64)
    forward = np.asarray(camera_cfg.get("forward", DEFAULT_EYE_IN_HAND_CAMERA_FORWARD), dtype=np.float64)
    up = np.asarray(camera_cfg.get("up", DEFAULT_EYE_IN_HAND_CAMERA_UP), dtype=np.float64)
    right, true_up = _camera_xyaxes(forward=forward, up=up)
    rotation = np.column_stack([right, true_up, _normalized(forward)])
    fovy = float(camera_cfg.get("fovy", DEFAULT_EYE_IN_HAND_CAMERA_FOVY))
    return local_position, rotation, fovy


def apply_eye_in_hand_camera_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_id: int,
    local_position: np.ndarray,
    rotation: np.ndarray,
    fovy: float,
) -> None:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=np.float64).reshape(9))
    model.cam_pos[camera_id] = np.asarray(local_position, dtype=np.float64)
    model.cam_quat[camera_id] = quat
    model.cam_fovy[camera_id] = float(fovy)
    mujoco.mj_forward(model, data)


def eye_in_hand_camera_xml(camera: dict[str, object] | None = None) -> str:
    cfg = dict(camera or {})
    name = str(cfg.get("name", DEFAULT_EYE_IN_HAND_CAMERA_NAME))
    pos = tuple(float(value) for value in cfg.get("pos", DEFAULT_EYE_IN_HAND_CAMERA_POS))
    forward = cfg.get("forward", DEFAULT_EYE_IN_HAND_CAMERA_FORWARD)
    up = cfg.get("up", DEFAULT_EYE_IN_HAND_CAMERA_UP)
    fovy = float(cfg.get("fovy", DEFAULT_EYE_IN_HAND_CAMERA_FOVY))
    xaxis, yaxis = _camera_xyaxes(forward=forward, up=up)
    return (
        f'                      <camera name="{name}" pos="{fmt(pos)}" '
        f'xyaxes="{fmt(xaxis)} {fmt(yaxis)}" fovy="{fovy:g}"/>'
    )


def _convert_arm_actuators_to_torque(xml: str) -> str:
    root = ET.fromstring(xml)
    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError("Could not find Panda actuator block.")

    torque_limits = {
        "actuator1": "-87 87",
        "actuator2": "-87 87",
        "actuator3": "-87 87",
        "actuator4": "-87 87",
        "actuator5": "-12 12",
        "actuator6": "-12 12",
        "actuator7": "-12 12",
    }
    for idx, child in enumerate(list(actuator)):
        name = child.get("name")
        if name not in torque_limits:
            continue
        motor = ET.Element(
            "motor",
            {
                "name": name,
                "joint": f"joint{name.removeprefix('actuator')}",
                "ctrlrange": torque_limits[name],
                "forcerange": torque_limits[name],
            },
        )
        actuator[idx] = motor

    keyframe = root.find("keyframe")
    if keyframe is not None:
        for key in keyframe.findall("key"):
            ctrl = key.get("ctrl")
            if ctrl is None:
                continue
            values = ctrl.split()
            if len(values) >= 7:
                values[:7] = ["0"] * 7
                key.set("ctrl", " ".join(values))

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def _safe_model_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def render_panda_with_attached_peg(
    peg: dict[str, object],
    *,
    actuator_mode: str = "position",
    model_name: str = "panda_with_peg",
    vision_camera: dict[str, object] | None = None,
) -> Path:
    panda_xml = PANDA_XML.read_text(encoding="utf-8")
    panda_xml = panda_xml.replace('meshdir="assets"', 'meshdir="../../third_party/mujoco_menagerie/franka_emika_panda/assets"', 1)
    insertion_marker = '                      <body name="left_finger" pos="0 0 0.0584">'
    if insertion_marker not in panda_xml:
        raise RuntimeError("Could not find Panda hand insertion point in Menagerie panda.xml.")
    camera_xml = eye_in_hand_camera_xml(vision_camera) if vision_camera is not None else ""
    camera_block = f"{camera_xml}\n" if camera_xml else ""
    patched = panda_xml.replace(insertion_marker, f"{camera_block}{hand_peg_body_xml(peg)}\n{insertion_marker}", 1)
    stem = _safe_model_stem(model_name)
    if actuator_mode == "position":
        output_path = GENERATED_DIR / f"{stem}.xml"
    elif actuator_mode == "torque":
        output_path = GENERATED_DIR / f"{stem}_torque.xml"
        patched = _convert_arm_actuators_to_torque(patched)
    else:
        raise ValueError(f"Unsupported scene.actuator_mode {actuator_mode!r}. Use 'position' or 'torque'.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8")
    return output_path


def render(config: dict[str, object]) -> str:
    scene = config["scene"]
    robot = scene.get("robot", "panda")
    if robot != "panda":
        raise ValueError(f"Unsupported scene.robot {robot!r}. Current scene generator supports 'panda'.")
    actuator_mode = str(scene.get("actuator_mode", "position"))
    visual = config["visual"]
    table = config["table"]
    peg = config["peg"]
    hole = config["hole"]
    vision = config.get("vision", {})
    if vision is not None and not isinstance(vision, dict):
        raise TypeError("vision must be a mapping when provided.")
    friction = _contact_attr(config, "friction", (0.9, 0.02, 0.002))
    solref = _contact_attr(config, "solref", (0.004, 1.0))
    solimp = _contact_attr(config, "solimp", (0.95, 0.99, 0.001))
    hole_site_z = float(hole["wall_half_height"])
    peg_attachment = str(peg.get("attachment", "free"))
    robot_include = "panda.xml"
    if peg_attachment == "hand":
        robot_path = render_panda_with_attached_peg(
            peg,
            actuator_mode=actuator_mode,
            model_name=f"panda_with_peg_{scene['name']}",
            vision_camera=vision.get("eye_in_hand_camera") if isinstance(vision, dict) else None,
        )
        robot_include = f"../generated/{robot_path.name}"

    return f"""<mujoco model="{scene["name"]}">
  <include file="{robot_include}"/>

  <statistic center="{fmt(visual["statistic_center"])}" extent="{visual["statistic_extent"]:g}"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.18 0.22 1"/>
    <global azimuth="135" elevation="-25"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.30 0.42 0.55" rgb2="0.03 0.04 0.05" width="512" height="3072"/>
    <texture type="2d" name="table_grid" builtin="checker" mark="edge" rgb1="0.28 0.30 0.31" rgb2="0.20 0.22 0.23"
      markrgb="0.72 0.72 0.72" width="300" height="300"/>
    <material name="table_mat" texture="table_grid" texuniform="true" texrepeat="4 4" reflectance="0.12"/>
    <material name="peg_mat" rgba="0.92 0.42 0.24 1"/>
    <material name="hole_mat" rgba="0.24 0.28 0.32 1"/>
{peg_mesh_asset_xml(peg)}
  </asset>

  <default>
    <default class="task_collision">
      <geom condim="4" friction="{friction}" solref="{solref}" solimp="{solimp}"/>
    </default>
  </default>

  <worldbody>
    <light name="task_key" pos="0.2 -0.7 1.8" dir="0.2 0.4 -1" directional="true"/>
    <geom name="floor" type="plane" size="1.5 1.5 0.05" material="table_mat"/>

    <body name="table" pos="{fmt(table["pos"])}">
      <geom name="table_top" type="box" size="{fmt(table["size"])}" material="table_mat" class="task_collision"/>
    </body>

    <body name="hole_fixture" pos="{fmt(hole["pos"])}">
      <site name="hole_center" pos="0 0 {hole_site_z:g}" size="0.008" rgba="0.18 0.62 0.72 1"/>
{wall_geoms(hole)}
    </body>

{peg_body_xml(peg) if peg_attachment == "free" else ""}
  </worldbody>
{peg_equality_xml(peg)}
</mujoco>
"""


def render_config_file(config_path: Path, *, output_path: Path | None = None) -> Path:
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target_path = output_path or Path(config["scene"]["output"])
    return render_config_to_file(config, target_path)


def render_config_to_file(config: dict[str, object], output_path: Path) -> Path:
    target_path = output_path
    if not target_path.is_absolute():
        target_path = ROOT / target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"{target_path.suffix}.tmp",
        dir=target_path.parent,
        encoding="utf-8",
        delete=False,
    ) as handle:
        handle.write(render(config))
        tmp_path = Path(handle.name)
    tmp_path.replace(target_path)
    return target_path


def render_runtime_config(config: dict[str, object], *, prefix: str = "runtime_scene_") -> Path:
    runtime_dir = ROOT / "models" / "scenes"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=runtime_dir,
        prefix=prefix,
        suffix=".xml",
        delete=False,
    ) as handle:
        output_path = Path(handle.name)
    return render_config_to_file(config, output_path)


def cleanup_runtime_scene(scene_path: Path) -> None:
    if scene_path.name.startswith("runtime_"):
        try:
            scene_path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    config_path = Path(args[0]) if args else DEFAULT_CONFIG
    output_path = render_config_file(config_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
