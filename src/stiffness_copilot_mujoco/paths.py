from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
SCENES_ROOT = MODELS_ROOT / "scenes"
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"
MENAGERIE_ROOT = THIRD_PARTY_ROOT / "mujoco_menagerie"
PANDA_SCENE = MENAGERIE_ROOT / "franka_emika_panda" / "scene.xml"
FR3_SCENE = MENAGERIE_ROOT / "franka_fr3" / "scene.xml"
