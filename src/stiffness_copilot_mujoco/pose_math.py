from __future__ import annotations

import numpy as np


def site_rotation(data, site_id: int) -> np.ndarray:
    return np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)


def orientation_error(current_rotation: np.ndarray, target_rotation: np.ndarray) -> np.ndarray:
    current = np.asarray(current_rotation, dtype=float).reshape(3, 3)
    target = np.asarray(target_rotation, dtype=float).reshape(3, 3)
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )
