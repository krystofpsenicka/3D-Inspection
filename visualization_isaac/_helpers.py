"""Shared visualization utilities (Isaac Sim variant — no Open3D)."""

import matplotlib.pyplot as plt
import numpy as np


def generate_tab20_colors(n: int) -> list[tuple]:
    """tab20 colours with near-reds filtered out (red is reserved for uncovered points)."""
    raw = plt.cm.tab20(np.linspace(0, 1, max(20, n)))
    colors = []
    for c in raw:
        r, g, b = c[:3]
        if r > 0.7 and g < 0.3 and b < 0.3:
            continue
        colors.append((r, g, b))
    return colors


def rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_matrix(np.asarray(rotmat)).as_quat(scalar_first=True).astype(np.float64)
