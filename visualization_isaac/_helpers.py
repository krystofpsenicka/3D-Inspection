"""Shared visualization utilities (Isaac Sim variant — no Open3D dependency)."""

import numpy as np
import matplotlib.pyplot as plt


def generate_tab20_colors(n: int) -> list[tuple]:
    """Generate tab20 colours, filtering out near-reds (reserved for uncovered points).

    Parameters
    ----------
    n : Number of distinct colours needed.

    Returns
    -------
    List of (r, g, b) tuples with at least *n* entries (wraps if necessary).
    """
    raw = plt.cm.tab20(np.linspace(0, 1, max(20, n)))
    colors = []
    for c in raw:
        r, g, b = c[:3]
        if r > 0.7 and g < 0.3 and b < 0.3:
            continue
        colors.append((r, g, b))
    return colors
