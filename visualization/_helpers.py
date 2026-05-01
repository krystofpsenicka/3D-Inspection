"""Shared visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


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


def show_geometries(
    geometries: list,
    window_name: str = "Open3D",
    width: int = 1920,
    height: int = 1080,
    point_size: float = 4.0,
    line_width: float = 2.0,
    mesh_show_back_face: bool = True,
):
    """Display a list of Open3D geometries with standard render options."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)

    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.line_width = line_width
    render_option.mesh_show_back_face = mesh_show_back_face

    for geom in geometries:
        vis.add_geometry(geom)

    vis.run()
    vis.destroy_window()
