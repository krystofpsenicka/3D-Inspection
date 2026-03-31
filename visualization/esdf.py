"""Visualize ESDF voxel grids in 3D (Open3D) and 2D (matplotlib)."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d

from shared.grid_utils import esdf_to_rgb
from shared.occupancy_grid import OccupancyGrid

logger = logging.getLogger(__name__)


class EsdfVisualizer:
    """Shared ESDF visualizer used by both VRP and visibility ESDF scripts.

    Parameters
    ----------
    occupancy_grid : OccupancyGrid object.
    raw_grid : Pre-inflation boolean occupancy grid.
    esdf : 3-D float array of signed distances (Euclidean Signed Distance Field).
    scaled_mesh : Optional trimesh mesh (already scaled/posed) for overlays.
    """

    def __init__(self, occupancy_grid: OccupancyGrid, raw_grid: np.ndarray,
                 esdf: np.ndarray, scaled_mesh: o3d.geometry.TriangleMesh = None):
        self.og = occupancy_grid
        self.raw = raw_grid
        self.esdf = esdf
        self.scaled_mesh = scaled_mesh

    # ------------------------------------------------------------------
    # 3-D
    # ------------------------------------------------------------------

    def visualize_3d(self, esdf_band: float = 2.0,
                     show_occupied: bool = False,
                     show_inflated: bool = False,
                     show_mesh: bool = True,
                     max_points: int = 300_000):
        """Open an interactive Open3D viewer of the ESDF voxel cloud."""
        geometries: list = []

        # 1. Ship mesh (grey)
        if show_mesh and self.scaled_mesh is not None:
            try:
                self.scaled_mesh.compute_vertex_normals()
                self.scaled_mesh.paint_uniform_color([0.6, 0.6, 0.6])
                geometries.append(self.scaled_mesh)
                print(f"Mesh: {len(self.scaled_mesh.vertices):,} verts, "
                      f"{len(self.scaled_mesh.faces):,} faces")
            except Exception as e:
                print(f"Could not load mesh: {e}")

        # 2. Near-surface ESDF voxels coloured by distance
        mask = np.abs(self.esdf) < esdf_band
        ijk = np.argwhere(mask)
        if len(ijk) > 0:
            if len(ijk) > max_points:
                rng = np.random.RandomState(42)
                ijk = ijk[rng.choice(len(ijk), max_points, replace=False)]
            centres = (self.og.origin
                       + (ijk.astype(np.float64) + 0.5) * self.og.resolution)
            values = self.esdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            colors = esdf_to_rgb(values, vmin=-esdf_band, vmax=esdf_band * 0.5)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(centres)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geometries.append(pcd)
            print(f"ESDF band  |d| < {esdf_band}m : {len(ijk):,} points")
        else:
            print("WARNING: no voxels fall within ESDF band")

        # 3. Raw occupied voxels (red)
        if show_occupied:
            occ_ijk = np.argwhere(self.raw)
            if len(occ_ijk) > max_points:
                rng = np.random.RandomState(42)
                occ_ijk = occ_ijk[rng.choice(len(occ_ijk), max_points, replace=False)]
            pts = (self.og.origin
                   + (occ_ijk.astype(np.float64) + 0.5) * self.og.resolution)
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pts)
            pcd2.paint_uniform_color([1.0, 0.3, 0.3])
            geometries.append(pcd2)
            print(f"Occupied voxels      : {len(occ_ijk):,} points")

        # 4. Inflation shell (blue)
        if show_inflated:
            shell = self.og.grid & ~self.raw
            shell_ijk = np.argwhere(shell)
            if len(shell_ijk) > max_points:
                rng = np.random.RandomState(99)
                shell_ijk = shell_ijk[rng.choice(len(shell_ijk), max_points,
                                                  replace=False)]
            pts = (self.og.origin
                   + (shell_ijk.astype(np.float64) + 0.5) * self.og.resolution)
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(pts)
            pcd3.paint_uniform_color([0.3, 0.3, 1.0])
            geometries.append(pcd3)
            print(f"Inflation shell      : {len(shell_ijk):,} points")

        # 5. Coordinate frame
        geometries.append(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

        print()
        print("Opening Open3D viewer ...")
        print("  Grey mesh  ")
        print("  Red -> White -> Blue = ESDF  (red = inside obstacle, blue = free space)")
        if show_occupied:
            print("  Red points          = raw occupied voxels")
        if show_inflated:
            print("  Blue points         = inflation shell")
        print("  Controls: left-drag = rotate | scroll = zoom | middle-drag = pan")

        o3d.visualization.draw_geometries(
            geometries, window_name="ESDF 3D", width=1600, height=900)

    # ------------------------------------------------------------------
    # 2-D matplotlib slice
    # ------------------------------------------------------------------

    def visualize_2d(self, axis: int = 2, slice_pos: float = 1.5,
                     vmin: float = -3.0, vmax: float = 1.0,
                     show_mesh: bool = True):
        """Show a 2-D ESDF + occupancy slice in matplotlib."""
        import matplotlib.pyplot as plt

        ax = axis
        axis_names = ["X", "Y", "Z"]
        slice_idx = int(round((slice_pos - self.og.origin[ax]) / self.og.resolution))
        slice_idx = int(np.clip(slice_idx, 0, self.raw.shape[ax] - 1))
        actual_pos = self.og.origin[ax] + (slice_idx + 0.5) * self.og.resolution
        print(f"Slicing at {axis_names[ax]}={actual_pos:.2f}m  (index {slice_idx})")

        if ax == 0:
            esdf_slice = self.esdf[slice_idx, :, :].T
            occ_slice = self.raw[slice_idx, :, :].T
            xlabel, ylabel = "Y (m)", "Z (m)"
            x_origin, y_origin = self.og.origin[1], self.og.origin[2]
            nx, ny = self.raw.shape[1], self.raw.shape[2]
        elif ax == 1:
            esdf_slice = self.esdf[:, slice_idx, :].T
            occ_slice = self.raw[:, slice_idx, :].T
            xlabel, ylabel = "X (m)", "Z (m)"
            x_origin, y_origin = self.og.origin[0], self.og.origin[2]
            nx, ny = self.raw.shape[0], self.raw.shape[2]
        else:
            esdf_slice = self.esdf[:, :, slice_idx].T
            occ_slice = self.raw[:, :, slice_idx].T
            xlabel, ylabel = "X (m)", "Y (m)"
            x_origin, y_origin = self.og.origin[0], self.og.origin[1]
            nx, ny = self.raw.shape[0], self.raw.shape[1]

        extent = [x_origin, x_origin + nx * self.og.resolution,
                  y_origin, y_origin + ny * self.og.resolution]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        ax1 = axes[0]
        im = ax1.imshow(esdf_slice, origin="lower", extent=extent,
                        cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="equal")
        ax1.contour(esdf_slice, levels=[0.0], colors="white", linewidths=1.5,
                    origin="lower", extent=extent)
        plt.colorbar(im, ax=ax1, label="ESDF (m) — +ve inside obstacle")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title(f"ESDF slice at {axis_names[ax]}={actual_pos:.2f}m")

        ax2 = axes[1]
        ax2.imshow(occ_slice.astype(float), origin="lower", extent=extent,
                   cmap="Greys", vmin=0, vmax=1, aspect="equal")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        ax2.set_title(
            f"Raw occupancy (pre-inflation) at {axis_names[ax]}={actual_pos:.2f}m")

        if show_mesh and self.scaled_mesh is not None:
            try:
                plane_origin = [0.0, 0.0, 0.0]
                plane_normal = [0.0, 0.0, 0.0]
                plane_origin[ax] = actual_pos
                plane_normal[ax] = 1.0
                cross = self.scaled_mesh.section(plane_origin=plane_origin,
                                                 plane_normal=plane_normal)
                if cross is not None:
                    for entity in cross.entities:
                        pts = cross.vertices[entity.points]
                        if ax == 0:
                            ax1.plot(pts[:, 1], pts[:, 2], "lime", lw=0.8)
                            ax2.plot(pts[:, 1], pts[:, 2], "lime", lw=0.8)
                        elif ax == 1:
                            ax1.plot(pts[:, 0], pts[:, 2], "lime", lw=0.8)
                            ax2.plot(pts[:, 0], pts[:, 2], "lime", lw=0.8)
                        else:
                            ax1.plot(pts[:, 0], pts[:, 1], "lime", lw=0.8)
                            ax2.plot(pts[:, 0], pts[:, 1], "lime", lw=0.8)
                    print("Mesh cross-section overlaid (green lines)")
                else:
                    print("Mesh cross-section is empty at this slice height")
            except Exception as e:
                print(f"Mesh overlay failed: {e}")

        plt.tight_layout()
        out_path = f"esdf_slice_{axis_names[ax]}{actual_pos:.1f}.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved to {out_path}")
        plt.show()
