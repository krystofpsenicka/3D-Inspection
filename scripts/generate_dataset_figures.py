"""Render the two dataset figures referenced by Bachelor-Thesis/chap03.tex § 3.3.

Outputs:
  Bachelor-Thesis/img/datasets/duke_of_lancaster.png
  Bachelor-Thesis/img/datasets/tosca_samples.png

Run:
  conda run -n isaaclab python 3D-Inspection/scripts/generate_dataset_figures.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dataset_figures")


REPO_ROOT = Path(__file__).resolve().parents[2]
INSPECTION_ROOT = REPO_ROOT / "3D-Inspection"
THESIS_IMG = REPO_ROOT / "Bachelor-Thesis" / "img" / "datasets"

DUKE_GLB = INSPECTION_ROOT / "models" / "duke_of_lancaster_uk_clipped.glb"
TOSCA_DIR = INSPECTION_ROOT / "models" / "TOSCA-dataset"

# Per the thesis text (chap03 § "Test Environment"), the Duke mesh is scaled at
# pipeline-time so that its longest axis matches a target length of 50 m.
DUKE_TARGET_LENGTH = 50.0


def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
    )
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    # Some TOSCA .off files have degenerate triangles that produce NaN normals.
    normals = np.asarray(o3d_mesh.vertex_normals)
    if not np.all(np.isfinite(normals)):
        normals = np.nan_to_num(normals, nan=0.0)
        # Re-normalize anything we touched.
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    o3d_mesh.paint_uniform_color([0.78, 0.78, 0.80])
    return o3d_mesh


def _render_mesh_offscreen(
    mesh: o3d.geometry.TriangleMesh,
    *,
    width: int,
    height: int,
    front: tuple[float, float, float],
    up: tuple[float, float, float],
    zoom: float = 0.7,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    lookat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Render mesh to an HxWx3 uint8 array via an invisible Open3D window.

    `lookat_offset` shifts the camera lookat point in world coordinates,
    which translates the mesh in the opposite direction in screen space.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    try:
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.background_color = np.asarray(background, dtype=np.float64)
        opt.mesh_show_back_face = True
        opt.light_on = True

        ctr = vis.get_view_control()
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center() + np.asarray(lookat_offset, dtype=np.float64)
        ctr.set_lookat(center)
        ctr.set_front(np.asarray(front, dtype=np.float64))
        ctr.set_up(np.asarray(up, dtype=np.float64))
        ctr.set_zoom(zoom)

        vis.poll_events()
        vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)
    finally:
        vis.destroy_window()

    img = (np.asarray(buf) * 255.0).clip(0, 255).astype(np.uint8)
    return img


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_duke(out_path: Path) -> None:
    log.info("Loading Duke of Lancaster mesh ...")
    tm = trimesh.load(str(DUKE_GLB), force="mesh")
    extents_native = tm.extents.copy()

    scale_factor = DUKE_TARGET_LENGTH / float(extents_native.max())
    extents_scaled = extents_native * scale_factor
    log.info(
        "  native extents (m): %.2f x %.2f x %.2f -> scaled to %.2f x %.2f x %.2f",
        *extents_native, *extents_scaled,
    )

    mesh = _trimesh_to_o3d(tm)

    # The Duke mesh's longest axis is X (~21 m). The GLB stores +Y as "down"
    # in this case, so we flip the world up vector. Render from a side-oblique
    # angle that shows both length and superstructure detail.
    img = _render_mesh_offscreen(
        mesh,
        width=2200, height=1100,
        front=(0.45, -0.55, 0.70),
        up=(0.0, -1.0, 0.0),
        zoom=0.17,
        lookat_offset=(5, 0.0, 2.0),
    )

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = _load_font(36)

    L_scaled = extents_scaled[0]   # X = length
    H_scaled = extents_scaled[1]   # Y = height
    W_scaled = extents_scaled[2]   # Z = width / beam
    label = (f"≈ {L_scaled:.0f} m (length)  ×  "
             f"{W_scaled:.1f} m (beam)  ×  "
             f"{H_scaled:.1f} m (height)")

    text_xy = (40, pil.height - 70)
    bbox = draw.textbbox(text_xy, label, font=font)
    pad = 10
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=(255, 255, 255), outline=(80, 80, 80), width=1,
    )
    draw.text(text_xy, label, fill=(20, 20, 20), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    log.info("Saved %s (%dx%d)", out_path, pil.width, pil.height)


# Selection of TOSCA classes for the sample grid. Six distinct classes
# (humans + animals) using the lowest-index pose available.
TOSCA_PICKS: list[tuple[str, str]] = [
    ("cat0",      "Cat"),
    ("centaur0",  "Centaur"),
    ("david0",    "David"),
    ("gorilla1",  "Gorilla"),  # gorilla0.off is not present in the dataset
    ("horse0",    "Horse"),
    ("michael0",  "Michael"),
]


def _render_tosca_tile(file_stem: str, tile_size: int) -> np.ndarray:
    path = TOSCA_DIR / f"{file_stem}.off"
    log.info("  rendering %s", path.name)
    tm = trimesh.load(str(path), force="mesh")
    mesh = _trimesh_to_o3d(tm)

    # TOSCA meshes are not consistently oriented across classes; for most poses
    # the up axis is +Z (humans/quadrupeds standing upright). Keep a single
    # camera convention so the grid feels uniform.
    return _render_mesh_offscreen(
        mesh,
        width=tile_size, height=tile_size,
        front=(0.4, -0.9, 0.45),
        up=(0.0, 0.0, 1.0),
        zoom=0.85,
    )


def render_tosca(out_path: Path) -> None:
    log.info("Rendering TOSCA sample grid ...")

    cols, rows = 3, 2
    tile = 720
    label_h = 60
    border = 6

    cell_w = tile + 2 * border
    cell_h = tile + label_h + 2 * border
    grid_w = cell_w * cols
    grid_h = cell_h * rows

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(32)

    for idx, (stem, pretty) in enumerate(TOSCA_PICKS):
        r, c = divmod(idx, cols)
        tile_img = Image.fromarray(_render_tosca_tile(stem, tile))

        cx = c * cell_w
        cy = r * cell_h
        # Light-grey border around tile.
        draw.rectangle(
            (cx, cy + label_h, cx + cell_w, cy + cell_h),
            outline=(180, 180, 180), width=border,
        )
        canvas.paste(tile_img, (cx + border, cy + label_h + border))

        # Class label, centered above the tile.
        bbox = draw.textbbox((0, 0), pretty, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (cx + (cell_w - text_w) // 2, cy + (label_h - text_h) // 2 - 6),
            pretty, fill=(20, 20, 20), font=font,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    log.info("Saved %s (%dx%d)", out_path, canvas.width, canvas.height)


def main() -> None:
    THESIS_IMG.mkdir(parents=True, exist_ok=True)
    render_duke(THESIS_IMG / "duke_of_lancaster.png")
    render_tosca(THESIS_IMG / "tosca_samples.png")
    log.info("Done.")


if __name__ == "__main__":
    main()
