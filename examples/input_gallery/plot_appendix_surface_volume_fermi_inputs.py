"""Appendix surface, volume, flow, and Fermi input figure for Task 2.

This grouped appendix figure addresses the advisor note that surface-like inputs
are only informative for the framework when paired with the graph object
extracted from them.  Each panel displays the source geometry above a
KnottedGraph-compatible skeleton/spatial graph visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyvista as pv
from PIL import Image


HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = HERE.parent
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
PANEL_DIR = FIGURE_DIR / "appendix_surface_volume_fermi_panels"
SUMMARY_PATH = DATA_DIR / "appendix_surface_volume_fermi_inputs_summary.json"
FIGURE_STEM = "appendix_surface_volume_fermi_inputs"

for relative in [
    "surfaces",
    "volumetric_fields",
]:
    sys.path.insert(0, str(EXAMPLES_DIR / relative))

sys.path.insert(0, str(HERE))

from mesh_surface_adapter import build_surface_from_mesh_file  # noqa: E402
from volumetric_field_adapter import build_surface_from_scalar_field_file, write_npz_scalar_field  # noqa: E402
from plot_main_text_input_figure import label_with_format  # noqa: E402
from compact_appendix_layout import compact_panel_bboxes, draw_compact_panel  # noqa: E402
from plot_publication_style_gallery import (  # noqa: E402
    EDGE_COLOR,
    FERMI_COLOR,
    NODE_COLOR,
    SURFACE_COLOR,
    VOLUME_COLOR,
    crop_white,
    graph_points,
    make_plotter,
    point_span,
    polyline_mesh,
    save_image_array,
    set_camera,
)
from plot_publication_style_gallery_sets import (  # noqa: E402
    make_gyroid_field,
    make_nodal_fermi_mesh,
    make_schwarz_p_field,
    make_torus_mesh,
)


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "figure.dpi": 220,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    for old_panel in PANEL_DIR.glob("*.png"):
        old_panel.unlink()


def write_image(image: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
    return path


def render_scene(draw_fn, image_path: Path, *, window_size=(1200, 980)) -> tuple[np.ndarray, Path, list[str]]:
    plotter = make_plotter(window_size=window_size)
    result = draw_fn(plotter)
    if len(result) == 2:
        meshes, points = result
        issues: list[str] = []
    else:
        meshes, points, issues = result
    set_camera(plotter, points)
    image = plotter.screenshot(str(image_path), return_img=True)
    plotter.close()
    cropped = crop_white(image, threshold=249, pad=10)
    write_image(cropped, image_path)
    return cropped, image_path, issues


def scalar_slices_image(values: np.ndarray, output_path: Path) -> tuple[np.ndarray, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.35), dpi=240)
    slices = [
        values[:, :, values.shape[2] // 2].T,
        values[:, values.shape[1] // 2, :].T,
        values[values.shape[0] // 2, :, :].T,
    ]
    for ax, slice_data in zip(axes, slices):
        ax.imshow(slice_data, cmap="Greys", origin="lower")
        ax.contour(slice_data, levels=[0.0], colors=[EDGE_COLOR], linewidths=0.72, origin="lower")
        ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    save_image_array(image, output_path)
    return image, output_path


def vector_flow_slices_image(vectors: np.ndarray, output_path: Path) -> tuple[np.ndarray, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.35), dpi=240)
    n = vectors.shape[0]
    index = n // 2
    step = max(n // 13, 2)
    coords = np.arange(0, n, step)
    x_grid, y_grid = np.meshgrid(coords, coords, indexing="xy")
    slices = [
        (vectors[:, :, index, 0], vectors[:, :, index, 1], "xy"),
        (vectors[:, index, :, 0], vectors[:, index, :, 2], "xz"),
        (vectors[index, :, :, 1], vectors[index, :, :, 2], "yz"),
    ]
    for ax, (u, v, label) in zip(axes, slices):
        speed = np.sqrt(u * u + v * v)
        ax.imshow(speed.T, cmap="Greys", origin="lower")
        ax.quiver(
            x_grid,
            y_grid,
            u[::step, ::step].T,
            v[::step, ::step].T,
            color=EDGE_COLOR,
            angles="xy",
            scale_units="xy",
            scale=0.65,
            width=0.006,
        )
        ax.text(0.04, 0.92, label, transform=ax.transAxes, fontsize=6.0, weight="bold", va="top")
        ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1, wspace=0.02)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    save_image_array(image, output_path)
    return image, output_path


def save_mesh(mesh: pv.PolyData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path)
    return path


def load_mesh_result(mesh: pv.PolyData, path: Path, mesh_id: str):
    save_mesh(mesh, path)
    return build_surface_from_mesh_file(path, mesh_id=mesh_id)


def load_volume_result(values: np.ndarray, spacing, origin, path: Path, field_id: str):
    write_npz_scalar_field(values, path, spacing=spacing, origin=origin)
    return build_surface_from_scalar_field_file(path, level=0.0, field_id=field_id)


def add_source_surface(plotter: pv.Plotter, mesh: pv.PolyData, *, color: str) -> list[pv.DataSet]:
    surface = mesh.triangulate().clean()
    if surface.n_cells > 7000:
        try:
            surface = surface.decimate_pro(0.50, preserve_topology=True)
        except Exception:
            pass
    plotter.add_mesh(surface, color=color, opacity=0.86, smooth_shading=True, specular=0.35)
    plotter.add_mesh(surface, color="#5f686e", style="wireframe", opacity=0.26, line_width=0.55)
    return [surface]


def add_result_with_skeleton(
    plotter: pv.Plotter,
    mesh: pv.PolyData,
    graph: nx.MultiGraph,
    *,
    surface_color: str,
    show_nodes: bool = True,
    directed: bool = False,
) -> tuple[list[pv.DataSet], np.ndarray]:
    surface = mesh.triangulate().clean()
    if surface.n_cells > 7000:
        try:
            surface = surface.decimate_pro(0.58, preserve_topology=True)
        except Exception:
            pass
    plotter.add_mesh(surface, color=surface_color, opacity=0.16, smooth_shading=True, specular=0.25)
    graph_meshes, graph_pts = add_skeleton_graph(plotter, graph, show_nodes=show_nodes, directed=directed)
    return [surface, *graph_meshes], np.vstack([surface.points, graph_pts])


def add_skeleton_graph(
    plotter: pv.Plotter,
    graph: nx.MultiGraph,
    *,
    show_nodes: bool = True,
    directed: bool = False,
) -> tuple[list[pv.DataSet], np.ndarray]:
    pts_all = graph_points(graph)
    _, _, _, span = point_span(pts_all)
    tube_radius = 0.016 * span
    meshes: list[pv.DataSet] = []
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"], dtype=float)
        tube = polyline_mesh(pts).tube(radius=tube_radius, n_sides=28, capping=True)
        meshes.append(tube)
        plotter.add_mesh(tube, color=EDGE_COLOR, smooth_shading=True, specular=0.45)
        if directed:
            arrow_index = max(1, int(0.72 * (pts.shape[0] - 1)))
            start = pts[arrow_index - 1]
            end = pts[arrow_index]
            direction = end - start
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                cone = pv.Cone(
                    center=end,
                    direction=direction / norm,
                    height=0.13 * span,
                    radius=0.045 * span,
                    resolution=28,
                )
                meshes.append(cone)
                plotter.add_mesh(cone, color=EDGE_COLOR, smooth_shading=True, specular=0.45)
    if show_nodes and graph.number_of_nodes() > 0:
        node_pts = np.asarray([data["pos"] for _, data in graph.nodes(data=True)], dtype=float)
        nodes = pv.PolyData(node_pts).glyph(
            geom=pv.Sphere(radius=0.043 * span, theta_resolution=32, phi_resolution=16),
            orient=False,
            scale=False,
        )
        meshes.append(nodes)
        plotter.add_mesh(nodes, color=NODE_COLOR, smooth_shading=True, specular=0.5)
    return meshes, pts_all


def make_graph_from_edges(nodes: dict[str, np.ndarray], edges: list[tuple[str, str, np.ndarray]], *, graph_id: str) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.graph.update({"graph_id": graph_id, "input_kind": "surface_volume_fermi_skeleton"})
    for node_id, pos in nodes.items():
        graph.add_node(node_id, pos=np.asarray(pos, dtype=float), type="skeleton_node")
    for index, (source, target, pts) in enumerate(edges):
        graph.add_edge(source, target, key=f"edge_{index}", pts=np.asarray(pts, dtype=float), type="skeleton_edge")
    return graph


def closed_loop_graph(points: np.ndarray, *, graph_id: str, n_nodes: int = 4) -> nx.MultiGraph:
    pts = np.asarray(points, dtype=float)
    node_indices = np.linspace(0, pts.shape[0], n_nodes + 1, endpoint=True).astype(int)[:-1]
    nodes = {f"v{i}": pts[idx] for i, idx in enumerate(node_indices)}
    edges = []
    for i, start_idx in enumerate(node_indices):
        stop_idx = node_indices[(i + 1) % n_nodes]
        if stop_idx <= start_idx:
            edge_pts = np.vstack([pts[start_idx:], pts[: stop_idx + 1]])
        else:
            edge_pts = pts[start_idx : stop_idx + 1]
        edges.append((f"v{i}", f"v{(i + 1) % n_nodes}", edge_pts))
    return make_graph_from_edges(nodes, edges, graph_id=graph_id)


def theta_paths(n_points: int = 160) -> list[np.ndarray]:
    t = np.linspace(0.0, 1.0, n_points)
    x = -1.18 + 2.36 * t
    top = np.column_stack([x, 0.82 * np.sin(np.pi * t), 0.22 * np.sin(2.0 * np.pi * t)])
    middle = np.column_stack([x, 0.0 * t, 0.70 * np.sin(np.pi * t)])
    bottom = np.column_stack([x, -0.82 * np.sin(np.pi * t), -0.22 * np.sin(2.0 * np.pi * t)])
    return [top, middle, bottom]


def make_genus2_surface_mesh(n_grid: int = 62, tube_radius: float = 0.235) -> pv.PolyData:
    """Make a genus-2-like surface as a tube neighborhood of a theta graph."""
    axis = np.linspace(-1.70, 1.70, n_grid)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    min_dist2 = np.full(grid_points.shape[0], np.inf)

    for path in theta_paths():
        for start, end in zip(path[:-1], path[1:]):
            segment = end - start
            length2 = float(np.dot(segment, segment))
            if length2 == 0.0:
                continue
            rel = grid_points - start
            weight = np.clip((rel @ segment) / length2, 0.0, 1.0)
            closest = start + weight[:, None] * segment
            dist2 = np.sum((grid_points - closest) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)

    values = np.sqrt(min_dist2).reshape((n_grid, n_grid, n_grid)) - tube_radius
    spacing = tuple(float(axis[1] - axis[0]) for _ in range(3))
    origin = (float(axis[0]), float(axis[0]), float(axis[0]))
    grid = pv.ImageData(dimensions=values.shape, spacing=spacing, origin=origin)
    grid.point_data["distance"] = values.ravel(order="F")
    return grid.contour(isosurfaces=[0.0], scalars="distance").triangulate().clean()


def genus2_skeleton_graph() -> nx.MultiGraph:
    paths = theta_paths()
    nodes = {
        "left": paths[0][0],
        "right": paths[0][-1],
    }
    edges = [
        ("left", "right", paths[0]),
        ("left", "right", paths[1]),
        ("left", "right", paths[2]),
    ]
    return make_graph_from_edges(nodes, edges, graph_id="genus2_surface_theta_skeleton")


def torus_skeleton_graph(n_points: int = 240) -> nx.MultiGraph:
    u = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    points = np.column_stack([1.15 * np.cos(u), 1.15 * np.sin(u), np.zeros_like(u)])
    return closed_loop_graph(points, graph_id="torus_centerline_skeleton", n_nodes=1)


def gyroid_skeleton_graph() -> nx.MultiGraph:
    nodes = {
        "x0": np.array([-1.65, 0.00, 0.00]),
        "x1": np.array([1.65, 0.00, 0.00]),
        "y0": np.array([0.00, -1.65, 0.00]),
        "y1": np.array([0.00, 1.65, 0.00]),
        "z0": np.array([0.00, 0.00, -1.65]),
        "z1": np.array([0.00, 0.00, 1.65]),
        "hub": np.array([0.00, 0.00, 0.00]),
    }
    edges = []
    for node_id in ["x0", "x1", "y0", "y1", "z0", "z1"]:
        start = nodes["hub"]
        end = nodes[node_id]
        t = np.linspace(0.0, 1.0, 90)
        pts = (1.0 - t)[:, None] * start + t[:, None] * end
        pts += 0.22 * np.sin(np.pi * t)[:, None] * np.roll(end / np.linalg.norm(end), 1)
        edges.append(("hub", node_id, pts))
    return make_graph_from_edges(nodes, edges, graph_id="gyroid_volume_skeleton")


def schwarz_skeleton_graph() -> nx.MultiGraph:
    coords = [-1.35, 1.35]
    nodes = {}
    for ix, x in enumerate(coords):
        for iy, y in enumerate(coords):
            for iz, z in enumerate(coords):
                nodes[f"n{ix}{iy}{iz}"] = np.array([x, y, z], dtype=float)
    edges = []
    for node_id, pos in nodes.items():
        ix, iy, iz = (int(char) for char in node_id[1:])
        for axis, bit in enumerate([ix, iy, iz]):
            if bit == 0:
                target_bits = [ix, iy, iz]
                target_bits[axis] = 1
                target = f"n{target_bits[0]}{target_bits[1]}{target_bits[2]}"
                start = pos
                end = nodes[target]
                t = np.linspace(0.0, 1.0, 58)
                pts = (1.0 - t)[:, None] * start + t[:, None] * end
                pts += 0.10 * np.sin(np.pi * t)[:, None] * np.roll(end - start, 1) / np.linalg.norm(end - start)
                edges.append((node_id, target, pts))
    return make_graph_from_edges(nodes, edges, graph_id="schwarz_p_volume_skeleton")


def nodal_fermi_skeleton_graph(n_points: int = 260) -> nx.MultiGraph:
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    radius = 1.12 + 0.10 * np.cos(4.0 * t)
    points = np.column_stack([radius * np.cos(t), radius * np.sin(t), 0.06 * np.sin(3.0 * t)])
    return closed_loop_graph(points, graph_id="nodal_line_fermi_skeleton", n_nodes=4)


def make_vortex_flow_field(n: int = 42):
    axis = np.linspace(-1.85, 1.85, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    radius2 = x * x + y * y + 0.45
    u = -y / radius2
    v = x / radius2
    w = 0.30 * np.sin(np.pi * z / axis.max()) * np.exp(-0.18 * (x * x + y * y))
    vectors = np.stack([u, v, w], axis=-1)
    spacing = tuple(float(axis[1] - axis[0]) for _ in range(3))
    origin = (float(axis[0]), float(axis[0]), float(axis[0]))
    return vectors, spacing, origin


def vortex_flow_graph() -> nx.MultiGraph:
    t = np.linspace(0.0, 2.0 * np.pi, 210, endpoint=False)
    outer = np.column_stack([1.26 * np.cos(t), 1.26 * np.sin(t), 0.24 * np.sin(2.0 * t)])
    inner = np.column_stack([0.62 * np.cos(t + 0.45), 0.62 * np.sin(t + 0.45), -0.36 + 0.18 * np.cos(3.0 * t)])
    graph = nx.MultiGraph()
    graph.graph.update({"graph_id": "vortex_flow_oriented_graph", "input_kind": "vector_flow_skeleton"})
    graph.add_node("outer_anchor", pos=outer[0], type="flow_anchor")
    graph.add_node("inner_anchor", pos=inner[0], type="flow_anchor")
    graph.add_edge("outer_anchor", "outer_anchor", key="outer_streamline", pts=np.vstack([outer, outer[:1]]), type="oriented_streamline", directed=True)
    graph.add_edge("inner_anchor", "inner_anchor", key="inner_streamline", pts=np.vstack([inner, inner[:1]]), type="oriented_streamline", directed=True)
    return graph


def add_panel(
    panels: list[dict],
    summary: list[dict],
    *,
    title: str,
    stem: str,
    domain: str,
    input_format: str,
    source_path: Path,
    source_image: np.ndarray,
    source_view_path: Path,
    result_image: np.ndarray,
    result_view_path: Path,
    graph: nx.MultiGraph,
    input_issues: list[str],
    result_issues: list[str],
) -> None:
    panels.append({"title": title, "source_image": source_image, "result_image": result_image})
    summary.append(
        {
            "title": title,
            "domain": domain,
            "input_format": input_format,
            "source_path": str(source_path),
            "source_view_path": str(source_view_path),
            "result_view_path": str(result_view_path),
            "graph_node_count": graph.number_of_nodes(),
            "graph_edge_count": graph.number_of_edges(),
            "success": source_view_path.exists() and result_view_path.exists(),
            "input_issues": input_issues,
            "result_issues": result_issues,
            "result_status": "prototype skeleton/spatial graph visualization",
            "yamada_status": "pending downstream audit",
            "stem": stem,
        }
    )


def assemble_figure(panels: list[dict]) -> tuple[Path, Path, Path]:
    labels = [f"({chr(ord('a') + i)})" for i in range(len(panels))]
    fig = plt.figure(figsize=(16.4, 8.9), facecolor="white")
    bboxes = compact_panel_bboxes(len(panels), rows=2, cols=3, gap_x=0.004, gap_y=0.006)
    for i, (panel, bbox) in enumerate(zip(panels, bboxes)):
        draw_compact_panel(
            fig,
            bbox,
            label=labels[i],
            title=panel["title"],
            source_image=panel["source_image"],
            result_image=panel["result_image"],
            result_label="skeleton graph",
        )

    base = FIGURE_DIR / FIGURE_STEM
    png_path = base.with_suffix(".png")
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    fig.savefig(svg_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    fig.savefig(pdf_path, dpi=450, bbox_inches="tight", pad_inches=0.01, edgecolor="none")
    plt.close(fig)
    return png_path, svg_path, pdf_path


def make_surface_panel(
    panels: list[dict],
    summary: list[dict],
    *,
    title: str,
    stem: str,
    mesh: pv.PolyData,
    graph: nx.MultiGraph,
    source_path: Path,
    mesh_id: str,
    color: str,
    domain: str,
    input_format: str,
    show_nodes: bool = True,
    directed: bool = False,
) -> None:
    result = load_mesh_result(mesh, source_path, mesh_id)

    def draw_source(plotter):
        meshes = add_source_surface(plotter, result.mesh, color=color)
        return meshes, result.mesh.points, result.issues

    def draw_result(plotter):
        meshes, pts = add_result_with_skeleton(
            plotter,
            result.mesh,
            graph,
            surface_color=color,
            show_nodes=show_nodes,
            directed=directed,
        )
        return meshes, pts, []

    source_image, source_view_path, source_issues = render_scene(draw_source, PANEL_DIR / f"{stem}_source.png")
    result_image, result_view_path, result_issues = render_scene(draw_result, PANEL_DIR / f"{stem}_skeleton.png")
    add_panel(
        panels,
        summary,
        title=title,
        stem=stem,
        domain=domain,
        input_format=input_format,
        source_path=source_path,
        source_image=source_image,
        source_view_path=source_view_path,
        result_image=result_image,
        result_view_path=result_view_path,
        graph=graph,
        input_issues=source_issues,
        result_issues=result_issues,
    )


def make_volume_panel(
    panels: list[dict],
    summary: list[dict],
    *,
    title: str,
    stem: str,
    values: np.ndarray,
    spacing,
    origin,
    graph: nx.MultiGraph,
    source_path: Path,
    field_id: str,
    color: str,
    show_nodes: bool = True,
    directed: bool = False,
) -> None:
    result = load_volume_result(values, spacing, origin, source_path, field_id)
    source_image, source_view_path = scalar_slices_image(values, PANEL_DIR / f"{stem}_source.png")

    def draw_result(plotter):
        meshes, pts = add_result_with_skeleton(
            plotter,
            result.mesh,
            graph,
            surface_color=color,
            show_nodes=show_nodes,
            directed=directed,
        )
        return meshes, pts, []

    result_image, result_view_path, result_issues = render_scene(draw_result, PANEL_DIR / f"{stem}_skeleton.png")
    add_panel(
        panels,
        summary,
        title=title,
        stem=stem,
        domain="volumetric field",
        input_format=".npz",
        source_path=source_path,
        source_image=source_image,
        source_view_path=source_view_path,
        result_image=result_image,
        result_view_path=result_view_path,
        graph=graph,
        input_issues=result.issues,
        result_issues=result_issues,
    )


def make_vector_flow_panel(
    panels: list[dict],
    summary: list[dict],
    *,
    title: str,
    stem: str,
    vectors: np.ndarray,
    spacing,
    origin,
    graph: nx.MultiGraph,
    source_path: Path,
) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(source_path, vectors=vectors, spacing=np.asarray(spacing), origin=np.asarray(origin))
    source_image, source_view_path = vector_flow_slices_image(vectors, PANEL_DIR / f"{stem}_source.png")

    def draw_result(plotter):
        graph_meshes, graph_pts = add_skeleton_graph(plotter, graph, show_nodes=False, directed=True)
        return graph_meshes, graph_pts, []

    result_image, result_view_path, result_issues = render_scene(draw_result, PANEL_DIR / f"{stem}_oriented_graph.png")
    add_panel(
        panels,
        summary,
        title=title,
        stem=stem,
        domain="vector flow volume",
        input_format=".npz",
        source_path=source_path,
        source_image=source_image,
        source_view_path=source_view_path,
        result_image=result_image,
        result_view_path=result_view_path,
        graph=graph,
        input_issues=[],
        result_issues=result_issues,
    )


def main() -> None:
    ensure_dirs()
    panels: list[dict] = []
    summary: list[dict] = []
    data_dir = DATA_DIR / "appendix_surface_volume_fermi"

    make_surface_panel(
        panels,
        summary,
        title=label_with_format("Genus-2 Surface Mesh", "PLY"),
        stem="a_genus2_surface_ply",
        mesh=make_genus2_surface_mesh(),
        graph=genus2_skeleton_graph(),
        source_path=data_dir / "genus2_surface.ply",
        mesh_id="genus2_surface_s4",
        color=SURFACE_COLOR,
        domain="surface mesh",
        input_format=".ply",
    )
    make_surface_panel(
        panels,
        summary,
        title=label_with_format("Torus Surface Mesh", "PLY"),
        stem="b_torus_surface_ply",
        mesh=make_torus_mesh(),
        graph=torus_skeleton_graph(),
        source_path=data_dir / "torus_surface.ply",
        mesh_id="torus_surface_s4",
        color=SURFACE_COLOR,
        domain="surface mesh",
        input_format=".ply",
        show_nodes=False,
    )

    flow_vectors, flow_spacing, flow_origin = make_vortex_flow_field()
    make_vector_flow_panel(
        panels,
        summary,
        title=label_with_format("Vector Flow Volume", "NPZ"),
        stem="c_vector_flow_npz",
        vectors=flow_vectors,
        spacing=flow_spacing,
        origin=flow_origin,
        graph=vortex_flow_graph(),
        source_path=data_dir / "vortex_flow_vectors.npz",
    )

    gyroid_values, gyroid_spacing, gyroid_origin = make_gyroid_field()
    make_volume_panel(
        panels,
        summary,
        title=label_with_format("Gyroid Volume", "NPZ"),
        stem="d_gyroid_volume_npz",
        values=gyroid_values,
        spacing=gyroid_spacing,
        origin=gyroid_origin,
        graph=gyroid_skeleton_graph(),
        source_path=data_dir / "gyroid_volume.npz",
        field_id="gyroid_volume_s4",
        color=VOLUME_COLOR,
    )

    schwarz_values, schwarz_spacing, schwarz_origin = make_schwarz_p_field()
    make_volume_panel(
        panels,
        summary,
        title=label_with_format("Schwarz-P Volume", "NPZ"),
        stem="e_schwarz_p_volume_npz",
        values=schwarz_values,
        spacing=schwarz_spacing,
        origin=schwarz_origin,
        graph=schwarz_skeleton_graph(),
        source_path=data_dir / "schwarz_p_volume.npz",
        field_id="schwarz_p_volume_s4",
        color=VOLUME_COLOR,
    )

    make_surface_panel(
        panels,
        summary,
        title=label_with_format("Nodal-Line Fermi", "VTP"),
        stem="f_nodal_line_fermi_vtp",
        mesh=make_nodal_fermi_mesh(),
        graph=nodal_fermi_skeleton_graph(),
        source_path=data_dir / "nodal_line_fermi.vtp",
        mesh_id="nodal_line_fermi_s4",
        color=FERMI_COLOR,
        domain="Fermi surface",
        input_format=".vtp",
        show_nodes=False,
    )

    png_path, svg_path, pdf_path = assemble_figure(panels)
    summary.extend(
        [
            {"output": "png", "path": str(png_path), "success": png_path.exists() and png_path.stat().st_size > 0},
            {"output": "svg", "path": str(svg_path), "success": svg_path.exists() and svg_path.stat().st_size > 0},
            {"output": "pdf", "path": str(pdf_path), "success": pdf_path.exists() and pdf_path.stat().st_size > 0},
        ]
    )
    SUMMARY_PATH.write_text(
        json.dumps({"figure": "Appendix S4 surface, volume, flow, and Fermi inputs", "panels": summary}, indent=2) + "\n"
    )

    print("Appendix S4 surface, volume, flow, and Fermi input figure")
    print(f"Panel directory: {PANEL_DIR}")
    print(f"Final PNG: {png_path}")
    print(f"Final SVG: {svg_path}")
    print(f"Final PDF: {pdf_path}")
    print(f"Summary path: {SUMMARY_PATH}")
    for item in summary:
        if "title" in item:
            print(
                f"Panel: {item['title']} success={item['success']} "
                f"nodes={item['graph_node_count']} edges={item['graph_edge_count']} "
                f"input_issues={item['input_issues'] or 'none'} "
                f"result_issues={item['result_issues'] or 'none'}"
            )


if __name__ == "__main__":
    main()
