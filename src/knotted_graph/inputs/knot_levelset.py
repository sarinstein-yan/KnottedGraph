"""Sampling, level-set, and tubular-neighborhood helpers for analytic knot fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .knot_field import KnotFunction


@dataclass(frozen=True)
class FieldSample:
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    values: np.ndarray

    def __post_init__(self) -> None:
        expected = tuple(len(axis) for axis in self.axes)
        if self.values.ndim != 3 or self.values.shape != expected:
            raise ValueError("field values must be 3-D and match the sampled axes")
        if any(len(axis) < 2 for axis in self.axes):
            raise ValueError("each sampled axis needs at least two points")

    @property
    def origin(self) -> np.ndarray:
        return np.asarray([axis[0] for axis in self.axes], dtype=float)

    @property
    def spacing(self) -> np.ndarray:
        return np.asarray([axis[1] - axis[0] for axis in self.axes], dtype=float)

    @property
    def abs_values(self) -> np.ndarray:
        return np.abs(self.values)

    @property
    def span(self):
        return tuple((float(axis[0]), float(axis[-1])) for axis in self.axes)


@dataclass(frozen=True)
class LevelSetMesh:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    values: np.ndarray
    radius: float

    def to_pyvista(self):
        try:
            import pyvista as pv
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "LevelSetMesh.to_pyvista requires pyvista; install the surface extra."
            ) from exc
        faces = np.column_stack(
            [np.full(len(self.faces), 3, dtype=np.int64), self.faces.astype(np.int64)]
        ).ravel()
        return pv.PolyData(self.vertices, faces)


@dataclass(frozen=True)
class LevelSetDiagnostics:
    radius: float
    occupied_voxels: int
    volume_components: int
    surface_components: int
    surface_euler_characteristic: int
    surface_is_closed: bool
    touches_box_boundary: bool
    total_boundary_genus: int | None
    expected_components: int | None
    matches_expected_tubular_neighborhood: bool | None


@dataclass(frozen=True)
class TubularConvergenceReport:
    radius: float
    dimensions: tuple[int, ...]
    diagnostics: tuple[LevelSetDiagnostics, ...]
    converged: bool


def _dimensions(dimension: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(dimension, int):
        values = (dimension,) * 3
    else:
        values = tuple(int(value) for value in dimension)
    if len(values) != 3 or any(value < 2 for value in values):
        raise ValueError("dimension must be an int or three integers >= 2")
    return values


def touches_box_boundary(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0]) or np.any(mask[-1])
        or np.any(mask[:, 0]) or np.any(mask[:, -1])
        or np.any(mask[:, :, 0]) or np.any(mask[:, :, -1])
    )


def sample_field(field: "KnotFunction", *, span, dimension=96) -> FieldSample:
    dims = _dimensions(dimension)
    if len(span) != 3 or any(len(b) != 2 or b[0] >= b[1] for b in span):
        raise ValueError("span must contain three increasing (min, max) bounds")
    axes = tuple(
        np.linspace(float(bounds[0]), float(bounds[1]), size)
        for bounds, size in zip(span, dims)
    )
    values = field(
        axes[0][:, None, None],
        axes[1][None, :, None],
        axes[2][None, None, :],
    )
    return FieldSample(axes=axes, values=np.asarray(values, dtype=np.complex128))


def sublevel_mask(
    field: "KnotFunction",
    radius: float,
    *,
    sample: FieldSample | None = None,
    span,
    dimension=96,
    require_compact: bool = True,
) -> tuple[np.ndarray, FieldSample]:
    if radius <= 0:
        raise ValueError("radius must be positive")
    pole = field.projection_pole_value
    if pole is not None and radius >= abs(pole):
        raise ValueError(
            "radius includes the stereographic projection pole, so the S3 sublevel "
            "is non-compact in this R3 chart; choose a smaller radius or another chart"
        )
    sample = sample or sample_field(field, span=span, dimension=dimension)
    mask = sample.abs_values <= float(radius)
    if not np.any(mask):
        raise ValueError("sublevel set is empty on the sampled grid")
    if require_compact and touches_box_boundary(mask):
        raise ValueError(
            "sublevel set touches the sampling-box boundary; enlarge span before "
            "treating it as a compact handlebody"
        )
    return mask, sample


def level_surface(
    field: "KnotFunction",
    radius: float,
    *,
    sample: FieldSample | None = None,
    span,
    dimension=96,
    require_compact: bool = True,
) -> LevelSetMesh:
    sample = sample or sample_field(field, span=span, dimension=dimension)
    sublevel_mask(
        field, radius, sample=sample, span=span, dimension=dimension,
        require_compact=require_compact,
    )
    try:
        from skimage import measure
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "level_surface requires scikit-image; install the knot-fields extra."
        ) from exc
    vertices, faces, normals, values = measure.marching_cubes(
        sample.abs_values,
        level=float(radius),
        spacing=tuple(float(value) for value in sample.spacing),
    )
    return LevelSetMesh(
        vertices=np.asarray(vertices + sample.origin, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        normals=np.asarray(normals, dtype=float),
        values=np.asarray(values, dtype=float),
        radius=float(radius),
    )


def to_spatial_graph(
    field: "KnotFunction",
    radius: float,
    *,
    sample: FieldSample | None = None,
    span,
    dimension=96,
    skeleton_padding: int = 1,
    max_junction_degree: int | None = None,
    adaptive_max_hops: int = 4,
    anomaly_ratio: float = 0.15,
) -> nx.MultiGraph:
    from knotted_graph.extraction import skeleton_image_to_graph, skeletonize_volume

    mask, sample = sublevel_mask(
        field, radius, sample=sample, span=span, dimension=dimension
    )
    skeleton = skeletonize_volume(mask, padding=skeleton_padding)
    graph = skeleton_image_to_graph(
        skeleton,
        max_junction_degree=max_junction_degree,
        adaptive_max_hops=adaptive_max_hops,
        anomaly_ratio=anomaly_ratio,
    )
    origin, spacing = sample.origin, sample.spacing
    for _, data in graph.nodes(data=True):
        data["pos"] = origin + spacing * np.asarray(data["pos"], dtype=float)
    for _, _, _, data in graph.edges(keys=True, data=True):
        points = origin + spacing * np.asarray(data["pts"], dtype=float)
        data["pts"] = points
        data["weight"] = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    graph.graph.update(
        source="KnotFunction",
        knot_function_name=field.name,
        sublevel_radius=float(radius),
        sample_span=sample.span,
        sample_shape=sample.values.shape,
    )
    return graph


def _surface_topology(mesh: LevelSetMesh) -> tuple[int, int, bool, int | None]:
    faces, vertices = mesh.faces, mesh.vertices
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edge_counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        union(int(a), int(b)); union(int(b), int(c)); union(int(c), int(a))
        for left, right in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(left), int(right))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    used = np.unique(faces)
    components = len({find(int(vertex)) for vertex in used})
    euler = int(len(used) - len(edge_counts) + len(faces))
    closed = bool(edge_counts) and all(count == 2 for count in edge_counts.values())
    genus = None
    if closed:
        candidate = components - euler / 2
        rounded = int(round(candidate))
        if abs(candidate - rounded) < 1e-8 and rounded >= 0:
            genus = rounded
    return components, euler, closed, genus


def diagnose_level(field: "KnotFunction", radius: float, *, sample=None, span, dimension=96):
    try:
        from skimage import measure
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "diagnose_level requires scikit-image; install the knot-fields extra."
        ) from exc
    mask, sample = sublevel_mask(
        field, radius, sample=sample, span=span, dimension=dimension,
        require_compact=False,
    )
    volume_components = int(measure.label(mask, connectivity=3).max())
    mesh = level_surface(
        field, radius, sample=sample, span=span, dimension=dimension,
        require_compact=False,
    )
    surface_components, euler, closed, genus = _surface_topology(mesh)
    boundary = touches_box_boundary(mask)
    expected = field.expected_components
    matches = None if expected is None else bool(
        not boundary and closed
        and volume_components == expected
        and surface_components == expected
        and genus == expected
    )
    return LevelSetDiagnostics(
        radius=float(radius),
        occupied_voxels=int(np.count_nonzero(mask)),
        volume_components=volume_components,
        surface_components=surface_components,
        surface_euler_characteristic=euler,
        surface_is_closed=closed,
        touches_box_boundary=boundary,
        total_boundary_genus=genus,
        expected_components=expected,
        matches_expected_tubular_neighborhood=matches,
    )


def tubular_convergence(
    field: "KnotFunction", radius: float, *, dimensions=(64, 96, 128), span
) -> TubularConvergenceReport:
    dimensions = tuple(int(value) for value in dimensions)
    if len(dimensions) < 2 or any(value < 2 for value in dimensions):
        raise ValueError("dimensions must contain at least two resolutions >= 2")
    diagnostics = tuple(
        diagnose_level(field, radius, span=span, dimension=dimension)
        for dimension in dimensions
    )
    def signature(item):
        return (
            item.volume_components, item.surface_components,
            item.surface_euler_characteristic, item.surface_is_closed,
            item.touches_box_boundary, item.total_boundary_genus,
            item.matches_expected_tubular_neighborhood,
        )
    converged = bool(
        signature(diagnostics[-1]) == signature(diagnostics[-2])
        and diagnostics[-1].matches_expected_tubular_neighborhood is not False
        and not diagnostics[-1].touches_box_boundary
    )
    return TubularConvergenceReport(float(radius), dimensions, diagnostics, converged)


__all__ = [
    "FieldSample", "LevelSetDiagnostics", "LevelSetMesh",
    "TubularConvergenceReport", "diagnose_level", "level_surface",
    "sample_field", "sublevel_mask", "to_spatial_graph", "tubular_convergence",
]
