"""Unified Yamada phase-map scans for knot fields and Hamiltonians.

The existing applications layer already has specialized workflows for analytic
knot-field tubes, two-band nodal Hamiltonians, and Hermitian material band-gap
surfaces.  This module adds a small common front door: sample a
``(lambda, parameter)`` grid, extract one embedded graph per cell, and classify
the cells by their Yamada polynomial.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.core import EmbeddingValidationError, total_edge_pts

PhaseSourceKind = Literal[
    "auto",
    "nodal",
    "material",
    "knot",
    "bloch",
    "hamiltonian",
    "2band",
    "knot_field",
    "knot_function",
]
MaterialMode = Literal["gap", "energy"]
MaterialBandAlignment = Literal["strict", "pad"]


@dataclass(frozen=True)
class YamadaPhaseRecord:
    """One cell in a Yamada phase map."""

    lam: float
    parameter: float
    parameter_name: str
    source_kind: str
    nodes: int
    edges: int
    components: int
    cycle_rank: int
    degree_sequence: tuple[int, ...]
    total_edge_points: int
    yamada: sp.Expr | None
    phase_signature: str
    error: str | None = None


@dataclass
class YamadaPhaseMapResult:
    """Result returned by :func:`make_yamada_phase_map`."""

    lambdas: np.ndarray
    parameters: np.ndarray
    parameter_name: str
    source_kind: str
    records: list[YamadaPhaseRecord]
    metadata: dict[str, Any]
    _graph_factory: Callable[[float, float], nx.MultiGraph] | None = None
    _object_factory: Callable[[float, float], Any] | None = None

    def record_grid(self) -> np.ndarray:
        lookup = {
            (float(record.parameter), float(record.lam)): record
            for record in self.records
        }
        grid = np.empty((len(self.parameters), len(self.lambdas)), dtype=object)
        for row, parameter in enumerate(self.parameters):
            for column, lam in enumerate(self.lambdas):
                grid[row, column] = lookup[(float(parameter), float(lam))]
        return grid

    def phase_grid(self) -> tuple[np.ndarray, dict[int, str]]:
        signatures = sorted({record.phase_signature for record in self.records})
        ids = {signature: index for index, signature in enumerate(signatures)}
        grid = np.empty((len(self.parameters), len(self.lambdas)), dtype=int)
        for row, records in enumerate(self.record_grid()):
            for column, record in enumerate(records):
                grid[row, column] = ids[record.phase_signature]
        return grid, {index: signature for signature, index in ids.items()}

    def transition_intervals(self) -> list[dict[str, Any]]:
        grid = self.record_grid()
        changes: list[dict[str, Any]] = []
        for row, parameter in enumerate(self.parameters):
            for column in range(1, len(self.lambdas)):
                left = grid[row, column - 1]
                right = grid[row, column]
                if left.phase_signature != right.phase_signature:
                    changes.append(
                        {
                            self.parameter_name: float(parameter),
                            "lambda_left": float(self.lambdas[column - 1]),
                            "lambda_right": float(self.lambdas[column]),
                            "phase_left": left.phase_signature,
                            "phase_right": right.phase_signature,
                        }
                    )
        return changes

    def plot_phase_diagram(self, ax=None, *, title: str | None = None):
        """Plot the finite-grid phase map with nearest-neighbor cells."""
        import matplotlib.pyplot as plt

        labels, legend = self.phase_grid()
        if ax is None:
            _, ax = plt.subplots()
        image = ax.imshow(
            labels,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(
                float(self.lambdas[0]),
                float(self.lambdas[-1]),
                float(self.parameters[0]),
                float(self.parameters[-1]),
            ),
        )
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(self.parameter_name)
        ax.set_title(title or f"{self.source_kind} Yamada phase map")
        image._knotted_graph_phase_legend = legend
        return ax

    def graph_at(
        self,
        lam_or_record: float | YamadaPhaseRecord,
        parameter: float | None = None,
    ) -> nx.MultiGraph:
        """Recompute the embedded graph for one phase-map cell."""
        if self._graph_factory is None:
            raise ValueError("this result does not keep a graph factory")
        lam, value = _coerce_cell(lam_or_record, parameter)
        return self._graph_factory(lam, value)

    def object_at(
        self,
        lam_or_record: float | YamadaPhaseRecord,
        parameter: float | None = None,
    ) -> Any:
        """Recompute the source object for one phase-map cell.

        For nodal and material scans this is a skeleton/surface object.  For
        knot-field scans it is the interpolated ``KnotFunction``.
        """
        if self._object_factory is None:
            raise ValueError("this result does not keep an object factory")
        lam, value = _coerce_cell(lam_or_record, parameter)
        return self._object_factory(lam, value)


class MaterialBandEnergySurface:
    """Hermitian multiband level-set surface for one selected band.

    ``MaterialFermiSurface`` handles pairwise band-gap tubes.  This small
    subclass-like wrapper reuses that implementation for the other common
    material use case: a tube around ``E_band(k) == energy``.
    """

    def __new__(cls, *args, **kwargs):
        from knotted_graph.applications.material_surface import MaterialFermiSurface

        class _EnergySurface(MaterialFermiSurface):
            def __init__(
                self,
                char: sp.Matrix,
                *,
                energy: float,
                band_index: int,
                energy_tol: float,
                reference_band_pair: tuple[int, int] | None = None,
                **surface_options,
            ) -> None:
                hamiltonian = sp.Matrix(char)
                if hamiltonian.rows < 2:
                    raise ValueError(
                        "energy-mode material scans require at least two bands"
                    )
                if reference_band_pair is None:
                    reference_band_pair = (0, 1)
                super().__init__(
                    hamiltonian,
                    band_pair=reference_band_pair,
                    gap_tol=float(energy_tol),
                    **surface_options,
                )
                self.energy = float(energy)
                self.energy_tol = float(energy_tol)
                self.band_index = int(band_index)
                if not 0 <= self.band_index < self.n_bands:
                    raise ValueError(
                        f"band_index must be in [0,{self.n_bands - 1}]. "
                        f"Got {band_index}."
                    )

            @cached_property
            def spectrum(self):
                return self.eigvals_sorted[..., self.band_index] - self.energy

            @cached_property
            def band_gap(self):
                return np.abs(self.spectrum)

        return _EnergySurface(*args, **kwargs)


def pad_material_hamiltonian(
    hamiltonian,
    size: int,
    *,
    padding_energy: float | sp.Expr = 0.0,
) -> sp.Matrix:
    """Embed a smaller Hermitian material Hamiltonian in a larger band space.

    Extra bands are added as uncoupled, flat bands at ``padding_energy``.  This
    is useful for interpolation studies between effective models with different
    band counts, e.g. a two-band model deformed into a three-band model.
    """

    matrix = sp.Matrix(hamiltonian)
    if matrix.rows != matrix.cols:
        raise ValueError("material Hamiltonians must be square")
    size = int(size)
    if size < matrix.rows:
        raise ValueError(
            f"cannot pad a {matrix.rows}x{matrix.cols} Hamiltonian down to size {size}"
        )
    if size == matrix.rows:
        return matrix

    padded = sp.zeros(size, size)
    for row in range(matrix.rows):
        for col in range(matrix.cols):
            padded[row, col] = matrix[row, col]
    pad_value = sp.sympify(padding_energy)
    for index in range(matrix.rows, size):
        padded[index, index] = pad_value
    return padded


def align_material_hamiltonians(
    start,
    end,
    *,
    padding_energy: float | sp.Expr = 0.0,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Return two material Hamiltonians embedded in the same band dimension."""

    h0 = sp.Matrix(start)
    h1 = sp.Matrix(end)
    if h0.rows != h0.cols or h1.rows != h1.cols:
        raise ValueError("material Hamiltonians must be square")
    size = max(h0.rows, h1.rows)
    return (
        pad_material_hamiltonian(h0, size, padding_energy=padding_energy),
        pad_material_hamiltonian(h1, size, padding_energy=padding_energy),
    )


def make_yamada_phase_map(
    start,
    end=None,
    *,
    source_kind: PhaseSourceKind = "auto",
    lambdas: Sequence[float],
    parameters: Sequence[float],
    parameter_name: str | None = None,
    dimension: int | Sequence[int] = 96,
    span=None,
    k_symbols: Sequence[sp.Symbol] | None = None,
    axis_scale=None,
    band_pair: tuple[int, int] = (0, 1),
    band_index: int | None = None,
    material_mode: MaterialMode = "gap",
    material_band_alignment: MaterialBandAlignment = "strict",
    material_padding_energy: float | sp.Expr = 0.0,
    energy_tol: float = 1e-2,
    normalize_yamada: bool = True,
    yamada_variable: sp.Symbol | None = None,
    yamada_options: dict[str, Any] | None = None,
    graph_options: dict[str, Any] | None = None,
    surface_options: dict[str, Any] | None = None,
    knot_options: dict[str, Any] | None = None,
    graph_transform: Callable[[nx.MultiGraph], nx.MultiGraph] | None = None,
    force_genus_zero_vertex: bool = True,
    continue_on_error: bool = True,
    keep_factories: bool = True,
) -> YamadaPhaseMapResult:
    """Compute a Yamada phase map from two endpoints.

    Parameters
    ----------
    start, end
        Endpoints for the deformation.  Accepted forms are:

        - ``NodalBlochPath`` as ``start`` for two-band nodal scans.
        - two 3-component Bloch vectors, two 2x2 SymPy Hamiltonians, or two
          factories ``factory(parameter)`` returning either of those.
        - two Hermitian material Hamiltonians for ``source_kind="material"``.
          If their band counts differ, set ``material_band_alignment="pad"``
          to add uncoupled flat bands to the smaller model.
        - ``KnotFunctionPath`` as ``start`` or two ``KnotFunction`` objects for
          ``source_kind="knot"``.

    parameters
        The second axis of the phase map.  For nodal scans this is normally
        ``gamma``.  For knot scans this is the tube radius.  For material scans
        it is either ``gap_tol`` or energy, according to ``material_mode``.
    material_mode
        ``"gap"`` builds pairwise band-gap tubes with ``MaterialFermiSurface``.
        ``"energy"`` builds a tube around the selected band level
        ``E_band(k; lambda) == parameter``.
    force_genus_zero_vertex
        For nodal and material scans, collapse a closed genus-zero filled
        region to the one-vertex Yamada phase.  This is the handlebody rule used
        by the benchmark phase maps.
    """
    resolved_kind = _resolve_source_kind(
        source_kind,
        start,
        end,
        material_mode=material_mode,
        band_index=band_index,
    )
    lambdas_arr = _validate_axis(lambdas, "lambdas")
    parameters_arr = _validate_axis(parameters, "parameters")
    if np.any((lambdas_arr < 0.0) | (lambdas_arr > 1.0)):
        raise ValueError("all lambda samples must lie in [0, 1]")

    variable = yamada_variable or sp.Symbol("A")
    yamada_kwargs = {"normalize": normalize_yamada, **dict(yamada_options or {})}
    graph_kwargs = dict(graph_options or {})
    surface_kwargs = dict(surface_options or {})
    knot_kwargs = dict(knot_options or {})

    if resolved_kind == "nodal":
        default_parameter = "gamma"
        object_factory, graph_factory = _nodal_factories(
            start,
            end,
            dimension=dimension,
            span=span,
            k_symbols=k_symbols,
            axis_scale=axis_scale,
            graph_options=graph_kwargs,
            force_genus_zero_vertex=force_genus_zero_vertex,
        )
    elif resolved_kind == "material":
        default_parameter = "energy" if material_mode == "energy" else "gap_tol"
        object_factory, graph_factory = _material_factories(
            start,
            end,
            dimension=int(dimension),
            span=span,
            k_symbols=k_symbols,
            axis_scale=axis_scale,
            band_pair=band_pair,
            band_index=band_index,
            material_mode=material_mode,
            material_band_alignment=material_band_alignment,
            material_padding_energy=material_padding_energy,
            energy_tol=energy_tol,
            graph_options=graph_kwargs,
            surface_options=surface_kwargs,
            force_genus_zero_vertex=force_genus_zero_vertex,
        )
    else:
        default_parameter = "radius"
        object_factory, graph_factory = _knot_factories(
            start,
            end,
            dimension=dimension,
            span=span,
            graph_options=graph_kwargs,
            knot_options=knot_kwargs,
        )

    name = parameter_name or default_parameter
    records: list[YamadaPhaseRecord] = []
    for parameter in parameters_arr:
        for lam in lambdas_arr:
            graph = nx.MultiGraph()
            yamada = None
            error = None
            try:
                graph = graph_factory(float(lam), float(parameter))
                if graph_transform is not None:
                    graph = graph_transform(graph)
                yamada = _compute_yamada(graph, variable, yamada_kwargs)
                signature = _phase_signature(graph, yamada, None)
            except Exception as exc:
                if not continue_on_error:
                    raise
                error = f"{type(exc).__name__}: {exc}"
                signature = "error:" + error
            records.append(
                YamadaPhaseRecord(
                    lam=float(lam),
                    parameter=float(parameter),
                    parameter_name=name,
                    source_kind=resolved_kind,
                    yamada=yamada,
                    phase_signature=signature,
                    error=error,
                    **_graph_summary(graph),
                )
            )

    return YamadaPhaseMapResult(
        lambdas=lambdas_arr.copy(),
        parameters=parameters_arr.copy(),
        parameter_name=name,
        source_kind=resolved_kind,
        records=records,
        metadata={
            "source_kind": resolved_kind,
            "material_mode": material_mode if resolved_kind == "material" else None,
            "material_band_alignment": (
                material_band_alignment if resolved_kind == "material" else None
            ),
            "material_padding_energy": (
                material_padding_energy if resolved_kind == "material" else None
            ),
            "dimension": dimension,
            "span": span,
            "band_pair": band_pair if resolved_kind == "material" else None,
            "band_index": band_index if resolved_kind == "material" else None,
            "energy_tol": energy_tol if material_mode == "energy" else None,
            "force_genus_zero_vertex": force_genus_zero_vertex,
        },
        _graph_factory=graph_factory if keep_factories else None,
        _object_factory=object_factory if keep_factories else None,
    )


def _coerce_cell(
    lam_or_record: float | YamadaPhaseRecord,
    parameter: float | None,
) -> tuple[float, float]:
    if isinstance(lam_or_record, YamadaPhaseRecord):
        if parameter is not None:
            raise ValueError("pass either a record or (lam, parameter), not both")
        return float(lam_or_record.lam), float(lam_or_record.parameter)
    if parameter is None:
        raise ValueError("parameter is required when the first argument is a lambda")
    return float(lam_or_record), float(parameter)


def _validate_axis(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be non-empty and one-dimensional")
    return arr


def _resolve_source_kind(
    source_kind: PhaseSourceKind,
    start,
    end,
    *,
    material_mode: MaterialMode,
    band_index: int | None,
) -> Literal["nodal", "material", "knot"]:
    if source_kind != "auto":
        if material_mode not in ("gap", "energy"):
            raise ValueError("material_mode must be 'gap' or 'energy'")
        if source_kind == "hamiltonian":
            if (
                _is_matrix_like(start)
                and (
                    sp.Matrix(start).shape != (2, 2)
                    or material_mode == "energy"
                    or band_index is not None
                )
            ):
                return "material"
            return "nodal"
        aliases = {
            "bloch": "nodal",
            "2band": "nodal",
            "knot_field": "knot",
            "knot_function": "knot",
        }
        resolved = aliases.get(source_kind, source_kind)
        if resolved not in ("nodal", "material", "knot"):
            raise ValueError(
                "source_kind must be 'auto', 'nodal', 'material', 'knot', "
                "'bloch', 'hamiltonian', '2band', 'knot_field', or 'knot_function'"
            )
        return resolved  # type: ignore[return-value]

    from knotted_graph.applications.nodal.deformation import NodalBlochPath
    from knotted_graph.inputs import KnotFunction, KnotFunctionPath

    if isinstance(start, (KnotFunction, KnotFunctionPath)):
        return "knot"
    if isinstance(start, NodalBlochPath):
        return "nodal"
    if _is_matrix_like(start):
        matrix = sp.Matrix(start)
        if matrix.shape != (2, 2) or material_mode == "energy" or band_index is not None:
            return "material"
        return "nodal"
    if _is_bloch_vector_like(start):
        return "nodal"
    if callable(start):
        raise ValueError(
            "bare callables are ambiguous. Pass source_kind='knot', 'nodal', "
            "or 'material'."
        )
    if end is not None and _is_matrix_like(end):
        return "material"
    raise ValueError("could not infer source_kind")


def _nodal_factories(
    start,
    end,
    *,
    dimension: int | Sequence[int],
    span,
    k_symbols,
    axis_scale,
    graph_options,
    force_genus_zero_vertex,
):
    from knotted_graph.applications.nodal.deformation import NodalBlochPath
    from knotted_graph.applications.nodal.skeleton import NodalSkeleton

    path = start if isinstance(start, NodalBlochPath) else None
    if path is None and end is None:
        raise ValueError("nodal scans require a NodalBlochPath or two endpoints")

    def char_at(lam: float, parameter: float):
        if path is not None:
            return path.at(parameter, lam)
        left = _resolve_nodal_endpoint(start, parameter)
        right = _resolve_nodal_endpoint(end, parameter)
        return _linear_blend(left, right, lam)

    def object_factory(lam: float, parameter: float):
        kwargs = {"char": char_at(lam, parameter), "dimension": int(dimension)}
        if span is not None:
            kwargs["span"] = span
        if k_symbols is not None:
            kwargs["k_symbols"] = tuple(k_symbols)
        if axis_scale is not None:
            kwargs["axis_scale"] = axis_scale
        return NodalSkeleton(**kwargs)

    def graph_factory(lam: float, parameter: float):
        return _graph_from_skeleton_like(
            object_factory(lam, parameter),
            graph_options,
            force_genus_zero_vertex=force_genus_zero_vertex,
        )

    return object_factory, graph_factory


def _material_factories(
    start,
    end,
    *,
    dimension: int,
    span,
    k_symbols,
    axis_scale,
    band_pair,
    band_index,
    material_mode: MaterialMode,
    material_band_alignment: MaterialBandAlignment,
    material_padding_energy: float | sp.Expr,
    energy_tol,
    graph_options,
    surface_options,
    force_genus_zero_vertex,
):
    from knotted_graph.applications.material_surface import MaterialFermiSurface

    h0 = _resolve_hamiltonian_endpoint(start, k_symbols)
    h1 = _resolve_hamiltonian_endpoint(end, k_symbols) if end is not None else h0
    if material_band_alignment not in {"strict", "pad"}:
        raise ValueError("material_band_alignment must be 'strict' or 'pad'")
    if h0.shape != h1.shape:
        if material_band_alignment != "pad":
            raise ValueError(
                f"Hamiltonian shapes differ: {h0.shape} vs {h1.shape}. "
                "Use material_band_alignment='pad' to add inert flat bands."
            )
        h0, h1 = align_material_hamiltonians(
            h0,
            h1,
            padding_energy=material_padding_energy,
        )
    if h0.rows != h0.cols:
        raise ValueError("material scans require square Hamiltonians")

    def h_at(lam: float):
        return sp.Matrix(_linear_blend(h0, h1, lam))

    def common_options():
        options = {
            "dimension": dimension,
            "check_pt_symmetry": False,
            **surface_options,
        }
        if span is not None:
            options["span"] = span
        if k_symbols is not None:
            options["k_symbols"] = tuple(k_symbols)
        if axis_scale is not None:
            options["axis_scale"] = axis_scale
        return options

    def object_factory(lam: float, parameter: float):
        if material_mode == "gap":
            return MaterialFermiSurface(
                h_at(lam),
                band_pair=band_pair,
                gap_tol=float(parameter),
                **common_options(),
            )
        if band_index is None:
            raise ValueError("band_index is required for material_mode='energy'")
        return MaterialBandEnergySurface(
            h_at(lam),
            energy=float(parameter),
            band_index=int(band_index),
            energy_tol=float(energy_tol),
            reference_band_pair=band_pair,
            **common_options(),
        )

    def graph_factory(lam: float, parameter: float):
        return _graph_from_skeleton_like(
            object_factory(lam, parameter),
            graph_options,
            force_genus_zero_vertex=force_genus_zero_vertex,
        )

    return object_factory, graph_factory


def _knot_factories(
    start,
    end,
    *,
    dimension: int | Sequence[int],
    span,
    graph_options,
    knot_options,
):
    from knotted_graph.inputs import DEFAULT_SPAN, KnotFunction, KnotFunctionPath

    if isinstance(start, KnotFunctionPath):
        path = start
    else:
        if end is None:
            raise ValueError("knot scans require a KnotFunctionPath or two endpoints")
        left = start if isinstance(start, KnotFunction) else KnotFunction.from_function(start)
        right = end if isinstance(end, KnotFunction) else KnotFunction.from_function(end)
        path = KnotFunctionPath(left, right, **knot_options)

    resolved_span = DEFAULT_SPAN if span is None else span
    sample_cache: dict[float, Any] = {}

    def object_factory(lam: float, parameter: float):
        return path.at(lam)

    def graph_factory(lam: float, parameter: float):
        if lam not in sample_cache:
            sample_cache[lam] = object_factory(lam, parameter).sample(
                span=resolved_span,
                dimension=dimension,
            )
        return object_factory(lam, parameter).to_spatial_graph(
            parameter,
            sample=sample_cache[lam],
            **graph_options,
        )

    return object_factory, graph_factory


def _resolve_nodal_endpoint(value, parameter: float):
    if callable(value) and not _is_matrix_like(value):
        value = value(float(parameter))
    if _is_matrix_like(value):
        matrix = sp.Matrix(value)
        if matrix.shape != (2, 2):
            raise ValueError("nodal Hamiltonian endpoints must be 2x2 matrices")
        return matrix
    if _is_bloch_vector_like(value):
        if len(value) != 3:
            raise ValueError("Bloch-vector endpoints must have three components")
        return tuple(sp.sympify(component) for component in value)
    raise TypeError("nodal endpoint must be a 2x2 matrix or 3-component vector")


def _resolve_hamiltonian_endpoint(value, k_symbols):
    if value is None:
        raise ValueError("missing Hamiltonian endpoint")
    if callable(value) and not _is_matrix_like(value):
        if k_symbols is not None:
            try:
                value = value(k_symbols=tuple(k_symbols))
            except TypeError:
                value = value()
        else:
            value = value()
    if not _is_matrix_like(value):
        raise TypeError("Hamiltonian endpoint must be a SymPy matrix or factory")
    return sp.Matrix(value)


def _linear_blend(left, right, lam: float):
    lam = float(lam)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1]")
    if _is_matrix_like(left) or _is_matrix_like(right):
        left_matrix = sp.Matrix(left)
        right_matrix = sp.Matrix(right)
        if left_matrix.shape != right_matrix.shape:
            raise ValueError(
                f"endpoint shapes differ: {left_matrix.shape} vs {right_matrix.shape}"
            )
        return (1.0 - lam) * left_matrix + lam * right_matrix
    if len(left) != len(right):
        raise ValueError("endpoint vector lengths differ")
    return tuple(
        sp.expand((1.0 - lam) * sp.sympify(a) + lam * sp.sympify(b))
        for a, b in zip(left, right)
    )


def _is_matrix_like(value) -> bool:
    return isinstance(value, (sp.MatrixBase, sp.ImmutableMatrix))


def _is_bloch_vector_like(value) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
        and not _is_matrix_like(value)
    )


def _one_vertex_graph() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node(0, pos=(0.0, 0.0, 0.0))
    return graph


def _interior_boundary_faces(mask: np.ndarray) -> list[str]:
    mask = np.asarray(mask, dtype=bool)
    faces: list[str] = []
    for label, touched in (
        ("axis0_min", mask[0, :, :].any()),
        ("axis0_max", mask[-1, :, :].any()),
        ("axis1_min", mask[:, 0, :].any()),
        ("axis1_max", mask[:, -1, :].any()),
        ("axis2_min", mask[:, :, 0].any()),
        ("axis2_max", mask[:, :, -1].any()),
    ):
        if bool(touched):
            faces.append(label)
    return faces


def _closed_genus_zero_interior(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return True
    boundary_faces = _interior_boundary_faces(mask)
    if boundary_faces:
        return False
    from skimage.measure import euler_number, label

    _, component_count = label(mask, connectivity=3, return_num=True)
    euler = int(euler_number(mask, connectivity=3))
    handle_rank = max(0, int(component_count) - euler)
    return handle_rank == 0


def _graph_from_skeleton_like(
    obj,
    graph_options: dict[str, Any],
    *,
    force_genus_zero_vertex: bool,
) -> nx.MultiGraph:
    if force_genus_zero_vertex and hasattr(obj, "_interior_mask"):
        if _closed_genus_zero_interior(np.asarray(obj._interior_mask, dtype=bool)):
            return _one_vertex_graph()
    try:
        graph = obj.skeleton_graph(**graph_options)
    except (EmbeddingValidationError, ValueError) as exc:
        message = str(exc)
        if "collapsed to fewer than two distinct points" in message:
            retry_options = dict(graph_options)
            retry_options["smooth_epsilon"] = 0
            try:
                graph = obj.skeleton_graph(**retry_options)
            except (EmbeddingValidationError, ValueError):
                return _one_vertex_graph()
            if graph.number_of_edges() == 0:
                return _one_vertex_graph()
            return graph
        if (
            "graph has no edges" in message
            or "skeleton image is empty" in message
            or "does not contain any True voxels" in message
            or "Skeletonization produced no points" in message
        ):
            return _one_vertex_graph()
        raise
    if graph.number_of_edges() == 0:
        return _one_vertex_graph()
    return graph


def _graph_summary(graph: nx.MultiGraph) -> dict[str, Any]:
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    components = nx.number_connected_components(graph) if nodes else 0
    cycle_rank = edges - nodes + components
    degrees = tuple(sorted((degree for _, degree in graph.degree()), reverse=True))
    try:
        edge_points = int(total_edge_pts(graph))
    except Exception:
        edge_points = 0
    return {
        "nodes": nodes,
        "edges": edges,
        "components": components,
        "cycle_rank": cycle_rank,
        "degree_sequence": degrees,
        "total_edge_points": edge_points,
    }


def _compute_yamada(
    graph: nx.MultiGraph,
    variable: sp.Symbol,
    yamada_options: dict[str, Any],
) -> sp.Expr:
    from knotted_graph.invariants.yamada import compute_graph_yamada_polynomial

    if graph.number_of_nodes() == 0:
        raise ValueError("cannot compute Yamada polynomial for an empty graph")
    if graph.number_of_edges() == 0:
        return compute_graph_yamada_polynomial(graph, variable)

    from knotted_graph.projection import compute_yamada_polynomial

    try:
        return compute_yamada_polynomial(graph, variable, **yamada_options)
    except Exception:
        return compute_graph_yamada_polynomial(graph, variable)


def _phase_signature(
    graph: nx.MultiGraph,
    yamada: sp.Expr | None,
    error: str | None,
) -> str:
    if yamada is not None:
        try:
            canonical = sp.factor(sp.together(sp.expand(yamada)))
        except Exception:
            canonical = sp.expand(yamada)
        return "yamada:" + sp.srepr(canonical)
    if error is not None:
        return "error:" + error
    return "graph:" + repr(tuple(_graph_summary(graph).values()))


__all__ = [
    "align_material_hamiltonians",
    "MaterialBandEnergySurface",
    "pad_material_hamiltonian",
    "YamadaPhaseMapResult",
    "YamadaPhaseRecord",
    "make_yamada_phase_map",
]
