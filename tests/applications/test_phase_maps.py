import networkx as nx
import sympy as sp

from knotted_graph.applications.phase_maps import make_yamada_phase_map
from knotted_graph.inputs import KnotFunction


def _triangle_graph():
    graph = nx.MultiGraph()
    graph.add_node(0, pos=(0.0, 0.0, 0.0))
    graph.add_node(1, pos=(1.0, 0.0, 0.0))
    graph.add_node(2, pos=(0.0, 1.0, 0.0))
    graph.add_edge(0, 1, pts=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    graph.add_edge(1, 2, pts=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    graph.add_edge(2, 0, pts=[(0.0, 1.0, 0.0), (0.0, 0.0, 0.0)])
    return graph


def _patch_yamada(monkeypatch):
    import knotted_graph.invariants.yamada
    import knotted_graph.projection

    A = sp.Symbol("A")

    def fake_yamada(graph, variable, **kwargs):
        return A + graph.number_of_edges()

    monkeypatch.setattr(
        knotted_graph.invariants.yamada,
        "compute_graph_yamada_polynomial",
        fake_yamada,
    )
    monkeypatch.setattr(knotted_graph.projection, "compute_yamada_polynomial", fake_yamada)


def test_unified_phase_map_accepts_nodal_bloch_factories(monkeypatch):
    from knotted_graph.applications.nodal.skeleton import NodalSkeleton

    _patch_yamada(monkeypatch)
    monkeypatch.setattr(NodalSkeleton, "skeleton_graph", lambda self, **kwargs: _triangle_graph())

    kx, ky, kz = sp.symbols("kx ky kz", real=True)

    def start(gamma):
        return (kx, ky, kz + sp.I * gamma)

    def end(gamma):
        return (kx + gamma, ky, kz + sp.I * gamma)

    result = make_yamada_phase_map(
        start,
        end,
        source_kind="nodal",
        lambdas=[0.0, 1.0],
        parameters=[0.1, 0.2],
        dimension=6,
        force_genus_zero_vertex=False,
    )

    assert result.source_kind == "nodal"
    assert result.parameter_name == "gamma"
    assert result.record_grid().shape == (2, 2)
    assert all(record.error is None for record in result.records)
    assert {record.edges for record in result.records} == {3}


def test_unified_phase_map_accepts_material_gap_and_energy_modes(monkeypatch):
    from knotted_graph.applications.material_surface import MaterialFermiSurface

    _patch_yamada(monkeypatch)
    monkeypatch.setattr(MaterialFermiSurface, "skeleton_graph", lambda self, **kwargs: _triangle_graph())

    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    h0 = sp.diag(kx, ky, kz)
    h1 = sp.diag(kx + sp.Rational(1, 10), ky, kz)

    gap_result = make_yamada_phase_map(
        h0,
        h1,
        source_kind="material",
        material_mode="gap",
        band_pair=(0, 1),
        lambdas=[0.0, 1.0],
        parameters=[0.02, 0.04],
        k_symbols=(kx, ky, kz),
        dimension=5,
        force_genus_zero_vertex=False,
    )
    energy_result = make_yamada_phase_map(
        h0,
        h1,
        source_kind="material",
        material_mode="energy",
        band_index=1,
        lambdas=[0.0, 1.0],
        parameters=[-0.1, 0.1],
        k_symbols=(kx, ky, kz),
        dimension=5,
        force_genus_zero_vertex=False,
    )

    assert gap_result.parameter_name == "gap_tol"
    assert energy_result.parameter_name == "energy"
    assert all(record.source_kind == "material" for record in gap_result.records)
    assert all(record.source_kind == "material" for record in energy_result.records)
    assert all(record.error is None for record in energy_result.records)

    alias_result = make_yamada_phase_map(
        h0,
        h1,
        source_kind="hamiltonian",
        material_mode="energy",
        band_index=1,
        lambdas=[0.0],
        parameters=[0.0],
        k_symbols=(kx, ky, kz),
        dimension=5,
        force_genus_zero_vertex=False,
    )
    assert alias_result.source_kind == "material"


def test_unified_phase_map_accepts_knot_functions(monkeypatch):
    _patch_yamada(monkeypatch)

    sample_calls = []

    def fake_sample(self, **kwargs):
        sample_calls.append(self.name)
        return object()

    def fake_graph(self, radius, *, sample=None, **kwargs):
        assert sample is not None
        return _triangle_graph()

    monkeypatch.setattr(KnotFunction, "sample", fake_sample)
    monkeypatch.setattr(KnotFunction, "to_spatial_graph", fake_graph)

    start = KnotFunction.from_function(lambda x, y, z: x + 1j * y, name="xy")
    end = KnotFunction.from_function(lambda x, y, z: x + 1j * z, name="xz")
    result = make_yamada_phase_map(
        start,
        end,
        source_kind="knot",
        lambdas=[0.0, 0.5, 1.0],
        parameters=[0.1, 0.2],
        dimension=6,
    )

    assert result.source_kind == "knot"
    assert result.parameter_name == "radius"
    assert result.record_grid().shape == (2, 3)
    assert len(sample_calls) == 3
    assert all(record.error is None for record in result.records)


def test_unified_phase_map_collapses_closed_genus_zero_masks_to_vertex(monkeypatch):
    from knotted_graph.applications.nodal.skeleton import NodalSkeleton

    _patch_yamada(monkeypatch)

    mask = __import__("numpy").zeros((7, 7, 7), dtype=bool)
    mask[2:5, 2:5, 2:5] = True
    monkeypatch.setattr(NodalSkeleton, "_interior_mask", property(lambda self: mask))

    def fail_skeleton_graph(self, **kwargs):
        raise AssertionError("closed genus-zero masks should collapse before graph extraction")

    monkeypatch.setattr(NodalSkeleton, "skeleton_graph", fail_skeleton_graph)

    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    result = make_yamada_phase_map(
        (kx, ky, kz),
        (kx + 1, ky, kz),
        source_kind="nodal",
        lambdas=[0.0],
        parameters=[0.0],
        dimension=7,
    )

    [record] = result.records
    assert record.error is None
    assert record.nodes == 1
    assert record.edges == 0


def test_unified_phase_map_collapses_no_core_skeletons_to_vertex(monkeypatch):
    from knotted_graph.applications.nodal.skeleton import NodalSkeleton
    from knotted_graph.core import EmbeddingValidationError

    _patch_yamada(monkeypatch)

    mask = __import__("numpy").zeros((7, 7, 7), dtype=bool)
    mask[0, 0, 0] = True
    monkeypatch.setattr(NodalSkeleton, "_interior_mask", property(lambda self: mask))

    def empty_skeleton_graph(self, **kwargs):
        raise EmbeddingValidationError(["graph has no edges"])

    monkeypatch.setattr(NodalSkeleton, "skeleton_graph", empty_skeleton_graph)

    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    result = make_yamada_phase_map(
        (kx, ky, kz),
        (kx + 1, ky, kz),
        source_kind="nodal",
        lambdas=[0.0],
        parameters=[0.0],
        dimension=7,
    )

    [record] = result.records
    assert record.error is None
    assert record.nodes == 1
    assert record.edges == 0
