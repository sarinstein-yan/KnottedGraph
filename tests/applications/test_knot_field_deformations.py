import networkx as nx
import sympy as sp

from knotted_graph.applications.knot_deformation import KnotDeformationScan
from knotted_graph.applications.nodal.deformation import NodalBlochPath
from knotted_graph.inputs import KnotFunction, KnotFunctionPath


def test_nodal_bloch_path_matches_old_notebook_linear_blend():
    x = sp.Symbol("x")

    def start(gamma):
        return (x + gamma, 2 * x, 3 * x - gamma)

    def end(gamma):
        return (2 * x - gamma, 4 * x, x + gamma)

    path = NodalBlochPath(start, end)
    gamma, lam = 0.25, 0.4
    left, right = path.endpoints(gamma)
    expected = tuple(sp.expand((1 - lam) * a + lam * b) for a, b in zip(left, right))
    assert all(sp.simplify(a - b) == 0 for a, b in zip(path.at(gamma, lam), expected))


def test_componentwise_bloch_weights_are_supported():
    def start(gamma):
        return (gamma, gamma + 1, gamma + 2)

    def end(gamma):
        return (10 + gamma, 20 + gamma, 30 + gamma)

    path = NodalBlochPath(start, end)
    assert path.at_components(0.5, (0, 0.5, 1)) == (
        sp.Float(0.5), sp.Float(11.0), sp.Float(30.5)
    )


def test_knot_deformation_scan_reuses_one_field_sample_per_lambda(monkeypatch):
    start = KnotFunction.from_function(lambda x, y, z: x + 1j * y)
    end = KnotFunction.from_function(lambda x, y, z: x + 1j * z)
    path = KnotFunctionPath(start, end, sample_count=64)
    sample_calls = []
    original_sample = KnotFunction.sample

    def counted_sample(self, *args, **kwargs):
        sample_calls.append(self.name)
        return original_sample(self, *args, **kwargs)

    def fake_graph(self, radius, *, sample=None, **kwargs):
        assert sample is not None
        graph = nx.MultiGraph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        return graph

    monkeypatch.setattr(KnotFunction, "sample", counted_sample)
    monkeypatch.setattr(KnotFunction, "to_spatial_graph", fake_graph)
    result = KnotDeformationScan(
        path,
        lambdas=[0, 0.5, 1],
        radii=[0.1, 0.2],
        span=((-1, 1),) * 3,
        dimension=8,
    ).run()
    assert len(sample_calls) == 3
    assert len(result.records) == 6
    assert result.record_grid().shape == (2, 3)
    assert all(record.cycle_rank == 1 for record in result.records)
