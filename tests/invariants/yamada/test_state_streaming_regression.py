import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada import (
    Yamada,
    compute_yamada_from_states,
)
from knotted_graph.projection import PDCode


def _assert_equal(left, right):
    assert sp.simplify(
        sp.together(
            sp.expand(left - right)
        )
    ) == 0


def _multi_crossing_theta(component_count=3):
    graph = nx.MultiGraph()

    for component in range(component_count):
        y_offset = 5.0 * component
        sign = 1.0 if component % 2 == 0 else -1.0

        left = f"u{component}"
        right = f"v{component}"

        graph.add_node(left, pos=np.array([-2.0, y_offset, 0.0]))
        graph.add_node(right, pos=np.array([2.0, y_offset, 0.0]))

        curves = [
            np.array(
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, -1.0, 0.5 * sign],
                    [1.0, 1.0, 0.5 * sign],
                    [2.0, 0.0, 0.0],
                ]
            ),
            np.array(
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 1.0, -0.5 * sign],
                    [1.0, -1.0, -0.5 * sign],
                    [2.0, 0.0, 0.0],
                ]
            ),
            np.array(
                [
                    [-2.0, 0.0, 0.0],
                    [-1.0, 2.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
        ]

        for points in curves:
            shifted = points.copy()
            shifted[:, 1] += y_offset
            graph.add_edge(left, right, pts=shifted)

    return graph


def test_streaming_compute_matches_materialized_reference():
    A = sp.Symbol("A")

    processor = PDCode(_multi_crossing_theta(3))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))

    assert len(processor.crossings) == 3

    calculator = Yamada.from_PDCode(processor)
    state_graphs, exponents = calculator._build_state_graphs()

    assert len(state_graphs) == 3**3
    assert len(exponents) == 3**3

    for method in ("recursive", "negami"):
        for normalize in (False, True):
            materialized = compute_yamada_from_states(
                state_graphs,
                exponents,
                A,
                normalize=normalize,
                n_jobs=1,
                method=method,
            )
            streamed = calculator.compute(
                A,
                normalize=normalize,
                n_jobs=1,
                method=method,
            )
            _assert_equal(streamed, materialized)


def test_normal_compute_no_longer_calls_materializing_helper(monkeypatch):
    A = sp.Symbol("A")

    processor = PDCode(_multi_crossing_theta(2))
    processor.compute(rotation_angles=(0.0, 0.0, 0.0))
    calculator = Yamada.from_PDCode(processor)

    def fail_if_called(self):
        raise AssertionError(
            "_build_state_graphs should not be used by Yamada.compute"
        )

    monkeypatch.setattr(Yamada, "_build_state_graphs", fail_if_called)

    calculator.compute(
        A,
        normalize=True,
        n_jobs=1,
        method="recursive",
    )
