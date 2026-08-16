import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    compute_yamada_from_states,
)
from knotted_graph.invariants.yamada.recursive import (
    YamadaRecursiveEvaluator,
)
from knotted_graph.projection import PDCode, generate_isotopy_angles


A = sp.Symbol("A")
X, Y = sp.symbols("x y")



def _assert_equal(left, right):
    assert sp.simplify(sp.together(sp.expand(left - right))) == 0


def _theta_abstract():
    graph = nx.MultiGraph()
    graph.add_nodes_from([0, 1])
    for _ in range(3):
        graph.add_edge(0, 1)
    return graph


def _theta_surface_embedding(amplitude=5.0, samples=41):
    graph = nx.MultiGraph()
    t = np.linspace(0.0, 1.0, samples)

    def zfun(x, y):
        return amplitude * (
            0.45 * np.sin(0.8 * x)
            + 0.30 * np.cos(1.1 * y)
            + 0.03 * x * y
        )

    endpoints = {
        "u": (-3.0, 0.0),
        "v": (3.0, 0.0),
    }
    for node, (x, y) in endpoints.items():
        graph.add_node(node, pos=np.array([x, y, zfun(x, y)]))

    for sign in (1.0, 0.0, -1.0):
        x = -3.0 + 6.0 * t
        y = sign * 1.8 * np.sin(np.pi * t)
        pts = np.column_stack(
            [x, y, [zfun(xi, yi) for xi, yi in zip(x, y)]]
        )
        graph.add_edge("u", "v", pts=pts)

    return graph


def _surface_lifted_planar_embedding(
    graph,
    *,
    amplitude=5.0,
    samples=11,
    scale=3.0,
):
    """Embed a plane graph on a corrugated surface z=f(x,y).

    The embedding is ambient-isotopic to the original plane embedding, while
    generic projections can contain crossings and therefore exercise PDCode.
    """
    pos2 = nx.planar_layout(graph)
    embedded = nx.MultiGraph()

    def zfun(x, y):
        return amplitude * (
            0.45 * np.sin(0.8 * x)
            + 0.30 * np.cos(1.1 * y)
            + 0.12 * x * y / scale**2
        )

    for node, xy in pos2.items():
        x, y = scale * np.asarray(xy, dtype=float)
        embedded.add_node(
            node,
            pos=np.array([x, y, zfun(x, y)]),
        )

    for u, v in graph.edges():
        p0 = scale * np.asarray(pos2[u], dtype=float)
        p1 = scale * np.asarray(pos2[v], dtype=float)
        t = np.linspace(0.0, 1.0, samples)
        xy = (1.0 - t[:, None]) * p0 + t[:, None] * p1
        pts = np.column_stack(
            [
                xy[:, 0],
                xy[:, 1],
                [zfun(x, y) for x, y in xy],
            ]
        )
        embedded.add_edge(u, v, pts=pts)

    return embedded


def _spring_3d_embedding(graph, *, seed, scale=3.0):
    """Deterministic generic straight-line 3D embedding."""
    positions = nx.spring_layout(
        graph,
        dim=3,
        seed=seed,
        scale=scale,
    )

    embedded = nx.MultiGraph()
    for node, point in positions.items():
        embedded.add_node(
            node,
            pos=np.asarray(point, dtype=float),
        )

    for u, v in graph.edges():
        embedded.add_edge(
            u,
            v,
            pts=np.vstack([positions[u], positions[v]]),
        )

    return embedded


def _screen(graph, *, views=12):
    results = []
    for index, angles in enumerate(generate_isotopy_angles(views)):
        processor = PDCode(graph)
        try:
            pd_code = processor.compute(
                rotation_angles=tuple(float(x) for x in angles),
                rotation_order="ZYX",
            )
        except Exception:
            continue
        results.append(
            {
                "index": index,
                "angles": tuple(float(x) for x in angles),
                "crossings": len(processor.crossings),
                "pd_code": pd_code,
                "processor": processor,
            }
        )
    return results


def _select(screened, *, count=4, max_crossings=2):
    candidates = [
        item for item in screened
        if item["crossings"] <= max_crossings
    ]
    buckets = {
        crossing_count: [
            item for item in candidates
            if item["crossings"] == crossing_count
        ]
        for crossing_count in range(max_crossings + 1)
    }

    # Prefer crossing-containing projections, but keep crossing-free controls.
    order = list(range(1, max_crossings + 1)) + [0]
    chosen = []
    while len(chosen) < count:
        changed = False
        for crossing_count in order:
            if buckets[crossing_count]:
                chosen.append(buckets[crossing_count].pop(0))
                changed = True
                if len(chosen) == count:
                    break
        if not changed:
            break
    return chosen


def _evaluate_pd(processor, evaluator):
    calculator = Yamada(
        vertices=list(processor.vertices.values()),
        crossings=list(processor.crossings.values()),
        arcs=list(processor.arcs.values()),
    )
    state_graphs, exponents = calculator._build_state_graphs()
    values = [evaluator.compute(graph) for graph in state_graphs]
    raw = sp.expand(
        sum(
            A**exponent * value
            for exponent, value in zip(exponents, values)
        )
    )
    normalized = processor.compute_yamada(
        A,
        normalize=True,
        n_jobs=1,
        method="recursive",
    )
    return raw, normalized


def _projection_invariance_check(
    abstract_graph,
    embedded_graph,
    *,
    expected_normalized=None,
    views=16,
    evaluate=4,
):
    screened = _screen(embedded_graph, views=views)
    selected = _select(
        screened,
        count=evaluate,
        max_crossings=2,
    )
    assert len(selected) >= 2

    evaluator = YamadaRecursiveEvaluator(A)
    normalized_values = []

    for item in selected:
        _, normalized = _evaluate_pd(item["processor"], evaluator)
        normalized_values.append(normalized)

    reference = normalized_values[0]
    for value in normalized_values[1:]:
        _assert_equal(value, reference)

    if expected_normalized is not None:
        _assert_equal(reference, expected_normalized)

    # One independent backend spot-check on a low-crossing PD.
    first = selected[0]["processor"]
    calculator = Yamada(
        vertices=list(first.vertices.values()),
        crossings=list(first.crossings.values()),
        arcs=list(first.arcs.values()),
    )
    states, exponents = calculator._build_state_graphs()

    negami_normalized = first.compute_yamada(
        A,
        normalize=True,
        n_jobs=1,
        method="negami",
    )
    _assert_equal(negami_normalized, reference)

    return screened, selected, reference


def test_named_planar_trivalent_graphs_across_multiple_projections():
    cases = [
        (
            _theta_abstract(),
            _theta_surface_embedding(),
        ),
        (
            nx.complete_graph(4),
            _surface_lifted_planar_embedding(nx.complete_graph(4)),
        ),
        (
            nx.circular_ladder_graph(3),
            _surface_lifted_planar_embedding(nx.circular_ladder_graph(3)),
        ),
        (
            nx.cubical_graph(),
            _surface_lifted_planar_embedding(nx.cubical_graph()),
        ),
    ]

    for abstract_graph, embedded_graph in cases:
        expected = compute_yamada_from_states(
            [nx.MultiGraph(abstract_graph)],
            [0],
            A,
            normalize=True,
            n_jobs=1,
            method="negami",
        )
        _projection_invariance_check(
            abstract_graph,
            embedded_graph,
            expected_normalized=expected,
        )


def test_nonplanar_and_unpublished_trivalent_embeddings_across_projections():
    graph_cases = [
        (
            nx.complete_bipartite_graph(3, 3),
            1,
        ),
        (
            nx.random_regular_graph(3, 8, seed=11),
            11,
        ),
    ]

    for abstract_graph, embedding_seed in graph_cases:
        assert all(degree == 3 for _, degree in abstract_graph.degree())
        embedded = _spring_3d_embedding(
            abstract_graph,
            seed=embedding_seed,
        )
        _projection_invariance_check(
            abstract_graph,
            embedded,
            expected_normalized=None,
        )
