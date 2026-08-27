import inspect
import warnings

import networkx as nx
import numpy as np
import pytest
import sympy as sp

from knotted_graph.invariants.yamada.polynomial import (
    Yamada,
    compute_yamada_from_states,
)
from knotted_graph.projection import pd_code
from knotted_graph.projection.pd_code import (
    PDCode,
    ProjectionResult,
    compute_yamada_polynomial,
    sample_projections,
    select_projection,
)
from knotted_graph.projection.rotations import (
    generate_isotopy_angles,
    get_rotation_matrix,
)


def _embedded_edge() -> nx.MultiGraph:
    graph = nx.MultiGraph()
    graph.add_node("u", pos=np.array([0.0, 0.0, 0.0]))
    graph.add_node("v", pos=np.array([1.0, 0.0, 0.0]))
    graph.add_edge(
        "u",
        "v",
        pts=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    return graph


@pytest.mark.parametrize("sample_count", [True, 2.5])
def test_projection_sampling_rejects_non_integer_sample_counts(sample_count):
    with pytest.raises(TypeError, match="num_rotation_samples must be a positive integer"):
        sample_projections(_embedded_edge(), num_rotation_samples=sample_count)


@pytest.mark.parametrize("sample_count", [0, -1])
def test_projection_sampling_rejects_non_positive_sample_counts(sample_count):
    with pytest.raises(ValueError, match="num_rotation_samples must be a positive integer"):
        sample_projections(_embedded_edge(), num_rotation_samples=sample_count)


@pytest.mark.parametrize("sample_count", [True, 1.5])
def test_rotation_sampling_rejects_non_integer_counts(sample_count):
    with pytest.raises(TypeError, match="N must be a positive integer"):
        generate_isotopy_angles(sample_count)


@pytest.mark.parametrize(
    ("order", "exception", "message"),
    [
        (None, TypeError, "rotation_order must be a string"),
        ("XY", ValueError, "three-character Euler-axis sequence"),
        ("XYQ", ValueError, "containing only x, y, and z"),
        ("Zyx", ValueError, "one case consistently"),
    ],
)
def test_rotation_order_errors_explain_the_accepted_convention(
    order,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        get_rotation_matrix((0.0, 0.0, 0.0), order=order)


def test_pd_code_validates_rotation_order_even_without_rotation_angles():
    processor = PDCode(_embedded_edge())

    with pytest.raises(ValueError, match="three-character Euler-axis sequence"):
        processor.compute(rotation_order="bad")


def test_explicit_projection_does_not_validate_unused_sample_count(monkeypatch):
    expected = ProjectionResult(
        processor=object(),
        rotation_angles=(0.0, 0.0, 0.0),
        rotation_order="ZYX",
        pd_code="explicit",
        num_crossings=0,
    )
    monkeypatch.setattr(pd_code, "_compute_projection", lambda *args: expected)

    result = select_projection(
        _embedded_edge(),
        rotation_angles=(0.0, 0.0, 0.0),
        num_rotation_samples=0,
    )

    assert result is expected


def test_partial_projection_failures_emit_one_runtime_warning(monkeypatch):
    monkeypatch.setattr(
        pd_code,
        "generate_isotopy_angles",
        lambda count, order: np.asarray(
            [[float(index), 0.0, 0.0] for index in range(count)]
        ),
    )

    def fake_projection(graph, rotation_angles, rotation_order):
        if int(rotation_angles[0]) % 2:
            raise ValueError(f"invalid view {rotation_angles[0]:.0f}")
        return ProjectionResult(
            processor=object(),
            rotation_angles=rotation_angles,
            rotation_order=rotation_order,
            pd_code="valid",
            num_crossings=int(rotation_angles[0]),
        )

    monkeypatch.setattr(pd_code, "_compute_projection", fake_projection)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projections = sample_projections(_embedded_edge(), num_rotation_samples=4)

    runtime_warnings = [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    assert len(runtime_warnings) == 1
    assert "2 of 4 projection samples failed" in str(runtime_warnings[0].message)
    assert [result.rotation_angles[0] for result in projections] == [0.0, 2.0]


def test_all_failed_projection_samples_raise_with_diagnostics(monkeypatch):
    monkeypatch.setattr(
        pd_code,
        "generate_isotopy_angles",
        lambda count, order: np.asarray(
            [[float(index), 0.0, 0.0] for index in range(count)]
        ),
    )

    def fail_projection(graph, rotation_angles, rotation_order):
        raise ValueError(f"invalid view {rotation_angles[0]:.0f}")

    monkeypatch.setattr(pd_code, "_compute_projection", fail_projection)

    with pytest.raises(RuntimeError, match=r"All projection samples failed:.*sample 0"):
        sample_projections(_embedded_edge(), num_rotation_samples=2)


def test_public_yamada_parallel_defaults_are_single_process():
    public_callables = [
        PDCode.compute_yamada,
        compute_yamada_polynomial,
        compute_yamada_from_states,
        Yamada.compute,
    ]

    for callable_ in public_callables:
        assert inspect.signature(callable_).parameters["n_jobs"].default == 1


@pytest.mark.parametrize(("n_jobs", "exception"), [(0, ValueError), (True, TypeError), (1.5, TypeError)])
def test_public_yamada_rejects_invalid_worker_counts_before_projection(
    monkeypatch,
    n_jobs,
    exception,
):
    monkeypatch.setattr(
        pd_code,
        "select_projection",
        lambda *args, **kwargs: pytest.fail("projection should not run"),
    )

    with pytest.raises(exception, match="n_jobs"):
        compute_yamada_polynomial(_embedded_edge(), sp.Symbol("Y"), n_jobs=n_jobs)


def test_public_yamada_rejects_unknown_method_before_projection(monkeypatch):
    monkeypatch.setattr(
        pd_code,
        "select_projection",
        lambda *args, **kwargs: pytest.fail("projection should not run"),
    )

    with pytest.raises(ValueError, match="method must be either"):
        compute_yamada_polynomial(_embedded_edge(), sp.Symbol("Y"), method="unknown")


@pytest.mark.parametrize(
    ("threshold", "exception"),
    [(-1, ValueError), (True, TypeError), (2.5, TypeError)],
)
def test_public_yamada_rejects_invalid_warning_threshold_before_projection(
    monkeypatch,
    threshold,
    exception,
):
    monkeypatch.setattr(
        pd_code,
        "select_projection",
        lambda *args, **kwargs: pytest.fail("projection should not run"),
    )

    with pytest.raises(exception, match="crossing_warning_threshold"):
        compute_yamada_polynomial(
            _embedded_edge(),
            sp.Symbol("Y"),
            crossing_warning_threshold=threshold,
        )
