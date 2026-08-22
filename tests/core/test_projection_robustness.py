import numpy as np
import pytest
from shapely import LineString, MultiLineString, Point

from knotted_graph.projection.geom import Crossing
from knotted_graph.projection.pd_code import PDCode, find_all_crossings


def test_crossing_at_internal_polyline_sample_is_not_dropped():
    # The two strands share XY=(1,1), but one is at z=+1 and the other at z=-1.
    # The crossing lands exactly on an artificial sampling point of both lines.
    lines = MultiLineString(
        [
            LineString([(0, 0, 1), (1, 1, 1), (2, 2, 1)]),
            LineString([(0, 2, -1), (1, 1, -1), (2, 0, -1)]),
        ]
    )
    crossings = find_all_crossings(lines)
    assert len(crossings) == 1
    assert crossings[0].distance(Point(1, 1)) < 1e-12


def test_true_shared_3d_endpoint_is_not_a_crossing():
    lines = MultiLineString(
        [
            LineString([(0, 0, 0), (1, 1, 0)]),
            LineString([(1, 1, 0), (2, 0, 0)]),
        ]
    )
    assert find_all_crossings(lines) == []


def test_crossing_distance_dedup_merges_internal_sample_duplicates():
    values = [(1.0, 3), (1.0 + 1e-10, 3), (2.0, 4)]
    result = PDCode._deduplicate_crossing_distances(values, tolerance=1e-8)
    assert len(result) == 2
    assert [crossing_id for _, crossing_id in result] == [3, 4]


def test_crossing_accepts_well_separated_transverse_half_edges():
    crossing = Crossing(id=0, point=Point(0.0, 0.0))
    for arc_id, angle in enumerate((0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi)):
        crossing.add_incident_arc(arc_id, angle)
    assert len(crossing.incident_arcs) == 4


def test_crossing_rejects_numerically_tangent_half_edges():
    crossing = Crossing(id=0, point=Point(0.0, 0.0))
    crossing.add_incident_arc(0, 0.0)
    crossing.add_incident_arc(1, 1e-12)
    crossing.add_incident_arc(2, np.pi)
    with pytest.raises(ValueError, match="numerically tangent"):
        crossing.add_incident_arc(3, np.pi + 1e-12)
