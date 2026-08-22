import numpy as np
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


def test_crossing_preserves_repeated_arc_incidences():
    crossing = Crossing(id=0, point=Point(0, 0))
    crossing.add_incident_arc(5, -np.pi)
    crossing.add_incident_arc(6, -np.pi / 2)
    crossing.add_incident_arc(5, 0.0)
    crossing.add_incident_arc(7, np.pi / 2)
    assert crossing._raw_ccw_ordered_arcs.count(5) == 2
    crossing._correctly_overstrand = True
    assert crossing.pd_code.startswith("X[")
    assert crossing.pd_code.count("5") == 2
