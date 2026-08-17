from __future__ import annotations

import collections
import itertools
import json

from benchmark_yamada_end_to_end import multi_crossing_theta
from knotted_graph.invariants.yamada.polynomial import Yamada
from knotted_graph.projection import PDCode


def main():
    for crossings in range(2, 9):
        graph = multi_crossing_theta(crossings)
        processor = PDCode(graph)
        processor.compute(rotation_angles=(0.0, 0.0, 0.0))
        calculator = Yamada.from_PDCode(processor)

        by_graph = collections.Counter()
        by_graph_exponent = collections.Counter()
        for compact_graph, exponent in calculator._iter_compact_states():
            by_graph[compact_graph] += 1
            by_graph_exponent[(compact_graph, exponent)] += 1

        total = 3**crossings
        print(json.dumps({
            "crossings": crossings,
            "states": total,
            "unique_graphs": len(by_graph),
            "unique_graph_exponents": len(by_graph_exponent),
            "graph_duplication": total / len(by_graph),
            "graph_exponent_duplication": total / len(by_graph_exponent),
        }, separators=(",", ":")))


if __name__ == "__main__":
    main()
