from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import sympy as sp

from knotted_graph.projection import PDCode, compute_yamada_polynomial

import discover_yamada_theta_collisions as core

A = sp.Symbol("A")

# These two same-Yamada pairs are especially valuable because each member has an
# independently identified 8_20 constituent knot. Therefore, once the theta
# reconstruction is verified to have an eight-crossing projection, every member
# has theta crossing number exactly eight: c(theta) >= c(8_20)=8 and c(theta)<=8.
TARGET_PAIRS = [
    ((32, 58, 0.12), (39, 153, 0.05)),
    ((32, 197, 0.12), (39, 102, 0.05)),
]


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


@dataclass(frozen=True)
class Presentation:
    generator_count: int
    crossing_relations: tuple[tuple[int, int, int, int], ...]
    vertex_relations: tuple[tuple[tuple[int, int], ...], ...]


def _endpoint_z(arc, crossing_id: int) -> float:
    if arc.start_type == "x" and arc.start_id == crossing_id:
        return float(arc.line.coords[0][2])
    if arc.end_type == "x" and arc.end_id == crossing_id:
        return float(arc.line.coords[-1][2])
    raise ValueError("arc is not incident to crossing")


def _outgoing_tangent(arc, crossing_id: int) -> np.ndarray:
    if arc.start_type == "x" and arc.start_id == crossing_id:
        p0 = np.asarray(arc.line.coords[0][:2], dtype=float)
        p1 = np.asarray(arc.line.coords[1][:2], dtype=float)
        return p1 - p0
    if arc.end_type == "x" and arc.end_id == crossing_id:
        p0 = np.asarray(arc.line.coords[-2][:2], dtype=float)
        p1 = np.asarray(arc.line.coords[-1][:2], dtype=float)
        return p1 - p0
    raise ValueError("arc is not incident to crossing")


def _incoming_outgoing(arcs, crossing_id: int):
    incoming = [arc for arc in arcs if arc.end_type == "x" and arc.end_id == crossing_id]
    outgoing = [arc for arc in arcs if arc.start_type == "x" and arc.start_id == crossing_id]
    if len(incoming) != 1 or len(outgoing) != 1:
        raise ValueError(
            f"crossing {crossing_id}: expected one incoming and one outgoing arc "
            f"on a strand, got {len(incoming)} and {len(outgoing)}"
        )
    return incoming[0], outgoing[0]


def complement_group_presentation(graph: nx.MultiGraph) -> Presentation:
    """Wirtinger presentation of the spatial-graph complement group."""
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    arc_ids = sorted(pd.arcs)
    uf = UnionFind(arc_ids)

    raw_crossings = []
    for crossing_id, crossing in pd.crossings.items():
        incident_ids = list(dict.fromkeys(crossing._raw_ccw_ordered_arcs))
        if len(incident_ids) != 4:
            raise ValueError(f"crossing {crossing_id}: expected four distinct arcs")
        incident = [pd.arcs[arc_id] for arc_id in incident_ids]
        by_z = sorted(incident, key=lambda arc: _endpoint_z(arc, crossing_id))
        under_arcs, over_arcs = by_z[:2], by_z[2:]
        under_in, under_out = _incoming_outgoing(under_arcs, crossing_id)
        over_in, over_out = _incoming_outgoing(over_arcs, crossing_id)
        uf.union(over_in.id, over_out.id)

        over_tangent = _outgoing_tangent(over_out, crossing_id)
        under_tangent = _outgoing_tangent(under_out, crossing_id)
        determinant = float(
            over_tangent[0] * under_tangent[1]
            - over_tangent[1] * under_tangent[0]
        )
        if abs(determinant) < 1e-12:
            raise ValueError(f"crossing {crossing_id}: degenerate oriented tangents")
        sign = 1 if determinant > 0 else -1
        raw_crossings.append((under_out.id, over_out.id, under_in.id, sign))

    roots = sorted({uf.find(arc_id) for arc_id in arc_ids})
    root_to_generator = {root: i for i, root in enumerate(roots)}

    def generator(arc_id: int) -> int:
        return root_to_generator[uf.find(arc_id)]

    crossing_relations = tuple(
        (generator(out_id), generator(over_id), generator(in_id), sign)
        for out_id, over_id, in_id, sign in raw_crossings
    )

    vertex_relations = []
    for vertex in pd.vertices.values():
        word = []
        for arc_id in vertex.ccw_ordered_arcs:
            arc = pd.arcs[arc_id]
            if arc.start_type == "v" and arc.start_id == vertex.id:
                exponent = 1
            elif arc.end_type == "v" and arc.end_id == vertex.id:
                exponent = -1
            else:
                raise ValueError("vertex incidence is inconsistent")
            word.append((generator(arc_id), exponent))
        vertex_relations.append(tuple(word))

    return Presentation(
        generator_count=len(roots),
        crossing_relations=crossing_relations,
        vertex_relations=tuple(vertex_relations),
    )


def permutation_group(n: int):
    elements = list(itertools.permutations(range(n)))
    identity = tuple(range(n))
    index = {element: i for i, element in enumerate(elements)}

    def compose(a, b):
        return tuple(a[b[i]] for i in range(n))

    multiplication = [[index[compose(a, b)] for b in elements] for a in elements]
    inverse = []
    identity_index = index[identity]
    for a_index in range(len(elements)):
        inverse.append(
            next(
                b_index
                for b_index in range(len(elements))
                if multiplication[a_index][b_index] == identity_index
                and multiplication[b_index][a_index] == identity_index
            )
        )
    return elements, multiplication, inverse, identity_index


def count_homomorphisms(presentation: Presentation, degree: int) -> int:
    elements, mul, inv, identity = permutation_group(degree)
    order = len(elements)
    assignment: list[int | None] = [None] * presentation.generator_count

    relation_occurrence = Counter()
    for out, over, incoming, _ in presentation.crossing_relations:
        relation_occurrence.update((out, over, incoming))
    for word in presentation.vertex_relations:
        relation_occurrence.update(generator for generator, _ in word)

    def product(values):
        result = identity
        for value in values:
            result = mul[result][value]
        return result

    def pow1(value: int, exponent: int) -> int:
        return value if exponent == 1 else inv[value]

    def propagate() -> bool:
        changed = True
        while changed:
            changed = False
            for out, over, incoming, sign in presentation.crossing_relations:
                go, gb, ga = assignment[out], assignment[over], assignment[incoming]
                if gb is not None and ga is not None:
                    left_b = inv[gb] if sign == 1 else gb
                    right_b = gb if sign == 1 else inv[gb]
                    expected = mul[mul[left_b][ga]][right_b]
                    if go is None:
                        assignment[out] = expected
                        changed = True
                    elif go != expected:
                        return False
                elif gb is not None and go is not None:
                    left_b = gb if sign == 1 else inv[gb]
                    right_b = inv[gb] if sign == 1 else gb
                    expected = mul[mul[left_b][go]][right_b]
                    if ga is None:
                        assignment[incoming] = expected
                        changed = True
                    elif ga != expected:
                        return False

            for word in presentation.vertex_relations:
                unknown_positions = [
                    i for i, (generator_id, _) in enumerate(word)
                    if assignment[generator_id] is None
                ]
                if not unknown_positions:
                    values = [pow1(assignment[g], exponent) for g, exponent in word]  # type: ignore[arg-type]
                    if product(values) != identity:
                        return False
                elif len(unknown_positions) == 1:
                    missing_pos = unknown_positions[0]
                    missing_generator, exponent = word[missing_pos]
                    prefix = product(
                        pow1(assignment[g], e)  # type: ignore[arg-type]
                        for g, e in word[:missing_pos]
                    )
                    suffix = product(
                        pow1(assignment[g], e)  # type: ignore[arg-type]
                        for g, e in word[missing_pos + 1 :]
                    )
                    target_power = mul[inv[prefix]][inv[suffix]]
                    target = target_power if exponent == 1 else inv[target_power]
                    current = assignment[missing_generator]
                    if current is None:
                        assignment[missing_generator] = target
                        changed = True
                    elif current != target:
                        return False
        return True

    def recurse() -> int:
        snapshot = assignment.copy()
        if not propagate():
            assignment[:] = snapshot
            return 0
        try:
            generator_id = max(
                (i for i, value in enumerate(assignment) if value is None),
                key=lambda i: relation_occurrence[i],
            )
        except ValueError:
            assignment[:] = snapshot
            return 1

        total = 0
        propagated_snapshot = assignment.copy()
        for value in range(order):
            assignment[:] = propagated_snapshot
            assignment[generator_id] = value
            total += recurse()
        assignment[:] = snapshot
        return total

    return recurse()


def trivial_theta() -> nx.MultiGraph:
    u = np.array([-1.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0])
    t = np.linspace(0.0, 1.0, 40)
    x = -1.0 + 2.0 * t
    graph = nx.MultiGraph()
    graph.add_node("u", pos=u)
    graph.add_node("v", pos=v)
    graph.add_edge("u", "v", pts=np.c_[x, np.zeros_like(t), np.zeros_like(t)])
    graph.add_edge("u", "v", pts=np.c_[x, 0.7 * np.sin(np.pi * t), np.zeros_like(t)])
    graph.add_edge("u", "v", pts=np.c_[x, -0.7 * np.sin(np.pi * t), np.zeros_like(t)])
    return graph


def reconstruct(shadows, descriptor):
    shadow_index, bits, fraction = descriptor
    shadow = {item.index: item for item in shadows}[shadow_index]
    graph, _ = core.spatial_theta(shadow, bits, approach_fraction=fraction)
    result = compute_yamada_polynomial(
        graph,
        A,
        rotation_angles=(0.0, 0.0, 0.0),
        normalize=True,
        crossing_warning_threshold=None,
        return_result=True,
    )
    if int(result.projection.num_crossings) != 8:
        raise AssertionError("candidate no longer reconstructs with exactly 8 crossings")
    return graph, sp.expand(result.polynomial)


def run(plantri: str, output: Path) -> dict:
    trivial_presentation = complement_group_presentation(trivial_theta())
    sanity = {}
    for degree in (3, 4):
        observed = count_homomorphisms(trivial_presentation, degree)
        expected = math.factorial(degree) ** 2
        if observed != expected:
            raise AssertionError(
                f"trivial theta S_{degree} representation count {observed} != {expected}"
            )
        sanity[f"S{degree}"] = observed

    shadows = core.generate_shadows(plantri, 8)
    pair_results = []
    for pair_index, (left_desc, right_desc) in enumerate(TARGET_PAIRS):
        left_graph, left_yamada = reconstruct(shadows, left_desc)
        right_graph, right_yamada = reconstruct(shadows, right_desc)
        if sp.simplify(left_yamada - right_yamada) != 0:
            raise AssertionError(f"pair {pair_index}: Yamada equality did not reproduce")

        left_presentation = complement_group_presentation(left_graph)
        right_presentation = complement_group_presentation(right_graph)
        fingerprints = {}
        distinguished = False
        for degree in (3, 4):
            left_count = count_homomorphisms(left_presentation, degree)
            right_count = count_homomorphisms(right_presentation, degree)
            fingerprints[f"S{degree}"] = {"left": left_count, "right": right_count}
            distinguished = distinguished or left_count != right_count

        record = {
            "pair_index": pair_index,
            "left": {
                "shadow": left_desc[0],
                "bits": left_desc[1],
                "bitstring": format(left_desc[1], "08b"),
            },
            "right": {
                "shadow": right_desc[0],
                "bits": right_desc[1],
                "bitstring": format(right_desc[1], "08b"),
            },
            "normalized_yamada": str(left_yamada),
            "left_generator_count": left_presentation.generator_count,
            "right_generator_count": right_presentation.generator_count,
            "finite_group_hom_counts": fingerprints,
            "complement_group_distinguished": distinguished,
            "exact_crossing_number_argument": "Each candidate has an 8_20 constituent knot, hence theta crossing number >= 8; the certified projection has 8 crossings, hence theta crossing number = 8.",
        }
        pair_results.append(record)
        print("COMPLEMENT_GROUP_RESULT=" + json.dumps(record, sort_keys=True), flush=True)

    result = {
        "trivial_theta_sanity": sanity,
        "pairs": pair_results,
        "distinguished_pair_count": sum(
            int(record["complement_group_distinguished"]) for record in pair_results
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(
        "SUMMARY="
        + json.dumps(
            {
                "distinguished_pair_count": result["distinguished_pair_count"],
                "trivial_theta_sanity": sanity,
            },
            sort_keys=True,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output)


if __name__ == "__main__":
    main()
