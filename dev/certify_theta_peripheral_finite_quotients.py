from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from knotted_graph.projection import PDCode

import certify_theta_complement_group as cg
import discover_yamada_theta_collisions as core

TARGET_PAIRS = [
    ("pair13", (32, 58, 0.12), (39, 153, 0.05)),
    ("pair16", (32, 197, 0.12), (39, 102, 0.05)),
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


def presentation_and_vertex_meridians(graph):
    """Return cg.Presentation and the three based edge meridians at vertex ``u``.

    The meridians are based at the same trivalent vertex.  Changing the common
    base path conjugates all three simultaneously.  An ambient isotopy of an
    unlabeled theta-curve may permute the three edges; exchanging the two
    vertices simultaneously inverts the oriented meridians.  The canonical
    finite-quotient orbit used below quotients by exactly these ambiguities.
    """
    presentation = cg.complement_group_presentation(graph)

    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    arc_ids = sorted(pd.arcs)
    uf = UnionFind(arc_ids)

    # Match complement_group_presentation: identify the two arcs on every
    # over-passing strand before assigning generator numbers.
    for crossing_id, crossing in pd.crossings.items():
        incident_ids = list(dict.fromkeys(crossing._raw_ccw_ordered_arcs))
        incident = [pd.arcs[arc_id] for arc_id in incident_ids]
        by_z = sorted(incident, key=lambda arc: cg._endpoint_z(arc, crossing_id))
        over_arcs = by_z[2:]
        over_in, over_out = cg._incoming_outgoing(over_arcs, crossing_id)
        uf.union(over_in.id, over_out.id)

    roots = sorted({uf.find(arc_id) for arc_id in arc_ids})
    root_to_generator = {root: i for i, root in enumerate(roots)}

    def generator(arc_id):
        return root_to_generator[uf.find(arc_id)]

    u_vertex = next(vertex for vertex in pd.vertices.values() if vertex.key == "u")
    role_to_generator = {}
    for arc_id in u_vertex.ccw_ordered_arcs:
        arc = pd.arcs[arc_id]
        edge_data = pd.skeleton_graph.edges[arc.edge_key]
        role = int(edge_data["role"])
        role_to_generator[role] = generator(arc_id)
    if sorted(role_to_generator) != [0, 1, 2]:
        raise AssertionError(f"failed to recover three theta edge roles: {role_to_generator}")
    return presentation, tuple(role_to_generator[role] for role in range(3))


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


def canonical_peripheral_orbit(
    triple: tuple[int, int, int],
    *,
    elements,
    mul,
    inv,
) -> str:
    """Canonicalize the based meridian triple under isotopy ambiguities.

    We quotient by simultaneous target-group conjugation, arbitrary permutation
    of the three unlabeled theta edges, and simultaneous inversion (vertex
    exchange).  The resulting orbit is an invariant of the unlabeled embedded
    theta-curve for each finite-group representation.
    """
    candidates = []
    for conjugator in range(len(elements)):
        h_inv = inv[conjugator]
        for invert_all in (False, True):
            transformed = []
            for value in triple:
                g = inv[value] if invert_all else value
                conjugated = mul[mul[conjugator][g]][h_inv]
                transformed.append(elements[conjugated])
            # Edge labels are forgotten.
            candidates.append(tuple(sorted(transformed)))
    return repr(min(candidates))


def peripheral_histogram(presentation, vertex_generators, degree: int) -> dict:
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
                    values = [
                        pow1(assignment[g], exponent)  # type: ignore[arg-type]
                        for g, exponent in word
                    ]
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

    histogram = Counter()

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
            triple = tuple(int(assignment[g]) for g in vertex_generators)
            histogram[
                canonical_peripheral_orbit(
                    triple,
                    elements=elements,
                    mul=mul,
                    inv=inv,
                )
            ] += 1
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

    total = recurse()
    return {
        "degree": degree,
        "homomorphism_count": total,
        "peripheral_orbit_count": len(histogram),
        "histogram": dict(sorted(histogram.items())),
    }


def reconstruct(shadows, descriptor):
    shadow_index, bits, fraction = descriptor
    shadow = {item.index: item for item in shadows}[shadow_index]
    graph, _ = core.spatial_theta(shadow, bits, approach_fraction=fraction)
    return graph


def run(plantri: str, output: Path) -> dict:
    shadows = core.generate_shadows(plantri, 8)
    pair_results = []
    for pair_name, left_desc, right_desc in TARGET_PAIRS:
        left_graph = reconstruct(shadows, left_desc)
        right_graph = reconstruct(shadows, right_desc)
        left_presentation, left_meridians = presentation_and_vertex_meridians(left_graph)
        right_presentation, right_meridians = presentation_and_vertex_meridians(right_graph)

        degrees = {}
        distinguished = False
        for degree in (3, 4):
            left = peripheral_histogram(left_presentation, left_meridians, degree)
            right = peripheral_histogram(right_presentation, right_meridians, degree)
            same = left["histogram"] == right["histogram"]
            degrees[f"S{degree}"] = {"same_peripheral_histogram": same, "left": left, "right": right}
            distinguished = distinguished or not same

        record = {
            "pair": pair_name,
            "left": {"shadow": left_desc[0], "bits": left_desc[1], "vertex_generators": left_meridians},
            "right": {"shadow": right_desc[0], "bits": right_desc[1], "vertex_generators": right_meridians},
            "degrees": degrees,
            "peripheral_finite_quotient_distinguishes": distinguished,
        }
        pair_results.append(record)
        print("PERIPHERAL_QUOTIENT_RESULT=" + json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "invariant": (
            "histogram over all homomorphisms pi1(complement)->S_n of the common-basepoint "
            "trivalent-vertex meridian triple, modulo simultaneous conjugation, arbitrary "
            "edge permutation, and simultaneous inversion"
        ),
        "pairs": pair_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plantri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.plantri, args.output)


if __name__ == "__main__":
    main()
