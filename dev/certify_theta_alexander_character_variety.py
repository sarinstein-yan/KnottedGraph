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
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def presentation_with_roles(graph):
    """Complement presentation plus abstract-edge role of each Wirtinger generator."""
    presentation = cg.complement_group_presentation(graph)
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    arc_ids = sorted(pd.arcs)
    uf = UnionFind(arc_ids)

    for crossing_id, crossing in pd.crossings.items():
        incident_ids = list(dict.fromkeys(crossing._raw_ccw_ordered_arcs))
        incident = [pd.arcs[arc_id] for arc_id in incident_ids]
        by_z = sorted(incident, key=lambda arc: cg._endpoint_z(arc, crossing_id))
        over_in, over_out = cg._incoming_outgoing(by_z[2:], crossing_id)
        uf.union(over_in.id, over_out.id)

    roots = sorted({uf.find(a) for a in arc_ids})
    root_to_generator = {root: i for i, root in enumerate(roots)}
    generator_roles = [None] * len(roots)
    for arc_id in arc_ids:
        generator = root_to_generator[uf.find(arc_id)]
        role = int(pd.skeleton_graph.edges[pd.arcs[arc_id].edge_key]["role"])
        if generator_roles[generator] is None:
            generator_roles[generator] = role
        elif generator_roles[generator] != role:
            raise AssertionError("overstrand union mixed theta-edge roles")
    if any(role is None for role in generator_roles):
        raise AssertionError("missing generator role")
    return presentation, tuple(int(role) for role in generator_roles)


def relation_words(presentation):
    """Return relations as words (generator, exponent) with value 1."""
    words = []
    for out, over, incoming, sign in presentation.crossing_relations:
        # out = over^{-s} incoming over^{s}
        # hence out^{-1} over^{-s} incoming over^{s} = 1.
        words.append(((out, -1), (over, -sign), (incoming, 1), (over, sign)))
    words.extend(presentation.vertex_relations)
    return tuple(words)


def role_value(role: int, x: int, y: int, p: int) -> int:
    if role == 0:
        return x % p
    if role == 1:
        return y % p
    if role == 2:
        return pow((x * y) % p, p - 2, p)
    raise ValueError(role)


def fox_matrix_mod_p(presentation, roles, x: int, y: int, p: int):
    words = relation_words(presentation)
    n = presentation.generator_count
    values = [role_value(role, x, y, p) for role in roles]
    matrix = [[0] * n for _ in words]
    for row, word in enumerate(words):
        prefix = 1
        for generator, exponent in word:
            value = values[generator]
            if exponent == 1:
                matrix[row][generator] = (matrix[row][generator] + prefix) % p
                prefix = (prefix * value) % p
            elif exponent == -1:
                inv_value = pow(value, p - 2, p)
                matrix[row][generator] = (
                    matrix[row][generator] - prefix * inv_value
                ) % p
                prefix = (prefix * inv_value) % p
            else:
                raise AssertionError("relations must use unit exponents")
        if prefix % p != 1:
            raise AssertionError(
                f"relation failed abelianization at character ({x},{y}) mod {p}: {prefix}"
            )
    return matrix


def rank_mod_p(matrix, p: int) -> int:
    a = [row[:] for row in matrix]
    rows = len(a)
    cols = len(a[0]) if rows else 0
    rank = 0
    for col in range(cols):
        pivot = next((r for r in range(rank, rows) if a[r][col] % p), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        inv = pow(a[rank][col] % p, p - 2, p)
        a[rank] = [(value * inv) % p for value in a[rank]]
        for r in range(rows):
            if r == rank:
                continue
            factor = a[r][col] % p
            if factor:
                a[r] = [
                    (left - factor * right) % p
                    for left, right in zip(a[r], a[rank])
                ]
        rank += 1
        if rank == rows:
            break
    return rank


def character_profile(presentation, roles, p: int) -> dict:
    histogram = Counter()
    exceptional = []
    n = presentation.generator_count
    for x, y in itertools.product(range(1, p), repeat=2):
        matrix = fox_matrix_mod_p(presentation, roles, x, y, p)
        rank = rank_mod_p(matrix, p)
        nullity = n - rank
        histogram[nullity] += 1
        if nullity > 2:
            exceptional.append([x, y, nullity])
    # Full character-torus histogram is invariant under any GL(2,Z) change of
    # basis whose determinant is nonzero mod p; in particular under all theta
    # edge permutations and global orientation reversal.
    return {
        "prime": p,
        "character_count": (p - 1) ** 2,
        "nullity_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "exceptional_characters": exceptional,
    }


def reconstruct(shadows, desc):
    shadow_index, bits, fraction = desc
    graph, _ = core.spatial_theta(
        {s.index: s for s in shadows}[shadow_index], bits, approach_fraction=fraction
    )
    return graph


def run(plantri: str, output: Path) -> dict:
    shadows = core.generate_shadows(plantri, 8)
    results = []
    for name, left_desc, right_desc in TARGET_PAIRS:
        lp, lr = presentation_with_roles(reconstruct(shadows, left_desc))
        rp, rr = presentation_with_roles(reconstruct(shadows, right_desc))
        prime_results = {}
        distinguished = False
        for p in (5, 7, 11, 13, 17, 19):
            left = character_profile(lp, lr, p)
            right = character_profile(rp, rr, p)
            same_hist = left["nullity_histogram"] == right["nullity_histogram"]
            prime_results[str(p)] = {
                "same_nullity_histogram": same_hist,
                "left": left,
                "right": right,
            }
            distinguished = distinguished or not same_hist
        record = {
            "pair": name,
            "left": {"shadow": left_desc[0], "bits": left_desc[1], "generator_roles": lr},
            "right": {"shadow": right_desc[0], "bits": right_desc[1], "generator_roles": rr},
            "finite_field_profiles": prime_results,
            "alexander_character_profile_distinguishes": distinguished,
        }
        results.append(record)
        print("ALEXANDER_CHARACTER_RESULT=" + json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "invariant": (
            "histogram of Fox-Alexander matrix nullity over all nonzero characters "
            "(x,y) in F_p^* x F_p^*, using edge meridians m0=x,m1=y,m2=(xy)^-1"
        ),
        "pairs": results,
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
