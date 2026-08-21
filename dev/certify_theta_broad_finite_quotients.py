from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from knotted_graph.projection import PDCode

import certify_theta_complement_group as cg
import discover_yamada_theta_collisions as core

TARGET_PAIRS = [
    ("pair13", (32, 58, 0.12), (39, 153, 0.05)),
    ("pair16", (32, 197, 0.12), (39, 102, 0.05)),
]


@dataclass(frozen=True)
class FiniteGroup:
    name: str
    labels: tuple[str, ...]
    mul: tuple[tuple[int, ...], ...]
    inv: tuple[int, ...]
    identity: int


class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, v):
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def presentation_and_vertex_meridians(graph):
    presentation = cg.complement_group_presentation(graph)
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    arc_ids = sorted(pd.arcs)
    uf = UnionFind(arc_ids)
    for crossing_id, crossing in pd.crossings.items():
        incident_ids = list(dict.fromkeys(crossing._raw_ccw_ordered_arcs))
        incident = [pd.arcs[a] for a in incident_ids]
        by_z = sorted(incident, key=lambda arc: cg._endpoint_z(arc, crossing_id))
        over_in, over_out = cg._incoming_outgoing(by_z[2:], crossing_id)
        uf.union(over_in.id, over_out.id)
    roots = sorted({uf.find(a) for a in arc_ids})
    root_to_generator = {r: i for i, r in enumerate(roots)}

    def gen(arc_id):
        return root_to_generator[uf.find(arc_id)]

    vertex = next(v for v in pd.vertices.values() if v.key == "u")
    role_to_gen = {}
    for arc_id in vertex.ccw_ordered_arcs:
        arc = pd.arcs[arc_id]
        role = int(pd.skeleton_graph.edges[arc.edge_key]["role"])
        role_to_gen[role] = gen(arc_id)
    return presentation, tuple(role_to_gen[r] for r in range(3))


def _make_group(name, elements, compose, identity_element):
    elements = tuple(elements)
    index = {g: i for i, g in enumerate(elements)}
    mul = tuple(tuple(index[compose(a, b)] for b in elements) for a in elements)
    identity = index[identity_element]
    inv = []
    for i in range(len(elements)):
        inv.append(next(j for j in range(len(elements)) if mul[i][j] == identity and mul[j][i] == identity))
    return FiniteGroup(name, tuple(map(str, elements)), mul, tuple(inv), identity)


def dihedral(n: int) -> FiniteGroup:
    elements = [(r, f) for f in (0, 1) for r in range(n)]
    def compose(a, b):
        r, f = a; s, g = b
        return ((r + (-1 if f else 1) * s) % n, (f + g) % 2)
    return _make_group(f"D{n}", elements, compose, (0, 0))


def quaternion8() -> FiniteGroup:
    # Elements sign * basis in {1,i,j,k}, encoded (sign,basis), sign=+/-1.
    elements = [(s, b) for s in (1, -1) for b in range(4)]
    # basis multiplication: (sign,basis)
    table = {
        (0,0):(1,0),(0,1):(1,1),(0,2):(1,2),(0,3):(1,3),
        (1,0):(1,1),(2,0):(1,2),(3,0):(1,3),
        (1,1):(-1,0),(2,2):(-1,0),(3,3):(-1,0),
        (1,2):(1,3),(2,3):(1,1),(3,1):(1,2),
        (2,1):(-1,3),(3,2):(-1,1),(1,3):(-1,2),
    }
    def compose(a,b):
        sa,ba=a; sb,bb=b
        s,c = table[(ba,bb)]
        return (sa*sb*s,c)
    return _make_group("Q8", elements, compose, (1,0))


def parity(p):
    inversions = sum(p[i] > p[j] for i in range(len(p)) for j in range(i+1,len(p)))
    return inversions % 2


def alternating(n: int) -> FiniteGroup:
    elements = [p for p in itertools.permutations(range(n)) if parity(p) == 0]
    def compose(a,b):
        return tuple(a[b[i]] for i in range(n))
    return _make_group(f"A{n}", elements, compose, tuple(range(n)))


def canonical_peripheral(triple, group: FiniteGroup):
    candidates = []
    for h in range(len(group.labels)):
        hi = group.inv[h]
        for invert in (False, True):
            vals = []
            for g in triple:
                if invert:
                    g = group.inv[g]
                vals.append(group.labels[group.mul[group.mul[h][g]][hi]])
            candidates.append(tuple(sorted(vals)))
    return repr(min(candidates))


def profile(presentation, peripheral_generators, group: FiniteGroup):
    assignment = [None] * presentation.generator_count
    occurrence = Counter()
    for out, over, incoming, _ in presentation.crossing_relations:
        occurrence.update((out, over, incoming))
    for word in presentation.vertex_relations:
        occurrence.update(g for g,_ in word)

    mul, inv, identity = group.mul, group.inv, group.identity
    def prod(vals):
        x = identity
        for v in vals: x = mul[x][v]
        return x
    def power(v,e): return v if e == 1 else inv[v]

    def propagate():
        changed = True
        while changed:
            changed = False
            for out, over, incoming, sign in presentation.crossing_relations:
                go, gb, ga = assignment[out], assignment[over], assignment[incoming]
                if gb is not None and ga is not None:
                    lb = inv[gb] if sign == 1 else gb
                    rb = gb if sign == 1 else inv[gb]
                    expected = mul[mul[lb][ga]][rb]
                    if go is None:
                        assignment[out] = expected; changed = True
                    elif go != expected: return False
                elif gb is not None and go is not None:
                    lb = gb if sign == 1 else inv[gb]
                    rb = inv[gb] if sign == 1 else gb
                    expected = mul[mul[lb][go]][rb]
                    if ga is None:
                        assignment[incoming] = expected; changed = True
                    elif ga != expected: return False
            for word in presentation.vertex_relations:
                unknown = [i for i,(g,_) in enumerate(word) if assignment[g] is None]
                if not unknown:
                    if prod(power(assignment[g],e) for g,e in word) != identity: return False
                elif len(unknown) == 1:
                    i = unknown[0]; g,e = word[i]
                    prefix = prod(power(assignment[a],b) for a,b in word[:i])
                    suffix = prod(power(assignment[a],b) for a,b in word[i+1:])
                    target_power = mul[inv[prefix]][inv[suffix]]
                    target = target_power if e == 1 else inv[target_power]
                    if assignment[g] is None:
                        assignment[g] = target; changed = True
                    elif assignment[g] != target: return False
        return True

    hist = Counter()
    image_order_hist = Counter()

    def generated_subgroup_order(gens):
        seen = {identity}; frontier = [identity]
        while frontier:
            a = frontier.pop()
            for g in gens:
                for b in (mul[a][g], mul[g][a]):
                    if b not in seen:
                        seen.add(b); frontier.append(b)
        return len(seen)

    def recurse():
        snapshot = assignment.copy()
        if not propagate(): assignment[:] = snapshot; return 0
        unassigned = [i for i,v in enumerate(assignment) if v is None]
        if not unassigned:
            triple = tuple(int(assignment[g]) for g in peripheral_generators)
            hist[canonical_peripheral(triple, group)] += 1
            image_order_hist[generated_subgroup_order(set(int(v) for v in assignment))] += 1
            assignment[:] = snapshot
            return 1
        g = max(unassigned, key=lambda i: occurrence[i])
        stable = assignment.copy(); total = 0
        for value in range(len(group.labels)):
            assignment[:] = stable; assignment[g] = value; total += recurse()
        assignment[:] = snapshot
        return total

    total = recurse()
    return {
        "group": group.name,
        "order": len(group.labels),
        "homomorphism_count": total,
        "peripheral_histogram": dict(sorted(hist.items())),
        "image_subgroup_order_histogram": {str(k):v for k,v in sorted(image_order_hist.items())},
    }


def reconstruct(shadows, desc):
    shadow,bits,fraction=desc
    graph,_=core.spatial_theta({s.index:s for s in shadows}[shadow],bits,approach_fraction=fraction)
    return graph


def run(plantri: str, output: Path):
    shadows=core.generate_shadows(plantri,8)
    groups=[dihedral(n) for n in range(3,11)] + [quaternion8(), alternating(4), alternating(5)]
    results=[]
    for name,ld,rd in TARGET_PAIRS:
        lp,lm=presentation_and_vertex_meridians(reconstruct(shadows,ld))
        rp,rm=presentation_and_vertex_meridians(reconstruct(shadows,rd))
        gres={}; distinguished=False
        for group in groups:
            left=profile(lp,lm,group); right=profile(rp,rm,group)
            same=(left["homomorphism_count"]==right["homomorphism_count"] and
                  left["peripheral_histogram"]==right["peripheral_histogram"] and
                  left["image_subgroup_order_histogram"]==right["image_subgroup_order_histogram"])
            gres[group.name]={"same_profile":same,"left":left,"right":right}
            distinguished = distinguished or not same
            print("FINITE_GROUP="+json.dumps({"pair":name,"group":group.name,"same":same,"left_count":left['homomorphism_count'],"right_count":right['homomorphism_count']},sort_keys=True),flush=True)
        record={"pair":name,"left":{"shadow":ld[0],"bits":ld[1]},"right":{"shadow":rd[0],"bits":rd[1]},"groups":gres,"broad_finite_quotient_distinguishes":distinguished}
        results.append(record)
    payload={"pairs":results}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2,sort_keys=True))
    return payload


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plantri',required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); run(a.plantri,a.output)

if __name__=='__main__': main()
