from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import topoly
from topoly.params import Closure

from knotted_graph.projection import PDCode

import certify_theta_complement_group as cg
import discover_yamada_theta_collisions as core

TARGET_PAIRS = [
    ("pair13", (32, 58, 0.12), (39, 153, 0.05)),
    ("pair16", (32, 197, 0.12), (39, 102, 0.05)),
]


def crossing_data(graph):
    pd = PDCode(graph)
    pd.compute(rotation_angles=(0.0, 0.0, 0.0))
    v = np.zeros((3, 3), dtype=int)
    details = []
    for crossing_id, crossing in pd.crossings.items():
        incident_ids = list(dict.fromkeys(crossing._raw_ccw_ordered_arcs))
        incident = [pd.arcs[a] for a in incident_ids]
        by_z = sorted(incident, key=lambda arc: cg._endpoint_z(arc, crossing_id))
        under_in, under_out = cg._incoming_outgoing(by_z[:2], crossing_id)
        over_in, over_out = cg._incoming_outgoing(by_z[2:], crossing_id)
        over_t = cg._outgoing_tangent(over_out, crossing_id)
        under_t = cg._outgoing_tangent(under_out, crossing_id)
        det = float(over_t[0] * under_t[1] - over_t[1] * under_t[0])
        if abs(det) < 1e-12:
            raise AssertionError("degenerate crossing")
        sign = 1 if det > 0 else -1
        role_under = int(pd.skeleton_graph.edges[under_out.edge_key]["role"])
        role_over = int(pd.skeleton_graph.edges[over_out.edge_key]["role"])
        v[role_under, role_over] += sign
        if role_under != role_over:
            v[role_over, role_under] += sign
        details.append({"crossing": crossing_id, "roles": [role_under, role_over], "sign": sign})
    m = [
        int(-2 * v[0, 0] + v[0, 1] + v[0, 2] - v[1, 2]),
        int(-2 * v[1, 1] + v[0, 1] + v[1, 2] - v[0, 2]),
        int(-2 * v[2, 2] + v[0, 2] + v[1, 2] - v[0, 1]),
    ]
    return v.tolist(), m, details


def resample(points: np.ndarray, samples: int = 401):
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    total = cumulative[-1]
    targets = np.linspace(0.0, total, samples)
    out = np.empty((samples, 3))
    seg = 0
    for i, x in enumerate(targets):
        while seg + 1 < len(cumulative) - 1 and cumulative[seg + 1] < x:
            seg += 1
        den = cumulative[seg + 1] - cumulative[seg]
        t = 0.0 if den == 0 else (x - cumulative[seg]) / den
        out[i] = points[seg] + t * (points[seg + 1] - points[seg])
    return out, np.linspace(0.0, 1.0, samples)


def ribbon_sides(points: np.ndarray, half_twists: int, eps: float, trim: float):
    center, s = resample(points)
    tangent = np.gradient(center, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-14)
    normal = np.c_[-tangent[:, 1], tangent[:, 0], np.zeros(len(tangent))]
    norm = np.linalg.norm(normal, axis=1)
    bad = norm < 1e-8
    if np.any(bad):
        fallback = np.cross(tangent[bad], np.array([1.0, 0.0, 0.0]))
        bad2 = np.linalg.norm(fallback, axis=1) < 1e-8
        fallback[bad2] = np.cross(tangent[bad][bad2], np.array([0.0, 1.0, 0.0]))
        normal[bad] = fallback
        norm = np.linalg.norm(normal, axis=1)
    normal /= norm[:, None]
    binormal = np.cross(tangent, normal)
    binormal /= np.maximum(np.linalg.norm(binormal, axis=1)[:, None], 1e-14)
    angle = math.pi * half_twists * s
    offset = np.cos(angle)[:, None] * normal + np.sin(angle)[:, None] * binormal
    plus = center + eps * offset
    minus = center - eps * offset
    keep = (s >= trim) & (s <= 1.0 - trim)
    return {1: plus[keep], -1: minus[keep]}


def vertex_pairings(side_paths, center: np.ndarray, vertex: str):
    index = 0 if vertex == "u" else -1
    items = []
    for role in range(3):
        for side in (1, -1):
            point = side_paths[role][side][index]
            vec = point[:2] - center[:2]
            items.append((math.atan2(vec[1], vec[0]), role, side, point))
    items.sort()
    pairs = []
    used = set()
    n = len(items)
    for k in range(n):
        a = items[k]
        b = items[(k + 1) % n]
        ka, kb = (a[1], a[2]), (b[1], b[2])
        if a[1] != b[1] and ka not in used and kb not in used:
            pairs.append((ka, kb))
            used.add(ka); used.add(kb)
    if len(pairs) != 3 or len(used) != 6:
        raise AssertionError(f"vertex {vertex}: could not pair ribbon sides: {pairs}")
    mapping = {}
    for a, b in pairs:
        mapping[a] = b; mapping[b] = a
    return mapping


def connector(a: np.ndarray, b: np.ndarray, center: np.ndarray, steps: int = 16):
    va, vb = a[:2] - center[:2], b[:2] - center[:2]
    aa, ab = math.atan2(va[1], va[0]), math.atan2(vb[1], vb[0])
    delta = (ab - aa) % (2 * math.pi)
    if delta > math.pi:
        delta -= 2 * math.pi
    ts = np.linspace(0.0, 1.0, steps)
    ra, rb = np.linalg.norm(va), np.linalg.norm(vb)
    pts = []
    for t in ts:
        angle = aa + t * delta
        radius = (1-t) * ra + t * rb
        xy = center[:2] + radius * np.array([math.cos(angle), math.sin(angle)])
        z = (1-t) * a[2] + t * b[2]
        pts.append([xy[0], xy[1], z])
    return np.asarray(pts)


def build_boundary(graph, m, eps: float, trim: float):
    edge_points = [None] * 3
    for _, _, data in graph.edges(data=True):
        edge_points[int(data["role"])] = np.asarray(data["pts"], dtype=float)
    side_paths = {role: ribbon_sides(edge_points[role], m[role], eps, trim) for role in range(3)}
    centers = {name: np.asarray(graph.nodes[name]["pos"], dtype=float) for name in ("u", "v")}
    pair = {name: vertex_pairings(side_paths, centers[name], name) for name in ("u", "v")}

    visited = set(); components = []
    for start_role in range(3):
        for start_side in (1, -1):
            state = (start_role, start_side, 1)  # direction +1 means u->v
            if state in visited or (start_role, start_side, -1) in visited:
                continue
            points = []; current = state; first = state
            for _ in range(20):
                role, side, direction = current
                visited.add(current)
                path = side_paths[role][side]
                if direction == 1:
                    seg = path
                    vertex = "v"
                else:
                    seg = path[::-1]
                    vertex = "u"
                if points:
                    points.extend(seg[1:].tolist())
                else:
                    points.extend(seg.tolist())
                nxt_role, nxt_side = pair[vertex][(role, side)]
                a = np.asarray(points[-1])
                b = side_paths[nxt_role][nxt_side][-1 if vertex == "v" else 0]
                conn = connector(a, b, centers[vertex])
                points.extend(conn[1:].tolist())
                next_direction = -direction
                current = (nxt_role, nxt_side, next_direction)
                if current == first:
                    break
            else:
                raise AssertionError("boundary trace did not close")
            comp = np.asarray(points, dtype=float)
            if not np.allclose(comp[0], comp[-1]):
                comp = np.vstack([comp, comp[0]])
            components.append(comp)
    # The directed-state visit duplicates each physical component in reverse;
    # canonicalize by selecting unique coordinate sets via starting point cloud.
    unique = []
    for comp in components:
        if not any(np.allclose(np.sort(comp[:-1], axis=0), np.sort(other[:-1], axis=0)) for other in unique):
            unique.append(comp)
    if len(unique) != 3:
        # More robust duplicate test by rounded point-set keys.
        keys = {}; unique=[]
        for comp in components:
            key = tuple(sorted(map(tuple, np.round(comp[:-1], 8))))
            if key not in keys:
                keys[key]=1; unique.append(comp)
    if len(unique) != 3:
        raise AssertionError(f"expected 3 boundary components, got {len(unique)} from {len(components)} traces")
    return unique


def stable(value):
    if isinstance(value, dict):
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    return str(value)


def topoly_link_invariants(components):
    coords = np.vstack(components)
    cumulative = np.cumsum([len(c) for c in components])
    break_candidates = [
        [int(cumulative[0]-1), int(cumulative[1]-1)],
        [int(cumulative[0]), int(cumulative[1])],
    ]
    errors=[]
    for breaks in break_candidates:
        try:
            links = topoly.find_links(coords.tolist(), breaks=breaks, components=3, output="list", output_type="pdcode")
            if not links:
                raise RuntimeError("find_links returned no 3-component link")
            pdcode = links[0]
            homfly = topoly.homfly(pdcode, closure=Closure.CLOSED, chiral=True, translate=False, max_cross=45, run_parallel=False)
            jones = topoly.jones(pdcode, closure=Closure.CLOSED, chiral=True, translate=False, max_cross=45, run_parallel=False)
            return {"breaks": breaks, "pdcode": stable(pdcode), "homfly": stable(homfly), "jones": stable(jones)}
        except Exception as exc:
            errors.append(f"breaks={breaks}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Topoly link construction failed: " + " | ".join(errors))


def run(plantri: str, output: Path, xyz_dir: Path):
    shadows = core.generate_shadows(plantri, 8); by={s.index:s for s in shadows}
    xyz_dir.mkdir(parents=True, exist_ok=True)
    results=[]
    for pair_name, ld, rd in TARGET_PAIRS:
        pair_record={"pair":pair_name,"sides":{}}
        side_values={}
        for side_name, desc in (("left",ld),("right",rd)):
            shadow,bits,fraction=desc
            graph,_=core.spatial_theta(by[shadow],bits,approach_fraction=fraction)
            v,m,crossings=crossing_data(graph)
            trials=[]
            for eps,trim in ((0.006,0.025),(0.0045,0.025),(0.006,0.035)):
                # Try the published sign convention and its global mirror convention;
                # accept only a construction with zero pairwise GLN.
                accepted=None
                for twist_sign in (1,-1):
                    mm=[twist_sign*x for x in m]
                    comps=build_boundary(graph,mm,eps,trim)
                    gln=[[0.0]*3 for _ in range(3)]
                    for i in range(3):
                        for j in range(i+1,3):
                            val=float(topoly.gln(comps[i].tolist(),comps[j].tolist()))
                            gln[i][j]=gln[j][i]=val
                    if all(abs(gln[i][j])<0.15 for i in range(3) for j in range(i+1,3)):
                        accepted=(mm,comps,gln,twist_sign); break
                if accepted is None:
                    raise AssertionError(f"{pair_name}/{side_name}: no zero-linking convention for eps={eps}, trim={trim}, m={m}")
                mm,comps,gln,twist_sign=accepted
                inv=topoly_link_invariants(comps)
                trials.append({"eps":eps,"trim":trim,"raw_twist_parameters":m,"used_twist_parameters":mm,"twist_sign":twist_sign,"gln":gln,"invariants":inv})
                if eps==0.006 and trim==0.025:
                    for i,c in enumerate(comps): np.savetxt(xyz_dir/f"{pair_name}_{side_name}_component{i}.xyz",c,fmt="%.16g")
            signatures={(json.dumps(t["invariants"]["homfly"],sort_keys=True),json.dumps(t["invariants"]["jones"],sort_keys=True)) for t in trials}
            if len(signatures)!=1:
                raise AssertionError(f"{pair_name}/{side_name}: associated-link polynomial unstable across geometric trials")
            stable_inv=trials[0]["invariants"]
            pair_record["sides"][side_name]={"shadow":shadow,"bits":bits,"v_ij":v,"twist_parameters":m,"stable_homfly":stable_inv["homfly"],"stable_jones":stable_inv["jones"],"trials":trials}
            side_values[side_name]=stable_inv
        pair_record["homfly_distinguishes"] = side_values["left"]["homfly"] != side_values["right"]["homfly"]
        pair_record["jones_same"] = side_values["left"]["jones"] == side_values["right"]["jones"]
        results.append(pair_record)
        print("ASSOCIATED_LINK_RESULT="+json.dumps(pair_record,sort_keys=True),flush=True)
    payload={"construction":"Kauffman-Wolcott-Zhao associated link: blackboard band surface corrected by Vesnin-Oshmarina twist parameters to zero Seifert form","pairs":results}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2,sort_keys=True)); return payload


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plantri',required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--xyz-dir',type=Path,required=True); a=ap.parse_args(); run(a.plantri,a.output,a.xyz_dir)

if __name__=='__main__': main()
