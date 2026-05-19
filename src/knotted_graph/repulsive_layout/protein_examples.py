from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from .models import CurveNetwork


ARC_COLORS = {
    "arc1": "#c33a2f",
    "arc2": "#2c67c7",
    "arc3": "#2e9f55",
    "arc1_closure": "#c33a2f",
    "arc2_backbone": "#2c67c7",
    "arc3_mg_bridge": "#2e9f55",
    "arc1_ca_closure": "#c33a2f",
    "arc2_cys_bridge": "#2c67c7",
    "arc3_backbone": "#2e9f55",
}

ION_RESNAMES = {"CA", "MG", "NA", "K", "ZN", "MN", "FE", "CU", "CO", "NI"}


@dataclass(frozen=True)
class ProteinExampleSpec:
    sample_id: str
    pdb_id: str
    chain_id: str
    default_total_arc_points: int
    description: str


PROTEIN_EXAMPLES: dict[str, ProteinExampleSpec] = {
    "1aoc": ProteinExampleSpec(
        sample_id="1aoc",
        pdb_id="1AOC",
        chain_id="A",
        default_total_arc_points=42,
        description="1AOC chain A theta_31 graph used in Repulsion_protein.ipynb.",
    ),
    "3ulk": ProteinExampleSpec(
        sample_id="3ulk",
        pdb_id="3ULK",
        chain_id="A",
        default_total_arc_points=72,
        description="3ULK chain A theta_41 protein-derived graph.",
    ),
    "5osq": ProteinExampleSpec(
        sample_id="5osq",
        pdb_id="5OSQ",
        chain_id="A",
        default_total_arc_points=72,
        description="5OSQ chain A protein-derived theta graph.",
    ),
}


def available_samples() -> tuple[str, ...]:
    return tuple(PROTEIN_EXAMPLES)


def ensure_pdb(pdb_id: str, cache_dir: Path, pdb_path: Path | None = None) -> Path:
    if pdb_path is not None:
        pdb_path = pdb_path.resolve()
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        return pdb_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{pdb_id.upper()}.pdb"
    if output.exists():
        return output
    urlretrieve(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", output)
    return output


def _load_chain(pdb_path: Path, chain_id: str):
    try:
        from Bio.PDB import PDBParser
    except ImportError as exc:
        raise RuntimeError("biopython is required for protein examples. Install knotted_graph[repulsion].") from exc

    structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
    return structure[0][chain_id]


def residues_with_alpha_carbon(chain) -> dict[int, object]:
    return {
        res.id[1]: res
        for res in chain
        if "CA" in res and res.resname.strip().upper() not in ION_RESNAMES
    }


def ion_residues(chain, *names: str) -> dict[int, object]:
    allowed = {name.upper() for name in names}
    return {
        res.id[1]: res
        for res in chain
        if res.id[0].strip() and res.resname.strip().upper() in allowed
    }


def residue_coord(residues: dict[int, object], residue_number: int) -> np.ndarray:
    return np.asarray(residues[residue_number]["CA"].coord, dtype=float)


def residue_by_curve_index(residues: list[object], curve_index: int) -> np.ndarray:
    return np.asarray(residues[curve_index - 1]["CA"].coord, dtype=float)


def ion_coord(residue) -> np.ndarray:
    return np.asarray(next(iter(residue.child_dict.values())).coord, dtype=float)


def backbone_by_number(residues: dict[int, object], a: int, b: int) -> np.ndarray:
    step = 1 if b >= a else -1
    missing = [i for i in range(a, b + step, step) if i not in residues]
    if missing:
        raise ValueError(f"Missing CA residues in backbone segment {a}->{b}: {missing[:20]}")
    return np.asarray([residue_coord(residues, i) for i in range(a, b + step, step)], dtype=float)


def backbone_by_curve_index(residues: list[object], a: int, b: int) -> np.ndarray:
    step = 1 if b >= a else -1
    idxs = list(range(a, b + step, step))
    return np.asarray([residue_by_curve_index(residues, i) for i in idxs], dtype=float)


def bridge(points: list[np.ndarray]) -> np.ndarray:
    return np.vstack(points)


def concat_polylines(parts: list[np.ndarray]) -> np.ndarray:
    out: list[np.ndarray] = []
    for part in parts:
        p = np.asarray(part, dtype=float)
        if not out:
            out.append(p)
        elif np.allclose(out[-1][-1], p[0]):
            out.append(p[1:])
        else:
            out.append(p)
    return np.vstack(out)


def resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= n_points:
        return pts.copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    if total < 1e-12:
        return np.repeat(pts[:1], n_points, axis=0)

    targets = np.linspace(0.0, total, n_points)
    out = []
    j = 0
    for t in targets:
        while j + 1 < len(cumulative) and cumulative[j + 1] < t:
            j += 1
        if j + 1 >= len(cumulative):
            out.append(pts[-1])
            continue
        a = cumulative[j]
        b = cumulative[j + 1]
        alpha = 0.0 if b - a < 1e-12 else (t - a) / (b - a)
        out.append((1.0 - alpha) * pts[j] + alpha * pts[j + 1])
    return np.asarray(out, dtype=float)


def allocate_arc_points(
    arcs: dict[str, np.ndarray],
    arc_order: tuple[str, ...],
    total_points: int,
    min_points: int,
) -> tuple[dict[str, int], dict[str, float]]:
    lengths = {
        name: float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
        for name, poly in arcs.items()
    }
    remaining = total_points - min_points * len(arc_order)
    if remaining < 0:
        raise ValueError("total_points must be at least min_points * number_of_arcs")

    total_length = sum(lengths.values())
    extras = {name: 0 for name in arc_order}
    if total_length > 1e-12 and remaining > 0:
        raw = {name: remaining * lengths[name] / total_length for name in arc_order}
        for name in arc_order:
            extras[name] = int(math.floor(raw[name]))
        leftovers = remaining - sum(extras.values())
        ranked = sorted(arc_order, key=lambda name: raw[name] - extras[name], reverse=True)
        for name in ranked[:leftovers]:
            extras[name] += 1

    return {name: min_points + extras[name] for name in arc_order}, lengths


def enclosing_sphere_point(points: np.ndarray, seed: int, radius_scale: float) -> np.ndarray:
    center = points.mean(axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1)) * radius_scale
    rng = np.random.default_rng(seed)
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return center + radius * v


def deterministic_closure_point(points: np.ndarray, direction: np.ndarray, radius_scale: float) -> np.ndarray:
    center = points.mean(axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    return center + radius_scale * radius * direction


def build_protein_example(
    sample_id: str,
    pdb_cache: Path,
    total_arc_points: int | None = None,
    seed: int = 7,
    pdb_path: Path | None = None,
) -> CurveNetwork:
    sample_id = sample_id.lower()
    if sample_id not in PROTEIN_EXAMPLES:
        raise ValueError(f"Unknown protein example {sample_id!r}. Choices: {', '.join(available_samples())}")

    spec = PROTEIN_EXAMPLES[sample_id]
    pdb = ensure_pdb(spec.pdb_id, pdb_cache, pdb_path=pdb_path)
    total_points = total_arc_points or spec.default_total_arc_points

    if sample_id == "1aoc":
        return build_1aoc(pdb, spec.chain_id, total_points, seed)
    if sample_id == "3ulk":
        return build_3ulk(pdb, spec.chain_id, total_points)
    if sample_id == "5osq":
        return build_5osq(pdb, spec.chain_id, total_points)
    raise AssertionError(sample_id)


def build_1aoc(pdb_path: Path, chain_id: str, total_points: int, seed: int) -> CurveNetwork:
    chain = _load_chain(pdb_path, chain_id)
    residues = [res for res in chain if not res.id[0].strip() and "CA" in res]
    all_points = np.asarray([res["CA"].coord for res in residues], dtype=float)
    closure_pt = enclosing_sphere_point(all_points, seed=seed, radius_scale=0.1)

    arc_order = ("arc1", "arc2", "arc3")
    raw_arcs = {
        "arc1": concat_polylines(
            [
                backbone_by_curve_index(residues, 140, 161),
                bridge([residue_by_curve_index(residues, 161), residue_by_curve_index(residues, 60)]),
                backbone_by_curve_index(residues, 60, 65),
                bridge([residue_by_curve_index(residues, 65), residue_by_curve_index(residues, 121)]),
                backbone_by_curve_index(residues, 121, 134),
            ]
        ),
        "arc2": concat_polylines(
            [
                bridge([residue_by_curve_index(residues, 140), residue_by_curve_index(residues, 88)]),
                backbone_by_curve_index(residues, 88, 95),
                bridge([residue_by_curve_index(residues, 95), residue_by_curve_index(residues, 10)]),
                backbone_by_curve_index(residues, 10, 8),
                bridge([residue_by_curve_index(residues, 8), closure_pt, residue_by_curve_index(residues, 167)]),
                backbone_by_curve_index(residues, 167, 172),
                bridge([residue_by_curve_index(residues, 172), residue_by_curve_index(residues, 134)]),
            ]
        ),
        "arc3": backbone_by_curve_index(residues, 140, 134),
    }
    targets, raw_lengths = allocate_arc_points(raw_arcs, arc_order, total_points, min_points=8)
    arc_polylines = {name: resample_polyline(raw_arcs[name], targets[name]) for name in arc_order}

    node_order = ("C140", "C134")
    return CurveNetwork(
        name="1AOC theta_31",
        node_order=node_order,
        node_positions={
            "C140": arc_polylines["arc1"][0].copy(),
            "C134": arc_polylines["arc1"][-1].copy(),
        },
        arc_order=arc_order,
        arc_polylines=arc_polylines,
        arc_specs={
            "arc1": "C140 ... C161 / C60 ... C65 / C121 ... C134",
            "arc2": "C140 / C88 ... C95 / C10 ... C8 / closure / C167 ... C172 / C134",
            "arc3": "C140 ... C134",
        },
        node_colors={"C140": "#c6a21f", "C134": "#1e4ca0"},
        arc_colors={name: ARC_COLORS[name] for name in arc_order},
        metadata={
            "sample_id": "1aoc",
            "pdb_id": "1AOC",
            "chain": chain_id,
            "closure_point": closure_pt.tolist(),
            "raw_arc_lengths": raw_lengths,
            "resampled_arc_points": targets,
        },
    )


def build_3ulk(pdb_path: Path, chain_id: str, total_points: int) -> CurveNetwork:
    chain = _load_chain(pdb_path, chain_id)
    residues = residues_with_alpha_carbon(chain)
    ions = ion_residues(chain, "MG")
    chain_points = np.asarray([res["CA"].coord for res in residues.values()], dtype=float)
    closure_pt = deterministic_closure_point(chain_points, np.array([0.33, -0.52, 0.79]), radius_scale=1.8)
    mg498 = ion_coord(ions[498])

    arc_order = ("arc1_closure", "arc2_backbone", "arc3_mg_bridge")
    raw_arcs = {
        "arc1_closure": concat_polylines(
            [
                backbone_by_number(residues, 217, 1),
                bridge([residue_coord(residues, 1), closure_pt, residue_coord(residues, 489)]),
                backbone_by_number(residues, 489, 393),
            ]
        ),
        "arc2_backbone": backbone_by_number(residues, 217, 393),
        "arc3_mg_bridge": bridge([residue_coord(residues, 217), mg498, residue_coord(residues, 393)]),
    }
    targets, raw_lengths = allocate_arc_points(raw_arcs, arc_order, total_points, min_points=10)
    arc_polylines = {name: resample_polyline(raw_arcs[name], targets[name]) for name in arc_order}

    node_order = ("D217", "E393")
    return CurveNetwork(
        name="3ULK theta_41",
        node_order=node_order,
        node_positions={
            "D217": arc_polylines["arc1_closure"][0].copy(),
            "E393": arc_polylines["arc1_closure"][-1].copy(),
        },
        arc_order=arc_order,
        arc_polylines=arc_polylines,
        arc_specs={
            "arc1_closure": "D217 ... M1 / closure / V489 ... E393",
            "arc2_backbone": "D217 ... E393",
            "arc3_mg_bridge": "D217 / Mg498 / E393",
        },
        node_colors={"D217": "#c6a21f", "E393": "#1e4ca0"},
        arc_colors={name: ARC_COLORS[name] for name in arc_order},
        metadata={
            "sample_id": "3ulk",
            "pdb_id": "3ULK",
            "chain": chain_id,
            "closure_point": closure_pt.tolist(),
            "raw_arc_lengths": raw_lengths,
            "resampled_arc_points": targets,
        },
    )


def build_5osq(pdb_path: Path, chain_id: str, total_points: int) -> CurveNetwork:
    chain = _load_chain(pdb_path, chain_id)
    residues = residues_with_alpha_carbon(chain)
    ions = ion_residues(chain, "CA")
    chain_points = np.asarray([res["CA"].coord for res in residues.values()], dtype=float)
    closure_pt = deterministic_closure_point(chain_points, np.array([-0.44, 0.68, 0.59]), radius_scale=1.8)
    ca503 = ion_coord(ions[503])
    ca504 = ion_coord(ions[504])

    arc_order = ("arc1_ca_closure", "arc2_cys_bridge", "arc3_backbone")
    raw_arcs = {
        "arc1_ca_closure": concat_polylines(
            [
                bridge([residue_coord(residues, 437), ca504, residue_coord(residues, 212)]),
                backbone_by_number(residues, 212, 366),
                bridge([residue_coord(residues, 366), ca503, residue_coord(residues, 182)]),
                backbone_by_number(residues, 182, 3),
                bridge([residue_coord(residues, 3), closure_pt, residue_coord(residues, 474)]),
                backbone_by_number(residues, 474, 469),
            ]
        ),
        "arc2_cys_bridge": concat_polylines(
            [
                backbone_by_number(residues, 437, 376),
                bridge([residue_coord(residues, 376), residue_coord(residues, 469)]),
            ]
        ),
        "arc3_backbone": backbone_by_number(residues, 437, 469),
    }
    targets, raw_lengths = allocate_arc_points(raw_arcs, arc_order, total_points, min_points=10)
    arc_polylines = {name: resample_polyline(raw_arcs[name], targets[name]) for name in arc_order}

    node_order = ("D437", "C469")
    return CurveNetwork(
        name="5OSQ theta",
        node_order=node_order,
        node_positions={
            "D437": arc_polylines["arc1_ca_closure"][0].copy(),
            "C469": arc_polylines["arc1_ca_closure"][-1].copy(),
        },
        arc_order=arc_order,
        arc_polylines=arc_polylines,
        arc_specs={
            "arc1_ca_closure": "D437 / Ca504 / S212 ... Q366 / Ca503 / V182 ... T3 / closure / L474 ... C469",
            "arc2_cys_bridge": "D437 ... C376 / C469",
            "arc3_backbone": "D437 ... C469",
        },
        node_colors={"D437": "#c6a21f", "C469": "#1e4ca0"},
        arc_colors={name: ARC_COLORS[name] for name in arc_order},
        metadata={
            "sample_id": "5osq",
            "pdb_id": "5OSQ",
            "chain": chain_id,
            "closure_point": closure_pt.tolist(),
            "raw_arc_lengths": raw_lengths,
            "resampled_arc_points": targets,
        },
    )


def set_special_node_distance(network: CurveNetwork, target_distance: float) -> tuple[float, float]:
    if len(network.node_order) != 2:
        raise ValueError("Special-node distance adjustment expects exactly two nodes")
    if target_distance <= 0:
        raise ValueError("target_distance must be positive")

    node_a, node_b = network.node_order
    p0 = np.asarray(network.node_positions[node_a], dtype=float)
    p1 = np.asarray(network.node_positions[node_b], dtype=float)
    current_distance = float(np.linalg.norm(p1 - p0))
    if current_distance < 1e-12:
        raise ValueError("Cannot set node distance because the two special nodes coincide")

    center = 0.5 * (p0 + p1)
    axis = (p1 - p0) / current_distance
    new_p0 = center - 0.5 * target_distance * axis
    new_p1 = center + 0.5 * target_distance * axis

    network.node_positions[node_a] = new_p0
    network.node_positions[node_b] = new_p1
    for arc_name in network.arc_order:
        polyline = np.asarray(network.arc_polylines[arc_name], dtype=float).copy()
        polyline[0] = new_p0
        polyline[-1] = new_p1
        network.arc_polylines[arc_name] = polyline

    return current_distance, target_distance
