"""Deterministic crosslink-rewiring null models and robustness statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from math import prod
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np

from knotted_graph.inputs.crosslinks import (
    METAL_RESIDUES,
    CrosslinkEndpoint,
    CrosslinkRecord,
    CrosslinkedProteinInputResult,
    ResidueKey,
    build_crosslinked_protein_graph,
)
from knotted_graph.layout.repulsive.graph_io import graph_to_curve_arrays
from knotted_graph.layout.repulsive.metrics import clearance_report

from .models import ProteinTopologyAnalysis


@dataclass
class NullGraphRecord:
    replicate: int
    seed: int
    graph: nx.MultiGraph
    crosslinks: list[CrosslinkRecord]
    issues: list[str]


@dataclass(frozen=True)
class NullRobustnessComparison:
    natural_robustness_r1: float
    null_values: tuple[float, ...]
    null_mean: float
    null_standard_deviation: float
    z_score: float | None
    empirical_p_greater_equal: float
    empirical_p_less_equal: float


def canonicalize_null_embedding(
    graph: nx.MultiGraph,
    *,
    seed: int,
    points_per_edge: int = 7,
) -> nx.MultiGraph:
    """Embed a rewired null graph deterministically with low projection complexity.

    This mode intentionally discards the folded-protein coordinates after endpoint
    rewiring.  It is an abstract-connectivity null, useful when coordinate-preserving
    rewires create diagrams too large for exact Yamada evaluation.  Planar abstract
    graphs use a planar layout; nonplanar graphs use a seeded spring layout.  Edge
    interiors occupy deterministic height layers so projected crossings are genuine
    over/under crossings rather than 3-D intersections.
    """

    if points_per_edge < 3:
        raise ValueError("points_per_edge must be at least 3")
    result = graph.copy()
    simple = nx.Graph(result)
    planar, _ = nx.check_planarity(simple)
    if planar:
        layout = nx.planar_layout(simple, scale=10.0)
        algorithm = "planar"
    else:
        layout = nx.spring_layout(
            simple,
            seed=seed,
            dim=2,
            scale=10.0,
            iterations=1000,
        )
        algorithm = "seeded_spring"
    for node, xy in layout.items():
        result.nodes[node]["pos"] = np.asarray([xy[0], xy[1], 0.0], dtype=float)

    grouped: dict[frozenset[object], list[tuple[object, object, object]]] = {}
    edge_refs = list(result.edges(keys=True))
    for u, v, key in edge_refs:
        grouped.setdefault(frozenset((u, v)), []).append((u, v, key))
    ordered_edges = sorted(
        edge_refs,
        key=lambda edge: (repr(edge[0]), repr(edge[1]), repr(edge[2])),
    )
    edge_layers = {
        edge: index - 0.5 * (len(ordered_edges) - 1)
        for index, edge in enumerate(ordered_edges)
    }
    parameters = np.linspace(0.0, 1.0, points_per_edge)
    for parallel_edges in grouped.values():
        parallel_edges.sort(key=lambda edge: repr(edge[2]))
        center = 0.5 * (len(parallel_edges) - 1)
        for lane, edge in enumerate(parallel_edges):
            u, v, key = edge
            start = np.asarray(result.nodes[u]["pos"], dtype=float)
            end = np.asarray(result.nodes[v]["pos"], dtype=float)
            delta = end[:2] - start[:2]
            length = float(np.linalg.norm(delta))
            perpendicular = (
                np.asarray([-delta[1], delta[0]], dtype=float) / length
                if length > 1e-12
                else np.asarray([1.0, 0.0], dtype=float)
            )
            points = []
            for parameter in parameters:
                point = (1.0 - parameter) * start + parameter * end
                envelope = float(np.sin(np.pi * parameter))
                point[:2] += perpendicular * (0.6 * (lane - center) * envelope)
                point[2] += 0.15 * edge_layers[edge] * envelope
                points.append(point.copy())
            result.edges[u, v, key]["pts"] = np.asarray(points, dtype=float)

    vertices, mapping = graph_to_curve_arrays(result)
    clearance = clearance_report(
        vertices,
        mapping.edge_vertex_indices,
        mapping.edge_order,
    )
    minimum_distance = float(clearance["min_non_adjacent_segment_distance"])
    if not np.isfinite(minimum_distance) or minimum_distance <= 1e-8:
        raise RuntimeError(
            "Canonical null embedding has no positive non-adjacent edge clearance"
        )
    result.graph.update(
        null_embedding_mode="canonical_low_crossing",
        canonical_layout_algorithm=algorithm,
        canonical_layout_seed=seed,
        canonical_points_per_edge=points_per_edge,
        canonical_min_non_adjacent_segment_distance=minimum_distance,
    )
    return result


def _residue_key(record: Mapping[str, object]) -> ResidueKey:
    return ResidueKey(
        chain_id=str(record.get("chain_id", "?")),
        sequence_id=str(record.get("sequence_id", "?")),
        insertion_code=str(record.get("insertion_code", "")),
    )


def _residue_number(residue: ResidueKey) -> int | None:
    try:
        return int(residue.sequence_id)
    except ValueError:
        return None


def _candidate_pools(
    atom_records: Sequence[Mapping[str, object]],
    chain_ids: Sequence[str],
    backbone_atom: str,
) -> tuple[
    dict[tuple[str, str], list[tuple[ResidueKey, str]]],
    dict[str, list[tuple[ResidueKey, str]]],
]:
    by_chain_and_name: dict[tuple[str, str], list[tuple[ResidueKey, str]]] = {}
    by_chain: dict[str, list[tuple[ResidueKey, str]]] = {}
    seen: set[ResidueKey] = set()
    selected = set(chain_ids)
    for record in atom_records:
        if str(record.get("group", "ATOM")).upper() != "ATOM":
            continue
        if str(record.get("atom_name", "")).upper() != backbone_atom.upper():
            continue
        residue = _residue_key(record)
        if residue.chain_id not in selected or residue in seen:
            continue
        seen.add(residue)
        residue_name = str(record.get("residue_name", "UNK")).upper()
        item = (residue, residue_name)
        by_chain.setdefault(residue.chain_id, []).append(item)
        by_chain_and_name.setdefault((residue.chain_id, residue_name), []).append(item)
    return by_chain_and_name, by_chain


def _endpoint_candidates(
    endpoint: CrosslinkEndpoint,
    by_chain_and_name: Mapping[tuple[str, str], list[tuple[ResidueKey, str]]],
    by_chain: Mapping[str, list[tuple[ResidueKey, str]]],
    *,
    preserve_residue_names: bool,
) -> list[tuple[ResidueKey, str]]:
    if preserve_residue_names:
        named = by_chain_and_name.get(
            (endpoint.residue.chain_id, endpoint.residue_name.upper()),
            [],
        )
        if named:
            return list(named)
    return list(by_chain.get(endpoint.residue.chain_id, []))


def _replace_endpoint(
    template: CrosslinkEndpoint,
    candidate: tuple[ResidueKey, str],
) -> CrosslinkEndpoint:
    residue, residue_name = candidate
    return CrosslinkEndpoint(
        residue=residue,
        residue_name=residue_name,
        atom_name=template.atom_name,
    )


def _randomized_pair(
    record: CrosslinkRecord,
    by_chain_and_name: Mapping[tuple[str, str], list[tuple[ResidueKey, str]]],
    by_chain: Mapping[str, list[tuple[ResidueKey, str]]],
    rng: np.random.Generator,
    *,
    preserve_residue_names: bool,
) -> tuple[CrosslinkEndpoint, CrosslinkEndpoint]:
    endpoint_a = record.endpoint_a
    endpoint_b = record.endpoint_b
    a_is_metal = endpoint_a.residue_name.upper() in METAL_RESIDUES
    b_is_metal = endpoint_b.residue_name.upper() in METAL_RESIDUES
    if a_is_metal != b_is_metal:
        fixed = endpoint_a if a_is_metal else endpoint_b
        template = endpoint_b if a_is_metal else endpoint_a
        candidates = _endpoint_candidates(
            template,
            by_chain_and_name,
            by_chain,
            preserve_residue_names=preserve_residue_names,
        )
        if not candidates:
            return endpoint_a, endpoint_b
        randomized = _replace_endpoint(
            template, candidates[int(rng.integers(len(candidates)))]
        )
        return (fixed, randomized) if a_is_metal else (randomized, fixed)

    candidates_a = _endpoint_candidates(
        endpoint_a,
        by_chain_and_name,
        by_chain,
        preserve_residue_names=preserve_residue_names,
    )
    candidates_b = _endpoint_candidates(
        endpoint_b,
        by_chain_and_name,
        by_chain,
        preserve_residue_names=preserve_residue_names,
    )
    possible = [
        (candidate_a, candidate_b)
        for candidate_a in candidates_a
        for candidate_b in candidates_b
        if candidate_a[0] != candidate_b[0]
    ]
    if not possible:
        return endpoint_a, endpoint_b

    target_a = _residue_number(endpoint_a.residue)
    target_b = _residue_number(endpoint_b.residue)
    target_separation = (
        abs(target_a - target_b)
        if target_a is not None
        and target_b is not None
        and endpoint_a.residue.chain_id == endpoint_b.residue.chain_id
        else None
    )
    original_key = frozenset((endpoint_a.residue, endpoint_b.residue))
    alternatives = [
        pair
        for pair in possible
        if frozenset((pair[0][0], pair[1][0])) != original_key
    ]
    if alternatives:
        possible = alternatives
    if target_separation is not None:
        deviations = np.asarray(
            [
                abs(
                    abs(
                        (_residue_number(pair[0][0]) or 0)
                        - (_residue_number(pair[1][0]) or 0)
                    )
                    - target_separation
                )
                for pair in possible
            ],
            dtype=float,
        )
        scale = max(2.0, 0.2 * target_separation)
        weights = np.exp(-deviations / scale)
        weights /= weights.sum()
        selected = int(rng.choice(len(possible), p=weights))
    else:
        selected = int(rng.integers(len(possible)))
    candidate_a, candidate_b = possible[selected]
    return _replace_endpoint(endpoint_a, candidate_a), _replace_endpoint(
        endpoint_b, candidate_b
    )


def _endpoint_pair_key(
    first: CrosslinkEndpoint,
    second: CrosslinkEndpoint,
) -> frozenset[ResidueKey]:
    return frozenset((first.residue, second.residue))


def _matching_span_score(
    pairs: Sequence[tuple[CrosslinkEndpoint, CrosslinkEndpoint]],
    target_spans: Sequence[int],
) -> float:
    spans = []
    for first, second in pairs:
        first_number = _residue_number(first.residue)
        second_number = _residue_number(second.residue)
        if first_number is None or second_number is None:
            return 0.0
        spans.append(abs(first_number - second_number))
    return float(
        sum(abs(first - second) for first, second in zip(sorted(spans), target_spans))
    )


def _randomized_intrachain_disulfide_matching(
    records: Sequence[CrosslinkRecord],
    rng: np.random.Generator,
) -> list[tuple[CrosslinkEndpoint, CrosslinkEndpoint]] | None:
    """Re-pair the same cysteine endpoints without reusing a physical site."""

    if len(records) < 2:
        return None
    endpoints = [
        endpoint
        for record in records
        for endpoint in (record.endpoint_a, record.endpoint_b)
    ]
    if len({endpoint.residue for endpoint in endpoints}) != len(endpoints):
        return None
    original = {
        _endpoint_pair_key(record.endpoint_a, record.endpoint_b) for record in records
    }
    target_spans = sorted(
        abs(
            (_residue_number(record.endpoint_a.residue) or 0)
            - (_residue_number(record.endpoint_b.residue) or 0)
        )
        for record in records
    )
    candidates: list[
        tuple[float, list[tuple[CrosslinkEndpoint, CrosslinkEndpoint]]]
    ] = []
    attempts = max(500, 100 * len(records))
    for _ in range(attempts):
        order = rng.permutation(len(endpoints))
        pairs = [
            (endpoints[int(order[index])], endpoints[int(order[index + 1])])
            for index in range(0, len(order), 2)
        ]
        pair_keys = {_endpoint_pair_key(first, second) for first, second in pairs}
        if len(pair_keys) != len(pairs) or pair_keys == original:
            continue
        if any(first.residue == second.residue for first, second in pairs):
            continue
        candidates.append((_matching_span_score(pairs, target_spans), pairs))
    if not candidates:
        return None
    best_score = min(score for score, _ in candidates)
    tolerance = max(2.0 * len(records), 0.15 * sum(target_spans))
    plausible = [
        pairs for score, pairs in candidates if score <= best_score + tolerance
    ]
    chosen = plausible[int(rng.integers(len(plausible)))]
    return sorted(
        chosen,
        key=lambda pair: tuple(
            sorted((pair[0].residue.label, pair[1].residue.label))
        ),
    )


def _all_endpoint_perfect_matchings(
    endpoints: tuple[CrosslinkEndpoint, ...],
):
    """Yield every unordered perfect matching exactly once."""

    if not endpoints:
        yield ()
        return
    first = endpoints[0]
    for index in range(1, len(endpoints)):
        second = endpoints[index]
        remainder = endpoints[1:index] + endpoints[index + 1 :]
        for suffix in _all_endpoint_perfect_matchings(remainder):
            yield ((first, second), *suffix)


def _eligible_intrachain_disulfide_matchings(
    records: Sequence[CrosslinkRecord],
    *,
    max_enumerated_matchings: int,
) -> list[tuple[tuple[CrosslinkEndpoint, CrosslinkEndpoint], ...]]:
    """Enumerate the exact span-matched non-native perfect-matching null."""

    if len(records) < 2:
        return []
    endpoints = tuple(
        endpoint
        for record in records
        for endpoint in (record.endpoint_a, record.endpoint_b)
    )
    if len({endpoint.residue for endpoint in endpoints}) != len(endpoints):
        raise ValueError("Disulfide endpoints must be unique physical residues")
    matching_count = prod(range(1, len(endpoints), 2))
    if matching_count > max_enumerated_matchings:
        raise ValueError(
            f"Exact disulfide null has {matching_count} perfect matchings, "
            f"exceeding max_enumerated_matchings={max_enumerated_matchings}"
        )
    original = {
        _endpoint_pair_key(record.endpoint_a, record.endpoint_b) for record in records
    }
    target_spans = sorted(
        abs(
            (_residue_number(record.endpoint_a.residue) or 0)
            - (_residue_number(record.endpoint_b.residue) or 0)
        )
        for record in records
    )
    candidates = []
    for matching in _all_endpoint_perfect_matchings(endpoints):
        pair_keys = {
            _endpoint_pair_key(first, second) for first, second in matching
        }
        if pair_keys == original:
            continue
        ordered = tuple(
            sorted(
                matching,
                key=lambda pair: tuple(
                    sorted((pair[0].residue.label, pair[1].residue.label))
                ),
            )
        )
        candidates.append((_matching_span_score(ordered, target_spans), ordered))
    if not candidates:
        return []
    best_score = min(score for score, _ in candidates)
    tolerance = max(2.0 * len(records), 0.15 * sum(target_spans))
    eligible = {
        tuple(
            sorted(
                (
                    tuple(sorted((first.residue.label, second.residue.label)))
                    for first, second in matching
                )
            )
        ): matching
        for score, matching in candidates
        if score <= best_score + tolerance
    }
    return [eligible[key] for key in sorted(eligible)]


def randomize_crosslinks(
    crosslinks: Sequence[CrosslinkRecord],
    atom_records: Sequence[Mapping[str, object]],
    *,
    chain_ids: Sequence[str],
    backbone_atom: str = "CA",
    seed: int = 0,
    replicate: int = 0,
    preserve_residue_names: bool = True,
) -> list[CrosslinkRecord]:
    """Rewire endpoints while preserving type, chain pair, and approximate span."""

    rng = np.random.default_rng(seed)
    by_chain_and_name, by_chain = _candidate_pools(
        atom_records,
        chain_ids,
        backbone_atom,
    )
    randomized_by_index: dict[int, CrosslinkRecord] = {}
    disulfide_groups: dict[tuple[str, str], list[tuple[int, CrosslinkRecord]]] = {}
    for index, record in enumerate(crosslinks):
        chain_a, chain_b = record.chains
        if record.kind == "disulfide" and chain_a == chain_b:
            disulfide_groups.setdefault((chain_a, chain_b), []).append((index, record))
    for indexed_records in disulfide_groups.values():
        records = [record for _, record in indexed_records]
        matching = _randomized_intrachain_disulfide_matching(records, rng)
        if matching is None:
            continue
        for (index, record), (endpoint_a, endpoint_b) in zip(
            indexed_records, matching
        ):
            randomized_by_index[index] = CrosslinkRecord(
                crosslink_id=f"null:{replicate}:{index}:{record.crosslink_id}",
                kind=record.kind,
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                source_record="null_disulfide_perfect_matching",
                distance=None,
                metadata={"source_crosslink_id": record.crosslink_id, "seed": seed},
            )

    randomized: list[CrosslinkRecord] = []
    occupied: set[tuple[str, tuple[tuple[str, str, str, str], ...]]] = set()
    for index, record in enumerate(crosslinks):
        if index in randomized_by_index:
            candidate_record = randomized_by_index[index]
            occupied.add(
                (candidate_record.kind, candidate_record.canonical_endpoint_key)
            )
            randomized.append(candidate_record)
            continue
        for _ in range(100):
            candidate_a, candidate_b = _randomized_pair(
                record,
                by_chain_and_name,
                by_chain,
                rng,
                preserve_residue_names=preserve_residue_names,
            )
            candidate_record = CrosslinkRecord(
                crosslink_id=f"null:{replicate}:{index}:{record.crosslink_id}",
                kind=record.kind,
                endpoint_a=candidate_a,
                endpoint_b=candidate_b,
                source_record="null_rewire",
                distance=None,
                metadata={"source_crosslink_id": record.crosslink_id, "seed": seed},
            )
            key = (candidate_record.kind, candidate_record.canonical_endpoint_key)
            if key not in occupied:
                break
        occupied.add(key)
        randomized.append(candidate_record)
    return randomized


def generate_null_graphs(
    protein: CrosslinkedProteinInputResult,
    *,
    replicates: int,
    seed: int = 0,
    preserve_residue_names: bool = True,
    embedding_mode: str = "coordinate_preserving",
) -> list[NullGraphRecord]:
    """Build a reproducible ensemble of rewired embedded protein graphs."""

    if replicates < 0:
        raise ValueError("replicates must be non-negative")
    if embedding_mode not in {"coordinate_preserving", "canonical_low_crossing"}:
        raise ValueError(
            "embedding_mode must be 'coordinate_preserving' or 'canonical_low_crossing'"
        )
    seed_sequence = np.random.SeedSequence(seed)
    child_sequences = seed_sequence.spawn(replicates)
    output: list[NullGraphRecord] = []
    for replicate, child in enumerate(child_sequences):
        child_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        randomized = randomize_crosslinks(
            protein.crosslinks,
            protein.atom_records,
            chain_ids=protein.chain_ids,
            backbone_atom=protein.backbone_atom,
            seed=child_seed,
            replicate=replicate,
            preserve_residue_names=preserve_residue_names,
        )
        graph, included, issues = build_crosslinked_protein_graph(
            protein.atom_records,
            randomized,
            pdb_id=protein.pdb_id,
            source_format=protein.source_format,
            chain_ids=protein.chain_ids,
            backbone_atom=protein.backbone_atom,
        )
        graph.graph.update(
            input_id=f"{protein.graph.graph['input_id']}_null_{replicate}",
            null_model="crosslink_rewire",
            null_replicate=replicate,
            null_seed=child_seed,
            null_embedding_mode=embedding_mode,
        )
        if embedding_mode == "canonical_low_crossing":
            graph = canonicalize_null_embedding(graph, seed=child_seed)
        output.append(NullGraphRecord(replicate, child_seed, graph, included, issues))
    return output


def generate_unique_disulfide_null_graphs(
    protein: CrosslinkedProteinInputResult,
    *,
    max_nulls: int | None = None,
    seed: int = 0,
    embedding_mode: str = "coordinate_preserving",
    max_enumerated_matchings: int = 100_000,
) -> list[NullGraphRecord]:
    """Build unique span-matched disulfide rewires without pseudoreplication.

    The current exact design intentionally accepts one intrachain disulfide
    group and no other crosslink chemistry.  Every eligible non-native perfect
    matching is used when it fits under ``max_nulls``; otherwise a seeded
    sample is drawn without replacement.
    """

    if max_nulls is not None and max_nulls < 0:
        raise ValueError("max_nulls must be non-negative")
    if embedding_mode not in {"coordinate_preserving", "canonical_low_crossing"}:
        raise ValueError(
            "embedding_mode must be 'coordinate_preserving' or "
            "'canonical_low_crossing'"
        )
    records = list(protein.crosslinks)
    if not records:
        return []
    chains = {record.chains for record in records}
    if (
        any(record.kind != "disulfide" for record in records)
        or len(chains) != 1
        or any(first != second for first, second in chains)
    ):
        raise ValueError(
            "unique_disulfide_matchings requires one intrachain disulfide-only group"
        )
    eligible = _eligible_intrachain_disulfide_matchings(
        records,
        max_enumerated_matchings=max_enumerated_matchings,
    )
    ensemble_size = len(eligible)
    selected = list(eligible)
    if max_nulls is not None and len(selected) > max_nulls:
        rng = np.random.default_rng(seed)
        indices = sorted(
            int(index)
            for index in rng.choice(len(selected), size=max_nulls, replace=False)
        )
        selected = [selected[index] for index in indices]
    exhaustive = len(selected) == ensemble_size
    output = []
    for replicate, matching in enumerate(selected):
        rewired = [
            CrosslinkRecord(
                crosslink_id=f"null:{replicate}:{index}:{record.crosslink_id}",
                kind="disulfide",
                endpoint_a=endpoint_a,
                endpoint_b=endpoint_b,
                source_record="exact_unique_disulfide_perfect_matching",
                distance=None,
                metadata={
                    "source_crosslink_id": record.crosslink_id,
                    "ensemble_index": replicate,
                },
            )
            for index, (record, (endpoint_a, endpoint_b)) in enumerate(
                zip(records, matching)
            )
        ]
        graph, included, issues = build_crosslinked_protein_graph(
            protein.atom_records,
            rewired,
            pdb_id=protein.pdb_id,
            source_format=protein.source_format,
            chain_ids=protein.chain_ids,
            backbone_atom=protein.backbone_atom,
        )
        child_seed = int(np.random.SeedSequence([seed, replicate]).generate_state(1)[0])
        graph.graph.update(
            input_id=f"{protein.graph.graph['input_id']}_null_{replicate}",
            null_model="exact_unique_disulfide_endpoint_matching",
            null_replicate=replicate,
            null_seed=child_seed,
            null_embedding_mode=embedding_mode,
            null_ensemble_size=ensemble_size,
            null_ensemble_selected_count=len(selected),
            null_ensemble_exhaustive=exhaustive,
            null_sampling_without_replacement=True,
        )
        if embedding_mode == "canonical_low_crossing":
            graph = canonicalize_null_embedding(graph, seed=child_seed)
        output.append(
            NullGraphRecord(replicate, child_seed, graph, included, issues)
        )
    return output


def compare_robustness_to_null(
    natural: ProteinTopologyAnalysis,
    null_analyses: Iterable[ProteinTopologyAnalysis],
) -> NullRobustnessComparison:
    """Compare natural R1 with a finite null ensemble using empirical tails."""

    if natural.robustness_r1 is None:
        raise ValueError("Natural analysis has no robustness_r1 value")
    values = np.asarray(
        [
            analysis.robustness_r1
            for analysis in null_analyses
            if analysis.robustness_r1 is not None
        ],
        dtype=float,
    )
    if len(values) == 0:
        raise ValueError("Null analyses contain no robustness_r1 values")
    natural_value = float(natural.robustness_r1)
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    z_score = (
        (natural_value - mean) / standard_deviation
        if standard_deviation > 0.0
        else None
    )
    return NullRobustnessComparison(
        natural_robustness_r1=natural_value,
        null_values=tuple(float(value) for value in values),
        null_mean=mean,
        null_standard_deviation=standard_deviation,
        z_score=z_score,
        empirical_p_greater_equal=float(
            (1 + np.sum(values >= natural_value)) / (len(values) + 1)
        ),
        empirical_p_less_equal=float(
            (1 + np.sum(values <= natural_value)) / (len(values) + 1)
        ),
    )


def crosslink_content_signature(graph: nx.MultiGraph) -> str:
    """Return a stable count signature for crosslink chemistry and chain scope."""

    counts: Counter[tuple[str, str]] = Counter()
    for u, v, data in graph.edges(data=True):
        if data.get("edge_kind") != "crosslink":
            continue
        chain_u = str(graph.nodes[u].get("chain_id", ""))
        chain_v = str(graph.nodes[v].get("chain_id", ""))
        scope = "intra_chain" if chain_u == chain_v else "inter_chain"
        counts[(str(data.get("crosslink_type", "other")), scope)] += 1
    payload = [(*key, count) for key, count in sorted(counts.items())]
    return json.dumps(payload, separators=(",", ":"))


def crosslink_motif_signature(graph: nx.MultiGraph) -> str:
    """Compatibility alias for :func:`crosslink_content_signature`.

    Chemistry/scope counts are not a decomposition into local topological
    motifs such as knots, lassos, theta curves, or handcuff graphs.
    """

    return crosslink_content_signature(graph)
