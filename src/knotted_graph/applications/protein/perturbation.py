"""Crosslink deletion scans and cooperative-edge discovery."""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

import networkx as nx

from .fingerprint import FingerprintComputer
from .graph import crosslink_edges, extract_crosslink_core, remove_crosslinks
from .models import (
    EdgeImpactRecord,
    FingerprintRecord,
    MinimumGeneratingSetSearch,
    PairImpactRecord,
    ProteinTopologyAnalysis,
    SubsetFingerprintRecord,
)


class Fingerprinter(Protocol):
    def compute(
        self,
        graph: nx.MultiGraph,
        *,
        removed_crosslink_ids: tuple[str, ...] = (),
        metadata: dict | None = None,
    ) -> FingerprintRecord: ...


def _changed(
    baseline: FingerprintRecord,
    perturbed: FingerprintRecord,
) -> bool | None:
    if not baseline.success or not perturbed.success:
        return None
    return not baseline.same_fingerprint(perturbed)


def _minimal_changed_subsets(
    records: list[SubsetFingerprintRecord],
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    changed = sorted(
        (record.removed_crosslink_ids for record in records if record.changed is True),
        key=lambda subset: (len(subset), subset),
    )
    inclusion_minimal = [
        subset
        for subset in changed
        if not any(
            set(smaller).issubset(subset)
            for smaller in changed
            if len(smaller) < len(subset)
        )
    ]
    if not changed:
        return inclusion_minimal, []
    minimum_size = len(changed[0])
    minimum_cardinality = [subset for subset in changed if len(subset) == minimum_size]
    return inclusion_minimal, minimum_cardinality


def _minimum_generating_crosslink_sets(
    crosslink_ids: tuple[str, ...],
    baseline: FingerprintRecord,
    records: list[SubsetFingerprintRecord],
    *,
    enumerate_all_subsets: bool,
) -> tuple[int | None, list[tuple[str, ...]], str]:
    """Find the smallest retained set reproducing the full fingerprint.

    A record stores deleted crosslinks, so its complementary retained set is
    the candidate generating set.  The minimum is reported only after every
    one of the ``2**m`` states has a successful exact fingerprint; otherwise a
    smaller, failed or unevaluated candidate could invalidate the minimum.
    """

    if not baseline.success:
        return None, [], "baseline_error"
    if not enumerate_all_subsets:
        return None, [], "not_enumerated"

    expected_perturbations = (1 << len(crosslink_ids)) - 1
    if len(records) != expected_perturbations or any(
        not record.fingerprint.success for record in records
    ):
        return None, [], "incomplete"

    crosslink_set = set(crosslink_ids)
    candidates = [crosslink_ids]
    candidates.extend(
        tuple(
            crosslink_id
            for crosslink_id in crosslink_ids
            if crosslink_id not in set(record.removed_crosslink_ids)
        )
        for record in records
        if record.changed is False
        and set(record.removed_crosslink_ids).issubset(crosslink_set)
    )
    minimum_count = min(len(candidate) for candidate in candidates)
    minimum_sets = sorted(
        {candidate for candidate in candidates if len(candidate) == minimum_count}
    )
    return minimum_count, minimum_sets, "ok"


def analyze_crosslink_perturbations(
    graph: nx.MultiGraph,
    *,
    fingerprinter: Fingerprinter | None = None,
    include_pairs: bool = True,
    enumerate_all_subsets: bool = False,
    max_exact_crosslinks: int = 12,
    suppress_degree_two: bool = True,
    extract_core: bool = True,
) -> ProteinTopologyAnalysis:
    """Compute baseline, single-edge, pair, and optional all-subset fingerprints."""

    if max_exact_crosslinks < 0:
        raise ValueError("max_exact_crosslinks must be non-negative")
    computer = fingerprinter or FingerprintComputer()
    analysis_graph = extract_crosslink_core(graph) if extract_core else graph
    refs = crosslink_edges(analysis_graph)
    ids = tuple(ref.crosslink_id for ref in refs)
    baseline = computer.compute(analysis_graph, removed_crosslink_ids=())

    singles: list[EdgeImpactRecord] = []
    single_changed: dict[str, bool | None] = {}
    subset_by_ids: dict[tuple[str, ...], SubsetFingerprintRecord] = {}
    for ref in refs:
        removed = (ref.crosslink_id,)
        perturbed_graph = remove_crosslinks(
            analysis_graph,
            removed,
            suppress_degree_two=suppress_degree_two,
        )
        if extract_core:
            perturbed_graph = extract_crosslink_core(perturbed_graph)
        fingerprint = computer.compute(
            perturbed_graph,
            removed_crosslink_ids=removed,
        )
        changed = _changed(baseline, fingerprint)
        single_changed[ref.crosslink_id] = changed
        singles.append(
            EdgeImpactRecord(
                crosslink_id=ref.crosslink_id,
                crosslink_type=ref.crosslink_type,
                endpoint_a=ref.endpoint_a,
                endpoint_b=ref.endpoint_b,
                changed=changed,
                fingerprint=fingerprint,
            )
        )
        subset_by_ids[removed] = SubsetFingerprintRecord(removed, changed, fingerprint)

    pairs: list[PairImpactRecord] = []
    if include_pairs:
        for crosslink_i, crosslink_j in combinations(ids, 2):
            removed = (crosslink_i, crosslink_j)
            perturbed_graph = remove_crosslinks(
                analysis_graph,
                removed,
                suppress_degree_two=suppress_degree_two,
            )
            if extract_core:
                perturbed_graph = extract_crosslink_core(perturbed_graph)
            fingerprint = computer.compute(
                perturbed_graph,
                removed_crosslink_ids=removed,
            )
            changed = _changed(baseline, fingerprint)
            single_i = single_changed[crosslink_i]
            single_j = single_changed[crosslink_j]
            if changed is None or single_i is None or single_j is None:
                cooperative = None
                synergy_score = None
            else:
                cooperative = changed and not single_i and not single_j
                synergy_score = int(changed) - max(int(single_i), int(single_j))
            pairs.append(
                PairImpactRecord(
                    crosslink_i=crosslink_i,
                    crosslink_j=crosslink_j,
                    changed=changed,
                    cooperative=cooperative,
                    synergy_score=synergy_score,
                    fingerprint=fingerprint,
                )
            )
            subset_by_ids[removed] = SubsetFingerprintRecord(
                removed, changed, fingerprint
            )

    if enumerate_all_subsets:
        if len(ids) > max_exact_crosslinks:
            raise ValueError(
                f"Exact subset enumeration requested for {len(ids)} crosslinks, "
                f"exceeding max_exact_crosslinks={max_exact_crosslinks}."
            )
        for size in range(1, len(ids) + 1):
            for subset in combinations(ids, size):
                if subset in subset_by_ids:
                    continue
                perturbed_graph = remove_crosslinks(
                    analysis_graph,
                    subset,
                    suppress_degree_two=suppress_degree_two,
                )
                if extract_core:
                    perturbed_graph = extract_crosslink_core(perturbed_graph)
                fingerprint = computer.compute(
                    perturbed_graph,
                    removed_crosslink_ids=subset,
                )
                subset_by_ids[subset] = SubsetFingerprintRecord(
                    subset,
                    _changed(baseline, fingerprint),
                    fingerprint,
                )

    subsets = sorted(
        subset_by_ids.values(),
        key=lambda record: (
            len(record.removed_crosslink_ids),
            record.removed_crosslink_ids,
        ),
    )
    minimal, minimum_cardinality = _minimal_changed_subsets(subsets)
    (
        minimum_generating_count,
        minimum_generating_sets,
        minimum_generating_status,
    ) = _minimum_generating_crosslink_sets(
        ids,
        baseline,
        subsets,
        enumerate_all_subsets=enumerate_all_subsets,
    )
    successful = sum(record.changed is not None for record in singles)
    changed_count = sum(record.changed is True for record in singles)
    topological_fraction = changed_count / successful if successful else None
    robustness = (
        1.0 - topological_fraction if topological_fraction is not None else None
    )
    return ProteinTopologyAnalysis(
        input_id=str(analysis_graph.graph.get("input_id", "protein_crosslinks")),
        pdb_id=analysis_graph.graph.get("pdb_id"),
        chain_ids=tuple(analysis_graph.graph.get("chain_ids", ())),
        crosslink_ids=ids,
        baseline=baseline,
        singles=singles,
        pairs=pairs,
        subsets=subsets,
        topological_fraction=topological_fraction,
        robustness_r1=robustness,
        successful_single_count=successful,
        failed_single_count=len(singles) - successful,
        cooperative_pair_count=sum(record.cooperative is True for record in pairs),
        minimal_changed_subsets=minimal,
        minimum_cardinality_subsets=minimum_cardinality,
        minimum_generating_crosslink_count=minimum_generating_count,
        minimum_generating_crosslink_sets=minimum_generating_sets,
        minimum_generating_crosslink_status=minimum_generating_status,
        metadata={
            "include_pairs": include_pairs,
            "enumerate_all_subsets": enumerate_all_subsets,
            "suppress_degree_two": suppress_degree_two,
            "extract_core": extract_core,
            "core_excluded_crosslink_ids": analysis_graph.graph.get(
                "core_excluded_crosslink_ids",
                (),
            ),
        },
    )


def search_minimum_generating_crosslink_sets(
    graph: nx.MultiGraph,
    *,
    fingerprinter: Fingerprinter | None = None,
    max_retained_crosslinks: int,
    suppress_degree_two: bool = True,
    extract_core: bool = True,
) -> MinimumGeneratingSetSearch:
    """Search retained-edge levels in increasing order with rigorous bounds.

    A successful first matching level proves the exact minimum because every
    smaller retained set has already been evaluated.  If the requested bound
    is exhausted, ``proven_lower_bound`` records the first untested size.
    Fingerprint failures make the result explicitly incomplete.
    """

    if max_retained_crosslinks < 0:
        raise ValueError("max_retained_crosslinks must be non-negative")
    computer = fingerprinter or FingerprintComputer()
    analysis_graph = extract_crosslink_core(graph) if extract_core else graph
    ids = tuple(ref.crosslink_id for ref in crosslink_edges(analysis_graph))
    requested = min(max_retained_crosslinks, len(ids))
    baseline = computer.compute(analysis_graph, removed_crosslink_ids=())
    if not baseline.success:
        return MinimumGeneratingSetSearch(
            status="baseline_error",
            crosslink_ids=ids,
            maximum_retained_size_requested=requested,
            maximum_retained_size_evaluated=0,
            proven_minimum_count=None,
            proven_minimum_sets=(),
            proven_lower_bound=0,
            successful_state_count=0,
            failed_state_count=1,
            states=(),
            baseline=baseline,
            issues=(f"{baseline.error_type}: {baseline.error_message}",),
        )

    records: list[SubsetFingerprintRecord] = []
    issues = []
    maximum_evaluated = -1
    for retained_size in range(requested + 1):
        level_records = []
        for retained in combinations(ids, retained_size):
            retained_set = set(retained)
            removed = tuple(
                crosslink_id
                for crosslink_id in ids
                if crosslink_id not in retained_set
            )
            if not removed:
                fingerprint = baseline
            else:
                state_graph = remove_crosslinks(
                    analysis_graph,
                    removed,
                    suppress_degree_two=suppress_degree_two,
                )
                if extract_core:
                    state_graph = extract_crosslink_core(state_graph)
                fingerprint = computer.compute(
                    state_graph,
                    removed_crosslink_ids=removed,
                )
            record = SubsetFingerprintRecord(
                removed_crosslink_ids=removed,
                changed=_changed(baseline, fingerprint),
                fingerprint=fingerprint,
            )
            level_records.append(record)
            records.append(record)
            if not fingerprint.success:
                issues.append(
                    f"retained={retained}: {fingerprint.error_type}: "
                    f"{fingerprint.error_message}"
                )
        maximum_evaluated = retained_size
        if any(not record.fingerprint.success for record in records):
            break
        matches = [
            tuple(
                crosslink_id
                for crosslink_id in ids
                if crosslink_id not in set(record.removed_crosslink_ids)
            )
            for record in level_records
            if record.changed is False
        ]
        if matches:
            return MinimumGeneratingSetSearch(
                status="proven",
                crosslink_ids=ids,
                maximum_retained_size_requested=requested,
                maximum_retained_size_evaluated=maximum_evaluated,
                proven_minimum_count=retained_size,
                proven_minimum_sets=tuple(matches),
                proven_lower_bound=retained_size,
                successful_state_count=len(records),
                failed_state_count=0,
                states=tuple(records),
                baseline=baseline,
            )

    failed_count = sum(not record.fingerprint.success for record in records)
    if failed_count:
        status = "incomplete"
        lower_bound = max(0, maximum_evaluated)
    else:
        status = "none_up_to_bound"
        lower_bound = requested + 1
    return MinimumGeneratingSetSearch(
        status=status,
        crosslink_ids=ids,
        maximum_retained_size_requested=requested,
        maximum_retained_size_evaluated=max(0, maximum_evaluated),
        proven_minimum_count=None,
        proven_minimum_sets=(),
        proven_lower_bound=lower_bound,
        successful_state_count=sum(
            record.fingerprint.success for record in records
        ),
        failed_state_count=failed_count,
        states=tuple(records),
        baseline=baseline,
        issues=tuple(issues),
    )
