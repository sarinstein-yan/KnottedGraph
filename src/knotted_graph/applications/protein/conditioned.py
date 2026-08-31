"""Abstract-connectivity-conditioned protein embedding robustness.

Raw Yamada fingerprints necessarily respond when an edge deletion changes the
abstract graph.  This module separates that unavoidable graph change from
spatial embedding complexity by comparing every observed state with a
deterministic low-crossing reference embedding of the *same* abstract graph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Protocol

import networkx as nx
import numpy as np

from .fingerprint import FingerprintComputer
from .graph import crosslink_edges, extract_crosslink_core, remove_crosslinks
from .models import FingerprintRecord, ProteinTopologyAnalysis
from .null_models import canonicalize_null_embedding
from .perturbation import analyze_crosslink_perturbations


class _Fingerprinter(Protocol):
    def compute(
        self,
        graph: nx.MultiGraph,
        *,
        removed_crosslink_ids: tuple[str, ...] = (),
        metadata: dict | None = None,
    ) -> FingerprintRecord: ...


@dataclass(frozen=True)
class ConditionedEmbeddingState:
    """Observed/reference comparison for one abstract graph state."""

    removed_crosslink_ids: tuple[str, ...]
    observed_status: str
    observed_fingerprint_id: str | None
    reference_status: str
    reference_fingerprint_id: str | None
    embedding_nontrivial: bool | None
    reference_layout_algorithm: str | None
    reference_minimum_clearance: float | None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConditionedEdgeImpact:
    """Whether deleting one edge removes baseline excess embedding topology."""

    crosslink_id: str
    deleted_state_nontrivial: bool | None
    information_carrying: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConditionedPairImpact:
    """Conditioned response and cooperativity of one deleted edge pair."""

    crosslink_i: str
    crosslink_j: str
    deleted_state_nontrivial: bool | None
    information_carrying: bool | None
    cooperative: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ConditionedSubsetImpact:
    """Conditioned topology loss for an arbitrary deleted crosslink subset.

    ``strictly_cooperative`` is true exactly when this subset removes the
    baseline excess embedding topology and no non-empty proper subset does.
    For subsets of size two this is the pairwise cooperativity definition.
    """

    removed_crosslink_ids: tuple[str, ...]
    order: int
    deleted_state_nontrivial: bool | None
    information_carrying: bool | None
    strictly_cooperative: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AbstractConditionedRobustness:
    """Single-edge robustness after conditioning on abstract connectivity."""

    status: str
    reference_mode: str
    reference_seed: int
    baseline_embedding_nontrivial: bool | None
    conditioned_state_robustness_r1: float | None
    entanglement_retention_r1: float | None
    topology_carrying_edge_count: int
    topology_carrying_edge_fraction: float | None
    successful_single_count: int
    failed_single_count: int
    cooperative_pair_count: int
    successful_pair_count: int
    failed_pair_count: int
    maximum_subset_order_evaluated: int
    successful_subset_count: int
    failed_subset_count: int
    strictly_cooperative_subset_count: int
    minimum_information_carrying_subset_size: int | None
    minimum_information_carrying_subsets: tuple[tuple[str, ...], ...]
    baseline: ConditionedEmbeddingState
    singles: tuple[ConditionedEdgeImpact, ...]
    pairs: tuple[ConditionedPairImpact, ...]
    subsets: tuple[ConditionedSubsetImpact, ...]
    states: tuple[ConditionedEmbeddingState, ...]
    issues: tuple[str, ...] = ()

    @property
    def has_topology_carrying_edge(self) -> bool | None:
        if self.baseline_embedding_nontrivial is None:
            return None
        return self.topology_carrying_edge_count > 0

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reference_mode": self.reference_mode,
            "reference_seed": self.reference_seed,
            "baseline_embedding_nontrivial": self.baseline_embedding_nontrivial,
            "conditioned_state_robustness_r1": (
                self.conditioned_state_robustness_r1
            ),
            "entanglement_retention_r1": self.entanglement_retention_r1,
            "topology_carrying_edge_count": self.topology_carrying_edge_count,
            "topology_carrying_edge_fraction": (
                self.topology_carrying_edge_fraction
            ),
            "has_topology_carrying_edge": self.has_topology_carrying_edge,
            "successful_single_count": self.successful_single_count,
            "failed_single_count": self.failed_single_count,
            "cooperative_pair_count": self.cooperative_pair_count,
            "successful_pair_count": self.successful_pair_count,
            "failed_pair_count": self.failed_pair_count,
            "maximum_subset_order_evaluated": (
                self.maximum_subset_order_evaluated
            ),
            "successful_subset_count": self.successful_subset_count,
            "failed_subset_count": self.failed_subset_count,
            "strictly_cooperative_subset_count": (
                self.strictly_cooperative_subset_count
            ),
            "minimum_information_carrying_subset_size": (
                self.minimum_information_carrying_subset_size
            ),
            "minimum_information_carrying_subsets": [
                list(subset)
                for subset in self.minimum_information_carrying_subsets
            ],
            "baseline": self.baseline.to_dict(),
            "singles": [record.to_dict() for record in self.singles],
            "pairs": [record.to_dict() for record in self.pairs],
            "subsets": [record.to_dict() for record in self.subsets],
            "states": [record.to_dict() for record in self.states],
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class ConditionedNullComparison:
    """Finite-sample comparison of topology-carrying edge fractions."""

    natural_value: float
    null_values: tuple[float, ...]
    null_mean: float
    null_standard_deviation: float
    z_score: float | None
    empirical_p_greater_equal: float
    empirical_p_less_equal: float

    def to_dict(self) -> dict:
        return asdict(self)


def _conditioned_state(
    graph: nx.MultiGraph,
    observed: FingerprintRecord,
    *,
    fingerprinter: _Fingerprinter,
    removed_crosslink_ids: tuple[str, ...],
    reference_seed: int,
) -> ConditionedEmbeddingState:
    try:
        if graph.number_of_edges() == 0:
            reference_graph = graph.copy()
            algorithm = "empty"
            clearance = None
        else:
            reference_graph = canonicalize_null_embedding(
                graph,
                seed=reference_seed,
            )
            algorithm = str(
                reference_graph.graph.get("canonical_layout_algorithm", "unknown")
            )
            clearance_value = reference_graph.graph.get(
                "canonical_min_non_adjacent_segment_distance"
            )
            clearance = (
                float(clearance_value) if clearance_value is not None else None
            )
        reference = fingerprinter.compute(
            reference_graph,
            removed_crosslink_ids=removed_crosslink_ids,
            metadata={"fingerprint_role": "abstract_conditioned_reference"},
        )
        nontrivial = (
            not observed.same_fingerprint(reference)
            if observed.success and reference.success
            else None
        )
        error_type = reference.error_type if not reference.success else observed.error_type
        error_message = (
            reference.error_message if not reference.success else observed.error_message
        )
        return ConditionedEmbeddingState(
            removed_crosslink_ids=removed_crosslink_ids,
            observed_status=observed.status,
            observed_fingerprint_id=observed.fingerprint_id,
            reference_status=reference.status,
            reference_fingerprint_id=reference.fingerprint_id,
            embedding_nontrivial=nontrivial,
            reference_layout_algorithm=algorithm,
            reference_minimum_clearance=clearance,
            error_type=error_type,
            error_message=error_message,
        )
    except Exception as exc:
        return ConditionedEmbeddingState(
            removed_crosslink_ids=removed_crosslink_ids,
            observed_status=observed.status,
            observed_fingerprint_id=observed.fingerprint_id,
            reference_status="error",
            reference_fingerprint_id=None,
            embedding_nontrivial=None,
            reference_layout_algorithm=None,
            reference_minimum_clearance=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def analyze_abstract_conditioned_robustness(
    graph: nx.MultiGraph,
    *,
    analysis: ProteinTopologyAnalysis | None = None,
    fingerprinter: _Fingerprinter | None = None,
    reference_seed: int = 0,
    include_pairs: bool = False,
    max_subset_order: int | None = None,
) -> AbstractConditionedRobustness:
    """Compare deleted-subset embeddings to same-connectivity references.

    ``max_subset_order`` enables strict higher-order cooperativity searches.
    The legacy ``include_pairs`` flag maps to order two when an explicit order
    is not supplied.
    """

    requested_order = (
        max_subset_order
        if max_subset_order is not None
        else (2 if include_pairs else 1)
    )
    if requested_order < 1:
        raise ValueError("max_subset_order must be at least 1")

    computer = fingerprinter or FingerprintComputer()
    observed_analysis = analysis or analyze_crosslink_perturbations(
        graph,
        fingerprinter=computer,
        include_pairs=False,
        enumerate_all_subsets=False,
    )
    analysis_graph = extract_crosslink_core(graph)
    refs = crosslink_edges(analysis_graph)
    ids = tuple(ref.crosslink_id for ref in refs)
    maximum_order = min(requested_order, len(ids)) if ids else 0
    if ids != observed_analysis.crosslink_ids:
        raise ValueError(
            "analysis crosslink IDs do not match the extracted graph core: "
            f"{observed_analysis.crosslink_ids!r} != {ids!r}"
        )

    baseline = _conditioned_state(
        analysis_graph,
        observed_analysis.baseline,
        fingerprinter=computer,
        removed_crosslink_ids=(),
        reference_seed=reference_seed,
    )
    if baseline.embedding_nontrivial is None:
        issue = (
            f"{baseline.error_type}: {baseline.error_message}"
            if baseline.error_type or baseline.error_message
            else "baseline conditioned state could not be evaluated"
        )
        return AbstractConditionedRobustness(
            status="baseline_error",
            reference_mode="same_abstract_graph_canonical_low_crossing",
            reference_seed=reference_seed,
            baseline_embedding_nontrivial=None,
            conditioned_state_robustness_r1=None,
            entanglement_retention_r1=None,
            topology_carrying_edge_count=0,
            topology_carrying_edge_fraction=None,
            successful_single_count=0,
            failed_single_count=len(ids),
            cooperative_pair_count=0,
            successful_pair_count=0,
            failed_pair_count=0,
            maximum_subset_order_evaluated=0,
            successful_subset_count=0,
            failed_subset_count=0,
            strictly_cooperative_subset_count=0,
            minimum_information_carrying_subset_size=None,
            minimum_information_carrying_subsets=(),
            baseline=baseline,
            singles=(),
            pairs=(),
            subsets=(),
            states=(baseline,),
            issues=(issue,),
        )
    observed_singles = {
        record.crosslink_id: record.fingerprint for record in observed_analysis.singles
    }
    states = [baseline]
    edge_impacts = []
    for ref in refs:
        removed = (ref.crosslink_id,)
        state_graph = extract_crosslink_core(
            remove_crosslinks(analysis_graph, removed)
        )
        observed = observed_singles.get(ref.crosslink_id)
        if observed is None:
            observed = computer.compute(
                state_graph,
                removed_crosslink_ids=removed,
            )
        state = _conditioned_state(
            state_graph,
            observed,
            fingerprinter=computer,
            removed_crosslink_ids=removed,
            reference_seed=reference_seed,
        )
        states.append(state)
        if baseline.embedding_nontrivial is None or state.embedding_nontrivial is None:
            carrying = None
        else:
            carrying = baseline.embedding_nontrivial and not state.embedding_nontrivial
        edge_impacts.append(
            ConditionedEdgeImpact(
                crosslink_id=ref.crosslink_id,
                deleted_state_nontrivial=state.embedding_nontrivial,
                information_carrying=carrying,
            )
        )

    observed_by_subset: dict[tuple[str, ...], FingerprintRecord] = {
        (record.crosslink_id,): record.fingerprint
        for record in observed_analysis.singles
    }
    observed_by_subset.update(
        {
            (record.crosslink_i, record.crosslink_j): record.fingerprint
            for record in observed_analysis.pairs
        }
    )
    observed_by_subset.update(
        {
            tuple(record.removed_crosslink_ids): record.fingerprint
            for record in observed_analysis.subsets
        }
    )
    pair_impacts = []
    subset_impacts = []
    information_by_subset: dict[tuple[str, ...], bool | None] = {
        (record.crosslink_id,): record.information_carrying
        for record in edge_impacts
    }
    for order in range(2, maximum_order + 1):
        for removed in combinations(ids, order):
            state_graph = extract_crosslink_core(
                remove_crosslinks(analysis_graph, removed)
            )
            observed = observed_by_subset.get(removed)
            if observed is None:
                observed = computer.compute(
                    state_graph,
                    removed_crosslink_ids=removed,
                )
            state = _conditioned_state(
                state_graph,
                observed,
                fingerprinter=computer,
                removed_crosslink_ids=removed,
                reference_seed=reference_seed,
            )
            states.append(state)
            if (
                baseline.embedding_nontrivial is None
                or state.embedding_nontrivial is None
            ):
                carrying = None
            else:
                carrying = (
                    baseline.embedding_nontrivial
                    and not state.embedding_nontrivial
                )
            proper_impacts = [
                information_by_subset[proper]
                for proper_order in range(1, order)
                for proper in combinations(removed, proper_order)
            ]
            if carrying is not True:
                cooperative = carrying
            elif any(value is True for value in proper_impacts):
                cooperative = False
            elif any(value is None for value in proper_impacts):
                cooperative = None
            else:
                cooperative = True
            information_by_subset[removed] = carrying
            subset_impacts.append(
                ConditionedSubsetImpact(
                    removed_crosslink_ids=removed,
                    order=order,
                    deleted_state_nontrivial=state.embedding_nontrivial,
                    information_carrying=carrying,
                    strictly_cooperative=cooperative,
                )
            )
            if order == 2:
                pair_impacts.append(
                    ConditionedPairImpact(
                        crosslink_i=removed[0],
                        crosslink_j=removed[1],
                        deleted_state_nontrivial=state.embedding_nontrivial,
                        information_carrying=carrying,
                        cooperative=cooperative,
                    )
                )

    successful_states = [
        state
        for state in states[1:]
        if len(state.removed_crosslink_ids) == 1
        and state.embedding_nontrivial is not None
    ]
    successful = len(successful_states)
    failed = len(edge_impacts) - successful
    successful_pairs = sum(
        record.deleted_state_nontrivial is not None for record in pair_impacts
    )
    failed_pairs = len(pair_impacts) - successful_pairs
    successful_subsets = sum(
        record.deleted_state_nontrivial is not None for record in subset_impacts
    )
    failed_subsets = len(subset_impacts) - successful_subsets
    carrying_count = sum(record.information_carrying is True for record in edge_impacts)
    carrying_subsets = [
        subset
        for subset, carrying in information_by_subset.items()
        if carrying is True
    ]
    minimum_carrying_size = (
        min(map(len, carrying_subsets)) if carrying_subsets else None
    )
    minimum_carrying_subsets = tuple(
        subset
        for subset in carrying_subsets
        if len(subset) == minimum_carrying_size
    )
    if baseline.embedding_nontrivial is None or not successful:
        robustness = None
        carrying_fraction = None
    else:
        robustness = sum(
            state.embedding_nontrivial == baseline.embedding_nontrivial
            for state in successful_states
        ) / successful
        carrying_fraction = carrying_count / successful
    entanglement_retention = (
        sum(state.embedding_nontrivial is True for state in successful_states)
        / successful
        if baseline.embedding_nontrivial is True and successful
        else None
    )
    if baseline.embedding_nontrivial is None:
        status = "baseline_error"
    elif failed or failed_subsets:
        status = "partial"
    else:
        status = "ok"
    issues = tuple(
        f"{state.removed_crosslink_ids or ('baseline',)}: "
        f"{state.error_type}: {state.error_message}"
        for state in states
        if state.embedding_nontrivial is None
    )
    return AbstractConditionedRobustness(
        status=status,
        reference_mode="same_abstract_graph_canonical_low_crossing",
        reference_seed=reference_seed,
        baseline_embedding_nontrivial=baseline.embedding_nontrivial,
        conditioned_state_robustness_r1=robustness,
        entanglement_retention_r1=entanglement_retention,
        topology_carrying_edge_count=carrying_count,
        topology_carrying_edge_fraction=carrying_fraction,
        successful_single_count=successful,
        failed_single_count=failed,
        cooperative_pair_count=sum(
            record.cooperative is True for record in pair_impacts
        ),
        successful_pair_count=successful_pairs,
        failed_pair_count=failed_pairs,
        maximum_subset_order_evaluated=maximum_order,
        successful_subset_count=successful_subsets,
        failed_subset_count=failed_subsets,
        strictly_cooperative_subset_count=sum(
            record.strictly_cooperative is True
            for record in subset_impacts
            if record.order >= 2
        ),
        minimum_information_carrying_subset_size=minimum_carrying_size,
        minimum_information_carrying_subsets=minimum_carrying_subsets,
        baseline=baseline,
        singles=tuple(edge_impacts),
        pairs=tuple(pair_impacts),
        subsets=tuple(subset_impacts),
        states=tuple(states),
        issues=issues,
    )


def compare_conditioned_topology_to_null(
    natural: AbstractConditionedRobustness,
    null_analyses: list[AbstractConditionedRobustness],
) -> ConditionedNullComparison:
    """Compare topology-carrying edge fractions with finite-sample tails."""

    if natural.topology_carrying_edge_fraction is None:
        raise ValueError("Natural conditioned analysis has no successful edge fraction")
    values = np.asarray(
        [
            analysis.topology_carrying_edge_fraction
            for analysis in null_analyses
            if analysis.topology_carrying_edge_fraction is not None
        ],
        dtype=float,
    )
    if len(values) == 0:
        raise ValueError("Null conditioned analyses contain no successful edge fractions")
    natural_value = float(natural.topology_carrying_edge_fraction)
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return ConditionedNullComparison(
        natural_value=natural_value,
        null_values=tuple(float(value) for value in values),
        null_mean=mean,
        null_standard_deviation=standard_deviation,
        z_score=(
            (natural_value - mean) / standard_deviation
            if standard_deviation > 0.0
            else None
        ),
        empirical_p_greater_equal=float(
            (1 + np.sum(values >= natural_value)) / (len(values) + 1)
        ),
        empirical_p_less_equal=float(
            (1 + np.sum(values <= natural_value)) / (len(values) + 1)
        ),
    )
