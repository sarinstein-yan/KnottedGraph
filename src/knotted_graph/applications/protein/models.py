"""Result models for protein crosslink topology analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FingerprintRecord:
    """Canonical Yamada fingerprint plus projection and runtime provenance."""

    cache_key: str
    embedding_hash: str
    status: str
    polynomial: str | None
    canonical_terms: tuple[tuple[int, str], ...]
    fingerprint_id: str | None
    pd_code: str | None
    rotation_angles: tuple[float, float, float] | None
    rotation_order: str
    crossing_count: int | None
    runtime_seconds: float
    removed_crosslink_ids: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None
    from_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def success(self) -> bool:
        return self.status == "ok"

    def same_fingerprint(self, other: "FingerprintRecord") -> bool:
        return (
            self.success
            and other.success
            and self.canonical_terms == other.canonical_terms
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, from_cache: bool = False
    ) -> "FingerprintRecord":
        payload = dict(data)
        payload["canonical_terms"] = tuple(
            (int(exponent), str(coefficient))
            for exponent, coefficient in payload.get("canonical_terms", [])
        )
        angles = payload.get("rotation_angles")
        payload["rotation_angles"] = tuple(angles) if angles is not None else None
        payload["removed_crosslink_ids"] = tuple(
            payload.get("removed_crosslink_ids", [])
        )
        payload["from_cache"] = from_cache
        return cls(**payload)


@dataclass(frozen=True)
class EdgeImpactRecord:
    """Fingerprint response to deleting one crosslink edge."""

    crosslink_id: str
    crosslink_type: str
    endpoint_a: str
    endpoint_b: str
    changed: bool | None
    fingerprint: FingerprintRecord

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint.to_dict()
        return payload


@dataclass(frozen=True)
class PairImpactRecord:
    """Fingerprint response and cooperativity for one deleted edge pair."""

    crosslink_i: str
    crosslink_j: str
    changed: bool | None
    cooperative: bool | None
    synergy_score: int | None
    fingerprint: FingerprintRecord

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint.to_dict()
        return payload


@dataclass(frozen=True)
class SubsetFingerprintRecord:
    """Fingerprint response to deleting an arbitrary crosslink subset."""

    removed_crosslink_ids: tuple[str, ...]
    changed: bool | None
    fingerprint: FingerprintRecord

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint.to_dict()
        return payload


@dataclass
class ProteinTopologyAnalysis:
    """Complete edge, pair, and optional all-subset analysis for one graph."""

    input_id: str
    pdb_id: str | None
    chain_ids: tuple[str, ...]
    crosslink_ids: tuple[str, ...]
    baseline: FingerprintRecord
    singles: list[EdgeImpactRecord]
    pairs: list[PairImpactRecord]
    subsets: list[SubsetFingerprintRecord]
    topological_fraction: float | None
    robustness_r1: float | None
    successful_single_count: int
    failed_single_count: int
    cooperative_pair_count: int
    minimal_changed_subsets: list[tuple[str, ...]]
    minimum_cardinality_subsets: list[tuple[str, ...]]
    minimum_generating_crosslink_count: int | None
    minimum_generating_crosslink_sets: list[tuple[str, ...]]
    minimum_generating_crosslink_status: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "pdb_id": self.pdb_id,
            "chain_ids": self.chain_ids,
            "crosslink_ids": self.crosslink_ids,
            "baseline": self.baseline.to_dict(),
            "singles": [record.to_dict() for record in self.singles],
            "pairs": [record.to_dict() for record in self.pairs],
            "subsets": [record.to_dict() for record in self.subsets],
            "topological_fraction": self.topological_fraction,
            "robustness_r1": self.robustness_r1,
            "successful_single_count": self.successful_single_count,
            "failed_single_count": self.failed_single_count,
            "cooperative_pair_count": self.cooperative_pair_count,
            "minimal_changed_subsets": self.minimal_changed_subsets,
            "minimum_cardinality_subsets": self.minimum_cardinality_subsets,
            "minimum_generating_crosslink_count": (
                self.minimum_generating_crosslink_count
            ),
            "minimum_generating_crosslink_sets": (
                self.minimum_generating_crosslink_sets
            ),
            "minimum_generating_crosslink_status": (
                self.minimum_generating_crosslink_status
            ),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MinimumGeneratingSetSearch:
    """Rigorous retained-edge search for a minimum fingerprint generator."""

    status: str
    crosslink_ids: tuple[str, ...]
    maximum_retained_size_requested: int
    maximum_retained_size_evaluated: int
    proven_minimum_count: int | None
    proven_minimum_sets: tuple[tuple[str, ...], ...]
    proven_lower_bound: int
    successful_state_count: int
    failed_state_count: int
    states: tuple[SubsetFingerprintRecord, ...]
    baseline: FingerprintRecord
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "crosslink_ids": list(self.crosslink_ids),
            "maximum_retained_size_requested": (
                self.maximum_retained_size_requested
            ),
            "maximum_retained_size_evaluated": (
                self.maximum_retained_size_evaluated
            ),
            "proven_minimum_count": self.proven_minimum_count,
            "proven_minimum_sets": [
                list(subset) for subset in self.proven_minimum_sets
            ],
            "proven_lower_bound": self.proven_lower_bound,
            "successful_state_count": self.successful_state_count,
            "failed_state_count": self.failed_state_count,
            "states": [record.to_dict() for record in self.states],
            "baseline": self.baseline.to_dict(),
            "issues": list(self.issues),
        }
