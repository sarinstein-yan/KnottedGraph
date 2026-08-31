"""Repulsor preprocessing with explicit topology-fingerprint validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from knotted_graph.layout.repulsive import (
    DecimationOptions,
    DriverConfig,
    ResamplingOptions,
    SolverOptions,
    relax_spatial_graph,
)
from knotted_graph.layout.repulsive.driver import effective_repulsor_root

from .fingerprint import FingerprintComputer
from .graph import extract_crosslink_core
from .models import FingerprintRecord, ProteinTopologyAnalysis
from .perturbation import Fingerprinter, analyze_crosslink_perturbations


@dataclass(frozen=True)
class RepulsorAvailability:
    available: bool
    repulsor_root: Path
    header_exists: bool
    driver_source_exists: bool
    driver_binary_exists: bool
    message: str


@dataclass
class RepulsionTopologyResult:
    status: str
    topology_preserved: bool | None
    initial_fingerprint: FingerprintRecord
    relaxed_fingerprint: FingerprintRecord | None
    analysis: ProteinTopologyAnalysis | None
    workspace: Path
    layout_metadata: dict[str, Any] | None = None
    relaxed_graph: nx.MultiGraph | None = field(default=None, repr=False, compare=False)
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "topology_preserved": self.topology_preserved,
            "initial_fingerprint": self.initial_fingerprint.to_dict(),
            "relaxed_fingerprint": (
                self.relaxed_fingerprint.to_dict() if self.relaxed_fingerprint else None
            ),
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "workspace": str(self.workspace),
            "layout_metadata": self.layout_metadata,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def check_repulsor_availability(
    driver_config: DriverConfig | None = None,
) -> RepulsorAvailability:
    """Check source/header presence without compiling or changing the system."""

    config = driver_config or DriverConfig()
    root = effective_repulsor_root(config)
    header_exists = (root / "Repulsor.hpp").exists()
    source_exists = Path(config.driver_source).exists()
    binary_exists = Path(config.driver_binary).exists()
    available = header_exists and source_exists
    if available:
        message = "Repulsor sources are available; the driver will build lazily."
    else:
        message = (
            "Repulsor sources are unavailable. Run scripts/bootstrap_repulsion.py "
            "or set REPULSOR_ROOT."
        )
    return RepulsorAvailability(
        available=available,
        repulsor_root=root,
        header_exists=header_exists,
        driver_source_exists=source_exists,
        driver_binary_exists=binary_exists,
        message=message,
    )


def relax_and_analyze_crosslinks(
    graph: nx.MultiGraph,
    workspace: str | Path,
    *,
    fingerprinter: Fingerprinter | None = None,
    solver_options: SolverOptions | None = None,
    decimation_options: DecimationOptions | None = None,
    driver_config: DriverConfig | None = None,
    resampling_options: ResamplingOptions | None = None,
    force_build: bool = False,
    keep_workspace: bool = True,
    save_steps: bool = False,
    pin_graph_nodes: bool = False,
    pin_node_collar_points: int = 0,
    simplify_after_layout: bool = True,
    verify_topology: bool = False,
    require_topology_preserved: bool = True,
    allow_certificate_only: bool = False,
    include_pairs: bool = True,
    enumerate_all_subsets: bool = False,
    max_exact_crosslinks: int = 12,
) -> RepulsionTopologyResult:
    """Relax once, verify the baseline fingerprint, then scan relaxed deletions."""

    output_workspace = Path(workspace).resolve()
    computer = fingerprinter or FingerprintComputer(
        output_workspace / "fingerprint_cache"
    )
    initial_core = extract_crosslink_core(graph)
    initial = computer.compute(initial_core, removed_crosslink_ids=())
    try:
        layout = relax_spatial_graph(
            initial_core,
            output_workspace,
            solver_options=solver_options,
            decimation_options=(
                decimation_options
                if decimation_options is not None
                else DecimationOptions(min_points_per_edge=3)
            ),
            driver_config=driver_config,
            force_build=force_build,
            keep_workspace=keep_workspace,
            save_steps=save_steps,
            pin_graph_nodes=pin_graph_nodes,
            pin_node_collar_points=pin_node_collar_points,
            simplify_after_layout=simplify_after_layout,
            resampling_options=resampling_options,
            verify_topology=verify_topology,
        )
    except Exception as exc:
        return RepulsionTopologyResult(
            status="layout_error",
            topology_preserved=None,
            initial_fingerprint=initial,
            relaxed_fingerprint=None,
            analysis=None,
            workspace=output_workspace,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    relaxed_graph = layout.graph
    relaxed_graph.graph.update(
        repulsor_preprocessed=True,
        repulsor_workspace=str(output_workspace),
    )
    relaxed_core = extract_crosslink_core(relaxed_graph)
    relaxed = computer.compute(
        relaxed_core,
        removed_crosslink_ids=(),
        metadata={"stage": "post_repulsor"},
    )
    if not relaxed.success:
        return RepulsionTopologyResult(
            status="relaxed_fingerprint_error",
            topology_preserved=None,
            initial_fingerprint=initial,
            relaxed_fingerprint=relaxed,
            analysis=None,
            workspace=output_workspace,
            layout_metadata=layout.metadata,
            relaxed_graph=relaxed_graph,
            error_type=relaxed.error_type,
            error_message=relaxed.error_message,
        )
    if initial.success:
        topology_preserved = initial.same_fingerprint(relaxed)
        validation_mode = "fingerprint"
        if require_topology_preserved and not topology_preserved:
            return RepulsionTopologyResult(
                status="topology_mismatch",
                topology_preserved=False,
                initial_fingerprint=initial,
                relaxed_fingerprint=relaxed,
                analysis=None,
                workspace=output_workspace,
                layout_metadata=layout.metadata,
                relaxed_graph=relaxed_graph,
                error_type="TopologyMismatch",
                error_message=(
                    "Repulsor output has a different canonical Yamada fingerprint; "
                    "crosslink perturbations were not evaluated."
                ),
            )
    else:
        topology_preserved = None
        validation_mode = "repulsor_safe_step_certificate"
        certificate_valid = bool(
            (layout.metadata.get("certificate") or {}).get("valid", False)
        )
        if not allow_certificate_only or not certificate_valid:
            reason = (
                "certificate-only validation is disabled"
                if not allow_certificate_only
                else "Repulsor safe-step certificate is not valid"
            )
            return RepulsionTopologyResult(
                status="initial_fingerprint_error",
                topology_preserved=None,
                initial_fingerprint=initial,
                relaxed_fingerprint=relaxed,
                analysis=None,
                workspace=output_workspace,
                layout_metadata=layout.metadata,
                relaxed_graph=relaxed_graph,
                error_type=initial.error_type,
                error_message=f"{initial.error_message}; {reason}",
            )
    analysis = analyze_crosslink_perturbations(
        relaxed_graph,
        fingerprinter=computer,
        include_pairs=include_pairs,
        enumerate_all_subsets=enumerate_all_subsets,
        max_exact_crosslinks=max_exact_crosslinks,
    )
    analysis.metadata["repulsor"] = {
        "workspace": str(output_workspace),
        "topology_preserved": topology_preserved,
        "validation_mode": validation_mode,
        "solver_options": asdict(solver_options or SolverOptions()),
    }
    return RepulsionTopologyResult(
        status=(
            "certificate_only"
            if validation_mode == "repulsor_safe_step_certificate"
            else "ok"
            if topology_preserved
            else "topology_mismatch_allowed"
        ),
        topology_preserved=topology_preserved,
        initial_fingerprint=initial,
        relaxed_fingerprint=relaxed,
        analysis=analysis,
        workspace=output_workspace,
        layout_metadata=layout.metadata,
        relaxed_graph=relaxed_graph,
    )
