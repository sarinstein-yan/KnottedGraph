# Hamiltonian Yamada Phase Maps

This advanced workflow samples a two-parameter family of in-memory Hamiltonian
or Bloch-vector models and records the topology extracted at each grid cell.
It is not a Hamiltonian-file parser and it is not a proof of a continuum phase
boundary.

<div class="kg-hero">
  <p class="kg-lead">Use the interactive result to select a transition and a phase region, then inspect a representative exceptional surface and its simplified spatial-graph skeleton. Use the notebook when you need to regenerate the grid, caches, audits, or figures.</p>
  <div class="kg-link-row">
    <a href="../demos/hamiltonian_yamada_phase_map.html">Open the interactive result full screen</a>
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/applications/06_hamiltonian_yamada_phase_maps.ipynb">Open the reproduction notebook</a>
    <a href="../api/applications.html">Phase-map API</a>
  </div>
</div>

## Interactive result

<iframe
  class="kg-interactive-demo"
  src="../demos/hamiltonian_yamada_phase_map.html"
  title="Interactive Hamiltonian Yamada phase map and representative geometry"
  loading="lazy"
  sandbox="allow-scripts allow-same-origin"
></iframe>

<p class="kg-caption">Choose a classification mode and transition above the phase map. Clicking a stable region updates the representative 3-D exceptional surface, skeleton, graph statistics, and polynomial. The embedded artifact loads Plotly from a pinned CDN URL; use the full-screen link if an iframe is blocked.</p>

## Data flow

For a sampled cell $(\lambda,\Gamma)$, the notebook follows

$$
H(\mathbf{k};\lambda,\Gamma)
\longrightarrow \text{filled exceptional region}
\longrightarrow \text{skeleton}
\longrightarrow G\subset\mathbb{R}^3
\longrightarrow \Upsilon(G;Y).
$$

The reusable `make_yamada_phase_map(...)` API stores one record per cell,
including graph size, connected components, cycle rank, polynomial or error,
and a phase signature. The notebook adds row caches, connected-region
stabilization, endpoint checks, classic phase labels, and a second view that
groups regions up to selected contraction moves.

## Interpretation boundaries

- A colored cell represents a finite-grid computation, not an analytically
  exact phase boundary.
- A failed extraction remains an error record; it must not be relabeled as a
  zero invariant.
- The displayed surface gives physical/geometric context. The black skeleton
  is the graph used for topological analysis.
- “Up to contraction moves” is a stated equivalence convention and should not
  be confused with literal equality of the classic Yamada signatures.
- Resolution, sampling-window contact, endpoint behavior, and representative
  geometry should be checked before a region is interpreted scientifically.

The full notebook is resource-heavy and is statically checked in normal pull
requests rather than run automatically as a beginner tutorial.
