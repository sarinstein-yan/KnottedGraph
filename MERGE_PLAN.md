# KnottedGraph TDD Merge Plan

This branch is intended to become the future public main branch. Git history can
be rewritten to make the result clean, so the merge should be organized around
small, test-backed commits rather than preserving the current branch topology.

## Target Package Identity

KnottedGraph should be treated as a general computational and pure mathematical
package for spatial graphs and their invariants.

`NodalSkeleton` is no longer the high-level package wrapper. It is a specific
physics application that motivated the repository and remains one route into the
generic spatial-graph pipeline. The public architecture should make the generic
pipeline primary:

1. Input adapters normalize external scientific and mathematical data.
2. Core spatial-graph contracts carry embedded graph data between modules.
3. Optional application workflows, including `NodalSkeleton`, produce those core
   spatial graphs.
4. Projection and PD encoding convert spatial graphs into symbolic diagrams.
5. The Yamada engine evaluates invariants through Negami and recursive routes.
6. Repulsive layout is an optional geometric simplification stage before
   projection.
7. Visualization and examples sit at the edges and should not be required for
   lightweight algebraic use.

## Desired Module Boundaries

| Layer | Intended modules | Contract |
|---|---|---|
| Public API | `knotted_graph.__init__`, docs, examples | Export stable generic functions first; expose application workflows explicitly. |
| Inputs | `knotted_graph.inputs` | Convert PDB, mmCIF, polymers, CSV graphs, meshes, and coordinate chains into core objects. |
| Core graph contract | shared utilities, `networkx.MultiGraph` conventions | Nodes have finite 3D `pos`; edges may have 3D polyline `pts`; parallel edges and edge keys are preserved. |
| Applications | `NodalSkeleton`, surface/physics helpers | Domain-specific workflows that produce core spatial graphs. |
| Projection and PD | `knotted_graph.yamada.pd_code`, `geom`, `util` | Convert embedded graphs to `PDCode`, `Vertex`, `Crossing`, and `Arc` objects. |
| Yamada engine | `knotted_graph.yamada.polynomial`, `recursive` | Shared crossing-state generation with selectable Negami or recursive graph evaluation. |
| Repulsive layout | `knotted_graph.repulsive_layout` | Optional 3D curve-network relaxation that accepts and returns the core graph contract. |
| Visualization | `vis`, plotting helpers | Optional diagnostics and presentation outputs. |

## Current Branch State

Current branch: `refactor-merge-development`.

Staged work currently includes:

- `src/knotted_graph/yamada/polynomial.py`: combined Yamada fix.
- `tests/test_yamada_combined_fix.py`: regression tests for Negami exponent and crossing-port resolution.
- `pyproject.toml` and `uv.lock`: package/environment changes.
- `getting_started.ipynb -> doc/getting_started.ipynb`: documentation move.

Known verification:

```bash
uv run --with pytest python -m pytest -q tests/test_yamada_combined_fix.py
```

Expected result:

```text
3 passed
```

## Branch Inventory

| Branch | Relevant content | Merge stance |
|---|---|---|
| `origin/main` | Math/protein artifacts, recursive Yamada files, data/notebooks/cache files, asset deletions. | Selective cherry-pick only. Do not wholesale merge. |
| `origin/dev` | Small Yamada update line. | Treat as superseded by the current combined Yamada fix unless a diff review finds unique tests or docs. |
| `origin/input-adapter` | `knotted_graph.inputs`, adapter tests, input-gallery examples. | Merge early after core Yamada fix. This aligns with the generic package architecture. |
| `origin/add-repulsive-curves` | `knotted_graph.repulsive_layout`, tests, CLI, bootstrap script, third-party notices, large vendored dependency trees. | Merge selectively. Keep Python integration and tests first; decide vendoring policy explicitly. |

## History Rewrite Strategy

Use a clean integration branch and rewrite commit history before publication.

Recommended commit sequence:

1. `test(yamada): capture combined Negami and crossing-resolution regressions`
2. `fix(yamada): resolve crossings through half-edge ports`
3. `fix(yamada): use inverse removed-edge exponent in Negami polynomial`
4. `build: move package metadata to uv build`
5. `docs: move getting started notebook under doc`
6. `feat(yamada): add recursive deletion-contraction backend`
7. `test(yamada): compare recursive and Negami backends on small graphs`
8. `feat(inputs): add generic input adapter layer`
9. `test(inputs): enforce core spatial-graph adapter contract`
10. `feat(repulsive-layout): add optional graph relaxation API`
11. `test(repulsive-layout): preserve graph topology and metadata with fake driver`
12. `docs(architecture): document generic package architecture and NodalSkeleton role`

Keep generated caches, large ad hoc datasets, exploratory notebooks, and vendored
third-party source out of the public history unless they are intentionally part
of the distribution policy.

## Merge Phases

### Phase 0: Stabilize Current Work

Goal: make the staged Yamada fix a clean, tested base.

Actions:

1. Split staged changes if necessary:
   - Yamada code and tests.
   - Packaging and lockfile.
   - Notebook/doc move.
2. Run the focused Yamada regression test.
3. Add one import smoke test proving the algebraic Yamada module can be imported
   without requiring heavy visualization or skeleton dependencies.
4. Commit the Yamada fix only after the focused tests pass.

Required tests:

```bash
uv run --with pytest python -m pytest -q tests/test_yamada_combined_fix.py
uv run --with pytest python -m pytest -q tests/test_import_boundaries.py
```

### Phase 1: Add Recursive Yamada Backend

Goal: satisfy the paper's claim that the engine supports both recursive
deletion-contraction and Negami state-sum routes.

Source:

- Selectively import `src/knotted_graph/yamada/recursive.py` from `origin/main`.
- Consider `src/knotted_graph/yamada/Graphs_collection.py` only if it is needed
  for examples or tests.
- Update `src/knotted_graph/yamada/__init__.py` deliberately.

Required design decision:

- Expose the backend as an explicit option, for example `method="negami"` and
  `method="recursive"`, or keep recursive as a lower-level graph API until it
  is wired into full PD-code evaluation.

Required tests:

```bash
uv run --with pytest python -m pytest -q tests/test_yamada_recursive.py
uv run --with pytest python -m pytest -q tests/test_yamada_backend_equivalence.py
```

Test cases:

- Empty graph evaluates to `1`.
- Bouquet `B_n` matches `-(-sigma)^n`.
- Cycle graph matches `sigma`.
- Theta graph matches `(sigma + (-sigma)^s) / (sigma + 1)`.
- Disjoint union factorizes.
- Bridge-containing graph evaluates to zero under the Yamada specialization.
- Recursive and Negami backends agree on a curated set of small crossing-free
  multigraphs.
- Memoization key is invariant under node relabeling and preserves parallel
  edges and loops.

### Phase 2: Merge Generic Input Adapter Layer

Goal: make the repository a generic spatial-graph package, with `NodalSkeleton`
as one application path rather than the central wrapper.

Source:

- Merge `origin/input-adapter` after the Yamada base is green.
- Keep `src/knotted_graph/inputs/*`.
- Keep adapter unit tests.
- Review examples and gallery scripts separately from the library code.

Required tests:

```bash
uv run --with pytest python -m pytest -q \
  tests/test_inputs_coordinate_chain.py \
  tests/test_inputs_spatial_graph_csv.py \
  tests/test_inputs_polymer.py \
  tests/test_inputs_biomolecular.py \
  tests/test_inputs_surface_mesh.py
```

Contract tests to add or preserve:

- Every graph adapter returns a `networkx.MultiGraph`.
- Every node has finite numeric `pos` with shape `(3,)`.
- Every edge preserves key identity for parallel edges.
- Every edge `pts`, when present, has shape `(n, 3)`.
- Edge `pts[0]` and `pts[-1]` match endpoint node positions unless the adapter
  explicitly documents a different convention.
- Bad schemas fail with targeted `ValueError` messages.
- Optional dependencies are skipped cleanly when unavailable.

### Phase 3: Reframe `NodalSkeleton` as an Application

Goal: keep the existing physics workflow while making generic spatial-graph
computation the public center of the package.

Actions:

1. Avoid presenting `NodalSkeleton` as the package-level input wrapper in docs.
2. Document it as an application workflow:
   `Hamiltonian/Bloch vector -> skeleton -> core spatial graph -> PD/Yamada`.
3. Add or update a method/property that clearly exports the core graph contract.
4. Ensure `NodalSkeleton` imports do not make generic algebra modules expensive.

Required tests:

```bash
uv run --with pytest python -m pytest -q tests/test_nodal_skeleton_core_contract.py
uv run --with pytest python -m pytest -q tests/test_import_boundaries.py
```

Contract tests:

- `NodalSkeleton.skeleton_graph` is a `networkx.MultiGraph`.
- Nodes have finite `pos`.
- Edges have `pts` when geometric paths are available.
- Graph summary and Yamada projection paths consume the same core graph contract
  used by non-physics adapters.
- Importing `knotted_graph.yamada.polynomial` does not import `pyvista`,
  `skimage`, or `poly2graph`.

### Phase 4: Merge Repulsive Layout Selectively

Goal: add optional 3D graph simplification without making it mandatory for
algebraic users or forcing a large vendored dependency into the core package.

Source:

- Select from `origin/add-repulsive-curves`:
  - `src/knotted_graph/repulsive_layout/*`
  - `tests/test_repulsive_layout_*.py`
  - `scripts/bootstrap_repulsion.py`
  - `THIRD_PARTY_NOTICES.md`
- Treat `external/repulsive-curves/*` and
  `src/knotted_graph/repulsive_layout/vendor/*` as a separate policy decision.

Recommended vendoring policy:

1. Prefer bootstrap/download/build instructions for public package history.
2. Keep vendored source out of the default package unless reproducibility
   requires it.
3. If vendoring is chosen, isolate it in one commit with license notices and
   package-exclusion rules reviewed explicitly.

Required tests:

```bash
uv run --with pytest python -m pytest -q \
  tests/test_repulsive_layout_spatial_graph_api.py \
  tests/test_repulsive_layout_topology.py \
  tests/test_repulsive_layout_driver.py \
  tests/test_repulsive_layout_downsampling.py \
  tests/test_repulsive_layout_cli.py
```

Contract tests:

- `relax_spatial_graph` accepts the core `MultiGraph` contract.
- Node labels, edge keys, graph attrs, node attrs, and edge attrs are preserved.
- Pinned graph nodes remain fixed by default.
- Returned graph is a new graph and does not mutate the input graph.
- Fake-driver tests cover API behavior without compiling external C++.
- Independent topology verifier detects a swept crossing.
- Topology verification remains opt-in and does not slow normal tests.

### Phase 5: Integration Tests Across the Generic Pipeline

Goal: prove that the architecture works as a generic package, not only as a
physics-specific workflow.

Required tests:

```bash
uv run --with pytest python -m pytest -q tests/test_pipeline_generic_graph_to_yamada.py
uv run --with pytest python -m pytest -q tests/test_pipeline_inputs_to_core.py
uv run --with pytest python -m pytest -q tests/test_pipeline_repulsive_to_pd.py
```

Integration scenarios:

- Coordinate chain input -> core graph -> PD encoding.
- Spatial-graph CSV -> core graph -> projection -> PDCode.
- Small hand-built graph -> recursive Yamada backend.
- Small PDCode with crossings -> Negami Yamada backend.
- Repulsive layout with fake driver -> core graph remains valid -> projection
  still succeeds.
- `NodalSkeleton` example -> core graph -> same downstream generic APIs.

### Phase 6: Documentation and Architecture Cleanup

Goal: make docs match the final public package shape.

Required updates:

- Update `dev/Architecture.md`.
- Replace the old "Input Wrapper = NodalSkeleton" mapping with
  `knotted_graph.inputs` plus application workflows.
- Change "Repulsive Curves" anchor from `yamada/planar_diagram.py` to
  `repulsive_layout`.
- State that `NodalSkeleton` is a domain application built on the generic core.
- Update the paper's section 10 before publication.
- Add a short public architecture note in user-facing docs.

Documentation tests/checks:

```bash
uv run python -m compileall src
uv run --with pytest python -m pytest -q tests/test_import_boundaries.py
```

## Test Suite Shape

Use markers to keep fast tests default and expensive tests explicit.

| Marker | Meaning | Default |
|---|---|---|
| no marker | Pure Python unit and contract tests | Run always. |
| `integration` | Cross-module pipeline tests with small fixtures | Run in normal CI if stable. |
| `slow` | Expensive projection, skeletonization, or larger graph tests | Run in scheduled or release CI. |
| `external` | Requires Repulsor, PyVista rendering, system tools, or network data | Opt-in only. |

Recommended default command:

```bash
uv run --with pytest python -m pytest -q
```

Recommended release command:

```bash
uv run --with pytest python -m pytest -q -m "not external"
```

Recommended external smoke command:

```bash
uv run --with pytest python -m pytest -q -m external
```

## Public Main Acceptance Criteria

The branch is ready to become public main when all of these are true:

- Generic package imports work without `NodalSkeleton`.
- `NodalSkeleton` is documented as an application workflow.
- Input adapters produce the same core graph contract used by Yamada and
  repulsive layout.
- Yamada has regression coverage for the current combined fix.
- Recursive and Negami routes are either both wired into the public API or the
  documentation clearly states the current exposure level.
- Repulsive layout is optional and tested with fake-driver unit tests.
- Vendored third-party code policy is explicit.
- No exploratory caches, large local datasets, or accidental generated files are
  included in public history.
- `uv run --with pytest python -m pytest -q` passes.
- `dev/Architecture.md` and the paper architecture section agree with the code.

