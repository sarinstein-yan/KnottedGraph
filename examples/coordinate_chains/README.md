# Coordinate Chain Input Prototype

This folder contains a generic coordinate-curve adapter for Task 2. It is meant
to cover polymer chains, simulation outputs, hand-built curves, and other data
sources that already provide ordered 3D points.

The stable coordinate-chain adapter is also available through the core
`knotted_graph.inputs` API.

## Supported Inputs

The prototype currently supports:

- direct NumPy arrays with shape `(N, 3)`;
- `.npy` coordinate arrays;
- CSV files with `x`, `y`, `z` columns;
- JSON files with `points` or `coords`;
- TSV / whitespace `.dat` tables with x y z columns;
- XYZ-style files with either `label x y z` rows or bare `x y z` rows.

The adapter can represent:

- open coordinate chains, with a `start` node and an `end` node;
- closed coordinate loops, with a `loop_anchor` self-loop edge.

Both cases produce a `networkx.MultiGraph` with:

- node attribute `pos`;
- edge attribute `pts`;
- graph metadata `input_kind="coordinate_curve"`;
- graph metadata `is_closed=True/False`.

## Smoke Test

Run from the repository root:

```bash
PYTHONPATH=src python examples/coordinate_chains/plot_coordinate_chain_examples.py
```

The smoke test generates three small example inputs:

- `data/open_helix_polymer.csv`
- `data/open_helix_polymer.json`
- `data/open_helix_polymer.tsv`
- `data/open_helix_polymer.dat`
- `data/closed_trefoil_ring.xyz`
- `data/closed_trefoil_ring.npy`

and produces:

- `figures/open_helix_polymer_csv.png`
- `figures/open_helix_polymer_csv_graph.html`
- `figures/open_helix_polymer_json.png`
- `figures/open_helix_polymer_json_graph.html`
- `figures/open_helix_polymer_tsv.png`
- `figures/open_helix_polymer_tsv_graph.html`
- `figures/open_helix_polymer_dat.png`
- `figures/open_helix_polymer_dat_graph.html`
- `figures/closed_trefoil_ring_xyz.png`
- `figures/closed_trefoil_ring_xyz_graph.html`
- `figures/closed_trefoil_ring_npy.png`
- `figures/closed_trefoil_ring_npy_graph.html`

## Example API Shape

Current prototype calls look like:

```python
from coordinate_curve_adapter import build_curve_from_csv

result = build_curve_from_csv(
    "data/open_helix_polymer.csv",
    closed=False,
    curve_id="open_helix_polymer",
)

G = result.graph
coords = result.coords
```

Core-library API:

```python
from knotted_graph.inputs import from_coordinate_chain

result = from_coordinate_chain("ring.xyz", closed=True, closure="direct")
```

## Current Limits

This prototype does not yet support:

- trajectory files with many frames;
- LAMMPS/GROMACS-specific formats;
- automatic periodic-boundary unwrapping;
- gap detection or segment splitting;
- arbitrary meshes/surfaces.

Those should be added incrementally as separate Task 2 adapters.
