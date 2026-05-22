# Polymer Snapshot Inputs

This Task 2 smoke test uses the public polymer snapshot input adapters as input
coordinate curves.

Supported inputs:

- LAMMPS dump files with `id mol type x y z` atom rows.
- GROMACS `.gro` snapshots.

The adapter extracts an ordered chain, converts it into a coordinate curve, and
then builds the same `networkx.MultiGraph` convention used by the plotting path:
node `pos` attributes and edge `pts` arrays.

Run from the repository root:

```bash
PYTHONPATH=src python examples/polymers/plot_polymer_snapshot_examples.py
```
