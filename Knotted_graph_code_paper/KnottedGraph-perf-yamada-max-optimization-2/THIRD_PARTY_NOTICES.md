# Third-Party Notices

KnottedGraph's `knotted_graph.layout.repulsive` workflow can use the external
Repulsor project to relax embedded curve networks. Repulsor is not vendored in
this repository.

## Repulsor

- Upstream: https://github.com/HenrikSchumacher/Repulsor
- Pinned paper/reproducibility revision:
  `adc56b61f65f5958b59cbd7e1539f44ed0c5e993`
- Expected local checkout: set `REPULSOR_ROOT`, or run
  `python scripts/bootstrap_repulsion.py` to prepare `external/Repulsor`.
- License: MIT
- Copyright: Copyright (c) 2022 Henrik Schumacher

The bootstrap script validates the selected Repulsor revision and initializes
its submodules. Users can intentionally override the pin with
`--repulsor-ref`, but such an override should be recorded when reproducing
paper calculations.

The C++ driver in
`src/knotted_graph/layout/repulsive/repulsor_curve_driver.cpp`
is compiled against the user-provided Repulsor checkout by
`knotted_graph.layout.repulsive.driver.build_driver()`.

## Repulsor Dependencies

Repulsor uses additional upstream dependencies, including Henrik Schumacher's
Tensors and Tools libraries. Those dependencies are managed through the
upstream Repulsor checkout and its submodules rather than being vendored into
this repository.

The reference KnottedGraph driver additionally links against OpenBLAS, LAPACK,
LAPACKE, `fmt`, AMD/SuiteSparse, and pthreads. These are native system
dependencies and are not installed by the Python `repulsion` extra.
