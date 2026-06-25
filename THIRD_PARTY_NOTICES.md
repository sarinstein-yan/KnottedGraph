# Third-Party Notices

KnottedGraph's `knotted_graph.repulsive_layout` workflow can use the external
Repulsor project to relax embedded curve networks. Repulsor is not vendored in
this repository.

## Repulsor

- Upstream: https://github.com/HenrikSchumacher/Repulsor
- Expected local checkout: set `REPULSOR_ROOT`, or run
  `python scripts/bootstrap_repulsion.py` to prepare `external/Repulsor`.
- License: MIT
- Copyright: Copyright (c) 2022 Henrik Schumacher

The C++ driver in `src/knotted_graph/repulsive_layout/repulsor_curve_driver.cpp`
is compiled against a user-provided Repulsor checkout by
`knotted_graph.repulsive_layout.driver.build_driver()`.

## Repulsor Dependencies

Repulsor uses additional upstream header-only dependencies, including
Henrik Schumacher's Tensors and Tools libraries. Those dependencies are managed
by the upstream Repulsor checkout and its submodules rather than by this
repository.
