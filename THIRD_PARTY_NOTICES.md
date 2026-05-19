# Third-Party Notices

This repository vendors a minimal snapshot of Repulsor for the
`knotted_graph.repulsive_layout` workflow.

## Repulsor

- Upstream: https://github.com/HenrikSchumacher/Repulsor
- Vendored path: `src/knotted_graph/repulsive_layout/vendor/Repulsor`
- Upstream commit used for this snapshot: `e83fd16e80a734f62125132deb1d9954f1788629`
- License: MIT
- Copyright: Copyright (c) 2022 Henrik Schumacher

The upstream MIT license is retained at:

- `src/knotted_graph/repulsive_layout/vendor/Repulsor/LICENSE`

## Tensors

Repulsor depends on the Tensors header-only library.

- Upstream: https://github.com/HenrikSchumacher/Tensors
- Vendored path: `src/knotted_graph/repulsive_layout/vendor/Repulsor/submodules/Tensors`
- Upstream commit used for this snapshot: `ad1567e5d6b3508efa0868773bb356ae28311192`
- License: MIT
- Copyright: Copyright (c) 2022 HenrikSchumacher

The upstream MIT license is retained at:

- `src/knotted_graph/repulsive_layout/vendor/Repulsor/submodules/Tensors/LICENSE`

Compatibility note: the vendored `GMRES.hpp` contains a small compile
compatibility patch in `Stats()` that removes an unreachable malformed string
expression. The numerical solver logic is unchanged.

## Tools

Tensors depends on the Tools header-only library.

- Upstream: https://github.com/HenrikSchumacher/Tools
- Vendored path: `src/knotted_graph/repulsive_layout/vendor/Repulsor/submodules/Tensors/submodules/Tools`
- Upstream commit used for this snapshot: `a5db6dc2c3318c9aba28b73800950b8bb140e468`
- License: MIT
- Copyright: Copyright (c) 2022 HenrikSchumacher

The upstream MIT license is retained at:

- `src/knotted_graph/repulsive_layout/vendor/Repulsor/submodules/Tensors/submodules/Tools/LICENSE`
