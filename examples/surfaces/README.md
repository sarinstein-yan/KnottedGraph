# Arbitrary Surface / Mesh Input Prototype

This folder contains examples for Task 2 surface and mesh inputs. Unlike
coordinate curves and spatial graphs, the output target here is a PyVista
`PolyData` surface mesh, not a `networkx.MultiGraph`.

This does not modify the downstream topology pipeline.

## Supported Inputs

The core API currently supports PyVista-readable surface mesh files:

- `.obj`
- `.off`
- `.ply`
- `.stl`
- `.vtk`
- `.vtp`

The adapter loads each file, extracts surface geometry, cleans it, triangulates
it, and validates basic mesh properties.

## Smoke Test

Run from the repository root:

```bash
PYTHONPATH=src python examples/surfaces/plot_mesh_surface_examples.py
```

The smoke test creates and reloads:

- `data/sphere_surface.ply`
- `data/torus_surface.obj`
- `data/ellipsoid_surface.stl`
- `data/cube_surface.off`

and produces:

- `figures/sphere_surface_ply.png`
- `figures/sphere_surface_ply.html`
- `figures/sphere_surface_ply.svg`
- `figures/torus_surface_obj.png`
- `figures/torus_surface_obj.html`
- `figures/torus_surface_obj.svg`
- `figures/ellipsoid_surface_stl.png`
- `figures/ellipsoid_surface_stl.html`
- `figures/ellipsoid_surface_stl.svg`
- `figures/cube_surface_off.png`
- `figures/cube_surface_off.html`
- `figures/cube_surface_off.svg`

## Example API Shape

Current prototype calls look like:

```python
from mesh_surface_adapter import build_surface_from_mesh_file

result = build_surface_from_mesh_file("mesh.obj")
mesh = result.mesh
```

Core-library API:

```python
from knotted_graph.inputs import from_surface_mesh

result = from_surface_mesh("surface.obj")
mesh = result.mesh
```

## Current Limits

This prototype does not yet support:

- implicit surfaces or level sets;
- mesh repair beyond clean/triangulate;
- nonmanifold topology validation;
- conversion from surface mesh to skeleton graph.

Those should be separate Task 2 or later-stage tasks, not changes to PD/Yamada.
