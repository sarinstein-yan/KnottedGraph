# Volumetric Scalar-Field Inputs

This Task 2 smoke test explores scalar volumes as inputs. The adapter extracts
an isosurface from a 3D scalar field and sends the result to the same mesh
surface path used by the surface examples.

Supported prototype inputs:

- `.npy`: one 3D scalar array.
- `.npz`: `values`, plus optional `spacing` and `origin` arrays.

Outputs:

- generated scalar-field files in `data/`,
- extracted `.vtp` isosurface meshes in `data/`,
- PNG, HTML, and SVG previews in `figures/`.

Run from the repository root:

```bash
PYTHONPATH=src python examples/volumetric_fields/plot_volumetric_field_examples.py
```
