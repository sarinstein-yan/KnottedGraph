# Fermi Surface Input Prototype

This folder demonstrates a Fermi surface as an arbitrary surface/mesh input for
Task 2. It uses the mesh adapter from `examples/surfaces`.

## Example

The smoke test samples a simple cubic tight-binding dispersion,

```text
E(k) = cos(kx) + cos(ky) + cos(kz) - mu
```

and extracts the isosurface `E(k)=0` as a PyVista mesh. The mesh is saved as
`.vtp`, then reloaded through the generic surface adapter.

## Smoke Test

Run from the repository root:

```bash
PYTHONPATH=src python examples/fermi_surfaces/plot_fermi_surface_example.py
```

Expected outputs:

- `data/tight_binding_fermi_surface.vtp`
- `figures/tight_binding_fermi_surface.png`
- `figures/tight_binding_fermi_surface.html`
- `figures/tight_binding_fermi_surface.svg`

## Current Limits

This is a surface example, not a full band-structure workflow. It does not yet
read Hamiltonians, band data files, or volumetric datasets from external tools.
