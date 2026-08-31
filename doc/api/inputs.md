# Inputs

The input namespace includes two different kinds of public entry point:

- file/array adapters that normalize external coordinate, biomolecular,
  polymer, spatial-CSV, and surface data; and
- analytic constructors such as `KnotFunction.from_name`,
  `KnotFunction.torus`, and `KnotFunction.from_braid`.

Constructing a named, torus, braid-derived, or custom `KnotFunction` uses the
base installation. Sampling its 3-D level set or converting it with
`KnotFunction.to_spatial_graph` requires the `knot-fields` extra and scales
with the sampled volume. A braid construction report validates the chosen
finite Fourier realization; it is not a formal proof certificate for the
underlying existence theorem. See {doc}`../applications/analytic_knot_fields`
before treating a finite-grid extraction as publication evidence.

```{eval-rst}
.. automodule:: knotted_graph.inputs
   :members:
   :undoc-members:
```
