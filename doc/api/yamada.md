# Yamada Invariant

KnottedGraph exposes two related entry paths:

- crossing-free abstract graphs can be evaluated directly with
  {py:func}`knotted_graph.invariants.yamada.compute_yamada_polynomial_recursive`;
- embedded spatial graphs first pass through projection and PD-code construction
  with {py:func}`knotted_graph.projection.compute_yamada_polynomial`.

The embedded entry point chooses a projection, resolves its crossings, and then
evaluates the state graphs. Pass `return_result=True` when you need the selected
rotation, PD code, and crossing count for provenance.

```python
import sympy as sp
from knotted_graph.projection import compute_yamada_polynomial

Y = sp.Symbol("Y")
result = compute_yamada_polynomial(
    graph,
    Y,
    n_jobs=1,
    return_result=True,
)
print(result.polynomial)
print(result.projection.num_crossings)
```

State resolution grows exponentially with diagram crossing count. The safe
default is one worker; `n_jobs=-1` is an explicit request to use all available
cores. The crossing warning indicates expected cost, not an invalid result.
Also note that a bridge (cut edge) makes the Yamada polynomial zero, so zero is
not in itself evidence of a failed calculation.

The onboarding examples use `Y` and write the invariant as
`\Upsilon(G;Y)`. Backend formulas may use other temporary symbols internally.
See [Troubleshooting](../troubleshooting.md) before increasing parallelism.

```{eval-rst}
.. automodule:: knotted_graph.invariants.yamada.polynomial
   :members:

.. automodule:: knotted_graph.invariants.yamada.recursive
   :members:
```
