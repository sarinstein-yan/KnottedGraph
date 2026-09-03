# Protein Derived Spatial Graphs

<div class="kg-hero">
  <p class="kg-lead">The protein applications notebook is the place for protein-coordinate input, graph construction, possible repulsive relaxation, projection, and invariant output. This page points to that notebook and uses the repulsive-curves figure as the geometric workflow reference.</p>
  <div class="kg-link-row">
    <a href="https://github.com/sarinstein-yan/KnottedGraph/blob/codex/arbitrary-knot-user-integration/User_guide/applications/03_protein_applications.ipynb">Open 03_protein_applications.ipynb</a>
    <a href="../user_guide/input_adapters.html">PDB/mmCIF input guide</a>
  </div>
</div>

<div class="kg-wide-figure">
  <img src="../site_figures/repulsive_curves.png" alt="Repulsive curves workflow for embedded spatial graphs">
</div>

## What is implemented now

The public input layer can extract an **ordered backbone trace** from a local
PDB/mmCIF file or an RCSB identifier:

```python
from knotted_graph.inputs import from_protein_ca_backbone

result = from_protein_ca_backbone(
    "1CRN",
    chain_id="A",
    model_id=1,
)

print(result.coords.shape)
print(result.graph.number_of_nodes(), result.graph.number_of_edges())
print(result.issues)
```

For nucleic acids use the corresponding backbone helper or select the desired
atom explicitly. Multiple matching chains are intentionally not chosen
silently; supply `chain_id` after inspecting the available chains. Remote IDs
need network access on the first download, while local files work offline.

The result represents the sampled trace as one geometric curve edge (or a
self-loop when explicitly closed). Residues/atoms are not automatically
converted into one graph vertex each. Read {doc}`../user_guide/input_adapters`
for atom selection, closure, metadata, and mmCIF parser boundaries.

## What is not yet a generic workflow

A protein-derived network may mean very different things: a contact graph,
residue-interaction network, cavity skeleton, domain graph, repulsive layout,
or simply the ordered backbone. KnottedGraph does not currently choose one of
these meanings for the user.

Before creating a derived spatial graph, document:

- which atoms/residues and model/chain were selected;
- the distance, contact, domain, or cavity rule that creates topology;
- coordinate units and any periodic/closure treatment;
- what each node and edge represents scientifically; and
- validation and perturbation checks for the derived graph.

Only after that mapping is explicit should the graph enter cleanup, optional
repulsive relaxation, projection, and invariant computation. The native
Repulsor route is a separate external-backend workflow; it is not invoked by
the PDB/mmCIF adapters.

## Recommended next step

If an ordered backbone is the intended object, continue directly with
{doc}`../user_guide/workflow_overview`. If you need a domain-specific contact or
cavity graph, treat its construction as an application method and validate it
before presenting the result as a library-supported generic conversion.
