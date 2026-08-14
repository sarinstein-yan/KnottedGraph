# Protein-Derived Theta Graphs

Biomolecular workflows often start from coordinate data but need the same
embedded-graph contract used by projection, visualization, repulsive layout,
and Yamada computation. The protein helpers currently expose three built-in
theta-graph examples: `1aoc`, `3ulk`, and `5osq`.

## Build A Protein Theta Graph

```python
from pathlib import Path

from knotted_graph.layout.repulsive import (
    available_samples,
    build_protein_example,
    curve_network_to_multigraph,
    set_special_node_distance,
)

print(available_samples())

network = build_protein_example(
    "1aoc",
    pdb_cache=Path("pdb-cache"),
    total_arc_points=42,
)
set_special_node_distance(network, target_distance=9.0)

graph = curve_network_to_multigraph(network)
source, target = network.node_order

print(network.name)
print(network.node_order)
print(network.arc_order)
print(graph.graph["input_kind"])
print(graph.number_of_nodes(), graph.number_of_edges())
print({key: len(graph.edges[source, target, key]["pts"]) for key in network.arc_order})
```

Output:

```text
('1aoc', '3ulk', '5osq')
1AOC theta_31
('C140', 'C134')
('arc1', 'arc2', 'arc3')
curve_network
(2, 3)
{'arc1': 18, 'arc2': 14, 'arc3': 10}
```

The printed values confirm that the protein example has two graph vertices and
three embedded arcs. The arc lengths are the numbers of sampled 3D points kept
on each protein-derived curve after resampling.

Plot the converted theta graph:

```python
from knotted_graph.visualization import plot_3D_graph_plotly

fig = plot_3D_graph_plotly(graph)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.35, y=1.65, z=1.2)),
    ),
)
fig.show()
```

```{image} ../assets/plot_outputs/protein_1aoc_theta_graph_plotly.png
:alt: Output from plotting the 1AOC protein-derived theta graph
:width: 75%
:align: center
```

## Compare The Built-In Protein Examples

Use the same code for all supported samples. If a PDB file is not already
available locally, the helper downloads it into `pdb_cache`.

```python
from knotted_graph.projection import select_projection

for sample in ("1aoc", "3ulk", "5osq"):
    network = build_protein_example(
        sample,
        pdb_cache=Path("pdb-cache"),
        total_arc_points=42,
    )
    if sample == "1aoc":
        set_special_node_distance(network, target_distance=9.0)

    graph = curve_network_to_multigraph(network)
    projection = select_projection(graph, num_rotation_samples=12)

    print(
        sample,
        network.name,
        (graph.number_of_nodes(), graph.number_of_edges()),
        {key: len(network.arc_polylines[key]) for key in network.arc_order},
        projection.num_crossings,
    )
```

Output:

```text
1aoc 1AOC theta_31 (2, 3) {'arc1': 18, 'arc2': 14, 'arc3': 10} 5
3ulk 3ULK theta_41 (2, 3) {'arc1_closure': 18, 'arc2_backbone': 14, 'arc3_mg_bridge': 10} 27
5osq 5OSQ theta (2, 3) {'arc1_ca_closure': 19, 'arc2_cys_bridge': 12, 'arc3_backbone': 11} 25
```

The larger selected crossing counts for `3ulk` and `5osq` are the reason those
examples are useful for the repulsive-layout workflow before invariant
calculation.

```python
from knotted_graph.visualization import plot_3D_graph_plotly

network_3ulk = build_protein_example(
    "3ulk",
    pdb_cache=Path("pdb-cache"),
    total_arc_points=42,
)
graph_3ulk = curve_network_to_multigraph(network_3ulk)

fig = plot_3D_graph_plotly(graph_3ulk)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.35, y=1.65, z=1.2)),
    ),
)
fig.show()
```

```{image} ../assets/plot_outputs/protein_3ulk_theta_graph_plotly.png
:alt: Output from plotting the 3ULK protein-derived theta graph
:width: 75%
:align: center
```

```python
network_5osq = build_protein_example(
    "5osq",
    pdb_cache=Path("pdb-cache"),
    total_arc_points=42,
)
graph_5osq = curve_network_to_multigraph(network_5osq)

fig = plot_3D_graph_plotly(graph_5osq)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(visible=True),
        yaxis=dict(visible=True),
        zaxis=dict(visible=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.35, y=1.65, z=1.2)),
    ),
)
fig.show()
```

```{image} ../assets/plot_outputs/protein_5osq_theta_graph_plotly.png
:alt: Output from plotting the 5OSQ protein-derived theta graph
:width: 75%
:align: center
```

## Projection and Yamada Output

The `1aoc` graph is small enough to project and evaluate immediately.

```python
import sympy as sp

from knotted_graph.projection import compute_yamada_polynomial, select_projection

Y = sp.Symbol("Y")

network = build_protein_example(
    "1aoc",
    pdb_cache=Path("pdb-cache"),
    total_arc_points=42,
)
set_special_node_distance(network, target_distance=9.0)
graph = curve_network_to_multigraph(network)

projection = select_projection(graph, num_rotation_samples=12)
result = compute_yamada_polynomial(
    graph,
    Y,
    num_rotation_samples=12,
    return_result=True,
    n_jobs=1,
)

print(f"selected_crossings = {projection.num_crossings}")
print(f"rotation_angles = {tuple(round(a, 2) for a in projection.rotation_angles)}")
print(f"pd_code = {projection.pd_code}")
print(f"Upsilon(G; Y) = {sp.expand(result.polynomial)}")
```

Output:

```text
selected_crossings = 5
rotation_angles = (-84.98, 77.98, 0.0)
pd_code = V[11,0,6];V[12,10,5];X[12,9,11,10];X[8,3,7,4];X[1,8,0,9];X[4,7,5,6]
Upsilon(G; Y) = -Y**12 - Y**11 - Y**10 - Y**9 - Y**8 - Y**6 - Y**4 + 1
```

Plot the selected protein projection:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 4))
for arc in projection.arcs:
    ax.plot(*arc.line.xy, color="tab:blue", linewidth=2.6)
for vertex in projection.vertices:
    ax.scatter(*vertex.point.xy, color="tab:red", zorder=3)
for crossing in projection.crossings:
    ax.scatter(*crossing.point.xy, marker="x", color="black", zorder=4)

ax.set_aspect("equal")
ax.axis("off")
plt.show()
```

```{image} ../assets/plot_outputs/protein_1aoc_selected_projection.png
:alt: Output from plotting the selected projection of the 1AOC protein-derived theta graph
:width: 75%
:align: center
```

## Repulsive Layout Before Large Invariants

For `3ulk` and `5osq`, compute a cleaner representative before asking for a
large Yamada calculation. Repulsive layout changes the geometric representative,
not the graph incidence.

```python
from knotted_graph.layout.repulsive import SolverOptions, relax_spatial_graph

layout = relax_spatial_graph(
    graph,
    workspace="protein-layout",
    solver_options=SolverOptions(steps=100, max_time=20, threads=1),
    save_steps=True,
    keep_workspace=True,
    verify_topology=True,
)

relaxed_graph = layout.graph
relaxed = compute_yamada_polynomial(
    relaxed_graph,
    Y,
    num_rotation_samples=12,
    return_result=True,
    n_jobs=1,
)

print(f"relaxed_nodes_edges = {(relaxed_graph.number_of_nodes(), relaxed_graph.number_of_edges())}")
print(f"Upsilon(G_relaxed; Y) = {sp.expand(relaxed.polynomial)}")
print(f"same_yamada = {sp.expand(relaxed.polynomial - result.polynomial) == 0}")
print(Path(layout.metadata["final_obj"]).name)
```

For a topology-preserving `1aoc` run, the expected invariant check is:

```text
relaxed_nodes_edges = (2, 3)
Upsilon(G_relaxed; Y) = -Y**12 - Y**11 - Y**10 - Y**9 - Y**8 - Y**6 - Y**4 + 1
same_yamada = True
final_simplified.obj
```

The exact relaxed coordinates and selected projection can change with solver
settings, but the `same_yamada` line should remain `True`. If it is false,
inspect the saved workspace files and projection before using the relaxed
embedding in a paper figure. The before-and-after plotting pattern for a
relaxed graph is shown in [Repulsive Layout](../user_guide/repulsive_layout.md).
