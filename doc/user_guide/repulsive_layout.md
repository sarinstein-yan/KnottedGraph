# Repulsive Layout

Repulsive layout is an optional workflow for simplifying a complicated spatial
embedding while preserving graph incidence. It is useful when a biological,
polymer, or geometric embedding creates too many close contacts or projection
crossings for a readable planar diagram or tractable Yamada computation.

## Relaxing A Spatial Graph

Any graph shown below can be passed to `relax_spatial_graph` once the external
Repulsor driver is available. The code path mirrors the stages in the figure:
prepare a spatial graph, write the Repulsor workspace, run topology-safe
layout, convert the final curve back to a `MultiGraph`, then use the ordinary
plotting and projection tools.

```python
from pathlib import Path

from knotted_graph.layout.repulsive import SolverOptions
from knotted_graph.layout.repulsive import relax_spatial_graph

layout = relax_spatial_graph(
    graph,
    workspace="repulsive-workspace",
    solver_options=SolverOptions(steps=100, max_time=20, threads=1),
    save_steps=True,
    keep_workspace=True,
    verify_topology=True,
)

relaxed_graph = layout.graph
metadata = layout.metadata
final_obj = layout.final_obj
```

Successful runs return an object shaped like this:

```text
type(layout).__name__
'GraphLayoutResult'

layout.graph.number_of_nodes(), layout.graph.number_of_edges()
(same_node_count, same_edge_count)

layout.final_obj
PosixPath('repulsive-workspace/final_simplified.obj')

sorted(layout.metadata)[:5]
['certificate', 'clearance_report_json', 'clearance_summary', 'compactness', 'curve_mapping']
```

The graph incidence is preserved: nodes and edges represent the same topology,
but their `pos` and `pts` coordinates have been replaced by a cleaner embedding.
The workspace files are useful when users want to inspect intermediate layout
steps or create figures.

Inspect the same stages that are shown in the repulsive-curve figure:

```python
workspace = Path(layout.workspace)

print(f"input_nodes_edges = {(graph.number_of_nodes(), graph.number_of_edges())}")
print(f"relaxed_nodes_edges = {(relaxed_graph.number_of_nodes(), relaxed_graph.number_of_edges())}")
print(f"initial_obj = {Path(metadata['initial_obj']).name}")
print(f"curve_file = {Path(metadata['curve_txt']).name}")
print(f"final_obj = {Path(metadata['final_obj']).name}")
print(f"topology_verified = {metadata['topology_verification']['verified']}")
print(f"clearance_summary = {metadata['clearance_summary']}")
```

Typical output:

```text
input_nodes_edges = (2, 3)
relaxed_nodes_edges = (2, 3)
initial_obj = initial.obj
curve_file = repulsor_curve.txt
final_obj = final_simplified.obj
topology_verified = True
clearance_summary = {'initial_min_distance': ..., 'relaxed_min_distance': ..., 'final_min_distance': ...}
```

Use the same Plotly style before and after layout. The axes are intentionally
visible: for layout debugging the user should see where close approaches occur
in 3D, not only an isolated curve drawing.

```python
from knotted_graph.visualization import plot_3D_graph_plotly
```

For a paper figure or debugging page, put the two plots next to each other so
the user can see what the repulsive pass changed:

```python
from plotly.subplots import make_subplots

before_fig = plot_3D_graph_plotly(graph)
after_fig = plot_3D_graph_plotly(relaxed_graph)

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    horizontal_spacing=0.02,
)
for trace in before_fig.data:
    fig.add_trace(trace, row=1, col=1)
for trace in after_fig.data:
    fig.add_trace(trace, row=1, col=2)

fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
    scene2=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
    showlegend=False,
)
fig.show()
```

```{image} ../assets/plot_outputs/repulsive_before_after_plotly.png
:alt: Output from plotting a tangled theta graph before and after repulsive layout
:width: 92%
:align: center
```

The left panel is the input embedding. The right panel is the cleaner embedded
representative returned as `layout.graph`. The nodes and edge keys are the same;
only the geometric representative has changed.

The figures in the next sections show three actual input graphs accepted by the
repulsive-layout API. After a solver run, replace each input graph with
`layout.graph` and call the same plotting helper.

## Example 1: Geometric Theta Graph

```python
import networkx as nx
import numpy as np


def tangled_theta_graph(samples=130):
    graph = nx.MultiGraph()
    source = np.array([0.0, 0.0, 0.0])
    target = np.array([3.0, 0.0, 0.0])
    graph.add_node("u", pos=source)
    graph.add_node("v", pos=target)

    s = np.linspace(0.0, 1.0, samples)
    envelope = np.sin(np.pi * s)
    phases = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]

    for index, phase in enumerate(phases, start=1):
        angle = 2 * np.pi * 1.15 * s + phase
        pts = np.column_stack(
            [
                3.0 * s,
                0.92 * envelope * np.cos(angle) + 0.12 * np.sin(5 * np.pi * s + phase),
                0.92 * envelope * np.sin(angle),
            ]
        )
        pts[0] = source
        pts[-1] = target
        graph.add_edge("u", "v", key=f"arc_{index}", pts=pts)

    return graph


graph = tangled_theta_graph()
fig = plot_3D_graph_plotly(graph)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/plot_outputs/repulsive_example_geometric_theta.png
:alt: Output from plotting a geometric theta graph input for repulsive layout
:width: 70%
:align: center
```

This is the smallest useful repulsive-layout example: two graph vertices and
three curved arcs. It is good for checking that graph incidence and edge keys
are preserved.

## Example 2: Protein-Derived Theta Graph

```python
from pathlib import Path

from knotted_graph.layout.repulsive.curve_io import curve_network_to_multigraph
from knotted_graph.layout.repulsive.protein_examples import (
    build_protein_example,
    set_special_node_distance,
)

network = build_protein_example(
    "1aoc",
    pdb_cache=Path("pdb-cache"),
    total_arc_points=64,
)
set_special_node_distance(network, target_distance=9.0)

graph = curve_network_to_multigraph(network)
fig = plot_3D_graph_plotly(graph)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/plot_outputs/repulsive_example_protein_theta.png
:alt: Output from plotting a protein-derived theta graph input for repulsive layout
:width: 70%
:align: center
```

This example starts from the protein helper used in the repulsive-curve
workflow. The conversion step returns the same spatial `MultiGraph(pos/pts)`
contract used by projection, plotting, and invariant code.

## Example 3: Closed Polymer Loop

```python
import numpy as np

from knotted_graph.inputs.coordinate_chain import coordinates_to_multigraph

t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
coords = np.column_stack(
    [
        (2 + np.cos(3 * t)) * np.cos(2 * t),
        (2 + np.cos(3 * t)) * np.sin(2 * t),
        np.sin(3 * t),
    ]
)
coords = np.vstack([coords, coords[0]])

graph = coordinates_to_multigraph(
    coords,
    closed=True,
    closure="direct",
    input_id="closed_polymer_loop",
)

fig = plot_3D_graph_plotly(graph)
fig.update_layout(
    title=None,
    scene=dict(
        xaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        yaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        zaxis=dict(
            visible=True,
            title="",
            showticklabels=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="black",
            linewidth=2,
        ),
        aspectmode="data",
        camera=dict(eye=dict(x=1.45, y=1.55, z=1.18)),
    ),
)
fig.show()
```

```{image} ../assets/plot_outputs/repulsive_example_polymer_loop.png
:alt: Output from plotting a closed polymer-loop input for repulsive layout
:width: 70%
:align: center
```

This example shows the single-loop case. It is useful when a simulation or
coordinate-chain input has no distinguished graph vertices except the closure
anchor.

## Checking Topology After Layout

For small examples where projection and invariant calculation are tractable,
compare the Yamada polynomial before and after layout.

```python
import sympy as sp
from pathlib import Path

from knotted_graph.layout.repulsive import SolverOptions, relax_spatial_graph
from knotted_graph.layout.repulsive.curve_io import curve_network_to_multigraph
from knotted_graph.layout.repulsive.protein_examples import (
    build_protein_example,
    set_special_node_distance,
)
from knotted_graph.projection import compute_yamada_polynomial

Y = sp.Symbol("Y")

network = build_protein_example(
    "1aoc",
    pdb_cache=Path("pdb-cache"),
    total_arc_points=42,
)
set_special_node_distance(network, target_distance=9.0)
graph = curve_network_to_multigraph(network)

layout = relax_spatial_graph(
    graph,
    workspace="protein-layout",
    solver_options=SolverOptions(steps=100, max_time=20, threads=1),
    save_steps=True,
    keep_workspace=True,
    verify_topology=True,
)
relaxed_graph = layout.graph

before = compute_yamada_polynomial(
    graph,
    Y,
    return_result=True,
    n_jobs=1,
)
after = compute_yamada_polynomial(
    relaxed_graph,
    Y,
    return_result=True,
    n_jobs=1,
)

print(f"Upsilon(input; Y) = {sp.expand(before.polynomial)}")
print(f"Upsilon(relaxed; Y) = {sp.expand(after.polynomial)}")
print(before.polynomial == after.polynomial)
```

Expected output for a topology-preserving run:

```text
Upsilon(input; Y) = -Y**12 - Y**11 - Y**10 - Y**9 - Y**8 - Y**6 - Y**4 + 1
Upsilon(relaxed; Y) = -Y**12 - Y**11 - Y**10 - Y**9 - Y**8 - Y**6 - Y**4 + 1
True
```

For larger examples, first inspect the saved layout workspace, clearance report,
and topology certificate. Then compute invariants on selected projections when
the crossing count is small enough.

## When To Use It

Use repulsive layout when:

- the embedded graph has near self-contacts or long geometric detours;
- projection produces many crossings;
- a cleaner planar diagram is needed for figures or PD-code inspection;
- the topology is fixed but the geometric representative can be improved.

Skip it when the input graph is already simple enough for projection.

## Optional Dependencies

Repulsive layout uses optional dependencies and an external driver workflow.
Install the package with the `repulsion` extra when using this chapter's tools.
