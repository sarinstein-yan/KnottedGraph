# Abstract Spatial Graph Input Prototype

This folder contains an examples-level adapter for user-friendly abstract
spatial graph inputs. It is part of Task 2 and does not touch the downstream
PD-code or Yamada-polynomial pipeline.

## Status

Spatial Graph CSV is already supported as a Task 2 prototype. The current goal
is to demonstrate that user-provided 3D component/interconnect data can be
converted into the same internal graph object used by the rest of the library.

Typical source domains include:

- electric circuits;
- pipe or cooling networks;
- mechanical component-interconnect systems;
- other embedded spatial networks.

The output is not a PD code. The output is an embedded
`networkx.MultiGraph(pos/pts)`, which can then be plotted or passed to later
topology-processing steps when appropriate.

## Supported Input

The current prototype supports a JSON graph format:

```json
{
  "graph_id": "theta_graph",
  "nodes": {
    "u": [-1, 0, 0],
    "v": [1, 0, 0]
  },
  "edges": [
    {
      "id": "arc_0",
      "source": "u",
      "target": "v",
      "points": [[-1, 0, 0], [0, 0.5, 1], [1, 0, 0]]
    }
  ]
}
```

Each node must have a 3D `pos`. Each edge may provide `points` or `pts`.
If no edge points are provided, the adapter creates a straight segment between
the source and target node.

It also supports a node/edge CSV pair:

- node CSV: `id,x,y,z`
- edge CSV: `id,source,target,points_json`

The optional `points_json` field stores an embedded edge as
`[[x, y, z], ...]`. If it is empty, the edge becomes a straight segment between
the source and target nodes.

The adapter returns a `networkx.MultiGraph` with:

- node attribute `pos`;
- edge attribute `pts`;
- graph metadata `input_kind="abstract_spatial_graph"`.

## Recommended Public CSV Schema

The public CSV adapter uses explicit column names:

`nodes.csv`

```csv
node_id,x,y,z,label,type
1,0,0,0,Component 1,component
2,1,0,0,Component 2,component
3,1,1,0,Component 3,component
4,0,1,0,Component 4,component
```

`edges.csv`

```csv
edge_id,source,target,label,type,points_json
e1,1,2,Wire 1,wire,
e2,2,3,Pipe 1,pipe,
e3,3,4,Wire 2,wire,
e4,4,1,Pipe 2,pipe,
```

Recommended columns:

- `nodes.csv` required: `node_id`, `x`, `y`, `z`;
- `nodes.csv` optional metadata: `label`, `type`;
- `edges.csv` required: `edge_id`, `source`, `target`;
- `edges.csv` optional metadata: `label`, `type`;
- `edges.csv` optional geometry: `points_json`.

The optional `points_json` column stores a full embedded 3D polyline for curved
wires, pipes, or cables:

```csv
edge_id,source,target,label,type,points_json
e1,1,2,Curved Wire,wire,"[[0,0,0],[0.3,0.2,0.8],[0.7,0.2,0.8],[1,0,0]]"
```

If `points_json` is empty, the adapter should create a straight segment from the
source node position to the target node position. If `points_json` is present,
its first point should match the source node position and its last point should
match the target node position.

For backward compatibility with the current prototype, the implementation may
also accept `id` as an alias for `node_id` in node CSV files, and `id` or `key`
as aliases for `edge_id` in edge CSV files.

## Proposed Public API

The public API is:

```python
from knotted_graph.inputs import from_spatial_graph_csv

result = from_spatial_graph_csv(
    "nodes.csv",
    "edges.csv",
    metadata={"name": "cooling_network_demo"},
)
G = result.graph
```

It should return a `networkx.MultiGraph` with node `pos` and edge `pts`
attributes, preserving optional columns such as `label` and `type` as metadata
where possible.

## Smoke Test

Run from the repository root:

```bash
PYTHONPATH=src python examples/spatial_graphs/plot_spatial_graph_examples.py
PYTHONPATH=src python examples/spatial_graphs/plot_spatial_graph_csv_examples.py
```

The smoke test writes and loads:

- `data/theta_graph.json`
- `data/vascular_bifurcation_nodes.csv`
- `data/vascular_bifurcation_edges.csv`

and produces:

- `figures/theta_graph_json.png`
- `figures/theta_graph_json_graph.html`
- `figures/vascular_bifurcation_csv.png`
- `figures/vascular_bifurcation_csv_graph.html`

## Current Limits

This prototype does not yet support:

- graph schemas with faces or surfaces;
- automatic smoothing/interpolation;
- graph simplification;
- topology calculations.

Those should be added separately as Task 2 input adapters, not as changes to
the downstream topology pipeline.
