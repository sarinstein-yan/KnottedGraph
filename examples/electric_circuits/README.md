# Electric Circuit Spatial-Network Prototype

This folder represents an electric circuit as an embedded spatial graph. It is
part of Task 2's goal of showing that the library can accept inputs from fields
beyond the package's original examples.

## Supported Input

The smoke test writes a JSON graph with:

- circuit junctions as nodes;
- wires/components as embedded edges;
- edge metadata such as `component="resistor"` or `component="capacitor"`.

The JSON is loaded through the abstract spatial-graph adapter in
`examples/spatial_graphs`.

## Smoke Test

Run from the repository root:

```bash
PYTHONPATH=src python examples/electric_circuits/plot_circuit_spatial_network.py
```

Expected outputs:

- `data/rc_filter_spatial_circuit.json`
- `figures/rc_filter_spatial_circuit.png`
- `figures/rc_filter_spatial_circuit_graph.html`

## Current Limits

This example is geometric only. It does not solve circuit equations or model
electrical behavior; it only demonstrates an electric circuit as a spatial
network input.
