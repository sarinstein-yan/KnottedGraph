Example graphs for quick testing.

Files:
- `k5.edgelist`: complete graph on 5 nodes.
- `k33.edgelist`: complete bipartite graph K3,3.
- `toy_multicellular_network.edgelist`: a synthetic 18-node, 36-edge modular graph with dense intra-group structure and a handful of inter-group bridges. Intended as a rough stand-in for a multicellular interaction graph.

Quick start:
```bash
python python/repulsive_graph_layout.py python/example_graphs/toy_multicellular_network.edgelist \
  --workspace build/toy_multicellular \
  --steps 20 \
  --samples-per-edge 12 \
  --engine ctypes \
  --solver build/bin/librcurves_shared.dll
```

Then render a static figure:
```bash
python python/render_layout.py build/toy_multicellular/layout.json \
  --output build/toy_multicellular/layout.png
```
