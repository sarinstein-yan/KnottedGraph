# Installation

The core package requires Python 3.11 or newer.

```bash
pip install knotted_graph
```

For development from a checkout:

```bash
git clone https://github.com/sarinstein-yan/KnottedGraph.git
cd KnottedGraph
uv sync --all-groups
```

## Optional Extras

Install extras only for the workflows you need:

```bash
pip install "knotted_graph[nodal]"
pip install "knotted_graph[surface]"
pip install "knotted_graph[repulsion]"
pip install "knotted_graph[all]"
```

The nodal-skeleton application uses optional packages such as PyVista,
scikit-image, poly2graph, and minorminer. The generic graph, projection, and
Yamada APIs do not import those optional stacks.

## Documentation Build

```bash
uv run --group docs python -m sphinx -b html -W --keep-going doc doc/_build/html
```
