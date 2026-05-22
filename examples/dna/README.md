# DNA Backbone Input Prototype

This folder contains examples for the public nucleic-acid PDB input adapter.
It is part of Task 2 and does not modify the downstream PD/Yamada pipeline.

## Supported Input

The core API downloads an RCSB PDB file and extracts one named atom from a DNA
or RNA chain. The smoke test uses:

- PDB ID: `1BNA`
- chain: `A`
- atom: `P`
- meaning: phosphate trace of one DNA strand

The output is the same graph convention used by the curve examples:

- `networkx.MultiGraph`
- node attribute `pos`
- edge attribute `pts`
- graph metadata `input_kind="nucleic_acid_backbone"`

## Smoke Test

Run from the repository root:

```bash
PYTHONPATH=src python examples/dna/plot_1bna_dna_backbone.py
```

Expected outputs:

- `data/1BNA.pdb`
- `data/1BNA_P_coords.npy`
- `figures/1bna_chainA_phosphate_backbone.png`
- `figures/1bna_chainA_phosphate_backbone_graph.html`

## Current Limits

This example currently traces one chain and one atom type. It does not yet
combine both DNA strands, infer base-pair geometry, or handle mmCIF files.
