# Protein Backbone Input Prototype

This folder contains examples for Task 2: making input formats more
user-friendly. The protein PDB adapter is now part of the core
`knotted_graph.inputs` API.

## Current Scope

The core API supports:

- downloading a PDB file from RCSB PDB by PDB ID;
- manually parsing `ATOM` records without Biopython;
- extracting primary C-alpha atoms with atom name `CA`;
- selecting a protein chain by `chain_id`;
- selecting a PDB `MODEL` by `model_id` for NMR/multi-model files;
- saving C-alpha coordinates as a small NumPy array;
- converting one C-alpha backbone into a `networkx.MultiGraph`;
- rendering a Plotly graph or PyVista tube/silhouette figure.

The adapter does not do:

- open-curve closure;
- knot-type detection;
- Yamada polynomial calculation;
- PD code simplification;
- recursive curve-library changes;
- paper-figure generation.

## Internal Convention

The adapter produces the same graph convention used downstream in the library:

- graph type: `networkx.MultiGraph`;
- node attribute: `pos`, a 3D coordinate with shape `(3,)`;
- edge attribute: `pts`, ordered 3D points with shape `(N, 3)`;
- graph metadata: `pdb_id`, `chain_id`, `input_kind`, `is_closed=False`.
- for multi-model files, `model_id` records the selected PDB model number.

For an open protein chain, the graph currently has two endpoint nodes:

- `start`;
- `end`.

The backbone is represented as one edge with key `curve`.

## Files

- `protein_backbone_adapter.py`  
  Compatibility wrapper around the core `knotted_graph.inputs` API.

- `plot_1crn_backbone.py`  
  1CRN static smoke test. Produces a Matplotlib PNG and a Plotly graph HTML.

- `plot_protein_backbone_pyvista.py`  
  Generic PyVista tube/silhouette renderer for any RCSB PDB ID and chain.

- `plot_1crn_backbone_pyvista.py`  
  Small wrapper around the generic PyVista renderer for 1CRN chain A.

## Example Usage

Run from the repository root:

```bash
PYTHONPATH=src python examples/proteins/plot_1crn_backbone.py
PYTHONPATH=src python examples/proteins/plot_1crn_backbone_pyvista.py
```

Run the generic renderer manually inside the conda environment:

```bash
python examples/proteins/plot_protein_backbone_pyvista.py \
  --pdb-id 1J85 \
  --chain-id A \
  --model-id 1 \
  --output-prefix 1j85_backbone
```

## Current Examples

### 1CRN

- PDB ID: `1CRN`
- Protein: crambin
- Source: RCSB PDB
- Download URL: `https://files.rcsb.org/download/1CRN.pdb`
- Purpose: small standard smoke test for the input adapter.

Expected outputs:

- `data/1CRN.pdb`
- `data/1CRN_ca_coords.npy`
- `figures/1crn_backbone.png`
- `figures/1crn_backbone_graph.html`
- `figures/1crn_backbone_tube.html`
- `figures/1crn_backbone_tube.png`
- `figures/1crn_backbone_tube.svg`

### 1J85

- PDB ID: `1J85`
- Protein: YibK methyltransferase from *Haemophilus influenzae*
- Source: RCSB PDB / KnotProt
- Download URL: `https://files.rcsb.org/download/1J85.pdb`
- Purpose: second smoke test using a known protein-knot example.

Expected outputs:

- `data/1J85.pdb`
- `data/1J85_ca_coords.npy`
- `figures/1j85_backbone_tube.html`
- `figures/1j85_backbone_tube.png`
- `figures/1j85_backbone_tube.svg`

### 4HHB

- PDB ID: `4HHB`
- Protein: hemoglobin
- Source: RCSB PDB
- Download URL: `https://files.rcsb.org/download/4HHB.pdb`
- Purpose: multi-chain PDB smoke test. The renderer can be run on chain `A`.

Expected outputs:

- `data/4HHB.pdb`
- `data/4HHB_ca_coords.npy`
- `figures/4hhb_chainA_backbone_tube.html`
- `figures/4hhb_chainA_backbone_tube.png`
- `figures/4hhb_chainA_backbone_tube.svg`

### 2K39

- PDB ID: `2K39`
- Protein: ubiquitin NMR ensemble
- Source: RCSB PDB
- Download URL: `https://files.rcsb.org/download/2K39.pdb`
- Purpose: NMR/multi-model PDB smoke test. The renderer can be run on chain
  `A`, model `1`.

Expected outputs:

- `data/2K39.pdb`
- `data/2K39_ca_coords.npy`
- `figures/2k39_chainA_model1_backbone_tube.html`
- `figures/2k39_chainA_model1_backbone_tube.png`
- `figures/2k39_chainA_model1_backbone_tube.svg`

## Notes on Closure

Protein C-alpha backbones are open chains. This prototype keeps them open and
sets `is_closed=False`. Closure is intentionally not applied here because this
folder is testing input conversion, not knot classification.

If knot detection is added later, closure should follow standard protein-knot
practice, such as the random-closure approach used by KnotProt, rather than a
new method invented in this prototype.
