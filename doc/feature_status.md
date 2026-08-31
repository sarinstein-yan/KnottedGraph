# Feature Status And Workflow Routes

KnottedGraph is currently **pre-alpha**. The status labels on this page describe
where a capability is available and how it is installed; they are not a
semantic-versioning or long-term API-stability guarantee.

Use this page to choose a supported starting point before opening an application
notebook or API page. Quantitative runtimes are stated only where they have been
measured. The final column otherwise records the dominant scaling variable.

Status legend:

- **Public** means a user-facing generic package entry point.
- **Application API** means a domain-specific workflow assembled on the generic
  core.
- **External backend** means the Python entry point is present, but execution
  depends on separately installed native software.
- **base** is available without an optional Python extra; **optional** names the
  required extra in the next column.

| Starting object or goal | Status | Required extra | Public call | Return object | Next step | Runtime / scaling |
| --- | --- | --- | --- | --- | --- | --- |
| Ordered coordinate array or CSV/DAT/JSON/NPY/TSV/TXT/XYZ | Public · base | none | `knotted_graph.inputs.from_coordinate_chain` | `CoordinateInputResult` | Inspect `.issues` and the closure choice, then use `.graph`. | Depends on the number of coordinate samples. |
| PDB ID or local PDB backbone | Public · base | none | `knotted_graph.inputs.from_pdb_backbone`; protein and nucleic-acid shortcuts are also available | `PDBBackboneInputResult` | Select chain/model and closure explicitly, inspect `.issues`, then use `.graph`. | Depends on record size; remote IDs also depend on network and cache state. |
| RCSB ID or local CIF/mmCIF atom trace | Public · base | none | `knotted_graph.inputs.from_mmcif_backbone` | `MMCIFBackboneInputResult` | Select chain/model and closure explicitly, inspect `.issues`, then use `.graph`. | Depends on record size; remote IDs also depend on network and cache state. |
| GROMACS GRO snapshot | Public · base | none | `knotted_graph.inputs.from_gromacs_gro` | `PolymerInputResult` | Check units, filters, and closure, then use `.graph`. | Depends on atom count. |
| First frame of a LAMMPS dump | Public · base | none | `knotted_graph.inputs.from_lammps_dump` | `PolymerInputResult` | Check molecule selection, ordering, and closure, then use `.graph`. | Depends on the selected atom count in the first frame. |
| Paired node/edge CSV spatial graph | Public · base | none | `knotted_graph.inputs.from_spatial_graph_csv` | `SpatialGraphInputResult` | Inspect `.issues`, then normalize or project `.graph`. | Depends on node, edge, and polyline-point counts. |
| Named knot/link, torus type, or Artin braid | Public · base | none | `knotted_graph.inputs.KnotFunction` | `KnotFunction` | Inspect the catalogue or braid construction report, then sample or diagnose a chosen level. | Symbolic construction depends on braid length and interpolation choices; no 3-D grid is sampled yet. |
| Analytic knot field needing a tubular handlebody or spatial graph | Public · optional | `knot-fields` | `knotted_graph.inputs.KnotFunction.to_spatial_graph` | `networkx.MultiGraph` | Run level diagnostics and a resolution check before using the extracted graph. | Field sampling grows with `dimension**3`; graph extraction also depends on level-set complexity. |
| OBJ/OFF/PLY/STL/VTK/VTP surface mesh | Public · optional | `surface` | `knotted_graph.inputs.from_surface_mesh` | `SurfaceInputResult` | Inspect `.mesh` and `.issues`, then choose an application-specific extraction route. | Depends on mesh size and PyVista operations. |
| Embedded `networkx.MultiGraph` with node `pos` and edge `pts` data | Public · base | none | `knotted_graph.core.ensure_embedding` | `networkx.MultiGraph` | Continue to projection after validation succeeds. | Depends on graph and sampled-polyline size. |
| Validated embedded graph needing an inspectable projection | Public · base | none | `knotted_graph.projection.select_projection` | `ProjectionResult` | Inspect the selected angles, crossing count, and PD code. | Candidate-view count multiplied by geometric intersection work. |
| Embedded graph needing its Yamada invariant and provenance | Public · base | none | `knotted_graph.projection.compute_yamada_polynomial(..., return_result=True)` | `YamadaComputationResult` | Retain both `.polynomial` and `.projection`. | Projection cost plus exponential crossing-state growth, approximately $3^c$. |
| Abstract undirected Graph/MultiGraph without crossing data | Public · base | none | `knotted_graph.invariants.yamada.compute_graph_yamada_polynomial` | `sympy.Expr` | Inspect or expand the Laurent polynomial. | Depends on graph topology and edge count. |
| Named structured mathematical graph family | Application API · base | none | `knotted_graph.applications.mathematical.build_graph_case` | `(networkx.MultiGraph, dict)` | Use the graph for direct invariant evaluation; use the position dictionary only for display. | Depends on generated graph size. |
| Deformation between two analytic knot fields | Application API · optional | `knot-fields` | `knotted_graph.applications.knot_deformation.KnotDeformationScan` | `KnotDeformationScanResult` | Inspect phase signatures, errors, and transition intervals across the sampled grid. | Number of lambda/radius cells multiplied by `dimension**3` field extraction and any requested invariant cost. |
| Already skeletonized 3-D boolean image | Public · optional | `nodal` | `knotted_graph.extraction.skeleton_image_to_graph` | `networkx.MultiGraph` | Validate and simplify the graph before projection. | Depends on voxel count and skeleton topology. |
| In-memory two-band non-Hermitian Hamiltonian or Bloch vector | Application API · optional | `nodal` | `knotted_graph.applications.nodal.NodalSkeleton` | `NodalSkeleton` | Inspect the sampled surface/mask, then call `.skeleton_graph()`. | Grid work grows with `dimension**3`. |
| In-memory Hermitian multiband Hamiltonian | Application API · optional | `nodal` | `knotted_graph.applications.materials.MaterialFermiSurface` | `MaterialFermiSurface` | Inspect the gap surface, then call `.skeleton_graph()`. | Depends on grid size and band count. |
| Knot-field or in-memory Hamiltonian Yamada phase map | Application API · optional | `knot-fields` for knot sources; `nodal` for Hamiltonian/material sources | `knotted_graph.applications.make_yamada_phase_map` | `YamadaPhaseMapResult` | Inspect cell errors, phase signatures, and transition intervals; recompute selected cells only when needed. | Number of phase-map cells multiplied by extraction and Yamada-evaluation cost. |
| Embedded graph needing repulsive relaxation | External backend | none for the direct graph call; separately installed native Repulsor solver | `knotted_graph.layout.repulsive.relax_spatial_graph` | `GraphLayoutResult` | Validate `result.graph` before projection. | Depends on native build and solver options; native runtime has not been independently audited. |
| Embedded graph needing an interactive 3-D view | Public · optional | `viz` | `knotted_graph.visualization.plot_3D_graph_plotly` | `plotly.graph_objects.Figure` | Display or export the returned figure. | Depends on graph/polyline size and the renderer. |

## Format and workflow boundaries

The table is a support matrix; input-gallery figures are not. The following
families are **not currently generic public file adapters**:

- GraphML;
- generic edge lists;
- SWC;
- spatial-graph JSON;
- NPZ scalar/vector fields or volumes;
- Hamiltonian files; and
- oriented-flow files.

JSON and NPY in the first table row mean ordered coordinate-chain data, not an
arbitrary serialized graph. The LAMMPS adapter reads the first frame and expects
unscaled `x/y/z` coordinates. The mmCIF adapter supports the documented
RCSB-style `_atom_site` subset with one complete data row per physical line.
Hamiltonian application workflows accept in-memory symbolic objects rather
than files. Named knots, torus types, and Artin braid words are public analytic
constructors; this does not imply a generic serialized-knot or braid-file
reader.

The surface loader returns `PyVista.PolyData`; it does not perform a generic
mesh-to-graph conversion. Likewise, application examples may contain GraphML,
SWC, NPZ, or Hamiltonian-derived objects after an external or
application-specific conversion step. Their presence in a figure does not add a
public parser to `knotted_graph.inputs`.

## Continue from here

- Follow the {doc}`installation` guide to install the required extra.
- Run the deterministic {doc}`quickstart` for a base-install smoke test.
- Use the {doc}`user_guide/index` to choose a tutorial.
- Open the {doc}`api/index` after selecting the relevant public entry point.
