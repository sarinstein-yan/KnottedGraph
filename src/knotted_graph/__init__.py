__version__ = "0.0.0"

__description__ = "A package for studying non-Hermitian nodal knotted graph."

__all__ = [
    "NodalKnot",

    "kron_batch",
    "eig_batch",
    "skeleton2graph",
    "skeleton2graph_batch",
    "shift_matrix", 
    "hk2hz",
    "hz2hk",
    "expand_hz_as_hop_dict",

    "planar_diagram_code",
    "find_best_view",

    "optimized_yamada",
    "computeNegami_cached",
    "build_state_graph",
    "igraph_multigraph_key",
    "parse_pd",
    "is_trivalent",
    # "computeNegami", 
    # "computeYamada",
    "BouquetGraph", 
    "ThetaGraph",

    "hop_dict_by_direction",
    "H_batch_from_hop_dict",
    "H_batch",

    "remove_leaf_nodes",

    "plot_3D_and_2D_projections",
    "plot_3D_graph",
    "plot_surface_modes",
]

from poly2graph import (
    kron_batch, 
    eig_batch,
    skeleton2graph,
    skeleton2graph_batch,
    shift_matrix,
    hk2hz, 
    hz2hk, 
    expand_hz_as_hop_dict,
)

from knotted_graph.NodalKnot import NodalKnot

from knotted_graph.pd_codes import (
    planar_diagram_code,
    find_best_view,
)

from knotted_graph.yamada import (
    optimized_yamada,
    computeNegami_cached,
    build_state_graph,
    igraph_multigraph_key,
    parse_pd,
    is_trivalent,
    BouquetGraph,
    ThetaGraph,
)

from knotted_graph.surface_modes import (
    hop_dict_by_direction,
    H_batch_from_hop_dict,
    H_batch,
)

from knotted_graph.util import (
    remove_leaf_nodes,
    remove_deg2_preserving_pts,
    is_PT_symmetric,
    total_edge_pts,
    get_all_pts, 
    get_edge_pts, 
    get_node_pts,
)

from knotted_graph.vis import (
    plot_3D_and_2D_projections,
    plot_3D_graph,
    plot_surface_modes,
)