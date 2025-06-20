__version__ = "0.0.0"

__description__ = "A package for working with non-Hermitian nodal knotted graph."

__all__ = [
    "NodalKnot",

    "skeleton2graph",
    "skeleton2graph_batch",

    "PlanarDiagram_Codes",
    "find_best_view"

    "is_trivalent",
    "computeNegami", "computeYamada",
    "BouquetGraph", "ThetaGraph",

    "shift_matrix", "hk2hz",
    "expand_hz_as_hop_dict",
    "hop_dict_by_direction",
    "H_obc_arr_from_hop_dict",
    "H_arr_obc",

    "remove_leaf_nodes",
    "kron_batched",
    "eig_batched",

    "plot_3D_and_2D_projections",
    "plot_3D_graph",
    "plot_surface_modes",
]


from knotted_graph.NodalKnot import NodalKnot

from knotted_graph.skeleton2graph import skeleton2graph, skeleton2graph_batch

from knotted_graph.pd_codes import (
    PlanarDiagram_Codes,
    find_best_view,
)

from knotted_graph.yamada import (
    optimized_yamada,
    computeNegami_cached,
    build_state_graph,
    igraph_multigraph_key,
    parse_pd,
)

from knotted_graph.surface_modes import (
    shift_matrix, hk2hz, 
    expand_hz_as_hop_dict,
    hop_dict_by_direction,
    H_obc_arr_from_hop_dict,
    H_arr_obc,
)

from knotted_graph.util import (
    remove_leaf_nodes,
    get_all_pts, get_edge_pts, get_node_pts,
    kron_batched, eig_batched,
)

from knotted_graph.vis import (
    plot_3D_and_2D_projections,
    plot_3D_graph,
    plot_surface_modes,
)