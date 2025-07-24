import sympy as sp
import networkx as nx
import itertools
from functools import cached_property
from dataclasses import dataclass
from joblib import Parallel, delayed
from .geom import Vertex, Crossing, Arc


__all__ = [
    "compute_negami",
    "compute_yamada_from_states",
    "Yamada",
]


def compute_negami(G: nx.MultiGraph, x: sp.Symbol, y: sp.Symbol) -> sp.Expr:
    """
    Compute the Negami polynomial for a graph G.
    Negami polynomial is a bivariate Laurent polynomial:
      h(G; x, y) = sum_{F ⊆ E(G)} (-x)^{|F|} * x^{μ(G-F)} * y^{β(G-F)},
    where:
      - μ(G-F) is the number of connected components of G with the edges in F removed,
      - β(G-F) = |E(G-F)| - |V(G-F)| + μ(G-F).
      
    Parameters:
    ----------
      G : networkx.MultiGraph
         The graph (assumed undirected) encoded as a NetworkX multigraph.
      x, y : sympy.Symbol
         The symbols for the polynomial.
    
    Returns:
    -------
    sp.Expr
        A sympy expression representing the Negami polynomial h(G; x, y).
    """
    # List all edges with keys (each edge is (u, v, key))
    edges = list(G.edges(keys=True))
    h_poly = sp.Integer(0)  # initialize polynomial to 0

    # Iterate over all subsets F of edges.
    for r in range(len(edges) + 1):
        for F in itertools.combinations(edges, r):
            H = G.copy()
            for (u, v, key) in F:
                H.remove_edge(u, v, key=key)
            
            mu = nx.number_connected_components(H)
            num_vertices = H.number_of_nodes()
            num_edges = H.number_of_edges()

            # β(G-F) = |E(G-F)| - |V(G-F)| + μ(G-F)
            beta = num_edges - num_vertices + mu
            
            h_poly += ((-x)**r) * (x**mu) * (y**beta)
    
    return sp.expand(h_poly)


def compute_yamada_from_states(
    state_graphs: list[nx.MultiGraph],
    exponents: list[int],
    A: sp.Symbol,
    normalize: bool = True,
    n_jobs: int = -1
) -> sp.Expr:
    """
    Compute the Yamada polynomial for a spatial graph diagram using its resolved states.

    The Yamada polynomial is computed as:
        Y(D; A) = ∑ₛ A^(p(s)-m(s)) · h(Dₛ; -1, -A-2-A^(-1)),
    where:
        - D is the spatial graph diagram,
        - Dₛ is the resolved state graph,
        - p(s) and m(s) are integers associated with the state,
        - h(G; x, y) is the Negami polynomial of the graph G.

    The computation involves:
        1. Calculating the Negami polynomial h(G; x, y) for each resolved state graph.
        2. Forming the term A^(p(s)-m(s)) * h(G; x, y) for each state.
        3. Substituting x = -1 and y = -A - 2 - A^(-1) into the polynomial.
        4. Optionally normalizing the polynomial by adjusting the lowest exponent of A to 0.

    Parameters:
    ---------
        state_graphs : list[nx.MultiGraph]
            List of resolved graphs (NetworkX MultiGraph objects) from the state resolutions.
        exponents : list[int]
            List of integers corresponding to p(s) - m(s) values for each state.
        A : sp.Symbol
            The symbol for the Yamada polynomial variable.
        normalize : bool, optional
            Whether to normalize the polynomial so that the lowest exponent of A is 0. Default is True.
        n_jobs : int, optional
            The number of parallel jobs to use for computing the Negami polynomials. Default is -1 (use all available cores).

    Returns:
        sp.Expr
            A sympy.Expr representing the Yamada polynomial Y(D; A).
    """
    x, y = sp.symbols('x y')
    
    # 1) launch all computeNegami in parallel threads
    h_values = Parallel(
        n_jobs=n_jobs,
        prefer='threads',
    )(
        delayed(compute_negami)(G, x, y)
        for G in state_graphs
    )
    
    # 2) form the total sum
    total_poly = sp.Integer(0)
    for h_val, exp in zip(h_values, exponents):
        total_poly += (A**exp) * h_val

    # 3) substitute and simplify
    Y = total_poly.xreplace({x: -1, y: -A-2-A**(-1)})
    Y = sp.expand(sp.simplify(Y))

    # 4) optionally normalize
    if normalize:
        terms = Y.as_ordered_terms()
        lowest_exp = min(t.as_coeff_exponent(A)[1] for t in terms)
        Y = Y * (-A)**(-lowest_exp)
        Y = sp.expand(sp.simplify(Y))

    return Y


@dataclass
class Yamada():
    vertices: list[Vertex]
    crossings: list[Crossing]
    arcs: list[Arc]

    @cached_property
    def _state_graph_base(self):
        G = nx.MultiGraph()
        num_v, num_x = len(self.vertices), len(self.crossings)
        G.add_nodes_from(range(num_v), type='v')
        G.add_nodes_from(range(num_v, num_v + num_x), type='x')

        arc_nodes = {}
        for arc in self.arcs:
            u = arc.start_id if arc.start_type == 'v' else arc.start_id + num_v
            v = arc.end_id if arc.end_type == 'v' else arc.end_id + num_v
            arc_nodes[arc.id] = (u, v)
            G.add_edge(u, v, id=arc.id)
        
        return G, arc_nodes

    def _build_state_graphs(self):
        G_base, arc_nodes = self._state_graph_base
        num_v, num_x = len(self.vertices), len(self.crossings)

        configurations = list(itertools.product([0, 1, 2], repeat=num_x))
        exponents = [s.count(0) - s.count(1) for s in configurations]
        state_graphs = []

        for config in configurations:
            G = G_base.copy()
            for x, spin in zip(self.crossings, config):
                x_node = x.id + num_v
                arc_ids = x.ccw_ordered_arcs

                if not arc_ids:
                    G.remove_node(x_node)
                    continue

                nbrs = [v if u == x_node else u 
                        for u, v in (arc_nodes[i] for i in arc_ids)]
                if spin == 0:
                    G.add_edge(nbrs[0], nbrs[3])
                    G.add_edge(nbrs[1], nbrs[2])
                    G.remove_node(x_node)
                elif spin == 1:
                    G.add_edge(nbrs[0], nbrs[2])
                    G.add_edge(nbrs[1], nbrs[3])
                    G.remove_node(x_node)
                elif spin == 2:
                    G.nodes[x_node]['type'] = 'x'
                else:
                    raise ValueError(f"Invalid spin configuration: {spin}")
            state_graphs.append(G)
        return state_graphs, exponents

    def compute(
            self, 
            variable: sp.Symbol,
            normalize: bool = True,
            n_jobs: int = -1
    ) -> sp.Expr:
        """
        Compute the Yamada polynomial for the knot diagram.
        """
        state_graphs, exponents = self._build_state_graphs()
        return compute_yamada_from_states(
            state_graphs, exponents, variable, normalize=normalize, n_jobs=n_jobs
        )