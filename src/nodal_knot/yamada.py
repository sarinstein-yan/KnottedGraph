import networkx as nx
import sympy as sp
import itertools

def is_trivalent(G):
    """
    Check if a graph is trivalent -- if all vertices have degree <= 3.
    
    Parameters:
      G : networkx.MultiGraph / networkx.Graph
         The undirected graph to check.
    
    Returns:
      True if the graph is trivalent, False otherwise.
    """
    degs = nx.degree(G)
    return all(degree <= 3 for node, degree in degs)

def computeNegami(G, x, y):
    """
    Compute the Negami polynomial for a graph G.
    Negami polynomial is a bivariate Laurent polynomial:
      h(G; x, y) = sum_{F ⊆ E(G)} (-x)^{|F|} * x^{μ(G-F)} * y^{β(G-F)},
    where:
      - μ(G-F) is the number of connected components of G with the edges in F removed,
      - β(G-F) = |E(G-F)| - |V(G-F)| + μ(G-F).
      
    Parameters:
      G : networkx.MultiGraph
         The graph (assumed undirected) encoded as a NetworkX multigraph.
      x, y : sympy.Symbol
         The symbols for the polynomial.
    
    Returns:
      A sympy.Expr representing h(G; x, y).
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
            
            # Compute the number of connected components
            components = list(nx.connected_components(H))
            mu = len(components)
            
            num_vertices = H.number_of_nodes()
            num_edges = H.number_of_edges()
            
            # β(G-F) = |E(G-F)| - |V(G-F)| + μ(G-F)
            beta = num_edges - num_vertices + mu
            
            term = ((-x)**r) * (x**mu) * (y**beta)
            h_poly += term

    return sp.simplify(h_poly)

def computeYamada(state_list, exponent_list, A):
    """
    Compute the Yamada polynomial for a spatial graph diagram using its resolved states.
    
    Given:
      - state_list: a list of resolved graphs (each a NetworkX MultiGraph) from the state resolutions.
      - exponent_list: a list of integers corresponding to p(s) - m(s) for each state.
      - A: a sympy.Symbol representing the Yamada polynomial variable.
    
    For each resolved graph G in state_list, compute h(G; x, y) using computeNegami,
    then form the term:
         A^(p(s)-m(s)) * h(G; x, y)
    and sum over all states. Finally, substitute:
         x = -1,   y = -A - 2 - A^(-1)
    to obtain the Yamada polynomial Y(D;A) defined by
         Y(D;A) = ∑ₛ A^(p(s)-m(s)) · h(Dₛ; -1, -A-2-A^(-1)).
    
    Parameters:
      state_list : list
         List of resolved graphs (NetworkX MultiGraph / Graph objects).
      exponent_list : list
         List of integers (p(s) - m(s) values) corresponding to each state.
      A : sympy.Symbol
         The symbol for the Yamada polynomial variable.
    
    Returns:
      A sympy.Expr representing the Yamada polynomial Y(D; A).
    """
    # Check that A is a sympy.Symbol
    assert isinstance(A, sp.Symbol), "A must be a sympy.Symbol."
    # Check that the state_list and exponent_list have the same length
    assert len(state_list) == len(exponent_list), "state_list and exponent_list must have the same length."
    # Define the auxiliary symbols x and y used in compute_h.
    x, y = sp.symbols('x y')
    
    total_poly = sp.Integer(0)
    # Sum over each state
    for G, exp in zip(state_list, exponent_list):
        h_val = computeNegami(G, x, y)
        total_poly += (A**exp) * h_val

    # Substitute x = -1 and y = -A - 2 - A^(-1) as per the definition.
    yamada_poly = total_poly.subs(x, -1).subs(y, -A - 2 - A**(-1))
    # Simplify the expression
    yamada_poly = sp.simplify(yamada_poly)
    # Expand the expression
    yamada_poly = sp.expand(yamada_poly)
    return yamada_poly

def BouquetGraph(n):
    """
    Construct the Bouquet_n graph.
    
    Parameters:
      n : int
        The number of petals in the Bouquet_n graph.
    
    Returns:
      A NetworkX MultiGraph representing the Bouquet_n graph.
    """
    edge_list = [(0, 0) for _ in range(n)]
    G = nx.from_edgelist(edge_list, nx.MultiGraph)
    return G

def ThetaGraph(n):
    """
    Construct the Theta_n graph.

    Parameters:
      n : int
        The number of edges in the Theta_n graph.

    Returns:
      A NetworkX MultiGraph representing the Theta_n graph.
    """
    edge_list = [(0, 1) for _ in range(n)]
    G = nx.from_edgelist(edge_list, nx.MultiGraph)
    return G





if __name__ == "__main__":
    
    # Define the variable for the Yamada polynomial
    A = sp.symbols('A')

    # State 1&2: A Bouquet-2
    G1 = BouquetGraph(2)
    # State 3: A Theta-4
    G2 = ThetaGraph(4)

    state_list = [G1, G1, G2]
    exponent_list = [1, -1, 0]

    Yamada = computeYamada(state_list, exponent_list, A)
    print("Yamada polynomial Y(D;A) =")
    sp.pprint(Yamada)