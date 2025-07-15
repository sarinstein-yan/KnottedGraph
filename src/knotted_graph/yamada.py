import networkx as nx
import sympy as sp
import itertools
import re
from collections import defaultdict
from functools import lru_cache
import igraph as ig

def igraph_multigraph_key(G):
    node_list = list(G.nodes())
    idx       = {n:i for i,n in enumerate(node_list)}
    edges     = [(idx[u], idx[v]) for u,v,_ in G.edges(keys=True)]
    igG       = ig.Graph(n=len(node_list), edges=edges, directed=False)

    # everyone the same color → Nauty still collapses isomorphic connectivity
    igG.vs['color'] = [0]*len(node_list)

    perm = igG.canonical_permutation(color=igG.vs['color'])
    canon = igG.permute_vertices(perm)
    return tuple(sorted(canon.get_edgelist()))


def parse_pd(pd_str):
    vertices, crossings = [], []
    for token in pd_str.strip().split(';'):
        token = token.strip()
        if not token:
            continue
        m = re.match(r'^(V|X)\[\s*([\d,]+)\s*\]$', token)
        if not m:
            raise ValueError(f"Bad token: {token!r}")
        kind, nums = m.groups()
        labels = list(map(int, nums.split(',')))
        if kind == 'V':
            vertices.append(labels)
        else:
            crossings.append(labels)
    return vertices, crossings

def build_state_graph(vertices, crossings, state):
    G = nx.MultiGraph()
    nV, nX = len(vertices), len(crossings)
    G.add_nodes_from(range(nV),   kind='V')
    G.add_nodes_from(range(nV, nV+nX), kind='X')
    label_ends = defaultdict(list)
    for vidx, v_lbls in enumerate(vertices):
        for lbl in v_lbls:
            label_ends[lbl].append(vidx)
    for xidx, (u1,u2,o1,o2) in enumerate(crossings):
        Xnode = nV + xidx
        for lbl in (u1,u2,o1,o2):
            label_ends[lbl].append(Xnode)
    for lbl, ends in label_ends.items():
        if len(ends) != 2:
            raise ValueError(f"Label {lbl} appears {len(ends)} times; expected 2")
        G.add_edge(*ends, label=lbl)
    for xidx, res in enumerate(state):
        Xnode = nV + xidx
        u1,u2,o1,o2 = crossings[xidx]
        nbr = {}
        for lbl in (u1,u2,o1,o2):
            a,b = label_ends[lbl]
            nbr[lbl] = b if a==Xnode else a
        if res == 0:
            G.add_edge(nbr[u1], nbr[o1])
            G.add_edge(nbr[u2], nbr[o2])
            G.remove_node(Xnode)
        elif res == 1:
            G.add_edge(nbr[u1], nbr[o2])
            G.add_edge(nbr[u2], nbr[o1])
            G.remove_node(Xnode)
        elif res == 2:
            G.nodes[Xnode]['kind'] = 'V'
        else:
            raise ValueError(f"Invalid state {res}")
    return G

def graph_key(G):
    # hashable summary: sorted list of (u,v,key)
    eds = []
    for u,v,k in G.edges(keys=True):
        a,b = (u,v) if u<=v else (v,u)
        eds.append((a,b,k))
    return tuple(sorted(eds))

# global store so our cached function can recover G
_graph_store = {}

x, y = sp.symbols('x y')
@lru_cache(maxsize=None)
def computeNegami_cached(key):
    G = _graph_store[key]
    h = sp.Integer(0)
    edges = list(G.edges(keys=True))
    for r in range(len(edges)+1):
        for F in itertools.combinations(edges, r):
            H = G.copy()
            for u,v,k in F:
                H.remove_edge(u, v, key=k)
            mu = nx.number_connected_components(H)
            V  = H.number_of_nodes()
            E  = H.number_of_edges()
            beta = E - V + mu
            h += ((-x)**r) * (x**mu) * (y**beta)
    return sp.simplify(h)

# ——— Main wrapper ———
def optimized_yamada(pd_code: str):
    # 1) parse once
    vertices, crossings = parse_pd(pd_code)
    # freeze them into tuples so we can use them in an lru_cache
    vert_t  = tuple(tuple(v) for v in vertices)
    cross_t = tuple(tuple(x) for x in crossings)

    # 2) cached constructor: key is the full state‐tuple
    @lru_cache(maxsize=None)
    def _build(state):
        # state is a tuple of 0/1/2
        return build_state_graph(vert_t, cross_t, list(state))

    # 3) enumerate states, but now use the cached builder
    configs = list(itertools.product([0,1,2], repeat=len(cross_t)))
    key_exps = []
    for cfg in configs:
        state = tuple(cfg)
        G = _build(state)                # will only build each unique state once
        k = igraph_multigraph_key(G)

        if k not in _graph_store:
            _graph_store[k] = G
        exp = cfg.count(0) - cfg.count(1)
        key_exps.append((k, exp))

    # 4) assemble and finish exactly as before…
    A = sp.symbols('A')
    total = sp.Integer(0)
    for key, group in itertools.groupby(sorted(key_exps, key=lambda t: t[0]), key=lambda t: t[0]):
        P = computeNegami_cached(key)
        for _, exp in group:
            total += (A**exp) * P

    Y = total.subs({x: -1, y: -A-2-A**(-1)})
    Y = sp.simplify(sp.expand(Y))
    m = min(t.as_coeff_exponent(A)[1] for t in Y.as_ordered_terms())
    return sp.expand(Y * A**(-m))



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