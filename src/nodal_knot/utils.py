import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from typing import Union

from plotly.express import colors
tableau_colors = colors.qualitative.T10

def remove_leaf_nodes(G: Union[nx.Graph, nx.MultiGraph]) -> Union[nx.Graph, nx.MultiGraph]:
    """
    Remove all leaf nodes (nodes with degree 1) and their incident edges from the graph.
    
    This function creates a copy of the input graph and then iteratively removes any node
    that has degree 1. Removing a node automatically removes its incident edge(s).
    The process repeats until no leaf nodes remain.
    
    Parameters:
        G : nx.Graph or nx.MultiGraph
            The input graph.
            
    Returns:
        H : nx.Graph or nx.MultiGraph
            A new graph with all leaf nodes (and their incident edges) removed.
    """
    H = G.copy()
    while True:
        # Identify all leaf nodes (nodes with degree exactly 1)
        leaf_nodes = [node for node, degree in H.degree() if degree == 1]
        if not leaf_nodes:
            break  # Exit when there are no leaf nodes left.
        for node in leaf_nodes:
            H.remove_node(node)
    return H


def plot_3D_and_2D_projections(points):
    """
    Plot the zero regions in 3D space and their 2D projections.

    Parameters:
    ----------
    points : Array-like
        Points in 3D (kx, ky, kz) space to plot.

    Returns:
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object for visualization.
    """
    # Separate coordinates for convenience
    kx_vals = [p[0] for p in points]
    ky_vals = [p[1] for p in points]
    kz_vals = [p[2] for p in points]
    
    # Build subplots with 1 scene (3D) and 3 cartesian 2D
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "3D Knot's Region (k<sub>x</sub>, k<sub>y</sub>, k<sub>z</sub>)",
            "Projection: k<sub>x</sub> vs k<sub>y</sub>",
            "Projection: k<sub>x</sub> vs k<sub>z</sub>",
            "Projection: k<sub>y</sub> vs k<sub>z</sub>"
        ],
        specs=[
            [{"type":"scene"}, {"type":"xy"}],
            [{"type":"xy"}, {"type":"xy"}]
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.1
    )
    
    # 3D scatter
    fig.add_trace(
        go.Scatter3d(
            x=kx_vals, 
            y=ky_vals, 
            z=kz_vals,
            mode='markers',
            marker=dict(size=2, 
                        color=tableau_colors[0],
                        opacity=.5),
            name='3D Points'
        ),
        row=1, col=1
    )
    
    # Projection: kx vs ky
    fig.add_trace(
        go.Scatter(
            x=kx_vals, 
            y=ky_vals,
            mode='markers',
            marker=dict(size=4, color=tableau_colors[1]),
            name='k<sub>x</sub> vs k<sub>y</sub>'
        ),
        row=1, col=2
    )
    
    # Projection: kx vs kz
    fig.add_trace(
        go.Scatter(
            x=kx_vals, 
            y=kz_vals,
            mode='markers',
            marker=dict(size=4, color=tableau_colors[2]),
            name='k<sub>x</sub> vs k<sub>z</sub>'
        ),
        row=2, col=1
    )
    
    # Projection: ky vs kz
    fig.add_trace(
        go.Scatter(
            x=ky_vals, 
            y=kz_vals,
            mode='markers',
            marker=dict(size=4, color=tableau_colors[3]),
            name='k<sub>y</sub> vs k<sub>z</sub>'
        ),
        row=2, col=2
    )
    
    # Update 3D axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title='k<sub>x</sub>',
            yaxis_title='k<sub>y</sub>',
            zaxis_title='k<sub>z</sub>'
        )
    )
    
    # Update 2D axis labels
    fig.update_xaxes(title_text='k<sub>x</sub>', row=1, col=2)
    fig.update_yaxes(title_text='k<sub>y</sub>', row=1, col=2)
    
    fig.update_xaxes(title_text='k<sub>x</sub>', row=2, col=1)
    fig.update_yaxes(title_text='k<sub>z</sub>', row=2, col=1)
    
    fig.update_xaxes(title_text='k<sub>y</sub>', row=2, col=2)
    fig.update_yaxes(title_text='k<sub>z</sub>', row=2, col=2)
    
    fig.update_layout(
        height=800, 
        width=1000,
        title="Thickened Nodal Knot"
    )
    
    return fig


def plot_3D_graph(G: Union[nx.Graph, nx.MultiGraph]) -> go.Figure:
    """
    Create a 3D Plotly visualization of a knotted graph.
    
    This function extracts edge and node data from the graph and creates an interactive 
    3D plot where:
      - Edges are displayed as blue lines.
      - Nodes (with degree ≠ 2) are displayed as red markers.
    
    Parameters:
    -----------
    G : nx.Graph
        The input graph with edge attribute 'pts' containing a NumPy array of shape (N, 3)
        representing the coordinates along each edge, and node attribute 'o' representing the
        node's 3D position.
        
    Returns:
    --------
    fig : go.Figure
        The Plotly figure object for the interactive 3D visualization.
    """
    # Create a list to hold Plotly traces for edges.
    edge_traces = []
    for u, v, data in G.edges(data=True):
        pts = data.get('pts')
        if pts is not None:
            if pts.ndim == 2 and pts.shape[1] == 3:
                # Extract x, y, z coordinates for the edge.
                x = pts[:, 0]
                y = pts[:, 1]
                z = pts[:, 2]
                trace = go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode='lines',
                    line=dict(color=tableau_colors[0], width=2),
                    hoverinfo='none',  # disable hover text
                    showlegend=False   # disable legend entry
                )
                edge_traces.append(trace)
            else:
                print(f"Edge {u}-{v} 'pts' data is not of shape (N, 3).")
        else:
            print(f"Edge {u}-{v} has no 'pts' attribute.")

    # Create a trace for nodes as red points, but only for nodes whose degree is not 2.
    node_positions = {}
    for n in G.nodes():
        if G.degree(n) != 2:
            data = G.nodes[n]
            if 'o' in data:
                node_positions[n] = data['o']

    if node_positions:
        xs = [coord[0] for coord in node_positions.values()]
        ys = [coord[1] for coord in node_positions.values()]
        zs = [coord[2] for coord in node_positions.values()]
        node_trace = go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers',
            marker=dict(size=5, color=tableau_colors[1]),
            hoverinfo='none',  # disable hover text
            showlegend=False   # disable legend entry
        )
    else:
        node_trace = None

    # Combine the traces into one Plotly figure.
    data_traces = edge_traces + ([node_trace] if node_trace is not None else [])
    fig = go.Figure(data=data_traces)

    # Update the layout for better viewing and disable the overall legend.
    fig.update_layout(
        title="Interactive 3D Graph Visualization<br>(Nodes with degree ≠ 2)",
        scene=dict(
            xaxis_title='k<sub>x</sub>',
            yaxis_title='k<sub>y</sub>',
            zaxis_title='k<sub>z</sub>'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,  # disable legend in the layout
        width=450,
        height=450
    )
    
    return fig