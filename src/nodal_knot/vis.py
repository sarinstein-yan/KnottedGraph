import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from typing import Union


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
            marker=dict(size=2, color='blue', opacity=.5),
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
            marker=dict(size=4, color='red'),
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
            marker=dict(size=4, color='green'),
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
            marker=dict(size=4, color='purple'),
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
    edge_color = 'blue'  # single color for all edges
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
                    line=dict(color=edge_color, width=2),
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
            marker=dict(size=5, color='red'),
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



def plot_surface_modes(eigvals_tuple, k_vals_tuple, Etol_tuple, nH_coeff):
    """
    Generates a single figure with three subplots (in a row) showing surface modes
    for different open boundary conditions.

    Parameters
    ----------
    eigvals_tuple : tuple of np.ndarray
        A tuple containing the eigenvalue arrays: (eigvals_obc_x, eigvals_obc_y, eigvals_obc_z).
        Each array is assumed to have shape (..., num_eigenvalues), where the last
        two dimensions correspond to the k-grid.
    k_vals_tuple : tuple of array-like
        A tuple containing the momentum values (kx_vals, ky_vals, kz_vals).
    Etol_tuple : tuple of float
        A tuple of energy tolerances: (Etol_x, Etol_y, Etol_z).
    nH_coeff : float or str
        A coefficient (or label) to be displayed in the title of each plot.

    Returns
    -------
    None
        Displays the figure with three subplots.
    """
    # Unpack inputs
    eigvals_obc_x, eigvals_obc_y, eigvals_obc_z = eigvals_tuple
    kx_vals, ky_vals, kz_vals = k_vals_tuple
    Etol_x, Etol_y, Etol_z = Etol_tuple

    # Create figure with three subplots arranged horizontally
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ------------------ Plot 1: OBC in x-direction ------------------ #
    # Here we use the ky and kz values.
    surface_modes_x = np.sum(np.abs(eigvals_obc_x) < Etol_x, axis=-1)
    sm_ky, sm_kz, sm_num = [], [], []
    for i, ky in enumerate(ky_vals):
        for j, kz in enumerate(kz_vals):
            sm = surface_modes_x[i, j]
            if sm > 0:
                sm_ky.append(ky)
                sm_kz.append(kz)
                sm_num.append(sm)
    ax = axes[0]
    sc = ax.scatter(sm_ky, sm_kz, s=5, 
                    c='k')
                    # c=sm_num, cmap='gray')
    ax.set_xlabel('$k_y$')
    ax.set_ylabel('$k_z$')
    ax.set_title(f'OBC in $x$-direction\n|E| tolerance = {Etol_x};  non-Hermitian strength: {nH_coeff}')

    # ------------------ Plot 2: OBC in y-direction ------------------ #
    # Here we use the kx and kz values.
    surface_modes_y = np.sum(np.abs(eigvals_obc_y) < Etol_y, axis=-1)
    sm_kx, sm_kz, sm_num = [], [], []
    for i, kz in enumerate(kz_vals):
        for j, kx in enumerate(kx_vals):
            sm = surface_modes_y[i, j]
            if sm > 0:
                sm_kx.append(kx)
                sm_kz.append(kz)
                sm_num.append(sm)
    ax = axes[1]
    sc = ax.scatter(sm_kx, sm_kz, s=5, 
                    c='k')
                    # c=sm_num, cmap='gray')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_z$')
    ax.set_title(f'OBC in $y$-direction\n|E| tolerance = {Etol_y};  non-Hermitian strength: {nH_coeff}')

    # ------------------ Plot 3: OBC in z-direction ------------------ #
    # Here we use the kx and ky values.
    surface_modes_z = np.sum(np.abs(eigvals_obc_z) < Etol_z, axis=-1)
    sm_kx, sm_ky, sm_num = [], [], []
    for i, kx in enumerate(kx_vals):
        for j, ky in enumerate(ky_vals):
            sm = surface_modes_z[i, j]
            if sm > 0:
                sm_kx.append(kx)
                sm_ky.append(ky)
                sm_num.append(sm)
    ax = axes[2]
    sc = ax.scatter(sm_kx, sm_ky, s=5, 
                    c='k')
                    # c=sm_num, cmap='gray')
    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    # Set axis limits if desired (adjust as needed)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f'OBC in $z$-direction\n|E| tolerance = {Etol_z};  non-Hermitian strength: {nH_coeff}')

    plt.tight_layout()
    return fig