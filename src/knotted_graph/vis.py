import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Union

__all__ = [
    "standard_petersen_layout",
    "draw_petersen_embedding",
    "plot_3D_and_projections_plotly",
    "plot_3D_graph_plotly",
    "plot_surface_modes",
]


def plot_3D_and_projections_plotly(points):
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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


def plot_3D_graph_plotly(G: Union[nx.Graph, nx.MultiGraph]) -> "go.Figure":
    """
    Robust 3D Plotly visualizer for a knotted graph.

    - Flattens mixed Python lists/arrays of edge 'pts' into a clean (N,3) array.
    - Plots edges as blue lines and **all** nodes as red markers.
    """
    import plotly.graph_objects as go
    edge_traces = []
    edge_color = 'blue'

    # — edge traces (unchanged) —
    for u, v, data in G.edges(data=True):
        raw_pts = data.get('pts', [])
        flat_pts = []
        for p in raw_pts:
            arr = np.asarray(p)
            if arr.ndim == 1 and arr.shape == (3,):
                flat_pts.append(arr)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                flat_pts.extend(arr)
            else:
                continue

        if len(flat_pts) < 2:
            continue

        pts_arr = np.stack(flat_pts)
        x, y, z = pts_arr[:,0], pts_arr[:,1], pts_arr[:,2]
        edge_traces.append(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=edge_color, width=2),
                hoverinfo='none',
                showlegend=False
            )
        )

    # — node traces: now include every node with an 'pos' attribute —
    node_x, node_y, node_z = [], [], []
    for n, data in G.nodes(data=True):
        if 'pos' in data:
            o = data['pos']
            if len(o) == 3:
                node_x.append(o[0])
                node_y.append(o[1])
                node_z.append(o[2])

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=5, color='red'),
        hoverinfo='none',
        showlegend=False
    ) if node_x else None

    # — combine and layout —
    traces = edge_traces + ([node_trace] if node_trace else [])
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Interactive 3D Graph Visualization<br>(All Nodes)",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        width=600,
        height=600
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


def standard_petersen_layout(R_outer=1.2, R_inner=0.7):
    coords = {}
    for i in range(5):
        angle = 2*math.pi*i/5
        coords[i] = (R_outer*math.cos(angle), R_outer*math.sin(angle))
    for i in range(5):
        angle = 2*math.pi*(i+0.5)/5
        coords[i+5] = (R_inner*math.cos(angle), R_inner*math.sin(angle))
    return coords


def draw_petersen_embedding(
        petersen_graph, 
        embedding,
        layout_func=standard_petersen_layout,
        box_size=(0.5,0.3),
        scale_factor=1.0,
        outer_color='lightblue',
        inner_color='lightcoral',
        edge_color='k', edge_width=3,
        circle_edgecolor='black',
        circle_facecolor='white',
        circle_lw=2,
        text_kwargs=None
    ):
    if text_kwargs is None:
        text_kwargs = dict(ha='center', va='center',
                           fontsize=10, fontweight='bold', zorder=11)

    positions = layout_func()
    fig, ax = plt.subplots(figsize=(8,8))

    # draw edges
    for u, v in petersen_graph.edges():
        pu = np.array(positions[u])*scale_factor
        pv = np.array(positions[v])*scale_factor
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                color=edge_color, lw=edge_width, zorder=0)

    bx, by = box_size
    for minor, (xc, yc) in positions.items():
        xc, yc = xc*scale_factor, yc*scale_factor
        x0, y0 = xc-bx/2, yc-by/2
        color = outer_color if minor < 5 else inner_color
        ax.add_patch(patches.Rectangle((x0,y0), bx, by,
                                       edgecolor='black',
                                       facecolor=color,
                                       lw=2, zorder=1))

        hosts = embedding.get(minor, [])
        n = len(hosts)
        if not n:
            continue

        r = by/2.5
        xs = [xc] if n==1 else np.linspace(x0+r, x0+bx-r, n)
        ys = [yc]*n

        for xi, yi, h in zip(xs, ys, hosts):
            ax.add_patch(patches.Circle((xi, yi), radius=r,
                                        edgecolor=circle_edgecolor,
                                        facecolor=circle_facecolor,
                                        lw=circle_lw, zorder=10))
            ax.text(xi, yi, str(h), **text_kwargs)

    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax
