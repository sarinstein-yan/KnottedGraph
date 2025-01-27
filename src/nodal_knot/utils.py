import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            "3D Curve (k<sub>x</sub>, k<sub>y</sub>, k<sub>z</sub>)",
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
            marker=dict(size=2, color='blue'),
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
        title="Nodal Knot in 3D and 2D Projections"
    )
    
    return fig