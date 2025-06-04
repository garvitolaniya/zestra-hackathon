import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata

def create_contour_plot(x, y, z, title="Contour Plot"):
    """
    Create an interactive contour plot
    """
    fig = go.Figure(data=
        go.Contour(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis',
            contours=dict(
                start=z.min(),
                end=z.max(),
                size=(z.max()-z.min())/20
            )
        )
    )
    fig.update_layout(title=title)
    return fig

def create_filled_contour(x, y, z, title="Filled Contour Plot"):
    """
    Create an interactive filled contour plot
    """
    fig = go.Figure(data=
        go.Contour(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis',
            contours=dict(
                start=z.min(),
                end=z.max(),
                size=(z.max()-z.min())/20
            ),
            contours_coloring='fill'
        )
    )
    fig.update_layout(title=title)
    return fig

def create_contour_with_scatter(x, y, z, scatter_x, scatter_y, title="Contour Plot with Scatter"):
    """
    Create a contour plot with scatter points overlay
    """
    fig = go.Figure()
    
    # Add contour
    fig.add_trace(go.Contour(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        showscale=True
    ))
    
    # Add scatter
    fig.add_trace(go.Scatter(
        x=scatter_x,
        y=scatter_y,
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            symbol='circle'
        ),
        name='Data Points'
    ))
    
    fig.update_layout(title=title)
    return fig 