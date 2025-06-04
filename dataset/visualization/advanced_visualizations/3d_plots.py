import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_3d_scatter(df, x_col, y_col, z_col, color_col=None):
    """
    Create an interactive 3D scatter plot
    """
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
        opacity=0.7
    )
    return fig

def create_3d_surface(x, y, z, title="3D Surface Plot"):
    """
    Create an interactive 3D surface plot
    """
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title=title)
    return fig

def create_3d_line(df, x_col, y_col, z_col, color_col=None):
    """
    Create an interactive 3D line plot
    """
    fig = px.line_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        title=f"3D Line Plot: {x_col} vs {y_col} vs {z_col}"
    )
    return fig 