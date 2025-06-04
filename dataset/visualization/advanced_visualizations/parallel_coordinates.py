import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_parallel_coordinates(df, dimensions, color_col=None):
    """
    Create an interactive parallel coordinates plot
    """
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df[color_col] if color_col else None,
                colorscale='Viridis'
            ),
            dimensions=[
                dict(
                    range=[df[dim].min(), df[dim].max()],
                    label=dim,
                    values=df[dim]
                ) for dim in dimensions
            ]
        )
    )
    
    fig.update_layout(
        title="Parallel Coordinates Plot",
        height=800
    )
    
    return fig

def create_parallel_categories(df, dimensions, color_col=None):
    """
    Create an interactive parallel categories plot
    """
    fig = go.Figure(data=
        go.Parcats(
            dimensions=[
                dict(
                    values=df[dim],
                    label=dim
                ) for dim in dimensions
            ],
            line=dict(
                color=df[color_col] if color_col else None,
                colorscale='Viridis'
            )
        )
    )
    
    fig.update_layout(
        title="Parallel Categories Plot",
        height=800
    )
    
    return fig

def create_parallel_coordinates_with_brush(df, dimensions, color_col=None):
    """
    Create an interactive parallel coordinates plot with brushing capability
    """
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df[color_col] if color_col else None,
                colorscale='Viridis'
            ),
            dimensions=[
                dict(
                    range=[df[dim].min(), df[dim].max()],
                    label=dim,
                    values=df[dim],
                    constraintrange=[df[dim].min(), df[dim].max()]
                ) for dim in dimensions
            ]
        )
    )
    
    fig.update_layout(
        title="Parallel Coordinates Plot with Brushing",
        height=800
    )
    
    return fig 