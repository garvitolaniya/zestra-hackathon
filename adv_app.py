from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from sklearn.ensemble import RandomForestRegressor
# Temporarily import DecisionTreeClassifier for decision boundary plot example
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
import json
import base64
import io
import matplotlib.pyplot as plt
import shap
import graphviz # For decision tree visualization
from sklearn.inspection import partial_dependence
# Remove sys.path manipulation as we are consolidating code
# Add the directory containing the 'dataset' package to the Python path (if needed for dataset loading)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = current_file_dir # Assuming adv_app.py is at the project root
sys.path.append(project_root_dir)
app = Flask(__name__)
# Get the correct path to train.csv
train_csv_path = os.path.join(project_root_dir, 'dataset', 'train.csv')
# Load the dataset
def load_data():
    try:
        df = pd.read_csv(train_csv_path)
        print(f"Successfully loaded data from {train_csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: train.csv not found at {train_csv_path}")
        return None
df = load_data()
# --- Visualization Functions (Copied from individual files) ---
def create_contour_plot(x, y, z, title="Contour Plot"):
    fig = go.Figure(data=
        go.Contour(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis',
            contours=dict(
                start=np.min(z) if z is not None else 0,
                end=np.max(z) if z is not None else 1,
                size=(np.max(z)-np.min(z))/20 if z is not None and np.max(z) != np.min(z) else 1
            )
        )
    )
    fig.update_layout(title=title)
    return fig
def create_filled_contour(x, y, z, title="Filled Contour Plot"):
    fig = go.Figure(data=
        go.Contour(
            z=z,
            x=x,
            y=y,
            colorscale='Viridis',
            contours=dict(
                start=np.min(z) if z is not None else 0,
                end=np.max(z) if z is not None else 1,
                size=(np.max(z)-np.min(z))/20 if z is not None and np.max(z) != np.min(z) else 1
            ),
            contours_coloring='fill'
        )
    )
    fig.update_layout(title=title)
    return fig
def create_contour_with_scatter(x, y, z, scatter_x, scatter_y, title="Contour Plot with Scatter"):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=z,
        x=x,
        y=y,
        colorscale='Viridis',
        showscale=True
    ))
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
def visualize_decision_tree(model, feature_names, class_names=None, max_depth=None):
    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        max_depth=max_depth
    )
    graph = graphviz.Source(dot_data)
    # Return DOT string so frontend can render it
    return graph.source
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_title('Feature Importance')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    # Convert matplotlib figure to PNG base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory
    return img_base64
def plot_decision_boundary(model, X, y, feature_names, resolution=100):
    if X.shape[1] != 2:
        # Cannot plot decision boundary for non-2D data
        return None
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='k')
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('Decision Boundary')
    plt.tight_layout()
    # Convert matplotlib figure to PNG base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory
    return img_base64
def create_shap_summary_plot(model, X, feature_names):
    # Check if the model is a tree-based model for TreeExplainer
    if not hasattr(model, 'tree_'):
         # For non-tree models, you might need shap.Explainer or other specific explainers
         return None # Indicate that visualization is not possible with this explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0] # Assuming regression or binary classification with shap_values[0] for the positive class or the single output
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    # Convert matplotlib figure to PNG base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory
    return img_base64
def create_shap_dependence_plot(model, X, feature_names, feature_idx):
    # Check if the model is a tree-based model for TreeExplainer
    if not hasattr(model, 'tree_'):
         return None # Indicate that visualization is not possible with this explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0] # Assuming regression or binary classification
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    # Convert matplotlib figure to PNG base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory
    return img_base64
def create_partial_dependence_plot(model, X, feature_names, feature_idx):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        # Calculate partial dependence
        pdp_results, axes = partial_dependence(
            model,
            X,
            [feature_idx],
            kind='average', 
            return_X_array=True # Ensure X_array is returned
        )
        # Plot
        ax.plot(axes[0], pdp_results[0])
        ax.set_xlabel(feature_names[feature_idx])
        ax.set_ylabel('Partial dependence')
        ax.set_title(f'Partial Dependence Plot for {feature_names[feature_idx]}')
        plt.tight_layout()
        # Convert matplotlib figure to PNG base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig) # Close the figure to free memory
        return img_base64
    except Exception as e:
        print(f"Error generating partial dependence plot: {e}")
        return None
def create_partial_dependence_2d(model, X, feature_names, feature_idx1, feature_idx2):
    if X.shape[1] < 2 or feature_idx1 is None or feature_idx2 is None:
        return None # Not enough features or invalid indices
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        # Calculate partial dependence
        pdp_results, axes = partial_dependence(
            model,
            X,
            [(feature_idx1, feature_idx2)], # Pass a tuple for 2D
            kind='average',
            grid_resolution=50, # Lower resolution for performance
            return_X_array=True
        )
        # Plot
        im = ax.imshow(
            pdp_results[0].T, # Transpose for correct orientation
            extent=[axes[0][0], axes[0][-1], axes[1][0], axes[1][-1]],
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        ax.set_xlabel(feature_names[feature_idx2])
        ax.set_ylabel(feature_names[feature_idx1])
        ax.set_title('2D Partial Dependence Plot')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        # Convert matplotlib figure to PNG base64 string
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig) # Close the figure to free memory
        return img_base64
    except Exception as e:
        print(f"Error generating 2D partial dependence plot: {e}")
        return None
def create_parallel_coordinates(df, dimensions, color_col=None):
    if not dimensions:
        return None
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df[color_col] if color_col and color_col in df.columns else None,
                colorscale='Viridis',
                showscale=color_col is not None
            ),
            dimensions=[
                dict(
                    range=[df[dim].min(), df[dim].max()],
                    label=dim,
                    values=df[dim]
                ) for dim in dimensions if dim in df.columns
            ]
        )
    )
    fig.update_layout(
        title="Parallel Coordinates Plot",
        height=800
    )
    return fig
def create_parallel_categories(df, dimensions, color_col=None):
    if not dimensions:
        return None
    fig = go.Figure(data=
        go.Parcats(
            dimensions=[
                dict(
                    values=df[dim].astype(str), # Convert to string for categorical
                    label=dim
                ) for dim in dimensions if dim in df.columns
            ],
            line=dict(
                color=df[color_col] if color_col and color_col in df.columns else None,
                colorscale='Viridis',
                showscale=color_col is not None
            )
        )
    )
    fig.update_layout(
        title="Parallel Categories Plot",
        height=800
    )
    return fig
def create_parallel_coordinates_with_brush(df, dimensions, color_col=None):
    if not dimensions:
        return None
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df[color_col] if color_col and color_col in df.columns else None,
                colorscale='Viridis',
                showscale=color_col is not None
            ),
            dimensions=[
                dict(
                    range=[df[dim].min(), df[dim].max()],
                    label=dim,
                    values=df[dim],
                    constraintrange=[df[dim].min(), df[dim].max()] # Add brushing capability
                ) for dim in dimensions if dim in df.columns
            ]
        )
    )
    fig.update_layout(
        title="Parallel Coordinates Plot with Brushing",
        height=800
    )
    return fig
# --- End Visualization Functions ---
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get_visualization', methods=['POST'])
def get_visualization():
    viz_type = request.json.get('type')
    
    if df is None:
         return jsonify({'error': 'Dataset not loaded. Please check the dataset file path.'}), 500

    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    numeric_cols = numeric_df.columns.tolist()
    
    # Prepare data for model-based visualizations (Decision Tree, SHAP, PDP)
    model_based_viz_types = ['decision_tree', 'shap', 'shap_dependence', 'partial_dependence', 'partial_dependence_2d']
    model = None
    X_cleaned = None
    y_cleaned = None
    feature_names = None

    if viz_type in model_based_viz_types:
        df_cleaned = numeric_df.dropna()
        if 'target' not in df_cleaned.columns:
             return jsonify({'error': "'target' column not found in the dataset for model-based analysis."}), 400
        X_cleaned = df_cleaned.drop(columns=['target'])
        y_cleaned = df_cleaned['target']
        feature_names = X_cleaned.columns.tolist()
        if X_cleaned.empty:
             return jsonify({'error': 'No complete observations for model training after dropping NaNs.'}), 400
        # Train a simple model for demonstration
        model = RandomForestRegressor(n_estimators=10, random_state=42) # Use RandomForestRegressor as before
        model.fit(X_cleaned, y_cleaned)

    if viz_type == 'contour':
        # For a real application, get x_var, y_var, z_var from request.json
        # Using first three numeric columns as default
        if len(numeric_cols) < 3:
             return jsonify({'error': 'Contour plot requires at least 3 numeric columns.'}), 400
             
        x_var = numeric_cols[0]
        y_var = numeric_cols[1]
        z_var = numeric_cols[2]
        
        x = np.linspace(numeric_df[x_var].min(), numeric_df[x_var].max(), 50)
        y = np.linspace(numeric_df[y_var].min(), numeric_df[y_var].max(), 50)
        X_grid, Y_grid = np.meshgrid(x, y)
        # Placeholder Z data - replace with actual calculation or interpolation
        Z_grid = np.full_like(X_grid, numeric_df[z_var].mean() if not numeric_df[z_var].empty else 0)

        fig = create_contour_plot(X_grid, Y_grid, Z_grid)
        return jsonify({'data': fig.data, 'layout': fig.layout}), 200
        
    elif viz_type == 'decision_tree':
        if model is None or X_cleaned is None:
             return jsonify({'error': 'Model or data not prepared for Decision Tree analysis.'}), 500
        if not hasattr(model, 'estimators_') or not model.estimators_:
             return jsonify({'error': 'Model does not have individual trees to visualize.'}), 400

        # Visualize one tree from the forest
        tree_to_viz = model.estimators_[0]
        dot_string = visualize_decision_tree(tree_to_viz, feature_names)
        # Return DOT string
        return jsonify({'dot_string': dot_string}), 200
        
    elif viz_type == 'shap':
        if model is None or X_cleaned is None:
             return jsonify({'error': 'Model or data not prepared for SHAP analysis.'}), 500
        # SHAP TreeExplainer requires a tree-based model. RandomForestRegressor is tree-based.
        # if not hasattr(model, 'tree_') and not hasattr(shap, 'Explainer'):
        #      return jsonify({'error': 'Model type not supported by SHAP TreeExplainer.'}), 400
        
        # Using a sample of the data for SHAP explanation for performance
        sample_size = min(100, len(X_cleaned))
        X_sample = X_cleaned.sample(sample_size, random_state=42) if sample_size > 0 else X_cleaned
        
        if X_sample.empty:
             return jsonify({'error': 'No data sample for SHAP analysis.'}), 400

        # SHAP Summary Plot
        img_base64 = create_shap_summary_plot(model, X_sample, feature_names)
        if img_base64 is None:
             return jsonify({'error': 'Could not generate SHAP summary plot.'}), 500 # More specific error needed from function
        return jsonify({'image_base64': img_base64, 'plot_type': 'matplotlib'}), 200
        
    elif viz_type == 'parallel_coordinates':
        # Using first few numeric columns as default
        dimensions = numeric_cols[:min(len(numeric_cols), 5)]
        
        if len(dimensions) < 2:
             return jsonify({'error': 'Parallel coordinates requires at least 2 numeric columns.'}), 400
             
        fig = create_parallel_coordinates(df, dimensions)
        return jsonify({'data': fig.data, 'layout': fig.layout}), 200

    # Adding routes for other potential plots (e.g., SHAP Dependence, Partial Dependence)
    elif viz_type == 'shap_dependence':
         if model is None or X_cleaned is None or feature_names is None:
              return jsonify({'error': 'Model, data, or feature names not prepared for SHAP dependence plot.'}), 500
         # SHAP TreeExplainer requires a tree-based model.
         # if not hasattr(model, 'tree_') and not hasattr(shap, 'Explainer'):
         #      return jsonify({'error': 'Model type not supported by SHAP TreeExplainer for dependence plot.'}), 400

         # For a real application, get feature_idx from request.json
         if not feature_names:
              return jsonify({'error': 'No features available for SHAP dependence plot.'}), 400
         
         # Using the first feature as default for the dependence plot
         feature_idx = 0
         
         sample_size = min(100, len(X_cleaned))
         X_sample = X_cleaned.sample(sample_size, random_state=42) if sample_size > 0 else X_cleaned

         if X_sample.empty:
              return jsonify({'error': 'No data sample for SHAP dependence plot.'}), 400

         img_base64 = create_shap_dependence_plot(model, X_sample, feature_names, feature_idx)
         if img_base64 is None:
              return jsonify({'error': 'Could not generate SHAP dependence plot.'}), 500 # More specific error needed from function
         return jsonify({'image_base64': img_base64, 'plot_type': 'matplotlib'}), 200

    elif viz_type == 'partial_dependence':
         if model is None or X_cleaned is None or feature_names is None:
              return jsonify({'error': 'Model, data, or feature names not prepared for Partial Dependence plot.'}), 500
         
         # For a real application, get feature_idx from request.json
         if not feature_names:
              return jsonify({'error': 'No features available for Partial Dependence plot.'}), 400
         
         # Using the first feature as default for the partial dependence plot
         feature_idx = 0

         img_base64 = create_partial_dependence_plot(model, X_cleaned, feature_names, feature_idx)
         if img_base64 is None:
              return jsonify({'error': 'Could not generate Partial Dependence plot.'}), 500 # More specific error needed from function
         return jsonify({'image_base64': img_base64, 'plot_type': 'matplotlib'}), 200

    elif viz_type == 'partial_dependence_2d':
         if model is None or X_cleaned is None or feature_names is None:
              return jsonify({'error': 'Model, data, or feature names not prepared for 2D Partial Dependence plot.'}), 500
         if len(feature_names) < 2:
              return jsonify({'error': '2D Partial Dependence plot requires at least 2 features.'}), 400

         # For a real application, get feature_idx1, feature_idx2 from request.json
         # Using the first two features as default
         feature_idx1 = 0
         feature_idx2 = 1

         img_base64 = create_partial_dependence_2d(model, X_cleaned, feature_names, feature_idx1, feature_idx2)
         if img_base64 is None:
              return jsonify({'error': 'Could not generate 2D Partial Dependence plot.'}), 500 # More specific error needed from function
         return jsonify({'image_base64': img_base64, 'plot_type': 'matplotlib'}), 200
        
    else:
        return jsonify({'error': 'Unknown visualization type.'}), 400

if __name__ == '__main__':
    # Ensure Flask app is run from the correct directory for template loading and data access
    # Use a relative path that works when adv_app.py is at the project root
    app.run(debug=True) 