import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add the directory containing the visualization modules to the Python path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
advanced_viz_dir = os.path.join(os.path.dirname(current_file_dir), 'advanced_visualizations')
sys.path.append(advanced_viz_dir)

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Removed: import advanced_visualizations

# Set page config
st.set_page_config(
    page_title="Solar Panel Dataset Analysis",
    page_icon="☀️",
    layout="wide"
)

# Get the current directory and construct the correct path to train.csv
# This path calculation seems to be for the dataset file, keep it separate.
current_dir_for_dataset = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
train_csv_path = os.path.join(current_dir_for_dataset, 'train.csv')

# Read the dataset
@st.cache_data
def load_data():
    return pd.read_csv(train_csv_path)

df = load_data()

# Title and description
st.title("Solar Panel Dataset Analysis Dashboard")
st.markdown("""
This dashboard helps you explore relationships between variables in the solar panel dataset.
Use the sidebar to navigate through different analyses.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis",
    ["Overview", "Null Value Analysis", "Correlation Analysis", "Variable Relationships", 
     "Impact Analysis", "3D Visualizations", "Contour Plots", "Decision Tree Analysis",
     "SHAP Analysis", "Parallel Coordinates"]
)

if page == "Overview":
    st.header("Dataset Overview")
    
    # Basic statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))
    
    with col2:
        st.metric("Numeric Columns", len(df.select_dtypes(include=['float64', 'int64']).columns))
        st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))
    
    # Display first few rows
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

elif page == "Null Value Analysis":
    st.header("Null Value Analysis")
    
    # Calculate null percentages
    null_percentages = (df.isnull().sum() / len(df)) * 100
    null_df = pd.DataFrame({
        'Column': null_percentages.index,
        'Null Percentage': null_percentages.values
    }).sort_values('Null Percentage', ascending=False)
    
    # Plot null percentages using plotly
    fig = px.bar(null_df, 
                 x='Null Percentage', 
                 y='Column',
                 title='Percentage of Null Values by Column',
                 orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    # Display null value statistics
    st.subheader("Null Value Statistics")
    st.dataframe(null_df)

elif page == "Correlation Analysis":
    st.header("Correlation Analysis")
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method='pearson', min_periods=1)
    
    # Create correlation heatmap using plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display correlation matrix
    st.subheader("Correlation Matrix")
    st.dataframe(corr_matrix)

elif page == "Variable Relationships":
    st.header("Variable Relationships")
    
    # Select variables to compare
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X Variable", numeric_cols)
    with col2:
        y_var = st.selectbox("Select Y Variable", numeric_cols)
    
    # Create scatter plot
    fig = px.scatter(df, 
                     x=x_var, 
                     y=y_var,
                     title=f'Relationship between {x_var} and {y_var}',
                     marginal_x='histogram',
                     marginal_y='histogram')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlation
    corr = df[[x_var, y_var]].corr().iloc[0,1]
    st.metric("Correlation", f"{corr:.3f}")
    
    # Show statistics
    st.subheader("Variable Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Statistics for {x_var}")
        st.write(df[x_var].describe())
    with col2:
        st.write(f"Statistics for {y_var}")
        st.write(df[y_var].describe())

elif page == "Impact Analysis":
    st.header("Impact Analysis")
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method='pearson', min_periods=1)
    
    # Calculate impact scores
    impact_scores = {}
    for col in numeric_df.columns:
        impact_scores[col] = {
            'Impact Score': abs(corr_matrix[col]).mean(),
            'Null Percentage': (df[col].isnull().sum() / len(df)) * 100,
            'Complete Observations': len(df) - df[col].isnull().sum()
        }
    
    impact_df = pd.DataFrame(impact_scores).T
    impact_df = impact_df.sort_values('Impact Score', ascending=False)
    
    # Plot impact scores
    fig = px.bar(impact_df.reset_index(),
                 x='Impact Score',
                 y='index',
                 title='Variable Impact Scores',
                 orientation='h',
                 hover_data=['Null Percentage', 'Complete Observations'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display impact scores table
    st.subheader("Impact Scores Table")
    st.dataframe(impact_df)

elif page == "3D Visualizations":
    st.header("3D Visualizations")
    from _3d_plots import create_3d_scatter, create_3d_surface, create_3d_line
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_var = st.selectbox("Select X Variable", numeric_cols, key='3d_x')
    with col2:
        y_var = st.selectbox("Select Y Variable", numeric_cols, key='3d_y')
    with col3:
        z_var = st.selectbox("Select Z Variable", numeric_cols, key='3d_z')
    
    viz_type = st.radio("Select Visualization Type", ["3D Scatter", "3D Surface", "3D Line"])
    
    if viz_type == "3D Scatter":
        fig = create_3d_scatter(df, x_var, y_var, z_var)
    elif viz_type == "3D Surface":
        # Create grid for surface plot
        x = np.linspace(df[x_var].min(), df[x_var].max(), 50)
        y = np.linspace(df[y_var].min(), df[y_var].max(), 50)
        X, Y = np.meshgrid(x, y)
        # For a surface plot, Z should typically be a function of X and Y. Using mean is a placeholder.
        # You might need to train a model or use interpolation here based on your data.
        Z = np.zeros_like(X)
        # Placeholder: Using the mean of the Z variable for simplicity
        Z = np.full_like(X, df[z_var].mean() if not df[z_var].empty else 0)
        fig = create_3d_surface(X, Y, Z)
    else:
        fig = create_3d_line(df, x_var, y_var, z_var)
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Contour Plots":
    st.header("Contour Plots")
    from contour_plots import create_contour_plot, create_filled_contour, create_contour_with_scatter
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Select X Variable", numeric_cols, key='contour_x')
    with col2:
        y_var = st.selectbox("Select Y Variable", numeric_cols, key='contour_y')
    
    # Create grid for contour plot
    x = np.linspace(df[x_var].min(), df[x_var].max(), 50)
    y = np.linspace(df[y_var].min(), df[y_var].max(), 50)
    X, Y = np.meshgrid(x, y)
    
    # For a contour plot, Z should typically be a function of X and Y.
    z_var_for_contour = st.selectbox("Select Z Variable for Contour", numeric_cols, key='contour_z')
    Z = np.full_like(X, df[z_var_for_contour].mean() if not df[z_var_for_contour].empty else 0)
    
    viz_type = st.radio("Select Contour Plot Type", ["Basic Contour", "Filled Contour", "Contour with Scatter"])
    
    if viz_type == "Basic Contour":
        fig = create_contour_plot(X, Y, Z)
    elif viz_type == "Filled Contour":
        fig = create_filled_contour(X, Y, Z)
    else:
        # For contour with scatter, we use the original data points
        scatter_x = df[x_var]
        scatter_y = df[y_var]
        fig = create_contour_with_scatter(X, Y, Z, scatter_x, scatter_y)
    
    st.plotly_chart(fig, use_container_width=True)

elif page == "Decision Tree Analysis":
    st.header("Decision Tree Analysis")
    from decision_tree_viz import visualize_decision_tree, plot_feature_importance, plot_decision_boundary
    # ... rest of Decision Tree Analysis block ...
    
    # Prepare data for decision tree
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Drop rows with NaNs for simplicity in model training
    df_cleaned = numeric_df.dropna()
    
    # Update the target column name check and assignment
    target_column_name = 'efficiency' # Set the target column name here
    
    if target_column_name not in df_cleaned.columns:
        st.warning(f"'{target_column_name}' column not found in the dataset. Please ensure your dataset has a column named '{target_column_name}' for Decision Tree and SHAP Analysis.")
    else:
        X = df_cleaned.drop(columns=[target_column_name])
        y = df_cleaned[target_column_name]
        
        if X.empty:
            st.warning("No complete observations available for training the model after dropping NaNs.")
        else:
            # Train a simple decision tree
            # Using a small number of estimators for faster demonstration
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Visualize decision tree
            st.subheader("Decision Tree Visualization")
            # Select one tree from the forest to visualize
            tree_to_viz = model.estimators_[0]
            tree_viz = visualize_decision_tree(tree_to_viz, X.columns)
            st.graphviz_chart(tree_viz)
            
            # Feature importance
            st.subheader("Feature Importance")
            fig = plot_feature_importance(model, X.columns)
            st.pyplot(fig)
            
            # Decision boundary (if 2D)
            if len(X.columns) >= 2:
                st.subheader("Decision Boundary")
                # Using only the first two features for 2D boundary plot
                X_2d = X.iloc[:, :2].values
                fig = plot_decision_boundary(model.estimators_[0], X_2d, y.values, X.columns[:2])
                st.pyplot(fig)
            else:
                st.info("Decision boundary plot requires at least 2 features.")

elif page == "SHAP Analysis":
    st.header("SHAP Analysis")
    from shap_pdp_plots import create_shap_summary_plot, create_shap_dependence_plot, create_partial_dependence_plot, create_partial_dependence_2d
    # ... rest of SHAP Analysis block ...
    
    # Prepare data
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Drop rows with NaNs for simplicity in model training
    df_cleaned = numeric_df.dropna()
    
    # Update the target column name check and assignment
    target_column_name = 'efficiency' # Set the target column name here

    if target_column_name not in df_cleaned.columns:
         st.warning(f"'{target_column_name}' column not found in the dataset. Please ensure your dataset has a column named '{target_column_name}' for Decision Tree and SHAP Analysis.")
    else:
        X = df_cleaned.drop(columns=[target_column_name])
        y = df_cleaned[target_column_name]
        
        if X.empty:
            st.warning("No complete observations available for training the model after dropping NaNs.")
        else:
            # Train a model (using a simple one for demonstration)
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # SHAP Summary Plot
            st.subheader("SHAP Summary Plot")
            # Using a sample of the data for SHAP explanation for performance
            sample_size = min(100, len(X))
            X_sample = X.sample(sample_size, random_state=42)
            fig = create_shap_summary_plot(model, X_sample, X.columns)
            st.pyplot(fig)
            
            # SHAP Dependence Plot
            st.subheader("SHAP Dependence Plot")
            if not X.columns.empty:
                feature_name = st.selectbox("Select Feature for Dependence Plot", X.columns.tolist(), key='shap_dep')
                feature_idx = X.columns.get_loc(feature_name)
                fig = create_shap_dependence_plot(model, X_sample, X.columns, feature_idx)
                st.pyplot(fig)
            else:
                st.info("No features available for SHAP Dependence plot.")
            
            # Partial Dependence Plot
            st.subheader("Partial Dependence Plot")
            if not X.columns.empty:
                feature_name_pdp = st.selectbox("Select Feature for Partial Dependence Plot", X.columns.tolist(), key='pdp')
                feature_idx_pdp = X.columns.get_loc(feature_name_pdp)
                fig = create_partial_dependence_plot(model, X, X.columns, feature_idx_pdp)
                st.pyplot(fig)
            else:
                 st.info("No features available for Partial Dependence plot.")

elif page == "Parallel Coordinates":
    st.header("Parallel Coordinates Analysis")
    from parallel_coordinates import create_parallel_coordinates, create_parallel_categories, create_parallel_coordinates_with_brush
    # ... rest of Parallel Coordinates block ...
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    selected_cols = st.multiselect("Select Variables", numeric_cols.tolist(), default=numeric_cols[:min(len(numeric_cols), 5)].tolist())
    
    if len(selected_cols) >= 2:
        viz_type = st.radio("Select Visualization Type", 
                          ["Parallel Coordinates", "Parallel Categories", "Parallel Coordinates with Brushing"])
        
        if viz_type == "Parallel Coordinates":
            # Check if there are enough unique values for parallel coordinates
            if df[selected_cols].nunique().min() > 1:
                 fig = create_parallel_coordinates(df, selected_cols)
                 st.plotly_chart(fig, use_container_width=True)
            else:
                 st.warning("Selected columns have too few unique values for Parallel Coordinates.")
        elif viz_type == "Parallel Categories":
             # Check if there are too many unique values for parallel categories
             if df[selected_cols].nunique().max() < 50: # Arbitrary threshold
                 fig = create_parallel_categories(df, selected_cols)
                 st.plotly_chart(fig, use_container_width=True)
             else:
                  st.warning("Selected columns have too many unique values for Parallel Categories. Consider using Parallel Coordinates instead.")
        else:
            # Check if there are enough unique values for parallel coordinates with brushing
            if df[selected_cols].nunique().min() > 1:
                 fig = create_parallel_coordinates_with_brush(df, selected_cols)
                 st.plotly_chart(fig, use_container_width=True)
            else:
                 st.warning("Selected columns have too few unique values for Parallel Coordinates with Brushing.")
    else:
        st.warning("Please select at least 2 variables for parallel coordinates visualization.")

# Add footer
st.markdown("---")
st.markdown("Dashboard created for Solar Panel Dataset Analysis") 