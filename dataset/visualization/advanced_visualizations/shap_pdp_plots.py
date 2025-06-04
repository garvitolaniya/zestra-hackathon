import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import partial_dependence

def create_shap_summary_plot(model, X, feature_names):
    """
    Create a SHAP summary plot
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    return fig

def create_shap_dependence_plot(model, X, feature_names, feature_idx):
    """
    Create a SHAP dependence plot for a specific feature
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    return fig

def create_partial_dependence_plot(model, X, feature_names, feature_idx):
    """
    Create a partial dependence plot for a specific feature
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate partial dependence
    pdp = partial_dependence(
        model,
        X,
        [feature_idx],
        kind='average'
    )
    
    # Plot
    ax.plot(pdp[1][0], pdp[0][0])
    ax.set_xlabel(feature_names[feature_idx])
    ax.set_ylabel('Partial dependence')
    ax.set_title(f'Partial Dependence Plot for {feature_names[feature_idx]}')
    plt.tight_layout()
    
    return fig

def create_partial_dependence_2d(model, X, feature_names, feature_idx1, feature_idx2):
    """
    Create a 2D partial dependence plot for two features
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate partial dependence
    pdp = partial_dependence(
        model,
        X,
        [feature_idx1, feature_idx2],
        kind='average'
    )
    
    # Plot
    im = ax.imshow(
        pdp[0][0],
        extent=[pdp[1][1][0], pdp[1][1][-1], pdp[1][0][0], pdp[1][0][-1]],
        aspect='auto',
        origin='lower',
        cmap='viridis'
    )
    
    ax.set_xlabel(feature_names[feature_idx2])
    ax.set_ylabel(feature_names[feature_idx1])
    ax.set_title('2D Partial Dependence Plot')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    return fig 