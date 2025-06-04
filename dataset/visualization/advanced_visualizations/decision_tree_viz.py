import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import export_graphviz

def visualize_decision_tree(model, feature_names, class_names=None, max_depth=None):
    """
    Create an interactive decision tree visualization
    """
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
    return graph

def plot_feature_importance(model, feature_names):
    """
    Create a bar plot of feature importance
    """
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
    
    return fig

def plot_decision_boundary(model, X, y, feature_names, resolution=100):
    """
    Create a decision boundary plot for 2D data
    """
    if len(feature_names) != 2:
        raise ValueError("Decision boundary plot requires exactly 2 features")
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title('Decision Boundary')
    
    return fig 