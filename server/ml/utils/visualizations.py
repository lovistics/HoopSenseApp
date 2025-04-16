"""
Visualization utilities for ML model analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Union, Optional
import io
import base64
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

from app.core.logger import logger


def plot_feature_importance(
    feature_names: List[str],
    importances: List[float],
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (12, 8),
    top_n: Optional[int] = None,
    horizontal: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importances.
    
    Args:
        feature_names: List of feature names
        importances: List of importance values
        title: Plot title
        figsize: Figure size
        top_n: Number of top features to show (None for all)
        horizontal: If True, plot horizontal bars
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Sort features by importance
    indices = np.argsort(importances)
    if top_n is not None:
        indices = indices[-top_n:]
    
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    if horizontal:
        ax.barh(range(len(sorted_names)), sorted_importances, align='center')
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance')
    else:
        ax.bar(range(len(sorted_names)), sorted_importances, align='center')
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=90)
        ax.set_ylabel('Importance')
    
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_shap_summary(
    shap_values,
    features: pd.DataFrame,
    max_display: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot SHAP summary plot for feature importance.
    
    Args:
        shap_values: SHAP values from explainer
        features: Feature dataframe
        max_display: Maximum number of features to display
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    import shap
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Create SHAP summary plot
    shap.summary_plot(shap_values, features, max_display=max_display, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix_sns(
    y_true: List[Any],
    y_pred: List[Any],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix using seaborn for better visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_roc_curves(
    y_true: List[Any],
    y_probs: Dict[str, List[float]],
    title: str = "ROC Curves",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        y_true: True labels
        y_probs: Dictionary of model name to predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot diagonal line for reference
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    # Plot ROC curve for each model
    for model_name, y_prob in y_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: List[Any],
    y_prob: List[float],
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Calculate precision and recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    ax.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
    
    # Plot random baseline
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    ax.plot([0, 1], [no_skill, no_skill], 'k--', label=f'Random (AUC = {no_skill:.3f})')
    
    # Set labels and title
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='best')
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_feature_distributions(
    data: pd.DataFrame,
    target_col: str,
    features: List[str],
    figsize: Tuple[int, int] = (15, 12),
    bins: int = 30,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distributions of features by target class.
    
    Args:
        data: DataFrame with features and target
        target_col: Column name of target variable
        features: List of feature columns to plot
        figsize: Figure size
        bins: Number of bins for histograms
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Determine number of rows and columns for subplots
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array if necessary
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot distributions for each feature
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            
            # Get unique target values
            target_values = data[target_col].unique()
            
            # Plot histogram for each target value
            for target_value in target_values:
                subset = data[data[target_col] == target_value]
                ax.hist(subset[feature], alpha=0.5, bins=bins, label=str(target_value))
            
            ax.set_title(feature)
            ax.set_xlabel(feature)
            ax.set_ylabel('Count')
            
            # Add legend to first plot only
            if i == 0:
                ax.legend()
    
    # Hide empty subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_prediction_calibration(
    y_true: List[Any],
    y_prob: List[float],
    n_bins: int = 10,
    title: str = "Calibration Plot",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve showing predicted vs actual probabilities.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for grouping predictions
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve
    
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    
    # Set labels and title
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(title)
    ax.legend(loc='best')
    
    # Set limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_win_probability_trends(
    game_data: pd.DataFrame,
    date_col: str = 'date',
    home_team_col: str = 'home_team',
    away_team_col: str = 'away_team',
    prob_col: str = 'home_win_probability',
    n_games: int = 10,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot win probability trends for recent games.
    
    Args:
        game_data: DataFrame with game predictions
        date_col: Column name for date
        home_team_col: Column name for home team
        away_team_col: Column name for away team
        prob_col: Column name for win probability
        n_games: Number of recent games to plot
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Sort data by date
    sorted_data = game_data.sort_values(by=date_col).tail(n_games)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot win probabilities
    dates = sorted_data[date_col].dt.strftime('%Y-%m-%d')
    probabilities = sorted_data[prob_col]
    
    # Create labels for x-axis
    labels = [f"{row[home_team_col]} vs {row[away_team_col]}" 
              for _, row in sorted_data.iterrows()]
    
    # Plot data
    ax.plot(range(len(probabilities)), probabilities, marker='o', linestyle='-', linewidth=2)
    
    # Add reference line at 0.5
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Game')
    ax.set_ylabel('Home Win Probability')
    ax.set_title('Win Probability Trends for Recent Games')
    
    # Set x-ticks
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim([0, 1])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_prediction_accuracy_by_confidence(
    y_true: List[Any],
    y_pred: List[Any],
    y_prob: List[float],
    n_bins: int = 10,
    title: str = "Prediction Accuracy by Confidence",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot prediction accuracy by confidence level.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction confidence/probability
        n_bins: Number of bins for grouping predictions
        title: Plot title
        figsize: Figure size
        save_path: Path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Adjust probabilities to represent confidence
    # For binary classification with probabilities near 0, confidence is 1-prob
    y_conf = np.where(y_prob < 0.5, 1 - y_prob, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bins
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate accuracy for each bin
    accuracies = []
    counts = []
    
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        
        # Find predictions in this confidence bin
        mask = (y_conf >= lower) & (y_conf < upper)
        
        # Skip if no predictions in this bin
        if np.sum(mask) == 0:
            accuracies.append(0)
            counts.append(0)
            continue
        
        # Calculate accuracy for this bin
        correct = (y_true[mask] == y_pred[mask])
        accuracy = np.mean(correct)
        count = np.sum(mask)
        
        accuracies.append(accuracy)
        counts.append(count)
    
    # Plot bars for accuracy
    ax.bar(bin_centers, accuracies, width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.6)
    
    # Plot diagonal line for perfect calibration
    ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', label='Perfect Calibration')
    
    # Set labels and title
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    
    # Add count labels
    for i, (x, y, count) in enumerate(zip(bin_centers, accuracies, counts)):
        ax.annotate(
            f'{count}',
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center'
        )
    
    # Set limits
    ax.set_xlim([0.45, 1.05])
    ax.set_ylim([0, 1.05])
    
    # Add legend
    ax.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def figure_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to base64 string for web display.
    
    Args:
        fig: Matplotlib figure
    
    Returns:
        Base64 encoded string of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str