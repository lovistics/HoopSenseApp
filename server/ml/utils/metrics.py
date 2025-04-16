"""
Evaluation metrics for machine learning model performance.
"""
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from app.core.logger import logger


def classification_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    y_prob: Optional[Union[List, np.ndarray]] = None,
    average: str = 'binary',
    labels: Optional[List] = None
) -> Dict[str, float]:
    """
    Calculate common classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities for positive class
        average: Averaging strategy for multiclass ('binary', 'micro', 'macro', 'weighted')
        labels: Optional list of labels to index the matrix
    
    Returns:
        Dictionary of metrics
    """
    # Convert inputs to numpy arrays if they're not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if binary or multiclass
    is_binary = len(np.unique(y_true)) <= 2
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # AUC and log loss if probabilities are provided
    if y_prob is not None:
        y_prob = np.array(y_prob)
        
        # For binary classification
        if is_binary:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            metrics['log_loss'] = log_loss(y_true, y_prob)
        # For multiclass, ensure probabilities are properly formatted
        elif y_prob.ndim > 1 and y_prob.shape[1] > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Failed to calculate AUC/log_loss: {e}")
    
    # Additional metrics
    try:
        # Confusion matrix values
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if is_binary:
            metrics['tn'] = cm[0, 0]
            metrics['fp'] = cm[0, 1]
            metrics['fn'] = cm[1, 0]
            metrics['tp'] = cm[1, 1]
    except Exception as e:
        logger.warning(f"Failed to calculate confusion matrix: {e}")
    
    return metrics


def regression_metrics(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray]
) -> Dict[str, float]:
    """
    Calculate common regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    # Convert inputs to numpy arrays if they're not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Mean absolute percentage error (handle zeros in y_true)
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        metrics['mape'] = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        metrics['mape'] = np.nan
    
    return metrics


def print_metrics_summary(metrics: Dict[str, float], title: str = "Model Performance") -> None:
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the summary
    """
    print(f"\n{'-' * 50}\n{title}\n{'-' * 50}")
    
    for metric, value in metrics.items():
        # Format based on metric type
        if metric in ['mape']:
            print(f"{metric.upper()}: {value:.2f}%")
        elif isinstance(value, float):
            print(f"{metric.upper()}: {value:.4f}")
        else:
            print(f"{metric.upper()}: {value}")
            
    print(f"{'-' * 50}\n")


def plot_confusion_matrix(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a confusion matrix for classification results.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        normalize: Whether to normalize the matrix
        title: Plot title
        cmap: Colormap
        figsize: Figure size
    
    Returns:
        Tuple of (figure, axes)
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax)
    
    # Set labels
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Add text annotations to cells
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Set labels and title
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    
    fig.tight_layout()
    return fig, ax


def plot_roc_curve(
    y_true: Union[List, np.ndarray],
    y_prob: Union[List, np.ndarray],
    figsize: Tuple[int, int] = (10, 8),
    title: str = "ROC Curve"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for positive class
        figsize: Figure size
        title: Plot title
    
    Returns:
        Tuple of (figure, axes)
    """
    from sklearn.metrics import roc_curve, auc
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    fig.tight_layout()
    return fig, ax


def calculate_basketball_prediction_metrics(
    predictions: pd.DataFrame,
    actual_column: str = 'actual_winner',
    pred_column: str = 'predicted_winner',
    prob_column: str = 'win_probability'
) -> Dict[str, Any]:
    """
    Calculate basketball prediction specific metrics.
    
    Args:
        predictions: DataFrame with predictions and actual results
        actual_column: Column name for actual winner
        pred_column: Column name for predicted winner
        prob_column: Column name for win probability
    
    Returns:
        Dictionary of basketball-specific metrics
    """
    # Standard classification metrics
    basic_metrics = classification_metrics(
        predictions[actual_column],
        predictions[pred_column],
        predictions[prob_column]
    )
    
    # Basketball specific metrics
    metrics = {
        'accuracy': basic_metrics['accuracy'],
        'auc': basic_metrics.get('auc', np.nan),
        'home_win_accuracy': accuracy_score(
            predictions[predictions[actual_column] == 'home'][actual_column],
            predictions[predictions[actual_column] == 'home'][pred_column]
        ),
        'away_win_accuracy': accuracy_score(
            predictions[predictions[actual_column] == 'away'][actual_column],
            predictions[predictions[actual_column] == 'away'][pred_column]
        ),
        'high_confidence_accuracy': accuracy_score(
            predictions[predictions[prob_column] > 0.7][actual_column],
            predictions[predictions[prob_column] > 0.7][pred_column]
        ) if sum(predictions[prob_column] > 0.7) > 0 else np.nan,
        'very_high_confidence_accuracy': accuracy_score(
            predictions[predictions[prob_column] > 0.85][actual_column],
            predictions[predictions[prob_column] > 0.85][pred_column]
        ) if sum(predictions[prob_column] > 0.85) > 0 else np.nan,
        'correct_predictions': basic_metrics.get('tp', 0) + basic_metrics.get('tn', 0),
        'total_predictions': len(predictions)
    }
    
    # Calculate streak analysis
    if len(predictions) > 0:
        # Sort by date if available
        if 'date' in predictions.columns:
            sorted_preds = predictions.sort_values(by='date')
        else:
            sorted_preds = predictions
            
        # Add correct column
        sorted_preds['correct'] = sorted_preds[actual_column] == sorted_preds[pred_column]
        
        # Find longest streak of correct predictions
        streaks = []
        current_streak = 0
        
        for correct in sorted_preds['correct']:
            if correct:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 0
                
        # Add final streak
        streaks.append(current_streak)
        
        metrics['best_streak'] = max(streaks)
        metrics['current_streak'] = streaks[-1]
    
    return metrics