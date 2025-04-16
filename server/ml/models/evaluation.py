"""
Model evaluation module for basketball prediction models.

This module provides methods for evaluating model performance,
generating performance reports, and analyzing prediction results.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from app.core.logger import logger
from app.core.config import settings
from app.ml.models.xgboost_model import BasketballXGBoostModel


class ModelEvaluator:
    """
    Evaluates basketball prediction models and generates performance reports.
    """
    
    def __init__(
        self,
        model: Optional[BasketballXGBoostModel] = None,
        report_dir: Optional[str] = None
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model: Optional model to evaluate
            report_dir: Directory for saving evaluation reports
        """
        self.model = model
        
        # Set report directory
        if report_dir is None:
            self.report_dir = os.path.join(settings.BASE_DIR, "ml", "reports")
        else:
            self.report_dir = report_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self.report_dir, exist_ok=True)
    
    def set_model(self, model: BasketballXGBoostModel) -> None:
        """
        Set the model to evaluate.
        
        Args:
            model: Model to evaluate
        """
        self.model = model
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        target_column: str = "target",
        exclude_columns: Optional[List[str]] = None,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test data with features and target
            target_column: Name of target column
            exclude_columns: Columns to exclude from features
            detailed: Whether to generate detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            ValueError: If model is not set or target column not in test data
        """
        if self.model is None:
            raise ValueError("Model must be set before evaluation")
        
        if target_column not in test_data.columns:
            raise ValueError(f"Target column '{target_column}' not in test data")
        
        # Set default exclude columns
        if exclude_columns is None:
            exclude_columns = ["game_id", "home_team_id", "away_team_id", "date"]
        
        # Prepare data
        X_test = test_data.drop([target_column] + [col for col in exclude_columns if col in test_data.columns], axis=1)
        y_test = test_data[target_column]
        
        # Convert target to numeric if necessary
        y_test_numeric = None
        if y_test.dtype == object:
            output_classes = sorted(y_test.unique())
            y_test_numeric = (y_test == output_classes[1]).astype(int)
        else:
            y_test_numeric = y_test
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Basic metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "log_loss": float(log_loss(y_test_numeric, y_pred_proba)),
            "auc": float(roc_auc_score(y_test_numeric, y_pred_proba)),
            "test_samples": len(X_test)
        }
        
        # Add detailed metrics if requested
        if detailed:
            # Add precision, recall, F1
            metrics["precision"] = float(precision_score(y_test_numeric, (y_pred_proba >= 0.5).astype(int)))
            metrics["recall"] = float(recall_score(y_test_numeric, (y_pred_proba >= 0.5).astype(int)))
            metrics["f1_score"] = float(f1_score(y_test_numeric, (y_pred_proba >= 0.5).astype(int)))
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test_numeric, (y_pred_proba >= 0.5).astype(int))
            metrics["confusion_matrix"] = cm.tolist()
            
            # Calculate metrics at different thresholds
            thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
            threshold_metrics = {}
            
            for threshold in thresholds:
                y_pred_threshold = (y_pred_proba >= threshold).astype(int)
                
                # Calculate metrics for predictions above this threshold
                above_threshold = (y_pred_proba >= threshold)
                if above_threshold.sum() > 0:
                    threshold_metrics[str(threshold)] = {
                        "samples": int(above_threshold.sum()),
                        "accuracy": float(accuracy_score(y_test_numeric[above_threshold], y_pred_threshold[above_threshold])),
                        "percentage_of_games": float(above_threshold.mean())
                    }
                else:
                    threshold_metrics[str(threshold)] = {
                        "samples": 0,
                        "accuracy": 0.0,
                        "percentage_of_games": 0.0
                    }
            
            metrics["threshold_metrics"] = threshold_metrics
            
            # Add feature importance
            feature_importance = self.model.get_feature_importance(top_n=50)
            metrics["feature_importance"] = feature_importance
        
        return metrics
    
    def evaluate_over_time(
        self,
        data: pd.DataFrame,
        target_column: str = "target",
        date_column: str = "date",
        exclude_columns: Optional[List[str]] = None,
        period: str = "month"
    ) -> Dict[str, Any]:
        """
        Evaluate model performance over time periods.
        
        Args:
            data: Data with features, target, and date
            target_column: Name of target column
            date_column: Name of date column
            exclude_columns: Columns to exclude from features
            period: Time period for grouping ('day', 'week', 'month')
            
        Returns:
            Dictionary of evaluation metrics by time period
            
        Raises:
            ValueError: If model is not set or required columns not in data
        """
        if self.model is None:
            raise ValueError("Model must be set before evaluation")
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not in data")
            
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not in data")
        
        # Set default exclude columns
        if exclude_columns is None:
            exclude_columns = ["game_id", "home_team_id", "away_team_id"]
            
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column])
        
        # Group data by time period
        if period == "day":
            data["period"] = data[date_column].dt.date
        elif period == "week":
            data["period"] = data[date_column].dt.to_period("W").apply(lambda x: x.start_time.date())
        else:  # Default to month
            data["period"] = data[date_column].dt.to_period("M").apply(lambda x: x.start_time.date())
        
        # Get unique periods
        periods = sorted(data["period"].unique())
        
        # Evaluate for each period
        results = {
            "period_metrics": {},
            "overall_metrics": {},
            "periods": [str(p) for p in periods]
        }
        
        all_y_test = []
        all_y_pred = []
        all_y_proba = []
        
        for period_value in periods:
            period_data = data[data["period"] == period_value]
            
            # Prepare data
            X_test = period_data.drop([target_column, date_column, "period"] + 
                                    [col for col in exclude_columns if col in period_data.columns], 
                                    axis=1)
            y_test = period_data[target_column]
            
            # Convert target to numeric if necessary
            y_test_numeric = None
            if y_test.dtype == object:
                output_classes = sorted(y_test.unique())
                y_test_numeric = (y_test == output_classes[1]).astype(int)
            else:
                y_test_numeric = y_test
            
            # Make predictions
            try:
                y_pred = self.model.predict(X_test)
                y_pred_proba = self.model.predict_proba(X_test)
                
                # Store for overall metrics
                all_y_test.extend(y_test_numeric)
                all_y_pred.extend(y_pred)
                all_y_proba.extend(y_pred_proba)
                
                # Calculate metrics
                period_metrics = {
                    "samples": len(X_test),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "auc": float(roc_auc_score(y_test_numeric, y_pred_proba))
                }
                
                results["period_metrics"][str(period_value)] = period_metrics
                
            except Exception as e:
                logger.error(f"Error evaluating period {period_value}: {str(e)}")
                results["period_metrics"][str(period_value)] = {
                    "samples": len(X_test),
                    "error": str(e)
                }
        
        # Calculate overall metrics
        if all_y_test:
            results["overall_metrics"] = {
                "accuracy": float(accuracy_score(all_y_test, all_y_pred)),
                "auc": float(roc_auc_score(all_y_test, all_y_proba)),
                "total_samples": len(all_y_test)
            }
        
        return results
    
    def analyze_errors(
        self,
        test_data: pd.DataFrame,
        target_column: str = "target",
        exclude_columns: Optional[List[str]] = None,
        error_analysis_type: str = "basic"
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors to identify patterns.
        
        Args:
            test_data: Test data with features and target
            target_column: Name of target column
            exclude_columns: Columns to exclude from features
            error_analysis_type: Type of analysis ('basic', 'detailed')
            
        Returns:
            Dictionary with error analysis results
            
        Raises:
            ValueError: If model is not set or target column not in test data
        """
        if self.model is None:
            raise ValueError("Model must be set before error analysis")
        
        if target_column not in test_data.columns:
            raise ValueError(f"Target column '{target_column}' not in test data")
        
        # Set default exclude columns
        if exclude_columns is None:
            exclude_columns = ["game_id", "date"]
        
        # Prepare data
        feature_cols = [col for col in test_data.columns 
                      if col != target_column and col not in exclude_columns]
        X_test = test_data[feature_cols]
        y_test = test_data[target_column]
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Add predictions to data
        test_with_preds = test_data.copy()
        test_with_preds["predicted"] = y_pred
        test_with_preds["probability"] = y_pred_proba
        test_with_preds["correct"] = (test_with_preds[target_column] == test_with_preds["predicted"])
        
        # Basic error analysis
        error_analysis = {
            "total_samples": len(test_with_preds),
            "correct_predictions": test_with_preds["correct"].sum(),
            "wrong_predictions": len(test_with_preds) - test_with_preds["correct"].sum(),
            "accuracy": float(test_with_preds["correct"].mean())
        }
        
        # Add confidence distribution
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        confidence_dist = {}
        
        for i in range(len(bins)-1):
            mask = (test_with_preds["probability"] >= bins[i]) & (test_with_preds["probability"] < bins[i+1])
            bin_data = test_with_preds[mask]
            
            if len(bin_data) > 0:
                confidence_dist[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] = {
                    "samples": len(bin_data),
                    "accuracy": float(bin_data["correct"].mean()),
                    "percentage": float(len(bin_data) / len(test_with_preds))
                }
        
        error_analysis["confidence_distribution"] = confidence_dist
        
        # For detailed analysis
        if error_analysis_type == "detailed" and len(feature_cols) > 0:
            # Analyze which features are most correlated with errors
            error_correlations = {}
            
            for feature in feature_cols:
                if test_data[feature].dtype in [np.float64, np.int64, np.bool_]:
                    correlation = np.corrcoef(test_with_preds[feature], ~test_with_preds["correct"])[0, 1]
                    error_correlations[feature] = float(correlation)
            
            # Sort by absolute correlation
            sorted_correlations = sorted(
                error_correlations.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            error_analysis["error_correlations"] = dict(sorted_correlations[:20])
            
            # Feature interactions in errors
            if "home_team_id" in test_data.columns and "away_team_id" in test_data.columns:
                # Home vs away errors
                home_mask = test_with_preds[target_column] == "home"
                away_mask = test_with_preds[target_column] == "away"
                
                error_analysis["home_away_bias"] = {
                    "home_team_wins": len(test_with_preds[home_mask]),
                    "away_team_wins": len(test_with_preds[away_mask]),
                    "home_accuracy": float(test_with_preds[home_mask]["correct"].mean()),
                    "away_accuracy": float(test_with_preds[away_mask]["correct"].mean()),
                }
            
            # Favorite vs underdog analysis
            if "odds_home_is_favorite" in test_data.columns:
                favorite_mask = test_with_preds["odds_home_is_favorite"] == 1
                underdog_mask = test_with_preds["odds_home_is_favorite"] == 0
                
                error_analysis["favorite_underdog_bias"] = {
                    "favorite_samples": int(favorite_mask.sum()),
                    "underdog_samples": int(underdog_mask.sum()),
                    "favorite_accuracy": float(test_with_preds[favorite_mask]["correct"].mean()),
                    "underdog_accuracy": float(test_with_preds[underdog_mask]["correct"].mean()),
                }
        
        return error_analysis
    
    def generate_report(
        self,
        test_data: pd.DataFrame,
        target_column: str = "target",
        exclude_columns: Optional[List[str]] = None,
        report_name: Optional[str] = None
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            test_data: Test data with features and target
            target_column: Name of target column
            exclude_columns: Columns to exclude from features
            report_name: Custom name for the report file
            
        Returns:
            Path to the saved report
            
        Raises:
            ValueError: If model is not set or target column not in test data
        """
        if self.model is None:
            raise ValueError("Model must be set before generating report")
        
        # Create report dictionary
        report = {
            "model_name": self.model.model_name,
            "model_version": self.model.model_version,
            "evaluation_time": datetime.now().isoformat(),
            "test_samples": len(test_data),
            "metrics": {},
            "error_analysis": {},
            "feature_importance": {}
        }
        
        # Get metrics
        try:
            metrics = self.evaluate(test_data, target_column, exclude_columns, detailed=True)
            report["metrics"] = metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            report["metrics_error"] = str(e)
        
        # Perform error analysis
        try:
            error_analysis = self.analyze_errors(test_data, target_column, exclude_columns)
            report["error_analysis"] = error_analysis
        except Exception as e:
            logger.error(f"Error in error analysis: {str(e)}")
            report["error_analysis_error"] = str(e)
        
        # Get feature importance
        try:
            feature_importance = self.model.get_feature_importance(top_n=50)
            report["feature_importance"] = feature_importance
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            report["feature_importance_error"] = str(e)
        
        # Generate filename if not provided
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"{self.model.model_name}_{self.model.model_version}_{timestamp}_report.json"
        
        # Save report
        report_path = os.path.join(self.report_dir, report_name)
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        return report_path
    
    @staticmethod
    def compare_models(
        model1: BasketballXGBoostModel,
        model2: BasketballXGBoostModel,
        test_data: pd.DataFrame,
        target_column: str = "target",
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two models on the same test data.
        
        Args:
            model1: First model to compare
            model2: Second model to compare
            test_data: Test data for evaluation
            target_column: Name of target column
            exclude_columns: Columns to exclude from features
            
        Returns:
            Dictionary with comparison results
        """
        # Set default exclude columns
        if exclude_columns is None:
            exclude_columns = ["game_id", "home_team_id", "away_team_id", "date"]
        
        # Create evaluators
        evaluator1 = ModelEvaluator(model1)
        evaluator2 = ModelEvaluator(model2)
        
        # Get metrics for both models
        metrics1 = evaluator1.evaluate(test_data, target_column, exclude_columns, detailed=True)
        metrics2 = evaluator2.evaluate(test_data, target_column, exclude_columns, detailed=True)
        
        # Create comparison
        comparison = {
            "model1": {
                "name": model1.model_name,
                "version": model1.model_version,
                "metrics": metrics1
            },
            "model2": {
                "name": model2.model_name,
                "version": model2.model_version,
                "metrics": metrics2
            },
            "differences": {}
        }
        
        # Calculate differences in key metrics
        key_metrics = ["accuracy", "auc", "precision", "recall", "f1_score"]
        
        for metric in key_metrics:
            if metric in metrics1 and metric in metrics2:
                diff = metrics1[metric] - metrics2[metric]
                pct_diff = diff / metrics2[metric] if metrics2[metric] != 0 else float('inf')
                
                comparison["differences"][metric] = {
                    "absolute": diff,
                    "percentage": pct_diff
                }
        
        # Add recommendation
        if comparison["differences"].get("accuracy", {}).get("absolute", 0) > 0:
            comparison["recommendation"] = "Model 1 is recommended (higher accuracy)"
        else:
            comparison["recommendation"] = "Model 2 is recommended (higher accuracy)"
        
        return comparison


def log_loss(y_true, y_pred, eps=1e-15):
    """
    Calculate log loss metric.
    
    Args:
        y_true: True values
        y_pred: Predicted probabilities
        eps: Small value to avoid log(0)
        
    Returns:
        Log loss value
    """
    # Ensure predictions are bounded away from 0 and 1
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
    # Compute log loss
    losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(losses)