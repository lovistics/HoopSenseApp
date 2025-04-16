"""
Model monitoring module for basketball prediction models.

This module provides functionality for monitoring model performance in production,
detecting data drift, tracking prediction outcomes, and generating monitoring reports.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
from collections import defaultdict, deque
import asyncio
import logging
import pickle
from scipy import stats

from app.core.logger import logger
from app.core.config import settings
from app.ml.models.xgboost_model import BasketballXGBoostModel
from app.ml.models.evaluation import ModelEvaluator
from app.db.mongodb import get_database


class ModelMonitor:
    """
    Monitors model performance in production and detects data drift.
    
    This class handles:
    - Tracking prediction outcomes
    - Detecting data and concept drift
    - Generating monitoring reports
    - Alerting on performance degradation
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        reference_data: Optional[pd.DataFrame] = None,
        monitor_dir: Optional[str] = None,
        alert_threshold: float = 0.1,
        window_size: int = 1000
    ):
        """
        Initialize the model monitor.
        
        Args:
            model_name: Name of the model to monitor
            model_version: Version of the model, if None uses active version
            reference_data: Reference data for drift detection
            monitor_dir: Directory for storing monitoring data
            alert_threshold: Threshold for triggering alerts
            window_size: Size of the rolling window for metrics
        """
        self.model_name = model_name
        self.model_version = model_version
        self.reference_data = reference_data
        self.alert_threshold = alert_threshold
        
        # Set monitoring directory
        if monitor_dir is None:
            self.monitor_dir = os.path.join(settings.BASE_DIR, "ml", "monitoring", "data")
        else:
            self.monitor_dir = monitor_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self.monitor_dir, exist_ok=True)
        
        # Initialize performance tracking
        self.predictions = deque(maxlen=window_size)
        self.performance_metrics = {
            "accuracy": deque(maxlen=window_size),
            "auc": deque(maxlen=window_size),
            "data_drift_score": deque(maxlen=window_size),
            "latency_ms": deque(maxlen=window_size)
        }
        
        # Feature distribution statistics from reference data
        self.feature_stats = {}
        if reference_data is not None:
            self._compute_reference_statistics(reference_data)
        
        # Load historical monitoring data if available
        self._load_monitoring_state()
    
    def _compute_reference_statistics(self, data: pd.DataFrame) -> None:
        """
        Compute reference statistics for each feature.
        
        Args:
            data: Reference data to compute statistics from
        """
        numeric_cols = data.select_dtypes(include=np.number).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # For numeric features
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                self.feature_stats[col] = {
                    "type": "numeric",
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "q25": float(col_data.quantile(0.25)),
                    "q50": float(col_data.quantile(0.5)),
                    "q75": float(col_data.quantile(0.75))
                }
        
        # For categorical features
        for col in categorical_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts(normalize=True).to_dict()
                self.feature_stats[col] = {
                    "type": "categorical",
                    "distribution": {str(k): float(v) for k, v in value_counts.items()}
                }
    
    def _load_monitoring_state(self) -> None:
        """
        Load historical monitoring state if available.
        """
        state_path = os.path.join(self.monitor_dir, f"{self.model_name}_monitor_state.pkl")
        if os.path.exists(state_path):
            try:
                with open(state_path, "rb") as f:
                    state = pickle.load(f)
                
                self.feature_stats = state.get("feature_stats", self.feature_stats)
                self.predictions = state.get("predictions", self.predictions)
                self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
                
                logger.info(f"Loaded monitoring state for model {self.model_name}")
            except Exception as e:
                logger.error(f"Error loading monitoring state: {str(e)}")
    
    def _save_monitoring_state(self) -> None:
        """
        Save current monitoring state.
        """
        state_path = os.path.join(self.monitor_dir, f"{self.model_name}_monitor_state.pkl")
        
        try:
            state = {
                "feature_stats": self.feature_stats,
                "predictions": self.predictions,
                "performance_metrics": self.performance_metrics
            }
            
            with open(state_path, "wb") as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved monitoring state for model {self.model_name}")
        except Exception as e:
            logger.error(f"Error saving monitoring state: {str(e)}")
    
    async def log_prediction(
        self,
        features: Dict[str, Any],
        prediction: Any,
        probability: float,
        prediction_id: str,
        latency_ms: Optional[float] = None
    ) -> None:
        """
        Log a prediction for monitoring.
        
        Args:
            features: Features used for prediction
            prediction: Model prediction
            probability: Prediction probability/confidence
            prediction_id: Unique ID for the prediction
            latency_ms: Prediction latency in milliseconds
        """
        timestamp = datetime.now()
        
        prediction_record = {
            "id": prediction_id,
            "timestamp": timestamp,
            "features": features,
            "prediction": prediction,
            "probability": probability,
            "latency_ms": latency_ms,
            "actual_outcome": None,  # Will be updated when outcome is known
            "accuracy": None
        }
        
        # Add to in-memory queue
        self.predictions.append(prediction_record)
        
        # Add latency metric if available
        if latency_ms is not None:
            self.performance_metrics["latency_ms"].append(latency_ms)
        
        # Check for data drift
        if self.feature_stats:
            drift_score = self._calculate_drift_score(features)
            prediction_record["drift_score"] = drift_score
            self.performance_metrics["data_drift_score"].append(drift_score)
            
            # Alert if drift score exceeds threshold
            if drift_score > self.alert_threshold:
                logger.warning(f"Data drift detected: score={drift_score:.4f}, threshold={self.alert_threshold}")
                await self._log_drift_alert(prediction_id, drift_score, features)
        
        # Persist to database asynchronously
        await self._persist_prediction_log(prediction_record)
        
        # Periodically save monitoring state (every 100 predictions)
        if len(self.predictions) % 100 == 0:
            self._save_monitoring_state()
    
    async def _persist_prediction_log(self, prediction_record: Dict[str, Any]) -> None:
        """
        Persist prediction log to database.
        
        Args:
            prediction_record: Prediction record to persist
        """
        try:
            # Convert non-serializable objects
            record = {k: v for k, v in prediction_record.items()}
            record["timestamp"] = record["timestamp"].isoformat()
            
            # Remove raw features which might be too large for efficient storage
            # Keep only a subset of important features or summary statistics
            if "features" in record:
                # Only keep a subset of key features or summary statistics
                important_features = self._extract_important_features(record["features"])
                record["features"] = important_features
            
            # Persist to MongoDB
            db = await get_database()
            await db[f"model_predictions_{self.model_name}"].insert_one(record)
            
        except Exception as e:
            logger.error(f"Error persisting prediction log: {str(e)}")
    
    def _extract_important_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract important features for persistence.
        
        Args:
            features: Complete feature set
            
        Returns:
            Subset of important features
        """
        # This should be customized based on domain knowledge
        # For now, return the original features
        # In a real system, you might want to filter to only the most important features
        return features
    
    async def log_prediction_outcome(
        self,
        prediction_id: str,
        actual_outcome: Any
    ) -> None:
        """
        Log the actual outcome of a previous prediction.
        
        Args:
            prediction_id: ID of the prediction
            actual_outcome: Actual outcome value
        """
        # Update in-memory record
        for pred in self.predictions:
            if pred["id"] == prediction_id:
                pred["actual_outcome"] = actual_outcome
                pred["accuracy"] = 1.0 if pred["prediction"] == actual_outcome else 0.0
                
                # Update rolling accuracy
                self.performance_metrics["accuracy"].append(pred["accuracy"])
                
                # Alert if accuracy drops below threshold
                rolling_accuracy = np.mean(self.performance_metrics["accuracy"])
                if rolling_accuracy < (1 - self.alert_threshold) and len(self.performance_metrics["accuracy"]) >= 50:
                    logger.warning(f"Accuracy alert: rolling_accuracy={rolling_accuracy:.4f}, threshold={1-self.alert_threshold}")
                    await self._log_accuracy_alert(rolling_accuracy)
                
                break
        
        # Update database record
        try:
            db = await get_database()
            await db[f"model_predictions_{self.model_name}"].update_one(
                {"id": prediction_id},
                {"$set": {
                    "actual_outcome": actual_outcome,
                    "accuracy": 1.0 if self._get_prediction(prediction_id) == actual_outcome else 0.0
                }}
            )
        except Exception as e:
            logger.error(f"Error updating prediction outcome: {str(e)}")
    
    def _get_prediction(self, prediction_id: str) -> Any:
        """
        Get the prediction value for a given prediction ID.
        
        Args:
            prediction_id: ID of the prediction
            
        Returns:
            The prediction value or None if not found
        """
        for pred in self.predictions:
            if pred["id"] == prediction_id:
                return pred["prediction"]
        
        # If not found in memory, try the database
        return None
    
    def _calculate_drift_score(self, features: Dict[str, Any]) -> float:
        """
        Calculate drift score for a single prediction's features.
        
        Args:
            features: Features to calculate drift score for
            
        Returns:
            Drift score (0-1 where higher indicates more drift)
        """
        if not self.feature_stats:
            return 0.0
        
        feature_drift_scores = []
        
        for feat_name, feat_value in features.items():
            if feat_name in self.feature_stats:
                stats = self.feature_stats[feat_name]
                
                if stats["type"] == "numeric" and isinstance(feat_value, (int, float)):
                    # For numeric features, calculate z-score
                    if stats["std"] > 0:
                        z_score = abs((feat_value - stats["mean"]) / stats["std"])
                        # Convert to probability of observing this extreme a value
                        p_value = 2 * (1 - stats.norm.cdf(z_score))
                        feature_drift_scores.append(1 - p_value)
                
                elif stats["type"] == "categorical" and feat_value is not None:
                    # For categorical features, check if value exists and its probability
                    str_value = str(feat_value)
                    if str_value in stats["distribution"]:
                        # Less common values contribute more to drift score
                        feature_drift_scores.append(1 - stats["distribution"][str_value])
                    else:
                        # Unseen value
                        feature_drift_scores.append(1.0)
        
        # Combine feature drift scores (average for now, could be weighted)
        if feature_drift_scores:
            return np.mean(feature_drift_scores)
        else:
            return 0.0
    
    async def _log_drift_alert(
        self,
        prediction_id: str,
        drift_score: float,
        features: Dict[str, Any]
    ) -> None:
        """
        Log a data drift alert.
        
        Args:
            prediction_id: ID of the prediction
            drift_score: Calculated drift score
            features: Features that caused the drift
        """
        try:
            alert = {
                "type": "data_drift",
                "model_name": self.model_name,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "prediction_id": prediction_id,
                "drift_score": drift_score,
                "threshold": self.alert_threshold,
                "drifting_features": self._identify_drifting_features(features)
            }
            
            db = await get_database()
            await db["model_alerts"].insert_one(alert)
            
            # Here you would also integrate with external alerting systems if needed
            
        except Exception as e:
            logger.error(f"Error logging drift alert: {str(e)}")
    
    def _identify_drifting_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify which features are drifting the most.
        
        Args:
            features: Features to analyze
            
        Returns:
            Dictionary of drifting features with their drift scores
        """
        drifting_features = {}
        
        for feat_name, feat_value in features.items():
            if feat_name in self.feature_stats:
                stats = self.feature_stats[feat_name]
                
                if stats["type"] == "numeric" and isinstance(feat_value, (int, float)):
                    if stats["std"] > 0:
                        z_score = abs((feat_value - stats["mean"]) / stats["std"])
                        if z_score > 2:  # More than 2 standard deviations
                            drifting_features[feat_name] = {
                                "value": feat_value,
                                "expected_range": [
                                    stats["mean"] - 2 * stats["std"],
                                    stats["mean"] + 2 * stats["std"]
                                ],
                                "z_score": z_score
                            }
                
                elif stats["type"] == "categorical" and feat_value is not None:
                    str_value = str(feat_value)
                    if str_value not in stats["distribution"]:
                        drifting_features[feat_name] = {
                            "value": str_value,
                            "expected_values": list(stats["distribution"].keys()),
                            "new_category": True
                        }
                    elif stats["distribution"][str_value] < 0.01:  # Rare value
                        drifting_features[feat_name] = {
                            "value": str_value,
                            "expected_probability": stats["distribution"][str_value],
                            "rare_category": True
                        }
        
        return drifting_features
    
    async def _log_accuracy_alert(self, rolling_accuracy: float) -> None:
        """
        Log an accuracy alert.
        
        Args:
            rolling_accuracy: Current rolling accuracy
        """
        try:
            alert = {
                "type": "accuracy_drop",
                "model_name": self.model_name,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "rolling_accuracy": rolling_accuracy,
                "threshold": 1 - self.alert_threshold,
                "window_size": len(self.performance_metrics["accuracy"])
            }
            
            db = await get_database()
            await db["model_alerts"].insert_one(alert)
            
            # Here you would also integrate with external alerting systems if needed
            
        except Exception as e:
            logger.error(f"Error logging accuracy alert: {str(e)}")
    
    async def get_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a time period.
        
        Args:
            start_date: Start date for metrics
            end_date: End date for metrics
            
        Returns:
            Dictionary of performance metrics
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # Default to last 30 days
        
        try:
            db = await get_database()
            collection = db[f"model_predictions_{self.model_name}"]
            
            # Query for predictions in the date range with known outcomes
            query = {
                "timestamp": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()},
                "actual_outcome": {"$ne": None}
            }
            
            # Get all matching prediction records
            cursor = collection.find(query)
            predictions = await cursor.to_list(length=None)
            
            # Calculate metrics
            if not predictions:
                return {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "sample_count": 0,
                    "metrics": {}
                }
            
            # Calculate accuracy
            accuracy = sum(1 for p in predictions if p.get("accuracy", 0) > 0) / len(predictions)
            
            # Calculate average latency
            latencies = [p.get("latency_ms", 0) for p in predictions if p.get("latency_ms") is not None]
            avg_latency = sum(latencies) / len(latencies) if latencies else None
            
            # Calculate drift metrics
            drift_scores = [p.get("drift_score", 0) for p in predictions if p.get("drift_score") is not None]
            avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else None
            
            # Calculate confidence calibration
            bins = np.linspace(0.5, 1.0, num=6)  # 5 bins from 0.5 to 1.0
            calibration = {}
            
            for i in range(len(bins)-1):
                bin_preds = [p for p in predictions 
                            if p.get("probability", 0) >= bins[i] 
                            and p.get("probability", 0) < bins[i+1]
                            and p.get("actual_outcome") is not None]
                
                if bin_preds:
                    bin_accuracy = sum(1 for p in bin_preds if p.get("accuracy", 0) > 0) / len(bin_preds)
                    bin_avg_conf = sum(p.get("probability", 0) for p in bin_preds) / len(bin_preds)
                    
                    calibration[f"{bins[i]:.1f}-{bins[i+1]:.1f}"] = {
                        "count": len(bin_preds),
                        "avg_confidence": bin_avg_conf,
                        "accuracy": bin_accuracy,
                        "calibration_error": abs(bin_avg_conf - bin_accuracy)
                    }
            
            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "sample_count": len(predictions),
                "metrics": {
                    "accuracy": accuracy,
                    "avg_latency_ms": avg_latency,
                    "avg_drift_score": avg_drift,
                    "calibration": calibration
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "error": str(e)
            }
    
    async def get_alerts(
        self,
        alert_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get alerts for a time period.
        
        Args:
            alert_type: Type of alerts to get (None for all)
            start_date: Start date for alerts
            end_date: End date for alerts
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)  # Default to last 30 days
        
        try:
            db = await get_database()
            collection = db["model_alerts"]
            
            # Build query
            query = {
                "model_name": self.model_name,
                "timestamp": {"$gte": start_date.isoformat(), "$lte": end_date.isoformat()}
            }
            if alert_type:
                query["type"] = alert_type
            
            # Get alerts
            cursor = collection.find(query).sort("timestamp", -1).limit(limit)
            alerts = await cursor.to_list(length=limit)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {str(e)}")
            return []
    
    async def generate_daily_report(self, date: Optional[datetime] = None) -> str:
        """
        Generate daily model monitoring report.
        
        Args:
            date: Date to generate report for, defaults to yesterday
            
        Returns:
            Path to saved report
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)
        
        start_date = datetime(date.year, date.month, date.day, 0, 0, 0)
        end_date = datetime(date.year, date.month, date.day, 23, 59, 59)
        
        # Get performance metrics
        metrics = await self.get_performance_metrics(start_date, end_date)
        
        # Get alerts
        alerts = await self.get_alerts(None, start_date, end_date)
        
        # Create report
        report = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "report_date": date.strftime("%Y-%m-%d"),
            "generated_at": datetime.now().isoformat(),
            "metrics": metrics,
            "alerts": alerts,
            "summary": self._generate_report_summary(metrics, alerts)
        }
        
        # Save report
        report_path = os.path.join(
            self.monitor_dir, 
            f"{self.model_name}_{date.strftime('%Y%m%d')}_report.json"
        )
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Daily report saved to {report_path}")
        
        return report_path
    
    def _generate_report_summary(
        self,
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a summary of the monitoring report.
        
        Args:
            metrics: Performance metrics
            alerts: Alerts
            
        Returns:
            Report summary
        """
        # Count alerts by type
        alert_counts = {}
        for alert in alerts:
            alert_type = alert.get("type", "unknown")
            if alert_type not in alert_counts:
                alert_counts[alert_type] = 0
            alert_counts[alert_type] += 1
        
        # Determine overall status
        accuracy = metrics.get("metrics", {}).get("accuracy", 0)
        avg_drift = metrics.get("metrics", {}).get("avg_drift_score", 0)
        
        if len(alerts) > 10 or (accuracy and accuracy < 0.5) or (avg_drift and avg_drift > 0.3):
            status = "critical"
        elif len(alerts) > 5 or (accuracy and accuracy < 0.7) or (avg_drift and avg_drift > 0.2):
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "sample_count": metrics.get("sample_count", 0),
            "accuracy": metrics.get("metrics", {}).get("accuracy"),
            "alerts": len(alerts),
            "alert_counts": alert_counts,
            "recommendations": self._generate_recommendations(metrics, alerts, status)
        }
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]],
        status: str
    ) -> List[str]:
        """
        Generate recommendations based on monitoring data.
        
        Args:
            metrics: Performance metrics
            alerts: Alerts
            status: Overall status
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Add recommendations based on status
        if status == "critical":
            recommendations.append("Consider retraining the model with recent data")
            recommendations.append("Investigate recent data quality issues")
            
        # Add recommendations based on metrics
        accuracy = metrics.get("metrics", {}).get("accuracy", 0)
        if accuracy and accuracy < 0.7:
            recommendations.append("Review feature engineering and model architecture")
        
        # Add recommendations based on calibration
        calibration = metrics.get("metrics", {}).get("calibration", {})
        poorly_calibrated = [
            bin_range for bin_range, bin_data in calibration.items()
            if bin_data.get("calibration_error", 0) > 0.1 and bin_data.get("count", 0) > 10
        ]
        
        if poorly_calibrated:
            ranges_str = ", ".join(poorly_calibrated)
            recommendations.append(f"Recalibrate model confidence in ranges: {ranges_str}")
        
        # Add recommendations based on specific alert types
        data_drift_alerts = [a for a in alerts if a.get("type") == "data_drift"]
        if len(data_drift_alerts) > 5:
            drifting_features = self._identify_common_drifting_features(data_drift_alerts)
            if drifting_features:
                features_str = ", ".join(drifting_features[:3])
                recommendations.append(f"Review feature distributions for: {features_str}")
        
        return recommendations
    
    def _identify_common_drifting_features(
        self,
        data_drift_alerts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify commonly drifting features from alerts.
        
        Args:
            data_drift_alerts: List of data drift alerts
            
        Returns:
            List of commonly drifting features
        """
        feature_counts = defaultdict(int)
        
        for alert in data_drift_alerts:
            drifting_features = alert.get("drifting_features", {})
            for feature in drifting_features:
                feature_counts[feature] += 1
        
        # Sort by count
        sorted_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [feature for feature, count in sorted_features]