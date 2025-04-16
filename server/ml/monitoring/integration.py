"""
Integration module for connecting model evaluation, monitoring, and calibration.

This module provides utilities for creating an end-to-end model
lifecycle management workflow, connecting prediction services with
model evaluation, monitoring, and calibration components.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import uuid
import os

from app.core.logger import logger
from app.core.config import settings
from app.ml.models.xgboost_model import BasketballXGBoostModel
from app.ml.models.model_registry import ModelRegistry
from app.ml.models.evaluation import ModelEvaluator
from app.ml.monitoring.model_monitor import ModelMonitor
from app.ml.calibration.calibrator import ProbabilityCalibrator
from app.db.mongodb import get_database


class ModelLifecycleManager:
    """
    Manages the complete lifecycle of basketball prediction models.
    
    This class integrates:
    - Model evaluation
    - Monitoring
    - Calibration
    - A/B testing
    - Automated retraining
    """
    
    def __init__(
        self,
        model_name: str = "basketball_prediction",
        registry: Optional[ModelRegistry] = None,
        evaluator: Optional[ModelEvaluator] = None,
        monitor: Optional[ModelMonitor] = None,
        calibrator: Optional[ProbabilityCalibrator] = None
    ):
        """
        Initialize the model lifecycle manager.
        
        Args:
            model_name: Base name of the model
            registry: Model registry (or None to create new)
            evaluator: Model evaluator (or None to create new)
            monitor: Model monitor (or None to create new)
            calibrator: Probability calibrator (or None to create new)
        """
        self.model_name = model_name
        
        # Initialize components
        self.registry = registry or ModelRegistry()
        self.evaluator = evaluator or ModelEvaluator()
        self.monitor = monitor or ModelMonitor(model_name=model_name)
        self.calibrator = calibrator or ProbabilityCalibrator(model_name=model_name)
        
        # Current active model and version
        self.current_model = None
        self.current_version = None
        
        # A/B testing configuration
        self.ab_testing_enabled = False
        self.ab_testing_ratio = 0.0  # Percent of traffic to new model
        self.candidate_model = None
        self.candidate_version = None
        
        # Initialize performance tracking
        self.performance_history = {}
    
    async def initialize(self) -> None:
        """
        Initialize the lifecycle manager by loading the active model.
        """
        try:
            # Get active model from registry
            self.current_model, self.current_version = await self.registry.get_active_model(
                self.model_name
            )
            
            if self.current_model:
                # Set model in evaluator
                self.evaluator.set_model(self.current_model)
                
                # Update monitor with correct version
                self.monitor.model_version = self.current_version
                
                logger.info(
                    f"Initialized model lifecycle manager with model: "
                    f"{self.model_name} (version: {self.current_version})"
                )
            else:
                logger.warning(f"No active model found for {self.model_name}")
                
        except Exception as e:
            logger.error(f"Error initializing model lifecycle manager: {str(e)}")
    
    async def log_prediction_event(
        self,
        features: Dict[str, Any],
        prediction: Any,
        probability: float,
        prediction_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None
    ) -> str:
        """
        Log a prediction event for monitoring.
        
        Args:
            features: Features used for prediction
            prediction: Model prediction
            probability: Prediction probability/confidence
            prediction_id: Unique ID for the prediction (or None to generate)
            user_id: User ID associated with the prediction
            context: Additional context information
            latency_ms: Prediction latency in milliseconds
            
        Returns:
            Prediction ID
        """
        # Generate prediction ID if not provided
        if prediction_id is None:
            prediction_id = str(uuid.uuid4())
        
        # Add additional metadata
        metadata = {
            "user_id": user_id,
            "model_name": self.model_name,
            "model_version": self.current_version,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        # If using A/B testing, record which model made the prediction
        if self.ab_testing_enabled and hasattr(self, "_used_candidate_model"):
            metadata["ab_test"] = {
                "is_candidate": self._used_candidate_model,
                "model_version": self.candidate_version if self._used_candidate_model else self.current_version
            }
            
            # Remove temporary attribute
            if hasattr(self, "_used_candidate_model"):
                delattr(self, "_used_candidate_model")
        
        # Log to monitor
        await self.monitor.log_prediction(
            features=features,
            prediction=prediction,
            probability=probability,
            prediction_id=prediction_id,
            latency_ms=latency_ms
        )
        
        # Store additional metadata in MongoDB
        try:
            db = await get_database()
            await db["prediction_metadata"].insert_one({
                "prediction_id": prediction_id,
                **metadata
            })
        except Exception as e:
            logger.error(f"Error storing prediction metadata: {str(e)}")
        
        return prediction_id
    
    async def log_prediction_outcome(
        self,
        prediction_id: str,
        actual_outcome: Any,
        outcome_time: Optional[datetime] = None
    ) -> None:
        """
        Log the actual outcome of a previous prediction.
        
        Args:
            prediction_id: ID of the prediction
            actual_outcome: Actual outcome value
            outcome_time: Time when the outcome was determined
        """
        # Log to monitor
        await self.monitor.log_prediction_outcome(
            prediction_id=prediction_id,
            actual_outcome=actual_outcome
        )
        
        # Update metadata in MongoDB
        try:
            db = await get_database()
            await db["prediction_metadata"].update_one(
                {"prediction_id": prediction_id},
                {"$set": {
                    "actual_outcome": actual_outcome,
                    "outcome_time": outcome_time or datetime.now().isoformat()
                }}
            )
        except Exception as e:
            logger.error(f"Error updating prediction outcome metadata: {str(e)}")
    
    async def make_prediction(
        self,
        features: Dict[str, Any],
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        calibrate: bool = True
    ) -> Dict[str, Any]:
        """
        Make a prediction using the current model, with monitoring.
        
        Args:
            features: Features for prediction
            user_id: User ID associated with the prediction
            context: Additional context information
            calibrate: Whether to calibrate the probability
            
        Returns:
            Dictionary with prediction, probability, and prediction_id
            
        Raises:
            ValueError: If no active model is available
        """
        if self.current_model is None:
            raise ValueError("No active model available for prediction")
        
        # Start timing
        start_time = datetime.now()
        
        # Determine which model to use for A/B testing
        model_to_use = self.current_model
        
        if self.ab_testing_enabled and self.candidate_model is not None:
            # Simple random assignment
            if np.random.random() < self.ab_testing_ratio:
                model_to_use = self.candidate_model
                self._used_candidate_model = True
            else:
                self._used_candidate_model = False
        
        # Convert features to DataFrame for model
        features_df = pd.DataFrame([features])
        
        # Make prediction
        try:
            prediction = model_to_use.predict(features_df)[0]
            probability = model_to_use.predict_proba(features_df)[0]
            
            # Apply calibration if requested
            if calibrate and self.calibrator and hasattr(self.calibrator, "calibration_model"):
                probability = float(self.calibrator.calibrate(np.array([probability]))[0])
            
            # Calculate latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log prediction
            prediction_id = await self.log_prediction_event(
                features=features,
                prediction=prediction,
                probability=probability,
                user_id=user_id,
                context=context,
                latency_ms=latency_ms
            )
            
            return {
                "prediction": prediction,
                "probability": float(probability),
                "prediction_id": prediction_id,
                "model_version": self.current_version,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    async def evaluate_model(
        self,
        test_data: pd.DataFrame,
        target_column: str = "target",
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test data with features and target
            target_column: Name of target column
            model_version: Version of model to evaluate (or None for current)
            
        Returns:
            Evaluation metrics
            
        Raises:
            ValueError: If model version not found
        """
        if model_version is None:
            model_version = self.current_version
            model = self.current_model
        else:
            # Load specified model version
            model, _ = await self.registry.get_model(self.model_name, model_version)
            
            if model is None:
                raise ValueError(f"Model version {model_version} not found")
        
        # Set model in evaluator
        self.evaluator.set_model(model)
        
        # Run evaluation
        metrics = self.evaluator.evaluate(test_data, target_column=target_column, detailed=True)
        
        # Store metrics in performance history
        self.performance_history[model_version] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "test_samples": len(test_data)
        }
        
        # Store metrics in MongoDB
        try:
            db = await get_database()
            await db["model_performance"].insert_one({
                "model_name": self.model_name,
                "model_version": model_version,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "test_samples": len(test_data)
            })
        except Exception as e:
            logger.error(f"Error storing model performance: {str(e)}")
        
        return metrics
    
    async def generate_report(
        self,
        test_data: pd.DataFrame,
        target_column: str = "target",
        model_version: Optional[str] = None,
        report_name: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_data: Test data with features and target
            target_column: Name of target column
            model_version: Version of model to evaluate (or None for current)
            report_name: Custom name for the report file
            
        Returns:
            Path to the saved report
            
        Raises:
            ValueError: If model version not found
        """
        if model_version is None:
            model_version = self.current_version
            model = self.current_model
        else:
            # Load specified model version
            model, _ = await self.registry.get_model(self.model_name, model_version)
            
            if model is None:
                raise ValueError(f"Model version {model_version} not found")
        
        # Set model in evaluator
        self.evaluator.set_model(model)
        
        # Generate report
        report_path = self.evaluator.generate_report(test_data, target_column, report_name=report_name)
        
        return report_path
    
    async def fit_calibrator(
        self,
        validation_data: pd.DataFrame,
        target_column: str = "target",
        method: str = "isotonic",
        model_version: Optional[str] = None
    ) -> None:
        """
        Fit probability calibrator using validation data.
        
        Args:
            validation_data: Validation data with features and target
            target_column: Name of target column
            method: Calibration method ('isotonic', 'platt', or 'binning')
            model_version: Version of model to calibrate (or None for current)
            
        Raises:
            ValueError: If model version not found
        """
        if model_version is None:
            model_version = self.current_version
            model = self.current_model
        else:
            # Load specified model version
            model, _ = await self.registry.get_model(self.model_name, model_version)
            
            if model is None:
                raise ValueError(f"Model version {model_version} not found")
        
        # Create calibrator
        self.calibrator = ProbabilityCalibrator(
            method=method,
            model_name=self.model_name,
            model_version=model_version
        )
        
        # Prepare data
        X_val = validation_data.drop(target_column, axis=1)
        y_val = validation_data[target_column]
        
        # Convert target to numeric if necessary
        if y_val.dtype == object:
            output_classes = sorted(y_val.unique())
            y_val_numeric = (y_val == output_classes[1]).astype(int)
        else:
            y_val_numeric = y_val
        
        # Get uncalibrated probabilities
        uncalibrated_probs = model.predict_proba(X_val)
        
        # Fit calibrator
        self.calibrator.fit(uncalibrated_probs, y_val_numeric)
        
        # Save calibrator
        calibrator_path = self.calibrator.save()
        
        logger.info(f"Fitted and saved calibrator to {calibrator_path}")
    
    async def setup_ab_testing(
        self,
        candidate_version: str,
        traffic_ratio: float = 0.1
    ) -> None:
        """
        Setup A/B testing between current and candidate models.
        
        Args:
            candidate_version: Version of candidate model
            traffic_ratio: Fraction of traffic to route to candidate
            
        Raises:
            ValueError: If candidate model not found
        """
        # Load candidate model
        candidate_model, _ = await self.registry.get_model(self.model_name, candidate_version)
        
        if candidate_model is None:
            raise ValueError(f"Candidate model version {candidate_version} not found")
        
        # Setup A/B testing
        self.candidate_model = candidate_model
        self.candidate_version = candidate_version
        self.ab_testing_ratio = traffic_ratio
        self.ab_testing_enabled = True
        
        logger.info(
            f"A/B testing enabled: {self.current_version} vs {candidate_version} "
            f"with {traffic_ratio:.1%} traffic to candidate"
        )
    
    async def stop_ab_testing(self) -> None:
        """
        Stop A/B testing and return to single model.
        """
        self.ab_testing_enabled = False
        self.ab_testing_ratio = 0.0
        self.candidate_model = None
        self.candidate_version = None
        
        logger.info("A/B testing disabled")
    
    async def compare_ab_test_results(self) -> Dict[str, Any]:
        """
        Compare results of A/B testing between models.
        
        Returns:
            Comparison metrics
            
        Raises:
            ValueError: If A/B testing not enabled
        """
        if not self.ab_testing_enabled:
            raise ValueError("A/B testing not enabled")
        
        try:
            db = await get_database()
            
            # Get prediction metadata with outcomes
            cursor = db["prediction_metadata"].find({
                "model_name": self.model_name,
                "actual_outcome": {"$exists": True},
                "ab_test": {"$exists": True}
            })
            
            predictions = await cursor.to_list(length=None)
            
            if not predictions:
                return {
                    "error": "No completed predictions with outcomes found for A/B test"
                }
            
            # Separate predictions by model
            control_preds = [p for p in predictions if not p.get("ab_test", {}).get("is_candidate", False)]
            treatment_preds = [p for p in predictions if p.get("ab_test", {}).get("is_candidate", False)]
            
            # Calculate metrics for each group
            control_metrics = self._calculate_group_metrics(control_preds)
            treatment_metrics = self._calculate_group_metrics(treatment_preds)
            
            # Calculate statistical significance
            p_value = self._calculate_significance(control_preds, treatment_preds)
            
            # Get versions
            control_version = self.current_version
            treatment_version = self.candidate_version
            
            return {
                "control": {
                    "version": control_version,
                    "sample_count": len(control_preds),
                    "metrics": control_metrics
                },
                "treatment": {
                    "version": treatment_version,
                    "sample_count": len(treatment_preds),
                    "metrics": treatment_metrics
                },
                "comparison": {
                    "accuracy_diff": treatment_metrics["accuracy"] - control_metrics["accuracy"],
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "recommended_action": self._get_recommendation(
                        control_metrics, treatment_metrics, p_value
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error comparing A/B test results: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_group_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate metrics for a group of predictions.
        
        Args:
            predictions: List of prediction records
            
        Returns:
            Dictionary of metrics
        """
        if not predictions:
            return {
                "accuracy": 0.0,
                "avg_probability": 0.0,
                "calibration_error": 0.0
            }
        
        # Calculate accuracy
        correct = [
            p for p in predictions 
            if p.get("prediction") == p.get("actual_outcome")
        ]
        accuracy = len(correct) / len(predictions) if predictions else 0
        
        # Calculate average probability
        probabilities = [p.get("probability", 0) for p in predictions]
        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0
        
        # Calculate calibration error
        calibration_error = abs(accuracy - avg_probability)
        
        return {
            "accuracy": accuracy,
            "avg_probability": avg_probability,
            "calibration_error": calibration_error
        }
    
    def _calculate_significance(
        self,
        control_preds: List[Dict[str, Any]],
        treatment_preds: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate statistical significance of difference between groups.
        
        Args:
            control_preds: Control group predictions
            treatment_preds: Treatment group predictions
            
        Returns:
            p-value from statistical test
        """
        from scipy import stats
        
        # Convert to binary outcomes for statistical test
        control_outcomes = [
            1 if p.get("prediction") == p.get("actual_outcome") else 0
            for p in control_preds
        ]
        
        treatment_outcomes = [
            1 if p.get("prediction") == p.get("actual_outcome") else 0
            for p in treatment_preds
        ]
        
        # Perform two-sample t-test
        _, p_value = stats.ttest_ind(control_outcomes, treatment_outcomes, equal_var=False)
        
        return p_value
    
    def _get_recommendation(
        self,
        control_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        p_value: float
    ) -> str:
        """
        Get recommendation based on A/B test results.
        
        Args:
            control_metrics: Control group metrics
            treatment_metrics: Treatment group metrics
            p_value: Statistical significance
            
        Returns:
            Recommendation string
        """
        if p_value >= 0.05:
            return "Continue testing - results not statistically significant"
            
        if treatment_metrics["accuracy"] > control_metrics["accuracy"]:
            if treatment_metrics["accuracy"] - control_metrics["accuracy"] > 0.05:
                return "Promote candidate model to production"
            else:
                return "Consider promoting candidate model (small improvement)"
        else:
            return "Keep current model - candidate not better"
    
    async def promote_candidate_model(self) -> None:
        """
        Promote candidate model to production.
        
        Raises:
            ValueError: If A/B testing not enabled
        """
        if not self.ab_testing_enabled or self.candidate_model is None:
            raise ValueError("A/B testing not active or no candidate model")
        
        # Set candidate as active model
        await self.registry.set_active_model(
            self.model_name,
            self.candidate_version
        )
        
        # Update current model
        self.current_model = self.candidate_model
        self.current_version = self.candidate_version
        
        # Reset A/B testing
        await self.stop_ab_testing()
        
        # Update monitor
        self.monitor.model_version = self.current_version
        
        logger.info(f"Promoted model {self.model_name} version {self.current_version} to production")
    
    async def schedule_daily_tasks(self) -> None:
        """
        Schedule daily monitoring and maintenance tasks.
        """
        while True:
            try:
                # Generate daily report
                await self.monitor.generate_daily_report()
                
                # Check for model drift alerts
                alerts = await self.monitor.get_alerts(
                    start_date=datetime.now() - timedelta(days=1)
                )
                
                if alerts:
                    logger.warning(f"Found {len(alerts)} new alerts in the last 24 hours")
                    
                    # TODO: Add more sophisticated handling
                    
                # Wait until next day
                tomorrow = datetime.now() + timedelta(days=1)
                tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0)
                seconds_to_wait = (tomorrow - datetime.now()).total_seconds()
                
                await asyncio.sleep(seconds_to_wait)
                
            except Exception as e:
                logger.error(f"Error in daily tasks: {str(e)}")
                # Wait an hour and try again
                await asyncio.sleep(3600)


async def create_model_lifecycle_manager() -> ModelLifecycleManager:
    """
    Factory function to create and initialize a model lifecycle manager.
    
    Returns:
        Initialized ModelLifecycleManager
    """
    manager = ModelLifecycleManager()
    await manager.initialize()
    return manager
