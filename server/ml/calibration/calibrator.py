"""
Probability calibration module for basketball prediction models.

This module provides methods for calibrating prediction probabilities,
ensuring they reflect the true likelihood of outcomes.
"""
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

from app.core.logger import logger
from app.core.config import settings
from app.ml.models.xgboost_model import BasketballXGBoostModel


class ProbabilityCalibrator:
    """
    Calibrates prediction probabilities to reflect true outcome likelihoods.
    
    This class supports multiple calibration methods including:
    - Isotonic Regression
    - Platt Scaling (Logistic Regression)
    - Custom binning-based calibration
    """
    
    def __init__(
        self,
        method: str = "isotonic",
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        save_dir: Optional[str] = None
    ):
        """
        Initialize the probability calibrator.
        
        Args:
            method: Calibration method ('isotonic', 'platt', or 'binning')
            model_name: Name of the model to calibrate
            model_version: Version of the model
            save_dir: Directory for saving calibration models
        """
        self.method = method
        self.model_name = model_name
        self.model_version = model_version
        
        # Set calibration model
        self.calibration_model = None
        
        # Set save directory
        if save_dir is None:
            self.save_dir = os.path.join(settings.BASE_DIR, "ml", "calibration", "models")
        else:
            self.save_dir = save_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # For binning-based calibration
        self.bins = np.linspace(0, 1, 11)  # 10 bins by default
        self.bin_mapping = {}  # Maps bin index to calibrated probability
    
    def fit(
        self,
        uncalibrated_probs: np.ndarray,
        true_outcomes: np.ndarray
    ) -> None:
        """
        Fit calibration model using validation data.
        
        Args:
            uncalibrated_probs: Uncalibrated prediction probabilities
            true_outcomes: True binary outcomes (0 or 1)
            
        Raises:
            ValueError: If method is not supported
        """
        # Ensure inputs are numpy arrays
        uncalibrated_probs = np.array(uncalibrated_probs).reshape(-1, 1)
        true_outcomes = np.array(true_outcomes)
        
        if self.method == "isotonic":
            # Isotonic Regression (non-parametric, monotonic)
            self.calibration_model = IsotonicRegression(out_of_bounds="clip")
            self.calibration_model.fit(uncalibrated_probs.ravel(), true_outcomes)
            
        elif self.method == "platt":
            # Platt Scaling (Logistic Regression)
            self.calibration_model = LogisticRegression(C=1.0, solver='lbfgs')
            self.calibration_model.fit(uncalibrated_probs, true_outcomes)
            
        elif self.method == "binning":
            # Binning-based calibration
            digitized = np.digitize(uncalibrated_probs.ravel(), self.bins) - 1
            digitized = np.clip(digitized, 0, len(self.bins) - 2)
            
            # Calculate calibrated probability for each bin
            for i in range(len(self.bins) - 1):
                bin_mask = (digitized == i)
                if np.sum(bin_mask) > 0:
                    self.bin_mapping[i] = np.mean(true_outcomes[bin_mask])
                else:
                    # If no samples in the bin, use bin midpoint as fallback
                    self.bin_mapping[i] = (self.bins[i] + self.bins[i+1]) / 2
        
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
    
    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        """
        Calibrate prediction probabilities.
        
        Args:
            uncalibrated_probs: Uncalibrated prediction probabilities
            
        Returns:
            Calibrated probabilities
            
        Raises:
            ValueError: If calibration model is not fitted
        """
        if self.method in ["isotonic", "platt"] and self.calibration_model is None:
            raise ValueError("Calibration model not fitted")
            
        # Ensure input is numpy array
        uncalibrated_probs = np.array(uncalibrated_probs)
        
        if self.method == "isotonic":
            # Use isotonic regression
            return self.calibration_model.predict(uncalibrated_probs.reshape(-1, 1))
            
        elif self.method == "platt":
            # Use logistic regression
            return self.calibration_model.predict_proba(uncalibrated_probs.reshape(-1, 1))[:, 1]
            
        elif self.method == "binning":
            # Use bin mapping
            digitized = np.digitize(uncalibrated_probs.ravel(), self.bins) - 1
            digitized = np.clip(digitized, 0, len(self.bins) - 2)
            
            return np.array([self.bin_mapping[d] for d in digitized])
        
        # Fallback to original probabilities
        return uncalibrated_probs
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save calibration model.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where model was saved
            
        Raises:
            ValueError: If calibration model is not fitted
        """
        if self.method in ["isotonic", "platt"] and self.calibration_model is None:
            raise ValueError("No calibration model to save")
            
        if self.method == "binning" and not self.bin_mapping:
            raise ValueError("No calibration bins to save")
            
        # Generate filename if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_str = ""
            if self.model_name:
                model_str = f"{self.model_name}_"
                
            version_str = ""
            if self.model_version:
                version_str = f"{self.model_version}_"
                
            filepath = os.path.join(
                self.save_dir, 
                f"{model_str}{version_str}{self.method}_calibrator_{timestamp}.pkl"
            )
        
        # Save the model
        with open(filepath, "wb") as f:
            if self.method in ["isotonic", "platt"]:
                pickle.dump(self.calibration_model, f)
            else:
                pickle.dump(self.bin_mapping, f)
        
        logger.info(f"Calibration model saved to {filepath}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load calibration model.
        
        Args:
            filepath: Path to load the model from
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration model not found at {filepath}")
            
        with open(filepath, "rb") as f:
            if self.method in ["isotonic", "platt"]:
                self.calibration_model = pickle.load(f)
            else:
                self.bin_mapping = pickle.load(f)
                
        logger.info(f"Calibration model loaded from {filepath}")
    
    def evaluate(
        self, 
        uncalibrated_probs: np.ndarray, 
        true_outcomes: np.ndarray, 
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate calibration quality.
        
        Args:
            uncalibrated_probs: Uncalibrated prediction probabilities
            true_outcomes: True binary outcomes (0 or 1)
            n_bins: Number of bins for reliability diagram
            
        Returns:
            Dictionary with calibration metrics
        """
        # Ensure inputs are numpy arrays
        uncalibrated_probs = np.array(uncalibrated_probs).ravel()
        true_outcomes = np.array(true_outcomes)
        
        # Get calibrated probabilities
        calibrated_probs = self.calibrate(uncalibrated_probs)
        
        # Calculate reliability curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(
            true_outcomes, uncalibrated_probs, n_bins=n_bins
        )
        
        prob_true_cal, prob_pred_cal = calibration_curve(
            true_outcomes, calibrated_probs, n_bins=n_bins
        )
        
        # Calculate calibration error (root mean squared error)
        uncal_error = np.sqrt(np.mean((prob_true_uncal - prob_pred_uncal) ** 2))
        cal_error = np.sqrt(np.mean((prob_true_cal - prob_pred_cal) ** 2))
        
        # Calculate Expected Calibration Error (ECE)
        uncal_ece = self._expected_calibration_error(uncalibrated_probs, true_outcomes, n_bins)
        cal_ece = self._expected_calibration_error(calibrated_probs, true_outcomes, n_bins)
        
        return {
            "uncalibrated": {
                "reliability_curve": {
                    "prob_true": prob_true_uncal.tolist(),
                    "prob_pred": prob_pred_uncal.tolist()
                },
                "calibration_error": float(uncal_error),
                "expected_calibration_error": float(uncal_ece)
            },
            "calibrated": {
                "reliability_curve": {
                    "prob_true": prob_true_cal.tolist(),
                    "prob_pred": prob_pred_cal.tolist()
                },
                "calibration_error": float(cal_error),
                "expected_calibration_error": float(cal_ece)
            },
            "improvement": {
                "calibration_error_reduction": float(uncal_error - cal_error),
                "ece_reduction": float(uncal_ece - cal_ece)
            }
        }
    
    def _expected_calibration_error(
        self, 
        probs: np.ndarray, 
        true_outcomes: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            probs: Prediction probabilities
            true_outcomes: True binary outcomes (0 or 1)
            n_bins: Number of bins
            
        Returns:
            Expected Calibration Error
        """
        bin_indices = np.digitize(probs, np.linspace(0, 1, n_bins + 1)) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_sums = np.bincount(bin_indices, weights=probs, minlength=n_bins)
        bin_true = np.bincount(bin_indices, weights=true_outcomes, minlength=n_bins)
        
        # Avoid division by zero
        nonzero = bin_counts > 0
        bin_avg_pred = np.zeros(n_bins)
        bin_avg_true = np.zeros(n_bins)
        
        if np.any(nonzero):
            bin_avg_pred[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
            bin_avg_true[nonzero] = bin_true[nonzero] / bin_counts[nonzero]
        
        # Calculate ECE
        ece = np.sum(bin_counts[nonzero] / np.sum(bin_counts) * np.abs(bin_avg_pred[nonzero] - bin_avg_true[nonzero]))
        
        return ece
    
    @staticmethod
    def apply_temperature_scaling(
        probs: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.
        
        Args:
            probs: Prediction probabilities
            temperature: Temperature parameter (>0)
                         - T<1: Sharpen probabilities
                         - T=1: No change
                         - T>1: Soften probabilities
                         
        Returns:
            Temperature-scaled probabilities
            
        Raises:
            ValueError: If temperature is not positive
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
            
        # Apply temperature scaling (sharpening or softening)
        scaled_logits = np.log(probs / (1 - probs)) / temperature
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        
        return scaled_probs
    
    @staticmethod
    def optimize_temperature(
        probs: np.ndarray,
        true_outcomes: np.ndarray,
        metric: str = "ece"
    ) -> Tuple[float, float]:
        """
        Find optimal temperature parameter for calibration.
        
        Args:
            probs: Prediction probabilities
            true_outcomes: True binary outcomes (0 or 1)
            metric: Metric to optimize ('ece' or 'nll')
            
        Returns:
            Tuple of (optimal temperature, metric value)
        """
        from scipy.optimize import minimize_scalar
        
        # Define metric function
        def objective(temperature):
            scaled_probs = ProbabilityCalibrator.apply_temperature_scaling(probs, temperature)
            
            if metric == "ece":
                # Expected Calibration Error
                calibrator = ProbabilityCalibrator()
                ece = calibrator._expected_calibration_error(scaled_probs, true_outcomes)
                return ece
                
            else:  # Default to negative log likelihood
                # Clip probabilities to avoid log(0)
                eps = 1e-15
                scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
                
                # Calculate NLL
                nll = -np.mean(
                    true_outcomes * np.log(scaled_probs) + 
                    (1 - true_outcomes) * np.log(1 - scaled_probs)
                )
                return nll
        
        # Find optimal temperature
        result = minimize_scalar(
            objective,
            bounds=(0.1, 10.0),
            method="bounded"
        )
        
        return result.x, result.fun
    
    @staticmethod
    def calibrate_predictions(
        model: BasketballXGBoostModel,
        validation_data: pd.DataFrame,
        target_column: str = "target"
    ) -> ProbabilityCalibrator:
        """
        Create and fit a calibrator using model predictions on validation data.
        
        Args:
            model: Model to calibrate
            validation_data: Validation data for calibration
            target_column: Name of target column
            
        Returns:
            Fitted calibrator
        """
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
        
        # Create and fit calibrator
        calibrator = ProbabilityCalibrator(
            method="isotonic",
            model_name=model.model_name,
            model_version=model.model_version
        )
        
        calibrator.fit(uncalibrated_probs, y_val_numeric)
        
        return calibrator
