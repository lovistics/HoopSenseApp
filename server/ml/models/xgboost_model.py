"""
XGBoost model implementation for basketball predictions.
"""
import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from app.core.logger import logger
from app.core.config import settings

class BasketballXGBoostModel:
    """
    XGBoost model for basketball game predictions.
    """
    
    def __init__(
        self,
        model_name: str = "basketball_prediction_model",
        model_version: str = "v1.0",
        target_column: str = "target"
    ):
        """
        Initialize the XGBoost model.
        
        Args:
            model_name: Model name for saving/loading
            model_version: Model version
            target_column: Target column in training data
        """
        self.model_name = model_name
        self.model_version = model_version
        self.target_column = target_column
        self.model = None
        self.feature_names = []
        self.model_params = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.output_classes = []  # E.g., ["home", "away"]
        
        # Model parameters, can be overridden
        self.default_params = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error", "auc"],
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 100,
            "seed": 42
        }
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
        test_size: float = 0.2,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 20,
        save_model: bool = True,
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train the XGBoost model.
        
        Args:
            train_data: Training data DataFrame
            val_data: Optional validation data DataFrame
            test_size: Test size if val_data not provided
            params: Model hyperparameters (will use defaults if not provided)
            early_stopping_rounds: Number of rounds for early stopping
            save_model: Whether to save the model after training
            exclude_columns: Columns to exclude from features
            
        Returns:
            Dictionary of training results
        """
        # Store training start time
        training_start_time = datetime.now()
        
        if exclude_columns is None:
            exclude_columns = ["game_id", "home_team_id", "away_team_id", "date"]
        
        # Ensure target column is in the data
        if self.target_column not in train_data.columns:
            raise ValueError(f"Target column '{self.target_column}' not in training data")
        
        # Prepare data
        X_train, y_train = self._prepare_data(train_data, exclude_columns)
        
        # Prepare validation data if provided
        if val_data is not None:
            X_val, y_val = self._prepare_data(val_data, exclude_columns)
            eval_set = [(X_train, y_train), (X_val, y_val)]
        else:
            # Split training data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=test_size, random_state=42, stratify=y_train
            )
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Set model parameters
        self.model_params = params if params else self.default_params
        
        # Convert target to numeric for XGBoost
        y_train_numeric = self._prepare_target(y_train)
        y_val_numeric = self._prepare_target(y_val)
        
        eval_set = [(X_train, y_train_numeric), (X_val, y_val_numeric)]
        
        # Create DMatrix objects for faster training
        dtrain = xgb.DMatrix(X_train, label=y_train_numeric, feature_names=X_train.columns.tolist())
        dval = xgb.DMatrix(X_val, label=y_val_numeric, feature_names=X_train.columns.tolist())
        
        # Train model
        logger.info(f"Training XGBoost model with {len(X_train)} samples, {X_train.shape[1]} features")
        
        # Extract core params for XGBoost
        xgb_params = {k: v for k, v in self.model_params.items() 
                     if k not in ['n_estimators']}
        
        n_estimators = self.model_params.get('n_estimators', 100)
        
        # Train the model
        self.model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=10
        )
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance()
        
        # Evaluate on validation set
        val_predictions = self.predict(X_val)
        val_pred_proba = self.predict_proba(X_val)
        
        # Store metrics
        self.model_metrics = {
            "accuracy": accuracy_score(y_val, val_predictions),
            "log_loss": log_loss(y_val_numeric, val_pred_proba),
            "auc": roc_auc_score(y_val_numeric, val_pred_proba),
            "training_date": training_start_time.isoformat(),
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "feature_count": len(self.feature_names),
            "model_version": self.model_version
        }
        
        logger.info(f"Model training completed. Validation accuracy: {self.model_metrics['accuracy']:.4f}")
        
        # Save the model if requested
        if save_model:
            self.save()
        
        return {
            "model_metrics": self.model_metrics,
            "feature_importance": dict(sorted(self.feature_importance.items(), 
                                             key=lambda x: x[1], reverse=True)[:30]),
            "model_params": self.model_params,
            "output_classes": self.output_classes
        }
    
    def hyperparameter_tuning(
        self,
        train_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            train_data: Training data DataFrame
            param_grid: Parameter grid to search
            cv_folds: Number of cross-validation folds
            scoring: Metric to use for scoring
            exclude_columns: Columns to exclude from features
            
        Returns:
            Dictionary with best parameters and CV results
        """
        if exclude_columns is None:
            exclude_columns = ["game_id", "home_team_id", "away_team_id", "date"]
        
        # Prepare data
        X_train, y_train = self._prepare_data(train_data, exclude_columns)
        
        # Convert target to numeric if necessary
        if y_train.dtype == object:
            self.output_classes = sorted(y_train.unique())
            y_train_numeric = (y_train == self.output_classes[1]).astype(int)
        else:
            y_train_numeric = y_train
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'error', 'auc'],
            use_label_encoder=False
        )
        
        # Set up grid search
        logger.info(f"Starting hyperparameter tuning with {cv_folds} CV folds")
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            verbose=1,
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train_numeric)
        
        # Get best parameters
        best_params = grid_search.best_params_
        
        # Update model parameters
        self.model_params.update(best_params)
        
        # Return best parameters and CV results
        return {
            "best_params": best_params,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make class predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predicted classes
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Ensure X has the correct features
        X = self._validate_features(X)
        
        # Convert to DMatrix for prediction
        dmatrix = xgb.DMatrix(X, feature_names=X.columns.tolist())
        
        # Get probability predictions
        probabilities = self.model.predict(dmatrix)
        
        # Convert to class predictions
        if len(self.output_classes) == 2:
            # Binary classification
            predictions = (probabilities >= 0.5).astype(int)
            return np.array([self.output_classes[p] for p in predictions])
        else:
            # This is a binary model, but we'll handle the case where output_classes is empty
            return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        # Ensure X has the correct features
        X = self._validate_features(X)
        
        # Convert to DMatrix for prediction
        dmatrix = xgb.DMatrix(X, feature_names=X.columns.tolist())
        
        # Return probability predictions
        return self.model.predict(dmatrix)
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test data DataFrame
            exclude_columns: Columns to exclude from features
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
        
        if exclude_columns is None:
            exclude_columns = ["game_id", "home_team_id", "away_team_id", "date"]
        
        # Prepare test data
        X_test, y_test = self._prepare_data(test_data, exclude_columns)
        
        # Make predictions
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Convert target to numeric for metrics
        if y_test.dtype == object:
            if not self.output_classes:
                # If output_classes wasn't set, determine it from the test data
                self.output_classes = sorted(y_test.unique())
            
            y_test_numeric = (y_test == self.output_classes[1]).astype(int)
        else:
            y_test_numeric = y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        logloss = log_loss(y_test_numeric, probabilities)
        auc = roc_auc_score(y_test_numeric, probabilities)
        
        # Calculate class-specific metrics
        class_metrics = {}
        for cls in self.output_classes:
            cls_pred = (predictions == cls)
            cls_actual = (y_test == cls)
            cls_accuracy = (cls_pred == cls_actual).mean()
            
            class_metrics[cls] = {
                "accuracy": cls_accuracy,
                "count": cls_actual.sum(),
                "predicted": cls_pred.sum()
            }
        
        # Return all metrics
        return {
            "accuracy": accuracy,
            "log_loss": logloss,
            "auc": auc,
            "class_metrics": class_metrics,
            "test_samples": len(X_test)
        }
    
    def save(self, model_dir: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            model_dir: Directory to save the model
            
        Returns:
            Path to saved model directory
        """
        if self.model is None:
            raise ValueError("Model has not been trained")
        
        # Use default directory if not provided
        if model_dir is None:
            model_dir = os.path.join(settings.BASE_DIR, "ml", "models", "saved_models")
        
        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"{self.model_name}_{self.model_version}_{timestamp}")
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        model_file = os.path.join(model_path, "model.xgb")
        self.model.save_model(model_file)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_names": self.feature_names,
            "model_params": self.model_params,
            "model_metrics": self.model_metrics,
            "feature_importance": self.feature_importance,
            "output_classes": self.output_classes,
            "saved_at": timestamp
        }
        
        metadata_file = os.path.join(model_path, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load(
        self,
        model_path: str,
        use_latest: bool = False
    ) -> bool:
        """
        Load the model from disk.
        
        Args:
            model_path: Path to model directory or file
            use_latest: If True, load the latest model in the directory
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            # If model_path is a directory and use_latest is True, find the latest model
            if os.path.isdir(model_path) and use_latest:
                model_dirs = [os.path.join(model_path, d) for d in os.listdir(model_path)
                             if os.path.isdir(os.path.join(model_path, d)) and 
                             d.startswith(self.model_name)]
                
                if not model_dirs:
                    logger.error(f"No models found in {model_path}")
                    return False
                
                # Sort by timestamp in directory name
                model_dirs.sort(reverse=True)
                model_path = model_dirs[0]
                logger.info(f"Loading latest model from {model_path}")
            
            # If model_path is a directory, look for model.xgb file
            if os.path.isdir(model_path):
                model_file = os.path.join(model_path, "model.xgb")
                metadata_file = os.path.join(model_path, "metadata.json")
            else:
                # Assume model_path is the model file
                model_file = model_path
                metadata_file = os.path.join(os.path.dirname(model_path), "metadata.json")
            
            # Load model
            self.model = xgb.Booster()
            self.model.load_model(model_file)
            
            # Load metadata if available
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                
                self.model_name = metadata.get("model_name", self.model_name)
                self.model_version = metadata.get("model_version", self.model_version)
                self.feature_names = metadata.get("feature_names", [])
                self.model_params = metadata.get("model_params", {})
                self.model_metrics = metadata.get("model_metrics", {})
                self.feature_importance = metadata.get("feature_importance", {})
                self.output_classes = metadata.get("output_classes", ["0", "1"])
            
            logger.info(f"Model loaded successfully from {model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature importance
        """
        if not self.feature_importance:
            self.feature_importance = self._calculate_feature_importance()
        
        # Sort by importance
        sorted_importance = dict(sorted(self.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        # Return top N if specified
        if top_n is not None:
            return dict(list(sorted_importance.items())[:top_n])
        
        return sorted_importance
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        exclude_columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training or prediction.
        
        Args:
            data: Input DataFrame
            exclude_columns: Columns to exclude from features
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Handle categorical columns
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col not in exclude_columns and col != self.target_column:
                # One-hot encode categorical features
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
        
        # Extract features and target
        exclude_with_target = exclude_columns + [self.target_column]
        feature_cols = [col for col in df.columns if col not in exclude_with_target]
        
        X = df[feature_cols]
        y = df[self.target_column] if self.target_column in df.columns else None
        
        return X, y
    
    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare features for prediction.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Prepared features DataFrame
        """
        # Check if model has been trained
        if not self.feature_names:
            logger.warning("Model feature names not available. Using all provided features.")
            return X
        
        # Identify missing features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features in input data: {missing_features}. Will initialize with zeros.")
            
            # Add missing features as zero columns
            for feature in missing_features:
                X[feature] = 0
        
        # Extract only the needed features in the correct order
        return X[self.feature_names]
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance.
        
        Returns:
            Dictionary of feature importance
        """
        if self.model is None:
            logger.warning("Model not trained, cannot calculate feature importance")
            return {}
        
        # Get feature importance scores
        importance_scores = self.model.get_score(importance_type='gain')
        
        # If no scores are available, try other methods
        if not importance_scores:
            try:
                # Try getting feature scores by weight
                importance_scores = self.model.get_score(importance_type='weight')
            except Exception:
                logger.warning("Could not calculate feature importance")
                return {}
        
        # Create a dictionary with all features (including those with zero importance)
        importance_dict = {feature: 0.0 for feature in self.feature_names}
        
        # Update with actual importance values
        for feature, score in importance_scores.items():
            if feature in importance_dict:
                importance_dict[feature] = score
        
        # Normalize importance to sum to 1.0
        total_importance = sum(importance_dict.values())
        if total_importance > 0:
            importance_dict = {feature: score / total_importance 
                               for feature, score in importance_dict.items()}
        
        return importance_dict
    
    def _prepare_target(self, y):
        """
        Prepare target variable for XGBoost.
        
        Args:
            y: Target data
            
        Returns:
            Numeric target data and original target to numeric mapping
        """
        if y.dtype == object:
            # Store output classes if not already set
            if not self.output_classes:
                self.output_classes = sorted(y.unique())
            
            # Convert to 0/1 for binary classification
            y_numeric = (y == self.output_classes[1]).astype(int)
            return y_numeric
        else:
            return y