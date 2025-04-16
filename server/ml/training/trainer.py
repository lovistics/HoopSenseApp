"""
Model training pipeline for basketball prediction models.

This module orchestrates the complete model training process,
from data preparation to model registration, evaluation, and deployment.
"""
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import logging
import asyncio
from functools import wraps

from app.core.logger import logger
from app.core.config import settings
from app.ml.models.xgboost_model import BasketballXGBoostModel
from app.ml.models.model_registry import ModelRegistry
from app.ml.models.evaluation import ModelEvaluator
from app.ml.calibration.calibrator import ProbabilityCalibrator
from app.ml.features.team_features import TeamFeatureExtractor
from app.ml.features.odds_features import OddsFeatureExtractor
from app.db.mongodb import get_database
from app.data.repositories.games_repository import GamesRepository
from app.data.repositories.teams_repository import TeamsRepository
from app.data.repositories.odds_repository import OddsRepository


# Helper function to run async functions in sync context
def run_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


class ModelTrainingConfig:
    """
    Configuration for model training pipeline.
    """
    
    def __init__(
        self,
        model_name: str = "basketball_prediction_model",
        model_version: Optional[str] = None,
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        hyperparameter_tuning: bool = False,
        register_model: bool = True,
        calibrate_probabilities: bool = True,
        save_artifacts: bool = True,
        feature_selection: bool = True,
        description: str = "",
        tags: List[str] = [],
        **model_params
    ):
        """
        Initialize training configuration.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model, auto-generated if None
            data_start_date: Start date for training data (YYYY-MM-DD)
            data_end_date: End date for training data (YYYY-MM-DD)
            test_size: Fraction of data to use for testing
            validation_size: Fraction of data to use for validation
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            register_model: Whether to register the model in the registry
            calibrate_probabilities: Whether to calibrate prediction probabilities
            save_artifacts: Whether to save training artifacts
            feature_selection: Whether to perform feature selection
            description: Model description
            tags: Model tags
            **model_params: Additional model parameters
        """
        self.model_name = model_name
        
        # Generate model version if not provided
        if model_version is None:
            self.model_version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
        else:
            self.model_version = model_version
        
        # Set date range for training data
        self.data_start_date = data_start_date
        self.data_end_date = data_end_date
        
        # Set data splitting parameters
        self.test_size = test_size
        self.validation_size = validation_size
        
        # Set training options
        self.hyperparameter_tuning = hyperparameter_tuning
        self.register_model = register_model
        self.calibrate_probabilities = calibrate_probabilities
        self.save_artifacts = save_artifacts
        self.feature_selection = feature_selection
        
        # Set model metadata
        self.description = description
        self.tags = tags
        
        # Set model parameters (overrides defaults)
        self.model_params = {
            # Default model parameters
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error", "auc"],
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 200,
            "seed": 42,
            **model_params  # Override with provided parameters
        }
        
        # Set directories
        self.artifacts_dir = os.path.join(
            settings.BASE_DIR, "ml", "training", "artifacts", 
            f"{self.model_name}_{self.model_version}"
        )
        
        # Create artifacts directory if saving artifacts
        if self.save_artifacts:
            os.makedirs(self.artifacts_dir, exist_ok=True)
    
    def save(self) -> str:
        """
        Save the training configuration to a JSON file.
        
        Returns:
            Path to saved configuration file
        """
        if not self.save_artifacts:
            return ""
            
        config_path = os.path.join(self.artifacts_dir, "training_config.json")
        
        # Create a dictionary of the configuration
        config_dict = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "data_start_date": self.data_start_date,
            "data_end_date": self.data_end_date,
            "test_size": self.test_size,
            "validation_size": self.validation_size,
            "hyperparameter_tuning": self.hyperparameter_tuning,
            "register_model": self.register_model,
            "calibrate_probabilities": self.calibrate_probabilities,
            "save_artifacts": self.save_artifacts,
            "feature_selection": self.feature_selection,
            "description": self.description,
            "tags": self.tags,
            "model_params": self.model_params,
            "created_at": datetime.now().isoformat()
        }
        
        # Save the configuration
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        return config_path


class ModelTrainingPipeline:
    """
    Pipeline for training basketball prediction models.
    """
    
    def __init__(
        self,
        config: Optional[ModelTrainingConfig] = None
    ):
        """
        Initialize the training pipeline.
        
        Args:
            config: Training configuration
        """
        self.config = config or ModelTrainingConfig()
        
        # Initialize components
        self.model = BasketballXGBoostModel(
            model_name=self.config.model_name,
            model_version=self.config.model_version
        )
        
        self.registry = ModelRegistry()
        self.evaluator = ModelEvaluator(model=self.model)
        
        # Initialize feature extractors
        self.team_feature_extractor = TeamFeatureExtractor()
        self.odds_feature_extractor = OddsFeatureExtractor()
        
        # Initialize repositories
        self.games_repository = GamesRepository()
        self.teams_repository = TeamsRepository()
        self.odds_repository = OddsRepository()
        
        # Training metrics and artifacts
        self.training_metrics = {}
        self.feature_importance = {}
        self.artifacts = {}
    
    async def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for model training.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("Preparing training data")
        
        # Fetch games data
        games = await self.games_repository.get_games(
            start_date=self.config.data_start_date,
            end_date=self.config.data_end_date,
            include_teams=True,
            include_odds=True,
            completed_only=True  # Only include games with known outcomes
        )
        
        if not games:
            raise ValueError("No games data available for training")
        
        logger.info(f"Fetched {len(games)} games for training")
        
        # Convert to DataFrame
        games_df = pd.DataFrame([game.dict() for game in games])
        
        # Extract features
        features_df = await self._extract_features(games_df)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split into train+val and test
        train_val_data, test_data = train_test_split(
            features_df, 
            test_size=self.config.test_size,
            random_state=42,
            stratify=features_df["target"]
        )
        
        # Then split train+val into train and val
        val_size_relative = self.config.validation_size / (1 - self.config.test_size)
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_relative,
            random_state=42,
            stratify=train_val_data["target"]
        )
        
        logger.info(f"Data split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test samples")
        
        # Save datasets if configured
        if self.config.save_artifacts:
            train_path = os.path.join(self.config.artifacts_dir, "train_data.csv")
            val_path = os.path.join(self.config.artifacts_dir, "val_data.csv")
            test_path = os.path.join(self.config.artifacts_dir, "test_data.csv")
            
            train_data.to_csv(train_path, index=False)
            val_data.to_csv(val_path, index=False)
            test_data.to_csv(test_path, index=False)
            
            self.artifacts["train_data_path"] = train_path
            self.artifacts["val_data_path"] = val_path
            self.artifacts["test_data_path"] = test_path
        
        return train_data, val_data, test_data
    
    async def _extract_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from games data.
        
        Args:
            games_df: DataFrame of games
            
        Returns:
            DataFrame with extracted features
        """
        # Extract team features
        team_features = await self.team_feature_extractor.extract_features(games_df)
        
        # Extract odds features
        odds_features = await self.odds_feature_extractor.extract_features(games_df)
        
        # Combine features
        features_df = pd.concat([games_df, team_features, odds_features], axis=1)
        
        # Define target (home team wins = 'home', away team wins = 'away')
        features_df["target"] = features_df.apply(
            lambda row: "home" if row["home_team_score"] > row["away_team_score"] else "away",
            axis=1
        )
        
        # Drop unnecessary columns if feature selection is enabled
        if self.config.feature_selection:
            # Basic feature selection - remove redundant or low-value features
            cols_to_drop = [
                "home_team_name", "away_team_name", "home_team_score", "away_team_score",
                "game_status", "stadium", "attendance", "created_at", "updated_at"
            ]
            
            # Only drop columns that exist
            cols_to_drop = [col for col in cols_to_drop if col in features_df.columns]
            
            features_df = features_df.drop(columns=cols_to_drop)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Log feature stats
        logger.info(f"Extracted {features_df.shape[1]} features for {len(features_df)} games")
        
        return features_df
    
    async def train_model(self) -> Dict[str, Any]:
        """
        Train the model using the prepared data.
        
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting model training for {self.config.model_name} ({self.config.model_version})")
        
        # Prepare training data
        train_data, val_data, test_data = await self.prepare_training_data()
        
        # Perform hyperparameter tuning if configured
        if self.config.hyperparameter_tuning:
            await self._hyperparameter_tuning(train_data)
        
        # Train the model
        training_results = self.model.train(
            train_data=train_data,
            val_data=val_data,
            params=self.config.model_params,
            save_model=False  # We'll handle saving through the registry
        )
        
        self.training_metrics = training_results["model_metrics"]
        self.feature_importance = training_results["feature_importance"]
        
        # Evaluate the model
        evaluation_metrics = self.evaluator.evaluate(test_data)
        
        # Update metrics with evaluation results
        self.training_metrics.update({
            "test_metrics": evaluation_metrics,
            "test_samples": len(test_data)
        })
        
        # Generate detailed evaluation report
        if self.config.save_artifacts:
            report_path = self.evaluator.generate_report(
                test_data, 
                report_name=f"{self.config.model_name}_{self.config.model_version}_evaluation.json"
            )
            self.artifacts["evaluation_report_path"] = report_path
        
        # Calibrate probabilities if configured
        if self.config.calibrate_probabilities:
            await self._calibrate_model(val_data)
        
        # Register the model if configured
        if self.config.register_model:
            await self._register_model()
        
        # Save training artifacts
        if self.config.save_artifacts:
            self._save_training_artifacts()
        
        logger.info(f"Model training completed: accuracy={evaluation_metrics['accuracy']:.4f}")
        
        return {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "metrics": self.training_metrics,
            "feature_importance": self.feature_importance,
            "artifacts": self.artifacts
        }
    
    async def _hyperparameter_tuning(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning.
        
        Args:
            train_data: Training data
            
        Returns:
            Dictionary with tuning results
        """
        logger.info("Starting hyperparameter tuning")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        # Perform tuning
        tuning_results = self.model.hyperparameter_tuning(
            train_data=train_data,
            param_grid=param_grid,
            cv_folds=5,
            scoring='accuracy'
        )
        
        # Update model parameters with best parameters
        self.config.model_params.update(tuning_results["best_params"])
        
        # Save tuning results if configured
        if self.config.save_artifacts:
            tuning_path = os.path.join(self.config.artifacts_dir, "hyperparameter_tuning.json")
            
            with open(tuning_path, "w") as f:
                # Convert numpy values to Python native types for JSON serialization
                tuning_results_serializable = {
                    "best_params": tuning_results["best_params"],
                    "best_score": float(tuning_results["best_score"])
                }
                json.dump(tuning_results_serializable, f, indent=2)
            
            self.artifacts["hyperparameter_tuning_path"] = tuning_path
        
        logger.info(f"Hyperparameter tuning completed: best score={tuning_results['best_score']:.4f}")
        
        return tuning_results
    
    async def _calibrate_model(self, val_data: pd.DataFrame) -> None:
        """
        Calibrate model probabilities.
        
        Args:
            val_data: Validation data
        """
        logger.info("Calibrating model probabilities")
        
        # Create calibrator
        calibrator = ProbabilityCalibrator(
            method="isotonic",
            model_name=self.config.model_name,
            model_version=self.config.model_version
        )
        
        # Calibrate using validation data
        calibrator = ProbabilityCalibrator.calibrate_predictions(
            self.model, val_data
        )
        
        # Save calibrator
        calibrator_path = calibrator.save()
        
        if self.config.save_artifacts:
            self.artifacts["calibrator_path"] = calibrator_path
        
        logger.info(f"Model calibration completed and saved to {calibrator_path}")
    
    async def _register_model(self) -> str:
        """
        Register the model in the model registry.
        
        Returns:
            Model ID
        """
        logger.info(f"Registering model {self.config.model_name} ({self.config.model_version})")
        
        # Save model locally first
        model_path = self.model.save()
        
        # Register model in registry
        model_id = await self.registry.register_model(
            model=self.model,
            model_path=model_path,
            description=self.config.description,
            tags=self.config.tags
        )
        
        logger.info(f"Model registered with ID: {model_id}")
        
        return model_id
    
    def _save_training_artifacts(self) -> Dict[str, str]:
        """
        Save training artifacts.
        
        Returns:
            Dictionary of artifact paths
        """
        if not self.config.save_artifacts:
            return {}
            
        # Save metrics
        metrics_path = os.path.join(self.config.artifacts_dir, "training_metrics.json")
        
        with open(metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)
        
        self.artifacts["metrics_path"] = metrics_path
        
        # Save feature importance
        importance_path = os.path.join(self.config.artifacts_dir, "feature_importance.json")
        
        with open(importance_path, "w") as f:
            json.dump(self.feature_importance, f, indent=2)
        
        self.artifacts["feature_importance_path"] = importance_path
        
        # Save configuration
        config_path = self.config.save()
        self.artifacts["config_path"] = config_path
        
        logger.info(f"Training artifacts saved to {self.config.artifacts_dir}")
        
        return self.artifacts
    
    @classmethod
    @run_async
    async def train_new_model(
        cls,
        model_name: str = "basketball_prediction_model",
        description: str = "",
        data_start_date: Optional[str] = None,
        data_end_date: Optional[str] = None,
        hyperparameter_tuning: bool = False,
        **model_params
    ) -> Dict[str, Any]:
        """
        Train a new model with the specified configuration.
        
        Args:
            model_name: Name of the model
            description: Model description
            data_start_date: Start date for training data
            data_end_date: End date for training data
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            **model_params: Additional model parameters
            
        Returns:
            Dictionary with training results
        """
        # Set date range if not provided
        if data_end_date is None:
            data_end_date = datetime.now().strftime("%Y-%m-%d")
        
        if data_start_date is None:
            # Default to 3 years of data
            start_date = datetime.now() - timedelta(days=365*3)
            data_start_date = start_date.strftime("%Y-%m-%d")
        
        # Create configuration
        config = ModelTrainingConfig(
            model_name=model_name,
            description=description,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            hyperparameter_tuning=hyperparameter_tuning,
            tags=["production", "automated_training"],
            **model_params
        )
        
        # Create and run pipeline
        pipeline = cls(config)
        results = await pipeline.train_model()
        
        return results
    
    @classmethod
    @run_async
    async def retrain_model(
        cls,
        model_name: str,
        model_version: Optional[str] = None,
        description: str = "Retrained model",
        data_window_days: int = 365,
        preserve_params: bool = True,
        **model_params
    ) -> Dict[str, Any]:
        """
        Retrain an existing model with new data.
        
        Args:
            model_name: Name of the model
            model_version: Version to retrain (None for active version)
            description: Model description
            data_window_days: Number of days of data to use
            preserve_params: Whether to preserve original model parameters
            **model_params: Additional model parameters to override
            
        Returns:
            Dictionary with training results
        """
        # Initialize registry
        registry = ModelRegistry()
        
        # Load existing model
        existing_model, existing_version = await registry.get_active_model(model_name)
        
        if existing_model is None:
            raise ValueError(f"No active model found for {model_name}")
        
        if model_version is not None:
            # Load specific version
            existing_model, existing_version = await registry.get_model(model_name, model_version)
            
            if existing_model is None:
                raise ValueError(f"Model version {model_version} not found")
        
        # Set date range
        data_end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = datetime.now() - timedelta(days=data_window_days)
        data_start_date = start_date.strftime("%Y-%m-%d")
        
        # Create configuration
        config = ModelTrainingConfig(
            model_name=model_name,
            description=description,
            data_start_date=data_start_date,
            data_end_date=data_end_date,
            tags=["production", "retraining"]
        )
        
        # Preserve parameters if requested
        if preserve_params:
            config.model_params.update(existing_model.model_params)
        
        # Override with provided parameters
        config.model_params.update(model_params)
        
        # Create and run pipeline
        pipeline = cls(config)
        results = await pipeline.train_model()
        
        return results


# Schedule functions for automated training
async def schedule_weekly_retraining():
    """
    Schedule weekly model retraining.
    """
    while True:
        try:
            logger.info("Starting scheduled weekly model retraining")
            
            # Retrain model
            await ModelTrainingPipeline.retrain_model(
                model_name="basketball_prediction_model",
                description="Weekly scheduled retraining",
                data_window_days=365,
                preserve_params=True
            )
            
            logger.info("Weekly model retraining completed")
            
            # Sleep for 7 days
            await asyncio.sleep(7 * 24 * 60 * 60)
            
        except Exception as e:
            logger.error(f"Error in weekly retraining: {str(e)}")
            # Sleep for 1 day before retrying
            await asyncio.sleep(24 * 60 * 60)


async def schedule_model_evaluation():
    """
    Schedule daily model evaluation.
    """
    while True:
        try:
            logger.info("Starting scheduled daily model evaluation")
            
            # Initialize registry and evaluator
            registry = ModelRegistry()
            model, version = await registry.get_active_model("basketball_prediction_model")
            
            if model is None:
                logger.warning("No active model found for evaluation")
                continue
            
            evaluator = ModelEvaluator(model=model)
            
            # Get recent games for evaluation
            games_repo = GamesRepository()
            recent_games = await games_repo.get_games(
                start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                include_teams=True,
                include_odds=True,
                completed_only=True
            )
            
            if not recent_games:
                logger.warning("No recent games found for evaluation")
                continue
            
            # Convert to DataFrame
            games_df = pd.DataFrame([game.dict() for game in recent_games])
            
            # Extract features
            pipeline = ModelTrainingPipeline()
            features_df = await pipeline._extract_features(games_df)
            
            # Evaluate model
            metrics = evaluator.evaluate(features_df)
            
            # Log results
            logger.info(f"Daily evaluation: model={version}, accuracy={metrics['accuracy']:.4f}")
            
            # Save results to database
            db = await get_database()
            await db["model_evaluation_logs"].insert_one({
                "model_name": "basketball_prediction_model",
                "model_version": version,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "sample_count": len(features_df)
            })
            
            # Sleep until next day
            tomorrow = datetime.now() + timedelta(days=1)
            tomorrow = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 2, 0, 0)  # 2 AM
            sleep_seconds = (tomorrow - datetime.now()).total_seconds()
            
            await asyncio.sleep(sleep_seconds)
            
        except Exception as e:
            logger.error(f"Error in daily evaluation: {str(e)}")
            # Sleep for 6 hours before retrying
            await asyncio.sleep(6 * 60 * 60)