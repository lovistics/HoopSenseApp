"""
Model registry for managing ML models.
"""
import os
import json
import shutil
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

from app.core.logger import logger
from app.core.config import settings
from app.db.models.prediction import PredictionModelSchema, ModelPerformance
from app.ml.models.xgboost_model import BasketballXGBoostModel
from app.data.repositories.prediction_repository import PredictionModelRepository


class ModelRegistry:
    """
    Registry for managing ML models, including versioning, 
    performance tracking, and model selection.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Directory for model storage
        """
        # Set models directory
        if models_dir is None:
            self.models_dir = os.path.join(settings.BASE_DIR, "ml", "models", "saved_models")
        else:
            self.models_dir = models_dir
        
        # Create directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize repository
        self.model_repository = PredictionModelRepository()
        
        # Current active model instance
        self.active_model = None
    
    async def register_model(
        self,
        model: BasketballXGBoostModel,
        model_path: str,
        description: str = "",
        tags: List[str] = []
    ) -> str:
        """
        Register a trained model in the registry.
        
        Args:
            model: Trained model instance
            model_path: Path to saved model files
            description: Model description
            tags: Model tags for categorization
            
        Returns:
            Registered model ID
            
        Raises:
            ValueError: If model registration fails or inputs are invalid
        """
        # Extract model metadata
        model_metrics = model.model_metrics
        
        # Create model performance object
        performance = ModelPerformance(
            accuracy=model_metrics.get("accuracy", 0.0),
            precision=model_metrics.get("precision", 0.0),
            recall=model_metrics.get("recall", 0.0),
            f1_score=model_metrics.get("f1_score", 0.0),
            auc=model_metrics.get("auc", 0.0)
        )
        
        # Create model schema
        model_schema = PredictionModelSchema(
            version=model.model_version,
            model_type=f"xgboost_{model.model_name}",
            features=model.feature_names,
            hyperparameters=model.model_params,
            performance=performance,
            training_date=datetime.now(),
            is_active=False  # Not active by default
        )
        
        # Save model to registry database
        model_id = await self.model_repository.create(model_schema)
        
        if not model_id:
            logger.error("Failed to register model in database")
            raise ValueError("Failed to register model in database")
        
        # Register model in filesystem
        registry_path = os.path.join(self.models_dir, model_id)
        
        # Create registry directory
        os.makedirs(registry_path, exist_ok=True)
        
        # Copy model files to registry
        if os.path.isdir(model_path):
            # Copy all files from model_path to registry_path
            for file_name in os.listdir(model_path):
                source_file = os.path.join(model_path, file_name)
                target_file = os.path.join(registry_path, file_name)
                
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, target_file)
        else:
            # model_path is the model file itself
            shutil.copy2(model_path, os.path.join(registry_path, "model.xgb"))
        
        # Create or update registry info file
        registry_info = {
            "model_id": model_id,
            "model_name": model.model_name,
            "model_version": model.model_version,
            "description": description,
            "tags": tags,
            "registered_at": datetime.now().isoformat(),
            "metrics": model_metrics,
            "feature_names": model.feature_names,
            "model_params": model.model_params
        }
        
        with open(os.path.join(registry_path, "registry_info.json"), "w") as f:
            json.dump(registry_info, f, indent=2)
        
        logger.info(f"Model registered with ID {model_id}")
        return model_id
    
    async def activate_model(self, model_id: str) -> bool:
        """
        Activate a registered model.
        
        Args:
            model_id: Model ID to activate
            
        Returns:
            True if activation was successful
            
        Raises:
            ValueError: If model is not found or activation fails
        """
        # Check if model exists in database
        model_record = await self.model_repository.find_by_id(model_id)
        
        if not model_record:
            logger.error(f"Model with ID {model_id} not found in database")
            raise ValueError(f"Model with ID {model_id} not found in database")
        
        # Check if model exists in registry filesystem
        registry_path = os.path.join(self.models_dir, model_id)
        if not os.path.exists(registry_path):
            logger.error(f"Model files for ID {model_id} not found in registry")
            raise ValueError(f"Model files for ID {model_id} not found in registry")
        
        # Activate in database
        _, activated = await self.model_repository.activate_model(model_id)
        
        if not activated:
            logger.error(f"Failed to activate model {model_id} in database")
            raise ValueError(f"Failed to activate model {model_id} in database")
        
        # Load the model in memory if needed
        if self.active_model is None or self.active_model.model_version != model_record.version:
            self.active_model = BasketballXGBoostModel(
                model_name=f"basketball_prediction_{model_record.version}",
                model_version=model_record.version
            )
            
            # Load model from registry
            loaded = self.active_model.load(registry_path)
            
            if not loaded:
                logger.error(f"Failed to load model {model_id} from registry")
                raise ValueError(f"Failed to load model {model_id} from registry")
        
        logger.info(f"Model {model_id} activated successfully")
        return True
    
    async def get_active_model(self) -> Optional[BasketballXGBoostModel]:
        """
        Get the currently active model.
        
        Returns:
            Active model instance
            
        Raises:
            ValueError: If active model exists in database but files are not found or model fails to load
        """
        # If active model already loaded, return it
        if self.active_model is not None:
            return self.active_model
        
        # Get active model from database
        active_record = await self.model_repository.find_active_model()
        
        if not active_record:
            logger.warning("No active model found in database")
            return None
        
        # Check if model exists in registry filesystem
        registry_path = os.path.join(self.models_dir, str(active_record.id))
        if not os.path.exists(registry_path):
            logger.error(f"Model files for active model {active_record.id} not found in registry")
            raise ValueError(f"Model files for active model {active_record.id} not found in registry")
        
        # Load the model
        self.active_model = BasketballXGBoostModel(
            model_name=f"basketball_prediction_{active_record.version}",
            model_version=active_record.version
        )
        
        # Load model from registry
        loaded = self.active_model.load(registry_path)
        
        if not loaded:
            logger.error(f"Failed to load active model {active_record.id} from registry")
            raise ValueError(f"Failed to load active model {active_record.id} from registry")
        
        logger.info(f"Active model {active_record.version} loaded successfully")
        return self.active_model
    
    async def list_models(
        self,
        active_only: bool = False,
        min_accuracy: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List registered models with filtering options.
        
        Args:
            active_only: Only return active models
            min_accuracy: Minimum accuracy filter
            tags: Filter by tags
            limit: Maximum number of models to return
            
        Returns:
            List of model information dictionaries
        """
        # Build filter
        filter_dict = {}
        
        if active_only:
            filter_dict["is_active"] = True
        
        if min_accuracy is not None:
            filter_dict["performance.accuracy"] = {"$gte": min_accuracy}
        
        # Query database
        models = await self.model_repository.find(
            filter=filter_dict,
            sort=[("training_date", -1)],
            limit=limit
        )
        
        # Format result
        result = []
        for model in models:
            model_info = {
                "id": str(model.id),
                "version": model.version,
                "model_type": model.model_type,
                "is_active": model.is_active,
                "training_date": model.training_date.isoformat() if hasattr(model.training_date, 'isoformat') else str(model.training_date),
                "accuracy": model.performance.accuracy,
                "auc": model.performance.auc,
                "features_count": len(model.features) if model.features else 0
            }
            
            # Check for additional info in registry
            registry_path = os.path.join(self.models_dir, str(model.id))
            info_file = os.path.join(registry_path, "registry_info.json")
            
            if os.path.exists(info_file):
                try:
                    with open(info_file, "r") as f:
                        registry_info = json.load(f)
                    
                    # Add description and tags if available
                    if "description" in registry_info:
                        model_info["description"] = registry_info["description"]
                    
                    if "tags" in registry_info:
                        model_info["tags"] = registry_info["tags"]
                        
                        # Filter by tags if specified
                        if tags is not None:
                            if not any(tag in model_info["tags"] for tag in tags):
                                continue
                except Exception as e:
                    logger.warning(f"Error reading registry info for model {model.id}: {str(e)}")
            
            result.append(model_info)
        
        return result
    
    async def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a registered model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary with model details
        """
        # Get model from database
        model_record = await self.model_repository.find_by_id(model_id)
        
        if not model_record:
            logger.error(f"Model with ID {model_id} not found in database")
            return {}
        
        # Check registry for additional information
        registry_path = os.path.join(self.models_dir, model_id)
        info_file = os.path.join(registry_path, "registry_info.json")
        
        registry_info = {}
        if os.path.exists(info_file):
            try:
                with open(info_file, "r") as f:
                    registry_info = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading registry info for model {model_id}: {str(e)}")
        
        # Combine information
        details = {
            "id": str(model_record.id),
            "version": model_record.version,
            "model_type": model_record.model_type,
            "is_active": model_record.is_active,
            "training_date": model_record.training_date.isoformat() if hasattr(model_record.training_date, 'isoformat') else str(model_record.training_date),
            "performance": model_record.performance.dict(),
            "hyperparameters": model_record.hyperparameters,
            "features": model_record.features[:50] + ["..."] if len(model_record.features) > 50 else model_record.features,
            "features_count": len(model_record.features) if model_record.features else 0
        }
        
        # Add registry info
        for key, value in registry_info.items():
            if key not in details and key != "feature_names":
                details[key] = value
        
        # Add feature importance if available
        if "metrics" in registry_info and "feature_importance" in registry_info:
            details["feature_importance"] = registry_info.get("feature_importance", {})
        
        return details
    
    async def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model ID to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            ValueError: If model is not found, is currently active, or deletion fails
        """
        # Check if model is currently active
        model_record = await self.model_repository.find_by_id(model_id)
        
        if not model_record:
            logger.error(f"Model with ID {model_id} not found in database")
            raise ValueError(f"Model with ID {model_id} not found in database")
        
        if model_record.is_active:
            logger.error(f"Cannot delete active model {model_id}. Deactivate first.")
            raise ValueError(f"Cannot delete active model {model_id}. Deactivate first.")
        
        # Delete from database
        deleted = await self.model_repository.delete(model_id)
        
        if not deleted:
            logger.error(f"Failed to delete model {model_id} from database")
            raise ValueError(f"Failed to delete model {model_id} from database")
        
        # Delete from filesystem
        registry_path = os.path.join(self.models_dir, model_id)
        
        if os.path.exists(registry_path):
            try:
                shutil.rmtree(registry_path)
            except Exception as e:
                logger.error(f"Error deleting model files for {model_id}: {str(e)}")
                raise ValueError(f"Error deleting model files for {model_id}: {str(e)}")
        
        logger.info(f"Model {model_id} deleted successfully")
        return True
    
    async def load_model_by_id(self, model_id: str) -> BasketballXGBoostModel:
        """
        Load a specific model by ID.
        
        Args:
            model_id: Model ID to load
            
        Returns:
            Loaded model instance
            
        Raises:
            ValueError: If model is not found or fails to load
        """
        # Get model record
        model_record = await self.model_repository.find_by_id(model_id)
        
        if not model_record:
            logger.error(f"Model with ID {model_id} not found in database")
            raise ValueError(f"Model with ID {model_id} not found in database")
        
        # Check if model exists in registry filesystem
        registry_path = os.path.join(self.models_dir, model_id)
        
        if not os.path.exists(registry_path):
            logger.error(f"Model files for {model_id} not found in registry")
            raise ValueError(f"Model files for {model_id} not found in registry")
        
        # Load the model
        model = BasketballXGBoostModel(
            model_name=f"basketball_prediction_{model_record.version}",
            model_version=model_record.version
        )
        
        # Load model from registry
        loaded = model.load(registry_path)
        
        if not loaded:
            logger.error(f"Failed to load model {model_id} from registry")
            raise ValueError(f"Failed to load model {model_id} from registry")
        
        logger.info(f"Model {model_id} loaded successfully")
        return model
    
    async def compare_models(
        self,
        model_ids: List[str],
        test_data: pd.DataFrame,
        target_column: str = "target"
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same test data.
        
        Args:
            model_ids: List of model IDs to compare
            test_data: Test data for evaluation
            target_column: Target column name
            
        Returns:
            Dictionary with comparison results
        """
        if not model_ids:
            logger.error("No model IDs provided for comparison")
            return {}
        
        # Validate test data
        if target_column not in test_data.columns:
            logger.error(f"Target column '{target_column}' not in test data")
            return {}
        
        # Load and evaluate each model
        results = {}
        
        for model_id in model_ids:
            model = await self.load_model_by_id(model_id)
            
            if model is None:
                logger.warning(f"Could not load model {model_id} for comparison")
                continue
            
            # Set target column
            model.target_column = target_column
            
            # Evaluate model
            evaluation = model.evaluate(test_data)
            
            # Get model details
            model_details = await self.get_model_details(model_id)
            model_version = model_details.get("version", model_id)
            
            # Store results
            results[model_version] = {
                "model_id": model_id,
                "metrics": evaluation,
                "is_active": model_details.get("is_active", False),
                "training_date": model_details.get("training_date", "Unknown")
            }
        
        return {
            "comparison": results,
            "test_samples": len(test_data),
            "compared_at": datetime.now().isoformat()
        }
    
    def export_model(self, model_id: str, export_dir: str) -> bool:
        """
        Export a model to a different directory.
        
        Args:
            model_id: Model ID to export
            export_dir: Directory to export to
            
        Returns:
            True if export was successful, False otherwise
        """
        # Check if model exists in registry
        registry_path = os.path.join(self.models_dir, model_id)
        
        if not os.path.exists(registry_path):
            logger.error(f"Model files for {model_id} not found in registry")
            return False
        
        # Create export directory
        os.makedirs(export_dir, exist_ok=True)
        
        # Copy files
        try:
            for file_name in os.listdir(registry_path):
                source_file = os.path.join(registry_path, file_name)
                target_file = os.path.join(export_dir, file_name)
                
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, target_file)
            
            logger.info(f"Model {model_id} exported to {export_dir}")
            return True
        except Exception as e:
            logger.error(f"Error exporting model {model_id}: {str(e)}")
            return False