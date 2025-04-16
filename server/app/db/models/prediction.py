"""
Prediction model for handling ML model predictions.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class ModelPerformance(MongoBaseModel):
    """Performance metrics for the model."""
    
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    
    @validator('accuracy', 'precision', 'recall', 'f1_score', 'auc')
    def validate_metric(cls, v):
        """Validate metric is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Metric must be between 0 and 1")
        return v


class PredictionModelSchema(MongoBaseModel):
    """Model representing an ML prediction model."""
    
    version: str
    model_type: str
    features: List[str]
    hyperparameters: Dict[str, Any]
    performance: ModelPerformance
    training_date: datetime
    is_active: bool = True


class GamePredictionItem(MongoBaseModel):
    """Individual game prediction for a user."""
    
    game_id: str
    predicted_winner: str
    actual_winner: Optional[str] = None
    is_correct: Optional[bool] = None
    confidence: int
    
    @validator('predicted_winner', 'actual_winner')
    def validate_winner(cls, v):
        """Validate winner is home, away, or None."""
        if v is not None and v not in ["home", "away"]:
            raise ValueError("Winner must be 'home', 'away', or None")
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Confidence must be between 0 and 100")
        return v


class PredictionHistory(MongoBaseModel):
    """Prediction history for a user."""
    
    user_id: str
    date: str
    predictions: List[GamePredictionItem]
    summary: Dict[str, Any]
    
    @validator('date')
    def validate_date_format(cls, v):
        """Validate date is in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class PredictionModelInDB(PredictionModelSchema):
    """Prediction model as stored in the database."""
    pass


class PredictionHistoryInDB(PredictionHistory):
    """Prediction history as stored in the database."""
    pass


# Collection names in MongoDB
MODEL_COLLECTION = "prediction_models"
HISTORY_COLLECTION = "prediction_history"


def get_model_dict(model: PredictionModelSchema) -> dict:
    """
    Convert prediction model to a dictionary for MongoDB storage.
    
    Args:
        model: Prediction model
        
    Returns:
        Dictionary representation for database
    """
    return model.dict_for_db()


def get_history_dict(history: PredictionHistory) -> dict:
    """
    Convert prediction history to a dictionary for MongoDB storage.
    
    Args:
        history: Prediction history
        
    Returns:
        Dictionary representation for database
    """
    return history.dict_for_db()