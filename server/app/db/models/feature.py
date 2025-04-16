"""
Feature model for handling ML features.
"""
from typing import Optional, Dict, Any

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class FeatureModel(MongoBaseModel):
    """Feature model representing ML features for a game."""
    
    game_id: str
    external_game_id: int
    home_team_id: str
    away_team_id: str
    feature_set: Dict[str, Any]
    
    @validator('feature_set')
    def validate_feature_set(cls, v):
        """Validate feature set is not empty."""
        if not v:
            raise ValueError("Feature set cannot be empty")
        return v


class FeatureInDB(FeatureModel):
    """Feature model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "features"


def get_feature_dict(feature: FeatureModel) -> dict:
    """
    Convert feature model to a dictionary for MongoDB storage.
    
    Args:
        feature: Feature model
        
    Returns:
        Dictionary representation for database
    """
    return feature.dict_for_db()