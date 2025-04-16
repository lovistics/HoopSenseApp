"""
Season model for handling basketball seasons.
"""
from datetime import datetime
from typing import Optional, List

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class SeasonModel(MongoBaseModel):
    """Season model representing a basketball season."""
    
    external_id: str
    name: str
    start_date: datetime
    end_date: datetime
    status: str  # active, completed, upcoming
    
    @validator('status')
    def validate_status(cls, v):
        """Validate that status is one of the allowed values."""
        allowed_values = ["active", "completed", "upcoming"]
        if v not in allowed_values:
            raise ValueError(f"Status must be one of {allowed_values}")
        return v
    
    @validator('end_date')
    def validate_end_date(cls, v, values):
        """Validate end date is after start date."""
        if 'start_date' in values and v < values['start_date']:
            raise ValueError("End date must be after start date")
        return v


class SeasonInDB(SeasonModel):
    """Season model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "seasons"


def get_season_dict(season: SeasonModel) -> dict:
    """
    Convert season model to a dictionary for MongoDB storage.
    
    Args:
        season: Season model
        
    Returns:
        Dictionary representation for database
    """
    return season.dict_for_db()