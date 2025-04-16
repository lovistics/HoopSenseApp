"""
Player model for handling basketball players.
"""
from typing import Optional

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class PlayerModel(MongoBaseModel):
    """Player model representing a basketball player."""
    
    external_id: int
    name: str
    number: Optional[str] = None
    position: Optional[str] = None
    country: Optional[str] = None
    country_id: Optional[str] = None
    age: Optional[int] = None
    height: Optional[float] = None  # in cm
    weight: Optional[float] = None  # in kg
    team_id: str
    
    @validator('age')
    def validate_age(cls, v):
        """Validate player age is realistic."""
        if v is not None and (v < 15 or v > 50):
            raise ValueError("Player age must be between 15 and 50")
        return v
    
    @validator('height')
    def validate_height(cls, v):
        """Validate player height is realistic."""
        if v is not None and (v < 150 or v > 250):
            raise ValueError("Player height must be between 150 and 250 cm")
        return v
    
    @validator('weight')
    def validate_weight(cls, v):
        """Validate player weight is realistic."""
        if v is not None and (v < 50 or v > 180):
            raise ValueError("Player weight must be between 50 and 180 kg")
        return v


class PlayerInDB(PlayerModel):
    """Player model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "players"


def get_player_dict(player: PlayerModel) -> dict:
    """
    Convert player model to a dictionary for MongoDB storage.
    
    Args:
        player: Player model
        
    Returns:
        Dictionary representation for database
    """
    return player.dict_for_db()