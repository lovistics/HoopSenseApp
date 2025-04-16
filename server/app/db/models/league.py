"""
League model for handling basketball leagues.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class LeagueSeason(MongoBaseModel):
    """Season within a league."""
    
    season_id: str
    external_season: str
    start_date: datetime
    end_date: datetime
    coverage: Dict[str, Any]


class LeagueModel(MongoBaseModel):
    """League model representing a basketball league."""
    
    external_id: int
    name: str
    type: str
    logo_url: Optional[str] = None
    country_id: str
    seasons: List[LeagueSeason] = Field(default_factory=list)
    
    @validator('name')
    def validate_name(cls, v):
        """Validate league name is not empty."""
        if not v.strip():
            raise ValueError("League name cannot be empty")
        return v.strip()


class LeagueInDB(LeagueModel):
    """League model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "leagues"


def get_league_dict(league: LeagueModel) -> dict:
    """
    Convert league model to a dictionary for MongoDB storage.
    
    Args:
        league: League model
        
    Returns:
        Dictionary representation for database
    """
    return league.dict_for_db()