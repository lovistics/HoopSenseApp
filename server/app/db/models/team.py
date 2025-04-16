"""
Team model for handling basketball teams.
"""
from typing import Optional

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class TeamModel(MongoBaseModel):
    """Team model representing a basketball team."""
    
    external_id: int
    name: str
    abbreviation: str
    is_national: bool = False
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    country_id: str
    league_id: str
    conference: Optional[str] = None
    division: Optional[str] = None
    
    @validator('abbreviation')
    def validate_abbreviation(cls, v):
        """Validate abbreviation format."""
        if not v:
            raise ValueError("Abbreviation cannot be empty")
        return v.upper()
    
    @validator('primary_color', 'secondary_color')
    def validate_color(cls, v):
        """Validate color format."""
        if v is not None and not v.startswith('#'):
            return f"#{v}" if v else v
        return v


class TeamInDB(TeamModel):
    """Team model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "teams"


def get_team_dict(team: TeamModel) -> dict:
    """
    Convert team model to a dictionary for MongoDB storage.
    
    Args:
        team: Team model
        
    Returns:
        Dictionary representation for database
    """
    return team.dict_for_db()