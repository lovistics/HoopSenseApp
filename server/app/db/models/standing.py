"""
Standing model for handling basketball standings.
"""
from typing import Optional, List, Dict, Any

from pydantic import Field, validator, root_validator

from app.db.models.base import MongoBaseModel, PyObjectId


class StandingTeam(MongoBaseModel):
    """Team information within standings."""
    
    id: str
    name: str
    logo: Optional[str] = None


class StandingGames(MongoBaseModel):
    """Games information for standings."""
    
    played: int
    wins: Dict[str, Any]
    losses: Dict[str, Any]
    
    @validator('played')
    def validate_played(cls, v):
        """Validate games played is not negative."""
        if v < 0:
            raise ValueError("Games played cannot be negative")
        return v


class StandingPoints(MongoBaseModel):
    """Points information for standings."""
    
    for_points: int
    against: int


class StandingEntry(MongoBaseModel):
    """Individual standing entry."""
    
    position: int
    team_id: str
    external_team_id: int
    team_name: str
    games: StandingGames
    win_percentage: float
    points: StandingPoints
    form: str
    description: Optional[str] = None
    
    @validator('win_percentage')
    def validate_win_percentage(cls, v):
        """Validate win percentage is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Win percentage must be between 0 and 1")
        return v
    
    @root_validator
    def validate_win_percentage_calculation(cls, values):
        """Validate win percentage matches games played/won."""
        games = values.get('games')
        win_pct = values.get('win_percentage')
        
        if games and win_pct is not None:
            # Check if games.played is zero to avoid division by zero
            if games.played == 0:
                if win_pct != 0:
                    raise ValueError("Win percentage should be 0 when no games played")
            elif games.wins.get('total') is not None:
                # Calculate expected win percentage
                expected_pct = games.wins.get('total') / games.played
                # Allow small rounding error
                if abs(win_pct - expected_pct) > 0.001:
                    values['win_percentage'] = expected_pct
        
        return values


class StandingModel(MongoBaseModel):
    """Standing model representing basketball standings."""
    
    league_id: str
    season_id: str
    external_league_id: int
    external_season: str
    stage: str
    group: Dict[str, Any]
    standings: List[StandingEntry]


class StandingInDB(StandingModel):
    """Standing model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "standings"


def get_standing_dict(standing: StandingModel) -> dict:
    """
    Convert standing model to a dictionary for MongoDB storage.
    
    Args:
        standing: Standing model
        
    Returns:
        Dictionary representation for database
    """
    return standing.dict_for_db()