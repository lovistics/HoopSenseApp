"""
Odds model for handling basketball betting odds.
"""
from typing import Optional, List, Dict, Any

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class OddValue(MongoBaseModel):
    """Individual odd value."""
    
    value: str
    odd: float
    
    @validator('odd')
    def validate_odd(cls, v):
        """Validate odd is positive."""
        if v <= 0:
            raise ValueError("Odd must be positive")
        return v


class Bet(MongoBaseModel):
    """Bet information."""
    
    id: int
    name: str
    values: List[OddValue]


class Bookmaker(MongoBaseModel):
    """Bookmaker information."""
    
    id: int
    name: str
    bets: List[Bet]


class OddsConsensus(MongoBaseModel):
    """Consensus odds across bookmakers."""
    
    home_win: float
    away_win: float
    implied_home_probability: float
    implied_away_probability: float
    
    @validator('home_win', 'away_win')
    def validate_odd(cls, v):
        """Validate odd is positive."""
        if v <= 0:
            raise ValueError("Odd must be positive")
        return v
    
    @validator('implied_home_probability', 'implied_away_probability')
    def validate_probability(cls, v):
        """Validate probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v


class OddsModel(MongoBaseModel):
    """Odds model representing basketball betting odds."""
    
    game_id: str
    external_game_id: int
    league_id: str
    season_id: str
    bookmakers: List[Bookmaker]
    consensus: OddsConsensus


class OddsInDB(OddsModel):
    """Odds model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "odds"


def get_odds_dict(odds: OddsModel) -> dict:
    """
    Convert odds model to a dictionary for MongoDB storage.
    
    Args:
        odds: Odds model
        
    Returns:
        Dictionary representation for database
    """
    return odds.dict_for_db()