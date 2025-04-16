"""
Game model for handling basketball games.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class GameScore(MongoBaseModel):
    """Score for a team in a game."""
    
    quarter_1: Optional[int] = None
    quarter_2: Optional[int] = None
    quarter_3: Optional[int] = None
    quarter_4: Optional[int] = None
    over_time: Optional[List[int]] = None
    total: Optional[int] = None
    
    @validator("*", pre=True, each_item=True)
    def convert_empty_to_none(cls, v):
        """Convert empty string to None for numeric fields."""
        if v == "":
            return None
        return v


class GameTeam(MongoBaseModel):
    """Team information within a game."""
    
    team_id: str
    external_id: int
    name: str
    scores: Optional[GameScore] = None


class GameStatus(MongoBaseModel):
    """Status of a game."""
    
    long: str
    short: str
    timer: Optional[str] = None


class AnalysisFactor(MongoBaseModel):
    """Factor influencing the prediction."""
    
    factor: str
    description: str
    impact: float
    
    @validator('impact')
    def validate_impact(cls, v):
        """Validate impact is within reasonable range."""
        if abs(v) > 100:
            raise ValueError("Impact cannot exceed 100 in absolute value")
        return v


class GamePrediction(MongoBaseModel):
    """Prediction for a game."""
    
    predicted_winner: str  # "home" or "away"
    home_win_probability: float
    confidence: int  # 0-100
    is_game_of_day: bool = False
    is_in_vip_betslip: bool = False
    analysis_factors: Optional[List[AnalysisFactor]] = None
    model_version: str
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator('predicted_winner')
    def validate_winner(cls, v):
        """Validate predicted winner is either home or away."""
        if v not in ["home", "away"]:
            raise ValueError("Winner must be either 'home' or 'away'")
        return v
    
    @validator('home_win_probability')
    def validate_probability(cls, v):
        """Validate probability is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Confidence must be between 0 and 100")
        return v


class GameModel(MongoBaseModel):
    """Game model representing a basketball game."""
    
    external_id: int
    league_id: str
    season_id: str
    date: datetime
    timestamp: int
    timezone: str
    stage: Optional[str] = None
    status: GameStatus
    home_team: GameTeam
    away_team: GameTeam
    venue: Optional[str] = None
    is_analyzed: bool = False
    prediction: Optional[GamePrediction] = None


class GameInDB(GameModel):
    """Game model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "games"


def get_game_dict(game: GameModel) -> dict:
    """
    Convert game model to a dictionary for MongoDB storage.
    
    Args:
        game: Game model
        
    Returns:
        Dictionary representation for database
    """
    return game.dict_for_db()