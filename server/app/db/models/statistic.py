"""
Statistics models for handling basketball statistics.
"""
from typing import Optional, Dict, Any

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class GameStatsBreakdown(MongoBaseModel):
    """Breakdown of game stats by home/away/all."""
    
    home: Optional[int] = None
    away: Optional[int] = None
    all: Optional[int] = None


class WinLossBreakdown(MongoBaseModel):
    """Breakdown of wins/losses with percentage."""
    
    total: int
    percentage: float
    
    @validator('percentage')
    def validate_percentage(cls, v):
        """Validate percentage is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        return v


class GamesBreakdown(MongoBaseModel):
    """Breakdown of games played, wins, and losses."""
    
    played: GameStatsBreakdown
    wins: Dict[str, WinLossBreakdown]
    loses: Dict[str, WinLossBreakdown]


class PointsBreakdownDetail(MongoBaseModel):
    """Detailed breakdown of points by home/away/all."""
    
    home: float
    away: float
    all: float


class PointsBreakdown(MongoBaseModel):
    """Breakdown of points for and against."""
    
    for_points: Dict[str, PointsBreakdownDetail]
    against: Dict[str, PointsBreakdownDetail]


class TeamStatisticsModel(MongoBaseModel):
    """Team statistics model."""
    
    team_id: str
    external_team_id: int
    league_id: str
    season_id: str
    games: GamesBreakdown
    points: PointsBreakdown
    form: str
    win_percentage: float
    offensive_rating: float
    defensive_rating: float
    net_rating: float
    pace: float
    true_shooting_percentage: float
    
    @validator('win_percentage', 'true_shooting_percentage')
    def validate_percentage(cls, v):
        """Validate percentage is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        return v


class FieldGoalStats(MongoBaseModel):
    """Field goal statistics."""
    
    made: int
    attempts: int
    percentage: Optional[float] = None
    
    @validator('percentage')
    def validate_percentage(cls, v, values):
        """Validate percentage calculation."""
        if v is not None:
            if not 0 <= v <= 1:
                raise ValueError("Percentage must be between 0 and 1")
            
            # Validate consistency with made/attempts
            attempts = values.get('attempts', 0)
            made = values.get('made', 0)
            
            if attempts > 0:
                expected_pct = made / attempts
                # Allow small rounding error
                if abs(v - expected_pct) > 0.001:
                    return expected_pct
        
        return v


class PlayerStatisticsModel(MongoBaseModel):
    """Player statistics model."""
    
    player_id: str
    external_player_id: int
    team_id: str
    game_id: str
    external_game_id: int
    type: str  # "starters" or "bench"
    minutes: str
    field_goals: FieldGoalStats
    three_points: FieldGoalStats
    free_throws: FieldGoalStats
    rebounds: int
    assists: int
    points: int
    
    @validator('type')
    def validate_type(cls, v):
        """Validate player type."""
        valid_types = ["starters", "bench"]
        if v not in valid_types:
            raise ValueError(f"Type must be one of {valid_types}")
        return v
    
    @validator('points')
    def validate_points_calculation(cls, values):
        """Validate points consistency with field goals."""
        field_goals = values.get('field_goals')
        three_points = values.get('three_points')
        free_throws = values.get('free_throws')
        points = values.get('points')
        
        if all(v is not None for v in [field_goals, three_points, free_throws, points]):
            # Calculate expected points
            expected_points = (
                (field_goals.made - three_points.made) * 2 +  # 2-point field goals
                three_points.made * 3 +  # 3-point field goals
                free_throws.made  # Free throws
            )
            
            # Allow small discrepancy
            if abs(points - expected_points) > 3:
                return expected_points
        
        return points


# Collection names in MongoDB
TEAM_STATS_COLLECTION = "team_statistics"
PLAYER_STATS_COLLECTION = "player_statistics"


def get_team_stats_dict(stats: TeamStatisticsModel) -> dict:
    """
    Convert team statistics model to a dictionary for MongoDB storage.
    
    Args:
        stats: Team statistics model
        
    Returns:
        Dictionary representation for database
    """
    return stats.dict_for_db()


def get_player_stats_dict(stats: PlayerStatisticsModel) -> dict:
    """
    Convert player statistics model to a dictionary for MongoDB storage.
    
    Args:
        stats: Player statistics model
        
    Returns:
        Dictionary representation for database
    """
    return stats.dict_for_db()