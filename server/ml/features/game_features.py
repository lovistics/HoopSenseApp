"""
Game-related feature extraction for basketball predictions.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from app.core.logger import logger
from app.db.models.game import GameInDB
from app.db.models.team import TeamInDB


class GameFeatureProcessor:
    """
    Extracts features related to the game context, such as:
    - Home court advantage
    - Rest days
    - Back-to-back games
    - Game time/day of week
    - Season information
    """
    
    def __init__(self):
        """Initialize the game feature processor."""
        self.feature_prefix = 'game_'
    
    async def extract_features(
        self,
        game: GameInDB,
        home_team: TeamInDB,
        away_team: TeamInDB
    ) -> Dict[str, Any]:
        """
        Extract game-related features.
        
        Args:
            game: Game data
            home_team: Home team data
            away_team: Away team data
            
        Returns:
            Dictionary of feature name to value
        """
        features = {}
        
        # Basic game information
        features[f'{self.feature_prefix}is_home_court'] = 1  # Always 1 for home team perspective
        
        # Extract time-related features
        features.update(self._extract_time_features(game))
        
        # Extract rest and schedule features
        # These would ideally come from previous games data, but we'll use placeholders
        # for now and implement this in the historical feature processor
        features[f'{self.feature_prefix}home_days_rest'] = 0
        features[f'{self.feature_prefix}away_days_rest'] = 0
        features[f'{self.feature_prefix}home_back_to_back'] = 0
        features[f'{self.feature_prefix}away_back_to_back'] = 0
        
        # Extract location-based features
        features.update(self._extract_location_features(game, home_team, away_team))
        
        # Extract season context features
        features.update(self._extract_season_features(game))
        
        return features
    
    def _extract_time_features(self, game: GameInDB) -> Dict[str, Any]:
        """
        Extract time-related features from the game.
        
        Args:
            game: Game data
            
        Returns:
            Dictionary of time-related features
        """
        features = {}
        game_date = game.date
        
        # Day of week (0 = Monday, 6 = Sunday)
        day_of_week = game_date.weekday()
        features[f'{self.feature_prefix}day_of_week'] = day_of_week
        
        # Weekend game (Friday, Saturday, Sunday)
        features[f'{self.feature_prefix}is_weekend'] = 1 if day_of_week >= 4 else 0
        
        # Game hour (in 24-hour format)
        hour = game_date.hour
        features[f'{self.feature_prefix}hour'] = hour
        
        # Game time category
        if hour < 12:
            time_category = 'morning'
        elif hour < 18:
            time_category = 'afternoon'
        else:
            time_category = 'evening'
            
        features[f'{self.feature_prefix}time_category'] = time_category
        
        # Month of season
        features[f'{self.feature_prefix}month'] = game_date.month
        
        # Part of season (early, mid, late)
        # This would ideally use season start/end dates
        # Here we're using a simplified approach based on month
        if game_date.month in [10, 11, 12]:
            season_part = 'early'
        elif game_date.month in [1, 2]:
            season_part = 'mid'
        else:
            season_part = 'late'
            
        features[f'{self.feature_prefix}season_part'] = season_part
        
        return features
    
    def _extract_location_features(
        self,
        game: GameInDB,
        home_team: TeamInDB,
        away_team: TeamInDB
    ) -> Dict[str, Any]:
        """
        Extract location-based features from the game.
        
        Args:
            game: Game data
            home_team: Home team data
            away_team: Away team data
            
        Returns:
            Dictionary of location-related features
        """
        features = {}
        
        # Home court strength 
        # Could be calculated from historical win percentage at home
        # For now, we'll use a placeholder
        features[f'{self.feature_prefix}home_court_strength'] = 0.6
        
        # Same conference game
        same_conference = (
            hasattr(home_team, 'conference') and 
            hasattr(away_team, 'conference') and
            home_team.conference == away_team.conference
        )
        features[f'{self.feature_prefix}same_conference'] = 1 if same_conference else 0
        
        # Same division game
        same_division = (
            hasattr(home_team, 'division') and 
            hasattr(away_team, 'division') and
            home_team.division == away_team.division
        )
        features[f'{self.feature_prefix}same_division'] = 1 if same_division else 0
        
        # Venue information if available
        if hasattr(game, 'venue') and game.venue:
            features[f'{self.feature_prefix}has_venue_info'] = 1
        else:
            features[f'{self.feature_prefix}has_venue_info'] = 0
        
        return features
    
    def _extract_season_features(self, game: GameInDB) -> Dict[str, Any]:
        """
        Extract season context features from the game.
        
        Args:
            game: Game data
            
        Returns:
            Dictionary of season-related features
        """
        features = {}
        
        # Game stage/type
        if hasattr(game, 'stage') and game.stage:
            features[f'{self.feature_prefix}stage'] = game.stage
            features[f'{self.feature_prefix}is_playoff'] = 1 if "playoff" in game.stage.lower() else 0
            features[f'{self.feature_prefix}is_regular_season'] = 1 if "regular" in game.stage.lower() else 0
        else:
            # Default to regular season if not specified
            features[f'{self.feature_prefix}stage'] = "Regular Season"
            features[f'{self.feature_prefix}is_playoff'] = 0
            features[f'{self.feature_prefix}is_regular_season'] = 1
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get a list of feature names generated by this processor.
        
        Returns:
            List of feature names
        """
        return [
            # Time-related features
            f'{self.feature_prefix}day_of_week',
            f'{self.feature_prefix}is_weekend',
            f'{self.feature_prefix}hour',
            f'{self.feature_prefix}time_category',
            f'{self.feature_prefix}month',
            f'{self.feature_prefix}season_part',
            
            # Location features
            f'{self.feature_prefix}is_home_court',
            f'{self.feature_prefix}home_court_strength',
            f'{self.feature_prefix}same_conference',
            f'{self.feature_prefix}same_division',
            f'{self.feature_prefix}has_venue_info',
            
            # Rest and schedule features
            f'{self.feature_prefix}home_days_rest',
            f'{self.feature_prefix}away_days_rest',
            f'{self.feature_prefix}home_back_to_back',
            f'{self.feature_prefix}away_back_to_back',
            
            # Season features
            f'{self.feature_prefix}stage',
            f'{self.feature_prefix}is_playoff',
            f'{self.feature_prefix}is_regular_season'
        ]