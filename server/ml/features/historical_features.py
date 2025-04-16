"""
Historical data feature extraction for basketball predictions.
"""
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from app.core.logger import logger
from app.db.models.game import GameInDB
from server.data.repositories.game_repository import GameRepository


class HistoricalFeatureProcessor:
    """
    Extracts features from historical game data, such as:
    - Team recent performance
    - Head-to-head matchups
    - Rest days and schedule effects
    - Streaks and momentum
    """
    
    def __init__(self):
        """Initialize the historical feature processor."""
        self.feature_prefix = 'hist_'
        self.game_repository = GameRepository()
    
    async def extract_features(
        self,
        game: GameInDB,
        lookback_games: int = 10,
        lookback_days: int = 60
    ) -> Dict[str, Any]:
        """
        Extract historical features.
        
        Args:
            game: Current game
            lookback_games: Number of previous games to consider
            lookback_days: Maximum number of days to look back
            
        Returns:
            Dictionary of feature name to value
        """
        features = {}
        
        # Get home team recent games
        home_team_id = game.home_team.team_id
        away_team_id = game.away_team.team_id
        
        # Get recent games for each team
        home_recent_games = await self._get_recent_team_games(
            team_id=home_team_id,
            game_date=game.date,
            lookback_games=lookback_games,
            lookback_days=lookback_days
        )
        
        away_recent_games = await self._get_recent_team_games(
            team_id=away_team_id,
            game_date=game.date,
            lookback_games=lookback_games,
            lookback_days=lookback_days
        )
        
        # Get head-to-head games
        h2h_games = await self._get_head_to_head_games(
            team1_id=home_team_id,
            team2_id=away_team_id,
            game_date=game.date,
            lookback_games=10,
            lookback_days=365 * 2  # Look back up to 2 years for H2H
        )
        
        # Extract rest days and schedule features
        features.update(await self._extract_schedule_features(
            game=game,
            home_recent_games=home_recent_games,
            away_recent_games=away_recent_games
        ))
        
        # Extract recent performance features
        features.update(self._extract_recent_performance_features(
            home_recent_games=home_recent_games,
            away_recent_games=away_recent_games
        ))
        
        # Extract streak and momentum features
        features.update(self._extract_streak_features(
            home_recent_games=home_recent_games,
            away_recent_games=away_recent_games
        ))
        
        # Extract head-to-head features
        features.update(self._extract_head_to_head_features(
            h2h_games=h2h_games,
            home_team_id=home_team_id
        ))
        
        return features
    
    async def _get_recent_team_games(
        self,
        team_id: str,
        game_date: datetime,
        lookback_games: int = 10,
        lookback_days: int = 60
    ) -> List[GameInDB]:
        """
        Get recent games for a team before a specified date.
        
        Args:
            team_id: Team ID
            game_date: Reference date (games before this date)
            lookback_games: Maximum number of games to retrieve
            lookback_days: Maximum number of days to look back
            
        Returns:
            List of recent games for the team
        """
        # Calculate earliest date to consider
        earliest_date = game_date - timedelta(days=lookback_days)
        
        # Query games where this team participated
        recent_games = await self.game_repository.find(
            filter={
                "$or": [
                    {"home_team.team_id": team_id},
                    {"away_team.team_id": team_id}
                ],
                "date": {"$lt": game_date, "$gte": earliest_date},
                "status.short": {"$in": ["FT", "Final", "Finished", "AOT"]}
            },
            sort=[("date", -1)],
            limit=lookback_games
        )
        
        return recent_games
    
    async def _get_head_to_head_games(
        self,
        team1_id: str,
        team2_id: str,
        game_date: datetime,
        lookback_games: int = 10,
        lookback_days: int = 365
    ) -> List[GameInDB]:
        """
        Get head-to-head games between two teams before a specified date.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            game_date: Reference date (games before this date)
            lookback_games: Maximum number of games to retrieve
            lookback_days: Maximum number of days to look back
            
        Returns:
            List of head-to-head games
        """
        # Calculate earliest date to consider
        earliest_date = game_date - timedelta(days=lookback_days)
        
        # Query head-to-head games
        h2h_games = await self.game_repository.find(
            filter={
                "$or": [
                    {"$and": [{"home_team.team_id": team1_id}, {"away_team.team_id": team2_id}]},
                    {"$and": [{"home_team.team_id": team2_id}, {"away_team.team_id": team1_id}]}
                ],
                "date": {"$lt": game_date, "$gte": earliest_date},
                "status.short": {"$in": ["FT", "Final", "Finished", "AOT"]}
            },
            sort=[("date", -1)],
            limit=lookback_games
        )
        
        return h2h_games
    
    async def _extract_schedule_features(
        self,
        game: GameInDB,
        home_recent_games: List[GameInDB],
        away_recent_games: List[GameInDB]
    ) -> Dict[str, Any]:
        """
        Extract rest days and schedule-related features.
        
        Args:
            game: Current game
            home_recent_games: Recent games for home team
            away_recent_games: Recent games for away team
            
        Returns:
            Dictionary of schedule-related features
        """
        features = {}
        
        # Calculate days since last game for each team
        home_rest_days = self._calculate_days_rest(game.date, home_recent_games)
        away_rest_days = self._calculate_days_rest(game.date, away_recent_games)
        
        features[f'{self.feature_prefix}home_days_rest'] = home_rest_days
        features[f'{self.feature_prefix}away_days_rest'] = away_rest_days
        features[f'{self.feature_prefix}rest_advantage'] = home_rest_days - away_rest_days
        
        # Identify back-to-back games
        features[f'{self.feature_prefix}home_back_to_back'] = 1 if home_rest_days <= 1 else 0
        features[f'{self.feature_prefix}away_back_to_back'] = 1 if away_rest_days <= 1 else 0
        
        # Identify third game in four nights
        home_third_in_four = await self._is_third_game_in_four_nights(game.home_team.team_id, game.date)
        away_third_in_four = await self._is_third_game_in_four_nights(game.away_team.team_id, game.date)
        
        features[f'{self.feature_prefix}home_third_in_four'] = 1 if home_third_in_four else 0
        features[f'{self.feature_prefix}away_third_in_four'] = 1 if away_third_in_four else 0
        
        # Road trip length (consecutive away games)
        home_road_games = await self._count_consecutive_road_games(game.home_team.team_id, game.date)
        away_road_games = await self._count_consecutive_road_games(game.away_team.team_id, game.date)
        
        features[f'{self.feature_prefix}home_road_trip_length'] = home_road_games
        features[f'{self.feature_prefix}away_road_trip_length'] = away_road_games
        
        # Home stand length (consecutive home games)
        home_home_games = await self._count_consecutive_home_games(game.home_team.team_id, game.date)
        away_home_games = await self._count_consecutive_home_games(game.away_team.team_id, game.date)
        
        features[f'{self.feature_prefix}home_stand_length'] = home_home_games
        
        # Travel distance/difficulty would be ideal but would require venue data
        # This is a placeholder for that concept
        features[f'{self.feature_prefix}schedule_advantage'] = (
            (3 * (away_road_games - home_road_games)) + 
            (2 * (features[f'{self.feature_prefix}away_back_to_back'] - features[f'{self.feature_prefix}home_back_to_back'])) +
            features[f'{self.feature_prefix}rest_advantage']
        )
        
        return features
    
    def _extract_recent_performance_features(
        self,
        home_recent_games: List[GameInDB],
        away_recent_games: List[GameInDB]
    ) -> Dict[str, Any]:
        """
        Extract features related to recent team performance.
        
        Args:
            home_recent_games: Recent games for home team
            away_recent_games: Recent games for away team
            
        Returns:
            Dictionary of recent performance features
        """
        features = {}
        
        # Calculate recent win percentage
        home_win_pct, home_wins, home_games = self._calculate_recent_win_percentage(
            home_recent_games, 
            is_home_team=True
        )
        
        away_win_pct, away_wins, away_games = self._calculate_recent_win_percentage(
            away_recent_games,
            is_home_team=False
        )
        
        features[f'{self.feature_prefix}home_recent_win_pct'] = home_win_pct
        features[f'{self.feature_prefix}away_recent_win_pct'] = away_win_pct
        features[f'{self.feature_prefix}recent_win_pct_diff'] = home_win_pct - away_win_pct
        
        # Calculate recent scoring averages
        home_scoring_avg, home_defense_avg = self._calculate_recent_scoring(home_recent_games, is_home_team=True)
        away_scoring_avg, away_defense_avg = self._calculate_recent_scoring(away_recent_games, is_home_team=False)
        
        features[f'{self.feature_prefix}home_recent_scoring_avg'] = home_scoring_avg
        features[f'{self.feature_prefix}home_recent_defense_avg'] = home_defense_avg
        features[f'{self.feature_prefix}away_recent_scoring_avg'] = away_scoring_avg
        features[f'{self.feature_prefix}away_recent_defense_avg'] = away_defense_avg
        
        # Calculate scoring differentials
        features[f'{self.feature_prefix}home_recent_point_diff'] = home_scoring_avg - home_defense_avg
        features[f'{self.feature_prefix}away_recent_point_diff'] = away_scoring_avg - away_defense_avg
        
        # Calculate expected scoring matchup
        features[f'{self.feature_prefix}home_off_vs_away_def'] = home_scoring_avg - away_defense_avg
        features[f'{self.feature_prefix}away_off_vs_home_def'] = away_scoring_avg - home_defense_avg
        
        # Net advantage based on recent performance
        features[f'{self.feature_prefix}net_recent_advantage'] = (
            features[f'{self.feature_prefix}home_off_vs_away_def'] - 
            features[f'{self.feature_prefix}away_off_vs_home_def']
        )
        
        return features
    
    def _extract_streak_features(
        self,
        home_recent_games: List[GameInDB],
        away_recent_games: List[GameInDB]
    ) -> Dict[str, Any]:
        """
        Extract streak and momentum-related features.
        
        Args:
            home_recent_games: Recent games for home team
            away_recent_games: Recent games for away team
            
        Returns:
            Dictionary of streak-related features
        """
        features = {}
        
        # Calculate current streaks
        home_streak, home_streak_type = self._calculate_current_streak(home_recent_games, is_home_team=True)
        away_streak, away_streak_type = self._calculate_current_streak(away_recent_games, is_home_team=False)
        
        # Convert to signed values (positive for wins, negative for losses)
        home_signed_streak = home_streak if home_streak_type == 'win' else -home_streak
        away_signed_streak = away_streak if away_streak_type == 'win' else -away_streak
        
        features[f'{self.feature_prefix}home_current_streak'] = home_signed_streak
        features[f'{self.feature_prefix}away_current_streak'] = away_signed_streak
        features[f'{self.feature_prefix}streak_advantage'] = home_signed_streak - away_signed_streak
        
        # Calculate momentum (weighted recent games)
        home_momentum = self._calculate_momentum_score(home_recent_games, is_home_team=True)
        away_momentum = self._calculate_momentum_score(away_recent_games, is_home_team=False)
        
        features[f'{self.feature_prefix}home_momentum'] = home_momentum
        features[f'{self.feature_prefix}away_momentum'] = away_momentum
        features[f'{self.feature_prefix}momentum_advantage'] = home_momentum - away_momentum
        
        # Calculate recent form as a single metric
        home_form = self._calculate_form_metric(home_recent_games, is_home_team=True)
        away_form = self._calculate_form_metric(away_recent_games, is_home_team=False)
        
        features[f'{self.feature_prefix}home_form'] = home_form
        features[f'{self.feature_prefix}away_form'] = away_form
        features[f'{self.feature_prefix}form_advantage'] = home_form - away_form
        
        return features
    
    def _extract_head_to_head_features(
        self,
        h2h_games: List[GameInDB],
        home_team_id: str
    ) -> Dict[str, Any]:
        """
        Extract head-to-head matchup features.
        
        Args:
            h2h_games: Head-to-head games between the teams
            home_team_id: ID of the home team
            
        Returns:
            Dictionary of head-to-head features
        """
        features = {}
        
        # Calculate head-to-head win percentage for home team
        home_wins = 0
        away_wins = 0
        
        if not h2h_games:
            # No head-to-head history
            features[f'{self.feature_prefix}h2h_games_count'] = 0
            features[f'{self.feature_prefix}home_h2h_win_pct'] = 0.5  # Default
            features[f'{self.feature_prefix}home_h2h_advantage'] = 0.0
            return features
            
        # Count wins for each team
        for game in h2h_games:
            home_score = game.home_team.scores.total
            away_score = game.away_team.scores.total
            
            if home_score is None or away_score is None:
                continue
                
            if game.home_team.team_id == home_team_id:
                if home_score > away_score:
                    home_wins += 1
                else:
                    away_wins += 1
            else:  # Away team in this game is current home team
                if away_score > home_score:
                    home_wins += 1
                else:
                    away_wins += 1
        
        total_games = home_wins + away_wins
        
        # Calculate features
        features[f'{self.feature_prefix}h2h_games_count'] = total_games
        
        if total_games > 0:
            features[f'{self.feature_prefix}home_h2h_win_pct'] = home_wins / total_games
            
            # Calculate advantage weighted by recency
            weighted_advantage = 0
            for i, game in enumerate(h2h_games):
                # More recent games have more weight
                weight = 1.0 - (i * 0.1)
                if weight < 0.3:
                    weight = 0.3  # Minimum weight
                
                home_score = game.home_team.scores.total
                away_score = game.away_team.scores.total
                
                if home_score is None or away_score is None:
                    continue
                
                # Calculate point differential as percentage of total points
                total_points = home_score + away_score
                if total_points == 0:
                    continue
                    
                if game.home_team.team_id == home_team_id:
                    diff = (home_score - away_score) / total_points
                else:
                    diff = (away_score - home_score) / total_points
                
                weighted_advantage += diff * weight
            
            features[f'{self.feature_prefix}home_h2h_advantage'] = weighted_advantage
        else:
            features[f'{self.feature_prefix}home_h2h_win_pct'] = 0.5
            features[f'{self.feature_prefix}home_h2h_advantage'] = 0.0
        
        return features
    
    def _calculate_days_rest(
        self,
        game_date: datetime,
        recent_games: List[GameInDB]
    ) -> int:
        """
        Calculate days of rest since last game.
        
        Args:
            game_date: Date of current game
            recent_games: List of recent games
            
        Returns:
            Number of days rest
        """
        if not recent_games:
            return 3  # Default if no recent games
        
        # Most recent game is first in the list (sorted by date descending)
        last_game_date = recent_games[0].date
        
        # Calculate days between games
        delta = game_date - last_game_date
        
        return delta.days
    
    async def _is_third_game_in_four_nights(
        self,
        team_id: str,
        game_date: datetime
    ) -> bool:
        """
        Check if this is the third game in four nights for a team.
        
        Args:
            team_id: Team ID
            game_date: Date of current game
            
        Returns:
            True if it's the third game in four nights, False otherwise
        """
        # Start date (3 days before current game)
        start_date = game_date - timedelta(days=3)
        
        # Count games in the 3-day window before current game
        recent_games = await self.game_repository.find(
            filter={
                "$or": [
                    {"home_team.team_id": team_id},
                    {"away_team.team_id": team_id}
                ],
                "date": {"$gte": start_date, "$lt": game_date},
                "status.short": {"$in": ["FT", "Final", "Finished", "AOT", "LIVE", "In Progress"]}
            }
        )
        
        return len(recent_games) >= 2
    
    async def _count_consecutive_road_games(
        self,
        team_id: str,
        game_date: datetime
    ) -> int:
        """
        Count consecutive road games for a team before this game.
        
        Args:
            team_id: Team ID
            game_date: Date of current game
            
        Returns:
            Number of consecutive road games
        """
        consecutive_count = 0
        
        # Get recent games for this team
        lookback_days = 30  # Reasonable maximum road trip length
        earliest_date = game_date - timedelta(days=lookback_days)
        
        recent_games = await self.game_repository.find(
            filter={
                "$or": [
                    {"home_team.team_id": team_id},
                    {"away_team.team_id": team_id}
                ],
                "date": {"$lt": game_date, "$gte": earliest_date},
                "status.short": {"$in": ["FT", "Final", "Finished", "AOT"]}
            },
            sort=[("date", -1)]  # Most recent games first
        )
        
        # Count consecutive road games
        for game in recent_games:
            if game.away_team.team_id == team_id:
                consecutive_count += 1
            else:
                # Home game breaks the streak
                break
        
        return consecutive_count
    
    async def _count_consecutive_home_games(
        self,
        team_id: str,
        game_date: datetime
    ) -> int:
        """
        Count consecutive home games for a team before this game.
        
        Args:
            team_id: Team ID
            game_date: Date of current game
            
        Returns:
            Number of consecutive home games
        """
        consecutive_count = 0
        
        # Get recent games for this team
        lookback_days = 30  # Reasonable maximum home stand length
        earliest_date = game_date - timedelta(days=lookback_days)
        
        recent_games = await self.game_repository.find(
            filter={
                "$or": [
                    {"home_team.team_id": team_id},
                    {"away_team.team_id": team_id}
                ],
                "date": {"$lt": game_date, "$gte": earliest_date},
                "status.short": {"$in": ["FT", "Final", "Finished", "AOT"]}
            },
            sort=[("date", -1)]  # Most recent games first
        )
        
        # Count consecutive home games
        for game in recent_games:
            if game.home_team.team_id == team_id:
                consecutive_count += 1
            else:
                # Away game breaks the streak
                break
        
        return consecutive_count
    
    def _calculate_recent_win_percentage(
        self,
        recent_games: List[GameInDB],
        is_home_team: bool
    ) -> Tuple[float, int, int]:
        """
        Calculate win percentage from recent games.
        
        Args:
            recent_games: List of recent games
            is_home_team: Whether to calculate for the home team (True) or away team (False)
            
        Returns:
            Tuple of (win_percentage, wins, total_games)
        """
        if not recent_games:
            return 0.5, 0, 0  # Default
        
        wins = 0
        games_played = 0
        
        for game in recent_games:
            # Skip games without scores
            if not game.home_team.scores or not game.away_team.scores:
                continue
                
            home_score = game.home_team.scores.total
            away_score = game.away_team.scores.total
            
            if home_score is None or away_score is None:
                continue
                
            games_played += 1
            
            # Determine if this team won
            if is_home_team:
                team_id = game.home_team.team_id if hasattr(game, 'home_team') else None
            else:
                team_id = game.away_team.team_id if hasattr(game, 'away_team') else None
            
            if team_id:
                if game.home_team.team_id == team_id:
                    if home_score > away_score:
                        wins += 1
                else:
                    if away_score > home_score:
                        wins += 1
        
        if games_played == 0:
            return 0.5, 0, 0
            
        return wins / games_played, wins, games_played
    
    def _calculate_recent_scoring(
        self,
        recent_games: List[GameInDB],
        is_home_team: bool
    ) -> Tuple[float, float]:
        """
        Calculate recent scoring and defense averages.
        
        Args:
            recent_games: List of recent games
            is_home_team: Whether to calculate for the home team (True) or away team (False)
            
        Returns:
            Tuple of (scoring_average, defense_average)
        """
        if not recent_games:
            return 100.0, 100.0  # Default
        
        total_points_scored = 0
        total_points_allowed = 0
        games_played = 0
        
        for game in recent_games:
            # Skip games without scores
            if not game.home_team.scores or not game.away_team.scores:
                continue
                
            home_score = game.home_team.scores.total
            away_score = game.away_team.scores.total
            
            if home_score is None or away_score is None:
                continue
                
            games_played += 1
            
            # Determine which scores to use based on whether this team is home or away
            if is_home_team:
                team_id = game.home_team.team_id if hasattr(game, 'home_team') else None
            else:
                team_id = game.away_team.team_id if hasattr(game, 'away_team') else None
            
            if team_id:
                if game.home_team.team_id == team_id:
                    total_points_scored += home_score
                    total_points_allowed += away_score
                else:
                    total_points_scored += away_score
                    total_points_allowed += home_score
        
        if games_played == 0:
            return 100.0, 100.0
            
        return total_points_scored / games_played, total_points_allowed / games_played
    
    def _calculate_current_streak(
        self,
        recent_games: List[GameInDB],
        is_home_team: bool
    ) -> Tuple[int, str]:
        """
        Calculate current streak length and type (win/loss).
        
        Args:
            recent_games: List of recent games
            is_home_team: Whether to calculate for the home team (True) or away team (False)
            
        Returns:
            Tuple of (streak_length, streak_type)
        """
        if not recent_games:
            return 0, 'none'
        
        streak = 0
        streak_type = 'none'
        
        for game in recent_games:
            # Skip games without scores
            if not game.home_team.scores or not game.away_team.scores:
                continue
                
            home_score = game.home_team.scores.total
            away_score = game.away_team.scores.total
            
            if home_score is None or away_score is None:
                continue
                
            # Determine if this team won
            if is_home_team:
                team_id = game.home_team.team_id if hasattr(game, 'home_team') else None
            else:
                team_id = game.away_team.team_id if hasattr(game, 'away_team') else None
            
            if team_id:
                is_win = (game.home_team.team_id == team_id and home_score > away_score) or \
                         (game.away_team.team_id == team_id and away_score > home_score)
                
                current_type = 'win' if is_win else 'loss'
                
                # If first game or continuing streak
                if streak_type == 'none' or streak_type == current_type:
                    streak_type = current_type
                    streak += 1
                else:
                    # Streak is broken
                    break
        
        return streak, streak_type
    
    def _calculate_momentum_score(
        self,
        recent_games: List[GameInDB],
        is_home_team: bool
    ) -> float:
        """
        Calculate a momentum score based on recent games.
        
        Args:
            recent_games: List of recent games
            is_home_team: Whether to calculate for the home team (True) or away team (False)
            
        Returns:
            Momentum score between -1 and 1
        """
        if not recent_games:
            return 0.0
        
        # Define weights for N most recent games (more recent = higher weight)
        max_games = min(len(recent_games), 10)
        weights = np.linspace(1.0, 0.1, max_games)
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        momentum = 0.0
        games_counted = 0
        
        for i, game in enumerate(recent_games[:max_games]):
            # Skip games without scores
            if not game.home_team.scores or not game.away_team.scores:
                continue
                
            home_score = game.home_team.scores.total
            away_score = game.away_team.scores.total
            
            if home_score is None or away_score is None:
                continue
                
            # Determine if this team won
            if is_home_team:
                team_id = game.home_team.team_id if hasattr(game, 'home_team') else None
            else:
                team_id = game.away_team.team_id if hasattr(game, 'away_team') else None
            
            if team_id:
                # Calculate point differential as percentage of total points
                if game.home_team.team_id == team_id:
                    points_for = home_score
                    points_against = away_score
                else:
                    points_for = away_score
                    points_against = home_score
                    
                total_points = points_for + points_against
                if total_points == 0:
                    continue
                    
                # Point differential (from -1 to 1)
                diff = (points_for - points_against) / total_points
                
                # Apply weight based on recency
                momentum += diff * weights[i]
                games_counted += 1
        
        if games_counted == 0:
            return 0.0
            
        # Scale to range from -1 to 1
        return max(min(momentum, 1.0), -1.0)
    
    def _calculate_form_metric(
        self,
        recent_games: List[GameInDB],
        is_home_team: bool
    ) -> float:
        """
        Calculate an overall form metric combining win/loss and scoring.
        
        Args:
            recent_games: List of recent games
            is_home_team: Whether to calculate for the home team (True) or away team (False)
            
        Returns:
            Form score between 0 and 1
        """
        # Combine win percentage, momentum, and scoring differential
        win_pct, _, _ = self._calculate_recent_win_percentage(recent_games, is_home_team)
        momentum = self._calculate_momentum_score(recent_games, is_home_team)
        scoring_avg, defense_avg = self._calculate_recent_scoring(recent_games, is_home_team)
        
        # Normalize scoring differential to [0, 1] range (assuming -30 to +30 point range)
        scoring_diff = scoring_avg - defense_avg
        norm_scoring_diff = (scoring_diff + 30) / 60
        norm_scoring_diff = max(0, min(1, norm_scoring_diff))
        
        # Normalize momentum to [0, 1] range
        norm_momentum = (momentum + 1) / 2
        
        # Combine metrics (weighted average)
        form = (0.5 * win_pct) + (0.3 * norm_momentum) + (0.2 * norm_scoring_diff)
        
        return form
    
    def get_feature_names(self) -> List[str]:
        """
        Get a list of feature names generated by this processor.
        
        Returns:
            List of feature names
        """
        return [
            # Schedule features
            f'{self.feature_prefix}home_days_rest',
            f'{self.feature_prefix}away_days_rest',
            f'{self.feature_prefix}rest_advantage',
            f'{self.feature_prefix}home_back_to_back',
            f'{self.feature_prefix}away_back_to_back',
            f'{self.feature_prefix}home_third_in_four',
            f'{self.feature_prefix}away_third_in_four',
            f'{self.feature_prefix}home_road_trip_length',
            f'{self.feature_prefix}away_road_trip_length',
            f'{self.feature_prefix}home_stand_length',
            f'{self.feature_prefix}schedule_advantage',
            
            # Recent performance features
            f'{self.feature_prefix}home_recent_win_pct',
            f'{self.feature_prefix}away_recent_win_pct',
            f'{self.feature_prefix}recent_win_pct_diff',
            f'{self.feature_prefix}home_recent_scoring_avg',
            f'{self.feature_prefix}home_recent_defense_avg',
            f'{self.feature_prefix}away_recent_scoring_avg',
            f'{self.feature_prefix}away_recent_defense_avg',
            f'{self.feature_prefix}home_recent_point_diff',
            f'{self.feature_prefix}away_recent_point_diff',
            f'{self.feature_prefix}home_off_vs_away_def',
            f'{self.feature_prefix}away_off_vs_home_def',
            f'{self.feature_prefix}net_recent_advantage',
            
            # Streak features
            f'{self.feature_prefix}home_current_streak',
            f'{self.feature_prefix}away_current_streak',
            f'{self.feature_prefix}streak_advantage',
            f'{self.feature_prefix}home_momentum',
            f'{self.feature_prefix}away_momentum',
            f'{self.feature_prefix}momentum_advantage',
            f'{self.feature_prefix}home_form',
            f'{self.feature_prefix}away_form',
            f'{self.feature_prefix}form_advantage',
            
            # Head-to-head features
            f'{self.feature_prefix}h2h_games_count',
            f'{self.feature_prefix}home_h2h_win_pct',
            f'{self.feature_prefix}home_h2h_advantage'
        ]