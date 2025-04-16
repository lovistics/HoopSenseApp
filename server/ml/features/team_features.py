"""
Team-related feature extraction for basketball predictions.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from app.core.logger import logger
from app.db.models.team import TeamInDB
from app.db.models.statistic import TeamStatisticsModel

class TeamFeatureProcessor:
    """
    Extracts features related to team statistics and matchups, such as:
    - Offensive and defensive ratings
    - Win percentages
    - Scoring differentials
    - Pace and style metrics
    - Head-to-head comparisons
    """
    
    def __init__(self):
        """Initialize the team feature processor."""
        self.feature_prefix = 'team_'
    
    async def extract_features(
        self,
        home_team: TeamInDB,
        away_team: TeamInDB,
        home_stats: TeamStatisticsModel,
        away_stats: TeamStatisticsModel
    ) -> Dict[str, Any]:
        """
        Extract team-related features.
        
        Args:
            home_team: Home team data
            away_team: Away team data
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            Dictionary of feature name to value
        """
        features = {}
        
        # Basic team records
        features.update(self._extract_record_features(home_stats, away_stats))
        
        # Offensive and defensive metrics
        features.update(self._extract_performance_metrics(home_stats, away_stats))
        
        # Team pace and style metrics
        features.update(self._extract_style_metrics(home_stats, away_stats))
        
        # Matchup-specific features
        features.update(self._extract_matchup_features(home_team, away_team, home_stats, away_stats))
        
        return features
    
    def _extract_record_features(
        self,
        home_stats: TeamStatisticsModel,
        away_stats: TeamStatisticsModel
    ) -> Dict[str, Any]:
        """
        Extract team record-related features.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            Dictionary of record-related features
        """
        features = {}
        
        # Win percentages
        features[f'{self.feature_prefix}home_win_pct'] = home_stats.win_percentage
        features[f'{self.feature_prefix}away_win_pct'] = away_stats.win_percentage
        features[f'{self.feature_prefix}win_pct_diff'] = home_stats.win_percentage - away_stats.win_percentage
        
        # Home/away specific records
        home_home_win_pct = self._get_home_win_percentage(home_stats)
        away_away_win_pct = self._get_away_win_percentage(away_stats)
        
        features[f'{self.feature_prefix}home_team_home_win_pct'] = home_home_win_pct
        features[f'{self.feature_prefix}away_team_away_win_pct'] = away_away_win_pct
        
        # Form (recent performance)
        features[f'{self.feature_prefix}home_form'] = self._calculate_form_score(home_stats.form)
        features[f'{self.feature_prefix}away_form'] = self._calculate_form_score(away_stats.form)
        features[f'{self.feature_prefix}form_diff'] = features[f'{self.feature_prefix}home_form'] - features[f'{self.feature_prefix}away_form']
        
        # Games played (experience/fatigue metric)
        features[f'{self.feature_prefix}home_games_played'] = home_stats.games.played.all if home_stats.games.played.all else 0
        features[f'{self.feature_prefix}away_games_played'] = away_stats.games.played.all if away_stats.games.played.all else 0
        
        return features
    
    def _extract_performance_metrics(
        self,
        home_stats: TeamStatisticsModel,
        away_stats: TeamStatisticsModel
    ) -> Dict[str, Any]:
        """
        Extract team offensive and defensive performance metrics.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            Dictionary of performance-related features
        """
        features = {}
        
        # Offensive and defensive ratings
        features[f'{self.feature_prefix}home_off_rating'] = home_stats.offensive_rating
        features[f'{self.feature_prefix}home_def_rating'] = home_stats.defensive_rating
        features[f'{self.feature_prefix}away_off_rating'] = away_stats.offensive_rating
        features[f'{self.feature_prefix}away_def_rating'] = away_stats.defensive_rating
        
        # Net ratings
        features[f'{self.feature_prefix}home_net_rating'] = home_stats.net_rating
        features[f'{self.feature_prefix}away_net_rating'] = away_stats.net_rating
        features[f'{self.feature_prefix}net_rating_diff'] = home_stats.net_rating - away_stats.net_rating
        
        # Points per game
        home_points_pg = self._get_points_per_game(home_stats)
        away_points_pg = self._get_points_per_game(away_stats)
        home_points_allowed_pg = self._get_points_allowed_per_game(home_stats)
        away_points_allowed_pg = self._get_points_allowed_per_game(away_stats)
        
        features[f'{self.feature_prefix}home_points_pg'] = home_points_pg
        features[f'{self.feature_prefix}away_points_pg'] = away_points_pg
        features[f'{self.feature_prefix}home_points_allowed_pg'] = home_points_allowed_pg
        features[f'{self.feature_prefix}away_points_allowed_pg'] = away_points_allowed_pg
        
        # Point differentials
        features[f'{self.feature_prefix}home_point_diff_pg'] = home_points_pg - home_points_allowed_pg
        features[f'{self.feature_prefix}away_point_diff_pg'] = away_points_pg - away_points_allowed_pg
        
        # Adjusted efficiency metrics (considering strength of opponent)
        # These are simplified estimates - in reality they would be more sophisticated
        features[f'{self.feature_prefix}home_adj_off_eff'] = home_stats.offensive_rating * (1 + 0.1 * home_stats.win_percentage)
        features[f'{self.feature_prefix}away_adj_off_eff'] = away_stats.offensive_rating * (1 + 0.1 * away_stats.win_percentage)
        features[f'{self.feature_prefix}home_adj_def_eff'] = home_stats.defensive_rating * (1 - 0.1 * home_stats.win_percentage)
        features[f'{self.feature_prefix}away_adj_def_eff'] = away_stats.defensive_rating * (1 - 0.1 * away_stats.win_percentage)
        
        return features
    
    def _extract_style_metrics(
        self,
        home_stats: TeamStatisticsModel,
        away_stats: TeamStatisticsModel
    ) -> Dict[str, Any]:
        """
        Extract team pace and style metrics.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            Dictionary of style-related features
        """
        features = {}
        
        # Pace (possessions per game)
        features[f'{self.feature_prefix}home_pace'] = home_stats.pace
        features[f'{self.feature_prefix}away_pace'] = away_stats.pace
        features[f'{self.feature_prefix}pace_diff'] = home_stats.pace - away_stats.pace
        
        # Average game pace (weighted average of both teams)
        features[f'{self.feature_prefix}expected_pace'] = (home_stats.pace + away_stats.pace) / 2
        
        # True shooting percentage
        features[f'{self.feature_prefix}home_ts_pct'] = home_stats.true_shooting_percentage
        features[f'{self.feature_prefix}away_ts_pct'] = away_stats.true_shooting_percentage
        features[f'{self.feature_prefix}ts_pct_diff'] = home_stats.true_shooting_percentage - away_stats.true_shooting_percentage
        
        # Offensive vs defensive style matchup
        # Higher value indicates home team's offense vs away team's defense advantage
        features[f'{self.feature_prefix}home_off_vs_away_def'] = home_stats.offensive_rating - away_stats.defensive_rating
        
        # Higher value indicates away team's offense vs home team's defense advantage
        features[f'{self.feature_prefix}away_off_vs_home_def'] = away_stats.offensive_rating - home_stats.defensive_rating
        
        # Net style advantage
        features[f'{self.feature_prefix}net_style_advantage'] = (
            features[f'{self.feature_prefix}home_off_vs_away_def'] - 
            features[f'{self.feature_prefix}away_off_vs_home_def']
        )
        
        return features
    
    def _extract_matchup_features(
        self,
        home_team: TeamInDB,
        away_team: TeamInDB,
        home_stats: TeamStatisticsModel,
        away_stats: TeamStatisticsModel
    ) -> Dict[str, Any]:
        """
        Extract matchup-specific features.
        
        Args:
            home_team: Home team data
            away_team: Away team data
            home_stats: Home team statistics
            away_stats: Away team statistics
            
        Returns:
            Dictionary of matchup-related features
        """
        features = {}
        
        # Conference/division matchup
        same_conference = (
            hasattr(home_team, 'conference') and 
            hasattr(away_team, 'conference') and
            home_team.conference == away_team.conference
        )
        
        same_division = (
            hasattr(home_team, 'division') and 
            hasattr(away_team, 'division') and
            home_team.division == away_team.division
        )
        
        features[f'{self.feature_prefix}same_conference'] = 1 if same_conference else 0
        features[f'{self.feature_prefix}same_division'] = 1 if same_division else 0
        
        # Matchup historical data would ideally come from head-to-head data
        # This is a placeholder that would be replaced with actual data
        features[f'{self.feature_prefix}home_team_h2h_advantage'] = 0.0
        
        # Team strength percentile difference
        # Higher values mean bigger gap in team quality
        win_pct_diff = abs(home_stats.win_percentage - away_stats.win_percentage)
        features[f'{self.feature_prefix}team_strength_gap'] = win_pct_diff
        
        # Home team underdog flag
        features[f'{self.feature_prefix}home_team_underdog'] = 1 if home_stats.win_percentage < away_stats.win_percentage else 0
        
        # Significant advantage flag (gap > 20%)
        features[f'{self.feature_prefix}significant_advantage'] = 1 if win_pct_diff > 0.2 else 0
        
        return features
    
    def _get_home_win_percentage(self, stats: TeamStatisticsModel) -> float:
        """Calculate home win percentage from team statistics."""
        if not stats or not stats.games.played.home or stats.games.played.home == 0:
            return 0.5  # Default if no data
        
        home_wins = stats.games.wins.get('home', {}).get('total', 0)
        return home_wins / stats.games.played.home
    
    def _get_away_win_percentage(self, stats: TeamStatisticsModel) -> float:
        """Calculate away win percentage from team statistics."""
        if not stats or not stats.games.played.away or stats.games.played.away == 0:
            return 0.5  # Default if no data
        
        away_wins = stats.games.wins.get('away', {}).get('total', 0)
        return away_wins / stats.games.played.away
    
    def _calculate_form_score(self, form_string: str) -> float:
        """
        Calculate a form score from the form string (e.g., "WLWWL").
        
        Args:
            form_string: String of recent results (W=win, L=loss, D=draw)
            
        Returns:
            Form score between 0 and 1
        """
        if not form_string:
            return 0.5  # Default if no form data
        
        # More recent games are weighted more heavily
        weights = np.linspace(0.5, 1.0, len(form_string))
        weights = weights / weights.sum()  # Normalize
        
        score = 0
        for i, result in enumerate(form_string):
            if result.upper() == 'W':
                score += weights[i]
            elif result.upper() == 'D':
                score += 0.5 * weights[i]
                
        return score
    
    def _get_points_per_game(self, stats: TeamStatisticsModel) -> float:
        """Calculate points per game from team statistics."""
        if not stats or not stats.games.played.all or stats.games.played.all == 0:
            return 0.0  # Default if no data
        
        # This is a bit hacky - we should actually get the data from the API
        # but we're working with what's available in the model
        try:
            return stats.points.for_points.get('average', {}).all
        except Exception:
            # Fallback calculation if the average field is not available
            total_points = stats.points.for_points.get('total', {}).all or 0
            games_played = stats.games.played.all or 1
            return total_points / games_played
    
    def _get_points_allowed_per_game(self, stats: TeamStatisticsModel) -> float:
        """Calculate points allowed per game from team statistics."""
        if not stats or not stats.games.played.all or stats.games.played.all == 0:
            return 0.0  # Default if no data
        
        try:
            return stats.points.against.get('average', {}).all
        except Exception:
            # Fallback calculation if the average field is not available
            total_points_against = stats.points.against.get('total', {}).all or 0
            games_played = stats.games.played.all or 1
            return total_points_against / games_played
    
    def get_feature_names(self) -> List[str]:
        """
        Get a list of feature names generated by this processor.
        
        Returns:
            List of feature names
        """
        return [
            # Record features
            f'{self.feature_prefix}home_win_pct',
            f'{self.feature_prefix}away_win_pct',
            f'{self.feature_prefix}win_pct_diff',
            f'{self.feature_prefix}home_team_home_win_pct',
            f'{self.feature_prefix}away_team_away_win_pct',
            f'{self.feature_prefix}home_form',
            f'{self.feature_prefix}away_form',
            f'{self.feature_prefix}form_diff',
            f'{self.feature_prefix}home_games_played',
            f'{self.feature_prefix}away_games_played',
            
            # Performance metrics
            f'{self.feature_prefix}home_off_rating',
            f'{self.feature_prefix}home_def_rating',
            f'{self.feature_prefix}away_off_rating',
            f'{self.feature_prefix}away_def_rating',
            f'{self.feature_prefix}home_net_rating',
            f'{self.feature_prefix}away_net_rating',
            f'{self.feature_prefix}net_rating_diff',
            f'{self.feature_prefix}home_points_pg',
            f'{self.feature_prefix}away_points_pg',
            f'{self.feature_prefix}home_points_allowed_pg',
            f'{self.feature_prefix}away_points_allowed_pg',
            f'{self.feature_prefix}home_point_diff_pg',
            f'{self.feature_prefix}away_point_diff_pg',
            f'{self.feature_prefix}home_adj_off_eff',
            f'{self.feature_prefix}away_adj_off_eff',
            f'{self.feature_prefix}home_adj_def_eff',
            f'{self.feature_prefix}away_adj_def_eff',
            
            # Style metrics
            f'{self.feature_prefix}home_pace',
            f'{self.feature_prefix}away_pace',
            f'{self.feature_prefix}pace_diff',
            f'{self.feature_prefix}expected_pace',
            f'{self.feature_prefix}home_ts_pct',
            f'{self.feature_prefix}away_ts_pct',
            f'{self.feature_prefix}ts_pct_diff',
            f'{self.feature_prefix}home_off_vs_away_def',
            f'{self.feature_prefix}away_off_vs_home_def',
            f'{self.feature_prefix}net_style_advantage',
            
            # Matchup features
            f'{self.feature_prefix}same_conference',
            f'{self.feature_prefix}same_division',
            f'{self.feature_prefix}home_team_h2h_advantage',
            f'{self.feature_prefix}team_strength_gap',
            f'{self.feature_prefix}home_team_underdog',
            f'{self.feature_prefix}significant_advantage'
        ]