"""
Statistics service for handling statistical calculations.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

from fastapi import HTTPException, status

from app.core.logger import logger
from app.db.models.statistic import TeamStatisticsModel, PlayerStatisticsModel
from server.data.repositories.stats_repository import TeamStatsRepository, PlayerStatsRepository
from server.data.repositories.game_repository import GameRepository
from server.data.repositories.team_repository import TeamRepository
from server.data.repositories.player_repository import PlayerRepository


class StatisticsService:
    """Service for statistics-related operations."""
    
    def __init__(self):
        """Initialize the statistics service with repositories."""
        self.team_stats_repository = TeamStatsRepository()
        self.player_stats_repository = PlayerStatsRepository()
        self.game_repository = GameRepository()
        self.team_repository = TeamRepository()
        self.player_repository = PlayerRepository()
    
    async def get_team_statistics(
        self,
        team_id: str,
        season_id: Optional[str] = None
    ) -> Optional[TeamStatisticsModel]:
        """
        Get statistics for a team.
        
        Args:
            team_id: The ID of the team
            season_id: Optional season ID filter
            
        Returns:
            Team statistics or None if not found
        """
        return await self.team_stats_repository.find_by_team_season(team_id, season_id)
    
    async def update_team_statistics(
        self,
        team_id: str,
        season_id: str,
        stats_data: Union[Dict[str, Any], TeamStatisticsModel]
    ) -> Optional[TeamStatisticsModel]:
        """
        Update statistics for a team.
        
        Args:
            team_id: The ID of the team
            season_id: The ID of the season
            stats_data: The statistics data to update
            
        Returns:
            Updated team statistics or None if update failed
        """
        # Convert to dict if it's a TeamStatisticsModel
        if isinstance(stats_data, TeamStatisticsModel):
            stats_dict = stats_data.dict(exclude={"id", "team_id", "season_id"})
        else:
            stats_dict = stats_data
        
        # Check if stats already exist
        existing_stats = await self.team_stats_repository.find_by_team_season(team_id, season_id)
        
        if existing_stats:
            # Update existing stats
            updated = await self.team_stats_repository.update(
                id=str(existing_stats.id),
                update={"$set": stats_dict}
            )
            
            if not updated:
                return None
                
            # Return updated stats
            return await self.team_stats_repository.find_by_id(str(existing_stats.id))
        else:
            # Create new stats
            if isinstance(stats_data, TeamStatisticsModel):
                stats_model = stats_data
            else:
                # Check if external_team_id is provided
                if "external_team_id" not in stats_dict:
                    # Get team to find external ID
                    team = await self.team_repository.find_by_id(team_id)
                    if not team:
                        return None
                    external_team_id = team.external_id
                else:
                    external_team_id = stats_dict["external_team_id"]
                
                # Create model
                stats_model = TeamStatisticsModel(
                    team_id=team_id,
                    external_team_id=external_team_id,
                    league_id=stats_dict.get("league_id", ""),
                    season_id=season_id,
                    **stats_dict
                )
            
            # Create stats
            stats_id = await self.team_stats_repository.create(stats_model)
            
            if not stats_id:
                return None
                
            # Return created stats
            return await self.team_stats_repository.find_by_id(stats_id)
    
    async def get_player_statistics(
        self,
        player_id: str,
        game_id: Optional[str] = None,
        limit: int = 10
    ) -> List[PlayerStatisticsModel]:
        """
        Get statistics for a player.
        
        Args:
            player_id: The ID of the player
            game_id: Optional game ID filter
            limit: Maximum number of statistics to return
            
        Returns:
            List of player statistics
        """
        if game_id:
            # Get statistics for a specific game
            stat = await self.player_stats_repository.find_by_player_game(player_id, game_id)
            return [stat] if stat else []
        else:
            # Get recent game statistics
            return await self.player_stats_repository.find_recent_games(player_id, limit)
    
    async def save_player_statistics(
        self,
        player_statistics: PlayerStatisticsModel
    ) -> Optional[PlayerStatisticsModel]:
        """
        Save statistics for a player.
        
        Args:
            player_statistics: The statistics to save
            
        Returns:
            Saved player statistics or None if save failed
        """
        # Check if statistics already exist
        existing_stats = await self.player_stats_repository.find_by_player_game(
            player_statistics.player_id,
            player_statistics.game_id
        )
        
        if existing_stats:
            # Update existing statistics
            updated = await self.player_stats_repository.update(
                id=str(existing_stats.id),
                update={"$set": player_statistics.dict(exclude={"id"})}
            )
            
            if not updated:
                return None
                
            # Return updated statistics
            return await self.player_stats_repository.find_by_id(str(existing_stats.id))
        else:
            # Create new statistics
            stats_id = await self.player_stats_repository.create(player_statistics)
            
            if not stats_id:
                return None
                
            # Return created statistics
            return await self.player_stats_repository.find_by_id(stats_id)
    
    async def calculate_team_form(
        self,
        team_id: str,
        games_limit: int = 5
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Calculate team form based on recent games.
        
        Args:
            team_id: The ID of the team
            games_limit: Maximum number of games to consider
            
        Returns:
            Tuple of (form string, game details)
        """
        # Find recent completed games for the team
        recent_games = await self.game_repository.find(
            filter={
                "$or": [
                    {"home_team.team_id": team_id},
                    {"away_team.team_id": team_id}
                ],
                "status.short": {"$in": ["FT", "Final", "Finished", "AOT"]}
            },
            sort=[("date", -1)],
            limit=games_limit
        )
        
        # Calculate form
        form_string = ""
        game_details = []
        
        for game in recent_games:
            is_home = game.home_team.team_id == team_id
            team_score = game.home_team.scores.total if is_home else game.away_team.scores.total
            opponent_score = game.away_team.scores.total if is_home else game.home_team.scores.total
            opponent_name = game.away_team.name if is_home else game.home_team.name
            
            # Handle missing scores
            if team_score is None or opponent_score is None:
                continue
            
            # Determine result
            if team_score > opponent_score:
                form_string += "W"
                result = "W"
            elif team_score < opponent_score:
                form_string += "L"
                result = "L"
            else:
                form_string += "D"
                result = "D"
                
            # Add game details
            game_details.append({
                "game_id": str(game.id),
                "opponent": opponent_name,
                "result": result,
                "score": f"{team_score}-{opponent_score}",
                "date": game.date,
                "is_home": is_home
            })
            
        return form_string, game_details
    
    async def calculate_season_averages(
        self,
        player_id: str,
        season_id: str
    ) -> Dict[str, Any]:
        """
        Calculate season averages for a player.
        
        Args:
            player_id: The ID of the player
            season_id: The ID of the season
            
        Returns:
            Dictionary of season averages
        """
        return await self.player_stats_repository.calculate_season_averages(
            player_id=player_id,
            season_id=season_id
        )
    
    async def calculate_head_to_head_stats(
        self,
        team1_id: str,
        team2_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate head-to-head statistics between two teams.
        
        Args:
            team1_id: The ID of the first team
            team2_id: The ID of the second team
            limit: Maximum number of games to consider
            
        Returns:
            Dictionary of head-to-head statistics
        """
        # Get head-to-head games
        games = await self.game_repository.find_head_to_head_games(
            team1_id=team1_id,
            team2_id=team2_id,
            limit=limit
        )
        
        if not games:
            return {}
            
        # Calculate stats
        team1_wins = 0
        team2_wins = 0
        team1_total_points = 0
        team2_total_points = 0
        game_details = []
        
        for game in games:
            team1_is_home = game.home_team.team_id == team1_id
            
            # Handle missing scores
            if (not game.home_team.scores or not game.away_team.scores or
                game.home_team.scores.total is None or game.away_team.scores.total is None):
                continue
                
            team1_score = game.home_team.scores.total if team1_is_home else game.away_team.scores.total
            team2_score = game.away_team.scores.total if team1_is_home else game.home_team.scores.total
            
            # Update totals
            team1_total_points += team1_score
            team2_total_points += team2_score
            
            # Determine winner
            if team1_score > team2_score:
                team1_wins += 1
            elif team2_score > team1_score:
                team2_wins += 1
                
            # Add game details
            game_details.append({
                "game_id": str(game.id),
                "date": game.date,
                "team1_score": team1_score,
                "team2_score": team2_score,
                "team1_is_home": team1_is_home,
                "venue": game.venue
            })
            
        # Calculate statistics
        total_games = len(game_details)
        
        if total_games == 0:
            return {
                "team1_wins": 0,
                "team2_wins": 0,
                "total_games": 0,
                "team1_win_percentage": 0,
                "team1_avg_points": 0,
                "team2_avg_points": 0,
                "avg_point_differential": 0,
                "recent_games": []
            }
        
        return {
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "total_games": total_games,
            "team1_win_percentage": round((team1_wins / total_games) * 100, 1),
            "team1_avg_points": round(team1_total_points / total_games, 1),
            "team2_avg_points": round(team2_total_points / total_games, 1),
            "avg_point_differential": round((team1_total_points - team2_total_points) / total_games, 1),
            "recent_games": game_details
        }
    
    async def get_top_performers(
        self,
        stat_field: str = "points",
        days: int = 7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top performers by a specific stat.
        
        Args:
            stat_field: Field to rank by (e.g., "points", "rebounds", "assists")
            days: Number of days to look back
            limit: Maximum number of players to return
            
        Returns:
            List of top performers
        """
        return await self.player_stats_repository.get_top_performers(
            stat_field=stat_field,
            days=days,
            limit=limit
        )