"""
Repository for statistics data operations.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from bson import ObjectId

from app.core.logger import logger
from app.db.models.statistic import (
    TeamStatisticsModel, PlayerStatisticsModel,
    TEAM_STATS_COLLECTION, PLAYER_STATS_COLLECTION
)
from server.data.repositories.base_repository import BaseRepository


class TeamStatsRepository(BaseRepository[TeamStatisticsModel]):
    """
    Repository for team statistics-related database operations.
    """
    
    def __init__(self):
        """Initialize the team statistics repository."""
        super().__init__(TEAM_STATS_COLLECTION, TeamStatisticsModel)
    
    async def find_by_team_season(
        self,
        team_id: str,
        season_id: str
    ) -> Optional[TeamStatisticsModel]:
        """
        Find statistics for a team in a season.
        
        Args:
            team_id: Team ID
            season_id: Season ID
            
        Returns:
            Team statistics or None if not found
        """
        return await self.find_one({
            "team_id": team_id,
            "season_id": season_id
        })
    
    async def find_by_teams(
        self,
        team_ids: List[str],
        season_id: Optional[str] = None
    ) -> List[TeamStatisticsModel]:
        """
        Find statistics for multiple teams.
        
        Args:
            team_ids: List of team IDs
            season_id: Optional season ID filter
            
        Returns:
            List of team statistics
        """
        # Build filter
        filter = {"team_id": {"$in": team_ids}}
        
        if season_id:
            filter["season_id"] = season_id
        
        # Get team statistics
        return await self.find(filter)
    
    async def get_team_comparison(
        self,
        team1_id: str,
        team2_id: str,
        season_id: str
    ) -> Tuple[Optional[TeamStatisticsModel], Optional[TeamStatisticsModel]]:
        """
        Get statistics for two teams for comparison.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            season_id: Season ID
            
        Returns:
            Tuple of (team1_stats, team2_stats)
        """
        team1_stats = await self.find_by_team_season(team1_id, season_id)
        team2_stats = await self.find_by_team_season(team2_id, season_id)
        
        return team1_stats, team2_stats
    
    async def get_league_stats_ranking(
        self,
        season_id: str,
        league_id: Optional[str] = None,
        stat_field: str = "win_percentage",
        limit: int = 30,
        sort_direction: int = -1  # -1 for descending, 1 for ascending
    ) -> List[Dict[str, Any]]:
        """
        Get team statistics ranking by a specific stat.
        
        Args:
            season_id: Season ID
            league_id: Optional league ID filter
            stat_field: Field to rank by (e.g., "win_percentage", "offensive_rating")
            limit: Maximum number of teams to return
            sort_direction: Sort direction (-1 for descending, 1 for ascending)
            
        Returns:
            List of team statistics ranked by the specified stat
        """
        # Build pipeline
        pipeline = [
            {"$match": {"season_id": season_id}}
        ]
        
        if league_id:
            pipeline[0]["$match"]["league_id"] = league_id
        
        pipeline.extend([
            {"$sort": {stat_field: sort_direction}},
            {"$limit": limit}
        ])
        
        # Execute aggregation
        return await self.aggregate(pipeline)


class PlayerStatsRepository(BaseRepository[PlayerStatisticsModel]):
    """
    Repository for player statistics-related database operations.
    """
    
    def __init__(self):
        """Initialize the player statistics repository."""
        super().__init__(PLAYER_STATS_COLLECTION, PlayerStatisticsModel)
    
    async def find_by_player_game(
        self,
        player_id: str,
        game_id: str
    ) -> Optional[PlayerStatisticsModel]:
        """
        Find statistics for a player in a game.
        
        Args:
            player_id: Player ID
            game_id: Game ID
            
        Returns:
            Player game statistics or None if not found
        """
        return await self.find_one({
            "player_id": player_id,
            "game_id": game_id
        })
    
    async def find_recent_games(
        self,
        player_id: str,
        limit: int = 10
    ) -> List[PlayerStatisticsModel]:
        """
        Find recent game statistics for a player.
        
        Args:
            player_id: Player ID
            limit: Maximum number of games to return
            
        Returns:
            List of recent game statistics
        """
        return await self.find(
            filter={"player_id": player_id},
            sort=[("created_at", -1)],
            limit=limit
        )
    
    async def find_by_game(
        self,
        game_id: str,
        team_id: Optional[str] = None,
        player_type: Optional[str] = None
    ) -> List[PlayerStatisticsModel]:
        """
        Find statistics for players in a game.
        
        Args:
            game_id: Game ID
            team_id: Optional team ID filter
            player_type: Optional player type filter (e.g., "starters", "bench")
            
        Returns:
            List of player game statistics
        """
        # Build filter
        filter = {"game_id": game_id}
        
        if team_id:
            filter["team_id"] = team_id
        
        if player_type:
            filter["type"] = player_type
        
        # Get player statistics
        return await self.find(
            filter=filter,
            sort=[("points", -1)]
        )
    
    async def calculate_season_averages(
        self,
        player_id: str,
        season_id: str
    ) -> Dict[str, Any]:
        """
        Calculate season averages for a player.
        
        Args:
            player_id: Player ID
            season_id: Season ID
            
        Returns:
            Season average statistics
        """
        # This requires game data with season_id, so we'll use aggregation pipeline
        pipeline = [
            {
                "$lookup": {
                    "from": "games",
                    "localField": "game_id",
                    "foreignField": "_id",
                    "as": "game"
                }
            },
            {"$unwind": "$game"},
            {
                "$match": {
                    "player_id": player_id,
                    "game.season_id": season_id
                }
            },
            {
                "$group": {
                    "_id": "$player_id",
                    "games_played": {"$sum": 1},
                    "total_points": {"$sum": "$points"},
                    "total_rebounds": {"$sum": "$rebounds"},
                    "total_assists": {"$sum": "$assists"},
                    "field_goals_made": {"$sum": "$field_goals.made"},
                    "field_goals_attempted": {"$sum": "$field_goals.attempts"},
                    "three_points_made": {"$sum": "$three_points.made"},
                    "three_points_attempted": {"$sum": "$three_points.attempts"},
                    "free_throws_made": {"$sum": "$free_throws.made"},
                    "free_throws_attempted": {"$sum": "$free_throws.attempts"}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "player_id": "$_id",
                    "games_played": 1,
                    "points_per_game": {"$divide": ["$total_points", "$games_played"]},
                    "rebounds_per_game": {"$divide": ["$total_rebounds", "$games_played"]},
                    "assists_per_game": {"$divide": ["$total_assists", "$games_played"]},
                    "field_goal_percentage": {
                        "$cond": [
                            {"$eq": ["$field_goals_attempted", 0]},
                            0,
                            {"$divide": ["$field_goals_made", "$field_goals_attempted"]}
                        ]
                    },
                    "three_point_percentage": {
                        "$cond": [
                            {"$eq": ["$three_points_attempted", 0]},
                            0,
                            {"$divide": ["$three_points_made", "$three_points_attempted"]}
                        ]
                    },
                    "free_throw_percentage": {
                        "$cond": [
                            {"$eq": ["$free_throws_attempted", 0]},
                            0,
                            {"$divide": ["$free_throws_made", "$free_throws_attempted"]}
                        ]
                    }
                }
            }
        ]
        
        results = await self.aggregate(pipeline)
        
        if not results:
            # Return default structure with zeros
            return {
                "player_id": player_id,
                "games_played": 0,
                "points_per_game": 0.0,
                "rebounds_per_game": 0.0,
                "assists_per_game": 0.0,
                "field_goal_percentage": 0.0,
                "three_point_percentage": 0.0,
                "free_throw_percentage": 0.0
            }
        
        return results[0]
    
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
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Build pipeline
        pipeline = [
            {
                "$match": {
                    "created_at": {"$gte": start_date, "$lte": end_date}
                }
            },
            {
                "$group": {
                    "_id": "$player_id",
                    "avg_stat": {"$avg": f"${stat_field}"},
                    "max_stat": {"$max": f"${stat_field}"},
                    "games_played": {"$sum": 1},
                    "last_game": {"$max": "$created_at"}
                }
            },
            {"$sort": {"avg_stat": -1}},
            {"$limit": limit},
            {
                "$lookup": {
                    "from": "players",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "player"
                }
            },
            {"$unwind": "$player"},
            {
                "$project": {
                    "player_id": "$_id",
                    "player_name": "$player.name",
                    "team_id": "$player.team_id",
                    "avg_stat": 1,
                    "max_stat": 1,
                    "games_played": 1,
                    "last_game": 1
                }
            }
        ]
        
        # Execute aggregation
        return await self.aggregate(pipeline)