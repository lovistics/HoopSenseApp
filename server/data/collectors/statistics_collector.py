"""
Collector for basketball statistics data.
"""
from typing import List, Dict, Any, Optional, Tuple, Union

from bson import ObjectId

from app.core.logger import logger
from app.db.models.statistic import (
    TeamStatisticsModel, PlayerStatisticsModel, 
    GamesBreakdown, WinLossBreakdown, GameStatsBreakdown,
    PointsBreakdown, PointsBreakdownDetail, FieldGoalStats,
    get_team_stats_dict, get_player_stats_dict,
    TEAM_STATS_COLLECTION, PLAYER_STATS_COLLECTION
)
from server.data.repositories.team_repository import TeamRepository
from server.data.repositories.player_repository import PlayerRepository
from server.data.repositories.game_repository import GameRepository
from server.data.repositories.league_repository import LeagueRepository
from server.data.repositories.season_repository import SeasonRepository
from server.data.providers.basketball_api import BasketballAPI
from server.data.processors.cleaner import DataCleaner
from server.data.processors.transformer import DataTransformer


class StatisticsCollector:
    """
    Collects and processes basketball statistics data.
    """
    
    def __init__(
        self, 
        api: Optional[BasketballAPI] = None,
        team_repository: Optional[TeamRepository] = None,
        player_repository: Optional[PlayerRepository] = None,
        game_repository: Optional[GameRepository] = None,
        league_repository: Optional[LeagueRepository] = None,
        season_repository: Optional[SeasonRepository] = None
    ):
        """
        Initialize the statistics collector.
        
        Args:
            api: Optional BasketballAPI instance
            team_repository: Optional TeamRepository instance
            player_repository: Optional PlayerRepository instance
            game_repository: Optional GameRepository instance
            league_repository: Optional LeagueRepository instance
            season_repository: Optional SeasonRepository instance
        """
        self.api = api or BasketballAPI()
        self.team_repository = team_repository or TeamRepository()
        self.player_repository = player_repository or PlayerRepository()
        self.game_repository = game_repository or GameRepository()
        self.league_repository = league_repository or LeagueRepository()
        self.season_repository = season_repository or SeasonRepository()
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
    
    async def collect_team_statistics(
        self,
        team_id: str,
        league_id: str,
        season_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Collect team statistics for a team in a league and season.
        
        Args:
            team_id: MongoDB team ID
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            Team statistics data or None if not found
        """
        # Get team from database
        team = await self.team_repository.find_by_id(team_id)
        if not team:
            logger.error(f"Team not found: {team_id}")
            return None
        
        # Get league from database
        league = await self.league_repository.find_by_id(league_id)
        if not league:
            logger.error(f"League not found: {league_id}")
            return None
        
        # Get season from database
        season = await self.season_repository.find_by_id(season_id)
        if not season:
            logger.error(f"Season not found: {season_id}")
            return None
        
        # Get external IDs for API calls
        api_team_id = team.external_id
        api_league_id = league.external_id
        api_season = None
        
        # Find the season in the league
        for ls in league.seasons:
            if ls.season_id == season_id:
                api_season = ls.external_season
                break
        
        if not api_season:
            logger.error(f"Season {season_id} not found in league {league_id}")
            return None
        
        logger.info(f"Collecting statistics for team {api_team_id}, league {api_league_id}, season {api_season}")
        
        # Fetch team statistics from API
        return await self.api.get_team_statistics(
            team_id=api_team_id,
            league_id=api_league_id,
            season=api_season
        )
    
    async def collect_player_statistics(
        self,
        player_id: str,
        game_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Collect player statistics for a game.
        
        Args:
            player_id: MongoDB player ID
            game_id: MongoDB game ID
            
        Returns:
            Player statistics data or None if not found
        """
        # Get player from database
        player = await self.player_repository.find_by_id(player_id)
        if not player:
            logger.error(f"Player not found: {player_id}")
            return None
        
        # Get game from database
        game = await self.game_repository.find_by_id(game_id)
        if not game:
            logger.error(f"Game not found: {game_id}")
            return None
        
        # Get external IDs for API calls
        api_player_id = player.external_id
        api_game_id = game.external_id
        
        logger.info(f"Collecting statistics for player {api_player_id}, game {api_game_id}")
        
        # Fetch game statistics from API
        game_stats = await self.api.get_game_statistics(game_id=api_game_id)
        
        # Find player statistics in game stats
        player_stats = None
        if game_stats and "players" in game_stats:
            for team_data in game_stats["players"]:
                for player_data in team_data.get("players", []):
                    if player_data["player"]["id"] == api_player_id:
                        player_stats = {
                            "player": player_data["player"],
                            "team": team_data["team"],
                            "game": {"id": api_game_id},
                            "statistics": player_data["statistics"]
                        }
                        break
        
        if not player_stats:
            logger.warning(f"No statistics found for player {api_player_id} in game {api_game_id}")
            return None
        
        return player_stats
    
    async def collect_game_statistics(
        self,
        game_id: str
    ) -> Dict[str, Any]:
        """
        Collect statistics for all players in a game.
        
        Args:
            game_id: MongoDB game ID
            
        Returns:
            Game statistics data
        """
        # Get game from database
        game = await self.game_repository.find_by_id(game_id)
        if not game:
            logger.error(f"Game not found: {game_id}")
            return {}
        
        # Get external ID for API call
        api_game_id = game.external_id
        
        logger.info(f"Collecting statistics for game {api_game_id}")
        
        # Fetch game statistics from API
        game_stats = await self.api.get_game_statistics(game_id=api_game_id)
        
        if not game_stats:
            logger.warning(f"No statistics found for game {api_game_id}")
            return {}
        
        return game_stats
    
    async def process_team_statistics(
        self, 
        stats_data: Dict[str, Any],
        team_id: str,
        league_id: str,
        season_id: str
    ) -> Optional[TeamStatisticsModel]:
        """
        Process raw team statistics data into a team statistics model.
        
        Args:
            stats_data: Raw team statistics data from API
            team_id: MongoDB team ID
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            Team statistics model or None if processing fails
        """
        try:
            # Get team for external ID
            team = await self.team_repository.find_by_id(team_id)
            if not team:
                logger.error(f"Team not found: {team_id}")
                return None
            
            # Check if we have valid statistics data
            if not stats_data or not isinstance(stats_data, dict):
                logger.error(f"Invalid statistics data for team {team_id}")
                return None
            
            # Process games data
            games_data = stats_data.get("games", {})
            games_breakdown = GamesBreakdown(
                played=GameStatsBreakdown(
                    home=games_data.get("played", {}).get("home", 0),
                    away=games_data.get("played", {}).get("away", 0),
                    all=games_data.get("played", {}).get("all", 0)
                ),
                wins={
                    "total": WinLossBreakdown(
                        total=games_data.get("wins", {}).get("all", {}).get("total", 0),
                        percentage=games_data.get("wins", {}).get("all", {}).get("percentage", 0.0)
                    ),
                    "home": WinLossBreakdown(
                        total=games_data.get("wins", {}).get("home", {}).get("total", 0),
                        percentage=games_data.get("wins", {}).get("home", {}).get("percentage", 0.0)
                    ),
                    "away": WinLossBreakdown(
                        total=games_data.get("wins", {}).get("away", {}).get("total", 0),
                        percentage=games_data.get("wins", {}).get("away", {}).get("percentage", 0.0)
                    )
                },
                loses={
                    "total": WinLossBreakdown(
                        total=games_data.get("loses", {}).get("all", {}).get("total", 0),
                        percentage=games_data.get("loses", {}).get("all", {}).get("percentage", 0.0)
                    ),
                    "home": WinLossBreakdown(
                        total=games_data.get("loses", {}).get("home", {}).get("total", 0),
                        percentage=games_data.get("loses", {}).get("home", {}).get("percentage", 0.0)
                    ),
                    "away": WinLossBreakdown(
                        total=games_data.get("loses", {}).get("away", {}).get("total", 0),
                        percentage=games_data.get("loses", {}).get("away", {}).get("percentage", 0.0)
                    )
                }
            )
            
            # Process points data
            points_data = stats_data.get("points", {})
            points_breakdown = PointsBreakdown(
                for_points={
                    "average": PointsBreakdownDetail(
                        home=points_data.get("for", {}).get("average", {}).get("home", 0.0),
                        away=points_data.get("for", {}).get("average", {}).get("away", 0.0),
                        all=points_data.get("for", {}).get("average", {}).get("all", 0.0)
                    ),
                    "total": PointsBreakdownDetail(
                        home=points_data.get("for", {}).get("total", {}).get("home", 0.0),
                        away=points_data.get("for", {}).get("total", {}).get("away", 0.0),
                        all=points_data.get("for", {}).get("total", {}).get("all", 0.0)
                    )
                },
                against={
                    "average": PointsBreakdownDetail(
                        home=points_data.get("against", {}).get("average", {}).get("home", 0.0),
                        away=points_data.get("against", {}).get("average", {}).get("away", 0.0),
                        all=points_data.get("against", {}).get("average", {}).get("all", 0.0)
                    ),
                    "total": PointsBreakdownDetail(
                        home=points_data.get("against", {}).get("total", {}).get("home", 0.0),
                        away=points_data.get("against", {}).get("total", {}).get("away", 0.0),
                        all=points_data.get("against", {}).get("total", {}).get("all", 0.0)
                    )
                }
            )
            
            # Get form (recent results)
            form = stats_data.get("form", "")
            
            # Calculate additional advanced statistics
            win_percentage = games_data.get("wins", {}).get("all", {}).get("percentage", 0.0)
            
            # Pace calculation (possessions per 48 minutes)
            # For simplicity, using average possessions as an approximation
            pace = 100.0  # Default value
            # In a real system, you'd use the formula:
            # pace = 48 * ((poss_team + poss_opponent) / (2 * minutes_played))
            
            # Offensive Rating (points per 100 possessions)
            offensive_rating = 0.0
            if games_breakdown.played.all > 0:
                for_points = points_breakdown.for_points["total"].all
                offensive_rating = (for_points / games_breakdown.played.all) * (100.0 / pace)
            
            # Defensive Rating (points allowed per 100 possessions)
            defensive_rating = 0.0
            if games_breakdown.played.all > 0:
                against_points = points_breakdown.against["total"].all
                defensive_rating = (against_points / games_breakdown.played.all) * (100.0 / pace)
            
            # Net Rating (offensive rating - defensive rating)
            net_rating = offensive_rating - defensive_rating
            
            # True Shooting Percentage
            # We don't have detailed shooting stats, so using a placeholder
            true_shooting_percentage = 0.55  # Default value
            
            # Create TeamStatisticsModel
            return TeamStatisticsModel(
                team_id=team_id,
                external_team_id=team.external_id,
                league_id=league_id,
                season_id=season_id,
                games=games_breakdown,
                points=points_breakdown,
                form=form,
                win_percentage=win_percentage,
                offensive_rating=offensive_rating,
                defensive_rating=defensive_rating,
                net_rating=net_rating,
                pace=pace,
                true_shooting_percentage=true_shooting_percentage
            )
            
        except Exception as e:
            logger.error(f"Error processing team statistics for team {team_id}: {str(e)}", exc_info=True)
            return None
    
    async def process_player_statistics(
        self, 
        stats_data: Dict[str, Any],
        player_id: str,
        game_id: str
    ) -> Optional[PlayerStatisticsModel]:
        """
        Process raw player statistics data into a player statistics model.
        
        Args:
            stats_data: Raw player statistics data from API
            player_id: MongoDB player ID
            game_id: MongoDB game ID
            
        Returns:
            Player statistics model or None if processing fails
        """
        try:
            # Get player and game for external IDs
            player = await self.player_repository.find_by_id(player_id)
            game = await self.game_repository.find_by_id(game_id)
            
            if not player or not game:
                logger.error(f"Player {player_id} or game {game_id} not found")
                return None
            
            # Check if we have valid statistics data
            if not stats_data or not isinstance(stats_data, dict):
                logger.error(f"Invalid statistics data for player {player_id}")
                return None
            
            # Get team ID
            team_id = player.team_id
            
            # Find the relevant statistics record
            statistics = stats_data.get("statistics", [])
            if not statistics or not isinstance(statistics, list):
                logger.error(f"No statistics found for player {player_id}")
                return None
            
            # Use the first statistics record (assuming there's just one)
            stat = statistics[0] if statistics else {}
            
            # Define player type (starters or bench)
            # This might not be explicitly provided, so we'll use a default
            player_type = "starters"  # Default to starters
            
            # Process minutes played
            minutes = stat.get("minutes", "0:00")
            
            # Process field goals
            field_goals = FieldGoalStats(
                made=stat.get("fgm", 0),
                attempts=stat.get("fga", 0),
                percentage=stat.get("fgp", 0.0)
            )
            
            # Process three-point shots
            three_points = FieldGoalStats(
                made=stat.get("tpm", 0),
                attempts=stat.get("tpa", 0),
                percentage=stat.get("tpp", 0.0)
            )
            
            # Process free throws
            free_throws = FieldGoalStats(
                made=stat.get("ftm", 0),
                attempts=stat.get("fta", 0),
                percentage=stat.get("ftp", 0.0)
            )
            
            # Get other basic statistics
            rebounds = stat.get("totReb", 0)
            assists = stat.get("assists", 0)
            points = stat.get("points", 0)
            
            # Create PlayerStatisticsModel
            return PlayerStatisticsModel(
                player_id=player_id,
                external_player_id=player.external_id,
                team_id=team_id,
                game_id=game_id,
                external_game_id=game.external_id,
                type=player_type,
                minutes=minutes,
                field_goals=field_goals,
                three_points=three_points,
                free_throws=free_throws,
                rebounds=rebounds,
                assists=assists,
                points=points
            )
            
        except Exception as e:
            logger.error(f"Error processing player statistics for player {player_id}, game {game_id}: {str(e)}", exc_info=True)
            return None
    
    async def process_game_statistics(
        self, 
        stats_data: Dict[str, Any],
        game_id: str
    ) -> List[PlayerStatisticsModel]:
        """
        Process raw game statistics data into player statistics models.
        
        Args:
            stats_data: Raw game statistics data from API
            game_id: MongoDB game ID
            
        Returns:
            List of player statistics models
        """
        player_stats_models = []
        
        try:
            # Get game
            game = await self.game_repository.find_by_id(game_id)
            if not game:
                logger.error(f"Game not found: {game_id}")
                return []
            
            # Check if we have valid statistics data
            if not stats_data or not isinstance(stats_data, dict) or "players" not in stats_data:
                logger.error(f"Invalid statistics data for game {game_id}")
                return []
            
            # Process each team's players
            for team_data in stats_data["players"]:
                team_external_id = team_data.get("team", {}).get("id")
                team = await self.team_repository.find_by_external_id(team_external_id)
                
                if not team:
                    logger.warning(f"Team not found: {team_external_id}")
                    continue
                
                team_id = str(team.id)
                
                # Process each player
                for player_data in team_data.get("players", []):
                    player_external_id = player_data.get("player", {}).get("id")
                    player = await self.player_repository.find_by_external_id(player_external_id)
                    
                    if not player:
                        logger.warning(f"Player not found: {player_external_id}")
                        continue
                    
                    player_id = str(player.id)
                    
                    # Build player statistics data
                    player_stats_data = {
                        "player": player_data.get("player", {}),
                        "team": team_data.get("team", {}),
                        "game": {"id": game.external_id},
                        "statistics": player_data.get("statistics", [])
                    }
                    
                    # Process the player statistics
                    player_stats_model = await self.process_player_statistics(
                        stats_data=player_stats_data,
                        player_id=player_id,
                        game_id=game_id
                    )
                    
                    if player_stats_model:
                        player_stats_models.append(player_stats_model)
                    
        except Exception as e:
            logger.error(f"Error processing game statistics for game {game_id}: {str(e)}", exc_info=True)
        
        logger.info(f"Processed {len(player_stats_models)} player statistics for game {game_id}")
        return player_stats_models
    
    async def save_team_statistics(self, stats: TeamStatisticsModel) -> Optional[str]:
        """
        Save team statistics to the database.
        
        Args:
            stats: Team statistics model to save
            
        Returns:
            ID of saved statistics or None if save fails
        """
        from server.data.repositories.stats_repository import TeamStatsRepository
        
        if not stats:
            return None
            
        team_stats_repository = TeamStatsRepository()
        
        try:
            # Check if statistics already exist for this team and season
            existing_stats = await team_stats_repository.find_by_team_season(
                stats.team_id,
                stats.season_id
            )
            
            if existing_stats:
                # Update existing statistics
                stats_dict = get_team_stats_dict(stats)
                updated = await team_stats_repository.update(
                    id=str(existing_stats.id),
                    update={"$set": stats_dict}
                )
                
                if updated:
                    return str(existing_stats.id)
                return None
            else:
                # Insert new statistics
                return await team_stats_repository.create(stats)
                
        except Exception as e:
            logger.error(f"Error saving team statistics for team {stats.team_id}: {str(e)}")
            return None
    
    async def save_player_statistics(self, stats: List[PlayerStatisticsModel]) -> int:
        """
        Save player statistics to the database.
        
        Args:
            stats: List of player statistics models to save
            
        Returns:
            Number of statistics saved
        """
        from server.data.repositories.stats_repository import PlayerStatsRepository
        
        if not stats:
            return 0
            
        player_stats_repository = PlayerStatsRepository()
        saved_count = 0
        
        for player_stats in stats:
            try:
                # Check if statistics already exist for this player and game
                existing_stats = await player_stats_repository.find_by_player_game(
                    player_stats.player_id,
                    player_stats.game_id
                )
                
                if existing_stats:
                    # Update existing statistics
                    stats_dict = get_player_stats_dict(player_stats)
                    updated = await player_stats_repository.update(
                        id=str(existing_stats.id),
                        update={"$set": stats_dict}
                    )
                    
                    if updated:
                        saved_count += 1
                else:
                    # Insert new statistics
                    stats_id = await player_stats_repository.create(player_stats)
                    if stats_id:
                        saved_count += 1
                        
            except Exception as e:
                logger.error(f"Error saving player statistics for player {player_stats.player_id}, game {player_stats.game_id}: {str(e)}")
                continue
        
        return saved_count
    
    async def collect_and_save_team_statistics(
        self,
        team_id: str,
        league_id: str,
        season_id: str
    ) -> Dict[str, Any]:
        """
        Collect and save team statistics.
        
        Args:
            team_id: MongoDB team ID
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            Summary of collection results
        """
        # Collect team statistics
        stats_data = await self.collect_team_statistics(
            team_id=team_id,
            league_id=league_id,
            season_id=season_id
        )
        
        if not stats_data:
            return {"stats_found": False, "stats_saved": False}
        
        # Process team statistics
        stats_model = await self.process_team_statistics(
            stats_data=stats_data,
            team_id=team_id,
            league_id=league_id,
            season_id=season_id
        )
        
        if not stats_model:
            return {"stats_found": True, "stats_processed": False, "stats_saved": False}
        
        # Save team statistics
        stats_id = await self.save_team_statistics(stats_model)
        
        return {
            "stats_found": True,
            "stats_processed": True,
            "stats_saved": bool(stats_id),
            "stats_id": stats_id
        }
    
    async def collect_and_save_game_statistics(
        self,
        game_id: str
    ) -> Dict[str, Any]:
        """
        Collect and save statistics for all players in a game.
        
        Args:
            game_id: MongoDB game ID
            
        Returns:
            Summary of collection results
        """
        # Collect game statistics
        stats_data = await self.collect_game_statistics(game_id)
        
        if not stats_data:
            return {"stats_found": False, "stats_saved": 0}
        
        # Process game statistics
        player_stats_models = await self.process_game_statistics(
            stats_data=stats_data,
            game_id=game_id
        )
        
        if not player_stats_models:
            return {"stats_found": True, "stats_processed": 0, "stats_saved": 0}
        
        # Save player statistics
        saved_count = await self.save_player_statistics(player_stats_models)
        
        return {
            "stats_found": True,
            "stats_processed": len(player_stats_models),
            "stats_saved": saved_count
        }
    
    async def collect_and_save_all_team_statistics(
        self,
        league_id: str,
        season_id: str
    ) -> Dict[str, Any]:
        """
        Collect and save statistics for all teams in a league.
        
        Args:
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            Summary of collection results
        """
        # Get all teams for this league
        teams = await self.team_repository.find_teams_by_league(league_id)
        
        if not teams:
            return {"teams_found": 0, "stats_saved": 0}
        
        teams_found = len(teams)
        stats_saved = 0
        
        # Process each team
        for team in teams:
            result = await self.collect_and_save_team_statistics(
                team_id=str(team.id),
                league_id=league_id,
                season_id=season_id
            )
            
            if result.get("stats_saved"):
                stats_saved += 1
        
        return {
            "teams_found": teams_found,
            "stats_saved": stats_saved
        }