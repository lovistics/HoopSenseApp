# server/data/collectors/game_collector.py
"""
Collector for basketball game data.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from bson import ObjectId

from app.core.logger import logger
from app.db.models.game import (
    GameModel, GameTeam, GameStatus, GameScore, get_game_dict
)
from server.data.repositories.game_repository import GameRepository
from server.data.repositories.team_repository import TeamRepository
from server.data.providers.basketball_api import BasketballAPI
from server.data.processors.transformer import DataTransformer
from server.data.processors.cleaner import DataCleaner


class GameCollector:
    """
    Collects and processes basketball game data.
    """
    
    def __init__(
        self, 
        api: Optional[BasketballAPI] = None,
        game_repository: Optional[GameRepository] = None,
        team_repository: Optional[TeamRepository] = None
    ):
        """
        Initialize the game collector.
        
        Args:
            api: Optional BasketballAPI instance
            game_repository: Optional GameRepository instance
            team_repository: Optional TeamRepository instance
        """
        self.api = api or BasketballAPI()
        self.game_repository = game_repository or GameRepository()
        self.team_repository = team_repository or TeamRepository()
        self.transformer = DataTransformer()
        self.cleaner = DataCleaner()
    
    async def collect_games_by_date(
        self,
        date: datetime,
        league_id: Optional[str] = None,
        season_id: Optional[str] = None
    ) -> List[GameModel]:
        """
        Collect games for a specific date.
        
        Args:
            date: The date to collect games for
            league_id: Optional MongoDB league ID filter
            season_id: Optional MongoDB season ID filter
            
        Returns:
            List of collected game models
        """
        # Format date for API
        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"Collecting games for date: {date_str}")
        
        # Get league and season info
        api_params = await self._get_league_season_params(league_id, season_id)
        if not api_params:
            logger.error("Failed to determine league and season for game collection")
            return []
        
        # Fetch games from API
        api_games = await self.api.get_games(
            league_id=api_params["api_league_id"],
            season=api_params["api_season"],
            date=date_str
        )
        
        if not api_games:
            logger.info(f"No games found for date {date_str}")
            return []
        
        # Transform API data to our models
        game_models = []
        
        for api_game in api_games:
            try:
                # Transform API game to our model format
                game_data = self.transformer.transform_api_game_to_model(api_game)
                
                # Get team IDs from our database
                home_team_external_id = game_data["home_team"]["external_id"]
                away_team_external_id = game_data["away_team"]["external_id"]
                
                home_team_db = await self.team_repository.find_by_external_id(home_team_external_id)
                away_team_db = await self.team_repository.find_by_external_id(away_team_external_id)
                
                if not home_team_db or not away_team_db:
                    logger.warning(f"Teams not found for game {api_game['id']}")
                    continue
                
                # Create the game status
                game_status = GameStatus(
                    long=game_data["status"]["long"],
                    short=game_data["status"]["short"],
                    timer=game_data["status"]["timer"]
                )
                
                # Create score objects if available
                home_score = None
                if game_data["home_team"]["scores"]:
                    home_score = GameScore(**game_data["home_team"]["scores"])
                
                away_score = None
                if game_data["away_team"]["scores"]:
                    away_score = GameScore(**game_data["away_team"]["scores"])
                
                # Create team objects
                home_team = GameTeam(
                    team_id=str(home_team_db.id),
                    external_id=home_team_external_id,
                    name=home_team_db.name,
                    scores=home_score
                )
                
                away_team = GameTeam(
                    team_id=str(away_team_db.id),
                    external_id=away_team_external_id,
                    name=away_team_db.name,
                    scores=away_score
                )
                
                # Create the game model
                game_model = GameModel(
                    external_id=game_data["external_id"],
                    league_id=api_params["league_id"],
                    season_id=api_params["season_id"],
                    date=game_data["date"],
                    timestamp=game_data["timestamp"],
                    timezone=game_data["timezone"],
                    stage=game_data["stage"],
                    status=game_status,
                    home_team=home_team,
                    away_team=away_team,
                    venue=game_data["venue"],
                    is_analyzed=False,
                    prediction=None
                )
                
                game_models.append(game_model)
                
            except Exception as e:
                logger.error(f"Error processing game {api_game.get('id', 'unknown')}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Processed {len(game_models)} games for date {date_str}")
        return game_models
    
    async def save_games(self, games: List[GameModel]) -> int:
        """
        Save games to the database.
        
        Args:
            games: List of game models to save
            
        Returns:
            Number of games saved or updated
        """
        if not games:
            return 0
            
        saved_count = 0
        
        for game in games:
            try:
                # Check if game already exists
                existing_game = await self.game_repository.find_by_external_id(game.external_id)
                
                if existing_game:
                    # Update existing game
                    game_dict = get_game_dict(game)
                    
                    # Don't overwrite existing prediction if present
                    if existing_game.prediction:
                        game_dict["prediction"] = existing_game.prediction.dict()
                        game_dict["is_analyzed"] = existing_game.is_analyzed
                    
                    # Update the game
                    updated = await self.game_repository.update(
                        id=str(existing_game.id),
                        update={"$set": game_dict}
                    )
                    
                    if updated:
                        saved_count += 1
                else:
                    # Insert new game
                    game_id = await self.game_repository.create(game)
                    if game_id:
                        saved_count += 1
                        
            except Exception as e:
                logger.error(f"Error saving game {game.external_id}: {str(e)}", exc_info=True)
                continue
        
        return saved_count
    
    async def collect_and_save_games_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        league_id: Optional[str] = None,
        season_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect and save games for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            league_id: Optional MongoDB league ID filter
            season_id: Optional MongoDB season ID filter
            
        Returns:
            Summary of collection results
        """
        total_days = (end_date - start_date).days + 1
        total_games_found = 0
        total_games_saved = 0
        
        logger.info(f"Collecting games from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        for day_offset in range(total_days):
            current_date = start_date + timedelta(days=day_offset)
            
            # Collect games for current date
            games = await self.collect_games_by_date(
                date=current_date,
                league_id=league_id,
                season_id=season_id
            )
            
            total_games_found += len(games)
            
            if games:
                saved_count = await self.save_games(games)
                total_games_saved += saved_count
                logger.info(f"Saved {saved_count} games for date {current_date.strftime('%Y-%m-%d')}")
        
        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "days_processed": total_days,
            "games_found": total_games_found,
            "games_saved": total_games_saved
        }
    
    async def update_live_games(self) -> Dict[str, Any]:
        """
        Update scores and status for live games.
        
        Returns:
            Summary of update results
        """
        # Get all currently live games from our database
        live_games = await self.game_repository.find_live_games()
        
        if not live_games:
            logger.info("No live games found for update")
            return {"updated": 0, "status": "No live games found"}
        
        logger.info(f"Found {len(live_games)} live games to update")
        
        # Group by league and season to minimize API calls
        league_season_games = {}
        for game in live_games:
            key = f"{game.league_id}:{game.season_id}"
            if key not in league_season_games:
                league_season_games[key] = []
            league_season_games[key].append(game)
        
        total_updated = 0
        failed_updates = 0
        
        # Process each league/season group
        for key, games in league_season_games.items():
            logger.debug(f"Processing {len(games)} games for league/season {key}")
            
            # Get API parameters for this league/season
            try:
                league_id, season_id = key.split(":")
                api_params = await self._get_league_season_params(league_id, season_id)
                
                if not api_params:
                    logger.warning(f"Couldn't get league/season parameters for {key}")
                    continue
                
                # Fetch all live games for this league/season
                api_games = await self.api.get_games(
                    league_id=api_params["api_league_id"],
                    season=api_params["api_season"],
                    status="LIVE"
                )
                
                if not api_games:
                    logger.info(f"No live games found for league {api_params['api_league_id']}, season {api_params['api_season']}")
                    continue
                
                # Map API games by ID for easy lookup
                api_games_map = {str(game["id"]): game for game in api_games}
                
                # Update each game
                for game in games:
                    await self._update_live_game(game, api_games_map)
                    total_updated += 1
            
            except Exception as e:
                logger.error(f"Error updating games for league/season {key}: {str(e)}", exc_info=True)
                failed_updates += len(games)
        
        return {
            "total_live_games": len(live_games),
            "updated": total_updated,
            "failed": failed_updates
        }
    
    async def _update_live_game(
        self, 
        game: GameModel, 
        api_games_map: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Update a single live game from API data.
        
        Args:
            game: The game to update
            api_games_map: Map of API games by ID
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            external_id = str(game.external_id)
            
            # If the game is not in the live games map, fetch it directly
            if external_id not in api_games_map:
                api_game = await self.api.get_game(game.external_id)
                if not api_game:
                    logger.warning(f"Game {external_id} not found in API")
                    return False
            else:
                api_game = api_games_map[external_id]
            
            # Update status
            updated_data = {
                "status.long": api_game["status"]["long"],
                "status.short": api_game["status"]["short"],
                "status.timer": api_game["status"].get("timer")
            }
            
            # Update scores if available
            if "scores" in api_game and api_game["scores"]:
                if "home" in api_game["scores"]:
                    updated_data.update({
                        "home_team.scores.quarter_1": api_game["scores"]["home"].get("quarter_1"),
                        "home_team.scores.quarter_2": api_game["scores"]["home"].get("quarter_2"),
                        "home_team.scores.quarter_3": api_game["scores"]["home"].get("quarter_3"),
                        "home_team.scores.quarter_4": api_game["scores"]["home"].get("quarter_4"),
                        "home_team.scores.over_time": api_game["scores"]["home"].get("over_time"),
                        "home_team.scores.total": api_game["scores"]["home"].get("total")
                    })
                
                if "away" in api_game["scores"]:
                    updated_data.update({
                        "away_team.scores.quarter_1": api_game["scores"]["away"].get("quarter_1"),
                        "away_team.scores.quarter_2": api_game["scores"]["away"].get("quarter_2"),
                        "away_team.scores.quarter_3": api_game["scores"]["away"].get("quarter_3"),
                        "away_team.scores.quarter_4": api_game["scores"]["away"].get("quarter_4"),
                        "away_team.scores.over_time": api_game["scores"]["away"].get("over_time"),
                        "away_team.scores.total": api_game["scores"]["away"].get("total")
                    })
            
            # Update the game
            updated = await self.game_repository.update(
                id=str(game.id),
                update={"$set": updated_data}
            )
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating live game {game.external_id}: {str(e)}", exc_info=True)
            return False
    
    async def _get_league_season_params(
        self, 
        league_id: Optional[str], 
        season_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get API parameters for league and season.
        
        Args:
            league_id: Optional MongoDB league ID
            season_id: Optional MongoDB season ID
            
        Returns:
            Dictionary with API parameters or None if not found
        """
        from server.data.repositories.league_repository import LeagueRepository
        from server.data.repositories.season_repository import SeasonRepository
        
        league_repo = LeagueRepository()
        season_repo = SeasonRepository()
        
        # If both league_id and season_id are provided, use them
        if league_id and season_id:
            league = await league_repo.find_by_id(league_id)
            season = await season_repo.find_by_id(season_id)
            
            if league and season:
                # Find the season in the league
                for league_season in league.seasons:
                    if league_season.season_id == str(season.id):
                        return {
                            "api_league_id": league.external_id,
                            "api_season": season.external_id,
                            "league_id": str(league.id),
                            "season_id": str(season.id)
                        }
        
        # Otherwise, find NBA league and current season
        nba_league = await league_repo.find_one({"name": "NBA"})
        
        if not nba_league:
            return None
        
        # Get current season
        current_date = datetime.utcnow()
        current_season = None
        
        # First, look through the league's seasons
        for league_season in nba_league.seasons:
            season = await season_repo.find_by_id(league_season.season_id)
            if season and season.start_date <= current_date <= season.end_date:
                current_season = season
                break
        
        # If no current season found in league, find one in the seasons collection
        if not current_season:
            seasons = await season_repo.find(
                filter={"status": "active"},
                sort=[("start_date", -1)],
                limit=1
            )
            
            if seasons:
                current_season = seasons[0]
        
        # If still no season found, get the most recent one
        if not current_season:
            seasons = await season_repo.find(
                filter={},
                sort=[("start_date", -1)],
                limit=1
            )
            
            if seasons:
                current_season = seasons[0]
        
        if not current_season:
            return None
        
        return {
            "api_league_id": nba_league.external_id,
            "api_season": current_season.external_id,
            "league_id": str(nba_league.id),
            "season_id": str(current_season.id)
        }