"""
Collector for basketball player data.
"""
from typing import List, Dict, Any, Optional

from bson import ObjectId

from app.core.logger import logger
from app.db.models.player import (
    PlayerModel, get_player_dict
)
from server.data.repositories.player_repository import PlayerRepository
from server.data.repositories.team_repository import TeamRepository
from server.data.repositories.season_repository import SeasonRepository
from server.data.repositories.country_repository import CountryRepository
from server.data.providers.basketball_api import BasketballAPI
from server.data.processors.cleaner import DataCleaner


class PlayerCollector:
    """
    Collects and processes basketball player data.
    """
    
    def __init__(
        self, 
        api: Optional[BasketballAPI] = None,
        player_repository: Optional[PlayerRepository] = None,
        team_repository: Optional[TeamRepository] = None,
        season_repository: Optional[SeasonRepository] = None,
        country_repository: Optional[CountryRepository] = None
    ):
        """
        Initialize the player collector.
        
        Args:
            api: Optional BasketballAPI instance
            player_repository: Optional PlayerRepository instance
            team_repository: Optional TeamRepository instance
            season_repository: Optional SeasonRepository instance
            country_repository: Optional CountryRepository instance
        """
        self.api = api or BasketballAPI()
        self.player_repository = player_repository or PlayerRepository()
        self.team_repository = team_repository or TeamRepository()
        self.season_repository = season_repository or SeasonRepository()
        self.country_repository = country_repository or CountryRepository()
        self.cleaner = DataCleaner()
    
    async def collect_players_for_team(
        self,
        team_id: str,
        season_id: str
    ) -> List[Dict[str, Any]]:
        """
        Collect player data for a team and season.
        
        Args:
            team_id: MongoDB team ID
            season_id: MongoDB season ID
            
        Returns:
            List of raw player data
        """
        # Get team from database
        team = await self.team_repository.find_by_id(team_id)
        
        if not team:
            logger.error(f"Team not found: {team_id}")
            return []
        
        # Get season from database
        season = await self.season_repository.find_by_id(season_id)
        
        if not season:
            logger.error(f"Season not found: {season_id}")
            return []
        
        # Get external IDs
        api_team_id = team.external_id
        api_season = season.external_id
        
        logger.info(f"Collecting players for team {api_team_id}, season {api_season}")
        
        # Fetch players from API
        return await self.api.get_players(
            team_id=api_team_id,
            season=api_season
        )
    
    async def process_players(
        self, 
        api_players: List[Dict[str, Any]],
        team_id: str
    ) -> List[PlayerModel]:
        """
        Process raw player data into player models.
        
        Args:
            api_players: List of raw player data from API
            team_id: MongoDB team ID
            
        Returns:
            List of player models
        """
        player_models = []
        
        for api_player in api_players:
            try:
                # Extract player data
                player_info = api_player["player"]
                
                # Get country ID if available
                country_id = None
                if "country" in player_info and player_info["country"]:
                    country_name = player_info["country"]
                    country = await self.country_repository.find_one({"name": country_name})
                    
                    if country:
                        country_id = str(country.id)
                
                # Clean height and weight values
                height = None
                weight = None
                
                if "height" in player_info and player_info["height"]:
                    height_value = player_info["height"]
                    
                    # Try to convert to numeric value (cm)
                    height = self.cleaner.extract_numeric_value(height_value)
                    
                    # If it's in feet/inches format, convert to cm
                    if height is None and "'" in height_value:
                        height = self.cleaner.convert_height_to_cm(height_value)
                
                if "weight" in player_info and player_info["weight"]:
                    weight_value = player_info["weight"]
                    
                    # Try to convert to numeric value (kg)
                    weight = self.cleaner.extract_numeric_value(weight_value)
                    
                    # If it's in pounds, convert to kg
                    if weight and "lbs" in weight_value:
                        weight = weight * 0.453592  # convert pounds to kg
                
                # Create player model
                player_model = PlayerModel(
                    external_id=player_info["id"],
                    name=player_info["name"],
                    number=player_info.get("number"),
                    position=player_info.get("position"),
                    country=player_info.get("country"),
                    country_id=country_id,
                    age=player_info.get("age"),
                    height=height,
                    weight=weight,
                    team_id=team_id
                )
                
                player_models.append(player_model)
                
            except Exception as e:
                logger.error(f"Error processing player {api_player.get('player', {}).get('id', 'unknown')}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Processed {len(player_models)} players")
        return player_models
    
    async def save_players(self, players: List[PlayerModel]) -> int:
        """
        Save players to the database.
        
        Args:
            players: List of player models to save
            
        Returns:
            Number of players saved or updated
        """
        if not players:
            return 0
            
        saved_count = 0
        
        for player in players:
            try:
                # Check if player already exists
                existing_player = await self.player_repository.find_by_external_id(player.external_id)
                
                if existing_player:
                    # Update existing player
                    player_dict = get_player_dict(player)
                    updated = await self.player_repository.update(
                        id=str(existing_player.id),
                        update={"$set": player_dict}
                    )
                    
                    if updated:
                        saved_count += 1
                else:
                    # Insert new player
                    player_id = await self.player_repository.create(player)
                    if player_id:
                        saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving player {player.external_id}: {str(e)}", exc_info=True)
                continue
        
        return saved_count
    
    async def collect_and_save_players(
        self,
        team_id: str,
        season_id: str
    ) -> Dict[str, Any]:
        """
        Collect and save players for a team and season.
        
        Args:
            team_id: MongoDB team ID
            season_id: MongoDB season ID
            
        Returns:
            Summary of collection results
        """
        # Collect players
        api_players = await self.collect_players_for_team(
            team_id=team_id,
            season_id=season_id
        )
        
        if not api_players:
            return {"players_found": 0, "players_saved": 0}
        
        # Process players
        player_models = await self.process_players(
            api_players=api_players,
            team_id=team_id
        )
        
        # Save players
        saved_count = await self.save_players(player_models)
        
        return {
            "players_found": len(api_players),
            "players_processed": len(player_models),
            "players_saved": saved_count
        }
    
    async def collect_and_save_players_for_all_teams(
        self,
        league_id: str,
        season_id: str
    ) -> Dict[str, Any]:
        """
        Collect and save players for all teams in a league and season.
        
        Args:
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            Summary of collection results
        """
        # Get all teams for this league
        teams = await self.team_repository.find({"league_id": league_id})
        
        if not teams:
            return {"teams_processed": 0, "players_found": 0, "players_saved": 0}
        
        total_players_found = 0
        total_players_saved = 0
        
        # Process each team
        for team in teams:
            team_id = str(team.id)
            
            # Collect and save players for this team
            result = await self.collect_and_save_players(
                team_id=team_id,
                season_id=season_id
            )
            
            total_players_found += result["players_found"]
            total_players_saved += result["players_saved"]
            
            logger.info(f"Processed players for team {team.name}: {result}")
        
        return {
            "teams_processed": len(teams),
            "players_found": total_players_found,
            "players_saved": total_players_saved
        }