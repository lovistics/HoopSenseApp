# server/data/collectors/team_collector.py
"""
Collector for basketball team data.
"""
from typing import List, Dict, Any, Optional

from bson import ObjectId

from app.core.logger import logger
from app.db.models.team import (
    TeamModel, get_team_dict
)
from server.data.repositories.team_repository import TeamRepository
from server.data.repositories.league_repository import LeagueRepository
from server.data.repositories.country_repository import CountryRepository
from server.data.repositories.season_repository import SeasonRepository
from server.data.providers.basketball_api import BasketballAPI
from server.data.processors.transformer import DataTransformer


class TeamCollector:
    """
    Collects and processes basketball team data.
    """
    
    def __init__(
        self, 
        api: Optional[BasketballAPI] = None,
        team_repository: Optional[TeamRepository] = None,
        league_repository: Optional[LeagueRepository] = None,
        country_repository: Optional[CountryRepository] = None,
        season_repository: Optional[SeasonRepository] = None
    ):
        """
        Initialize the team collector.
        
        Args:
            api: Optional BasketballAPI instance
            team_repository: Optional TeamRepository instance
            league_repository: Optional LeagueRepository instance
            country_repository: Optional CountryRepository instance
            season_repository: Optional SeasonRepository instance
        """
        self.api = api or BasketballAPI()
        self.team_repository = team_repository or TeamRepository()
        self.league_repository = league_repository or LeagueRepository()
        self.country_repository = country_repository or CountryRepository()
        self.season_repository = season_repository or SeasonRepository()
        self.transformer = DataTransformer()
    
    async def collect_teams(
        self,
        league_id: str,
        season_id: str
    ) -> List[Dict[str, Any]]:
        """
        Collect team data for a league and season.
        
        Args:
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            List of raw team data
        """
        # Get league and season from database
        league = await self.league_repository.find_by_id(league_id)
        
        if not league:
            logger.error(f"League not found: {league_id}")
            return []
        
        # Find the season in the league
        league_season = None
        for season in league.seasons:
            if season.season_id == season_id:
                league_season = season
                break
        
        if not league_season:
            logger.error(f"Season {season_id} not found in league {league_id}")
            return []
        
        # Get external IDs
        api_league_id = league.external_id
        api_season = league_season.external_season
        
        logger.info(f"Collecting teams for league {api_league_id}, season {api_season}")
        
        # Fetch teams from API
        return await self.api.get_teams(
            league_id=api_league_id,
            season=api_season
        )
    
    async def process_teams(
        self, 
        api_teams: List[Dict[str, Any]],
        league_id: str,
        country_id: Optional[str] = None
    ) -> List[TeamModel]:
        """
        Process raw team data into team models.
        
        Args:
            api_teams: List of raw team data from API
            league_id: MongoDB league ID
            country_id: Optional MongoDB country ID
            
        Returns:
            List of team models
        """
        team_models = []
        
        # Get country ID if not provided
        if not country_id:
            league = await self.league_repository.find_by_id(league_id)
            
            if league and hasattr(league, "country_id"):
                country_id = league.country_id
            else:
                # Try to find USA as default
                usa_country = await self.country_repository.find_one({"name": "USA"})
                if usa_country:
                    country_id = str(usa_country.id)
                else:
                    logger.error("Country ID not provided and default not found")
                    return []
        
        for api_team in api_teams:
            try:
                # Extract team data
                team_info = api_team["team"]
                
                # Generate abbreviation if not available
                abbreviation = team_info.get("code")
                if not abbreviation:
                    abbreviation = self.transformer.extract_team_abbreviation(team_info["name"])
                
                # Determine if this is a national team
                is_national = team_info.get("national", False)
                
                # Get team colors
                primary_color, secondary_color = self.transformer.get_team_colors(team_info["name"])
                
                # Extract conference and division for NBA teams
                conference = None
                division = None
                
                # For NBA teams, try to get conference and division
                if "NBA" in api_team.get("league", {}).get("name", ""):
                    # Try to determine conference and division
                    name = team_info["name"].lower()
                    
                    # Determine conference
                    eastern_teams = ["celtics", "nets", "knicks", "76ers", "raptors", 
                                     "bulls", "cavaliers", "pistons", "pacers", "bucks",
                                     "hawks", "hornets", "heat", "magic", "wizards"]
                    
                    western_teams = ["mavericks", "rockets", "grizzlies", "pelicans", "spurs",
                                      "nuggets", "timberwolves", "thunder", "trail blazers", "jazz",
                                      "warriors", "clippers", "lakers", "suns", "kings"]
                    
                    for eastern in eastern_teams:
                        if eastern in name:
                            conference = "Eastern"
                            break
                    
                    if not conference:
                        for western in western_teams:
                            if western in name:
                                conference = "Western"
                                break
                    
                    # Determine division
                    if conference == "Eastern":
                        if any(team in name for team in ["celtics", "nets", "knicks", "76ers", "raptors"]):
                            division = "Atlantic"
                        elif any(team in name for team in ["bulls", "cavaliers", "pistons", "pacers", "bucks"]):
                            division = "Central"
                        else:
                            division = "Southeast"
                    elif conference == "Western":
                        if any(team in name for team in ["nuggets", "timberwolves", "thunder", "trail blazers", "jazz"]):
                            division = "Northwest"
                        elif any(team in name for team in ["warriors", "clippers", "lakers", "suns", "kings"]):
                            division = "Pacific"
                        else:
                            division = "Southwest"
                
                # Create team model
                team_model = TeamModel(
                    external_id=team_info["id"],
                    name=team_info["name"],
                    abbreviation=abbreviation,
                    is_national=is_national,
                    logo_url=team_info.get("logo"),
                    primary_color=primary_color,
                    secondary_color=secondary_color,
                    country_id=country_id,
                    league_id=league_id,
                    conference=conference,
                    division=division
                )
                
                team_models.append(team_model)
                
            except Exception as e:
                logger.error(f"Error processing team {api_team.get('team', {}).get('id', 'unknown')}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Processed {len(team_models)} teams")
        return team_models
    
    async def save_teams(self, teams: List[TeamModel]) -> int:
        """
        Save teams to the database.
        
        Args:
            teams: List of team models to save
            
        Returns:
            Number of teams saved or updated
        """
        if not teams:
            return 0
            
        saved_count = 0
        
        for team in teams:
            try:
                # Check if team already exists
                existing_team = await self.team_repository.find_by_external_id(team.external_id)
                
                if existing_team:
                    # Update existing team
                    team_dict = get_team_dict(team)
                    
                    # Don't overwrite colors if they already exist
                    if existing_team.primary_color and not team.primary_color:
                        team_dict.pop("primary_color", None)
                    if existing_team.secondary_color and not team.secondary_color:
                        team_dict.pop("secondary_color", None)
                    
                    updated = await self.team_repository.update(
                        id=str(existing_team.id),
                        update={"$set": team_dict}
                    )
                    
                    if updated:
                        saved_count += 1
                else:
                    # Insert new team
                    team_id = await self.team_repository.create(team)
                    if team_id:
                        saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving team {team.external_id}: {str(e)}", exc_info=True)
                continue
        
        return saved_count
    
    async def collect_and_save_teams(
        self,
        league_id: str,
        season_id: str
    ) -> Dict[str, Any]:
        """
        Collect and save teams for a league and season.
        
        Args:
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            Summary of collection results
        """
        # Collect teams
        api_teams = await self.collect_teams(
            league_id=league_id,
            season_id=season_id
        )
        
        if not api_teams:
            return {"teams_found": 0, "teams_saved": 0}
        
        # Process teams
        team_models = await self.process_teams(
            api_teams=api_teams,
            league_id=league_id
        )
        
        # Save teams
        saved_count = await self.save_teams(team_models)
        
        return {
            "teams_found": len(api_teams),
            "teams_processed": len(team_models),
            "teams_saved": saved_count
        }