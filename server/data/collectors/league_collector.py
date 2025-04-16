# server/data/collectors/league_collector.py
"""
Collector for basketball league data.
"""
from typing import List, Dict, Any, Optional

from bson import ObjectId

from app.core.logger import logger
from app.db.models.league import (
    LeagueModel, LeagueSeason, get_league_dict
)
from server.data.repositories.league_repository import LeagueRepository
from server.data.repositories.country_repository import CountryRepository
from server.data.repositories.season_repository import SeasonRepository
from server.data.providers.basketball_api import BasketballAPI
from server.data.processors.cleaner import DataCleaner


class LeagueCollector:
    """
    Collects and processes basketball league data.
    """
    
    def __init__(
        self, 
        api: Optional[BasketballAPI] = None,
        league_repository: Optional[LeagueRepository] = None,
        country_repository: Optional[CountryRepository] = None,
        season_repository: Optional[SeasonRepository] = None
    ):
        """
        Initialize the league collector.
        
        Args:
            api: Optional BasketballAPI instance
            league_repository: Optional LeagueRepository instance
            country_repository: Optional CountryRepository instance
            season_repository: Optional SeasonRepository instance
        """
        self.api = api or BasketballAPI()
        self.league_repository = league_repository or LeagueRepository()
        self.country_repository = country_repository or CountryRepository()
        self.season_repository = season_repository or SeasonRepository()
        self.cleaner = DataCleaner()
    
    async def collect_leagues(
        self, 
        country_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect league data from the API.
        
        Args:
            country_id: Optional MongoDB country ID filter
            
        Returns:
            List of raw league data
        """
        # Get country external ID if provided
        api_country_id = None
        if country_id:
            country = await self.country_repository.find_by_id(country_id)
            if country:
                api_country_id = country.external_id
        
        # Fetch leagues from API
        logger.info(f"Collecting leagues from API{' for country ' + str(api_country_id) if api_country_id else ''}")
        leagues = await self.api.get_leagues(country_id=api_country_id)
        
        logger.info(f"Found {len(leagues)} leagues from API")
        return leagues
    
    async def collect_league_seasons(self, league_id: int) -> List[str]:
        """
        Collect seasons for a league from the API.
        
        Args:
            league_id: External league ID
            
        Returns:
            List of season data
        """
        logger.info(f"Collecting seasons for league {league_id}")
        return await self.api.get_league_seasons(league_id=league_id)
    
    async def process_leagues(
        self, 
        api_leagues: List[Dict[str, Any]]
    ) -> List[LeagueModel]:
        """
        Process raw league data into league models.
        
        Args:
            api_leagues: List of raw league data from API
            
        Returns:
            List of league models
        """
        league_models = []
        
        for api_league in api_leagues:
            try:
                # Get country ID from our database
                country = await self.country_repository.find_by_external_id(api_league["country"]["id"])
                
                if not country:
                    logger.warning(f"Country not found for league {api_league['id']}")
                    continue
                
                country_id = str(country.id)
                
                # Get seasons for this league
                api_seasons = await self.collect_league_seasons(api_league["id"])
                league_seasons = []
                
                for api_season in api_seasons:
                    # Look up or create season in our database
                    season = await self.season_repository.find_by_external_id(api_season)
                    
                    if not season:
                        # We should have seasons in our database
                        # This is a fallback but it's better to run the season collector first
                        logger.warning(f"Season {api_season} not found in database")
                        continue
                    
                    # Get more detailed season info
                    league_info = await self.api.get_league_information(
                        league_id=api_league["id"],
                        season=api_season
                    )
                    
                    # Skip if no detailed info available
                    if not league_info:
                        continue
                    
                    # Add season to league
                    season_coverage = {}
                    if "seasons" in league_info and league_info["seasons"]:
                        season_coverage = league_info["seasons"][0].get("coverage", {})
                    
                    league_seasons.append(LeagueSeason(
                        season_id=str(season.id),
                        external_season=api_season,
                        start_date=season.start_date,
                        end_date=season.end_date,
                        coverage=season_coverage
                    ))
                
                # Clean league name
                league_name = self.cleaner.clean_text(api_league["name"])
                
                # Create league model
                league_model = LeagueModel(
                    external_id=api_league["id"],
                    name=league_name,
                    type=api_league["type"],
                    logo_url=api_league.get("logo"),
                    country_id=country_id,
                    seasons=league_seasons
                )
                
                league_models.append(league_model)
                
            except Exception as e:
                logger.error(f"Error processing league {api_league.get('id', 'unknown')}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Processed {len(league_models)} leagues")
        return league_models
    
    async def save_leagues(self, leagues: List[LeagueModel]) -> int:
        """
        Save leagues to the database.
        
        Args:
            leagues: List of league models to save
            
        Returns:
            Number of leagues saved or updated
        """
        if not leagues:
            return 0
            
        saved_count = 0
        
        for league in leagues:
            try:
                # Check if league already exists
                existing_league = await self.league_repository.find_by_external_id(league.external_id)
                
                if existing_league:
                    # Update existing league
                    league_dict = get_league_dict(league)
                    
                    # Keep existing seasons and add new ones
                    existing_seasons = existing_league.seasons
                    new_seasons = league.seasons
                    
                    # Create map of existing seasons by ID
                    existing_seasons_map = {
                        season.season_id: season 
                        for season in existing_seasons
                    }
                    
                    # Update or add new seasons
                    for new_season in new_seasons:
                        existing_seasons_map[new_season.season_id] = new_season
                    
                    # Convert back to list
                    league_dict["seasons"] = [season.dict() for season in existing_seasons_map.values()]
                    
                    # Update
                    updated = await self.league_repository.update(
                        id=str(existing_league.id),
                        update={"$set": league_dict}
                    )
                    
                    if updated:
                        saved_count += 1
                else:
                    # Insert new league
                    league_id = await self.league_repository.create(league)
                    if league_id:
                        saved_count += 1
                        
            except Exception as e:
                logger.error(f"Error saving league {league.external_id}: {str(e)}", exc_info=True)
                continue
        
        return saved_count
    
    async def collect_and_save_leagues(
        self, 
        country_id: Optional[str] = None,
        focus_on_basketball: bool = True
    ) -> Dict[str, Any]:
        """
        Collect and save leagues.
        
        Args:
            country_id: Optional MongoDB country ID filter
            focus_on_basketball: If True, filter to basketball leagues only
            
        Returns:
            Summary of collection results
        """
        # Collect leagues
        api_leagues = await self.collect_leagues(country_id)
        
        if not api_leagues:
            return {"leagues_found": 0, "leagues_saved": 0}
        
        # Filter to basketball leagues if requested
        filtered_leagues = api_leagues
        if focus_on_basketball:
            filtered_leagues = [
                league for league in api_leagues 
                if "basketball" in league.get("name", "").lower() or
                   "basketball" in league.get("type", "").lower() or
                   "nba" in league.get("name", "").lower()
            ]
            logger.info(f"Filtered to {len(filtered_leagues)} basketball leagues from {len(api_leagues)} total leagues")
        
        # Process leagues
        league_models = await self.process_leagues(filtered_leagues)
        
        # Save leagues
        saved_count = await self.save_leagues(league_models)
        
        return {
            "leagues_found": len(api_leagues),
            "leagues_filtered": len(filtered_leagues) if focus_on_basketball else len(api_leagues),
            "leagues_processed": len(league_models),
            "leagues_saved": saved_count
        }