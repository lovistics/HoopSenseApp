"""
Collector for basketball season data.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.core.logger import logger
from app.db.models.season import (
    SeasonModel, get_season_dict
)
from server.data.repositories.season_repository import SeasonRepository
from server.data.providers.basketball_api import BasketballAPI
from server.data.processors.transformer import DataTransformer


class SeasonCollector:
    """
    Collects and processes basketball season data.
    """
    
    def __init__(
        self, 
        api: Optional[BasketballAPI] = None,
        season_repository: Optional[SeasonRepository] = None
    ):
        """
        Initialize the season collector.
        
        Args:
            api: Optional BasketballAPI instance
            season_repository: Optional SeasonRepository instance
        """
        self.api = api or BasketballAPI()
        self.season_repository = season_repository or SeasonRepository()
        self.transformer = DataTransformer()
    
    async def collect_seasons(self, league_id: int) -> List[str]:
        """
        Collect season data for a league.
        
        Args:
            league_id: External league ID
            
        Returns:
            List of season strings
        """
        logger.info(f"Collecting seasons for league {league_id}")
        return await self.api.get_league_seasons(league_id)
    
    async def get_season_details(
        self, 
        league_id: int, 
        season: str
    ) -> Dict[str, Any]:
        """
        Get details for a specific season.
        
        Args:
            league_id: External league ID
            season: Season string (e.g., "2023-2024")
            
        Returns:
            Season details
        """
        logger.info(f"Getting details for season {season} in league {league_id}")
        return await self.api.get_league_information(
            league_id=league_id,
            season=season
        )
    
    async def process_seasons(
        self, 
        api_seasons: List[str],
        league_id: int
    ) -> List[SeasonModel]:
        """
        Process raw season data into season models.
        
        Args:
            api_seasons: List of season strings
            league_id: External league ID for fetching details
            
        Returns:
            List of season models
        """
        season_models = []
        
        for api_season in api_seasons:
            try:
                # Get more details about this season
                season_details = await self.get_season_details(
                    league_id=league_id,
                    season=api_season
                )
                
                if not season_details or "seasons" not in season_details:
                    logger.warning(f"No details found for season {api_season}")
                    continue
                
                # Extract season info
                season_info = season_details["seasons"][0]
                
                # Parse dates
                start_date = None
                end_date = None
                
                if "start" in season_info:
                    start_date = self.transformer.parse_api_datetime(season_info["start"])
                if "end" in season_info:
                    end_date = self.transformer.parse_api_datetime(season_info["end"])
                
                # Fallback parsing from season string
                if not start_date or not end_date:
                    try:
                        # Try to parse from season string (e.g., "2023-2024")
                        years = api_season.split("-")
                        if len(years) == 2:
                            start_year = int(years[0])
                            end_year = int(years[1])
                            
                            # NBA season typically starts in October and ends in April
                            start_date = datetime(start_year, 10, 1)
                            end_date = datetime(end_year, 4, 30)
                    except Exception as e:
                        logger.error(f"Error parsing dates from season string: {str(e)}", exc_info=True)
                        # Use current date as fallback
                        current_year = datetime.utcnow().year
                        start_date = datetime(current_year, 1, 1)
                        end_date = datetime(current_year, 12, 31)
                
                # Determine status
                now = datetime.utcnow()
                status = "active"
                
                if now < start_date:
                    status = "upcoming"
                elif now > end_date:
                    status = "completed"
                
                # Create season model
                season_model = SeasonModel(
                    external_id=api_season,
                    name=api_season,
                    start_date=start_date,
                    end_date=end_date,
                    status=status
                )
                
                season_models.append(season_model)
                
            except Exception as e:
                logger.error(f"Error processing season {api_season}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Processed {len(season_models)} seasons")
        return season_models
    
    async def save_seasons(self, seasons: List[SeasonModel]) -> int:
        """
        Save seasons to the database.
        
        Args:
            seasons: List of season models to save
            
        Returns:
            Number of seasons saved or updated
        """
        if not seasons:
            return 0
            
        saved_count = 0
        
        for season in seasons:
            try:
                # Check if season already exists
                existing_season = await self.season_repository.find_by_external_id(season.external_id)
                
                if existing_season:
                    # Update existing season
                    season_dict = get_season_dict(season)
                    updated = await self.season_repository.update(
                        id=str(existing_season.id),
                        update={"$set": season_dict}
                    )
                    
                    if updated:
                        saved_count += 1
                else:
                    # Insert new season
                    season_id = await self.season_repository.create(season)
                    if season_id:
                        saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving season {season.external_id}: {str(e)}", exc_info=True)
                continue
        
        return saved_count
    
    async def collect_and_save_nba_seasons(self) -> Dict[str, Any]:
        """
        Collect and save NBA seasons.
        
        Returns:
            Summary of collection results
        """
        # NBA league ID in basketball API
        NBA_LEAGUE_ID = 12
        
        # Collect seasons
        api_seasons = await self.collect_seasons(NBA_LEAGUE_ID)
        
        if not api_seasons:
            return {"seasons_found": 0, "seasons_saved": 0}
        
        # Process seasons
        season_models = await self.process_seasons(api_seasons, NBA_LEAGUE_ID)
        
        # Save seasons
        saved_count = await self.save_seasons(season_models)
        
        return {
            "seasons_found": len(api_seasons),
            "seasons_processed": len(season_models),
            "seasons_saved": saved_count
        }