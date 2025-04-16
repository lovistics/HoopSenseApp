"""
Collector for basketball standings data.
"""
from typing import List, Dict, Any, Optional, Tuple

from bson import ObjectId

from app.core.logger import logger
from app.db.models.standing import (
    StandingModel, StandingInDB, StandingEntry, StandingGames, 
    StandingPoints, StandingTeam, get_standing_dict
)
from server.data.repositories.league_repository import LeagueRepository
from server.data.repositories.season_repository import SeasonRepository
from server.data.repositories.team_repository import TeamRepository
from server.data.providers.basketball_api import BasketballAPI
from server.data.processors.cleaner import DataCleaner


class StandingsCollector:
    """
    Collects and processes basketball standings data.
    """
    
    def __init__(
        self, 
        api: Optional[BasketballAPI] = None,
        league_repository: Optional[LeagueRepository] = None,
        season_repository: Optional[SeasonRepository] = None,
        team_repository: Optional[TeamRepository] = None
    ):
        """
        Initialize the standings collector.
        
        Args:
            api: Optional BasketballAPI instance
            league_repository: Optional LeagueRepository instance
            season_repository: Optional SeasonRepository instance
            team_repository: Optional TeamRepository instance
        """
        self.api = api or BasketballAPI()
        self.league_repository = league_repository or LeagueRepository()
        self.season_repository = season_repository or SeasonRepository()
        self.team_repository = team_repository or TeamRepository()
        self.cleaner = DataCleaner()
    
    async def collect_standings(
        self,
        league_id: str,
        season_id: str
    ) -> List[Dict[str, Any]]:
        """
        Collect standings data for a league and season.
        
        Args:
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            List of raw standings data
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
        
        # Get external IDs for API calls
        api_league_id = league.external_id
        api_season = league_season.external_season
        
        logger.info(f"Collecting standings for league {api_league_id}, season {api_season}")
        
        # Fetch standings from API
        standings_data = await self.api.get_standings(
            league_id=api_league_id,
            season=api_season
        )
        
        logger.info(f"Found {len(standings_data)} standings groups from API")
        return standings_data
    
    async def process_standings(
        self, 
        standings_data: List[Dict[str, Any]],
        league_id: str,
        season_id: str
    ) -> List[StandingModel]:
        """
        Process raw standings data into standing models.
        
        Args:
            standings_data: List of raw standings data from API
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            List of standing models
        """
        standing_models = []
        
        # Get league for external ID
        league = await self.league_repository.find_by_id(league_id)
        if not league:
            logger.error(f"League not found: {league_id}")
            return []
        
        # Get season for external ID
        season = await self.season_repository.find_by_id(season_id)
        if not season:
            logger.error(f"Season not found: {season_id}")
            return []
        
        for standings_group in standings_data:
            try:
                # Extract standings entries
                standings_entries = []
                
                # Check if the structure matches expected format
                if "league" not in standings_group:
                    logger.warning(f"Unexpected standings data format, missing 'league' key")
                    continue
                
                league_data = standings_group["league"]
                if "standings" not in league_data:
                    logger.warning(f"Unexpected league data format, missing 'standings' key")
                    continue
                
                group_data = {"name": "Regular Season"}
                if "stage" in league_data:
                    group_data["name"] = league_data["stage"]
                
                # Process each team in the standings
                for team_standing in league_data["standings"]:
                    # Find the team in the database
                    team_id = await self._find_team_id(team_standing["team"]["id"])
                    
                    if not team_id:
                        logger.warning(f"Team not found: {team_standing['team']['id']}")
                        continue
                    
                    # Build Standing Entry
                    try:
                        entry = StandingEntry(
                            position=team_standing["position"],
                            team_id=team_id,
                            external_team_id=team_standing["team"]["id"],
                            team_name=team_standing["team"]["name"],
                            games=StandingGames(
                                played=team_standing["games"]["played"],
                                wins=team_standing["games"]["win"],
                                losses=team_standing["games"]["lose"]
                            ),
                            win_percentage=self._calculate_win_percentage(
                                team_standing["games"]["played"],
                                team_standing["games"]["win"].get("total", 0)
                            ),
                            points=StandingPoints(
                                for_points=team_standing["points"]["for"],
                                against=team_standing["points"]["against"]
                            ),
                            form=team_standing.get("form", ""),
                            description=team_standing.get("description", "")
                        )
                        
                        standings_entries.append(entry)
                    except Exception as e:
                        logger.error(f"Error creating standing entry for team {team_standing['team']['id']}: {str(e)}")
                        continue
                
                # Create StandingModel
                standing_model = StandingModel(
                    league_id=league_id,
                    season_id=season_id,
                    external_league_id=league.external_id,
                    external_season=season.external_id,
                    stage=league_data.get("stage", "Regular Season"),
                    group=group_data,
                    standings=standings_entries
                )
                
                standing_models.append(standing_model)
                
            except Exception as e:
                logger.error(f"Error processing standings group: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Processed {len(standing_models)} standing models")
        return standing_models
    
    async def save_standings(self, standings: List[StandingModel]) -> int:
        """
        Save standings to the database.
        
        Args:
            standings: List of standing models to save
            
        Returns:
            Number of standings saved or updated
        """
        from server.data.repositories.standing_repository import StandingRepository
        
        if not standings:
            return 0
            
        standing_repository = StandingRepository()
        saved_count = 0
        
        for standing in standings:
            try:
                # Check if standings already exist for this league/season
                existing_standing = await standing_repository.find_by_league_season(
                    standing.league_id,
                    standing.season_id
                )
                
                if existing_standing:
                    # Update existing standings
                    standing_dict = get_standing_dict(standing)
                    updated = await standing_repository.update(
                        id=str(existing_standing.id),
                        update={"$set": standing_dict}
                    )
                    
                    if updated:
                        saved_count += 1
                else:
                    # Insert new standings
                    standing_id = await standing_repository.create(standing)
                    if standing_id:
                        saved_count += 1
                        
            except Exception as e:
                logger.error(f"Error saving standings for league {standing.league_id}, season {standing.season_id}: {str(e)}")
                continue
        
        return saved_count
    
    async def collect_and_save_standings(
        self,
        league_id: str,
        season_id: str
    ) -> Dict[str, Any]:
        """
        Collect and save standings for a league and season.
        
        Args:
            league_id: MongoDB league ID
            season_id: MongoDB season ID
            
        Returns:
            Summary of collection results
        """
        # Collect standings
        standings_data = await self.collect_standings(
            league_id=league_id,
            season_id=season_id
        )
        
        if not standings_data:
            return {"standings_found": 0, "standings_saved": 0}
        
        # Process standings
        standing_models = await self.process_standings(
            standings_data=standings_data,
            league_id=league_id,
            season_id=season_id
        )
        
        # Save standings
        saved_count = await self.save_standings(standing_models)
        
        return {
            "standings_found": len(standings_data),
            "standings_processed": len(standing_models),
            "standings_saved": saved_count
        }
    
    async def _find_team_id(self, external_id: int) -> Optional[str]:
        """
        Find MongoDB team ID by external ID.
        
        Args:
            external_id: External team ID
            
        Returns:
            MongoDB team ID or None if not found
        """
        team = await self.team_repository.find_by_external_id(external_id)
        return str(team.id) if team else None
    
    @staticmethod
    def _calculate_win_percentage(games_played: int, games_won: int) -> float:
        """
        Calculate win percentage.
        
        Args:
            games_played: Total games played
            games_won: Games won
            
        Returns:
            Win percentage as float between 0 and 1
        """
        if games_played == 0:
            return 0.0
        
        return games_won / games_played