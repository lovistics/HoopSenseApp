"""
Repository for team data operations.
"""
from typing import Dict, Any, List, Optional

from bson import ObjectId

from app.core.logger import logger
from app.db.models.team import TeamInDB, COLLECTION_NAME
from server.data.repositories.base_repository import BaseRepository


class TeamRepository(BaseRepository[TeamInDB]):
    """
    Repository for team-related database operations.
    """
    
    def __init__(self):
        """Initialize the team repository."""
        super().__init__(COLLECTION_NAME, TeamInDB)
    
    async def find_by_external_id(self, external_id: int) -> Optional[TeamInDB]:
        """
        Find a team by external ID.
        
        Args:
            external_id: External API team ID
            
        Returns:
            Team or None if not found
        """
        return await self.find_one({"external_id": external_id})
    
    async def find_by_name(self, name: str, exact: bool = False) -> Optional[TeamInDB]:
        """
        Find a team by name.
        
        Args:
            name: Team name to search for
            exact: If True, require exact match (case insensitive)
            
        Returns:
            Team or None if not found
        """
        if exact:
            return await self.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})
        else:
            return await self.find_one({"name": {"$regex": name, "$options": "i"}})
    
    async def find_by_abbreviation(self, abbreviation: str) -> Optional[TeamInDB]:
        """
        Find a team by abbreviation.
        
        Args:
            abbreviation: Team abbreviation (e.g., "GSW")
            
        Returns:
            Team or None if not found
        """
        return await self.find_one({"abbreviation": abbreviation})
    
    async def find_teams_by_league(
        self,
        league_id: str,
        conference: Optional[str] = None,
        division: Optional[str] = None
    ) -> List[TeamInDB]:
        """
        Find teams in a league.
        
        Args:
            league_id: League ID
            conference: Optional conference filter (e.g., "Eastern", "Western")
            division: Optional division filter (e.g., "Atlantic", "Pacific")
            
        Returns:
            List of teams
        """
        # Build filter
        filter = {"league_id": league_id}
        
        if conference:
            filter["conference"] = conference
        
        if division:
            filter["division"] = division
        
        # Get teams
        return await self.find(
            filter=filter,
            sort=[("name", 1)]
        )
    
    async def search_teams(
        self,
        query: str,
        limit: int = 20
    ) -> List[TeamInDB]:
        """
        Search for teams by name or abbreviation.
        
        Args:
            query: Search query
            limit: Maximum number of teams to return
            
        Returns:
            List of matching teams
        """
        # Build filter
        filter = {
            "$or": [
                {"name": {"$regex": query, "$options": "i"}},
                {"abbreviation": {"$regex": query, "$options": "i"}}
            ]
        }
        
        # Get teams
        return await self.find(
            filter=filter,
            sort=[("name", 1)],
            limit=limit
        )
    
    async def update_team_colors(
        self,
        team_id: str,
        primary_color: str,
        secondary_color: str
    ) -> bool:
        """
        Update a team's colors.
        
        Args:
            team_id: Team ID
            primary_color: Primary color hex code
            secondary_color: Secondary color hex code
            
        Returns:
            True if update successful, False otherwise
        """
        return await self.update(
            id=team_id,
            update={
                "$set": {
                    "primary_color": primary_color,
                    "secondary_color": secondary_color
                }
            }
        )