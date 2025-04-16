"""
Repository for player data operations.
"""
from typing import Dict, Any, List, Optional

from bson import ObjectId

from app.core.logger import logger
from app.db.models.player import PlayerInDB, COLLECTION_NAME
from server.data.repositories.base_repository import BaseRepository


class PlayerRepository(BaseRepository[PlayerInDB]):
    """
    Repository for player-related database operations.
    """
    
    def __init__(self):
        """Initialize the player repository."""
        super().__init__(COLLECTION_NAME, PlayerInDB)
    
    async def find_by_external_id(self, external_id: int) -> Optional[PlayerInDB]:
        """
        Find a player by external ID.
        
        Args:
            external_id: External API player ID
            
        Returns:
            Player or None if not found
        """
        return await self.find_one({"external_id": external_id})
    
    async def find_by_name(
        self, 
        name: str, 
        team_id: Optional[str] = None,
        exact: bool = False
    ) -> Optional[PlayerInDB]:
        """
        Find a player by name.
        
        Args:
            name: Player name to search for
            team_id: Optional team ID filter
            exact: If True, require exact match (case insensitive)
            
        Returns:
            Player or None if not found
        """
        # Build filter
        filter = {}
        
        if exact:
            filter["name"] = {"$regex": f"^{name}$", "$options": "i"}
        else:
            filter["name"] = {"$regex": name, "$options": "i"}
        
        if team_id:
            filter["team_id"] = team_id
        
        return await self.find_one(filter)
    
    async def find_players_by_team(
        self,
        team_id: str,
        position: Optional[str] = None
    ) -> List[PlayerInDB]:
        """
        Find players on a team.
        
        Args:
            team_id: Team ID
            position: Optional position filter (e.g., "Guard", "Forward")
            
        Returns:
            List of players
        """
        # Build filter
        filter = {"team_id": team_id}
        
        if position:
            filter["position"] = {"$regex": position, "$options": "i"}
        
        # Get players
        return await self.find(
            filter=filter,
            sort=[("name", 1)]
        )
    
    async def search_players(
        self,
        query: str,
        team_id: Optional[str] = None,
        limit: int = 20
    ) -> List[PlayerInDB]:
        """
        Search for players by name.
        
        Args:
            query: Search query
            team_id: Optional team ID filter
            limit: Maximum number of players to return
            
        Returns:
            List of matching players
        """
        # Build filter
        filter = {"name": {"$regex": query, "$options": "i"}}
        
        if team_id:
            filter["team_id"] = team_id
        
        # Get players
        return await self.find(
            filter=filter,
            sort=[("name", 1)],
            limit=limit
        )
    
    async def update_player_team(
        self,
        player_id: str,
        team_id: str
    ) -> bool:
        """
        Update a player's team.
        
        Args:
            player_id: Player ID
            team_id: New team ID
            
        Returns:
            True if update successful, False otherwise
        """
        return await self.update(
            id=player_id,
            update={
                "$set": {
                    "team_id": team_id
                }
            }
        )