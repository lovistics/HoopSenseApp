"""
Game service for handling game-related business logic.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union

from fastapi import HTTPException, status

from app.core.logger import logger
from app.db.models.game import GameInDB, GameModel, GamePrediction
from server.data.repositories.game_repository import GameRepository
from server.data.repositories.team_repository import TeamRepository


class GameService:
    """Service for game-related operations."""
    
    def __init__(self):
        """Initialize the game service with repositories."""
        self.game_repository = GameRepository()
        self.team_repository = TeamRepository()
    
    async def get_game_by_id(self, game_id: str) -> Optional[GameInDB]:
        """
        Get a game by ID.
        
        Args:
            game_id: The ID of the game to retrieve
            
        Returns:
            Game object or None if not found
        """
        return await self.game_repository.find_by_id(game_id)
    
    async def get_game_by_external_id(self, external_id: int) -> Optional[GameInDB]:
        """
        Get a game by external ID.
        
        Args:
            external_id: The external API ID of the game
            
        Returns:
            Game object or None if not found
        """
        return await self.game_repository.find_by_external_id(external_id)
    
    async def create_game(self, game: GameModel) -> GameInDB:
        """
        Create a new game.
        
        Args:
            game: The game model to create
            
        Returns:
            Created game
            
        Raises:
            HTTPException: If a game with the same external ID already exists
        """
        # Check if game with this external_id already exists
        existing = await self.get_game_by_external_id(game.external_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Game with external_id {game.external_id} already exists"
            )
        
        # Create game
        game_id = await self.game_repository.create(game)
        if not game_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create game"
            )
        
        # Return created game
        return await self.get_game_by_id(game_id)
    
    async def update_game(
        self, 
        game_id: str, 
        game_data: Dict[str, Any]
    ) -> Optional[GameInDB]:
        """
        Update a game by ID.
        
        Args:
            game_id: The ID of the game to update
            game_data: The data to update
            
        Returns:
            Updated game or None if not found
        """
        # Update game
        updated = await self.game_repository.update(
            id=game_id,
            update={"$set": game_data}
        )
        
        if not updated:
            return None
        
        # Return updated game
        return await self.get_game_by_id(game_id)
    
    async def update_game_prediction(
        self, 
        game_id: str, 
        prediction_data: Union[Dict[str, Any], GamePrediction]
    ) -> Optional[GameInDB]:
        """
        Update a game's prediction.
        
        Args:
            game_id: The ID of the game to update
            prediction_data: The prediction data to update
            
        Returns:
            Updated game or None if not found
        """
        # If prediction_data is a GamePrediction model, convert to dict
        if isinstance(prediction_data, GamePrediction):
            prediction_data = prediction_data.dict()
        
        # Update game prediction
        updated = await self.game_repository.update_prediction(game_id, prediction_data)
        
        if not updated:
            return None
        
        # Return updated game
        return await self.get_game_by_id(game_id)
    
    async def get_games_by_date(
        self, 
        date: datetime,
        league_id: Optional[str] = None
    ) -> List[GameInDB]:
        """
        Get games for a specific date.
        
        Args:
            date: The date to get games for
            league_id: Optional league ID filter
            
        Returns:
            List of games for the date
        """
        return await self.game_repository.find_games_by_date(
            date=date,
            league_id=league_id
        )
    
    async def get_live_games(
        self,
        league_id: Optional[str] = None
    ) -> List[GameInDB]:
        """
        Get currently live games.
        
        Args:
            league_id: Optional league ID filter
            
        Returns:
            List of live games
        """
        return await self.game_repository.find_live_games(league_id=league_id)
    
    async def get_upcoming_games(
        self, 
        days: int = 7,
        league_id: Optional[str] = None,
        team_id: Optional[str] = None
    ) -> List[GameInDB]:
        """
        Get upcoming games within a time window.
        
        Args:
            days: Number of days to look ahead
            league_id: Optional league ID filter
            team_id: Optional team ID filter
            
        Returns:
            List of upcoming games
        """
        return await self.game_repository.find_upcoming_games(
            days=days,
            league_id=league_id,
            team_id=team_id
        )
    
    async def get_game_of_the_day(
        self,
        date: Optional[datetime] = None
    ) -> Optional[GameInDB]:
        """
        Get the featured game of the day.
        
        Args:
            date: Optional date (defaults to today)
            
        Returns:
            Game of the day or None if not found
        """
        return await self.game_repository.find_game_of_the_day(date=date)
    
    async def get_betslip_games(
        self,
        date: Optional[datetime] = None
    ) -> List[GameInDB]:
        """
        Get games in the VIP betslip.
        
        Args:
            date: Optional date (defaults to today)
            
        Returns:
            List of betslip games
        """
        return await self.game_repository.find_betslip_games(date=date)
    
    async def get_head_to_head_games(
        self,
        team1_id: str,
        team2_id: str,
        limit: int = 10
    ) -> List[GameInDB]:
        """
        Get head-to-head games between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            limit: Maximum number of games to return
            
        Returns:
            List of head-to-head games
        """
        return await self.game_repository.find_head_to_head_games(
            team1_id=team1_id,
            team2_id=team2_id,
            limit=limit
        )
    
    async def search_games(
        self,
        query: str,
        limit: int = 20
    ) -> List[GameInDB]:
        """
        Search for games by team names.
        
        Args:
            query: Search query
            limit: Maximum number of games to return
            
        Returns:
            List of matching games
        """
        return await self.game_repository.search_games(
            query=query,
            limit=limit
        )
    
    async def mark_game_as_game_of_day(self, game_id: str) -> Optional[GameInDB]:
        """
        Mark a game as the game of the day.
        
        Args:
            game_id: The ID of the game to mark
            
        Returns:
            Updated game or None if not found
        """
        cleared, updated = await self.game_repository.mark_game_as_game_of_day(game_id)
        
        if not updated:
            return None
        
        return await self.get_game_by_id(game_id)
    
    async def update_betslip_games(
        self,
        date: datetime,
        game_ids: List[str]
    ) -> Tuple[int, int]:
        """
        Update the games in the VIP betslip for a specific date.
        
        Args:
            date: Date for the betslip
            game_ids: List of game IDs to include in the betslip
            
        Returns:
            Tuple of (cleared games count, updated games count)
        """
        return await self.game_repository.update_betslip_games(
            date=date,
            game_ids=game_ids
        )