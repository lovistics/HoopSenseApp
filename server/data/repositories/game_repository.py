"""
Repository for game data operations.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from bson import ObjectId

from app.core.logger import logger
from app.db.models.game import GameInDB, COLLECTION_NAME as GAMES_COLLECTION
from server.data.repositories.base_repository import BaseRepository


class GameRepository(BaseRepository[GameInDB]):
    """
    Repository for game-related database operations.
    """
    
    def __init__(self):
        """Initialize the game repository."""
        super().__init__(GAMES_COLLECTION, GameInDB)
    
    async def find_by_external_id(self, external_id: int) -> Optional[GameInDB]:
        """
        Find a game by external ID.
        
        Args:
            external_id: External API game ID
            
        Returns:
            Game document or None if not found
        """
        return await self.find_one({"external_id": external_id})
    
    async def find_by_date_range(
        self, 
        start_date: datetime,
        end_date: datetime,
        league_id: Optional[str] = None,
        team_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[GameInDB]:
        """
        Find games within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            league_id: Optional league ID filter
            team_id: Optional team ID filter
            status: Optional status filter (e.g., "LIVE", "FT")
            limit: Maximum number of games to return
            
        Returns:
            List of matching games
        """
        # Build filter
        filter = {
            "date": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        
        if league_id:
            filter["league_id"] = league_id
        
        if team_id:
            filter["$or"] = [
                {"home_team.team_id": team_id},
                {"away_team.team_id": team_id}
            ]
        
        if status:
            filter["status.short"] = status
        
        # Get games
        return await self.find(
            filter=filter,
            sort=[("date", 1)],
            limit=limit
        )
    
    async def find_games_by_date(
        self, 
        date: datetime,
        league_id: Optional[str] = None,
        analyzed_only: bool = False
    ) -> List[GameInDB]:
        """
        Find games for a specific date.
        
        Args:
            date: The date to find games for
            league_id: Optional league ID filter
            analyzed_only: If True, only return games with predictions
            
        Returns:
            List of matching games
        """
        # Create date range (start of day to end of day)
        start_of_day = datetime(date.year, date.month, date.day)
        end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)
        
        # Build filter
        filter = {
            "date": {
                "$gte": start_of_day,
                "$lte": end_of_day
            }
        }
        
        if league_id:
            filter["league_id"] = league_id
        
        if analyzed_only:
            filter["is_analyzed"] = True
            filter["prediction"] = {"$exists": True, "$ne": None}
        
        # Get games
        return await self.find(
            filter=filter,
            sort=[("date", 1)]
        )
    
    async def find_live_games(
        self,
        league_id: Optional[str] = None
    ) -> List[GameInDB]:
        """
        Find currently live games.
        
        Args:
            league_id: Optional league ID filter
            
        Returns:
            List of live games
        """
        # Build filter
        filter = {
            "$or": [
                {"status.short": "LIVE"},
                {"status.short": "Q1"},
                {"status.short": "Q2"},
                {"status.short": "Q3"},
                {"status.short": "Q4"},
                {"status.short": "OT"},
                {"status.short": "HT"},
                {"status.long": "In Progress"},
                {"status.long": "Game Started"}
            ]
        }
        
        if league_id:
            filter["league_id"] = league_id
        
        # Get games
        return await self.find(
            filter=filter,
            sort=[("date", 1)]
        )
    
    async def find_upcoming_games(
        self,
        days: int = 7,
        league_id: Optional[str] = None,
        team_id: Optional[str] = None
    ) -> List[GameInDB]:
        """
        Find upcoming games within a time window.
        
        Args:
            days: Number of days to look ahead
            league_id: Optional league ID filter
            team_id: Optional team ID filter
            
        Returns:
            List of upcoming games
        """
        now = datetime.utcnow()
        end_date = now + timedelta(days=days)
        
        # Build filter
        filter = {
            "date": {"$gte": now, "$lte": end_date},
            "status.long": {"$in": ["Not Started", "Scheduled"]}
        }
        
        if league_id:
            filter["league_id"] = league_id
        
        if team_id:
            filter["$or"] = [
                {"home_team.team_id": team_id},
                {"away_team.team_id": team_id}
            ]
        
        # Get games
        return await self.find(
            filter=filter,
            sort=[("date", 1)]
        )
    
    async def find_game_of_the_day(self, date: Optional[datetime] = None) -> Optional[GameInDB]:
        """
        Find the featured game of the day.
        
        Args:
            date: Optional date (defaults to today)
            
        Returns:
            Game of the day or None if not found
        """
        if not date:
            date = datetime.utcnow()
        
        # Create date range (start of day to end of day)
        start_of_day = datetime(date.year, date.month, date.day)
        end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)
        
        # Try to find game marked as game of the day
        game = await self.find_one({
            "date": {"$gte": start_of_day, "$lte": end_of_day},
            "prediction.is_game_of_day": True
        })
        
        if not game:
            # If no game is marked as game of the day, try to find one with prediction
            game = await self.find_one({
                "date": {"$gte": start_of_day, "$lte": end_of_day},
                "prediction": {"$exists": True, "$ne": None}
            })
        
        return game
    
    async def find_betslip_games(self, date: Optional[datetime] = None) -> List[GameInDB]:
        """
        Find games in the VIP betslip.
        
        Args:
            date: Optional date (defaults to today)
            
        Returns:
            List of betslip games
        """
        if not date:
            date = datetime.utcnow()
        
        # Create date range (start of day to end of day)
        start_of_day = datetime(date.year, date.month, date.day)
        end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)
        
        # Get games
        return await self.find(
            filter={
                "date": {"$gte": start_of_day, "$lte": end_of_day},
                "prediction.is_in_vip_betslip": True
            },
            sort=[("date", 1)]
        )
    
    async def find_head_to_head_games(
        self,
        team1_id: str,
        team2_id: str,
        limit: int = 10
    ) -> List[GameInDB]:
        """
        Find head-to-head games between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            limit: Maximum number of games to return
            
        Returns:
            List of head-to-head games
        """
        # Validate IDs
        if not ObjectId.is_valid(team1_id) or not ObjectId.is_valid(team2_id):
            return []
        
        # Build filter
        filter = {
            "$or": [
                {"home_team.team_id": team1_id, "away_team.team_id": team2_id},
                {"home_team.team_id": team2_id, "away_team.team_id": team1_id}
            ],
            "status.short": {"$in": ["FT", "Final", "Finished", "AOT"]}
        }
        
        # Get games
        return await self.find(
            filter=filter,
            sort=[("date", -1)],
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
        # Build filter
        filter = {
            "$or": [
                {"home_team.name": {"$regex": query, "$options": "i"}},
                {"away_team.name": {"$regex": query, "$options": "i"}}
            ]
        }
        
        # Get games
        return await self.find(
            filter=filter,
            sort=[("date", -1)],
            limit=limit
        )
    
    async def update_prediction(
        self,
        game_id: str,
        prediction_data: Dict[str, Any]
    ) -> bool:
        """
        Update a game's prediction.
        
        Args:
            game_id: Game ID
            prediction_data: Prediction data to update
            
        Returns:
            True if update successful, False otherwise
        """
        return await self.update(
            id=game_id,
            update={
                "$set": {
                    "prediction": prediction_data,
                    "is_analyzed": True
                }
            }
        )
    
    async def mark_game_as_game_of_day(self, game_id: str) -> Tuple[int, bool]:
        """
        Mark a game as the game of the day.
        
        Args:
            game_id: Game ID
            
        Returns:
            Tuple of (cleared game count, updated target game)
        """
        # Get game to find its date
        game = await self.find_by_id(game_id)
        if not game:
            return 0, False
        
        game_date = game.date
        start_of_day = datetime(game_date.year, game_date.month, game_date.day)
        end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)
        
        # Clear any other game of the day for this date
        clear_result = await self.update_by_filter(
            filter={
                "date": {"$gte": start_of_day, "$lte": end_of_day},
                "prediction.is_game_of_day": True,
                "_id": {"$ne": ObjectId(game_id)}
            },
            update={"$set": {"prediction.is_game_of_day": False}}
        )
        
        # Mark this game as game of the day
        update_result = await self.update(
            id=game_id,
            update={"$set": {"prediction.is_game_of_day": True}}
        )
        
        return clear_result, update_result
    
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
        # Create date range
        start_of_day = datetime(date.year, date.month, date.day)
        end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)
        
        # Clear all betslip games for this date
        clear_result = await self.update_by_filter(
            filter={
                "date": {"$gte": start_of_day, "$lte": end_of_day},
                "prediction.is_in_vip_betslip": True
            },
            update={"$set": {"prediction.is_in_vip_betslip": False}}
        )
        
        # Convert IDs to ObjectIds
        valid_game_ids = [ObjectId(gid) for gid in game_ids if ObjectId.is_valid(gid)]
        
        if not valid_game_ids:
            return clear_result, 0
        
        # Set the specified games as betslip games
        set_result = await self.update_by_filter(
            filter={
                "_id": {"$in": valid_game_ids},
                "date": {"$gte": start_of_day, "$lte": end_of_day}
            },
            update={"$set": {"prediction.is_in_vip_betslip": True}}
        )
        
        return clear_result, set_result