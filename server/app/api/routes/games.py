"""
API routes for game operations.
"""
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel

from app.core.logger import logger
from app.api.dependencies import get_current_user, get_vip_user, get_pagination_params
from app.db.models.game import GameInDB
from app.db.models.user import UserInDB
from app.services.game_service import GameService
from app.services.user_service import UserService

# Initialize services
game_service = GameService()
user_service = UserService()

# Create router
router = APIRouter()


class GameResponse(BaseModel):
    """
    Game response with additional fields.
    """
    game: GameInDB
    is_vip_only: bool
    has_access: bool


@router.get("/", response_model=List[GameResponse])
async def get_games(
    date: Optional[str] = None,
    status: Optional[str] = None,
    current_user: Optional[UserInDB] = Depends(get_current_user),
):
    """
    Get games by date or status.
    """
    # Parse date if provided
    parsed_date = None
    if date:
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD."
            )
    
    # Get games based on parameters
    if parsed_date:
        games = await game_service.get_games_by_date(parsed_date)
    elif status == "live":
        games = await game_service.get_live_games()
    elif status == "upcoming":
        games = await game_service.get_upcoming_games(days=7)
    else:
        # Default to today's games
        games = await game_service.get_games_by_date(datetime.utcnow())
    
    # Check user access for each game
    result = []
    for game in games:
        is_vip_only = False
        has_access = True
        
        # Check if game is VIP only and if user has access
        if game.prediction and game.prediction.is_in_vip_betslip:
            is_vip_only = True
            has_access = False
            
            # Check user access if authenticated
            if current_user:
                has_access = await user_service.check_access_to_betslip(
                    str(current_user.id),
                    game.date.strftime("%Y-%m-%d")
                )
        
        # Always allow access to game of the day
        if game.prediction and game.prediction.is_game_of_day:
            is_vip_only = False
            has_access = True
        
        result.append(GameResponse(
            game=game,
            is_vip_only=is_vip_only,
            has_access=has_access
        ))
    
    return result


@router.get("/game-of-day", response_model=GameInDB)
async def get_game_of_the_day():
    """
    Get the featured game of the day.
    """
    game = await game_service.get_game_of_the_day()
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No game of the day found"
        )
    return game


@router.get("/live", response_model=List[GameInDB])
async def get_live_games():
    """
    Get currently live games.
    """
    return await game_service.get_live_games()


@router.get("/upcoming", response_model=List[GameInDB])
async def get_upcoming_games(days: int = Query(7, ge=1, le=30)):
    """
    Get upcoming games within a time window.
    """
    return await game_service.get_upcoming_games(days=days)


@router.get("/betslip", response_model=List[GameResponse])
async def get_betslip_games(
    date: Optional[str] = None,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get games in the VIP betslip for a date.
    """
    # Parse date if provided
    parsed_date = None
    if date:
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD."
            )
    else:
        parsed_date = datetime.utcnow()
    
    # Get betslip games
    games = await game_service.get_betslip_games(parsed_date)
    
    # Check user access
    has_access = await user_service.check_access_to_betslip(
        str(current_user.id),
        parsed_date.strftime("%Y-%m-%d")
    )
    
    # Prepare response
    result = []
    for game in games:
        result.append(GameResponse(
            game=game,
            is_vip_only=True,
            has_access=has_access
        ))
    
    return result


@router.get("/{game_id}", response_model=GameResponse)
async def get_game(
    game_id: str = Path(..., title="The ID of the game to get"),
    current_user: Optional[UserInDB] = Depends(get_current_user),
):
    """
    Get a specific game by ID.
    """
    game = await game_service.get_game_by_id(game_id)
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    # Check if game is VIP only and if user has access
    is_vip_only = False
    has_access = True
    
    if game.prediction and game.prediction.is_in_vip_betslip:
        is_vip_only = True
        has_access = False
        
        # Check user access if authenticated
        if current_user:
            has_access = await user_service.check_access_to_betslip(
                str(current_user.id),
                game.date.strftime("%Y-%m-%d")
            )
    
    # Always allow access to game of the day
    if game.prediction and game.prediction.is_game_of_day:
        is_vip_only = False
        has_access = True
    
    return GameResponse(
        game=game,
        is_vip_only=is_vip_only,
        has_access=has_access
    )


@router.get("/{game_id}/h2h", response_model=List[GameInDB])
async def get_head_to_head(
    game_id: str = Path(..., title="The ID of the game"),
    limit: int = Query(10, ge=1, le=50),
):
    """
    Get head-to-head games between the teams in a specific game.
    """
    # Get the game to find the teams
    game = await game_service.get_game_by_id(game_id)
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    # Get head-to-head games
    h2h_games = await game_service.get_head_to_head_games(
        game.home_team.team_id,
        game.away_team.team_id,
        limit=limit
    )
    
    return h2h_games


@router.get("/search/{query}", response_model=List[GameInDB])
async def search_games(
    query: str = Path(..., title="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """
    Search for games by team names.
    """
    games = await game_service.search_games(query, limit=limit)
    return games


@router.post("/game-of-day/{game_id}", response_model=GameInDB)
async def set_game_of_day(
    game_id: str = Path(..., title="The ID of the game"),
    current_user: UserInDB = Depends(get_vip_user),
):
    """
    Set a game as the game of the day.
    Admin-only endpoint.
    """
    game = await game_service.mark_game_as_game_of_day(game_id)
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    return game