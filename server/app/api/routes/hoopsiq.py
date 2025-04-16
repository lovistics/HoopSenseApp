"""
API routes for HoopsIQ AI analysis.
"""
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel

from app.api.dependencies import get_current_user, get_vip_user
from app.db.models.user import UserInDB
from app.services.hoopsiq_service import HoopsIQService
from app.services.game_service import GameService
from app.services.user_service import UserService

# Initialize services
hoopsiq_service = HoopsIQService()
game_service = GameService()
user_service = UserService()

# Create router
router = APIRouter()


class InsightResponse(BaseModel):
    """
    AI insight response.
    """
    insight: str
    category: str
    confidence: int
    source: str


class ScenarioRequest(BaseModel):
    """
    Scenario analysis request.
    """
    query: str


class ScenarioResponse(BaseModel):
    """
    Scenario analysis response.
    """
    originalProbability: float
    newProbability: float
    explanation: str
    impactedAreas: List[Dict[str, Any]]


@router.get("/{game_id}/insights", response_model=List[InsightResponse])
async def get_insights(
    game_id: str = Path(..., title="The ID of the game"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get AI-generated insights for a game.
    """
    # Get the game to check if it's VIP only
    game = await game_service.get_game_by_id(game_id)
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    # Check access if game is in VIP betslip
    if game.prediction and game.prediction.is_in_vip_betslip:
        has_access = await user_service.check_access_to_betslip(
            str(current_user.id),
            game.date.strftime("%Y-%m-%d")
        )
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="VIP access required for this game"
            )
    
    # Get insights
    try:
        insights = await hoopsiq_service.get_insights(game_id)
        return [InsightResponse(**insight) for insight in insights]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating insights: {str(e)}"
        )


@router.post("/{game_id}/analyze", response_model=ScenarioResponse)
async def analyze_scenario(
    scenario: ScenarioRequest,
    game_id: str = Path(..., title="The ID of the game"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Analyze a what-if scenario for a game.
    """
    # Get the game to check if it's VIP only
    game = await game_service.get_game_by_id(game_id)
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    # For free users, limit number of queries per day
    if current_user.subscription.status == "free":
        # For simplicity, we're not implementing a real query counter here
        # In a production system, you'd track this in the database
        # Just showing the concept
        MAX_FREE_QUERIES = 3
        
        # Check if user has exceeded free queries
        # (this is a placeholder, real implementation would check the database)
        free_queries_used = 0  # This would be fetched from the database
        
        if free_queries_used >= MAX_FREE_QUERIES:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Free users are limited to {MAX_FREE_QUERIES} queries per day"
            )
    
    # Check access if game is in VIP betslip
    if game.prediction and game.prediction.is_in_vip_betslip:
        has_access = await user_service.check_access_to_betslip(
            str(current_user.id),
            game.date.strftime("%Y-%m-%d")
        )
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="VIP access required for this game"
            )
    
    # Analyze scenario
    try:
        result = await hoopsiq_service.analyze_scenario(game_id, scenario.query)
        return ScenarioResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing scenario: {str(e)}"
        )