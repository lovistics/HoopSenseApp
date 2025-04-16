"""
API routes for prediction operations.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel

from app.core.logger import logger
from app.api.dependencies import get_current_user, get_vip_user
from app.db.models.user import UserInDB
from app.db.models.game import GameInDB, GamePrediction
from app.db.models.prediction import PredictionHistory
from app.services.prediction_service import PredictionService
from app.services.game_service import GameService
from app.services.user_service import UserService

# Initialize services
prediction_service = PredictionService()
game_service = GameService()
user_service = UserService()

# Create router
router = APIRouter()


class PredictionResponse(BaseModel):
    """
    Prediction response model.
    """
    game_id: str
    prediction: GamePrediction
    has_access: bool


class AccuracyStatsResponse(BaseModel):
    """
    User prediction accuracy statistics.
    """
    total_predictions: int
    correct_predictions: int
    accuracy: float
    current_streak: int
    best_streak: int


class DailyBetslipResponse(BaseModel):
    """
    Daily betslip response.
    """
    date: str
    games: List[GameInDB]
    has_access: bool


class ExplanationResponse(BaseModel):
    """
    Prediction explanation response model.
    """
    game_id: str
    home_team: str
    away_team: str
    text_explanation: str
    prediction: str
    confidence: float
    top_features: Optional[Dict[str, float]] = None
    feature_groups: Optional[Dict[str, Dict[str, Any]]] = None
    game_prediction: Optional[Dict[str, Any]] = None
    date: Optional[str] = None


@router.get("/game/{game_id}", response_model=PredictionResponse)
async def get_game_prediction(
    game_id: str = Path(..., title="The ID of the game"),
    current_user: Optional[UserInDB] = Depends(get_current_user),
):
    """
    Get prediction for a specific game.
    """
    # Get the game with prediction
    game = await game_service.get_game_by_id(game_id)
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    if not game.prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No prediction available for this game"
        )
    
    # Check if game is VIP only and if user has access
    has_access = True
    
    if game.prediction.is_in_vip_betslip:
        has_access = False
        
        # Check user access if authenticated
        if current_user:
            has_access = await user_service.check_access_to_betslip(
                str(current_user.id),
                game.date.strftime("%Y-%m-%d")
            )
    
    # Always allow access to game of the day
    if game.prediction.is_game_of_day:
        has_access = True
    
    return PredictionResponse(
        game_id=str(game.id),
        prediction=game.prediction,
        has_access=has_access
    )


@router.get("/history", response_model=List[PredictionHistory])
async def get_prediction_history(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get prediction history for the current user.
    """
    # Parse dates if provided
    parsed_start = None
    parsed_end = None
    
    if start_date:
        try:
            parsed_start = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid start_date format. Use YYYY-MM-DD."
            )
    
    if end_date:
        try:
            parsed_end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid end_date format. Use YYYY-MM-DD."
            )
    
    # Get history
    history = await prediction_service.get_user_prediction_history(
        str(current_user.id),
        parsed_start,
        parsed_end
    )
    
    return history


@router.get("/stats", response_model=AccuracyStatsResponse)
async def get_accuracy_stats(
    days: int = Query(30, ge=1, le=365),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get prediction accuracy statistics for the current user.
    """
    stats = await prediction_service.get_user_accuracy_stats(
        str(current_user.id),
        days=days
    )
    
    return AccuracyStatsResponse(**stats)


@router.get("/betslip", response_model=DailyBetslipResponse)
async def get_daily_betslip(
    date: Optional[str] = None,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get games in the VIP betslip for a specific date.
    """
    # Parse date if provided
    parsed_date = None
    date_str = None
    
    if date:
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d")
            date_str = date
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD."
            )
    else:
        parsed_date = datetime.utcnow()
        date_str = parsed_date.strftime("%Y-%m-%d")
    
    # Get betslip games
    games = await game_service.get_betslip_games(parsed_date)
    
    # Check if user has access
    has_access = await user_service.check_access_to_betslip(
        str(current_user.id),
        date_str
    )
    
    return DailyBetslipResponse(
        date=date_str,
        games=games,
        has_access=has_access
    )


@router.post("/purchase-betslip", response_model=UserInDB)
async def purchase_betslip(
    date: str = Query(..., title="Date in YYYY-MM-DD format"),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Purchase access to a daily betslip.
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD."
        )
    
    # Check if user already has access
    has_access = await user_service.check_access_to_betslip(
        str(current_user.id),
        date
    )
    
    if has_access:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have access to this betslip"
        )
    
    # Add purchase to user
    # In a real system, you'd handle payment processing here
    updated_user = await user_service.add_betslip_purchase(
        str(current_user.id),
        date
    )
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process purchase"
        )
    
    return updated_user


@router.get("/game/{game_id}/explanation", response_model=ExplanationResponse)
async def get_game_prediction_explanation(
    game_id: str = Path(..., title="The ID of the game"),
    include_raw_shap: bool = Query(False, title="Include raw SHAP values in response"),
    current_user: Optional[UserInDB] = Depends(get_current_user),
):
    """
    Get an explanation for a game prediction showing which factors
    influenced the model's decision.
    """
    # Get game with prediction
    game = await game_service.get_game_by_id(game_id)
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Game not found"
        )
    
    if not game.prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No prediction available for this game"
        )
    
    # Check if game is VIP only and if user has access
    has_access = True
    
    if game.prediction.is_in_vip_betslip:
        has_access = False
        
        # Check user access if authenticated
        if current_user:
            has_access = await user_service.check_access_to_betslip(
                str(current_user.id),
                game.date.strftime("%Y-%m-%d")
            )
    
    # Always allow access to game of the day
    if game.prediction.is_game_of_day:
        has_access = True
    
    if not has_access:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this prediction explanation"
        )
    
    # Get explanation
    explanation = await prediction_service.get_prediction_explanation(
        game_id,
        include_raw_shap=include_raw_shap
    )
    
    return explanation


@router.post("/admin/set-betslip", response_model=Dict[str, Any])
async def set_betslip_games(
    date: str = Query(..., title="Date in YYYY-MM-DD format"),
    game_ids: List[str] = Query(..., title="List of game IDs to include in betslip"),
    current_user: UserInDB = Depends(get_vip_user),
):
    """
    Set games to include in the VIP betslip for a date.
    Admin-only endpoint.
    """
    try:
        parsed_date = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD."
        )
    
    # Update betslip games
    cleared, updated = await prediction_service.update_betslip_games(
        parsed_date,
        game_ids
    )
    
    return {
        "date": date,
        "cleared_games": cleared,
        "updated_games": updated,
        "game_ids": game_ids
    }