"""
API routes for user operations.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.logger import logger
from app.api.dependencies import get_current_user, get_vip_user, get_pagination_params
from app.db.models.user import (
    UserInDB, UserCreate, UserUpdate, UserResponse, UserSettings
)
from app.services.user_service import UserService

# Initialize services
user_service = UserService()

# Create router
router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Get information about the current logged-in user.
    """
    return UserResponse(
        _id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        subscription=current_user.subscription,
        betslip_purchases=current_user.betslip_purchases,
        settings=current_user.settings,
        created_at=current_user.created_at
    )


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate = Body(...),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Update information for the current logged-in user.
    """
    updated_user = await user_service.update_user(str(current_user.id), user_data)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user"
        )
    
    return UserResponse(
        _id=updated_user.id,
        email=updated_user.email,
        name=updated_user.name,
        subscription=updated_user.subscription,
        betslip_purchases=updated_user.betslip_purchases,
        settings=updated_user.settings,
        created_at=updated_user.created_at
    )


@router.put("/me/settings", response_model=UserSettings)
async def update_user_settings(
    settings: UserSettings = Body(...),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Update settings for the current logged-in user.
    """
    updated_user = await user_service.update_settings(
        str(current_user.id),
        settings
    )
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating settings"
        )
    
    return updated_user.settings


@router.post("/start-trial", response_model=UserResponse)
async def start_free_trial(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Start a free trial for the current user.
    """
    # Check if user is eligible for a trial
    if current_user.subscription.status != "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has an active subscription or trial"
        )
    
    updated_user = await user_service.start_free_trial(str(current_user.id))
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error starting trial"
        )
    
    return UserResponse(
        _id=updated_user.id,
        email=updated_user.email,
        name=updated_user.name,
        subscription=updated_user.subscription,
        betslip_purchases=updated_user.betslip_purchases,
        settings=updated_user.settings,
        created_at=updated_user.created_at
    )


@router.post("/upgrade", response_model=UserResponse)
async def upgrade_to_vip(
    months: int = Query(1, ge=1, le=12),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Upgrade current user to VIP subscription.
    """
    # In a real system, you'd handle payment processing here
    updated_user = await user_service.activate_vip(str(current_user.id), months)
    
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error upgrading subscription"
        )
    
    return UserResponse(
        _id=updated_user.id,
        email=updated_user.email,
        name=updated_user.name,
        subscription=updated_user.subscription,
        betslip_purchases=updated_user.betslip_purchases,
        settings=updated_user.settings,
        created_at=updated_user.created_at
    )


@router.get("/admin/users", response_model=List[UserResponse])
async def get_users(
    subscription_status: Optional[str] = None,
    pagination: Dict[str, int] = Depends(get_pagination_params),
    current_user: UserInDB = Depends(get_vip_user),  # Using VIP for admin access (simplified)
):
    """
    Admin endpoint to get users.
    """
    # In a real system, you'd have proper admin role checks
    users = await user_service.find_users_by_criteria(
        subscription_status=subscription_status,
        limit=pagination["limit"],
        skip=pagination["skip"]
    )
    
    return [
        UserResponse(
            _id=user.id,
            email=user.email,
            name=user.name,
            subscription=user.subscription,
            betslip_purchases=user.betslip_purchases,
            settings=user.settings,
            created_at=user.created_at
        )
        for user in users
    ]