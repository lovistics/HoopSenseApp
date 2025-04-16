"""
API routes for authentication operations.
"""
from datetime import datetime, timedelta
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.core.config import settings
from app.core.logger import logger
from app.core.security import create_tokens_for_user, decode_token
from app.api.dependencies import get_current_user
from app.db.models.user import UserInDB, UserCreate, UserResponse
from app.services.user_service import UserService

# Initialize services
user_service = UserService()

# Create router
router = APIRouter()


@router.post("/token", response_model=Dict[str, Any])
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user = await user_service.authenticate_user(
        form_data.username,  # OAuth2 uses username field for email
        form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_token(
    refresh_token: str,
):
    """
    Refresh access token using refresh token.
    """
    try:
        # Decode refresh token
        payload = decode_token(refresh_token)
        
        # Check if it's a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token type",
            )
            
        # Get user ID from token
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token",
            )
            
        # Get user
        user = await user_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
            
        # Create new tokens
        tokens = create_tokens_for_user(user)
        
        # Return tokens and user information
        return {
            **tokens,
            "user": UserResponse(
                _id=user.id,
                email=user.email,
                name=user.name,
                subscription=user.subscription,
                betslip_purchases=user.betslip_purchases,
                settings=user.settings,
                created_at=user.created_at
            )
        }
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
):
    """
    Register a new user account.
    """
    try:
        user = await user_service.create_user(user_data)
        return UserResponse(
            _id=user.id,
            email=user.email,
            name=user.name,
            subscription=user.subscription,
            betslip_purchases=user.betslip_purchases,
            settings=user.settings,
            created_at=user.created_at
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating user"
        )


@router.post("/logout")
async def logout(
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Logout user.
    
    Note: With JWTs, we don't actually invalidate the token server-side.
    In a production system, you'd use a token blacklist or shorter token lifetimes.
    """
    # This is just a stub - in a real system you might:
    # 1. Add the token to a blacklist
    # 2. Update the user's last logout time
    # 3. Clear user session data
    
    return {"detail": "Successfully logged out"}