"""
API dependencies and utilities for route handlers.
"""
from datetime import datetime
from typing import Callable, Dict, Optional, Any

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from app.core.config import settings
from app.core.logger import logger
from app.core.security import decode_token
from app.db.models.user import UserInDB
from app.services.user_service import UserService

# Initialize services
user_service = UserService()

# OAuth2 token URL path and scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=settings.API_V1_STR + "/auth/token")


async def rate_limit(request: Request) -> None:
    """
    Rate limiting dependency.
    
    Args:
        request: Request object
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    if not settings.RATE_LIMIT_ENABLED:
        return
    
    # This would be implemented with Redis in a production system
    # For now, just log that rate limiting would be applied
    client_ip = request.client.host if request.client else "unknown"
    logger.debug(f"Rate limiting check for IP: {client_ip}")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    rate_limit_check: None = Depends(rate_limit)
) -> UserInDB:
    """
    Get the current authenticated user based on JWT token.
    
    Args:
        token: JWT token
        rate_limit_check: Rate limiting dependency
        
    Returns:
        Current user
        
    Raises:
        HTTPException: If credentials invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = decode_token(token)
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None:
            logger.warning("Token has no subject claim")
            raise credentials_exception
        
        if token_type != "access":
            logger.warning(f"Invalid token type: {token_type}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type, access token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError as e:
        logger.warning(f"JWT validation error: {str(e)}")
        raise credentials_exception from e
    
    # Get user from database
    user = await user_service.get_user_by_id(user_id)
    if user is None:
        logger.warning(f"User not found: {user_id}")
        raise credentials_exception
        
    return user


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    Check if the current user is active.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current active user
    """
    # We might have an "is_active" flag in the future
    # For now, just return the current user
    return current_user


async def get_vip_user(
    current_user: UserInDB = Depends(get_current_user)
) -> UserInDB:
    """
    Check if the current user has VIP or trial access.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current VIP user
        
    Raises:
        HTTPException: If user does not have VIP access
    """
    now = datetime.utcnow()
    
    if current_user.subscription.status == "vip":
        if current_user.subscription.subscription_ends_at and current_user.subscription.subscription_ends_at > now:
            return current_user
    
    if current_user.subscription.status == "trial":
        if current_user.subscription.trial_ends_at and current_user.subscription.trial_ends_at > now:
            return current_user
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="VIP access required"
    )


def get_pagination_params(
    skip: int = 0, 
    limit: int = 100, 
    max_limit: int = 1000
) -> Dict[str, int]:
    """
    Common pagination parameters.
    
    Args:
        skip: Number of items to skip
        limit: Maximum number of items
        max_limit: Maximum allowed limit
        
    Returns:
        Dictionary of pagination parameters
    """
    # Ensure limit does not exceed max_limit
    if limit > max_limit:
        limit = max_limit
    
    return {
        "skip": max(0, skip),  # Ensure skip is not negative
        "limit": limit
    }


def get_service(service_class: Callable) -> Any:
    """
    Get a service instance.
    
    Args:
        service_class: Service class to instantiate
        
    Returns:
        Service instance
    """
    return service_class()