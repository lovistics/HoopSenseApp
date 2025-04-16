"""
Security utilities for authentication and authorization.
"""
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
from passlib.context import CryptContext
from pydantic import ValidationError

from app.core.config import settings
from app.core.logger import logger
from app.db.models.user import UserInDB, UserModel

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token settings
ALGORITHM = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches hash, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Generate a password hash.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def create_access_token(
    subject: Union[str, int], 
    expires_delta: Optional[timedelta] = None,
    scopes: Optional[list] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        subject: Token subject (typically user ID)
        expires_delta: Optional expiration override
        scopes: Optional token scopes
        
    Returns:
        Encoded JWT token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "exp": expire, 
        "sub": str(subject), 
        "type": "access",
        "iat": datetime.utcnow()
    }
    
    if scopes:
        to_encode["scopes"] = scopes
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(subject: Union[str, int], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        subject: Token subject (typically user ID)
        expires_delta: Optional expiration override
        
    Returns:
        Encoded JWT token
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "exp": expire, 
        "sub": str(subject), 
        "type": "refresh",
        "iat": datetime.utcnow()
    }
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode a JWT token.
    
    Args:
        token: JWT token to decode
        
    Returns:
        Decoded token payload
        
    Raises:
        JWTError: If token is invalid
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except ExpiredSignatureError as e:
        logger.warning(f"Expired token: {e}")
        raise JWTError("Token has expired") from e
    except JWTError as e:
        logger.error(f"Invalid token: {e}")
        raise
    except Exception as e:
        logger.error(f"Error decoding token: {e}")
        raise JWTError("Invalid token") from e


def is_token_expired(token: str) -> bool:
    """
    Check if a token is expired.
    
    Args:
        token: JWT token to check
        
    Returns:
        True if token is expired, False otherwise
    """
    try:
        payload = decode_token(token)
        expiration = datetime.fromtimestamp(payload.get("exp", 0))
        return datetime.utcnow() > expiration
    except Exception:
        return True


def create_tokens_for_user(user: Union[UserModel, UserInDB, Dict[str, Any]]) -> Dict[str, str]:
    """
    Create access and refresh tokens for a user.
    
    Args:
        user: User model or dictionary
        
    Returns:
        Dictionary containing tokens
    """
    user_id = str(user.id if hasattr(user, 'id') else user.get('id', ''))
    
    # Determine user role/scopes - this would be expanded in a real system
    scopes = ["user"]
    
    access_token = create_access_token(subject=user_id, scopes=scopes)
    refresh_token = create_refresh_token(subject=user_id)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }