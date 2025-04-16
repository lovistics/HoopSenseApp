"""
User model for handling user data.
"""
from datetime import datetime
from typing import Dict, Any, List, Optional

from pydantic import EmailStr, Field, validator
from pydantic.types import SecretStr

from app.db.models.base import MongoBaseModel, PyObjectId


class BetslipPurchase(MongoBaseModel):
    """Betslip purchase information."""
    
    date: str
    expires_at: datetime
    
    @validator('date')
    def validate_date_format(cls, v):
        """Validate date is in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class UserSubscription(MongoBaseModel):
    """User subscription information."""
    
    status: str  # "free", "trial", "vip"
    trial_ends_at: Optional[datetime] = None
    subscription_ends_at: Optional[datetime] = None
    
    @validator('status')
    def validate_status(cls, v):
        """Validate subscription status."""
        valid_statuses = ["free", "trial", "vip"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v


class UserNotificationSettings(MongoBaseModel):
    """User notification preferences."""
    
    game_start: bool = True
    close_games: bool = True
    prediction_results: bool = True


class UserSettings(MongoBaseModel):
    """User settings."""
    
    notifications: UserNotificationSettings = Field(default_factory=UserNotificationSettings)
    favorite_teams: List[str] = Field(default_factory=list)


class UserModel(MongoBaseModel):
    """User model representing an application user."""
    
    email: EmailStr
    password_hash: str
    name: str
    last_login: Optional[datetime] = None
    subscription: UserSubscription
    betslip_purchases: List[BetslipPurchase] = Field(default_factory=list)
    settings: UserSettings = Field(default_factory=UserSettings)


class UserInDB(UserModel):
    """User model as stored in the database."""
    pass


class UserCreate(MongoBaseModel):
    """Model for creating a new user."""
    
    email: EmailStr
    password: str
    name: str
    
    @validator('password')
    def password_min_length(cls, v):
        """Validate password meets minimum length."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v
    
    @validator('name')
    def name_not_empty(cls, v):
        """Validate name is not empty."""
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class UserUpdate(MongoBaseModel):
    """Model for updating a user."""
    
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    settings: Optional[UserSettings] = None
    
    @validator('name')
    def name_not_empty(cls, v):
        """Validate name is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip() if v else None


class UserResponse(MongoBaseModel):
    """User information returned to clients."""
    
    id: Optional[PyObjectId] = Field(alias="_id")
    email: EmailStr
    name: str
    subscription: UserSubscription
    betslip_purchases: List[BetslipPurchase] = Field(default_factory=list)
    settings: UserSettings
    created_at: datetime


# Collection name in MongoDB
COLLECTION_NAME = "users"