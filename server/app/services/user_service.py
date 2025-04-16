"""
User service for handling user-related business logic.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple

from bson import ObjectId
from fastapi import HTTPException, status
from pydantic import EmailStr

from app.core.logger import logger
from app.core.security import get_password_hash, verify_password, create_tokens_for_user
from app.db.models.user import UserInDB, UserCreate, UserUpdate, UserResponse, UserSettings, UserModel
from server.data.repositories.user_repository import UserRepository


class UserService:
    """Service for user-related operations."""
    
    def __init__(self):
        """Initialize the user service with repositories."""
        self.user_repository = UserRepository()
    
    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """
        Get a user by ID.
        
        Args:
            user_id: The ID of the user to retrieve
            
        Returns:
            User object or None if not found
        """
        return await self.user_repository.find_by_id(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Get a user by email.
        
        Args:
            email: The email of the user to retrieve
            
        Returns:
            User object or None if not found
        """
        return await self.user_repository.find_by_email(email)
    
    async def create_user(self, user_data: UserCreate) -> UserInDB:
        """
        Create a new user.
        
        Args:
            user_data: The user data to create
            
        Returns:
            Created user
            
        Raises:
            HTTPException: If a user with the same email already exists
        """
        # Check if email already exists
        existing_user = await self.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered"
            )
        
        # Create user with hashed password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user with free subscription by default
        user_model = UserModel(
            email=user_data.email,
            password_hash=hashed_password,
            name=user_data.name,
            subscription={
                "status": "free",
                "trial_ends_at": None,
                "subscription_ends_at": None
            },
            betslip_purchases=[],
            settings={
                "notifications": {
                    "game_start": True,
                    "close_games": True,
                    "prediction_results": True
                },
                "favorite_teams": []
            }
        )
        
        # Insert user
        user_id = await self.user_repository.create(user_model)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
        
        # Return created user
        return await self.get_user_by_id(user_id)
    
    async def authenticate_user(
        self,
        email: str,
        password: str
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user and return tokens if valid.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            Dictionary with tokens and user info if authentication successful, None otherwise
        """
        user = await self.get_user_by_email(email)
        
        if not user:
            return None
            
        if not verify_password(password, user.password_hash):
            return None
            
        # Update last login time
        await self.user_repository.update_last_login(str(user.id))
        
        # Create tokens
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
    
    async def update_user(
        self,
        user_id: str,
        update_data: Union[UserUpdate, Dict[str, Any]]
    ) -> Optional[UserInDB]:
        """
        Update user information.
        
        Args:
            user_id: The ID of the user to update
            update_data: The data to update
            
        Returns:
            Updated user or None if not found
            
        Raises:
            HTTPException: If updating to an email that already exists for another user
        """
        # Convert to dict if it's a UserUpdate
        if isinstance(update_data, UserUpdate):
            update_dict = update_data.dict(exclude_unset=True)
        else:
            update_dict = update_data
        
        # Check if email is being updated and if it's already taken
        if "email" in update_dict and update_dict["email"]:
            email_user = await self.get_user_by_email(update_dict["email"])
            if email_user and str(email_user.id) != user_id:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered to another user"
                )
        
        # Update the user
        updated = await self.user_repository.update(
            id=user_id,
            update={"$set": update_dict}
        )
        
        if not updated:
            return None
            
        # Return the updated user
        return await self.get_user_by_id(user_id)
    
    async def update_subscription(
        self,
        user_id: str,
        status: str,
        trial_ends_at: Optional[datetime] = None,
        subscription_ends_at: Optional[datetime] = None
    ) -> Optional[UserInDB]:
        """
        Update user subscription status.
        
        Args:
            user_id: The ID of the user to update
            status: Subscription status (free, trial, vip)
            trial_ends_at: Trial end date
            subscription_ends_at: Subscription end date
            
        Returns:
            Updated user or None if not found
            
        Raises:
            HTTPException: If status is invalid
        """
        # Validate status
        valid_statuses = ["free", "trial", "vip"]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Status must be one of {valid_statuses}"
            )
        
        # Update subscription
        updated = await self.user_repository.update_subscription(
            user_id=user_id,
            status=status,
            trial_ends_at=trial_ends_at,
            subscription_ends_at=subscription_ends_at
        )
        
        if not updated:
            return None
            
        # Return updated user
        return await self.get_user_by_id(user_id)
    
    async def start_free_trial(
        self,
        user_id: str,
        days: int = 3
    ) -> Optional[UserInDB]:
        """
        Start a free trial for a user.
        
        Args:
            user_id: The ID of the user
            days: Number of trial days
            
        Returns:
            Updated user or None if not found
            
        Raises:
            HTTPException: If user already has an active subscription or trial
        """
        # Get user to check current subscription
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
            
        # Check if user is eligible for a trial
        if user.subscription.status != "free":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User already has an active subscription or trial"
            )
            
        # Calculate trial end date
        trial_end = datetime.utcnow() + timedelta(days=days)
        
        # Update to trial status
        return await self.update_subscription(
            user_id=user_id,
            status="trial",
            trial_ends_at=trial_end
        )
    
    async def activate_vip(
        self,
        user_id: str,
        months: int = 1
    ) -> Optional[UserInDB]:
        """
        Activate VIP subscription for a user.
        
        Args:
            user_id: The ID of the user
            months: Number of subscription months
            
        Returns:
            Updated user or None if not found
        """
        # Calculate subscription end date
        subscription_end = datetime.utcnow() + timedelta(days=30 * months)
        
        # Update to VIP status
        return await self.update_subscription(
            user_id=user_id,
            status="vip",
            subscription_ends_at=subscription_end
        )
    
    async def add_betslip_purchase(
        self,
        user_id: str,
        date: str
    ) -> Optional[UserInDB]:
        """
        Add a betslip purchase for a user.
        
        Args:
            user_id: The ID of the user
            date: Betslip date (YYYY-MM-DD)
            
        Returns:
            Updated user or None if not found
            
        Raises:
            HTTPException: If date format is invalid
        """
        # Calculate expiry (end of the day after purchase date)
        try:
            purchase_date = datetime.strptime(date, "%Y-%m-%d")
            expires_at = datetime(
                purchase_date.year,
                purchase_date.month,
                purchase_date.day,
                23, 59, 59
            ) + timedelta(days=1)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Use YYYY-MM-DD."
            )
        
        # Add purchase
        updated = await self.user_repository.add_betslip_purchase(
            user_id=user_id,
            date=date,
            expires_at=expires_at
        )
        
        if not updated:
            return None
            
        # Return updated user
        return await self.get_user_by_id(user_id)
    
    async def update_settings(
        self,
        user_id: str,
        settings: Union[UserSettings, Dict[str, Any]]
    ) -> Optional[UserInDB]:
        """
        Update user settings.
        
        Args:
            user_id: The ID of the user
            settings: Settings to update
            
        Returns:
            Updated user or None if not found
        """
        # Convert to dict if it's a UserSettings
        if isinstance(settings, UserSettings):
            settings_dict = settings.dict()
        else:
            settings_dict = settings
        
        # Update settings
        updated = await self.user_repository.update_settings(
            user_id=user_id,
            settings=settings_dict
        )
        
        if not updated:
            return None
            
        # Return updated user
        return await self.get_user_by_id(user_id)
    
    async def check_access_to_betslip(
        self,
        user_id: str,
        date: str
    ) -> bool:
        """
        Check if a user has access to a betslip for a specific date.
        
        Args:
            user_id: The ID of the user
            date: Betslip date (YYYY-MM-DD)
            
        Returns:
            True if user has access, False otherwise
        """
        # Get user
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
            
        # Check subscription status
        if user.subscription.status == "vip":
            if user.subscription.subscription_ends_at and user.subscription.subscription_ends_at > datetime.utcnow():
                return True
            
        if user.subscription.status == "trial":
            if user.subscription.trial_ends_at and user.subscription.trial_ends_at > datetime.utcnow():
                return True
                
        # Check individual betslip purchases
        for purchase in user.betslip_purchases:
            if purchase.date == date and purchase.expires_at > datetime.utcnow():
                return True
                
        return False
    
    async def find_users_by_criteria(
        self,
        subscription_status: Optional[str] = None,
        limit: int = 100,
        skip: int = 0
    ) -> List[UserInDB]:
        """
        Find users by criteria.
        
        Args:
            subscription_status: Optional subscription status filter
            limit: Maximum number of users to return
            skip: Number of users to skip
            
        Returns:
            List of matching users
        """
        # Build filter
        filter = {}
        if subscription_status:
            filter["subscription.status"] = subscription_status
        
        # Get users
        return await self.user_repository.find(
            filter=filter,
            skip=skip,
            limit=limit
        )
    
    async def cleanup_expired_subscriptions(self) -> int:
        """
        Cleanup expired subscriptions and trials.
        
        Returns:
            Number of subscriptions cleaned up
        """
        return await self.user_repository.cleanup_expired_subscriptions()