"""
Repository for user data operations.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from bson import ObjectId

from app.core.logger import logger
from app.db.models.user import UserInDB, COLLECTION_NAME
from server.data.repositories.base_repository import BaseRepository


class UserRepository(BaseRepository[UserInDB]):
    """
    Repository for user-related database operations.
    """
    
    def __init__(self):
        """Initialize the user repository."""
        super().__init__(COLLECTION_NAME, UserInDB)
    
    async def find_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Find a user by email.
        
        Args:
            email: User email
            
        Returns:
            User or None if not found
        """
        return await self.find_one({"email": email})
    
    async def find_with_expired_subscriptions(self) -> List[UserInDB]:
        """
        Find users with expired subscriptions.
        
        Returns:
            List of users with expired subscriptions
        """
        now = datetime.utcnow()
        
        # Find users with expired VIP subscriptions
        vip_users = await self.find({
            "subscription.status": "vip",
            "subscription.subscription_ends_at": {"$lt": now}
        })
        
        # Find users with expired trials
        trial_users = await self.find({
            "subscription.status": "trial",
            "subscription.trial_ends_at": {"$lt": now}
        })
        
        return vip_users + trial_users
    
    async def find_users_by_subscription_status(
        self,
        status: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[UserInDB]:
        """
        Find users by subscription status.
        
        Args:
            status: Subscription status (free, trial, vip)
            skip: Number of users to skip
            limit: Maximum number of users to return
            
        Returns:
            List of matching users
        """
        return await self.find(
            filter={"subscription.status": status},
            skip=skip,
            limit=limit
        )
    
    async def find_with_betslip_access(
        self,
        date: str
    ) -> List[UserInDB]:
        """
        Find users with access to a specific betslip date.
        
        Args:
            date: Betslip date (YYYY-MM-DD)
            
        Returns:
            List of users with access
        """
        now = datetime.utcnow()
        
        # Find VIP users
        vip_users = await self.find({
            "subscription.status": "vip",
            "subscription.subscription_ends_at": {"$gt": now}
        })
        
        # Find trial users
        trial_users = await self.find({
            "subscription.status": "trial",
            "subscription.trial_ends_at": {"$gt": now}
        })
        
        # Find users with specific betslip purchase
        purchase_users = await self.find({
            "betslip_purchases.date": date,
            "betslip_purchases.expires_at": {"$gt": now}
        })
        
        # Combine and deduplicate users
        user_ids = set()
        unique_users = []
        
        for user in vip_users + trial_users + purchase_users:
            if str(user.id) not in user_ids:
                user_ids.add(str(user.id))
                unique_users.append(user)
        
        return unique_users
    
    async def update_last_login(self, user_id: str) -> bool:
        """
        Update a user's last login time.
        
        Args:
            user_id: User ID
            
        Returns:
            True if update successful, False otherwise
        """
        return await self.update(
            id=user_id,
            update={
                "$set": {
                    "last_login": datetime.utcnow()
                }
            }
        )
    
    async def update_subscription(
        self,
        user_id: str,
        status: str,
        trial_ends_at: Optional[datetime] = None,
        subscription_ends_at: Optional[datetime] = None
    ) -> bool:
        """
        Update a user's subscription.
        
        Args:
            user_id: User ID
            status: Subscription status (free, trial, vip)
            trial_ends_at: Trial end date
            subscription_ends_at: Subscription end date
            
        Returns:
            True if update successful, False otherwise
        """
        return await self.update(
            id=user_id,
            update={
                "$set": {
                    "subscription.status": status,
                    "subscription.trial_ends_at": trial_ends_at,
                    "subscription.subscription_ends_at": subscription_ends_at
                }
            }
        )
    
    async def add_betslip_purchase(
        self,
        user_id: str,
        date: str,
        expires_at: datetime
    ) -> bool:
        """
        Add a betslip purchase for a user.
        
        Args:
            user_id: User ID
            date: Betslip date (YYYY-MM-DD)
            expires_at: Expiration date
            
        Returns:
            True if update successful, False otherwise
        """
        return await self.update(
            id=user_id,
            update={
                "$push": {
                    "betslip_purchases": {
                        "date": date,
                        "expires_at": expires_at
                    }
                }
            }
        )
    
    async def update_settings(
        self,
        user_id: str,
        settings: Dict[str, Any]
    ) -> bool:
        """
        Update a user's settings.
        
        Args:
            user_id: User ID
            settings: Settings to update
            
        Returns:
            True if update successful, False otherwise
        """
        return await self.update(
            id=user_id,
            update={
                "$set": {
                    "settings": settings
                }
            }
        )
    
    async def cleanup_expired_subscriptions(self) -> int:
        """
        Reset expired subscriptions to free status.
        
        Returns:
            Number of subscriptions cleaned up
        """
        now = datetime.utcnow()
        
        # Reset expired VIP subscriptions
        vip_result = await self.update_by_filter(
            filter={
                "subscription.status": "vip",
                "subscription.subscription_ends_at": {"$lt": now}
            },
            update={
                "$set": {
                    "subscription.status": "free",
                    "subscription.subscription_ends_at": None
                }
            }
        )
        
        # Reset expired trials
        trial_result = await self.update_by_filter(
            filter={
                "subscription.status": "trial",
                "subscription.trial_ends_at": {"$lt": now}
            },
            update={
                "$set": {
                    "subscription.status": "free",
                    "subscription.trial_ends_at": None
                }
            }
        )
        
        return vip_result + trial_result