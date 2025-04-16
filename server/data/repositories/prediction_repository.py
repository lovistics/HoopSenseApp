"""
Repository for prediction data operations.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from bson import ObjectId

from app.core.logger import logger
from app.db.models.prediction import (
    PredictionHistory, PredictionHistoryInDB, PredictionModelSchema, 
    HISTORY_COLLECTION, MODEL_COLLECTION
)
from server.data.repositories.base_repository import BaseRepository


class PredictionModelRepository(BaseRepository[PredictionModelSchema]):
    """
    Repository for prediction model operations.
    """
    
    def __init__(self):
        """Initialize the prediction model repository."""
        super().__init__(MODEL_COLLECTION, PredictionModelSchema)
    
    async def find_active_model(self) -> Optional[PredictionModelSchema]:
        """
        Find the currently active prediction model.
        
        Returns:
            Active model or None if not found
        """
        return await self.find_one({"is_active": True})
    
    async def find_by_version(self, version: str) -> Optional[PredictionModelSchema]:
        """
        Find a model by version.
        
        Args:
            version: Model version string
            
        Returns:
            Model or None if not found
        """
        return await self.find_one({"version": version})
    
    async def activate_model(self, model_id: str) -> Tuple[int, bool]:
        """
        Activate a model and deactivate all others.
        
        Args:
            model_id: Model ID to activate
            
        Returns:
            Tuple of (deactivated count, activation success)
        """
        # Deactivate all models
        deactivate_count = await self.update_by_filter(
            filter={},
            update={"$set": {"is_active": False}}
        )
        
        # Activate the specified model
        activate_result = await self.update(
            id=model_id,
            update={"$set": {"is_active": True}}
        )
        
        return deactivate_count, activate_result


class PredictionHistoryRepository(BaseRepository[PredictionHistoryInDB]):
    """
    Repository for prediction history operations.
    """
    
    def __init__(self):
        """Initialize the prediction history repository."""
        super().__init__(HISTORY_COLLECTION, PredictionHistoryInDB)
    
    async def find_by_user_date(
        self,
        user_id: str,
        date: str
    ) -> Optional[PredictionHistoryInDB]:
        """
        Find prediction history for a user on a specific date.
        
        Args:
            user_id: User ID
            date: Date string (YYYY-MM-DD)
            
        Returns:
            Prediction history or None if not found
        """
        return await self.find_one({
            "user_id": user_id,
            "date": date
        })
    
    async def find_by_user_date_range(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PredictionHistoryInDB]:
        """
        Find prediction history for a user within a date range.
        
        Args:
            user_id: User ID
            start_date: Start date
            end_date: End date
            
        Returns:
            List of matching prediction history
        """
        # Convert dates to string format used in history (YYYY-MM-DD)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        return await self.find(
            filter={
                "user_id": user_id,
                "date": {"$gte": start_str, "$lte": end_str}
            },
            sort=[("date", -1)]
        )
    
    async def calculate_user_stats(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Calculate prediction accuracy statistics for a user.
        
        Args:
            user_id: User ID
            days: Number of days to consider
            
        Returns:
            Dictionary of statistics
        """
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get history within range
        history_records = await self.find_by_user_date_range(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Default stats
        stats = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "current_streak": 0,
            "best_streak": 0
        }
        
        if not history_records:
            return stats
        
        # Sort records by date ascending for streak calculation
        sorted_records = sorted(history_records, key=lambda x: x.date)
        
        # Calculate statistics
        total_predictions = 0
        correct_predictions = 0
        current_streak = 0
        best_streak = 0
        current_streak_active = True
        
        for record in sorted_records:
            if record.summary:
                total_predictions += record.summary.get("total", 0)
                correct_predictions += record.summary.get("correct", 0)
                
                # Calculate streak
                if record.summary.get("total", 0) > 0:
                    accuracy = record.summary.get("correct", 0) / record.summary.get("total", 0)
                    
                    # Streak is continued if accuracy is at least 50%
                    if accuracy >= 0.5 and current_streak_active:
                        current_streak += 1
                        best_streak = max(best_streak, current_streak)
                    else:
                        current_streak_active = False
                        current_streak = 0
        
        # Calculate overall accuracy
        accuracy = 0.0
        if total_predictions > 0:
            accuracy = round(correct_predictions / total_predictions * 100, 1)
        
        return {
            "total_predictions": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "current_streak": current_streak,
            "best_streak": best_streak
        }
    
    async def upsert_history(
        self,
        user_id: str,
        date: str,
        predictions: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create or update prediction history for a user on a date.
        
        Args:
            user_id: User ID
            date: Date string (YYYY-MM-DD)
            predictions: List of prediction items
            summary: Summary statistics
            
        Returns:
            ID of created/updated history or None if operation fails
        """
        # Check if history for this date already exists
        existing = await self.find_by_user_date(user_id, date)
        
        if existing:
            # Update existing record
            updated = await self.update(
                id=str(existing.id),
                update={
                    "$set": {
                        "predictions": predictions,
                        "summary": summary
                    }
                }
            )
            return str(existing.id) if updated else None
        else:
            # Create new record
            history = {
                "user_id": user_id,
                "date": date,
                "predictions": predictions,
                "summary": summary,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            return await self.create(history)