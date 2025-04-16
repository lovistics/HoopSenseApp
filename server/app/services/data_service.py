"""
Data service for coordinating data fetching and updates.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from fastapi import HTTPException, status

from app.core.logger import logger
from app.db.mongodb import get_db, get_collection, check_db_connection


class DataService:
    """Service for data coordination between collectors and database."""
    
    @staticmethod
    async def check_data_freshness(collection_name: str, days: int = 1) -> bool:
        """
        Check if data in a collection is fresh (updated within specified days).
        
        Args:
            collection_name: The name of the collection to check
            days: Number of days to consider fresh
            
        Returns:
            True if data is fresh, False if outdated or not present
        """
        collection = get_collection(collection_name)
        
        # Get the most recently updated document
        most_recent = await collection.find_one(
            {},
            sort=[("updated_at", -1)],
            projection={"updated_at": 1}
        )
        
        if not most_recent or "updated_at" not in most_recent:
            return False
            
        # Check if the most recent update is within the freshness period
        freshness_threshold = datetime.utcnow() - timedelta(days=days)
        return most_recent["updated_at"] > freshness_threshold
    
    @staticmethod
    async def record_data_update(
        collection_name: str,
        update_type: str,
        items_updated: int,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a data update event.
        
        Args:
            collection_name: The name of the collection updated
            update_type: The type of update (e.g., "full", "incremental")
            items_updated: Number of items updated
            error: Optional error message
            
        Returns:
            Created log entry
        """
        update_log_collection = get_collection("data_update_logs")
        
        log_entry = {
            "collection_name": collection_name,
            "update_type": update_type,
            "items_updated": items_updated,
            "error": error,
            "timestamp": datetime.utcnow()
        }
        
        result = await update_log_collection.insert_one(log_entry)
        
        return {
            "id": str(result.inserted_id),
            **log_entry
        }
    
    @staticmethod
    async def get_update_logs(
        collection_name: Optional[str] = None, 
        days: int = 7,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get data update logs.
        
        Args:
            collection_name: Optional collection name filter
            days: Number of days to look back
            limit: Maximum number of logs to return
            
        Returns:
            List of log entries
        """
        update_log_collection = get_collection("data_update_logs")
        
        # Build query
        query = {}
        if collection_name:
            query["collection_name"] = collection_name
            
        # Set date filter
        date_threshold = datetime.utcnow() - timedelta(days=days)
        query["timestamp"] = {"$gte": date_threshold}
        
        # Get logs
        logs = await update_log_collection.find(
            query,
            sort=[("timestamp", -1)]
        ).limit(limit).to_list(None)
        
        # Convert ObjectId to string for JSON serialization
        for log in logs:
            log["_id"] = str(log["_id"])
            
        return logs
    
    @staticmethod
    async def get_collections_status() -> List[Dict[str, Any]]:
        """
        Get the status of all data collections.
        
        Returns:
            List of collection status information
        """
        # List of collections to check
        collections = [
            "seasons", "countries", "leagues", "teams", "players",
            "games", "team_statistics", "player_statistics", "standings",
            "odds", "features", "prediction_models"
        ]
        
        status = []
        
        for collection_name in collections:
            collection = get_collection(collection_name)
            
            # Count documents
            count = await collection.count_documents({})
            
            # Get last update
            last_updated = await collection.find_one(
                {},
                sort=[("updated_at", -1)],
                projection={"updated_at": 1}
            )
            
            # Determine status
            last_update_date = last_updated.get("updated_at") if last_updated else None
            is_fresh = False
            
            if last_update_date:
                freshness_threshold = datetime.utcnow() - timedelta(days=1)
                is_fresh = last_update_date > freshness_threshold
            
            status.append({
                "collection": collection_name,
                "document_count": count,
                "last_updated": last_update_date,
                "is_fresh": is_fresh
            })
            
        return status
    
    @staticmethod
    async def trigger_data_refresh(collection_name: str) -> Dict[str, Any]:
        """
        Trigger a data refresh for a specific collection.
        Note: This is a stub that would be implemented to call the appropriate collector.
        
        Args:
            collection_name: The name of the collection to refresh
            
        Returns:
            Status information
        """
        # This would delegate to the appropriate data collector
        # For now, we'll just log the request
        logger.info(f"Data refresh requested for collection: {collection_name}")
        
        return {
            "collection": collection_name,
            "status": "refresh_requested",
            "timestamp": datetime.utcnow()
        }
    
    @staticmethod
    async def check_database_health() -> Dict[str, Any]:
        """
        Check the health of the database connection.
        
        Returns:
            Health status information
        """
        # Check connection
        is_connected = await check_db_connection()
        
        return {
            "status": "healthy" if is_connected else "unhealthy",
            "message": "Database connection is active" if is_connected else "Database connection is not available",
            "timestamp": datetime.utcnow()
        }