"""
Database connection module for MongoDB.
Handles connecting to MongoDB and provides db instance.
"""
import asyncio
import logging
import time
from typing import Dict, Optional, Any, Tuple, Callable, Union, TypeVar

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

from app.core.config import settings
from app.core.logger import logger

# Async MongoDB client for API operations
async_client: Optional[AsyncIOMotorClient] = None
db: Optional[AsyncIOMotorDatabase] = None

# Sync MongoDB client for scripts and utilities
sync_client: Optional[MongoClient] = None
sync_db: Optional[Any] = None

# Connection state tracking
_is_connected = False
_last_health_check = 0
_health_check_interval = 60  # seconds
_connection_attempts = 0
_max_connection_attempts = 5
_reconnect_delay = 1  # starting delay in seconds (will be increased with exponential backoff)


def _get_connection_params() -> Dict[str, Any]:
    """
    Get common MongoDB connection parameters.
    
    Returns:
        Dictionary of connection parameters
    """
    return {
        "serverSelectionTimeoutMS": 5000,
        "maxPoolSize": settings.MONGODB_MAX_CONNECTIONS,
        "minPoolSize": settings.MONGODB_MIN_CONNECTIONS,
        "waitQueueTimeoutMS": 10000,
        "connectTimeoutMS": 10000,
        "socketTimeoutMS": 30000,
        "retryWrites": True,
        "retryReads": True,
        "w": "majority"  # Ensure write durability
    }


async def connect_to_mongo() -> bool:
    """
    Create async database connection with retry logic.
    
    Returns:
        True if connection successful, False otherwise
    """
    global async_client, db, _is_connected, _connection_attempts, _reconnect_delay
    
    if _is_connected:
        return True
    
    try:
        # Configure client with reasonable defaults for a production system
        async_client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            **_get_connection_params()
        )
        db = async_client[settings.MONGODB_DB_NAME]
        
        # Verify connection
        await async_client.admin.command('ping')
        
        logger.info("Connected to MongoDB (async)")
        _is_connected = True
        _connection_attempts = 0
        _reconnect_delay = 1
        return True
        
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        _connection_attempts += 1
        delay = min(60, _reconnect_delay * (2 ** (_connection_attempts - 1)))  # Exponential backoff with max 60s
        
        logger.error(f"MongoDB connection failed (attempt {_connection_attempts}): {str(e)}")
        logger.info(f"Retrying in {delay} seconds...")
        
        if _connection_attempts >= _max_connection_attempts:
            logger.critical(f"Failed to connect to MongoDB after {_max_connection_attempts} attempts")
            return False
            
        await asyncio.sleep(delay)
        return await connect_to_mongo()
        
    except Exception as e:
        logger.critical(f"Unexpected error connecting to MongoDB: {str(e)}")
        return False


def connect_to_mongo_sync() -> bool:
    """
    Create synchronous database connection for scripts with retry logic.
    
    Returns:
        True if connection successful, False otherwise
    """
    global sync_client, sync_db, _connection_attempts, _reconnect_delay
    
    try:
        # Configure client with reasonable defaults
        sync_client = MongoClient(
            settings.MONGODB_URL,
            **_get_connection_params()
        )
        sync_db = sync_client[settings.MONGODB_DB_NAME]
        
        # Verify connection
        sync_client.admin.command('ping')
        
        logger.info("Connected to MongoDB (sync)")
        _connection_attempts = 0
        _reconnect_delay = 1
        return True
        
    except (ServerSelectionTimeoutError, ConnectionFailure) as e:
        _connection_attempts += 1
        delay = min(60, _reconnect_delay * (2 ** (_connection_attempts - 1)))
        
        logger.error(f"MongoDB sync connection failed (attempt {_connection_attempts}): {str(e)}")
        logger.info(f"Retrying in {delay} seconds...")
        
        if _connection_attempts >= _max_connection_attempts:
            logger.critical(f"Failed to connect to MongoDB (sync) after {_max_connection_attempts} attempts")
            return False
            
        time.sleep(delay)
        return connect_to_mongo_sync()
        
    except Exception as e:
        logger.critical(f"Unexpected error connecting to MongoDB (sync): {str(e)}")
        return False


async def check_db_connection() -> bool:
    """
    Check if the database connection is healthy.
    Performs periodic health check to detect connection issues early.
    
    Returns:
        True if connection is healthy, False otherwise
    """
    global _is_connected, _last_health_check
    
    # Skip frequent checks
    now = time.time()
    if _is_connected and (now - _last_health_check) < _health_check_interval:
        return True
        
    _last_health_check = now
    
    if not db or not async_client:
        _is_connected = False
        logger.warning("MongoDB connection not initialized")
        return False
        
    try:
        # Simple health check
        await async_client.admin.command('ping')
        _is_connected = True
        return True
    except Exception as e:
        _is_connected = False
        logger.error(f"MongoDB health check failed: {str(e)}")
        return False


async def close_mongo_connection() -> None:
    """Close async database connection."""
    global async_client, _is_connected
    if async_client:
        async_client.close()
        async_client = None
        db = None
        _is_connected = False
        logger.info("Disconnected from MongoDB (async)")


def close_mongo_connection_sync() -> None:
    """Close sync database connection."""
    global sync_client
    if sync_client:
        sync_client.close()
        sync_client = None
        sync_db = None
        logger.info("Disconnected from MongoDB (sync)")


def get_collection(collection_name: str, sync: bool = False):
    """
    Get a MongoDB collection by name.
    
    Args:
        collection_name: Name of the collection
        sync: Whether to return sync or async collection
        
    Returns:
        MongoDB collection
    """
    if sync:
        if not sync_db:
            raise ValueError("Sync MongoDB connection not initialized")
        return sync_db[collection_name]
    
    if not db:
        raise ValueError("Async MongoDB connection not initialized")
    return db[collection_name]


def get_db(sync: bool = False):
    """
    Get the database instance.
    
    Args:
        sync: Whether to return sync or async database
        
    Returns:
        MongoDB database
    """
    if sync:
        if not sync_db:
            raise ValueError("Sync MongoDB connection not initialized")
        return sync_db
    
    if not db:
        raise ValueError("Async MongoDB connection not initialized")
    return db


async def ensure_indexes():
    """
    Create indexes for collections to optimize query performance.
    Should be called once during application startup.
    """
    if not db:
        logger.error("Cannot create indexes: MongoDB connection not initialized")
        return
    
    # Collection and index definitions with format: (collection, index_field, options)
    index_definitions = [
        # Users collection indexes
        ("users", "email", {"unique": True}),
        ("users", "subscription.status", {}),
        
        # Games collection indexes
        ("games", "external_id", {"unique": True}),
        ("games", "date", {}),
        ("games", [("home_team.team_id", 1), ("away_team.team_id", 1)], {}),
        ("games", "prediction.is_game_of_day", {}),
        ("games", "prediction.is_in_vip_betslip", {}),
        ("games", "status.short", {}),
        
        # Teams collection indexes
        ("teams", "external_id", {"unique": True}),
        ("teams", "name", {}),
        ("teams", "league_id", {}),
        
        # Players collection indexes
        ("players", "external_id", {"unique": True}),
        ("players", "team_id", {}),
        ("players", "name", {}),
        
        # Team statistics collection indexes
        ("team_statistics", [("team_id", 1), ("season_id", 1)], {"unique": True}),
        
        # Player statistics collection indexes
        ("player_statistics", [("player_id", 1), ("game_id", 1)], {"unique": True}),
        ("player_statistics", "game_id", {}),
        
        # Prediction history collection indexes
        ("prediction_history", [("user_id", 1), ("date", 1)], {"unique": True}),
    ]
    
    successful = 0
    failed = 0
    
    # Create each index with individual error handling
    for collection_name, index_field, options in index_definitions:
        try:
            collection = db[collection_name]
            await collection.create_index(index_field, **options)
            successful += 1
        except Exception as e:
            failed += 1
            logger.error(f"Failed to create index on {collection_name}.{index_field}: {str(e)}")
    
    if failed > 0:
        logger.warning(f"Index creation completed with issues: {successful} successful, {failed} failed")
    else:
        logger.info(f"Successfully created {successful} MongoDB indexes")