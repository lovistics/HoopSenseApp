"""
Base repository providing common data access methods.
"""
from typing import Dict, Any, List, Optional, TypeVar, Generic, Union, Type
from datetime import datetime

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from pydantic import BaseModel

from app.core.logger import logger
from app.db.mongodb import get_collection


T = TypeVar('T', bound=BaseModel)


class BaseRepository(Generic[T]):
    """
    Base repository class with common CRUD operations.
    Generic type T is the model class returned by this repository.
    """
    
    def __init__(self, collection_name: str, model_class: Type[T]):
        """
        Initialize the repository.
        
        Args:
            collection_name: MongoDB collection name
            model_class: Pydantic model class for typecasting results
        """
        self.collection_name = collection_name
        self.model_class = model_class
        self._collection = None
    
    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Get the MongoDB collection."""
        if not self._collection:
            self._collection = get_collection(self.collection_name)
        return self._collection
    
    async def find_by_id(self, id: str) -> Optional[T]:
        """
        Find a document by ID.
        
        Args:
            id: Document ID
            
        Returns:
            Document model or None if not found
        """
        if not ObjectId.is_valid(id):
            return None
        
        try:
            document = await self.collection.find_one({"_id": ObjectId(id)})
            return self.model_class(**document) if document else None
        except Exception as e:
            logger.error(f"Error finding document by ID {id}: {str(e)}")
            return None
    
    async def find(
        self, 
        filter: Dict[str, Any], 
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[List[tuple]] = None,
        skip: int = 0,
        limit: Optional[int] = None
    ) -> List[T]:
        """
        Find documents matching a filter.
        
        Args:
            filter: Query filter
            projection: Fields to include/exclude
            sort: Sort specification
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List of matching document models
        """
        try:
            cursor = self.collection.find(filter, projection)
            
            if sort:
                cursor = cursor.sort(sort)
            
            cursor = cursor.skip(skip)
            
            if limit:
                cursor = cursor.limit(limit)
            
            documents = await cursor.to_list(None)
            return [self.model_class(**doc) for doc in documents]
        except Exception as e:
            logger.error(f"Error finding documents: {str(e)}")
            return []
    
    async def find_one(
        self, 
        filter: Dict[str, Any], 
        projection: Optional[Dict[str, Any]] = None
    ) -> Optional[T]:
        """
        Find a single document matching a filter.
        
        Args:
            filter: Query filter
            projection: Fields to include/exclude
            
        Returns:
            Matching document model or None
        """
        try:
            document = await self.collection.find_one(filter, projection)
            return self.model_class(**document) if document else None
        except Exception as e:
            logger.error(f"Error finding document: {str(e)}")
            return None
    
    async def count(self, filter: Dict[str, Any]) -> int:
        """
        Count documents matching a filter.
        
        Args:
            filter: Query filter
            
        Returns:
            Count of matching documents
        """
        try:
            return await self.collection.count_documents(filter)
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return 0
    
    async def create(self, document: Union[Dict[str, Any], BaseModel]) -> Optional[str]:
        """
        Create a new document.
        
        Args:
            document: Document to create as dict or Pydantic model
            
        Returns:
            ID of created document or None if creation fails
        """
        try:
            # Convert to dict if it's a BaseModel
            if isinstance(document, BaseModel):
                document_dict = document.dict(by_alias=True, exclude={"id"})
            else:
                document_dict = document
            
            # Add timestamps
            if "created_at" not in document_dict:
                document_dict["created_at"] = datetime.utcnow()
            if "updated_at" not in document_dict:
                document_dict["updated_at"] = datetime.utcnow()
                
            result = await self.collection.insert_one(document_dict)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            return None
    
    async def update(
        self, 
        id: str, 
        update: Dict[str, Any], 
        upsert: bool = False
    ) -> bool:
        """
        Update a document by ID.
        
        Args:
            id: Document ID
            update: Update operations
            upsert: Whether to insert if document doesn't exist
            
        Returns:
            True if update was successful, False otherwise
        """
        if not ObjectId.is_valid(id):
            return False
        
        try:
            # Add updated_at timestamp
            if "$set" in update:
                update["$set"]["updated_at"] = datetime.utcnow()
            else:
                update["$set"] = {"updated_at": datetime.utcnow()}
            
            result = await self.collection.update_one(
                {"_id": ObjectId(id)},
                update,
                upsert=upsert
            )
            
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
        except Exception as e:
            logger.error(f"Error updating document {id}: {str(e)}")
            return False
    
    async def update_by_filter(
        self, 
        filter: Dict[str, Any], 
        update: Dict[str, Any], 
        upsert: bool = False
    ) -> int:
        """
        Update documents matching a filter.
        
        Args:
            filter: Query filter
            update: Update operations
            upsert: Whether to insert if document doesn't exist
            
        Returns:
            Number of documents modified
        """
        try:
            # Add updated_at timestamp
            if "$set" in update:
                update["$set"]["updated_at"] = datetime.utcnow()
            else:
                update["$set"] = {"updated_at": datetime.utcnow()}
            
            result = await self.collection.update_many(
                filter,
                update,
                upsert=upsert
            )
            
            return result.modified_count
        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            return 0
    
    async def delete(self, id: str) -> bool:
        """
        Delete a document by ID.
        
        Args:
            id: Document ID
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not ObjectId.is_valid(id):
            return False
        
        try:
            result = await self.collection.delete_one({"_id": ObjectId(id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document {id}: {str(e)}")
            return False
    
    async def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """
        Delete documents matching a filter.
        
        Args:
            filter: Query filter
            
        Returns:
            Number of documents deleted
        """
        try:
            result = await self.collection.delete_many(filter)
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return 0
    
    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute an aggregation pipeline.
        
        Args:
            pipeline: Aggregation pipeline stages
            
        Returns:
            Aggregation results
        """
        try:
            result = await self.collection.aggregate(pipeline).to_list(None)
            return result
        except Exception as e:
            logger.error(f"Error executing aggregation: {str(e)}")
            return []