"""
Base model utilities for MongoDB integration.
"""
from datetime import datetime
from typing import Any, Dict, Optional, TypeVar, Type, cast

from bson import ObjectId
from pydantic import BaseModel, Field, validator


class PyObjectId(ObjectId):
    """Custom ObjectId type for proper serialization and validation."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


class MongoBaseModel(BaseModel):
    """
    Base model for MongoDB documents with consistent ID handling.
    """
    id: Optional[PyObjectId] = Field(alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str,
            PyObjectId: str
        }
    
    def dict_for_db(self, **kwargs) -> Dict[str, Any]:
        """
        Convert model to dict for MongoDB storage.
        Excludes 'id' field and uses '_id' for new documents where needed.
        """
        # Start with normal dict conversion
        dict_data = self.dict(by_alias=True, exclude={"id"}, **kwargs)
        
        # If this is an update to existing document, remove _id
        if "_id" in dict_data and dict_data["_id"] is None:
            dict_data.pop("_id", None)
        
        # Update timestamps
        dict_data["updated_at"] = datetime.utcnow()
        
        return dict_data


T = TypeVar('T', bound=MongoBaseModel)


def create_model_from_dict(model_class: Type[T], data: Dict[str, Any]) -> T:
    """
    Create a model instance from a dictionary with proper _id handling.
    
    Args:
        model_class: Pydantic model class
        data: Dictionary data from MongoDB
        
    Returns:
        Instance of model_class
    """
    # Handle the case where MongoDB returns _id as ObjectId
    if data and "_id" in data and isinstance(data["_id"], ObjectId):
        data["_id"] = str(data["_id"])
    
    return model_class(**data)