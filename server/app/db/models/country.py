"""
Country model for handling country data.
"""
from typing import Optional

from pydantic import Field, validator

from app.db.models.base import MongoBaseModel, PyObjectId


class CountryModel(MongoBaseModel):
    """Country model representing a country."""
    
    external_id: int
    name: str
    code: str
    flag_url: Optional[str] = None
    
    @validator('code')
    def validate_code(cls, v):
        """Validate country code."""
        if v and len(v) > 3:
            raise ValueError("Country code should be max 3 characters")
        return v.upper() if v else v
    
    @validator('name')
    def validate_name(cls, v):
        """Validate country name is not empty."""
        if not v.strip():
            raise ValueError("Country name cannot be empty")
        return v.strip()


class CountryInDB(CountryModel):
    """Country model as stored in the database."""
    pass


# Collection name in MongoDB
COLLECTION_NAME = "countries"


def get_country_dict(country: CountryModel) -> dict:
    """
    Convert country model to a dictionary for MongoDB storage.
    
    Args:
        country: Country model
        
    Returns:
        Dictionary representation for database
    """
    return country.dict_for_db()