"""
Application configuration settings.
"""
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from pydantic import AnyHttpUrl, EmailStr, Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    """
    
    # Base
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "HoopSense Backend"
    DESCRIPTION: str = "Basketball prediction API with AI analysis"
    VERSION: str = "0.1.0"
    
    # Path settings
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    # CORS
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = ["http://localhost:3000", "http://localhost:8080"]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Parse CORS origins from string to list."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30  # 30 days
    
    # MongoDB
    MONGODB_URL: str = Field("mongodb://localhost:27017", env="MONGODB_URL")
    MONGODB_DB_NAME: str = Field("hoopsense", env="MONGODB_DB_NAME")
    MONGODB_MAX_CONNECTIONS: int = 10
    MONGODB_MIN_CONNECTIONS: int = 1
    
    # Basketball API
    API_BASKETBALL_KEY: str = Field(..., env="API_BASKETBALL_KEY")
    API_BASKETBALL_URL: str = "https://api-basketball.p.rapidapi.com"
    API_BASKETBALL_HOST: str = "api-basketball.p.rapidapi.com"
    
    # Odds API
    API_ODDS_KEY: str = Field(..., env="API_ODDS_KEY")
    API_ODDS_URL: str = "https://api.the-odds-api.com/v4"
    
    @validator("API_BASKETBALL_KEY", "API_ODDS_KEY")
    def validate_api_keys(cls, v, values, **kwargs):
        """Validate that API keys are provided when not in development mode."""
        field_name = kwargs["field"].name
        environment = values.get("ENVIRONMENT", "development")
        
        if environment != "development" and (not v or len(v) < 8):  # Simple check that key has reasonable length
            raise ValueError(f"{field_name} must be provided in {environment} environment")
        return v
    
    # FastAPI Config
    FAST_API_RELOAD: bool = True
    
    # Features
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # OAuth2
    OAUTH2_TOKEN_URL: str = "/api/v1/auth/token"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None  # Set to a path to enable file logging
    
    # Environment
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    
    # Email notifications
    SMTP_TLS: bool = True
    SMTP_PORT: Optional[int] = None
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_MAX_REQUESTS: int = 100  # Max requests per minute
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create settings instance
settings = Settings()