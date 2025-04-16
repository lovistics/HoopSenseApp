"""
API routes package initializer.
"""
from fastapi import FastAPI, APIRouter

from app.core.config import settings
from app.api.routes import games, predictions, hoopsiq, users, auth


def setup_routes(app: FastAPI) -> None:
    """
    Set up all API routes.
    
    Args:
        app: FastAPI application
    """
    # Create main API router
    api_router = APIRouter(prefix=settings.API_V1_STR)
    
    # Include all route modules
    api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
    api_router.include_router(users.router, prefix="/users", tags=["Users"])
    api_router.include_router(games.router, prefix="/games", tags=["Games"])
    api_router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
    api_router.include_router(hoopsiq.router, prefix="/hoopsiq", tags=["HoopsIQ"])
    
    # Include the main API router in the app
    app.include_router(api_router)