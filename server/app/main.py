"""
FastAPI application entry point for HoopSense backend.
"""
import asyncio
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

from app.core.config import settings
from app.core.logger import logger
from app.db.mongodb import connect_to_mongo, close_mongo_connection, ensure_indexes
from app.api.middleware import setup_middlewares
from app.api.routes import setup_routes


# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up custom middleware
setup_middlewares(app)

# Set up API routes
setup_routes(app)


@app.on_event("startup")
async def startup_event():
    """Execute startup tasks."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    
    # Connect to MongoDB
    connected = await connect_to_mongo()
    if not connected:
        logger.critical("Failed to connect to MongoDB. Application may not function correctly.")
    
    # Create database indexes
    await ensure_indexes()


@app.on_event("shutdown")
async def shutdown_event():
    """Execute shutdown tasks."""
    logger.info(f"Shutting down {settings.PROJECT_NAME}")
    await close_mongo_connection()


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": settings.VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.FAST_API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )