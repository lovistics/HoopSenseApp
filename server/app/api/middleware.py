"""
Middleware for the FastAPI application.
"""
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
from typing import Callable

from app.core.config import settings
from app.core.logger import logger


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests with timing information.
    """
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """
        Process a request and log information.
        
        Args:
            request: Request object
            call_next: Next middleware function
            
        Returns:
            Response object
        """
        # Generate a unique request ID
        request_id = str(uuid.uuid4())
        
        # Start timer
        start_time = time.time()
        
        # Add request ID to request state for logging
        request.state.request_id = request_id
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Log the request
        logger.info(
            f"Request started | {request_id} | {client_ip} | {request.method} {request.url.path}"
        )
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log the response
            logger.info(
                f"Request completed | {request_id} | {request.method} {request.url.path} | "
                f"Status: {response.status_code} | Time: {process_time:.3f}s"
            )
            
            # Add custom headers with processing time and request ID
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            return response
        except Exception as e:
            # Log exceptions
            process_time = time.time() - start_time
            logger.error(
                f"Request failed | {request_id} | {request.method} {request.url.path} | "
                f"Error: {str(e)} | Time: {process_time:.3f}s"
            )
            raise


class CacheControlMiddleware(BaseHTTPMiddleware):
    """
    Middleware to set Cache-Control headers based on the endpoint.
    """
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """
        Process a request and set cache control headers.
        
        Args:
            request: Request object
            call_next: Next middleware function
            
        Returns:
            Response object
        """
        response = await call_next(request)
        path = request.url.path
        
        # Set Cache-Control header based on endpoint
        if path.startswith("/api/static"):
            # Static assets can be cached longer
            response.headers["Cache-Control"] = "public, max-age=86400"
        elif request.method in ["GET", "HEAD"] and not path.startswith("/api/v1/auth"):
            # Regular GET requests can be cached briefly
            response.headers["Cache-Control"] = "public, max-age=60"
        else:
            # Other requests shouldn't be cached
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        
        return response


def setup_middlewares(app: FastAPI) -> None:
    """
    Configure middleware for the application.
    
    Args:
        app: FastAPI application
    """
    # Add CORS middleware (if not already added in main.py)
    if not any(isinstance(middleware, CORSMiddleware) for middleware in app.user_middleware):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.BACKEND_CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add request logger middleware
    app.add_middleware(RequestLoggerMiddleware)
    
    # Add cache control middleware
    app.add_middleware(CacheControlMiddleware)