"""
Base class for data providers.
"""
import asyncio
import httpx
import time
import json
from typing import Dict, Any, Optional, List, Union, Callable

from app.core.config import settings
from app.core.logger import logger


class ProviderBase:
    """
    Base class for external data providers.
    Provides common functionality for API integrations.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        retry_count: int = 3,
        retry_delay: int = 2,
        max_concurrent_requests: int = 5
    ):
        """
        Initialize the provider with API settings.
        
        Args:
            base_url: The base URL for the API
            api_key: The API key for authentication
            headers: Additional headers to include in requests
            timeout: Request timeout in seconds
            retry_count: Number of retries for failed requests
            retry_delay: Initial delay between retries in seconds (will be increased with backoff)
            max_concurrent_requests: Maximum number of concurrent requests
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers or {}
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.max_concurrent_requests = max_concurrent_requests
        
        # Semaphore to limit concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Track rate limits
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
        self.rate_limit_alert_threshold = 10  # Alert when fewer than this many requests remain
    
    async def _make_request(
        self, 
        endpoint: str, 
        method: str = "GET", 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, Any]] = None,
        retries_left: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make an API request with retry and error handling.
        
        Args:
            endpoint: API endpoint to call
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            data: Form data for POST/PUT requests
            json_data: JSON data for POST/PUT requests
            custom_headers: Additional headers for this request
            retries_left: Number of retries left (used internally for recursion)
            
        Returns:
            API response data
            
        Raises:
            httpx.HTTPStatusError: If request failed after retries
            httpx.RequestError: If connection failed after retries
            ValueError: If method is unsupported
        """
        url = f"{self.base_url}{endpoint}"
        
        # Set default retries if not specified
        if retries_left is None:
            retries_left = self.retry_count
        
        # Combine default and custom headers
        headers = {**self.headers}
        if custom_headers:
            headers.update(custom_headers)
        
        # Check if we're approaching rate limit
        if self.rate_limit_remaining is not None and self.rate_limit_remaining <= self.rate_limit_alert_threshold:
            logger.warning(f"Rate limit running low: {self.rate_limit_remaining} requests remaining")
            
            # If we've hit the limit, wait for reset
            if self.rate_limit_remaining <= 0 and self.rate_limit_reset is not None:
                wait_time = max(0, self.rate_limit_reset - time.time())
                if wait_time > 0:
                    logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
        
        # Use semaphore to limit concurrent requests
        async with self.semaphore:
            try:
                # Log request (without sensitive data)
                safe_params = self._sanitize_params(params) if params else None
                logger.debug(f"API Request: {method} {url} | Params: {safe_params}")
                
                # Make the request
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method == "GET":
                        response = await client.get(url, params=params, headers=headers)
                    elif method == "POST":
                        response = await client.post(
                            url, 
                            params=params, 
                            data=data, 
                            json=json_data, 
                            headers=headers
                        )
                    elif method == "PUT":
                        response = await client.put(
                            url, 
                            params=params, 
                            data=data, 
                            json=json_data, 
                            headers=headers
                        )
                    elif method == "DELETE":
                        response = await client.delete(url, params=params, headers=headers)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Update rate limit tracking if headers are present
                self._update_rate_limits(response)
                
                # Check if the request was successful
                response.raise_for_status()
                
                # Parse and return response
                if response.headers.get("content-type", "").startswith("application/json"):
                    return response.json()
                return {"text": response.text}
                
            except httpx.HTTPStatusError as e:
                # Handle rate limiting
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", self.retry_delay))
                    logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                    await asyncio.sleep(retry_after)
                    # Retry without decrementing retries
                    return await self._make_request(
                        endpoint, method, params, data, json_data, custom_headers, retries_left
                    )
                
                # Handle authentication errors
                if e.response.status_code in (401, 403):
                    logger.error(f"Authentication error: {e.response.status_code} {e.response.reason_phrase}")
                    if retries_left > 0:
                        # For auth errors, wait longer before retry
                        await asyncio.sleep(self.retry_delay * 2)
                        return await self._make_request(
                            endpoint, method, params, data, json_data, custom_headers, retries_left - 1
                        )
                    raise
                
                # If it's a server error (5xx), retry
                if 500 <= e.response.status_code < 600:
                    if retries_left > 0:
                        wait_time = self.retry_delay * (2 ** (self.retry_count - retries_left))  # Exponential backoff
                        logger.warning(f"Server error: {e}. Retrying in {wait_time}s ({retries_left} retries left)")
                        await asyncio.sleep(wait_time)
                        return await self._make_request(
                            endpoint, method, params, data, json_data, custom_headers, retries_left - 1
                        )
                
                # Log the response content for debugging if available
                content = None
                try:
                    content = e.response.json()
                except:
                    try:
                        content = e.response.text
                    except:
                        pass
                
                logger.error(f"HTTP error {e.response.status_code}: {e} | Content: {content}")
                raise
                
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                # Network-related errors, retry
                if retries_left > 0:
                    wait_time = self.retry_delay * (2 ** (self.retry_count - retries_left))  # Exponential backoff
                    logger.warning(f"Request error: {e}. Retrying in {wait_time}s ({retries_left} retries left)")
                    await asyncio.sleep(wait_time)
                    return await self._make_request(
                        endpoint, method, params, data, json_data, custom_headers, retries_left - 1
                    )
                
                logger.error(f"Failed after {self.retry_count} retries: {e}")
                raise
            
            except Exception as e:
                logger.error(f"Unexpected error during API request to {url}: {e}", exc_info=True)
                raise
    
    def _update_rate_limits(self, response: httpx.Response) -> None:
        """
        Update rate limit tracking based on response headers.
        
        Args:
            response: API response with rate limit headers
        """
        # Check for common rate limit header patterns
        # X-RateLimit-* (GitHub, many others)
        if "X-RateLimit-Remaining" in response.headers:
            try:
                self.rate_limit_remaining = int(response.headers["X-RateLimit-Remaining"])
                
                if "X-RateLimit-Reset" in response.headers:
                    self.rate_limit_reset = int(response.headers["X-RateLimit-Reset"])
            except (ValueError, TypeError):
                pass
        
        # RateLimit-* (Standard from RFC)
        elif "RateLimit-Remaining" in response.headers:
            try:
                self.rate_limit_remaining = int(response.headers["RateLimit-Remaining"])
                
                if "RateLimit-Reset" in response.headers:
                    self.rate_limit_reset = int(response.headers["RateLimit-Reset"])
            except (ValueError, TypeError):
                pass
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive information from parameters for logging.
        
        Args:
            params: Query parameters
            
        Returns:
            Sanitized parameters for logging
        """
        if not params:
            return {}
            
        sanitized = params.copy()
        
        # List of sensitive parameter names to mask
        sensitive_params = ["key", "apiKey", "api_key", "secret", "password", "token", "auth"]
        
        for param in sensitive_params:
            if param in sanitized:
                sanitized[param] = "********"
        
        return sanitized
    
    async def paginate_all(
        self,
        endpoint: str,
        params: Dict[str, Any],
        result_key: str,
        page_param: str = "page",
        limit_param: Optional[str] = "per_page",
        total_items_key: Optional[str] = None,
        max_pages: Optional[int] = None,
        page_start: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Handle pagination and collect all results.
        
        Args:
            endpoint: API endpoint
            params: Base query parameters
            result_key: Key in the response that contains the results array
            page_param: Name of the page parameter
            limit_param: Name of the limit parameter
            total_items_key: Key that holds the total number of items (for optimization)
            max_pages: Maximum number of pages to retrieve
            page_start: Starting page number
            
        Returns:
            Combined list of all paginated results
        """
        all_results = []
        current_page = page_start
        
        # Make sure we have a reasonable page size if applicable
        if limit_param and limit_param in params:
            # If page size is too small, may cause too many requests
            if params[limit_param] < 10:
                logger.warning(f"Small page size ({params[limit_param]}) may cause excessive API requests")
        
        while True:
            # Update params with page number
            page_params = {**params, page_param: current_page}
            
            # Make the request
            try:
                response = await self._make_request(endpoint, params=page_params)
            
                # Extract and accumulate results
                if isinstance(response, dict) and result_key in response:
                    results = response.get(result_key, [])
                    
                    # If it's not a list, it might be a single item or invalid
                    if not isinstance(results, list):
                        logger.warning(f"Expected list for '{result_key}', got {type(results)}. Treating as single item.")
                        results = [results] if results else []
                    
                    all_results.extend(results)
                    
                    # Determine if we've reached the end
                    end_of_results = len(results) == 0
                    
                    # If we have total_items_key, we can check if we've retrieved all items
                    if total_items_key and total_items_key in response:
                        total_items = response[total_items_key]
                        retrieved_items = len(all_results)
                        if retrieved_items >= total_items:
                            logger.debug(f"Retrieved all {retrieved_items} items")
                            break
                    elif end_of_results:
                        logger.debug(f"Received empty page at page {current_page}")
                        break
                else:
                    logger.warning(f"Result key '{result_key}' not found in response: {response}")
                    break
            except Exception as e:
                logger.error(f"Error during pagination at page {current_page}: {e}")
                # Stop pagination on error
                break
            
            # Check if we've hit the max pages limit
            if max_pages and current_page >= page_start + max_pages - 1:
                logger.debug(f"Reached maximum page limit ({max_pages})")
                break
            
            # Move to next page
            current_page += 1
        
        return all_results