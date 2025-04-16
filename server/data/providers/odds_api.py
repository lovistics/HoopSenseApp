"""
Provider for sports betting odds from the Odds API.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from app.core.config import settings
from app.core.logger import logger
from server.data.providers.provider_base import ProviderBase


class OddsAPI(ProviderBase):
    """
    Client for the Odds API service.
    Provides methods to fetch sports betting odds.
    """
    
    def __init__(self):
        """Initialize the Odds API client."""
        super().__init__(
            base_url=settings.API_ODDS_URL,
            api_key=settings.API_ODDS_KEY,
        )
    
    async def get_sports(self) -> List[Dict[str, Any]]:
        """
        Get all available sports.
        
        Returns:
            List of sports
        """
        params = {"apiKey": self.api_key}
        return await self._make_request("/sports", params=params)
    
    async def get_odds(
        self,
        sport_key: str = "basketball_nba",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        date_format: str = "iso",
        odds_format: str = "decimal",
        bookmakers: Optional[str] = None,
        event_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get odds for a sport.
        
        Args:
            sport_key: The sport key (e.g., "basketball_nba")
            regions: Comma-separated list of regions (e.g., "us,eu")
            markets: Comma-separated list of markets (e.g., "h2h,spreads")
            date_format: Date format ("iso" or "unix")
            odds_format: Odds format ("decimal", "american", etc.)
            bookmakers: Optional comma-separated list of bookmakers
            event_ids: Optional list of event IDs to filter by
            
        Returns:
            List of games with odds
        """
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "dateFormat": date_format,
            "oddsFormat": odds_format
        }
        
        if bookmakers:
            params["bookmakers"] = bookmakers
            
        if event_ids:
            params["eventIds"] = ",".join(event_ids)
        
        endpoint = f"/sports/{sport_key}/odds"
        return await self._make_request(endpoint, params=params)
    
    async def get_event_odds(
        self,
        sport_key: str,
        event_id: str,
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        date_format: str = "iso",
        odds_format: str = "decimal",
        bookmakers: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get odds for a specific event.
        
        Args:
            sport_key: The sport key (e.g., "basketball_nba")
            event_id: The event ID
            regions: Comma-separated list of regions (e.g., "us,eu")
            markets: Comma-separated list of markets (e.g., "h2h,spreads")
            date_format: Date format ("iso" or "unix")
            odds_format: Odds format ("decimal", "american", etc.)
            bookmakers: Optional comma-separated list of bookmakers
            
        Returns:
            Event with odds
        """
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "dateFormat": date_format,
            "oddsFormat": odds_format
        }
        
        if bookmakers:
            params["bookmakers"] = bookmakers
        
        endpoint = f"/sports/{sport_key}/events/{event_id}/odds"
        return await self._make_request(endpoint, params=params)
    
    async def get_historical_odds(
        self,
        sport_key: str = "basketball_nba",
        date: Optional[str] = None,
        regions: str = "us",
        markets: str = "h2h",
        date_format: str = "iso",
        odds_format: str = "decimal"
    ) -> List[Dict[str, Any]]:
        """
        Get historical odds for a sport.
        
        Args:
            sport_key: The sport key (e.g., "basketball_nba")
            date: Date string (YYYY-MM-DD)
            regions: Comma-separated list of regions (e.g., "us,eu")
            markets: Comma-separated list of markets (e.g., "h2h,spreads")
            date_format: Date format ("iso" or "unix")
            odds_format: Odds format ("decimal", "american", etc.)
            
        Returns:
            List of historical odds
        """
        if not date:
            # Default to yesterday
            date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "date": date,
            "dateFormat": date_format,
            "oddsFormat": odds_format
        }
        
        endpoint = f"/sports/{sport_key}/odds-history"
        return await self._make_request(endpoint, params=params)
    
    async def get_scores(
        self,
        sport_key: str = "basketball_nba",
        date_format: str = "iso",
        desde: Optional[str] = None,
        hasta: Optional[str] = None,
        event_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get scores for a sport.
        
        Args:
            sport_key: The sport key (e.g., "basketball_nba")
            date_format: Date format ("iso" or "unix")
            desde: Start date (YYYY-MM-DD) for filtering
            hasta: End date (YYYY-MM-DD) for filtering
            event_ids: Optional list of event IDs to filter by
            
        Returns:
            List of scores
        """
        params = {
            "apiKey": self.api_key,
            "dateFormat": date_format
        }
        
        if desde:
            params["desde"] = desde
        if hasta:
            params["hasta"] = hasta
        if event_ids:
            params["eventIds"] = ",".join(event_ids)
        
        endpoint = f"/sports/{sport_key}/scores"
        return await self._make_request(endpoint, params=params)