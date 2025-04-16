# server/data/providers/basketball_api.py
"""
Provider for basketball data from the API-Basketball service.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union

from app.core.config import settings
from app.core.logger import logger
from server.data.providers.provider_base import ProviderBase


class BasketballAPI(ProviderBase):
    """
    Client for the API-Basketball service.
    Provides methods to fetch basketball data.
    """
    
    def __init__(self):
        """Initialize the Basketball API client."""
        super().__init__(
            base_url=settings.API_BASKETBALL_URL,
            api_key=settings.API_BASKETBALL_KEY,
            headers={
                "X-RapidAPI-Key": settings.API_BASKETBALL_KEY,
                "X-RapidAPI-Host": settings.API_BASKETBALL_HOST
            }
        )
    
    async def get_leagues(self, country_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get basketball leagues.
        
        Args:
            country_id: Optional filter by country ID
            
        Returns:
            List of leagues
        """
        params = {}
        if country_id:
            params["country"] = country_id
        
        response = await self._make_request("/leagues", params=params)
        return response.get("response", [])
    
    async def get_league_seasons(self, league_id: int) -> List[str]:
        """
        Get seasons for a league.
        
        Args:
            league_id: The league ID
            
        Returns:
            List of seasons
        """
        params = {"league": league_id}
        response = await self._make_request("/leagues/seasons", params=params)
        return response.get("response", [])
    
    async def get_league_information(self, league_id: int, season: str) -> Dict[str, Any]:
        """
        Get detailed information about a league and season.
        
        Args:
            league_id: The league ID
            season: The season (e.g., "2023-2024")
            
        Returns:
            League information
        """
        params = {"id": league_id, "season": season}
        response = await self._make_request("/leagues", params=params)
        
        if response.get("results", 0) > 0:
            return response.get("response", [])[0]
        return {}
    
    async def get_countries(self) -> List[Dict[str, Any]]:
        """
        Get all available countries.
        
        Returns:
            List of countries
        """
        response = await self._make_request("/countries")
        return response.get("response", [])
    
    async def get_teams(
        self, 
        league_id: int, 
        season: str
    ) -> List[Dict[str, Any]]:
        """
        Get teams in a league and season.
        
        Args:
            league_id: The league ID
            season: The season (e.g., "2023-2024")
            
        Returns:
            List of teams
        """
        params = {"league": league_id, "season": season}
        response = await self._make_request("/teams", params=params)
        return response.get("response", [])
    
    async def get_team_statistics(
        self, 
        team_id: int, 
        league_id: int, 
        season: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a team in a specific league and season.
        
        Args:
            team_id: The team ID
            league_id: The league ID
            season: The season (e.g., "2023-2024")
            
        Returns:
            Team statistics
        """
        params = {"team": team_id, "league": league_id, "season": season}
        response = await self._make_request("/teams/statistics", params=params)
        
        if response.get("results", 0) > 0:
            return response.get("response", {})
        return {}
    
    async def get_standings(
        self, 
        league_id: int, 
        season: str
    ) -> List[Dict[str, Any]]:
        """
        Get standings for a league and season.
        
        Args:
            league_id: The league ID
            season: The season (e.g., "2023-2024")
            
        Returns:
            List of standings groups
        """
        params = {"league": league_id, "season": season}
        response = await self._make_request("/standings", params=params)
        return response.get("response", [])
    
    async def get_games(
        self, 
        league_id: Optional[int] = None, 
        season: Optional[str] = None, 
        date: Optional[str] = None,
        team_id: Optional[int] = None,
        game_id: Optional[int] = None,
        status: Optional[str] = None,
        timezone: str = "America/New_York"
    ) -> List[Dict[str, Any]]:
        """
        Get games for a league and season.
        
        Args:
            league_id: Optional league ID
            season: Optional season (e.g., "2023-2024")
            date: Optional date filter (YYYY-MM-DD)
            team_id: Optional team ID filter
            game_id: Optional game ID filter
            status: Optional status filter (NS, LIVE, FT, etc.)
            timezone: Timezone for game times
            
        Returns:
            List of games
        """
        params = {"timezone": timezone}
        
        # Add optional filters
        if league_id:
            params["league"] = league_id
        if season:
            params["season"] = season
        if date:
            params["date"] = date
        if team_id:
            params["team"] = team_id
        if game_id:
            params["id"] = game_id
        if status:
            params["status"] = status
        
        response = await self._make_request("/games", params=params)
        return response.get("response", [])
    
    async def get_game(
        self, 
        game_id: int, 
        timezone: str = "America/New_York"
    ) -> Dict[str, Any]:
        """
        Get details for a specific game.
        
        Args:
            game_id: The game ID
            timezone: Timezone for game times
            
        Returns:
            Game details
        """
        params = {"id": game_id, "timezone": timezone}
        response = await self._make_request("/games", params=params)
        
        if response.get("results", 0) > 0:
            return response.get("response", [])[0]
        return {}
    
    async def get_game_statistics(self, game_id: int) -> Dict[str, Any]:
        """
        Get statistics for a specific game.
        
        Args:
            game_id: The game ID
            
        Returns:
            Game statistics
        """
        params = {"game": game_id}
        response = await self._make_request("/games/statistics", params=params)
        
        if response.get("results", 0) > 0:
            return response.get("response", {})
        return {}
    
    async def get_players(
        self, 
        team_id: int, 
        season: str
    ) -> List[Dict[str, Any]]:
        """
        Get players for a team in a season.
        
        Args:
            team_id: The team ID
            season: The season (e.g., "2023-2024")
            
        Returns:
            List of players
        """
        params = {"team": team_id, "season": season}
        response = await self._make_request("/players", params=params)
        return response.get("response", [])
    
    async def get_player_statistics(
        self,
        player_id: int,
        season: str,
        league_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get statistics for a player in a season.
        
        Args:
            player_id: The player ID
            season: The season (e.g., "2023-2024")
            league_id: Optional league ID filter
            
        Returns:
            Player statistics
        """
        params = {"id": player_id, "season": season}
        
        if league_id:
            params["league"] = league_id
        
        response = await self._make_request("/players/statistics", params=params)
        
        if response.get("results", 0) > 0:
            return response.get("response", [])
        return []