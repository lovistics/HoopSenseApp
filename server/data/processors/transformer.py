"""
Data transformation utilities for converting between API data and models.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
import re

from app.core.logger import logger


class DataTransformer:
    """
    Transforms data between different formats.
    """
    
    def parse_api_datetime(self, date_str: str) -> Optional[datetime]:
        """
        Parse API datetime string to datetime object.
        
        Args:
            date_str: Datetime string from API
            
        Returns:
            Datetime object or None if parsing fails
        """
        if not date_str:
            return None
            
        try:
            # Try parsing ISO format with timezone
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try parsing ISO format with timezone
                return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None)
            except ValueError:
                try:
                    # Try parsing ISO format without timezone
                    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    try:
                        # Try parsing date only
                        return datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        logger.error(f"Failed to parse date: {date_str}")
                        return None
    
    def format_datetime(self, dt: datetime, format_str: str = "%Y-%m-%d") -> str:
        """
        Format datetime object to string.
        
        Args:
            dt: Datetime object
            format_str: Format string
            
        Returns:
            Formatted date string
        """
        if not dt:
            return ""
        
        return dt.strftime(format_str)
    
    def convert_timestamp_to_datetime(self, timestamp: Union[int, str]) -> Optional[datetime]:
        """
        Convert Unix timestamp to datetime object.
        
        Args:
            timestamp: Unix timestamp in seconds
            
        Returns:
            Datetime object or None if conversion fails
        """
        if not timestamp:
            return None
        
        try:
            # Convert to integer if it's a string
            if isinstance(timestamp, str):
                timestamp = int(timestamp)
            
            # Convert to datetime
            return datetime.fromtimestamp(timestamp)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert timestamp {timestamp}: {str(e)}")
            return None
    
    def extract_team_abbreviation(self, name: str) -> str:
        """
        Extract or generate a team abbreviation from its name.
        
        Args:
            name: Full team name
            
        Returns:
            Team abbreviation (3 letters)
        """
        # Known abbreviations map
        abbr_map = {
            "Atlanta Hawks": "ATL",
            "Boston Celtics": "BOS",
            "Brooklyn Nets": "BKN",
            "Charlotte Hornets": "CHA",
            "Chicago Bulls": "CHI",
            "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL",
            "Denver Nuggets": "DEN",
            "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW",
            "Houston Rockets": "HOU",
            "Indiana Pacers": "IND",
            "Los Angeles Clippers": "LAC",
            "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL",
            "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP",
            "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL",
            "Philadelphia 76ers": "PHI",
            "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR",
            "Sacramento Kings": "SAC",
            "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR",
            "Utah Jazz": "UTA",
            "Washington Wizards": "WAS"
        }
        
        # Check if we have a known abbreviation
        if name in abbr_map:
            return abbr_map[name]
        
        # Generate abbreviation from name
        name = name.strip()
        words = name.split()
        
        if len(words) == 1:
            # Single word name - take first 3 letters
            return words[0][:3].upper()
        elif len(words) == 2:
            # Two word name - first letter of first word + first 2 letters of second word
            return (words[0][0] + words[1][:2]).upper()
        else:
            # Multi-word name - take first letter from first 3 words
            return ''.join(word[0] for word in words[:3]).upper()
    
    def get_team_colors(self, team_name: str) -> Tuple[str, str]:
        """
        Get primary and secondary colors for a team.
        
        Args:
            team_name: Team name
            
        Returns:
            Tuple of (primary_color, secondary_color)
        """
        # NBA team colors map (primary, secondary)
        colors_map = {
            "Atlanta Hawks": ("#E03A3E", "#C1D32F"),
            "Boston Celtics": ("#007A33", "#BA9653"),
            "Brooklyn Nets": ("#000000", "#FFFFFF"),
            "Charlotte Hornets": ("#1D1160", "#00788C"),
            "Chicago Bulls": ("#CE1141", "#000000"),
            "Cleveland Cavaliers": ("#860038", "#FDBB30"),
            "Dallas Mavericks": ("#00538C", "#B8C4CA"),
            "Denver Nuggets": ("#0E2240", "#FEC524"),
            "Detroit Pistons": ("#C8102E", "#1D42BA"),
            "Golden State Warriors": ("#1D428A", "#FFC72C"),
            "Houston Rockets": ("#CE1141", "#000000"),
            "Indiana Pacers": ("#002D62", "#FDBB30"),
            "Los Angeles Clippers": ("#C8102E", "#1D428A"),
            "Los Angeles Lakers": ("#552583", "#FDB927"),
            "Memphis Grizzlies": ("#5D76A9", "#12173F"),
            "Miami Heat": ("#98002E", "#F9A01B"),
            "Milwaukee Bucks": ("#00471B", "#EEE1C6"),
            "Minnesota Timberwolves": ("#0C2340", "#236192"),
            "New Orleans Pelicans": ("#0C2340", "#C8102E"),
            "New York Knicks": ("#006BB6", "#F58426"),
            "Oklahoma City Thunder": ("#007AC1", "#EF3B24"),
            "Orlando Magic": ("#0077C0", "#C4CED4"),
            "Philadelphia 76ers": ("#006BB6", "#ED174C"),
            "Phoenix Suns": ("#1D1160", "#E56020"),
            "Portland Trail Blazers": ("#E03A3E", "#000000"),
            "Sacramento Kings": ("#5A2D81", "#63727A"),
            "San Antonio Spurs": ("#C4CED4", "#000000"),
            "Toronto Raptors": ("#CE1141", "#000000"),
            "Utah Jazz": ("#002B5C", "#00471B"),
            "Washington Wizards": ("#002B5C", "#E31837")
        }
        
        # Look for exact match first
        if team_name in colors_map:
            return colors_map[team_name]
        
        # Try to find by partial match
        for known_name, colors in colors_map.items():
            # Extract the distinct part of the team name (e.g., "Lakers" from "Los Angeles Lakers")
            known_parts = known_name.split()
            name_parts = team_name.split()
            
            # Check for matches in the team name parts
            for part in name_parts:
                if part in known_parts and len(part) > 3:  # Avoid matching on "The", "Los", etc.
                    return colors
        
        # Default colors if not found
        return ("#004080", "#FDB927")  # Default blue and gold
    
    def transform_api_team_to_model(self, api_team: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform API team data to model format.
        
        Args:
            api_team: Team data from API
            
        Returns:
            Transformed team data ready for model creation
        """
        # Extract team name
        name = api_team.get("name", "")
        
        # Generate abbreviation if not provided
        if "code" in api_team and api_team["code"]:
            abbreviation = api_team["code"]
        else:
            abbreviation = self.extract_team_abbreviation(name)
        
        # Get colors
        primary_color, secondary_color = self.get_team_colors(name)
        
        return {
            "external_id": api_team.get("id"),
            "name": name,
            "abbreviation": abbreviation,
            "logo_url": api_team.get("logo"),
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "is_national": api_team.get("national", False)
        }
    
    def transform_api_game_to_model(self, api_game: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform API game data to model format.
        
        Args:
            api_game: Game data from API
            
        Returns:
            Transformed game data ready for model creation
        """
        # Parse date
        game_date = None
        if "date" in api_game:
            game_date = self.parse_api_datetime(api_game["date"])
        
        # Build status object
        status = {
            "long": api_game.get("status", {}).get("long", ""),
            "short": api_game.get("status", {}).get("short", ""),
            "timer": api_game.get("status", {}).get("timer")
        }
        
        # Build score objects if available
        home_score = None
        away_score = None
        
        if "scores" in api_game and api_game["scores"]:
            if "home" in api_game["scores"]:
                home_score = {
                    "quarter_1": api_game["scores"]["home"].get("quarter_1"),
                    "quarter_2": api_game["scores"]["home"].get("quarter_2"),
                    "quarter_3": api_game["scores"]["home"].get("quarter_3"),
                    "quarter_4": api_game["scores"]["home"].get("quarter_4"),
                    "over_time": api_game["scores"]["home"].get("over_time"),
                    "total": api_game["scores"]["home"].get("total")
                }
            
            if "away" in api_game["scores"]:
                away_score = {
                    "quarter_1": api_game["scores"]["away"].get("quarter_1"),
                    "quarter_2": api_game["scores"]["away"].get("quarter_2"),
                    "quarter_3": api_game["scores"]["away"].get("quarter_3"),
                    "quarter_4": api_game["scores"]["away"].get("quarter_4"),
                    "over_time": api_game["scores"]["away"].get("over_time"),
                    "total": api_game["scores"]["away"].get("total")
                }
        
        return {
            "external_id": api_game.get("id"),
            "date": game_date,
            "timestamp": api_game.get("timestamp"),
            "timezone": api_game.get("timezone", "UTC"),
            "stage": api_game.get("stage"),
            "status": status,
            "home_team": {
                "external_id": api_game.get("teams", {}).get("home", {}).get("id"),
                "name": api_game.get("teams", {}).get("home", {}).get("name", ""),
                "scores": home_score
            },
            "away_team": {
                "external_id": api_game.get("teams", {}).get("away", {}).get("id"),
                "name": api_game.get("teams", {}).get("away", {}).get("name", ""),
                "scores": away_score
            },
            "venue": api_game.get("arena", {}).get("name")
        }