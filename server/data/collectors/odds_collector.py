"""
Collector for basketball odds data.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

from bson import ObjectId

from app.core.logger import logger
from app.db.models.odd import (
    OddsModel, OddsInDB, Bookmaker, Bet, OddValue, OddsConsensus, get_odds_dict
)
from server.data.repositories.game_repository import GameRepository
from server.data.providers.odds_api import OddsAPI
from server.data.processors.transformer import DataTransformer
from server.data.processors.cleaner import DataCleaner


class OddsCollector:
    """
    Collects and processes basketball betting odds data.
    """
    
    def __init__(
        self, 
        odds_api: Optional[OddsAPI] = None,
        game_repository: Optional[GameRepository] = None
    ):
        """
        Initialize the odds collector.
        
        Args:
            odds_api: Optional OddsAPI instance
            game_repository: Optional GameRepository instance
        """
        self.odds_api = odds_api or OddsAPI()
        self.game_repository = game_repository or GameRepository()
        self.transformer = DataTransformer()
        self.cleaner = DataCleaner()
    
    async def collect_odds_for_sport(
        self,
        sport_key: str = "basketball_nba",
        days_ahead: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Collect odds data for a sport.
        
        Args:
            sport_key: The sport key (e.g., "basketball_nba")
            days_ahead: Number of days ahead to collect odds for
            
        Returns:
            List of raw odds data
        """
        logger.info(f"Collecting odds for sport: {sport_key}")
        
        # Get odds from API
        odds_data = await self.odds_api.get_odds(
            sport_key=sport_key,
            regions="us",
            markets="h2h,spreads,totals",
            odds_format="decimal"
        )
        
        logger.info(f"Found {len(odds_data)} events with odds")
        return odds_data
    
    async def collect_odds_for_game(
        self,
        game_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Collect odds for a specific game.
        
        Args:
            game_id: MongoDB game ID
            
        Returns:
            Odds data or None if not found
        """
        # First, get the game to get external ID info
        game = await self.game_repository.find_by_id(game_id)
        
        if not game:
            logger.error(f"Game not found: {game_id}")
            return None
        
        logger.info(f"Collecting odds for game: {game.external_id}")
        
        # Construct event ID as expected by Odds API
        # Note: This will need to be adjusted based on how event IDs are mapped
        # between your game data and the Odds API
        event_id = f"{game.external_id}"
        
        # Get odds from API
        try:
            odds_data = await self.odds_api.get_event_odds(
                sport_key="basketball_nba",
                event_id=event_id,
                regions="us",
                markets="h2h,spreads,totals",
                odds_format="decimal"
            )
            
            return odds_data
        except Exception as e:
            logger.error(f"Error collecting odds for game {game_id}: {str(e)}")
            return None
    
    async def process_odds_data(
        self,
        odds_data: List[Dict[str, Any]]
    ) -> List[OddsModel]:
        """
        Process raw odds data into odds models.
        
        Args:
            odds_data: Raw odds data from API
            
        Returns:
            List of odds models
        """
        odds_models = []
        
        for event_odds in odds_data:
            try:
                # Find the corresponding game in our database
                game = await self._find_game_by_teams(
                    home_team=event_odds.get("home_team", ""),
                    away_team=event_odds.get("away_team", ""),
                    commence_time=event_odds.get("commence_time")
                )
                
                if not game:
                    logger.warning(f"Could not find matching game for event: {event_odds.get('id')}")
                    continue
                
                # Process bookmakers data
                bookmakers_data = event_odds.get("bookmakers", [])
                bookmakers = []
                
                for bookmaker_data in bookmakers_data:
                    bets = []
                    
                    # Process markets/bets
                    for market_data in bookmaker_data.get("markets", []):
                        odd_values = []
                        
                        # Process outcomes
                        for outcome in market_data.get("outcomes", []):
                            odd_values.append(OddValue(
                                value=outcome.get("name", ""),
                                odd=outcome.get("price", 0.0)
                            ))
                        
                        # Add bet
                        bets.append(Bet(
                            id=self._generate_bet_id(market_data.get("key", "")),
                            name=market_data.get("key", ""),
                            values=odd_values
                        ))
                    
                    # Add bookmaker
                    bookmakers.append(Bookmaker(
                        id=self._generate_bookmaker_id(bookmaker_data.get("key", "")),
                        name=bookmaker_data.get("title", ""),
                        bets=bets
                    ))
                
                # Calculate consensus odds
                consensus = self._calculate_consensus_odds(bookmakers)
                
                # Create odds model
                odds_model = OddsModel(
                    game_id=str(game.id),
                    external_game_id=game.external_id,
                    league_id=game.league_id,
                    season_id=game.season_id,
                    bookmakers=bookmakers,
                    consensus=consensus
                )
                
                odds_models.append(odds_model)
                
            except Exception as e:
                logger.error(f"Error processing odds for event {event_odds.get('id', 'unknown')}: {str(e)}", exc_info=True)
                continue
        
        logger.info(f"Processed {len(odds_models)} odds models")
        return odds_models
    
    async def save_odds(self, odds_models: List[OddsModel]) -> int:
        """
        Save odds to the database.
        
        Args:
            odds_models: List of odds models to save
            
        Returns:
            Number of odds saved or updated
        """
        from server.data.repositories.odds_repository import OddsRepository
        
        if not odds_models:
            return 0
        
        odds_repository = OddsRepository()
        saved_count = 0
        
        for odds in odds_models:
            try:
                # Check if odds already exist for this game
                existing_odds = await odds_repository.find_by_game_id(odds.game_id)
                
                if existing_odds:
                    # Update existing odds
                    odds_dict = get_odds_dict(odds)
                    updated = await odds_repository.update(
                        id=str(existing_odds.id),
                        update={"$set": odds_dict}
                    )
                    
                    if updated:
                        saved_count += 1
                else:
                    # Insert new odds
                    odds_id = await odds_repository.create(odds)
                    if odds_id:
                        saved_count += 1
                        
            except Exception as e:
                logger.error(f"Error saving odds for game {odds.game_id}: {str(e)}")
                continue
        
        return saved_count
    
    async def collect_and_save_odds(
        self,
        sport_key: str = "basketball_nba",
        days_ahead: int = 3
    ) -> Dict[str, Any]:
        """
        Collect and save odds.
        
        Args:
            sport_key: The sport key (e.g., "basketball_nba")
            days_ahead: Number of days ahead to collect odds for
            
        Returns:
            Summary of collection results
        """
        # Collect odds
        odds_data = await self.collect_odds_for_sport(
            sport_key=sport_key,
            days_ahead=days_ahead
        )
        
        if not odds_data:
            return {"odds_found": 0, "odds_saved": 0}
        
        # Process odds
        odds_models = await self.process_odds_data(odds_data)
        
        # Save odds
        saved_count = await self.save_odds(odds_models)
        
        return {
            "odds_found": len(odds_data),
            "odds_processed": len(odds_models),
            "odds_saved": saved_count
        }
    
    async def _find_game_by_teams(
        self,
        home_team: str,
        away_team: str,
        commence_time: Optional[str] = None
    ) -> Optional[Any]:
        """
        Find a game by team names and start time.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            commence_time: Game start time
            
        Returns:
            Game or None if not found
        """
        # Clean team names
        home_team = self.cleaner.normalize_team_name(home_team)
        away_team = self.cleaner.normalize_team_name(away_team)
        
        # Parse commence time
        parsed_time = None
        if commence_time:
            parsed_time = self.transformer.parse_api_datetime(commence_time)
        
        # Build date range for better matching
        date_range = {}
        if parsed_time:
            # Look for games within a 12-hour window of the commence time
            start_time = parsed_time - timedelta(hours=12)
            end_time = parsed_time + timedelta(hours=12)
            date_range = {"date": {"$gte": start_time, "$lte": end_time}}
        
        # Try to find exact match with both teams
        game = await self.game_repository.find_one({
            "home_team.name": {"$regex": home_team, "$options": "i"},
            "away_team.name": {"$regex": away_team, "$options": "i"},
            **date_range
        })
        
        if game:
            return game
        
        # Try with swapped teams (in case API has different home/away designation)
        game = await self.game_repository.find_one({
            "home_team.name": {"$regex": away_team, "$options": "i"},
            "away_team.name": {"$regex": home_team, "$options": "i"},
            **date_range
        })
        
        if game:
            return game
        
        # Try more lenient search with just one team and date if still not found
        if parsed_time:
            start_of_day = datetime(parsed_time.year, parsed_time.month, parsed_time.day)
            end_of_day = start_of_day + timedelta(days=1) - timedelta(microseconds=1)
            
            return await self.game_repository.find_one({
                "$or": [
                    {"home_team.name": {"$regex": home_team, "$options": "i"}},
                    {"away_team.name": {"$regex": away_team, "$options": "i"}}
                ],
                "date": {"$gte": start_of_day, "$lte": end_of_day}
            })
        
        return None
    
    def _generate_bookmaker_id(self, key: str) -> int:
        """
        Generate a numeric ID for a bookmaker.
        
        Args:
            key: Bookmaker key
            
        Returns:
            Numeric ID
        """
        # Simple hash function for demo - in production would use a more robust approach
        return hash(key) % 10000
    
    def _generate_bet_id(self, key: str) -> int:
        """
        Generate a numeric ID for a bet.
        
        Args:
            key: Bet key
            
        Returns:
            Numeric ID
        """
        # Simple hash function for demo
        return hash(key) % 10000
    
    def _calculate_consensus_odds(self, bookmakers: List[Bookmaker]) -> OddsConsensus:
        """
        Calculate consensus odds from multiple bookmakers.
        
        Args:
            bookmakers: List of bookmakers with odds
            
        Returns:
            Consensus odds
        """
        # Default values
        home_win_odds = []
        away_win_odds = []
        
        # Extract home and away win odds from all bookmakers
        for bookmaker in bookmakers:
            for bet in bookmaker.bets:
                if bet.name.lower() == "h2h":
                    for value in bet.values:
                        if value.value.lower() in ["home", "1"]:
                            home_win_odds.append(value.odd)
                        elif value.value.lower() in ["away", "2"]:
                            away_win_odds.append(value.odd)
        
        # Calculate average odds
        home_win = sum(home_win_odds) / len(home_win_odds) if home_win_odds else 2.0
        away_win = sum(away_win_odds) / len(away_win_odds) if away_win_odds else 2.0
        
        # Calculate implied probabilities
        # For decimal odds, implied probability = 1 / decimal odds
        implied_home_probability = 1 / home_win if home_win > 0 else 0.5
        implied_away_probability = 1 / away_win if away_win > 0 else 0.5
        
        # Normalize probabilities to sum to 1.0
        total_probability = implied_home_probability + implied_away_probability
        implied_home_probability = implied_home_probability / total_probability
        implied_away_probability = implied_away_probability / total_probability
        
        return OddsConsensus(
            home_win=home_win,
            away_win=away_win,
            implied_home_probability=implied_home_probability,
            implied_away_probability=implied_away_probability
        )