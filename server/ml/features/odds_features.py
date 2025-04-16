"""
Betting odds feature extraction for basketball predictions.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import statistics

from app.core.logger import logger
from app.db.models.odd import OddsInDB
from app.data.repositories.game_repository import GameRepository
from app.data.repositories.odds_repository import OddsRepository

class OddsFeatureProcessor:
    """
    Extracts features from betting odds data, such as:
    - Implied probabilities
    - Market consensus
    - Line movements
    - Bookmaker agreement/disagreement
    """
    
    def __init__(self):
        """Initialize the odds feature processor."""
        self.feature_prefix = 'odds_'
        self.game_repository = GameRepository()
    
    async def extract_features(self, game_id: str) -> Dict[str, Any]:
        """
        Extract odds-related features for a game.
        
        Args:
            game_id: MongoDB game ID
            
        Returns:
            Dictionary of feature name to value
        """
        # Get odds data
        odds_repository = OddsRepository()
        odds = await odds_repository.find_by_game_id(game_id)
        
        if not odds:
            # Return default values if no odds data available
            return self._get_default_features()
        
        # Process odds data
        return self.process_odds(odds)
    
    def process_odds(self, odds: OddsInDB) -> Dict[str, Any]:
        """
        Process odds data into features.
        
        Args:
            odds: Odds data
            
        Returns:
            Dictionary of feature name to value
        """
        features = {}
        
        # Consensus odds features
        features.update(self._extract_consensus_features(odds))
        
        # Bookmaker agreement features
        features.update(self._extract_bookmaker_agreement_features(odds))
        
        # Market sentiment features
        features.update(self._extract_market_sentiment_features(odds))
        
        return features
    
    def _extract_consensus_features(self, odds: OddsInDB) -> Dict[str, Any]:
        """
        Extract features from consensus odds.
        
        Args:
            odds: Odds data
            
        Returns:
            Dictionary of consensus features
        """
        features = {}
        
        # Check if consensus data is available
        if not hasattr(odds, 'consensus') or not odds.consensus:
            return self._get_default_consensus_features()
        
        # Implied probabilities from consensus odds
        home_prob = odds.consensus.implied_home_probability
        away_prob = odds.consensus.implied_away_probability
        
        features[f'{self.feature_prefix}home_implied_prob'] = home_prob
        features[f'{self.feature_prefix}away_implied_prob'] = away_prob
        
        # Probability differential (home advantage in probability space)
        features[f'{self.feature_prefix}implied_prob_diff'] = home_prob - away_prob
        
        # Odds values
        features[f'{self.feature_prefix}home_odds'] = odds.consensus.home_win
        features[f'{self.feature_prefix}away_odds'] = odds.consensus.away_win
        
        # Log odds (better for modeling)
        features[f'{self.feature_prefix}home_log_odds'] = np.log(odds.consensus.home_win)
        features[f'{self.feature_prefix}away_log_odds'] = np.log(odds.consensus.away_win)
        
        # Favorite indicator (1 if home team is favorite, 0 otherwise)
        features[f'{self.feature_prefix}home_is_favorite'] = 1 if home_prob > away_prob else 0
        
        # If home team has more than 60% implied probability, flag as strong favorite
        features[f'{self.feature_prefix}home_strong_favorite'] = 1 if home_prob > 0.6 else 0
        
        # Estimated market-based win probability
        # This adjusts for overround (the total probability exceeding 100%)
        total_prob = home_prob + away_prob
        if total_prob > 0:
            adjusted_home_prob = home_prob / total_prob
            features[f'{self.feature_prefix}market_win_prob'] = adjusted_home_prob
        else:
            features[f'{self.feature_prefix}market_win_prob'] = 0.5
        
        return features
    
    def _extract_bookmaker_agreement_features(self, odds: OddsInDB) -> Dict[str, Any]:
        """
        Extract features related to bookmaker agreement/disagreement.
        
        Args:
            odds: Odds data
            
        Returns:
            Dictionary of bookmaker agreement features
        """
        features = {}
        
        # Check if bookmakers data is available
        if not hasattr(odds, 'bookmakers') or not odds.bookmakers:
            return self._get_default_bookmaker_features()
        
        # Collect moneyline (h2h) odds from all bookmakers
        home_odds_list = []
        away_odds_list = []
        
        for bookmaker in odds.bookmakers:
            for bet in bookmaker.bets:
                if bet.name.lower() == 'h2h':
                    for value in bet.values:
                        if value.value.lower() in ['home', '1']:
                            home_odds_list.append(value.odd)
                        elif value.value.lower() in ['away', '2']:
                            away_odds_list.append(value.odd)
        
        # Calculate agreement metrics if we have enough data
        if len(home_odds_list) >= 2 and len(away_odds_list) >= 2:
            # Standard deviation of odds (lower means more agreement)
            features[f'{self.feature_prefix}home_odds_std'] = np.std(home_odds_list)
            features[f'{self.feature_prefix}away_odds_std'] = np.std(away_odds_list)
            
            # Range of odds
            features[f'{self.feature_prefix}home_odds_range'] = max(home_odds_list) - min(home_odds_list)
            features[f'{self.feature_prefix}away_odds_range'] = max(away_odds_list) - min(away_odds_list)
            
            # Coefficient of variation (normalized measure of dispersion)
            features[f'{self.feature_prefix}home_odds_cv'] = np.std(home_odds_list) / np.mean(home_odds_list) if np.mean(home_odds_list) > 0 else 0
            features[f'{self.feature_prefix}away_odds_cv'] = np.std(away_odds_list) / np.mean(away_odds_list) if np.mean(away_odds_list) > 0 else 0
            
            # Bookmaker agreement score (1 = perfect agreement, 0 = high disagreement)
            # This is an arbitrary scale based on typical odds variation
            home_agreement = 1 - min(features[f'{self.feature_prefix}home_odds_cv'] * 5, 1)
            away_agreement = 1 - min(features[f'{self.feature_prefix}away_odds_cv'] * 5, 1)
            
            features[f'{self.feature_prefix}bookmaker_agreement'] = (home_agreement + away_agreement) / 2
            
            # Number of bookmakers (a signal of market liquidity)
            features[f'{self.feature_prefix}bookmaker_count'] = len(odds.bookmakers)
        else:
            # Default values if insufficient data
            features.update(self._get_default_bookmaker_features())
        
        return features
    
    def _extract_market_sentiment_features(self, odds: OddsInDB) -> Dict[str, Any]:
        """
        Extract features related to market sentiment.
        
        Args:
            odds: Odds data
            
        Returns:
            Dictionary of market sentiment features
        """
        features = {}
        
        # For these features, we would ideally track odds over time to detect line movements
        # Since we don't have historical odds in this implementation, we'll use placeholder values
        
        # Market confidence (derived from bookmaker agreement)
        if f'{self.feature_prefix}bookmaker_agreement' in features:
            market_confidence = features[f'{self.feature_prefix}bookmaker_agreement']
        else:
            bookmaker_features = self._extract_bookmaker_agreement_features(odds)
            market_confidence = bookmaker_features.get(f'{self.feature_prefix}bookmaker_agreement', 0.5)
        
        features[f'{self.feature_prefix}market_confidence'] = market_confidence
        
        # Value bet indicator (when model probability differs significantly from market probability)
        # This would require a model prediction, so we'll set a placeholder
        features[f'{self.feature_prefix}value_bet_indicator'] = 0
        
        # Sharp money indicator (when odds move against public perception)
        # This would require line movement data, so we'll set a placeholder
        features[f'{self.feature_prefix}sharp_money_indicator'] = 0
        
        # Market efficiency metric
        # Higher = more efficient market (harder to find value)
        # Calculated as inverse of standard deviation in implied probabilities
        if f'{self.feature_prefix}home_odds_std' in features:
            home_prob_std = features[f'{self.feature_prefix}home_odds_std']
            if home_prob_std > 0:
                features[f'{self.feature_prefix}market_efficiency'] = 1 / (5 * home_prob_std)
            else:
                features[f'{self.feature_prefix}market_efficiency'] = 0.5
        else:
            features[f'{self.feature_prefix}market_efficiency'] = 0.5
        
        return features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """
        Get default features when no odds data is available.
        
        Returns:
            Dictionary of default features
        """
        features = {}
        
        # Consensus features
        features.update(self._get_default_consensus_features())
        
        # Bookmaker agreement features
        features.update(self._get_default_bookmaker_features())
        
        # Market sentiment features
        features[f'{self.feature_prefix}market_confidence'] = 0.5
        features[f'{self.feature_prefix}value_bet_indicator'] = 0
        features[f'{self.feature_prefix}sharp_money_indicator'] = 0
        features[f'{self.feature_prefix}market_efficiency'] = 0.5
        
        return features
    
    def _get_default_consensus_features(self) -> Dict[str, Any]:
        """
        Get default consensus features when no odds data is available.
        
        Returns:
            Dictionary of default consensus features
        """
        features = {}
        
        # Use 50/50 probabilities as default
        features[f'{self.feature_prefix}home_implied_prob'] = 0.5
        features[f'{self.feature_prefix}away_implied_prob'] = 0.5
        features[f'{self.feature_prefix}implied_prob_diff'] = 0.0
        
        # Default odds of 2.0 (implied 50% probability)
        features[f'{self.feature_prefix}home_odds'] = 2.0
        features[f'{self.feature_prefix}away_odds'] = 2.0
        features[f'{self.feature_prefix}home_log_odds'] = np.log(2.0)
        features[f'{self.feature_prefix}away_log_odds'] = np.log(2.0)
        
        # No favorite
        features[f'{self.feature_prefix}home_is_favorite'] = 0
        features[f'{self.feature_prefix}home_strong_favorite'] = 0
        features[f'{self.feature_prefix}market_win_prob'] = 0.5
        
        return features
    
    def _get_default_bookmaker_features(self) -> Dict[str, Any]:
        """
        Get default bookmaker agreement features when no odds data is available.
        
        Returns:
            Dictionary of default bookmaker agreement features
        """
        features = {}
        
        # Default to moderate agreement
        features[f'{self.feature_prefix}home_odds_std'] = 0.1
        features[f'{self.feature_prefix}away_odds_std'] = 0.1
        features[f'{self.feature_prefix}home_odds_range'] = 0.2
        features[f'{self.feature_prefix}away_odds_range'] = 0.2
        features[f'{self.feature_prefix}home_odds_cv'] = 0.05
        features[f'{self.feature_prefix}away_odds_cv'] = 0.05
        features[f'{self.feature_prefix}bookmaker_agreement'] = 0.75
        features[f'{self.feature_prefix}bookmaker_count'] = 0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        Get a list of feature names generated by this processor.
        
        Returns:
            List of feature names
        """
        return [
            # Consensus features
            f'{self.feature_prefix}home_implied_prob',
            f'{self.feature_prefix}away_implied_prob',
            f'{self.feature_prefix}implied_prob_diff',
            f'{self.feature_prefix}home_odds',
            f'{self.feature_prefix}away_odds',
            f'{self.feature_prefix}home_log_odds',
            f'{self.feature_prefix}away_log_odds',
            f'{self.feature_prefix}home_is_favorite',
            f'{self.feature_prefix}home_strong_favorite',
            f'{self.feature_prefix}market_win_prob',
            
            # Bookmaker agreement features
            f'{self.feature_prefix}home_odds_std',
            f'{self.feature_prefix}away_odds_std',
            f'{self.feature_prefix}home_odds_range',
            f'{self.feature_prefix}away_odds_range',
            f'{self.feature_prefix}home_odds_cv',
            f'{self.feature_prefix}away_odds_cv',
            f'{self.feature_prefix}bookmaker_agreement',
            f'{self.feature_prefix}bookmaker_count',
            
            # Market sentiment features
            f'{self.feature_prefix}market_confidence',
            f'{self.feature_prefix}value_bet_indicator',
            f'{self.feature_prefix}sharp_money_indicator',
            f'{self.feature_prefix}market_efficiency'
        ]