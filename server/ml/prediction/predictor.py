"""
Basketball game prediction module.

This module handles the prediction process for basketball games, integrating
the model, feature extraction, and confidence calculation.
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from app.core.logger import logger
from app.db.models.game import GameInDB, GamePrediction
from app.ml.models.model_registry import ModelRegistry
from app.ml.features.team_features import TeamFeatureProcessor
from app.ml.features.odds_features import OddsFeatureProcessor


class BasketballPredictor:
    """
    Predicts basketball game outcomes using the active ML model.
    """
    
    def __init__(self):
        """Initialize the basketball predictor."""
        self.model_registry = ModelRegistry()
        self.team_feature_processor = TeamFeatureProcessor()
        self.odds_feature_processor = OddsFeatureProcessor()
        self.model = None
        self.features_cache = {}  # Cache features to avoid recomputation
    
    async def load_model(self) -> bool:
        """
        Load the active prediction model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            self.model = await self.model_registry.get_active_model()
            return self.model is not None
        except Exception as e:
            logger.error(f"Failed to load active model: {str(e)}")
            return False
    
    async def predict_game(
        self,
        game: GameInDB,
        refresh_features: bool = False
    ) -> Optional[GamePrediction]:
        """
        Predict the outcome of a basketball game.
        
        Args:
            game: Game data
            refresh_features: Whether to recalculate features even if cached
            
        Returns:
            GamePrediction object or None if prediction fails
        """
        # Ensure model is loaded
        if not self.model:
            loaded = await self.load_model()
            if not loaded:
                logger.error("No model available for prediction")
                return None
        
        # Check if game already has features in cache
        game_id = str(game.id)
        if game_id in self.features_cache and not refresh_features:
            features_df = self.features_cache[game_id]
        else:
            # Extract features
            features = await self._extract_features(game)
            if not features:
                logger.error(f"Failed to extract features for game {game_id}")
                return None
            
            features_df = pd.DataFrame([features])
            self.features_cache[game_id] = features_df
        
        try:
            # Make prediction
            prediction_result = self.model.predict(features_df)[0]
            probability = float(self.model.predict_proba(features_df)[0])
            
            # Determine winner based on prediction
            if self.model.output_classes and len(self.model.output_classes) > 1:
                # If using categorical outcome classes
                winner = "home" if prediction_result == self.model.output_classes[1] else "away"
            else:
                # If using binary prediction
                winner = "home" if probability >= 0.5 else "away"
            
            # Use actual probability or convert to the correct direction
            confidence = probability if winner == "home" else (1.0 - probability)
            
            # Create prediction
            prediction = GamePrediction(
                predicted_winner=winner,
                home_win_probability=probability if winner == "home" else (1.0 - probability),
                confidence=int(confidence * 100),  # Convert from 0-1 to 0-100
                model_version=self.model.model_version,
                created_at=datetime.utcnow(),
                is_game_of_day=False,  # Will be set by selection process
                is_in_vip_betslip=False  # Will be set by selection process
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting game {game_id}: {str(e)}")
            return None
    
    async def batch_predict(
        self,
        games: List[GameInDB],
        refresh_features: bool = False
    ) -> Dict[str, GamePrediction]:
        """
        Predict outcomes for multiple games.
        
        Args:
            games: List of games to predict
            refresh_features: Whether to recalculate features
            
        Returns:
            Dictionary mapping game IDs to predictions
        """
        predictions = {}
        
        # Ensure model is loaded
        if not self.model:
            loaded = await self.load_model()
            if not loaded:
                logger.error("No model available for batch prediction")
                return predictions
        
        # Process each game
        for game in games:
            game_id = str(game.id)
            prediction = await self.predict_game(game, refresh_features)
            
            if prediction:
                predictions[game_id] = prediction
        
        logger.info(f"Batch predicted {len(predictions)} out of {len(games)} games")
        return predictions
    
    async def _extract_features(self, game: GameInDB) -> Optional[Dict[str, Any]]:
        """
        Extract features for a game.
        
        Args:
            game: Game data
            
        Returns:
            Dictionary of features or None if extraction fails
        """
        try:
            features = {}
            
            # Extract team features
            if hasattr(game, 'home_team') and hasattr(game, 'away_team'):
                if hasattr(game, 'home_team_stats') and hasattr(game, 'away_team_stats'):
                    team_features = await self.team_feature_processor.extract_features(
                        game.home_team,
                        game.away_team,
                        game.home_team_stats,
                        game.away_team_stats
                    )
                    features.update(team_features)
                else:
                    logger.warning(f"Missing team stats for game {game.id}")
            else:
                logger.warning(f"Missing team data for game {game.id}")
            
            # Extract odds features
            odds_features = await self.odds_feature_processor.extract_features(str(game.id))
            features.update(odds_features)
            
            # Add game-specific features if needed
            if hasattr(game, 'date'):
                features['days_since_season_start'] = (game.date - game.season_start_date).days if hasattr(game, 'season_start_date') else 0
                
                # Check if game is on a back-to-back for either team
                features['home_back_to_back'] = 1 if game.is_home_back_to_back else 0
                features['away_back_to_back'] = 1 if game.is_away_back_to_back else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error for game {game.id}: {str(e)}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self.features_cache = {}
        logger.info("Feature cache cleared")