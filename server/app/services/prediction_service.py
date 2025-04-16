"""
Prediction service for handling game predictions.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple, Union

from bson import ObjectId
from fastapi import HTTPException, status

from app.core.logger import logger
from app.db.models.game import GameInDB, GamePrediction
from app.db.models.prediction import (
    PredictionModelSchema, PredictionModelInDB,
    PredictionHistory, PredictionHistoryInDB,
    GamePredictionItem
)
from app.data.repositories.prediction_repository import (
    PredictionModelRepository, PredictionHistoryRepository
)
from app.data.repositories.game_repository import GameRepository


class PredictionService:
    """Service for prediction-related operations."""
    
    def __init__(self):
        """Initialize the prediction service with repositories."""
        self.model_repository = PredictionModelRepository()
        self.history_repository = PredictionHistoryRepository()
        self.game_repository = GameRepository()
    
    async def get_active_model(self) -> Optional[PredictionModelInDB]:
        """
        Get the currently active prediction model.
        
        Returns:
            Active model or None if not found
        """
        return await self.model_repository.find_active_model()
    
    async def create_model(self, model: PredictionModelSchema) -> PredictionModelInDB:
        """
        Create a new prediction model.
        
        Args:
            model: The model to create
            
        Returns:
            Created model
            
        Raises:
            HTTPException: If a model with the same version already exists
        """
        # Check if model with this version already exists
        existing = await self.model_repository.find_by_version(model.version)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model with version {model.version} already exists"
            )
        
        # Create model
        model_id = await self.model_repository.create(model)
        if not model_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create model"
            )
        
        # Return created model
        return await self.model_repository.find_by_id(model_id)
    
    async def activate_model(self, model_id: str) -> Optional[PredictionModelInDB]:
        """
        Activate a prediction model and deactivate others.
        
        Args:
            model_id: The ID of the model to activate
            
        Returns:
            Activated model or None if not found
        """
        # Activate model
        deactivated, activated = await self.model_repository.activate_model(model_id)
        
        if not activated:
            return None
            
        # Return activated model
        return await self.model_repository.find_by_id(model_id)
    
    async def get_game_prediction(self, game_id: str) -> Optional[GamePrediction]:
        """
        Get prediction for a specific game.
        
        Args:
            game_id: The ID of the game
            
        Returns:
            Game prediction or None if not found
        """
        game = await self.game_repository.find_by_id(game_id)
        
        if not game or not game.prediction:
            return None
            
        return game.prediction
    
    async def save_prediction(
        self,
        game_id: str,
        prediction: Union[GamePrediction, Dict[str, Any]]
    ) -> Optional[GamePrediction]:
        """
        Save a prediction for a game.
        
        Args:
            game_id: The ID of the game
            prediction: The prediction to save
            
        Returns:
            Saved prediction or None if save failed
        """
        # Convert to GamePrediction if dict
        if isinstance(prediction, dict):
            prediction = GamePrediction(**prediction)
        
        # Save prediction
        updated = await self.game_repository.update_prediction(game_id, prediction)
        
        if not updated:
            return None
            
        # Return saved prediction
        return await self.get_game_prediction(game_id)
    
    async def get_user_prediction_history(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PredictionHistoryInDB]:
        """
        Get prediction history for a user.
        
        Args:
            user_id: The ID of the user
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of prediction history records
        """
        # Default to last 30 days if no dates provided
        if not start_date:
            end_date = end_date or datetime.utcnow()
            start_date = end_date - timedelta(days=30)
        
        return await self.history_repository.find_by_user_date_range(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
    
    async def save_user_prediction_history(
        self,
        user_id: str,
        date: str,
        predictions: List[GamePredictionItem],
        summary: Dict[str, Any]
    ) -> Optional[str]:
        """
        Save prediction history for a user.
        
        Args:
            user_id: The ID of the user
            date: The date (YYYY-MM-DD)
            predictions: List of game predictions
            summary: Summary statistics
            
        Returns:
            ID of saved history or None if save failed
        """
        return await self.history_repository.upsert_history(
            user_id=user_id,
            date=date,
            predictions=predictions,
            summary=summary
        )
    
    async def get_user_accuracy_stats(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get accuracy statistics for a user.
        
        Args:
            user_id: The ID of the user
            days: Number of days to consider
            
        Returns:
            Dictionary of statistics
        """
        return await self.history_repository.calculate_user_stats(
            user_id=user_id,
            days=days
        )
    
    async def get_daily_betslip(self, date: Optional[datetime] = None) -> List[str]:
        """
        Get the list of game IDs in the VIP betslip for a specific date.
        
        Args:
            date: Optional date (defaults to today)
            
        Returns:
            List of game IDs
        """
        # Get betslip games
        betslip_games = await self.game_repository.find_betslip_games(date)
        
        # Return game IDs
        return [str(game.id) for game in betslip_games]
    
    async def update_betslip_games(
        self,
        date: datetime,
        game_ids: List[str]
    ) -> Tuple[int, int]:
        """
        Update the games in the VIP betslip for a specific date.
        
        Args:
            date: The date
            game_ids: List of game IDs to include
            
        Returns:
            Tuple of (cleared games count, updated games count)
        """
        return await self.game_repository.update_betslip_games(
            date=date,
            game_ids=game_ids
        )
    
    async def get_prediction_explanation(
        self,
        game_id: str,
        include_raw_shap: bool = False
    ) -> Dict[str, Any]:
        """
        Generate an explanation for a game prediction.
        
        Args:
            game_id: The ID of the game
            include_raw_shap: Whether to include raw SHAP values in the response
            
        Returns:
            Dictionary with explanation information
            
        Raises:
            HTTPException: If game not found, no prediction available, or explanation fails
        """
        from app.ml.prediction.explanation import PredictionExplainer
        from app.ml.models.model_registry import ModelRegistry
        
        # Get the game
        game = await self.game_repository.find_by_id(game_id)
        
        if not game:
            logger.error(f"Game with ID {game_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
        
        if not game.prediction:
            logger.error(f"No prediction available for game {game_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No prediction available for this game"
            )
        
        # Get active model
        model_registry = ModelRegistry()
        try:
            model = await model_registry.get_active_model()
        except ValueError as e:
            logger.error(f"Failed to load active model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load prediction model"
            )
        
        if not model:
            logger.error("No active model available for explanation")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No active model available for explanation"
            )
        
        # Get features for the game
        # In a real implementation, you would retrieve the features from a feature store
        # or recalculate them based on the game data
        # For this example, we'll create a placeholder feature vector
        
        # Simplified approach: construct a feature dataframe
        # Ideally, you would retrieve the actual feature values used for the original prediction
        from app.ml.features.team_features import TeamFeatureProcessor
        from app.ml.features.odds_features import OddsFeatureProcessor
        
        # Create a dummy feature vector based on the game data
        features = {}
        
        # Add prediction confidence as probability
        features['prediction_confidence'] = game.prediction.confidence
        
        # Team features (simplified version)
        features['team_home_win_pct'] = game.home_team.win_percentage if hasattr(game.home_team, 'win_percentage') else 0.5
        features['team_away_win_pct'] = game.away_team.win_percentage if hasattr(game.away_team, 'win_percentage') else 0.5
        features['team_win_pct_diff'] = features['team_home_win_pct'] - features['team_away_win_pct']
        
        # Odds features (simplified version)
        features['odds_home_implied_prob'] = game.odds.home_win_probability if hasattr(game, 'odds') and hasattr(game.odds, 'home_win_probability') else 0.5
        features['odds_away_implied_prob'] = game.odds.away_win_probability if hasattr(game, 'odds') and hasattr(game.odds, 'away_win_probability') else 0.5
        features['odds_implied_prob_diff'] = features['odds_home_implied_prob'] - features['odds_away_implied_prob']
        
        # Create DataFrame
        import pandas as pd
        features_df = pd.DataFrame([features])
        
        # Initialize explainer
        explainer = PredictionExplainer(model)
        
        try:
            # Generate explanation
            explanation = explainer.explain_prediction(
                features_df,
                include_shap_values=include_raw_shap
            )
            
            # Add game context
            explanation['game_id'] = str(game.id)
            explanation['home_team'] = game.home_team.name if hasattr(game.home_team, 'name') else "Home Team"
            explanation['away_team'] = game.away_team.name if hasattr(game.away_team, 'name') else "Away Team"
            explanation['date'] = game.date.isoformat() if hasattr(game, 'date') else None
            
            # Add prediction from game
            explanation['game_prediction'] = {
                'winner': game.prediction.winner,
                'confidence': game.prediction.confidence,
                'is_game_of_day': game.prediction.is_game_of_day,
                'is_in_vip_betslip': game.prediction.is_in_vip_betslip
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation for game {game_id}: {str(e)}")
            # Return simplified explanation
            return {
                "game_id": str(game.id),
                "home_team": game.home_team.name if hasattr(game.home_team, 'name') else "Home Team",
                "away_team": game.away_team.name if hasattr(game.away_team, 'name') else "Away Team",
                "prediction": game.prediction.winner,
                "confidence": game.prediction.confidence,
                "text_explanation": f"The model predicts a {game.prediction.winner} win with {game.prediction.confidence:.1%} probability. Detailed explanation is not available due to an error."
            }