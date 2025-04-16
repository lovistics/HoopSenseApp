"""
Game selection module for basketball predictions.

This module handles the selection of games for VIP betslips and 
Game of the Day recommendations, applying selection criteria and
confidence thresholds.
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.core.logger import logger
from app.db.models.game import GameInDB, GamePrediction
from app.ml.prediction.predictor import BasketballPredictor


class GameSelector:
    """
    Selects games for VIP betslips and Game of the Day based on prediction quality.
    """
    
    def __init__(self, predictor: Optional[BasketballPredictor] = None):
        """
        Initialize the game selector.
        
        Args:
            predictor: Optional predictor instance
        """
        self.predictor = predictor or BasketballPredictor()
        self.min_confidence_betslip = 0.65  # Minimum confidence for VIP betslip inclusion
        self.min_confidence_game_of_day = 0.75  # Minimum confidence for Game of the Day
    
    def set_confidence_thresholds(
        self,
        betslip_threshold: float = 0.65,
        game_of_day_threshold: float = 0.75
    ) -> None:
        """
        Set confidence thresholds for selection.
        
        Args:
            betslip_threshold: Minimum confidence for VIP betslip inclusion
            game_of_day_threshold: Minimum confidence for Game of the Day
        """
        self.min_confidence_betslip = betslip_threshold
        self.min_confidence_game_of_day = game_of_day_threshold
        logger.info(f"Selection thresholds updated: betslip={betslip_threshold}, game_of_day={game_of_day_threshold}")
    
    async def select_betslip_games(
        self,
        date: datetime,
        max_games: int = 3
    ) -> List[GameInDB]:
        """
        Select games for the VIP betslip on a specific date.
        
        Args:
            date: Date to select games for
            max_games: Maximum number of games to include
            
        Returns:
            List of selected games
        """
        # Import here to avoid circular imports
        from app.data.repositories.game_repository import GameRepository
        
        # Get games for the date
        game_repo = GameRepository()
        date_str = date.strftime("%Y-%m-%d")
        games = await game_repo.find_by_date(date_str)
        
        if not games:
            logger.info(f"No games found for date {date_str}")
            return []
        
        # Get predictions for all games
        predictions = await self.predictor.batch_predict(games)
        
        # Filter games with sufficient confidence
        confident_games = []
        for game in games:
            game_id = str(game.id)
            if game_id in predictions:
                prediction = predictions[game_id]
                if prediction.confidence >= self.min_confidence_betslip:
                    # Add prediction to game
                    game.prediction = prediction
                    confident_games.append(game)
        
        # Sort by confidence (descending)
        confident_games.sort(key=lambda g: g.prediction.confidence, reverse=True)
        
        # Select top games
        selected_games = confident_games[:max_games]
        
        # Mark selected games as part of VIP betslip
        for game in selected_games:
            game.prediction.is_in_vip_betslip = True
        
        logger.info(f"Selected {len(selected_games)} games for VIP betslip on {date_str}")
        return selected_games
    
    async def select_game_of_day(
        self,
        date: datetime
    ) -> Optional[GameInDB]:
        """
        Select Game of the Day for a specific date.
        
        Args:
            date: Date to select game for
            
        Returns:
            Selected game or None if no suitable game found
        """
        # Import here to avoid circular imports
        from app.data.repositories.game_repository import GameRepository
        
        # Get games for the date
        game_repo = GameRepository()
        date_str = date.strftime("%Y-%m-%d")
        games = await game_repo.find_by_date(date_str)
        
        if not games:
            logger.info(f"No games found for date {date_str}")
            return None
        
        # Get predictions for all games
        predictions = await self.predictor.batch_predict(games)
        
        # Find best candidate
        best_game = None
        best_confidence = 0.0
        
        for game in games:
            game_id = str(game.id)
            if game_id in predictions:
                prediction = predictions[game_id]
                
                # Apply selection criteria
                if prediction.confidence >= self.min_confidence_game_of_day:
                    if prediction.confidence > best_confidence:
                        # Add prediction to game
                        game.prediction = prediction
                        best_game = game
                        best_confidence = prediction.confidence
        
        if best_game:
            # Mark as Game of the Day
            best_game.prediction.is_game_of_day = True
            logger.info(f"Selected Game of the Day for {date_str}: {best_game.home_team.name} vs {best_game.away_team.name} (confidence: {best_confidence:.2f})")
        else:
            logger.info(f"No suitable Game of the Day found for {date_str}")
        
        return best_game
    
    async def analyze_historical_performance(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Analyze the historical performance of selection criteria.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with performance metrics
        """
        # Import here to avoid circular imports
        from app.data.repositories.game_repository import GameRepository
        
        # Get completed games in date range
        game_repo = GameRepository()
        games = await game_repo.find_completed_in_date_range(start_date, end_date)
        
        if not games:
            logger.info(f"No completed games found between {start_date} and {end_date}")
            return {"games_analyzed": 0}
        
        # Analyze performance at different confidence thresholds
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        performance = {
            "games_analyzed": len(games),
            "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "thresholds": {}
        }
        
        # Get predictions for all games
        predictions = await self.predictor.batch_predict(games, refresh_features=True)
        
        # For each confidence threshold
        for threshold in thresholds:
            correct = 0
            total = 0
            
            for game in games:
                game_id = str(game.id)
                if game_id in predictions:
                    prediction = predictions[game_id]
                    
                    if prediction.confidence >= threshold:
                        total += 1
                        
                        # Check if prediction was correct
                        if hasattr(game, 'winner'):
                            if game.winner == prediction.winner:
                                correct += 1
            
            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0
            
            # Store metrics
            performance["thresholds"][str(threshold)] = {
                "total_predictions": total,
                "correct_predictions": correct,
                "accuracy": accuracy,
                "games_per_day_avg": total / (end_date - start_date).days if (end_date - start_date).days > 0 else 0
            }
        
        # Add recommended thresholds based on analysis
        best_accuracy = 0.0
        best_threshold = 0.65  # Default
        
        for threshold, metrics in performance["thresholds"].items():
            if metrics["total_predictions"] >= 50:  # Only consider thresholds with enough samples
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    best_threshold = float(threshold)
        
        performance["recommended_thresholds"] = {
            "betslip": max(0.60, best_threshold - 0.05),  # More inclusive for betslip
            "game_of_day": best_threshold  # Use best threshold for Game of the Day
        }
        
        logger.info(f"Historical analysis complete. Recommended betslip threshold: {performance['recommended_thresholds']['betslip']}, Game of the Day threshold: {performance['recommended_thresholds']['game_of_day']}")
        
        return performance
    
    async def optimize_selection_criteria(
        self,
        lookback_days: int = 30
    ) -> None:
        """
        Optimize selection criteria based on recent performance.
        
        Args:
            lookback_days: Number of days to analyze
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Analyze performance
        performance = await self.analyze_historical_performance(start_date, end_date)
        
        # Update thresholds if analysis was successful
        if "recommended_thresholds" in performance:
            self.set_confidence_thresholds(
                betslip_threshold=performance["recommended_thresholds"]["betslip"],
                game_of_day_threshold=performance["recommended_thresholds"]["game_of_day"]
            )