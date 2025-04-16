"""
Feature engineering pipeline for basketball predictions.
"""
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from functools import lru_cache

from app.core.logger import logger
from app.db.models.game import GameInDB
from app.db.models.team import TeamInDB
from app.db.models.statistic import TeamStatisticsModel
from app.db.models.odd import OddsInDB
from server.data.repositories.game_repository import GameRepository
from server.data.repositories.team_repository import TeamRepository
from server.data.repositories.stats_repository import TeamStatsRepository
from server.ml.features.game_features import GameFeatureProcessor
from server.ml.features.team_features import TeamFeatureProcessor
from server.ml.features.historical_features import HistoricalFeatureProcessor
from server.ml.features.odds_features import OddsFeatureProcessor


class FeaturePipeline:
    """
    Orchestrates the feature engineering process for basketball game predictions.
    """
    
    def __init__(self):
        """Initialize the feature pipeline with all processors."""
        # Initialize repositories
        self.game_repository = GameRepository()
        self.team_repository = TeamRepository()
        self.team_stats_repository = TeamStatsRepository()
        
        # Initialize feature processors
        self.game_processor = GameFeatureProcessor()
        self.team_processor = TeamFeatureProcessor()
        self.historical_processor = HistoricalFeatureProcessor()
        self.odds_processor = OddsFeatureProcessor()
        
        # Feature configuration
        self.feature_config = {
            'include_game_features': True,
            'include_team_features': True,
            'include_historical_features': True,
            'include_odds_features': True,
            'lookback_games': 10,  # Number of previous games to consider for historical features
            'team_stats_recency_weight': 0.7,  # Weight recent games more heavily
        }
    
    async def generate_features_for_game(
        self,
        game_id: str,
        include_target: bool = True
    ) -> Dict[str, Any]:
        """
        Generate features for a single game.
        
        Args:
            game_id: MongoDB ID of the game
            include_target: Whether to include the target variable (actual winner)
            
        Returns:
            Dictionary of feature name to value
        """
        # Get the game data
        game = await self.game_repository.find_by_id(game_id)
        if not game:
            logger.error(f"Game not found: {game_id}")
            return {}
        
        # Get team data
        home_team = await self.team_repository.find_by_id(game.home_team.team_id)
        away_team = await self.team_repository.find_by_id(game.away_team.team_id)
        
        if not home_team or not away_team:
            logger.error(f"Teams not found for game {game_id}")
            return {}
        
        # Get team statistics
        home_stats = await self.team_stats_repository.find_by_team_season(
            team_id=game.home_team.team_id,
            season_id=game.season_id
        )
        
        away_stats = await self.team_stats_repository.find_by_team_season(
            team_id=game.away_team.team_id,
            season_id=game.season_id
        )
        
        # Generate features using all processors
        features = {}
        
        # Game features
        if self.feature_config['include_game_features']:
            game_features = await self.game_processor.extract_features(game, home_team, away_team)
            features.update(game_features)
        
        # Team features
        if self.feature_config['include_team_features'] and home_stats and away_stats:
            team_features = await self.team_processor.extract_features(
                home_team=home_team, 
                away_team=away_team,
                home_stats=home_stats,
                away_stats=away_stats
            )
            features.update(team_features)
        
        # Historical features
        if self.feature_config['include_historical_features']:
            historical_features = await self.historical_processor.extract_features(
                game=game,
                lookback_games=self.feature_config['lookback_games']
            )
            features.update(historical_features)
        
        # Odds features
        if self.feature_config['include_odds_features']:
            odds_features = await self.odds_processor.extract_features(game_id)
            features.update(odds_features)
        
        # Add target variable if requested and available
        if include_target and game.status.short in ["FT", "Final", "Finished", "AOT"]:
            # Check scores to determine winner
            if hasattr(game.home_team, 'scores') and hasattr(game.away_team, 'scores'):
                if game.home_team.scores and game.away_team.scores:
                    home_score = game.home_team.scores.total
                    away_score = game.away_team.scores.total
                    
                    if home_score is not None and away_score is not None:
                        features['target'] = 'home' if home_score > away_score else 'away'
        
        # Add metadata (not features but useful for analysis)
        features['game_id'] = str(game.id)
        features['home_team_id'] = game.home_team.team_id
        features['away_team_id'] = game.away_team.team_id
        features['date'] = game.date
        
        return features
    
    async def generate_features_for_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        include_target: bool = True,
        only_completed: bool = False
    ) -> pd.DataFrame:
        """
        Generate features for all games within a date range.
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            include_target: Whether to include the target variable
            only_completed: If True, only include games that have been completed
            
        Returns:
            DataFrame with features for all games
        """
        # Get games within date range
        games = await self.game_repository.find_by_date_range(
            start_date=start_date,
            end_date=end_date
        )
        
        if only_completed:
            completed_statuses = ["FT", "Final", "Finished", "AOT"]
            games = [game for game in games if game.status.short in completed_statuses]
        
        logger.info(f"Generating features for {len(games)} games from {start_date} to {end_date}")
        
        # Generate features for each game
        all_features = []
        for game in games:
            features = await self.generate_features_for_game(
                game_id=str(game.id),
                include_target=include_target
            )
            
            if features:
                all_features.append(features)
        
        # Convert to DataFrame
        if all_features:
            return pd.DataFrame(all_features)
        else:
            return pd.DataFrame()
    
    async def generate_features_for_games(
        self,
        game_ids: List[str],
        include_target: bool = True
    ) -> pd.DataFrame:
        """
        Generate features for a list of games.
        
        Args:
            game_ids: List of MongoDB game IDs
            include_target: Whether to include the target variable
            
        Returns:
            DataFrame with features for all games
        """
        logger.info(f"Generating features for {len(game_ids)} games")
        
        # Generate features for each game
        all_features = []
        for game_id in game_ids:
            features = await self.generate_features_for_game(
                game_id=game_id,
                include_target=include_target
            )
            
            if features:
                all_features.append(features)
        
        # Convert to DataFrame
        if all_features:
            return pd.DataFrame(all_features)
        else:
            return pd.DataFrame()
    
    async def generate_training_data(
        self,
        season_id: str,
        lookback_days: int = 365,
        test_split_date: Optional[datetime] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate a complete dataset for training and testing.
        
        Args:
            season_id: MongoDB season ID to filter games
            lookback_days: Number of days to look back for training data
            test_split_date: Date to use for splitting train/test (None for automatic)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Determine date range
        if test_split_date is None:
            # Use current date as test split by default
            test_split_date = datetime.utcnow()
        
        end_date = test_split_date
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"Generating training data from {start_date} to {end_date}")
        
        # Get all games within date range
        all_games = await self.game_repository.find(
            filter={
                "date": {"$gte": start_date, "$lte": end_date},
                "season_id": season_id,
                "status.short": {"$in": ["FT", "Final", "Finished", "AOT"]}
            },
            sort=[("date", 1)]
        )
        
        logger.info(f"Found {len(all_games)} completed games")
        
        # Generate features for all games
        all_features = []
        for game in all_games:
            features = await self.generate_features_for_game(
                game_id=str(game.id),
                include_target=True
            )
            
            if features and 'target' in features:
                all_features.append(features)
        
        # Convert to DataFrame
        if not all_features:
            logger.warning("No features generated")
            return pd.DataFrame(), pd.DataFrame()
        
        df = pd.DataFrame(all_features)
        
        # Split into train and test based on date
        test_date = test_split_date - timedelta(days=14)  # Use last 14 days as test set
        train_df = df[df['date'] < test_date].copy()
        test_df = df[df['date'] >= test_date].copy()
        
        logger.info(f"Created training set with {len(train_df)} samples and test set with {len(test_df)} samples")
        
        return train_df, test_df
    
    async def transform_raw_game_to_features(
        self,
        game: GameInDB,
        home_team: TeamInDB,
        away_team: TeamInDB,
        home_stats: Optional[TeamStatisticsModel] = None,
        away_stats: Optional[TeamStatisticsModel] = None,
        odds: Optional[OddsInDB] = None
    ) -> Dict[str, Any]:
        """
        Transform a raw game and related data directly into features.
        Useful for generating predictions for upcoming games.
        
        Args:
            game: Game data
            home_team: Home team data
            away_team: Away team data
            home_stats: Optional home team statistics
            away_stats: Optional away team statistics
            odds: Optional odds data
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Game features
        if self.feature_config['include_game_features']:
            game_features = await self.game_processor.extract_features(game, home_team, away_team)
            features.update(game_features)
        
        # Team features
        if self.feature_config['include_team_features'] and home_stats and away_stats:
            team_features = await self.team_processor.extract_features(
                home_team=home_team, 
                away_team=away_team,
                home_stats=home_stats,
                away_stats=away_stats
            )
            features.update(team_features)
        
        # Historical features
        if self.feature_config['include_historical_features']:
            historical_features = await self.historical_processor.extract_features(
                game=game,
                lookback_games=self.feature_config['lookback_games']
            )
            features.update(historical_features)
        
        # Odds features
        if self.feature_config['include_odds_features'] and odds:
            # Use direct odds processing since we have the odds object
            odds_features = self.odds_processor.process_odds(odds)
            features.update(odds_features)
        elif self.feature_config['include_odds_features']:
            # Try to fetch odds if not provided
            odds_features = await self.odds_processor.extract_features(str(game.id))
            features.update(odds_features)
        
        # Add metadata
        features['game_id'] = str(game.id)
        features['home_team_id'] = game.home_team.team_id
        features['away_team_id'] = game.away_team.team_id
        features['date'] = game.date
        
        return features
    
    @lru_cache(maxsize=128)
    def get_feature_names(self) -> List[str]:
        """
        Get a list of all feature names generated by this pipeline.
        
        Returns:
            List of feature names
        """
        # Combine all feature names from all processors
        feature_names = []
        
        if self.feature_config['include_game_features']:
            feature_names.extend(self.game_processor.get_feature_names())
        
        if self.feature_config['include_team_features']:
            feature_names.extend(self.team_processor.get_feature_names())
        
        if self.feature_config['include_historical_features']:
            feature_names.extend(self.historical_processor.get_feature_names())
        
        if self.feature_config['include_odds_features']:
            feature_names.extend(self.odds_processor.get_feature_names())
        
        return feature_names
    
    def preprocess_features(
        self, 
        features_df: pd.DataFrame,
        drop_na: bool = True,
        encode_categorical: bool = True,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Preprocess features for modeling.
        
        Args:
            features_df: DataFrame of features
            drop_na: Whether to drop rows with NA values
            encode_categorical: Whether to one-hot encode categorical features
            normalize: Whether to normalize numerical features
            
        Returns:
            Preprocessed DataFrame
        """
        df = features_df.copy()
        
        # Metadata columns (not features)
        metadata_cols = ['game_id', 'home_team_id', 'away_team_id', 'date', 'target']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Handle missing values
        if drop_na:
            df = df.dropna(subset=feature_cols)
        else:
            # Fill missing values
            numerical_cols = df[feature_cols].select_dtypes(include=np.number).columns
            categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
            
            # Fill numerical columns with median
            for col in numerical_cols:
                df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical columns with mode
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
        # Encode categorical features
        if encode_categorical:
            categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                # Skip columns with too many unique values
                if df[col].nunique() < 10:
                    # One-hot encode
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(col, axis=1)
        
        # Normalize numerical features
        if normalize:
            from sklearn.preprocessing import StandardScaler
            
            # Identify numerical columns (excluding metadata)
            numerical_cols = df.select_dtypes(include=np.number).columns
            numerical_cols = [col for col in numerical_cols if col not in metadata_cols]
            
            if numerical_cols.any():
                scaler = StandardScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def update_feature_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update the feature configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.feature_config.update(config_updates)
        logger.info(f"Updated feature configuration: {self.feature_config}")