"""
Explanation module for basketball game predictions.

This module provides methods to generate human-readable explanations
for model predictions using SHAP (SHapley Additive exPlanations) values
and other interpretability techniques.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import shap
from collections import defaultdict

from app.core.logger import logger
from app.ml.models.xgboost_model import BasketballXGBoostModel


class PredictionExplainer:
    """
    Generates human-readable explanations for basketball game predictions.
    Uses SHAP values to determine feature importance and explain predictions.
    """
    
    def __init__(self, model: Optional[BasketballXGBoostModel] = None):
        """
        Initialize the prediction explainer.
        
        Args:
            model: Optional pre-loaded model
        """
        self.model = model
        self.explainer = None
        self.feature_groups = self._define_feature_groups()
    
    def set_model(self, model: BasketballXGBoostModel) -> None:
        """
        Set the model to explain predictions for.
        
        Args:
            model: The model to use for explanations
        """
        self.model = model
        self.explainer = None  # Reset explainer when model changes
    
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """
        Define groups of related features for explanation purposes.
        
        Returns:
            Dictionary mapping group names to lists of feature patterns
        """
        return {
            "team_offense": ["team_home_off_", "team_away_off_", "team_home_points_pg", 
                           "team_away_points_pg", "team_ts_pct"],
            "team_defense": ["team_home_def_", "team_away_def_", "team_home_points_allowed_pg", 
                           "team_away_points_allowed_pg"],
            "team_form": ["team_home_form", "team_away_form", "team_form_diff", 
                        "team_home_win_pct", "team_away_win_pct"],
            "matchup": ["team_net_style_advantage", "team_same_conference", 
                      "team_home_team_underdog", "team_team_strength_gap"],
            "betting_odds": ["odds_home_implied_prob", "odds_away_implied_prob", 
                           "odds_implied_prob_diff", "odds_market_win_prob"],
            "market_sentiment": ["odds_home_is_favorite", "odds_bookmaker_agreement", 
                               "odds_market_confidence"]
        }
    
    def _initialize_explainer(self, background_data: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize the SHAP explainer.
        
        Args:
            background_data: Optional DataFrame to use as background data for the explainer
        """
        if not self.model or not hasattr(self.model, 'model') or self.model.model is None:
            raise ValueError("Model must be set before initializing explainer")
        
        # If no background data provided, create a simple one with zeros
        if background_data is None and self.model.feature_names:
            background_data = pd.DataFrame(
                np.zeros((1, len(self.model.feature_names))),
                columns=self.model.feature_names
            )
        
        if self.model.feature_names and background_data is not None:
            try:
                # Initialize SHAP TreeExplainer
                self.explainer = shap.TreeExplainer(self.model.model)
                logger.info("SHAP explainer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
                self.explainer = None
    
    def explain_prediction(
        self, 
        features: pd.DataFrame,
        include_shap_values: bool = False,
        num_top_features: int = 5
    ) -> Dict[str, Any]:
        """
        Generate an explanation for a prediction.
        
        Args:
            features: DataFrame with features for the prediction
            include_shap_values: Whether to include raw SHAP values in the output
            num_top_features: Number of top features to include in the explanation
            
        Returns:
            Dictionary with explanation information
        """
        if not self.model or not hasattr(self.model, 'model') or self.model.model is None:
            raise ValueError("Model must be set before generating explanations")
        
        # Initialize explainer if not already done
        if self.explainer is None:
            self._initialize_explainer(features)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Get SHAP values
        try:
            # Convert features to format expected by SHAP
            feature_matrix = features[self.model.feature_names]
            shap_values = self.explainer.shap_values(feature_matrix)
            
            # Handle different SHAP return types
            if isinstance(shap_values, list):
                # For multi-class models, we take the values for the predicted class
                if prediction == self.model.output_classes[1]:
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    shap_values = shap_values[0]
            
            # Get base value (expected value)
            expected_value = self.explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1] if prediction == self.model.output_classes[1] else expected_value[0]
            
            # Prepare feature importance
            feature_importance = {}
            for i, feature in enumerate(feature_matrix.columns):
                feature_importance[feature] = float(shap_values[0][i])
            
            # Sort by absolute value for overall importance
            sorted_importance = sorted(
                feature_importance.items(), 
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            top_features = sorted_importance[:num_top_features]
            
            # Group features by type
            grouped_explanation = self._group_features_by_type(feature_importance)
            
            # Generate text explanation
            text_explanation = self._generate_text_explanation(
                prediction, 
                probability, 
                top_features, 
                grouped_explanation
            )
            
            # Build result
            explanation = {
                "prediction": prediction,
                "probability": float(probability),
                "text_explanation": text_explanation,
                "top_features": dict(top_features),
                "feature_groups": grouped_explanation,
            }
            
            # Include raw SHAP values if requested
            if include_shap_values:
                explanation["shap_values"] = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
                explanation["expected_value"] = float(expected_value)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {str(e)}")
            
            # Fallback to a simpler explanation
            return {
                "prediction": prediction,
                "probability": float(probability),
                "text_explanation": self._generate_fallback_explanation(prediction, probability),
                "error": str(e)
            }
    
    def _group_features_by_type(self, feature_importance: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Group features by their type/category for better explanations.
        
        Args:
            feature_importance: Dictionary of feature names to importance values
            
        Returns:
            Dictionary with grouped features and their aggregate importance
        """
        groups = {}
        
        # Initialize group sums
        for group_name in self.feature_groups:
            groups[group_name] = {
                "importance": 0.0,
                "features": {},
                "direction": 0  # 1 for positive, -1 for negative, 0 for neutral
            }
        
        # Assign features to groups and sum importance
        assigned_features = set()
        
        for feature, importance in feature_importance.items():
            for group_name, patterns in self.feature_groups.items():
                for pattern in patterns:
                    if pattern in feature:
                        groups[group_name]["importance"] += abs(importance)
                        groups[group_name]["features"][feature] = importance
                        groups[group_name]["direction"] += (1 if importance > 0 else -1)
                        assigned_features.add(feature)
                        break
        
        # Normalize direction to -1, 0, or 1
        for group_name in groups:
            if groups[group_name]["direction"] > 0:
                groups[group_name]["direction"] = 1
            elif groups[group_name]["direction"] < 0:
                groups[group_name]["direction"] = -1
        
        # Add "other" group for non-categorized features
        groups["other"] = {
            "importance": 0.0,
            "features": {},
            "direction": 0
        }
        
        for feature, importance in feature_importance.items():
            if feature not in assigned_features:
                groups["other"]["importance"] += abs(importance)
                groups["other"]["features"][feature] = importance
                groups["other"]["direction"] += (1 if importance > 0 else -1)
        
        # Normalize other direction
        if groups["other"]["direction"] > 0:
            groups["other"]["direction"] = 1
        elif groups["other"]["direction"] < 0:
            groups["other"]["direction"] = -1
            
        # Sort groups by importance
        return dict(sorted(groups.items(), key=lambda x: x[1]["importance"], reverse=True))
    
    def _generate_text_explanation(
        self,
        prediction: Any,
        probability: float,
        top_features: List[Tuple[str, float]],
        grouped_explanation: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate a human-readable explanation of the prediction.
        
        Args:
            prediction: The predicted outcome
            probability: The prediction probability
            top_features: List of top features and their importance
            grouped_explanation: Grouped feature importance
            
        Returns:
            Human-readable explanation
        """
        # Determine predicted winner
        winner = None
        confidence = ""
        
        if self.model and hasattr(self.model, "output_classes") and self.model.output_classes:
            if prediction == self.model.output_classes[1]:  # Home team wins
                winner = "home team"
            else:
                winner = "away team"
                
            # Confidence level
            if probability > 0.8:
                confidence = "high confidence"
            elif probability > 0.65:
                confidence = "moderate confidence"
            else:
                confidence = "low confidence"
        else:
            # Fallback
            winner = "home team" if str(prediction).lower() in ["1", "home", "true"] else "away team"
            confidence = "uncertain" if 0.4 < probability < 0.6 else "some confidence"
        
        # Create explanation
        explanation = [
            f"The model predicts a {winner} win with {confidence} ({probability:.1%} probability).",
            "\nKey factors influencing this prediction:"
        ]
        
        # Add top group explanations
        groups_added = 0
        for group_name, group_data in grouped_explanation.items():
            # Skip groups with minimal importance
            if group_data["importance"] < 0.01 or not group_data["features"]:
                continue
                
            if groups_added >= 3:  # Limit to top 3 groups
                break
                
            direction = "favoring the home team" if group_data["direction"] > 0 else "favoring the away team"
            if group_data["direction"] == 0:
                direction = "with mixed influence"
                
            if group_name == "team_offense":
                explanation.append(f"• Offensive performance {direction} (relative strength in scoring, shooting efficiency)")
            elif group_name == "team_defense":
                explanation.append(f"• Defensive metrics {direction} (points allowed, defensive rating)")
            elif group_name == "team_form":
                explanation.append(f"• Recent team form {direction} (win percentage, recent game results)")
            elif group_name == "matchup":
                explanation.append(f"• Head-to-head matchup factors {direction} (team styles, conference matchup)")
            elif group_name == "betting_odds":
                explanation.append(f"• Betting market assessment {direction} (implied win probabilities, market expectations)")
            elif group_name == "market_sentiment":
                explanation.append(f"• Market sentiment {direction} (bookmaker consensus, public perception)")
            else:
                explanation.append(f"• Other factors {direction} (miscellaneous statistics)")
                
            groups_added += 1
            
        # Add specific feature insights for the top features
        explanation.append("\nSpecific insights:")
        for feature, importance in top_features[:3]:  # Limit to top 3 features
            direction = "increases" if importance > 0 else "decreases"
            feature_name = self._get_readable_feature_name(feature)
            explanation.append(f"• {feature_name} {direction} home team win probability")
            
        return "\n".join(explanation)
    
    def _get_readable_feature_name(self, feature: str) -> str:
        """
        Convert a feature name to a human-readable format.
        
        Args:
            feature: Raw feature name
            
        Returns:
            Human-readable feature name
        """
        # Replace prefixes
        readable = feature.replace("team_", "").replace("odds_", "")
        
        # Replace underscores with spaces
        readable = readable.replace("_", " ")
        
        # Special case mappings
        mappings = {
            "home off rating": "Home team offensive rating",
            "away off rating": "Away team offensive rating",
            "home def rating": "Home team defensive rating",
            "away def rating": "Away team defensive rating",
            "net rating diff": "Net rating difference",
            "home points pg": "Home team points per game",
            "away points pg": "Away team points per game",
            "home implied prob": "Home team implied probability",
            "away implied prob": "Away team implied probability",
            "implied prob diff": "Implied probability difference",
            "home win pct": "Home team win percentage",
            "away win pct": "Away team win percentage",
            "win pct diff": "Win percentage difference",
            "home is favorite": "Home team is favorite",
            "bookmaker agreement": "Bookmaker agreement",
            "market confidence": "Market confidence",
            "home form": "Home team recent form",
            "away form": "Away team recent form",
            "form diff": "Form difference",
            "home team underdog": "Home team is underdog"
        }
        
        # Try exact match first
        if readable in mappings:
            return mappings[readable]
        
        # Try partial matches
        for key, value in mappings.items():
            if key in readable:
                return value
        
        # Capitalize first letter of each word if no mapping found
        return readable.title()
    
    def _generate_fallback_explanation(self, prediction: Any, probability: float) -> str:
        """
        Generate a simple explanation when SHAP explanation fails.
        
        Args:
            prediction: The predicted outcome
            probability: The prediction probability
            
        Returns:
            Simple explanation
        """
        # Determine predicted winner
        winner = None
        confidence = ""
        
        if self.model and hasattr(self.model, "output_classes") and self.model.output_classes:
            if prediction == self.model.output_classes[1]:  # Home team wins
                winner = "home team"
            else:
                winner = "away team"
                
            # Confidence level
            if probability > 0.8:
                confidence = "high confidence"
            elif probability > 0.65:
                confidence = "moderate confidence"
            else:
                confidence = "low confidence"
        else:
            # Fallback
            winner = "home team" if str(prediction).lower() in ["1", "home", "true"] else "away team"
            confidence = "uncertain" if 0.4 < probability < 0.6 else "some confidence"
        
        return (
            f"The model predicts a {winner} win with {confidence} ({probability:.1%} probability). "
            "Detailed explanation is not available for this prediction."
        )
    
    def get_explanations_for_games(
        self, 
        game_features: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate explanations for multiple games.
        
        Args:
            game_features: Dictionary mapping game IDs to feature DataFrames
            
        Returns:
            Dictionary mapping game IDs to explanations
        """
        explanations = {}
        
        for game_id, features in game_features.items():
            try:
                explanation = self.explain_prediction(features)
                explanations[game_id] = explanation
            except Exception as e:
                logger.error(f"Failed to generate explanation for game {game_id}: {str(e)}")
                explanations[game_id] = {
                    "error": str(e),
                    "text_explanation": "Explanation could not be generated due to an error."
                }
        
        return explanations