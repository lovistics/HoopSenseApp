"""
HoopsIQ service for AI-powered scenario analysis.
"""
from typing import List, Optional, Dict, Any, Tuple

from fastapi import HTTPException, status

from app.core.logger import logger
from app.db.models.game import GameInDB, GamePrediction, AnalysisFactor
from server.data.repositories.game_repository import GameRepository


class HoopsIQService:
    """Service for HoopsIQ AI analysis."""
    
    def __init__(self):
        """Initialize the HoopsIQ service with repositories."""
        self.game_repository = GameRepository()
    
    async def get_insights(self, game_id: str) -> List[Dict[str, Any]]:
        """
        Get AI-generated insights for a game.
        
        Args:
            game_id: The ID of the game
            
        Returns:
            List of insights
            
        Raises:
            HTTPException: If game not found
        """
        # Get the game and its prediction
        game = await self.game_repository.find_by_id(game_id)
        if not game:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
            
        if not game.prediction:
            return []
            
        # Generate insights based on prediction factors
        insights = []
        
        # Add insight based on home/away advantage
        home_away_insight = {
            "insight": "",
            "category": "venue",
            "confidence": 80,
            "source": "historical data"
        }
        
        if game.prediction.predicted_winner == "home":
            home_away_insight["insight"] = f"{game.home_team.name} has a strong home court advantage that significantly impacts this matchup."
        else:
            home_away_insight["insight"] = f"{game.away_team.name} tends to perform well on the road, offsetting the home court advantage."
            
        insights.append(home_away_insight)
        
        # Add insights based on confidence level
        confidence_insight = {
            "insight": "",
            "category": "prediction confidence",
            "confidence": game.prediction.confidence,
            "source": "prediction model"
        }
        
        winner = game.home_team.name if game.prediction.predicted_winner == "home" else game.away_team.name
        
        if game.prediction.confidence > 75:
            confidence_insight["insight"] = f"This is a high-confidence prediction favoring {winner}."
        elif game.prediction.confidence > 60:
            confidence_insight["insight"] = f"The model has moderate confidence in a {winner} win."
        else:
            confidence_insight["insight"] = f"This is a close matchup, but the model slightly favors {winner}."
            
        insights.append(confidence_insight)
        
        # Add insights based on analysis factors if available
        if game.prediction.analysis_factors:
            for factor in game.prediction.analysis_factors:
                factor_insight = {
                    "insight": f"{factor.description} This contributes {abs(factor.impact)}% to the prediction.",
                    "category": "key factor",
                    "confidence": min(90, abs(int(factor.impact)) * 5),
                    "source": "statistical analysis"
                }
                insights.append(factor_insight)
        
        return insights
    
    async def analyze_scenario(
        self,
        game_id: str, 
        query: str
    ) -> Dict[str, Any]:
        """
        Analyze a what-if scenario for a game.
        
        Args:
            game_id: The ID of the game
            query: The scenario query
            
        Returns:
            Scenario analysis results
            
        Raises:
            HTTPException: If game not found or has no prediction
        """
        # Get the game and its original prediction
        game = await self.game_repository.find_by_id(game_id)
        if not game:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Game not found"
            )
            
        if not game.prediction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No prediction available for this game"
            )
            
        # Store the original probability
        original_probability = game.prediction.home_win_probability * 100
        
        # Parse the query to determine scenario type
        scenario_type, adjustments = self._parse_scenario_query(query, game)
        
        # Apply adjustments to the prediction
        new_probability = original_probability
        impacted_areas = []
        
        for adjustment in adjustments:
            new_probability += adjustment["impact"]
            impacted_areas.append({
                "factor": adjustment["factor"],
                "impact": adjustment["impact"]
            })
        
        # Ensure probability is within valid range
        new_probability = max(5, min(95, new_probability))
        
        # Generate explanation based on scenario
        explanation = self._generate_scenario_explanation(
            scenario_type, 
            adjustments, 
            game
        )
        
        # Return the adjusted prediction with explanation
        return {
            "originalProbability": original_probability,
            "newProbability": new_probability,
            "explanation": explanation,
            "impactedAreas": impacted_areas
        }
    
    @staticmethod
    def _parse_scenario_query(
        query: str, 
        game: GameInDB
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse a scenario query to determine adjustments.
        
        Args:
            query: The scenario query
            game: The game
            
        Returns:
            Tuple of (scenario_type, adjustments)
        """
        # Normalize query to lowercase
        query_lower = query.lower()
        
        # Initialize variables
        scenario_type = "unknown"
        adjustments = []
        
        # Check for player-related scenarios
        if any(term in query_lower for term in ["player", "star", "injury", "injured", "out"]):
            scenario_type = "player_absence"
            
            # Determine which team is affected
            home_team_mentioned = any(term in query_lower for term in [
                game.home_team.name.lower(), 
                game.home_team.name.split()[-1].lower() if " " in game.home_team.name else ""
            ])
            
            away_team_mentioned = any(term in query_lower for term in [
                game.away_team.name.lower(), 
                game.away_team.name.split()[-1].lower() if " " in game.away_team.name else ""
            ])
            
            if home_team_mentioned:
                # Star player absence for home team
                adjustments.append({
                    "factor": "Star Player Impact",
                    "impact": -8.5
                })
                
                # Also affects supporting cast
                adjustments.append({
                    "factor": "Team Chemistry",
                    "impact": -3.2
                })
                
                # Might improve the bench in some scenarios
                adjustments.append({
                    "factor": "Bench Performance",
                    "impact": 2.5
                })
            elif away_team_mentioned:
                # Star player absence for away team
                adjustments.append({
                    "factor": "Star Player Impact",
                    "impact": 7.8
                })
                
                # Also affects supporting cast
                adjustments.append({
                    "factor": "Team Chemistry",
                    "impact": 2.9
                })
                
                # Might improve the bench in some scenarios
                adjustments.append({
                    "factor": "Bench Performance",
                    "impact": -2.1
                })
            else:
                # Generic player absence scenario
                adjustments.append({
                    "factor": "Player Availability",
                    "impact": -4.0
                })
        
        # Check for shooting-related scenarios
        elif any(term in query_lower for term in ["shoot", "shooting", "percentage", "3-point", "three point", "three-point"]):
            scenario_type = "shooting_performance"
            
            better_shooting = any(term in query_lower for term in ["better", "good", "great", "high", "improve"])
            worse_shooting = any(term in query_lower for term in ["worse", "bad", "poor", "low", "decrease"])
            
            # Determine which team's shooting is affected
            home_team_mentioned = any(term in query_lower for term in [
                game.home_team.name.lower(), 
                game.home_team.name.split()[-1].lower() if " " in game.home_team.name else ""
            ])
            
            away_team_mentioned = any(term in query_lower for term in [
                game.away_team.name.lower(), 
                game.away_team.name.split()[-1].lower() if " " in game.away_team.name else ""
            ])
            
            if home_team_mentioned and better_shooting:
                adjustments.append({
                    "factor": "Shooting Efficiency",
                    "impact": 6.8
                })
                
                if "three" in query_lower or "3-point" in query_lower:
                    adjustments.append({
                        "factor": "Three-Point Shooting",
                        "impact": 7.9
                    })
            elif home_team_mentioned and worse_shooting:
                adjustments.append({
                    "factor": "Shooting Efficiency",
                    "impact": -6.5
                })
                
                if "three" in query_lower or "3-point" in query_lower:
                    adjustments.append({
                        "factor": "Three-Point Shooting",
                        "impact": -8.1
                    })
            elif away_team_mentioned and better_shooting:
                adjustments.append({
                    "factor": "Opponent Shooting",
                    "impact": -6.3
                })
                
                if "three" in query_lower or "3-point" in query_lower:
                    adjustments.append({
                        "factor": "Opponent Three-Point Shooting",
                        "impact": -7.7
                    })
            elif away_team_mentioned and worse_shooting:
                adjustments.append({
                    "factor": "Opponent Shooting",
                    "impact": 6.1
                })
                
                if "three" in query_lower or "3-point" in query_lower:
                    adjustments.append({
                        "factor": "Opponent Three-Point Shooting",
                        "impact": 7.4
                    })
            else:
                # Generic shooting scenario
                if better_shooting:
                    adjustments.append({
                        "factor": "Overall Shooting",
                        "impact": 4.5
                    })
                elif worse_shooting:
                    adjustments.append({
                        "factor": "Overall Shooting",
                        "impact": -4.3
                    })
                else:
                    adjustments.append({
                        "factor": "Shooting Variability",
                        "impact": 2.1
                    })
        
        # Check for pace-related scenarios
        elif any(term in query_lower for term in ["pace", "tempo", "fast", "slow"]):
            scenario_type = "game_pace"
            
            faster_pace = any(term in query_lower for term in ["fast", "high", "up", "increase"])
            slower_pace = any(term in query_lower for term in ["slow", "low", "down", "decrease"])
            
            # Determine which team benefits from pace change
            home_team_pace_advantage = game.prediction.home_win_probability > 0.5
            
            if faster_pace:
                if home_team_pace_advantage:
                    adjustments.append({
                        "factor": "Game Pace",
                        "impact": 3.8
                    })
                else:
                    adjustments.append({
                        "factor": "Game Pace",
                        "impact": -3.5
                    })
            elif slower_pace:
                if home_team_pace_advantage:
                    adjustments.append({
                        "factor": "Game Pace",
                        "impact": -3.2
                    })
                else:
                    adjustments.append({
                        "factor": "Game Pace",
                        "impact": 3.0
                    })
            else:
                adjustments.append({
                    "factor": "Pace Variability",
                    "impact": 1.5
                })
                
        # Check for bench-related scenarios
        elif any(term in query_lower for term in ["bench", "second unit", "reserves", "rotation"]):
            scenario_type = "bench_performance"
            
            better_bench = any(term in query_lower for term in ["better", "good", "strong", "improve"])
            worse_bench = any(term in query_lower for term in ["worse", "bad", "weak", "decrease"])
            
            # Determine which team's bench is affected
            home_team_mentioned = any(term in query_lower for term in [
                game.home_team.name.lower(), 
                game.home_team.name.split()[-1].lower() if " " in game.home_team.name else ""
            ])
            
            away_team_mentioned = any(term in query_lower for term in [
                game.away_team.name.lower(), 
                game.away_team.name.split()[-1].lower() if " " in game.away_team.name else ""
            ])
            
            if home_team_mentioned and better_bench:
                adjustments.append({
                    "factor": "Bench Production",
                    "impact": 5.2
                })
            elif home_team_mentioned and worse_bench:
                adjustments.append({
                    "factor": "Bench Production",
                    "impact": -4.8
                })
            elif away_team_mentioned and better_bench:
                adjustments.append({
                    "factor": "Opponent Bench Production",
                    "impact": -4.9
                })
            elif away_team_mentioned and worse_bench:
                adjustments.append({
                    "factor": "Opponent Bench Production",
                    "impact": 4.5
                })
            else:
                # Generic bench scenario
                if better_bench:
                    adjustments.append({
                        "factor": "Overall Bench Impact",
                        "impact": 3.2
                    })
                elif worse_bench:
                    adjustments.append({
                        "factor": "Overall Bench Impact",
                        "impact": -3.0
                    })
                else:
                    adjustments.append({
                        "factor": "Bench Variability",
                        "impact": 1.8
                    })
        
        # Check for close game scenarios
        elif any(term in query_lower for term in ["close", "tight", "clutch", "down to the wire"]):
            scenario_type = "close_game"
            
            # Teams with higher win probability often have more experience in close games
            if game.prediction.home_win_probability > 0.65 or game.prediction.home_win_probability < 0.35:
                # Strong favorite in a close game means reduced advantage
                adjustments.append({
                    "factor": "Close Game Dynamics",
                    "impact": -8.5 if game.prediction.home_win_probability > 0.5 else 8.5
                })
                
                adjustments.append({
                    "factor": "Pressure Handling",
                    "impact": -4.2 if game.prediction.home_win_probability > 0.5 else 4.2
                })
            else:
                # Already close matchup gets even closer
                adjustments.append({
                    "factor": "Close Game Dynamics",
                    "impact": -2.5 if game.prediction.home_win_probability > 0.5 else 2.5
                })
                
                adjustments.append({
                    "factor": "Home Court Pressure",
                    "impact": 3.8
                })
        
        # If no specific scenario was identified, provide a generic adjustment
        if not adjustments:
            scenario_type = "general_adjustment"
            adjustments.append({
                "factor": "General Game Variability",
                "impact": -4.0 if game.prediction.home_win_probability > 0.5 else 4.0
            })
            
        return scenario_type, adjustments
    
    @staticmethod
    def _generate_scenario_explanation(
        scenario_type: str,
        adjustments: List[Dict[str, Any]],
        game: GameInDB
    ) -> str:
        """
        Generate explanation text for the scenario.
        
        Args:
            scenario_type: The type of scenario
            adjustments: The adjustments
            game: The game
            
        Returns:
            Explanation text
        """
        home_team = game.home_team.name
        away_team = game.away_team.name
        
        # Calculate the total impact
        total_impact = sum(adj["impact"] for adj in adjustments)
        
        # Determine which team benefits from the scenario
        benefits_home = total_impact > 0
        
        # Generate an explanation based on the scenario type
        if scenario_type == "player_absence":
            player_team = home_team if any(adj["impact"] < 0 for adj in adjustments) else away_team
            opposing_team = away_team if player_team == home_team else home_team
            
            return (
                f"If a key player is out for {player_team}, it would significantly impact their "
                f"offensive efficiency and team chemistry. Teams often see decreased shooting "
                f"percentages and ball movement when missing a star player. This scenario would "
                f"benefit {opposing_team}, changing the win probability by approximately "
                f"{abs(total_impact):.1f}%."
            )
            
        elif scenario_type == "shooting_performance":
            better_team = home_team if benefits_home else away_team
            worse_team = away_team if benefits_home else home_team
            
            if "Three-Point" in str(adjustments) or "three-point" in str(adjustments).lower():
                return (
                    f"A significant change in three-point shooting would have a major impact on this matchup. "
                    f"If {better_team} shoots better than their season average from beyond the arc, "
                    f"their win probability increases by approximately {abs(total_impact):.1f}%. "
                    f"Three-point variance is one of the highest impact factors in modern basketball."
                )
            else:
                return (
                    f"Changes in overall shooting efficiency would shift this matchup considerably. "
                    f"If {better_team} shoots better than their season average while {worse_team} "
                    f"remains consistent, the win probability shifts by approximately {abs(total_impact):.1f}%. "
                    f"Field goal percentage is strongly correlated with game outcomes."
                )
                
        elif scenario_type == "game_pace":
            better_team = home_team if benefits_home else away_team
            worse_team = away_team if benefits_home else home_team
            
            faster_pace = any("up" in adj["factor"].lower() or "increase" in adj["factor"].lower() for adj in adjustments)
            
            if faster_pace:
                return (
                    f"A faster pace would likely benefit {better_team} in this matchup. "
                    f"They average more possessions per game and have a higher offensive "
                    f"efficiency in transition. This scenario would change the win probability "
                    f"by approximately {abs(total_impact):.1f}% in their favor."
                )
            else:
                return (
                    f"A slower, half-court oriented game would likely benefit {better_team}. "
                    f"They are more efficient in set plays and have better half-court defense "
                    f"compared to {worse_team}. This pace change would shift the win probability "
                    f"by approximately {abs(total_impact):.1f}%."
                )
                
        elif scenario_type == "bench_performance":
            better_team = home_team if benefits_home else away_team
            worse_team = away_team if benefits_home else home_team
            
            return (
                f"The bench contribution could be a significant factor in this game. "
                f"If {better_team}'s second unit outperforms expectations, they gain an advantage "
                f"of approximately {abs(total_impact):.1f}% in win probability. Bench scoring "
                f"and defensive intensity often impact momentum swings throughout the game."
            )
            
        elif scenario_type == "close_game":
            better_team = home_team if benefits_home else away_team
            
            return (
                f"In a close game scenario, various factors like free throw shooting, timeout usage, "
                f"and clutch performance become magnified. {better_team} has a slight edge in "
                f"these situations based on season performance metrics. This changes the win "
                f"probability by approximately {abs(total_impact):.1f}% compared to the baseline prediction."
            )
            
        else:  # general_adjustment
            better_team = home_team if benefits_home else away_team
            
            return (
                f"Based on your query, I analyzed potential variations in game conditions and their "
                f"impact on this matchup. General game variability factors tend to favor {better_team} "
                f"slightly in this specific scenario, shifting the win probability by approximately "
                f"{abs(total_impact):.1f}% from the baseline prediction."
            )