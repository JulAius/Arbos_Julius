"""
Consensus gating: combine multiple model predictions into final trading signal.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from ..models.population import Individual, Population
from ..config import ConsensusConfig


class ConsensusGate:
    """
    Gating mechanism that combines predictions from multiple models.

    Requirements:
    - At least MIN_MODELS_AGREE models must agree on direction
    - Average confidence must exceed MIN_CONFIDENCE
    """

    def __init__(
        self,
        min_models_agree: int = ConsensusConfig.MIN_MODELS_AGREE,
        min_confidence: float = ConsensusConfig.MIN_CONFIDENCE
    ):
        self.min_models_agree = min_models_agree
        self.min_confidence = min_confidence

    def decide(
        self,
        models: List[Individual],
        X: pd.DataFrame
    ) -> pd.Series:
        """
        Generate final trading signals (UP=1, DOWN=0, NO_TRADE=-1).

        Args:
            models: list of trained model individuals
            X: feature DataFrame to predict on

        Returns:
            Series of signals (1, 0, or -1) aligned to X.index
        """
        signals = pd.Series(index=X.index, data=0)
        probas_matrix = []

        for ind in models:
            try:
                proba = ind.model.predict_proba(X)
                probas_matrix.append(proba)
            except Exception as e:
                continue

        if not probas_matrix:
            # No model worked, all NO_TRADE
            signals[:] = -1
            return signals

        probas_array = np.array(probas_matrix)  # shape: (n_models, n_samples)

        # Average confidence for each sample
        avg_proba = probas_array.mean(axis=0)

        # Count models predicting UP
        up_votes = (probas_array >= 0.5).sum(axis=0)

        down_votes = len(probas_matrix) - up_votes

        # Apply gating rules
        for i, (avg_p, up, down) in enumerate(zip(avg_proba, up_votes, down_votes)):
            # Check agreement
            if up >= self.min_models_agree:
                direction = 1
                confidence = avg_p
            elif down >= self.min_models_agree:
                direction = 0
                confidence = 1 - avg_p  # confidence in DOWN is 1 - avg_proba
            else:
                signals.iloc[i] = -1
                continue

            # Check confidence threshold
            if confidence >= self.min_confidence:
                signals.iloc[i] = direction
            else:
                signals.iloc[i] = -1

        return signals

    def adjust_thresholds(
        self,
        recent_performance: Dict[str, float],
        target_sharpe: float = 1.0,
        target_profit_factor: float = 1.2,
        target_accuracy: float = 0.65,
        target_bets: float = 90.0
    ):
        """
        Dynamically adjust consensus thresholds based on recent performance.

        Prioritizes risk-adjusted returns (Sharpe) and profitability (profit factor).
        If Sharpe or profit factor are below target, tighten thresholds to improve quality.
        If accuracy is high but Sharpe is low, also tighten (filter out low-quality trades).
        If bets are too low and accuracy acceptable, loosen to increase frequency.
        """
        sharpe = recent_performance.get("sharpe", -1000.0)
        profit_factor = recent_performance.get("profit_factor", 0.0)
        accuracy = recent_performance.get("accuracy", 0.5)
        bets_per_month = recent_performance.get("bets_per_month", 0.0)

        # If we already have sufficient accuracy and activity, avoid tightening thresholds
        # which could zero out trading. This allows the rescoring metric change to take effect.
        if accuracy >= target_accuracy and bets_per_month >= 150:
            # Maintain current thresholds; no adjustment this cycle
            return

        # If extremely few trades (or none), loosen to generate activity
        if bets_per_month < 30.0:
            self.min_confidence = max(0.5, self.min_confidence - 0.05)
            self.min_models_agree = max(1, self.min_models_agree - 1)
            # After loosening for low activity, skip further adjustments this cycle
            self.min_models_agree = max(1, min(5, self.min_models_agree))
            self.min_confidence = max(0.5, min(0.95, self.min_confidence))
            return

        # Primary: Sharpe and profitability
        if sharpe < target_sharpe or profit_factor < target_profit_factor:
            # Poor risk-adjusted/absolute performance: tighten to filter trades
            self.min_confidence = min(0.9, self.min_confidence + 0.05)
            self.min_models_agree = min(5, self.min_models_agree + 1)
        elif accuracy < target_accuracy:
            # Directional accuracy insufficient: tighten to require stronger agreement
            self.min_confidence = min(0.9, self.min_confidence + 0.03)
        elif bets_per_month < target_bets and accuracy >= target_accuracy:
            # Too few trades despite decent accuracy: loosen thresholds
            self.min_confidence = max(0.5, self.min_confidence - 0.02)
            self.min_models_agree = max(1, self.min_models_agree - 1)
        # else: keep thresholds stable

        # Clamp to reasonable ranges
        self.min_models_agree = max(1, min(5, self.min_models_agree))
        self.min_confidence = max(0.5, min(0.95, self.min_confidence))

    def get_params(self) -> dict:
        return {
            "min_models_agree": self.min_models_agree,
            "min_confidence": self.min_confidence
        }
