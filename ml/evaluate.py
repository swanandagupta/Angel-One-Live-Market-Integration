import numpy as np
import pandas as pd
from typing import Dict, Any, List
from utils.helpers import safe_divide

class StrategyEvaluator:
    """Evaluates and compares financial performance of SMMA-only vs SMMA+ML Filter strategy."""

    @staticmethod
    def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates financial quantitative metrics from a dataframe of completed trades."""
        if trades_df.empty or "pnl" not in trades_df.columns:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0
            }

        pnls = trades_df["pnl"].dropna().values
        if len(pnls) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "total_pnl": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0
            }

        total_trades = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = safe_divide(winning_trades, total_trades)

        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0

        avg_profit = safe_divide(gross_profit, winning_trades)
        avg_loss = safe_divide(gross_loss, losing_trades)
        total_pnl = float(np.sum(pnls))
        profit_factor = safe_divide(gross_profit, gross_loss)

        # Max Drawdown calculation over cumulative P/L
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdowns = peak - cum_pnl
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown
        }

    @classmethod
    def compare_strategies(
        cls,
        all_trades_df: pd.DataFrame,
        ml_probabilities: np.ndarray,
        threshold: float = 0.70
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compares Strategy A (SMMA-only) vs Strategy B (SMMA + ML Filter).
        """
        # Strategy A: Take all SMMA crossover trades
        metrics_a = cls.calculate_trade_metrics(all_trades_df)

        # Strategy B: Take only trades where ML probability >= threshold
        if not all_trades_df.empty and len(ml_probabilities) == len(all_trades_df):
            filtered_df = all_trades_df[ml_probabilities >= threshold]
        else:
            filtered_df = pd.DataFrame()

        metrics_b = cls.calculate_trade_metrics(filtered_df)

        return {
            "Strategy_A_SMMA_Only": metrics_a,
            "Strategy_B_SMMA_ML_Filter": metrics_b
        }
