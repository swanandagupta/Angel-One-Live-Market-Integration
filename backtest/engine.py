import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from config import Config
from utils.logger import logger
from broker.models import Candle
from data.historical_loader import HistoricalDataLoader
from ml.dataset import DatasetBuilder
from ml.predict import Predictor
from ml.evaluate import StrategyEvaluator

class BacktestEngine:
    """Event-driven and vectorized backtest simulation engine comparing SMMA-only vs SMMA+ML Filter."""

    def __init__(self, predictor: Optional[Predictor] = None):
        self.predictor = predictor or Predictor()

    def run_backtest(
        self,
        candles_by_symbol: Optional[Dict[str, List[Candle]]] = None,
        threshold: float = Config.ML_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Runs comprehensive backtest over candle history.
        Evaluates SMMA crossovers, calculates point-in-time features, gets ML probabilities,
        and computes Strategy A vs Strategy B financial metrics.
        """
        if candles_by_symbol is None:
            loader = HistoricalDataLoader()
            _, candles_by_symbol = loader.load_ticks_and_candles()

        # Build crossover dataset with point-in-time features and trade P/L outcomes
        df_crossovers = DatasetBuilder.build_dataset_from_candles(candles_by_symbol)

        if df_crossovers.empty:
            logger.warning("No crossover events generated during backtest.")
            empty_metrics = StrategyEvaluator.calculate_trade_metrics(pd.DataFrame())
            return {
                "crossovers_df": pd.DataFrame(),
                "metrics_smma_only": empty_metrics,
                "metrics_smma_ml": empty_metrics,
                "comparison": {}
            }

        # Sort chronologically to enforce point-in-time zero look-ahead constraint
        df_crossovers = df_crossovers.sort_values(by="timestamp").reset_index(drop=True)

        probabilities = []
        decisions = []

        # Evaluate ML prediction for each crossover using point-in-time features
        for _, row in df_crossovers.iterrows():
            feats = row.to_dict()
            sig = str(row.get("signal", "BUY"))
            
            dec, prob, th_used, status, model_name = self.predictor.predict(feats, signal_dir=sig, override_threshold=threshold)
            probabilities.append(prob if prob is not None else 0.0)
            decisions.append(dec)

        df_crossovers["ml_probability"] = probabilities
        df_crossovers["ml_decision"] = decisions

        probs_arr = np.array(probabilities)
        comparison = StrategyEvaluator.compare_strategies(df_crossovers, probs_arr, threshold=threshold)

        logger.info(
            f"Backtest completed over {len(df_crossovers)} crossovers. "
            f"SMMA-Only P/L: {comparison['Strategy_A_SMMA_Only']['total_pnl']:.2f}, "
            f"SMMA+ML P/L: {comparison['Strategy_B_SMMA_ML_Filter']['total_pnl']:.2f}"
        )

        return {
            "crossovers_df": df_crossovers,
            "metrics_smma_only": comparison["Strategy_A_SMMA_Only"],
            "metrics_smma_ml": comparison["Strategy_B_SMMA_ML_Filter"],
            "comparison": comparison
        }
