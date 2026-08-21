import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from utils.helpers import safe_divide

class FeatureExtractor:
    """Computes quantitative features at exact crossover timestamps without future data leakage."""

    # Ordered list of feature names used by ML model
    FEATURE_NAMES = [
        "smma20", "smma120", "smma_gap", "smma20_slope", "smma120_slope", "smma_gap_change",
        "ltp", "return_1m", "return_5m", "return_20m",
        "avg_ltp_20m", "avg_ltp_60m", "distance_from_avg_20m", "distance_from_avg_60m",
        "ltq", "avg_ltq_1m", "avg_ltq_2m", "avg_ltq_5m", "avg_ltq_20m",
        "ltq_2m_to_5m", "ltq_5m_to_20m",
        "etq_5m", "etq_20m", "etq_60m", "etq_5m_to_20m", "etq_20m_to_60m",
        "bid_quantity", "ask_quantity", "bid_ask_imbalance", "spread", "relative_spread",
        "rolling_std_5m", "rolling_std_20m",
        "signal_buy"
    ]

    # Genuine historical features calculable strictly from OHLC candle data (17 features)
    HISTORICAL_CANDLE_FEATURE_NAMES = [
        "smma20", "smma120", "smma_gap", "smma20_slope", "smma120_slope", "smma_gap_change",
        "ltp", "return_1m", "return_5m", "return_20m",
        "avg_ltp_20m", "avg_ltp_60m", "distance_from_avg_20m", "distance_from_avg_60m",
        "rolling_std_5m", "rolling_std_20m",
        "signal_buy"
    ]

    @staticmethod
    def extract_features(
        signal: str,
        ltp: float,
        smma20_curr: float,
        smma120_curr: float,
        smma20_prev: Optional[float],
        smma120_prev: Optional[float],
        close_prices: List[float],
        tick_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Extracts complete quantitative feature dictionary at crossover timestamp.
        """
        # SMMA calculations
        smma_gap = smma20_curr - smma120_curr
        smma20_prev_val = smma20_prev if smma20_prev is not None else smma20_curr
        smma120_prev_val = smma120_prev if smma120_prev is not None else smma120_curr
        
        smma20_slope = safe_divide(smma20_curr - smma20_prev_val, smma20_prev_val)
        smma120_slope = safe_divide(smma120_curr - smma120_prev_val, smma120_prev_val)
        smma_gap_prev = smma20_prev_val - smma120_prev_val
        smma_gap_change = smma_gap - smma_gap_prev

        # Price returns over historical candle close prices
        price_1m = close_prices[-2] if len(close_prices) >= 2 else ltp
        price_5m = close_prices[-6] if len(close_prices) >= 6 else close_prices[0] if close_prices else ltp
        price_20m = close_prices[-21] if len(close_prices) >= 21 else close_prices[0] if close_prices else ltp

        return_1m = safe_divide(ltp - price_1m, price_1m)
        return_5m = safe_divide(ltp - price_5m, price_5m)
        return_20m = safe_divide(ltp - price_20m, price_20m)

        # Volatility (rolling std dev of 1m returns)
        returns = []
        if len(close_prices) >= 2:
            arr = np.array(close_prices)
            pct = np.diff(arr) / arr[:-1]
            returns = list(pct)

        rolling_std_5m = float(np.std(returns[-5:])) if len(returns) >= 2 else 0.0
        rolling_std_20m = float(np.std(returns[-20:])) if len(returns) >= 2 else 0.0

        # Extract tick store metrics with safe defaults
        avg_ltp_20m = tick_metrics.get("avg_ltp_20m", ltp)
        avg_ltp_60m = tick_metrics.get("avg_ltp_60m", ltp)
        distance_from_avg_20m = safe_divide(ltp - avg_ltp_20m, avg_ltp_20m)
        distance_from_avg_60m = safe_divide(ltp - avg_ltp_60m, avg_ltp_60m)

        ltq = tick_metrics.get("ltq", 100.0)
        avg_ltq_1m = tick_metrics.get("avg_ltq_1m", ltq)
        avg_ltq_2m = tick_metrics.get("avg_ltq_2m", ltq)
        avg_ltq_5m = tick_metrics.get("avg_ltq_5m", ltq)
        avg_ltq_20m = tick_metrics.get("avg_ltq_20m", ltq)

        ltq_2m_to_5m = tick_metrics.get("ltq_2m_to_5m", safe_divide(avg_ltq_2m, avg_ltq_5m))
        ltq_5m_to_20m = tick_metrics.get("ltq_5m_to_20m", safe_divide(avg_ltq_5m, avg_ltq_20m))

        etq_5m = tick_metrics.get("etq_5m", avg_ltq_5m * 5)
        etq_20m = tick_metrics.get("etq_20m", avg_ltq_20m * 20)
        etq_60m = tick_metrics.get("etq_60m", avg_ltq_20m * 60)

        etq_5m_to_20m = tick_metrics.get("etq_5m_to_20m", safe_divide(etq_5m, safe_divide(etq_20m, 4.0)))
        etq_20m_to_60m = tick_metrics.get("etq_20m_to_60m", safe_divide(etq_20m, safe_divide(etq_60m, 3.0)))

        bid_q = tick_metrics.get("bid_quantity", 1_000_000.0)
        ask_q = tick_metrics.get("ask_quantity", 1_000_000.0)
        bid_ask_imbalance = tick_metrics.get("bid_ask_imbalance", safe_divide(bid_q - ask_q, bid_q + ask_q))

        bid_p = tick_metrics.get("bid_price", ltp * 0.999)
        ask_p = tick_metrics.get("ask_price", ltp * 1.001)
        spread = tick_metrics.get("spread", max(0.0, ask_p - bid_p))
        relative_spread = tick_metrics.get("relative_spread", safe_divide(spread, ltp))

        signal_buy = 1.0 if signal.upper() == "BUY" else 0.0

        features = {
            "smma20": float(smma20_curr),
            "smma120": float(smma120_curr),
            "smma_gap": float(smma_gap),
            "smma20_slope": float(smma20_slope),
            "smma120_slope": float(smma120_slope),
            "smma_gap_change": float(smma_gap_change),
            "ltp": float(ltp),
            "return_1m": float(return_1m),
            "return_5m": float(return_5m),
            "return_20m": float(return_20m),
            "avg_ltp_20m": float(avg_ltp_20m),
            "avg_ltp_60m": float(avg_ltp_60m),
            "distance_from_avg_20m": float(distance_from_avg_20m),
            "distance_from_avg_60m": float(distance_from_avg_60m),
            "ltq": float(ltq),
            "avg_ltq_1m": float(avg_ltq_1m),
            "avg_ltq_2m": float(avg_ltq_2m),
            "avg_ltq_5m": float(avg_ltq_5m),
            "avg_ltq_20m": float(avg_ltq_20m),
            "ltq_2m_to_5m": float(ltq_2m_to_5m),
            "ltq_5m_to_20m": float(ltq_5m_to_20m),
            "etq_5m": float(etq_5m),
            "etq_20m": float(etq_20m),
            "etq_60m": float(etq_60m),
            "etq_5m_to_20m": float(etq_5m_to_20m),
            "etq_20m_to_60m": float(etq_20m_to_60m),
            "bid_quantity": float(bid_q),
            "ask_quantity": float(ask_q),
            "bid_ask_imbalance": float(bid_ask_imbalance),
            "spread": float(spread),
            "relative_spread": float(relative_spread),
            "rolling_std_5m": float(rolling_std_5m),
            "rolling_std_20m": float(rolling_std_20m),
            "signal_buy": float(signal_buy)
        }
        return features

    @classmethod
    def features_to_vector(cls, features_dict: Dict[str, float]) -> List[float]:
        """Convert feature dict to ordered float list matching FEATURE_NAMES."""
        return [features_dict.get(name, 0.0) for name in cls.FEATURE_NAMES]
