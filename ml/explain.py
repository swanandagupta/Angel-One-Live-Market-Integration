import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from features.feature_engineering import FeatureExtractor
from utils.helpers import safe_divide

class ModelExplainer:
    """Provides human-readable factor explanations for ML predictions based on feature importances and contributions."""

    FEATURE_DESCRIPTIONS = {
        "smma20": "SMMA(20) level",
        "smma120": "SMMA(120) level",
        "smma_gap": "SMMA gap width",
        "smma20_slope": "SMMA(20) trajectory slope",
        "smma120_slope": "SMMA(120) trajectory slope",
        "smma_gap_change": "SMMA gap expansion rate",
        "ltp": "Last Traded Price",
        "return_1m": "1-minute price momentum",
        "return_5m": "5-minute price momentum",
        "return_20m": "20-minute price momentum",
        "avg_ltp_20m": "20-minute average price level",
        "avg_ltp_60m": "60-minute average price level",
        "distance_from_avg_20m": "Distance from 20m average price",
        "distance_from_avg_60m": "Distance from 60m average price",
        "ltq": "Last Traded Quantity",
        "avg_ltq_1m": "1-minute LTQ intensity",
        "avg_ltq_2m": "2-minute LTQ intensity",
        "avg_ltq_5m": "5-minute LTQ intensity",
        "avg_ltq_20m": "20-minute LTQ intensity",
        "ltq_2m_to_5m": "Short-term LTQ acceleration ratio",
        "ltq_5m_to_20m": "Medium-term LTQ acceleration ratio",
        "etq_5m": "5-minute Executed Total Quantity",
        "etq_20m": "20-minute Executed Total Quantity",
        "etq_60m": "60-minute Executed Total Quantity",
        "etq_5m_to_20m": "ETQ 5m/20m acceleration ratio",
        "etq_20m_to_60m": "ETQ 20m/60m acceleration ratio",
        "bid_quantity": "Bid-side market depth quantity",
        "ask_quantity": "Ask-side market depth quantity",
        "bid_ask_imbalance": "Order book bid/ask imbalance",
        "spread": "Bid-Ask Spread",
        "relative_spread": "Relative spread ratio",
        "rolling_std_5m": "5-minute price volatility",
        "rolling_std_20m": "20-minute price volatility",
        "signal_buy": "Buy Crossover Direction"
    }

    @staticmethod
    def get_feature_importances(model) -> Dict[str, float]:
        """Extract global feature importances from XGBoost model."""
        if not hasattr(model, "feature_importances_"):
            return {}
        importances = model.feature_importances_
        feature_cols = FeatureExtractor.FEATURE_NAMES
        res = {col: float(imp) for col, imp in zip(feature_cols, importances)}
        # Sort descending
        return dict(sorted(res.items(), key=lambda x: x[1], reverse=True))

    @classmethod
    def explain_prediction(
        cls,
        model,
        features_dict: Dict[str, float],
        probability: float,
        threshold: float = 0.70
    ) -> Dict[str, Any]:
        """
        Generates top positive (+) and negative (-) factor explanations for a single prediction.
        """
        decision = "ACCEPT" if probability >= threshold else "AVOID"
        global_importances = cls.get_feature_importances(model)

        positive_factors = []
        negative_factors = []

        # Analyze key domain signals
        imbalance = features_dict.get("bid_ask_imbalance", 0.0)
        ltq_accel = features_dict.get("ltq_2m_to_5m", 1.0)
        ret_5m = features_dict.get("return_5m", 0.0)
        gap_change = features_dict.get("smma_gap_change", 0.0)
        vol_5m = features_dict.get("rolling_std_5m", 0.0)
        dist_20m = features_dict.get("distance_from_avg_20m", 0.0)

        # 1. Order Book Imbalance
        if imbalance > 0.1:
            positive_factors.append(f"+ Strong bid-side order book dominance ({imbalance:+.2f})")
        elif imbalance < -0.1:
            negative_factors.append(f"- Heavy ask-side selling pressure ({imbalance:+.2f})")

        # 2. LTQ Execution Acceleration
        if ltq_accel > 1.2:
            positive_factors.append(f"+ High LTQ execution intensity acceleration ({ltq_accel:.2f}x)")
        elif ltq_accel < 0.8:
            negative_factors.append(f"- Weak LTQ execution activity ({ltq_accel:.2f}x)")

        # 3. Short-term Price Momentum
        if ret_5m > 0.003:
            positive_factors.append(f"+ Positive 5m price momentum ({ret_5m:+.2%})")
        elif ret_5m < -0.003:
            negative_factors.append(f"- Weak/Negative short-term price return ({ret_5m:+.2%})")

        # 4. SMMA Gap Expansion
        if gap_change > 0:
            positive_factors.append("+ Expanding SMMA gap divergence")
        else:
            negative_factors.append("- Narrowing/flat SMMA gap slope")

        # 5. Volatility Check
        if vol_5m > 0.01:
            negative_factors.append(f"- High short-term price volatility ({vol_5m:.3f})")
        else:
            positive_factors.append("+ Stable, low-volatility price environment")

        # Top feature importance ranking
        top_importances = list(global_importances.items())[:5]

        return {
            "decision": decision,
            "probability": probability,
            "threshold": threshold,
            "positive_factors": positive_factors if positive_factors else ["+ Baseline SMMA crossover alignment"],
            "negative_factors": negative_factors if negative_factors else ["- Moderate factor values"],
            "top_global_features": top_importances
        }
