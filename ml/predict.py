import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from config import Config
from utils.logger import logger
from features.feature_engineering import FeatureExtractor

class Predictor:
    """
    Production inference engine supporting direction-specific XGBoost models:
    - BUY Model: xgboost_buy_model_v2.joblib (Threshold = 0.55)
    - SELL Model: xgboost_sell_model_v2.joblib (Threshold = 0.70)
    """

    BUY_MODEL_PATH = Config.MODELS_DIR / "xgboost_buy_model_v2.joblib"
    SELL_MODEL_PATH = Config.MODELS_DIR / "xgboost_sell_model_v2.joblib"
    UNIFIED_MODEL_PATH = Config.MODEL_V2_PATH
    FALLBACK_MODEL_PATH = Config.MODEL_PATH

    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.70

    def __init__(self):
        self.buy_model = None
        self.sell_model = None
        self.unified_model = None
        self.load_models()

    def load_models(self) -> bool:
        """Load direction-specific production models and fallback models."""
        loaded_any = False
        
        # Load BUY Model
        if self.BUY_MODEL_PATH.exists():
            try:
                self.buy_model = joblib.load(self.BUY_MODEL_PATH)
                logger.info(f"Loaded production BUY ML model from {self.BUY_MODEL_PATH}")
                loaded_any = True
            except Exception as e:
                logger.error(f"Failed to load BUY ML model from {self.BUY_MODEL_PATH}: {e}")

        # Load SELL Model
        if self.SELL_MODEL_PATH.exists():
            try:
                self.sell_model = joblib.load(self.SELL_MODEL_PATH)
                logger.info(f"Loaded production SELL ML model from {self.SELL_MODEL_PATH}")
                loaded_any = True
            except Exception as e:
                logger.error(f"Failed to load SELL ML model from {self.SELL_MODEL_PATH}: {e}")

        # Load Unified Fallback Model
        fallback_path = self.UNIFIED_MODEL_PATH if self.UNIFIED_MODEL_PATH.exists() else self.FALLBACK_MODEL_PATH
        if fallback_path.exists():
            try:
                self.unified_model = joblib.load(fallback_path)
                logger.info(f"Loaded unified fallback ML model from {fallback_path}")
                loaded_any = True
            except Exception as e:
                logger.error(f"Failed to load unified ML model: {e}")

        return loaded_any

    def is_model_trained(self) -> bool:
        return (self.buy_model is not None or self.sell_model is not None or self.unified_model is not None)

    def predict(
        self,
        features_dict: Dict[str, float],
        signal_dir: str = "BUY",
        override_threshold: Optional[float] = None
    ) -> Tuple[str, Optional[float], float, str, str]:
        """
        Generates direction-specific ML prediction.
        Returns: (decision, probability, threshold_used, status_message, model_name)
        """
        if not self.is_model_trained():
            return "MODEL_UNAVAILABLE", None, Config.ML_THRESHOLD, "MODEL UNAVAILABLE", "N/A"

        sig_upper = signal_dir.upper()
        
        # Select directional model & threshold
        if sig_upper == "BUY" and self.buy_model is not None:
            active_model = self.buy_model
            default_thresh = self.BUY_THRESHOLD
            model_name = "BUY-XGB-V2"
        elif sig_upper == "SELL" and self.sell_model is not None:
            active_model = self.sell_model
            default_thresh = self.SELL_THRESHOLD
            model_name = "SELL-XGB-V2"
        elif self.unified_model is not None:
            active_model = self.unified_model
            default_thresh = Config.ML_THRESHOLD
            model_name = "UNIFIED-XGB"
        else:
            return "MODEL_UNAVAILABLE", None, Config.ML_THRESHOLD, "MODEL UNAVAILABLE", "N/A"

        thresh = override_threshold if override_threshold is not None else default_thresh

        try:
            # Determine feature schema
            n_expected = getattr(active_model, "n_features_in_", 17)
            if n_expected == 17:
                expected_names = FeatureExtractor.HISTORICAL_CANDLE_FEATURE_NAMES
            elif n_expected == 16:
                expected_names = [f for f in FeatureExtractor.HISTORICAL_CANDLE_FEATURE_NAMES if f != "signal_buy"]
            else:
                expected_names = FeatureExtractor.FEATURE_NAMES

            # Build feature vector & sanitize NaN/Inf
            vector = []
            for name in expected_names:
                val = features_dict.get(name, 0.0)
                if val is None or np.isnan(val) or np.isinf(val):
                    logger.warning(f"Feature '{name}' contained NaN/Inf ({val}). Replacing with 0.0.")
                    vector.append(0.0)
                else:
                    vector.append(float(val))

            df_input = pd.DataFrame([vector], columns=expected_names)
            
            # Execute model predict_proba
            prob_arr = active_model.predict_proba(df_input)
            
            # Dynamically resolve positive class (target=1) index
            classes = list(getattr(active_model, "classes_", [0, 1]))
            if 1 in classes:
                pos_idx = classes.index(1)
            elif len(classes) == 1:
                pos_idx = 0 if classes[0] == 1 else -1
            else:
                pos_idx = 1 if len(classes) > 1 else 0

            if pos_idx >= 0 and pos_idx < prob_arr.shape[1]:
                probability = float(prob_arr[0][pos_idx])
            else:
                probability = 0.0

            decision = "ACCEPT" if probability >= thresh else "AVOID"
            status = f"{decision} ({probability:.1%} confidence vs threshold {thresh:.1%})"
            return decision, probability, thresh, status, model_name
        except Exception as e:
            logger.error(f"Inference error during ML prediction: {e}")
            return "ML_ERROR", None, thresh, f"ML ERROR: {e}", model_name

    def explain_prediction(
        self,
        features_dict: Dict[str, float],
        signal_dir: str,
        probability: Optional[float],
        decision: str,
        threshold: Optional[float] = None
    ) -> str:
        """Generates deterministic quantitative feature explanation string with threshold breakdown."""
        if probability is None or decision in ["MODEL_UNAVAILABLE", "ML_ERROR"]:
            return "MODEL UNAVAILABLE"

        sig_upper = signal_dir.upper()
        effective_thresh = threshold if threshold is not None else (self.BUY_THRESHOLD if sig_upper == "BUY" else self.SELL_THRESHOLD)
        thresh_pct = f"{effective_thresh:.0%}"
        prob_pct = f"{probability:.1%}"

        s20 = features_dict.get("smma20", 0.0)
        s120 = features_dict.get("smma120", 0.0)
        ret5 = features_dict.get("return_5m", 0.0)
        slope120 = features_dict.get("smma120_slope", 0.0)
        gap = features_dict.get("smma_gap", 0.0)

        reasons = []
        if sig_upper == "BUY":
            if s20 > s120:
                reasons.append(f"SMMA20 ({s20:.2f}) > SMMA120 ({s120:.2f})")
            if ret5 > 0:
                reasons.append(f"Positive 5m Return ({ret5:+.2%})")
            if slope120 > 0:
                reasons.append("Positive SMMA120 Slope")
            if gap > 0:
                reasons.append(f"Expanding Gap (+{gap:.2f})")

            reason_str = " | ".join(reasons) if reasons else "Trend Alignment"
            if decision == "ACCEPT":
                return f"BUY probability {prob_pct} >= threshold {thresh_pct} ({reason_str})"
            else:
                return f"BUY probability {prob_pct} < threshold {thresh_pct} ({reason_str})"
        else:
            if s20 < s120:
                reasons.append(f"SMMA20 ({s20:.2f}) < SMMA120 ({s120:.2f})")
            if ret5 < 0:
                reasons.append(f"Negative 5m Return ({ret5:+.2%})")
            if slope120 < 0:
                reasons.append("Negative SMMA120 Slope")
            if gap < 0:
                reasons.append(f"Widening Downward Gap ({gap:.2f})")

            reason_str = " | ".join(reasons) if reasons else "Trend Alignment"
            if decision == "ACCEPT":
                return f"SELL probability {prob_pct} >= threshold {thresh_pct} ({reason_str})"
            else:
                return f"SELL probability {prob_pct} < threshold {thresh_pct} ({reason_str})"
