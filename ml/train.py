import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from config import Config
from utils.logger import logger
from features.feature_engineering import FeatureExtractor
from data.historical_loader import HistoricalDataLoader
from ml.dataset import DatasetBuilder

class ModelTrainer:
    """Trains XGBoost v2 model using strictly 17 candle-verifiable features and chronological splits."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Config.MODEL_V2_PATH
        self.meta_path = Config.MODELS_DIR / "model_metadata_v2.json"

    def train_model(self, df_dataset: Optional[pd.DataFrame] = None) -> Tuple[XGBClassifier, Dict[str, float]]:
        v2_csv = Config.DATA_STORAGE_DIR / "ml_dataset_v2.csv"
        
        if df_dataset is None or df_dataset.empty:
            if v2_csv.exists():
                logger.info(f"Loading ML dataset v2 from {v2_csv}")
                df_dataset = pd.read_csv(v2_csv)
            else:
                logger.info("Building dataset v2 from historical candles...")
                loader = HistoricalDataLoader()
                ticks, candles = loader.load_ticks_and_candles()
                df_dataset = DatasetBuilder.build_dataset_v2(candles)

        # Sort chronologically to prevent temporal data leakage
        df_dataset = df_dataset.sort_values(by="timestamp").reset_index(drop=True)

        feature_cols = FeatureExtractor.HISTORICAL_CANDLE_FEATURE_NAMES
        for col in feature_cols:
            if col not in df_dataset.columns:
                df_dataset[col] = 0.0

        X = df_dataset[feature_cols].copy()
        y = df_dataset["target"].astype(int)

        n_samples = len(X)
        train_end = int(n_samples * 0.70)
        val_end = int(n_samples * 0.85)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        # Class balance ratio for scale_pos_weight
        spw = (y_train == 0).sum() / float((y_train == 1).sum()) if (y_train == 1).sum() > 0 else 1.0

        logger.info(f"Temporal split prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        # Optimal XGBoost hyper-parameters selected via Validation Grid Search
        model = XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            min_child_weight=1,
            scale_pos_weight=spw,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss"
        )

        eval_set = [(X_val, y_val)] if len(X_val) > 0 and len(np.unique(y_val)) > 1 else None
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Evaluate metrics on Test set (thresh = 0.55)
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.55).astype(int)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
            "precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, test_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, test_preds, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
            "confusion_matrix": confusion_matrix(y_test, test_preds).tolist()
        }

        # Save trained v2 model and default model paths for compatibility
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(model, Config.MODEL_PATH)

        meta = {
            "model_type": "XGBClassifier",
            "feature_names": feature_cols,
            "feature_count": len(feature_cols),
            "training_dataset": str(v2_csv.absolute()),
            "training_date": pd.Timestamp.now().isoformat(),
            "hyperparameters": {
                "max_depth": 3, "learning_rate": 0.1, "n_estimators": 100,
                "min_child_weight": 1, "scale_pos_weight": round(spw, 4),
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0
            },
            "test_metrics": metrics,
            "selected_threshold": 0.55,
            "class_mapping": {0: "Loss", 1: "Profit"}
        }

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Successfully trained and saved v2 model to {self.model_path}. Metrics: {metrics}")
        return model, metrics

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model()
