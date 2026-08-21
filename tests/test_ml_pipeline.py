import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from broker.models import Candle
from data.historical_loader import HistoricalDataLoader
from ml.dataset import DatasetBuilder
from ml.train import ModelTrainer
from ml.predict import Predictor
from ml.explain import ModelExplainer
from features.feature_engineering import FeatureExtractor

def test_dataset_builder_from_candles():
    loader = HistoricalDataLoader()
    _, candles_by_symbol = loader.load_ticks_and_candles()
    df = DatasetBuilder.build_dataset_from_candles(candles_by_symbol)

    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "target" in df.columns
        assert "smma20" in df.columns
        assert "smma120" in df.columns
        assert "signal" in df.columns

def test_model_trainer_and_predictor(tmp_path):
    model_file = tmp_path / "test_xgb_model.joblib"
    trainer = ModelTrainer(model_path=model_file)
    
    loader = HistoricalDataLoader()
    _, candles_by_symbol = loader.load_ticks_and_candles()
    df_dataset = DatasetBuilder.build_dataset_from_candles(candles_by_symbol)

    model, metrics = trainer.train_model(df_dataset)

    assert model_file.exists()
    assert "accuracy" in metrics

    predictor = Predictor()
    assert predictor.is_model_trained() == True

    feats = {
        "smma20": 105.0, "smma120": 100.0, "smma_gap": 5.0,
        "smma20_slope": 0.01, "smma120_slope": 0.002, "smma_gap_change": 0.5,
        "ltp": 105.0, "return_1m": 0.005, "return_5m": 0.01, "return_20m": 0.02,
        "avg_ltp_20m": 102.0, "avg_ltp_60m": 100.0,
        "distance_from_avg_20m": 0.029, "distance_from_avg_60m": 0.05,
        "ltq": 1000.0, "avg_ltq_1m": 1000.0, "avg_ltq_2m": 1000.0,
        "avg_ltq_5m": 800.0, "avg_ltq_20m": 600.0,
        "ltq_2m_to_5m": 1.25, "ltq_5m_to_20m": 1.33,
        "etq_5m": 5000.0, "etq_20m": 12000.0, "etq_60m": 36000.0,
        "etq_5m_to_20m": 1.66, "etq_20m_to_60m": 1.0,
        "bid_quantity": 1_500_000.0, "ask_quantity": 1_100_000.0,
        "bid_ask_imbalance": 0.15, "spread": 0.10, "relative_spread": 0.001,
        "rolling_std_5m": 0.002, "rolling_std_20m": 0.005,
        "signal_buy": 1.0
    }

    decision, prob, th_used, status, model_name = predictor.predict(feats, signal_dir="BUY")
    assert decision in ["ACCEPT", "AVOID"]
    assert prob is not None
    assert 0.0 <= prob <= 1.0

def test_controlled_model_predict_proba_mapping():
    """Controlled unit test verifying direct predict_proba vs Predictor mapping."""
    predictor = Predictor()
    assert predictor.is_model_trained() == True

    sample_feats = {name: 1.0 for name in FeatureExtractor.FEATURE_NAMES}
    decision, prob, th_used, status, model_name = predictor.predict(sample_feats, signal_dir="BUY")
    assert prob is not None
    assert 0.0 <= prob <= 1.0

def test_predictor_unfitted_fallback(tmp_path):
    predictor = Predictor()
    predictor.buy_model = None
    predictor.sell_model = None
    predictor.unified_model = None
    assert predictor.is_model_trained() == False

    decision, prob, th_used, status, model_name = predictor.predict({}, signal_dir="BUY")
    assert decision == "MODEL_UNAVAILABLE"
    assert prob is None
