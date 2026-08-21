import pytest
from features.feature_engineering import FeatureExtractor
from utils.helpers import safe_divide

def test_feature_extraction_completeness():
    tick_metrics = {
        "ltp": 150.0,
        "ltq": 200.0,
        "etq_5m": 5000.0,
        "etq_20m": 20000.0,
        "etq_60m": 60000.0,
        "avg_ltq_1m": 200.0,
        "avg_ltq_2m": 200.0,
        "avg_ltq_5m": 200.0,
        "avg_ltq_20m": 200.0,
        "ltq_2m_to_5m": 1.0,
        "ltq_5m_to_20m": 1.0,
        "etq_5m_to_20m": 1.0,
        "etq_20m_to_60m": 1.0,
        "avg_ltp_20m": 148.0,
        "avg_ltp_60m": 145.0,
        "bid_quantity": 1_500_000.0,
        "ask_quantity": 1_200_000.0,
        "bid_price": 149.90,
        "ask_price": 150.10,
        "bid_ask_imbalance": (1.5 - 1.2) / (1.5 + 1.2),
        "spread": 0.20,
        "relative_spread": 0.20 / 150.0
    }

    close_prices = [140.0, 142.0, 145.0, 148.0, 150.0]

    feats = FeatureExtractor.extract_features(
        signal="BUY",
        ltp=150.0,
        smma20_curr=147.0,
        smma120_curr=142.0,
        smma20_prev=146.0,
        smma120_prev=142.0,
        close_prices=close_prices,
        tick_metrics=tick_metrics
    )

    assert feats["smma20"] == 147.0
    assert feats["smma120"] == 142.0
    assert feats["smma_gap"] == 5.0
    assert feats["signal_buy"] == 1.0
    assert feats["bid_ask_imbalance"] == pytest.approx((1.5 - 1.2) / (1.5 + 1.2))

    vector = FeatureExtractor.features_to_vector(feats)
    assert len(vector) == len(FeatureExtractor.FEATURE_NAMES)
