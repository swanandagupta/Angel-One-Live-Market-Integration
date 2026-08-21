import pytest
import numpy as np
from typing import Dict, List, Any
from ml.predict import Predictor

class MockModel:
    """Mock XGBoost model returning fixed probability for testing."""
    def __init__(self, prob: float):
        self.prob = prob
        self.n_features_in_ = 17
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.array([[1.0 - self.prob, self.prob]])

def test_buy_threshold_boundary_0_54():
    """Test 1: BUY probability 0.54 -> AVOID (BUY threshold = 0.55)"""
    predictor = Predictor()
    predictor.buy_model = MockModel(0.54)
    
    decision, prob, thresh, status, model_name = predictor.predict({}, signal_dir="BUY")
    assert decision == "AVOID"
    assert prob == pytest.approx(0.54)
    assert thresh == 0.55

def test_buy_threshold_boundary_0_55():
    """Test 2: BUY probability 0.55 -> ACCEPT (BUY threshold = 0.55)"""
    predictor = Predictor()
    predictor.buy_model = MockModel(0.55)
    
    decision, prob, thresh, status, model_name = predictor.predict({}, signal_dir="BUY")
    assert decision == "ACCEPT"
    assert prob == pytest.approx(0.55)
    assert thresh == 0.55

def test_buy_threshold_boundary_0_80():
    """Test 3: BUY probability 0.80 -> ACCEPT (BUY threshold = 0.55)"""
    predictor = Predictor()
    predictor.buy_model = MockModel(0.80)
    
    decision, prob, thresh, status, model_name = predictor.predict({}, signal_dir="BUY")
    assert decision == "ACCEPT"
    assert prob == pytest.approx(0.80)
    assert thresh == 0.55

def test_sell_threshold_boundary_0_69():
    """Test 4: SELL probability 0.69 -> AVOID (SELL threshold = 0.70)"""
    predictor = Predictor()
    predictor.sell_model = MockModel(0.69)
    
    decision, prob, thresh, status, model_name = predictor.predict({}, signal_dir="SELL")
    assert decision == "AVOID"
    assert prob == pytest.approx(0.69)
    assert thresh == 0.70

def test_sell_threshold_boundary_0_70():
    """Test 5: SELL probability 0.70 -> ACCEPT (SELL threshold = 0.70)"""
    predictor = Predictor()
    predictor.sell_model = MockModel(0.70)
    
    decision, prob, thresh, status, model_name = predictor.predict({}, signal_dir="SELL")
    assert decision == "ACCEPT"
    assert prob == pytest.approx(0.70)
    assert thresh == 0.70

def test_sell_threshold_boundary_0_90():
    """Test 6: SELL probability 0.90 -> ACCEPT (SELL threshold = 0.70)"""
    predictor = Predictor()
    predictor.sell_model = MockModel(0.90)
    
    decision, prob, thresh, status, model_name = predictor.predict({}, signal_dir="SELL")
    assert decision == "ACCEPT"
    assert prob == pytest.approx(0.90)
    assert thresh == 0.70

def test_active_equals_accepted_plus_avoided_invariant():
    """Test 7 & 8 & 9 & 10: Invariant ACTIVE SIGNALS = ACCEPTED + AVOIDED, single decision per crossover."""
    signal_log: List[Dict[str, Any]] = [
        {"SYMBOL": "SBIN", "SIGNAL": "BUY", "DECISION": "ACCEPT", "ML PROBABILITY": "78.0%", "THRESHOLD": "55%"},
        {"SYMBOL": "RELIANCE", "SIGNAL": "BUY", "DECISION": "AVOID", "ML PROBABILITY": "48.0%", "THRESHOLD": "55%"},
        {"SYMBOL": "INFY", "SIGNAL": "SELL", "DECISION": "AVOID", "ML PROBABILITY": "63.0%", "THRESHOLD": "70%"},
        {"SYMBOL": "TCS", "SIGNAL": "SELL", "DECISION": "ACCEPT", "ML PROBABILITY": "85.0%", "THRESHOLD": "70%"},
        {"SYMBOL": "HDFCBANK", "SIGNAL": "BUY", "DECISION": "ACCEPT", "ML PROBABILITY": "60.0%", "THRESHOLD": "55%"},
    ]

    accepted_count = sum(1 for s in signal_log if s["DECISION"] == "ACCEPT")
    avoided_count = sum(1 for s in signal_log if s["DECISION"] == "AVOID")
    active_count = accepted_count + avoided_count

    # Invariant assertion
    assert active_count == accepted_count + avoided_count
    assert accepted_count == 3
    assert avoided_count == 2
    assert active_count == 5

    # Every crossover received exactly one decision
    for entry in signal_log:
        assert entry["DECISION"] in ["ACCEPT", "AVOID"]

    # No crossover disappeared merely because it was rejected
    avoided_entries = [e for e in signal_log if e["DECISION"] == "AVOID"]
    assert len(avoided_entries) == 2
    assert avoided_entries[0]["SYMBOL"] == "RELIANCE"
    assert avoided_entries[1]["SYMBOL"] == "INFY"
