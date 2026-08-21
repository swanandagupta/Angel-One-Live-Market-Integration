import pytest
from datetime import datetime
from strategy.crossover import CrossoverDetector

def test_buy_crossover_signal():
    detector = CrossoverDetector()
    ts1 = datetime(2026, 8, 20, 10, 0)
    ts2 = datetime(2026, 8, 20, 10, 1)

    # Initial state: prev_smma20 <= prev_smma120
    event1 = detector.update("SBIN", ts1, ltp=100.0, smma20=45.0, smma120=50.0)
    assert event1 is None

    # Transition: smma20 > smma120 -> Should emit BUY Crossover
    event2 = detector.update("SBIN", ts2, ltp=102.0, smma20=52.0, smma120=50.0)
    assert event2 is not None
    assert event2.signal == "BUY"
    assert event2.symbol == "SBIN"

def test_sell_crossover_signal():
    detector = CrossoverDetector()
    ts1 = datetime(2026, 8, 20, 10, 0)
    ts2 = datetime(2026, 8, 20, 10, 1)

    # Initial state: prev_smma20 >= prev_smma120
    event1 = detector.update("RELIANCE", ts1, ltp=2500.0, smma20=2520.0, smma120=2500.0)
    assert event1 is None

    # Transition: smma20 < smma120 -> Should emit SELL Crossover
    event2 = detector.update("RELIANCE", ts2, ltp=2480.0, smma20=2490.0, smma120=2500.0)
    assert event2 is not None
    assert event2.signal == "SELL"
    assert event2.symbol == "RELIANCE"

def test_duplicate_crossover_prevention():
    detector = CrossoverDetector()
    ts1 = datetime(2026, 8, 20, 10, 0)
    ts2 = datetime(2026, 8, 20, 10, 1)
    ts3 = datetime(2026, 8, 20, 10, 2)

    detector.update("INFY", ts1, ltp=1500.0, smma20=1490.0, smma120=1500.0)
    
    # First crossover -> BUY
    e2 = detector.update("INFY", ts2, ltp=1510.0, smma20=1505.0, smma120=1500.0)
    assert e2 is not None and e2.signal == "BUY"

    # Subsequent tick where SMMA20 stays above SMMA120 -> Should NOT re-emit BUY
    e3 = detector.update("INFY", ts3, ltp=1520.0, smma20=1515.0, smma120=1500.0)
    assert e3 is None

def test_no_crossover_and_exact_equality():
    detector = CrossoverDetector()
    ts1 = datetime(2026, 8, 20, 10, 0)
    ts2 = datetime(2026, 8, 20, 10, 1)

    # Exact equality initial state: smma20 == smma120
    e1 = detector.update("TCS", ts1, ltp=3000.0, smma20=3000.0, smma120=3000.0)
    assert e1 is None

    # Equal again -> No crossover
    e2 = detector.update("TCS", ts2, ltp=3000.0, smma20=3000.0, smma120=3000.0)
    assert e2 is None

def test_alternating_crossovers():
    detector = CrossoverDetector()
    symbol = "HDFCBANK"
    
    # Tick 1: Below
    detector.update(symbol, datetime(2026, 8, 20, 10, 0), ltp=150.0, smma20=140.0, smma120=150.0)
    
    # Tick 2: BUY Crossover
    e1 = detector.update(symbol, datetime(2026, 8, 20, 10, 1), ltp=155.0, smma20=155.0, smma120=150.0)
    assert e1 is not None and e1.signal == "BUY"

    # Tick 3: SELL Crossover
    e2 = detector.update(symbol, datetime(2026, 8, 20, 10, 2), ltp=145.0, smma20=145.0, smma120=150.0)
    assert e2 is not None and e2.signal == "SELL"

    # Tick 4: BUY Crossover again
    e3 = detector.update(symbol, datetime(2026, 8, 20, 10, 3), ltp=160.0, smma20=152.0, smma120=150.0)
    assert e3 is not None and e3.signal == "BUY"

