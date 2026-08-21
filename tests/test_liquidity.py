import pytest
from scanner.liquidity import LiquidityFilter
from broker.models import MarketTick
from datetime import datetime

def test_price_filtering():
    lf = LiquidityFilter(min_ltp=30.0, max_ltp=500.0)

    assert lf.is_price_qualified(29.9) == False
    assert lf.is_price_qualified(30.0) == True
    assert lf.is_price_qualified(250.0) == True
    assert lf.is_price_qualified(500.0) == True
    assert lf.is_price_qualified(500.1) == False
    assert lf.is_price_qualified(None) == False

def test_liquidity_depth_filtering():
    lf = LiquidityFilter(min_bid_qty=1_000_000, min_ask_qty=1_000_000)

    # 1,000,000 is NOT > 1,000,000
    assert lf.is_liquidity_qualified(1_000_000, 1_000_000) == False
    assert lf.is_liquidity_qualified(1_000_001, 1_000_000) == False
    assert lf.is_liquidity_qualified(1_000_000, 1_000_001) == False
    assert lf.is_liquidity_qualified(1_000_001, 1_000_001) == True
    assert lf.is_liquidity_qualified(None, 2_000_000) == False

def test_evaluate_tick():
    lf = LiquidityFilter()
    tick_qual = MarketTick(
        timestamp=datetime.now(), symbol="SBIN", ltp=150.0, ltq=100,
        bid_price=149.9, bid_quantity=1_200_000, ask_price=150.1, ask_quantity=1_500_000
    )
    res = lf.evaluate_tick(tick_qual)
    assert res["price_qualified"] == True
    assert res["liquidity_qualified"] == True
    assert res["fully_qualified"] == True

    tick_unqual = MarketTick(
        timestamp=datetime.now(), symbol="MRF", ltp=120000.0, ltq=10,
        bid_price=119900.0, bid_quantity=500, ask_price=120100.0, ask_quantity=500
    )
    res_un = lf.evaluate_tick(tick_unqual)
    assert res_un["price_qualified"] == False
    assert res_un["liquidity_qualified"] == False
    assert res_un["fully_qualified"] == False
