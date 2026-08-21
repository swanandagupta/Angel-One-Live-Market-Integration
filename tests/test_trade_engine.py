import pytest
from datetime import datetime
from broker.models import CrossoverEvent, Trade
from strategy.trade_engine import TradeEngine

def test_trade_engine_buy_and_sell_pnl():
    engine = TradeEngine()

    ts1 = datetime(2026, 8, 20, 10, 0)
    ts2 = datetime(2026, 8, 20, 10, 15)

    # 1. Open BUY crossover trade at LTP = 100.0
    event_buy = CrossoverEvent(
        timestamp=ts1, symbol="SBIN", signal="BUY", ltp=100.0,
        smma20=102.0, smma120=100.0, smma_gap=2.0
    )
    trade_buy = engine.process_crossover(event_buy, ml_probability=0.85, decision="ACCEPT")
    assert trade_buy.decision == "ACCEPT"
    assert "SBIN" in engine.active_trades

    # 2. Opposite SELL crossover at LTP = 110.0 -> Should close BUY trade with P/L = +10.0
    event_sell = CrossoverEvent(
        timestamp=ts2, symbol="SBIN", signal="SELL", ltp=110.0,
        smma20=108.0, smma120=110.0, smma_gap=-2.0
    )
    trade_sell = engine.process_crossover(event_sell, ml_probability=0.80, decision="ACCEPT")

    # Completed trade check: BUY Entry 100, Exit 110 => P/L = +10.0, profitable = 1
    assert len(engine.completed_trades) == 1
    closed_trade = engine.completed_trades[0]
    assert closed_trade.signal == "BUY"
    assert closed_trade.entry_price == 100.0
    assert closed_trade.exit_price == 110.0
    assert closed_trade.pnl == 10.0
    assert closed_trade.profitable == 1

def test_trade_engine_buy_loss_and_sell_pnl():
    engine = TradeEngine()
    ts1 = datetime(2026, 8, 20, 10, 0)
    ts2 = datetime(2026, 8, 20, 10, 10)
    ts3 = datetime(2026, 8, 20, 10, 20)

    # BUY Entry 100, Exit 95 => P/L = -5.0, profitable = 0
    e_buy = CrossoverEvent(timestamp=ts1, symbol="INFY", signal="BUY", ltp=100.0, smma20=102.0, smma120=100.0, smma_gap=2.0)
    engine.process_crossover(e_buy, ml_probability=0.9, decision="ACCEPT")

    # SELL Crossover at 95 closes BUY with P/L = -5.0
    e_sell = CrossoverEvent(timestamp=ts2, symbol="INFY", signal="SELL", ltp=95.0, smma20=94.0, smma120=95.0, smma_gap=-1.0)
    engine.process_crossover(e_sell, ml_probability=0.9, decision="ACCEPT")

    t1 = engine.completed_trades[0]
    assert t1.pnl == -5.0
    assert t1.profitable == 0

    # SELL Entry 95, Exit 85 => P/L = 95 - 85 = +10.0, profitable = 1
    e_buy2 = CrossoverEvent(timestamp=ts3, symbol="INFY", signal="BUY", ltp=85.0, smma20=86.0, smma120=85.0, smma_gap=1.0)
    engine.process_crossover(e_buy2, ml_probability=0.9, decision="ACCEPT")

    t2 = engine.completed_trades[1]
    assert t2.signal == "SELL"
    assert t2.entry_price == 95.0
    assert t2.exit_price == 85.0
    assert t2.pnl == 10.0
    assert t2.profitable == 1

def test_trade_engine_sell_loss():
    engine = TradeEngine()
    ts1 = datetime(2026, 8, 20, 10, 0)
    ts2 = datetime(2026, 8, 20, 10, 10)

    # SELL Entry 100, Exit 110 => P/L = 100 - 110 = -10.0, profitable = 0
    e_sell = CrossoverEvent(timestamp=ts1, symbol="TCS", signal="SELL", ltp=100.0, smma20=98.0, smma120=100.0, smma_gap=-2.0)
    engine.process_crossover(e_sell, ml_probability=0.85, decision="ACCEPT")

    e_buy = CrossoverEvent(timestamp=ts2, symbol="TCS", signal="BUY", ltp=110.0, smma20=112.0, smma120=110.0, smma_gap=2.0)
    engine.process_crossover(e_buy, ml_probability=0.85, decision="ACCEPT")

    closed = engine.completed_trades[0]
    assert closed.signal == "SELL"
    assert closed.entry_price == 100.0
    assert closed.exit_price == 110.0
    assert closed.pnl == -10.0
    assert closed.profitable == 0

def test_trade_engine_avoided_trade_handling():
    engine = TradeEngine()
    ts = datetime(2026, 8, 20, 10, 0)

    event = CrossoverEvent(
        timestamp=ts, symbol="TCS", signal="BUY", ltp=3200.0,
        smma20=3210.0, smma120=3200.0, smma_gap=10.0
    )
    # ML probability is 0.40 < threshold 0.55 -> decision is AVOID
    trade = engine.process_crossover(event, ml_probability=0.40, decision="AVOID")
    
    assert trade.decision == "AVOID"
    # Should not open position in active portfolio
    assert "TCS" not in engine.active_trades

def test_temporal_integrity_rejection():
    """Verifies that trade exits with timestamp <= entry_time are rejected."""
    engine = TradeEngine()
    ts1 = datetime(2026, 8, 20, 10, 30)
    ts2 = datetime(2026, 8, 20, 10, 15)  # Earlier timestamp!

    # 1. Open BUY trade at 10:30
    e_buy = CrossoverEvent(timestamp=ts1, symbol="RELIANCE", signal="BUY", ltp=2500.0, smma20=2510.0, smma120=2500.0, smma_gap=10.0)
    engine.process_crossover(e_buy, ml_probability=0.80, decision="ACCEPT")
    assert "RELIANCE" in engine.active_trades

    # 2. Emitting SELL crossover with earlier timestamp 10:15 MUST BE REJECTED
    e_sell = CrossoverEvent(timestamp=ts2, symbol="RELIANCE", signal="SELL", ltp=2550.0, smma20=2490.0, smma120=2500.0, smma_gap=-10.0)
    engine.process_crossover(e_sell, ml_probability=0.80, decision="ACCEPT")

    # Completed trades list should remain empty because completion was rejected
    assert len(engine.completed_trades) == 0
    # Position should still be active
    assert "RELIANCE" in engine.active_trades

def test_validate_trade_integrity_pass():
    """Verifies validate_trade_integrity passes on valid trades and detects invalid trades."""
    engine = TradeEngine()
    ts1 = datetime(2026, 8, 20, 10, 0)
    ts2 = datetime(2026, 8, 20, 10, 30)

    e_buy = CrossoverEvent(timestamp=ts1, symbol="SBIN", signal="BUY", ltp=100.0, smma20=102.0, smma120=100.0, smma_gap=2.0)
    engine.process_crossover(e_buy, ml_probability=0.80, decision="ACCEPT")

    e_sell = CrossoverEvent(timestamp=ts2, symbol="SBIN", signal="SELL", ltp=115.0, smma20=98.0, smma120=100.0, smma_gap=-2.0)
    engine.process_crossover(e_sell, ml_probability=0.80, decision="ACCEPT")

    is_pass, err_cnt, errs = TradeEngine.validate_trade_integrity(engine.completed_trades)
    assert is_pass == True
    assert err_cnt == 0
    assert len(errs) == 0
