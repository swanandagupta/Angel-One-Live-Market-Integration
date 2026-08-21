import pytest
import pandas as pd
from backtest.engine import BacktestEngine
from data.historical_loader import HistoricalDataLoader
from ml.evaluate import StrategyEvaluator

def test_strategy_evaluator():
    trades = [
        {"pnl": 10.0},
        {"pnl": -5.0},
        {"pnl": 15.0},
        {"pnl": -3.0},
        {"pnl": 20.0}
    ]
    df = pd.DataFrame(trades)
    metrics = StrategyEvaluator.calculate_trade_metrics(df)

    assert metrics["total_trades"] == 5
    assert metrics["winning_trades"] == 3
    assert metrics["losing_trades"] == 2
    assert metrics["win_rate"] == pytest.approx(3.0 / 5.0)
    assert metrics["total_pnl"] == pytest.approx(37.0)
    assert metrics["avg_profit"] == pytest.approx(45.0 / 3.0)
    assert metrics["avg_loss"] == pytest.approx(8.0 / 2.0)
    assert metrics["profit_factor"] == pytest.approx(45.0 / 8.0)

def test_backtest_engine_run():
    loader = HistoricalDataLoader()
    _, candles_by_symbol = loader.load_ticks_and_candles()
    
    engine = BacktestEngine()
    results = engine.run_backtest(candles_by_symbol=candles_by_symbol, threshold=0.70)

    assert "crossovers_df" in results
    assert "metrics_smma_only" in results
    assert "metrics_smma_ml" in results
    assert "comparison" in results

    crossovers_df = results["crossovers_df"]
    if not crossovers_df.empty:
        assert "ml_probability" in crossovers_df.columns
        assert "ml_decision" in crossovers_df.columns
