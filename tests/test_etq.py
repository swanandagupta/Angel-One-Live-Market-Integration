import pytest
from datetime import datetime, timedelta
from broker.models import MarketTick
from data.tick_store import TickStore

def test_etq_and_ltq_calculations():
    store = TickStore()
    now = datetime.now()

    # Add 5 ticks over the last 4 minutes
    for i in range(5):
        t = MarketTick(
            timestamp=now - timedelta(minutes=i),
            symbol="SBIN",
            ltp=100.0 + i,
            ltq=500.0,
            bid_price=99.0,
            bid_quantity=1_200_000,
            ask_price=101.0,
            ask_quantity=1_100_000
        )
        store.add_tick(t)

    # Sum of LTQ over 5m should be 5 * 500 = 2500
    etq_5m = store.calculate_etq("SBIN", 5)
    assert etq_5m == pytest.approx(2500.0)

    # Average LTQ over 5m
    avg_ltq_5m = store.calculate_avg_ltq("SBIN", 5)
    assert avg_ltq_5m == pytest.approx(500.0)

    # Metrics dictionary check
    metrics = store.get_metrics("SBIN")
    assert metrics["etq_5m"] == pytest.approx(2500.0)
    assert metrics["bid_ask_imbalance"] > 0
