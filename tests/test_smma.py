import pytest
import numpy as np
from indicators.smma import calculate_smma_series, update_smma, SMMACalculator

def test_calculate_smma_series_expected_values():
    prices = [10.0, 12.0, 14.0, 16.0, 18.0]
    period = 3
    smma_result = calculate_smma_series(prices, period)

    assert len(smma_result) == 5
    assert smma_result[0] is None
    assert smma_result[1] is None

    # First value at index 2 (period 3 SMA: (10 + 12 + 14) / 3 = 12.0)
    assert smma_result[2] == pytest.approx(12.0)

    # Next value at index 3: (12 * 2 + 16) / 3 = 40 / 3 = 13.3333333
    assert smma_result[3] == pytest.approx(40.0 / 3.0)

    # Next value at index 4: ((40/3) * 2 + 18) / 3 = 44.6666667 / 3 = 14.8888889
    expected_4 = ((40.0 / 3.0) * 2.0 + 18.0) / 3.0
    assert smma_result[4] == pytest.approx(expected_4)

def test_smma_insufficient_data():
    prices = [10.0, 20.0]
    period = 5
    smma_result = calculate_smma_series(prices, period)
    assert smma_result == [None, None]

def test_smma_nan_handling():
    prices = [10.0, 12.0, 14.0, np.nan, 18.0]
    period = 3
    smma_result = calculate_smma_series(prices, period)
    assert smma_result[2] == pytest.approx(12.0)
    # At index 3 (NaN price), SMMA remains previous value 12.0
    assert smma_result[3] == pytest.approx(12.0)
    # At index 4 (18.0): (12.0 * 2 + 18.0) / 3 = 42 / 3 = 14.0
    assert smma_result[4] == pytest.approx(14.0)

def test_update_smma_incremental():
    prev_smma = 100.0
    new_price = 110.0
    period = 20
    # (100 * 19 + 110) / 20 = (1900 + 110) / 20 = 2010 / 20 = 100.5
    updated = update_smma(prev_smma, new_price, period)
    assert updated == pytest.approx(100.5)

def test_update_smma_none_and_nan():
    assert update_smma(None, 50.0, 20) == 50.0
    assert update_smma(100.0, np.nan, 20) == 100.0
    assert update_smma(100.0, None, 20) == 100.0

def test_smma_calculator_class():
    calc = SMMACalculator(period_fast=2, period_slow=3)
    prices = [10.0, 20.0, 30.0, 40.0]
    calc.initialize_from_prices(prices)

    assert calc.smma_fast is not None
    assert calc.smma_slow is not None
    
    # Update single step
    fast, slow = calc.update(50.0)
    assert fast is not None and slow is not None

