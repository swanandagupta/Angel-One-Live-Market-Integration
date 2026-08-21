from typing import List, Optional, Union
import numpy as np
import pandas as pd

def calculate_smma_series(prices: Union[List[float], np.ndarray, pd.Series], period: int) -> List[Optional[float]]:
    """
    Calculates Smoothed Moving Average (SMMA / Modified Moving Average) for a price series.
    
    Formula:
    For t = period: SMMA_t = sum(Price_1 to Price_n) / n  (Simple Average)
    For t > period: SMMA_t = (SMMA_(t-1) * (n - 1) + Price_t) / n
    
    Returns a list of same length as prices, padded with None for indices < period - 1.
    """
    if len(prices) < period:
        return [None] * len(prices)

    prices_arr = np.asarray(prices, dtype=float)
    result = [None] * len(prices_arr)

    # Initial SMA value at index (period - 1)
    first_window = prices_arr[:period]
    valid_window = first_window[~np.isnan(first_window)] if np.any(np.isnan(first_window)) else first_window
    first_smma = np.mean(valid_window) if len(valid_window) > 0 else 0.0
    result[period - 1] = float(first_smma)

    prev_smma = first_smma
    for i in range(period, len(prices_arr)):
        p = prices_arr[i]
        if np.isnan(p):
            curr_smma = prev_smma
        else:
            curr_smma = (prev_smma * (period - 1) + p) / period
        result[i] = float(curr_smma)
        prev_smma = curr_smma

    return result

def update_smma(prev_smma: float, current_price: float, period: int) -> float:
    """
    Calculates single incremental SMMA step given previous SMMA value and new price.
    Formula: (SMMA_(t-1) * (n - 1) + Price_t) / n
    """
    if prev_smma is None:
        return current_price
    if current_price is None or np.isnan(current_price):
        return prev_smma
    return (prev_smma * (period - 1) + current_price) / period

class SMMACalculator:
    """Stateful SMMA calculator for a specific symbol maintaining SMMA20 and SMMA120."""

    def __init__(self, period_fast: int = 20, period_slow: int = 120):
        self.period_fast = period_fast
        self.period_slow = period_slow
        self.smma_fast: Optional[float] = None
        self.smma_slow: Optional[float] = None

    def initialize_from_prices(self, prices: List[float]) -> None:
        """Warms up indicator state using historical price series."""
        fast_series = calculate_smma_series(prices, self.period_fast)
        slow_series = calculate_smma_series(prices, self.period_slow)

        # Get latest non-None values
        valid_fast = [v for v in fast_series if v is not None]
        valid_slow = [v for v in slow_series if v is not None]

        self.smma_fast = valid_fast[-1] if valid_fast else None
        self.smma_slow = valid_slow[-1] if valid_slow else None

    def update(self, price: float) -> tuple[Optional[float], Optional[float]]:
        """Updates SMMA values with new candle close price."""
        if self.smma_fast is None:
            self.smma_fast = price
        else:
            self.smma_fast = update_smma(self.smma_fast, price, self.period_fast)

        if self.smma_slow is None:
            self.smma_slow = price
        else:
            self.smma_slow = update_smma(self.smma_slow, price, self.period_slow)

        return self.smma_fast, self.smma_slow
