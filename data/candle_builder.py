from datetime import datetime
from typing import Dict, List, Optional
from broker.models import MarketTick, Candle

class CandleBuilder:
    """Aggregates incoming ticks into 1-minute OHLC candles per symbol."""

    def __init__(self, max_candles_per_symbol: int = 500):
        self.max_candles_per_symbol = max_candles_per_symbol
        # Completed candles per symbol: List[Candle]
        self._completed_candles: Dict[str, List[Candle]] = {}
        # Current forming candle state: Dict[str, Dict]
        self._current_builder: Dict[str, Dict] = {}

    def process_tick(self, tick: MarketTick) -> Optional[Candle]:
        """
        Process a market tick. Returns a completed Candle if a minute boundary passed, else None.
        """
        symbol = tick.symbol
        # Floor timestamp to start of minute (e.g. 09:15:34 -> 09:15:00)
        minute_ts = tick.timestamp.replace(second=0, microsecond=0)

        if symbol not in self._completed_candles:
            self._completed_candles[symbol] = []

        completed_candle = None

        if symbol not in self._current_builder:
            self._current_builder[symbol] = {
                "minute_ts": minute_ts,
                "open": tick.ltp,
                "high": tick.ltp,
                "low": tick.ltp,
                "close": tick.ltp,
                "volume": tick.ltq
            }
        else:
            builder = self._current_builder[symbol]
            if minute_ts > builder["minute_ts"]:
                # Save previous completed candle
                completed_candle = Candle(
                    timestamp=builder["minute_ts"],
                    symbol=symbol,
                    open=builder["open"],
                    high=builder["high"],
                    low=builder["low"],
                    close=builder["close"],
                    volume=builder["volume"]
                )
                self._completed_candles[symbol].append(completed_candle)

                # Trim completed candles list
                if len(self._completed_candles[symbol]) > self.max_candles_per_symbol:
                    self._completed_candles[symbol].pop(0)

                # Reset for new minute
                self._current_builder[symbol] = {
                    "minute_ts": minute_ts,
                    "open": tick.ltp,
                    "high": tick.ltp,
                    "low": tick.ltp,
                    "close": tick.ltp,
                    "volume": tick.ltq
                }
            else:
                # Update current minute candle
                builder["high"] = max(builder["high"], tick.ltp)
                builder["low"] = min(builder["low"], tick.ltp)
                builder["close"] = tick.ltp
                builder["volume"] += tick.ltq

        return completed_candle

    def add_historical_candles(self, symbol: str, candles: List[Candle]) -> None:
        """Pre-load historical candles for indicator warm-up."""
        if symbol not in self._completed_candles:
            self._completed_candles[symbol] = []
        self._completed_candles[symbol].extend(candles)
        # Keep sorted by timestamp
        self._completed_candles[symbol].sort(key=lambda c: c.timestamp)
        if len(self._completed_candles[symbol]) > self.max_candles_per_symbol:
            self._completed_candles[symbol] = self._completed_candles[symbol][-self.max_candles_per_symbol:]

    def get_candles(self, symbol: str) -> List[Candle]:
        """Returns completed candles for a symbol."""
        return self._completed_candles.get(symbol, [])

    def get_close_prices(self, symbol: str) -> List[float]:
        """Returns list of close prices from completed candles."""
        return [c.close for c in self.get_candles(symbol)]
