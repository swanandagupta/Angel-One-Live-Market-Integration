from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from broker.models import MarketTick
from utils.helpers import safe_divide

class TickStore:
    """In-memory high-frequency rolling tick buffer per symbol for real-time LTQ, ETQ, and depth calculations."""

    def __init__(self, max_history_minutes: int = 65):
        self.max_history_minutes = max_history_minutes
        # Store deque of MarketTick per symbol
        self._buffers: Dict[str, deque] = {}
        # Latest tick snapshot per symbol
        self._latest: Dict[str, MarketTick] = {}

    def add_tick(self, tick: MarketTick) -> None:
        symbol = tick.symbol
        if symbol not in self._buffers:
            self._buffers[symbol] = deque()
        
        self._buffers[symbol].append(tick)
        self._latest[symbol] = tick
        self._purge_old_ticks(symbol, tick.timestamp)

    def _purge_old_ticks(self, symbol: str, current_time: datetime) -> None:
        cutoff = current_time - timedelta(minutes=self.max_history_minutes)
        buffer = self._buffers[symbol]
        while buffer and buffer[0].timestamp < cutoff:
            buffer.popleft()

    def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        return self._latest.get(symbol)

    def get_all_symbols(self) -> List[str]:
        return list(self._latest.keys())

    def get_ticks_in_window(self, symbol: str, minutes: float) -> List[MarketTick]:
        if symbol not in self._buffers or not self._buffers[symbol]:
            return []
        latest_time = self._buffers[symbol][-1].timestamp
        cutoff = latest_time - timedelta(minutes=minutes)
        return [t for t in self._buffers[symbol] if t.timestamp >= cutoff]

    def calculate_etq(self, symbol: str, minutes: float) -> float:
        """ETQ_window = sum(LTQ within window)"""
        ticks = self.get_ticks_in_window(symbol, minutes)
        return sum(t.ltq for t in ticks)

    def calculate_avg_ltq(self, symbol: str, minutes: float) -> float:
        """Calculate average LTQ per tick in window."""
        ticks = self.get_ticks_in_window(symbol, minutes)
        if not ticks:
            return 0.0
        return sum(t.ltq for t in ticks) / len(ticks)

    def calculate_avg_ltp(self, symbol: str, minutes: float) -> float:
        """Calculate rolling average LTP in window."""
        ticks = self.get_ticks_in_window(symbol, minutes)
        if not ticks:
            latest = self.get_latest_tick(symbol)
            return latest.ltp if latest else 0.0
        return sum(t.ltp for t in ticks) / len(ticks)

    def get_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculates full LTQ/ETQ and rolling price metrics for a symbol."""
        latest = self.get_latest_tick(symbol)
        if not latest:
            return {}

        ltp = latest.ltp
        ltq = latest.ltq
        bid_q = latest.bid_quantity
        ask_q = latest.ask_quantity
        bid_p = latest.bid_price
        ask_p = latest.ask_price

        # ETQ calculations
        etq_5m = self.calculate_etq(symbol, 5)
        etq_20m = self.calculate_etq(symbol, 20)
        etq_60m = self.calculate_etq(symbol, 60)

        # Average LTQ calculations
        avg_ltq_1m = self.calculate_avg_ltq(symbol, 1)
        avg_ltq_2m = self.calculate_avg_ltq(symbol, 2)
        avg_ltq_5m = self.calculate_avg_ltq(symbol, 5)
        avg_ltq_20m = self.calculate_avg_ltq(symbol, 20)

        # LTQ ratios
        ltq_2m_to_5m = safe_divide(avg_ltq_2m, avg_ltq_5m)
        ltq_5m_to_20m = safe_divide(avg_ltq_5m, avg_ltq_20m)

        # ETQ ratios
        etq_5m_to_20m = safe_divide(etq_5m, safe_divide(etq_20m, 4.0))
        etq_20m_to_60m = safe_divide(etq_20m, safe_divide(etq_60m, 3.0))

        # Average LTP calculations
        avg_ltp_20m = self.calculate_avg_ltp(symbol, 20)
        avg_ltp_60m = self.calculate_avg_ltp(symbol, 60)

        distance_from_avg_20m = safe_divide(ltp - avg_ltp_20m, avg_ltp_20m)
        distance_from_avg_60m = safe_divide(ltp - avg_ltp_60m, avg_ltp_60m)

        # Market Depth metrics
        bid_ask_imbalance = safe_divide(bid_q - ask_q, bid_q + ask_q)
        spread = max(0.0, ask_p - bid_p)
        relative_spread = safe_divide(spread, ltp)

        return {
            "ltp": ltp,
            "ltq": ltq,
            "etq_5m": etq_5m,
            "etq_20m": etq_20m,
            "etq_60m": etq_60m,
            "avg_ltq_1m": avg_ltq_1m,
            "avg_ltq_2m": avg_ltq_2m,
            "avg_ltq_5m": avg_ltq_5m,
            "avg_ltq_20m": avg_ltq_20m,
            "ltq_2m_to_5m": ltq_2m_to_5m,
            "ltq_5m_to_20m": ltq_5m_to_20m,
            "etq_5m_to_20m": etq_5m_to_20m,
            "etq_20m_to_60m": etq_20m_to_60m,
            "avg_ltp_20m": avg_ltp_20m,
            "avg_ltp_60m": avg_ltp_60m,
            "distance_from_avg_20m": distance_from_avg_20m,
            "distance_from_avg_60m": distance_from_avg_60m,
            "bid_quantity": bid_q,
            "ask_quantity": ask_q,
            "bid_price": bid_p,
            "ask_price": ask_p,
            "bid_ask_imbalance": bid_ask_imbalance,
            "spread": spread,
            "relative_spread": relative_spread
        }
