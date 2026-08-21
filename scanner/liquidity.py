from typing import Dict, Any, List
from config import Config
from broker.models import MarketTick
from utils.logger import logger

class LiquidityFilter:
    """Filters stocks based on price (₹30 <= LTP <= ₹500) and order book depth (>1,000,000 bid & ask quantity)."""

    def __init__(self,
                 min_ltp: float = Config.MIN_LTP,
                 max_ltp: float = Config.MAX_LTP,
                 min_bid_qty: float = Config.MIN_BID_QTY,
                 min_ask_qty: float = Config.MIN_ASK_QTY):
        self.min_ltp = min_ltp
        self.max_ltp = max_ltp
        self.min_bid_qty = min_bid_qty
        self.min_ask_qty = min_ask_qty

    def is_price_qualified(self, ltp: float) -> bool:
        """Check if 30 <= LTP <= 500."""
        if ltp is None:
            return False
        return self.min_ltp <= float(ltp) <= self.max_ltp

    def is_liquidity_qualified(self, bid_quantity: float, ask_quantity: float) -> bool:
        """Check if Bid Quantity > 1,000,000 and Ask Quantity > 1,000,000."""
        if bid_quantity is None or ask_quantity is None:
            return False
        return float(bid_quantity) > self.min_bid_qty and float(ask_quantity) > self.min_ask_qty

    def evaluate_tick(self, tick: MarketTick) -> Dict[str, bool]:
        """Evaluates price and liquidity qualification for a single MarketTick."""
        price_ok = self.is_price_qualified(tick.ltp)
        liquidity_ok = self.is_liquidity_qualified(tick.bid_quantity, tick.ask_quantity)
        return {
            "price_qualified": price_ok,
            "liquidity_qualified": liquidity_ok,
            "fully_qualified": price_ok and liquidity_ok
        }

    def filter_ticks(self, ticks: List[MarketTick]) -> List[MarketTick]:
        """Returns only ticks that satisfy both price and depth requirements."""
        qualifying = []
        for t in ticks:
            res = self.evaluate_tick(t)
            if res["fully_qualified"]:
                qualifying.append(t)
        return qualifying
