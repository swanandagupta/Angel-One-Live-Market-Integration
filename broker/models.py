from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class MarketTick:
    timestamp: datetime
    symbol: str
    ltp: float
    ltq: float
    bid_price: float
    bid_quantity: float
    ask_price: float
    ask_quantity: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "ltp": self.ltp,
            "ltq": self.ltq,
            "bid_price": self.bid_price,
            "bid_quantity": self.bid_quantity,
            "ask_price": self.ask_price,
            "ask_quantity": self.ask_quantity,
        }

@dataclass
class Candle:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

@dataclass
class CrossoverEvent:
    timestamp: datetime
    symbol: str
    signal: str  # "BUY" or "SELL"
    ltp: float
    smma20: float
    smma120: float
    smma_gap: float
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "signal": self.signal,
            "ltp": self.ltp,
            "smma20": self.smma20,
            "smma120": self.smma120,
            "smma_gap": self.smma_gap,
            "features": self.features,
        }

@dataclass
class Trade:
    symbol: str
    signal: str  # "BUY" or "SELL"
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    profitable: Optional[int] = None  # 1 if pnl > 0 else 0
    ml_probability: float = 0.0
    decision: str = "PENDING"  # "ACCEPT", "AVOID", "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "profitable": self.profitable,
            "ml_probability": self.ml_probability,
            "decision": self.decision,
        }
