from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Any, Optional
from broker.models import MarketTick, Candle

class BaseBroker(ABC):
    """Abstract base class establishing common broker interface for FYERS, Angel One, and Demo brokers."""

    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and establish connection with broker API/WebSocket."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully disconnect from broker WebSocket and API sessions."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check active connection status."""
        pass

    @abstractmethod
    def get_instruments(self) -> List[str]:
        """Fetch complete universe of NSE equity stock symbols."""
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Fetch current market depth and tick snapshot for requested symbols."""
        pass

    @abstractmethod
    def get_historical_candles(self, symbol: str, timeframe: str = "1m", num_candles: int = 200) -> List[Candle]:
        """Fetch historical OHLC candles for indicator warm-up."""
        pass

    @abstractmethod
    def subscribe_ticks(self, symbols: List[str], callback: Callable[[MarketTick], None]) -> None:
        """Subscribe to live market depth / tick stream with a callback handler."""
        pass
