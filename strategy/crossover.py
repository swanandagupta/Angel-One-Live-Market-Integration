from datetime import datetime
from typing import Dict, Optional, Tuple, List
from broker.models import CrossoverEvent
from utils.logger import logger

class CrossoverDetector:
    """
    Stateful detector tracking SMMA20 and SMMA120 per symbol to emit single-fire crossover signals.
    """

    def __init__(self):
        # Per symbol state: { "prev_smma20": float, "prev_smma120": float, "last_signal": str }
        self._states: Dict[str, Dict[str, Optional[float]]] = {}

    def update(
        self,
        symbol: str,
        timestamp: datetime,
        ltp: float,
        smma20: float,
        smma120: float,
        features: Optional[Dict[str, float]] = None
    ) -> Optional[CrossoverEvent]:
        """
        Evaluates new SMMA20 and SMMA120 values for a symbol.
        Emits CrossoverEvent EXACTLY ONCE on trend transition.
        """
        if smma20 is None or smma120 is None:
            return None

        if symbol not in self._states:
            self._states[symbol] = {
                "prev_smma20": smma20,
                "prev_smma120": smma120,
                "last_signal": "ABOVE" if smma20 > smma120 else "BELOW"
            }
            return None

        state = self._states[symbol]
        prev_smma20 = state["prev_smma20"]
        prev_smma120 = state["prev_smma120"]

        event = None

        # Check BUY Crossover: prev_smma20 <= prev_smma120 AND curr_smma20 > curr_smma120
        if prev_smma20 <= prev_smma120 and smma20 > smma120:
            if state["last_signal"] != "BUY":
                event = CrossoverEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal="BUY",
                    ltp=ltp,
                    smma20=smma20,
                    smma120=smma120,
                    smma_gap=smma20 - smma120,
                    features=features or {}
                )
                state["last_signal"] = "BUY"
                logger.info(f"BUY Crossover detected for {symbol} at LTP {ltp} (SMMA20: {smma20:.2f}, SMMA120: {smma120:.2f})")

        # Check SELL Crossover: prev_smma20 >= prev_smma120 AND curr_smma20 < curr_smma120
        elif prev_smma20 >= prev_smma120 and smma20 < smma120:
            if state["last_signal"] != "SELL":
                event = CrossoverEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal="SELL",
                    ltp=ltp,
                    smma20=smma20,
                    smma120=smma120,
                    smma_gap=smma20 - smma120,
                    features=features or {}
                )
                state["last_signal"] = "SELL"
                logger.info(f"SELL Crossover detected for {symbol} at LTP {ltp} (SMMA20: {smma20:.2f}, SMMA120: {smma120:.2f})")

        # Update previous SMMA state
        state["prev_smma20"] = smma20
        state["prev_smma120"] = smma120

        return event

    def get_prev_smma(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Returns (prev_smma20, prev_smma120) for feature engineering slope calculations."""
        state = self._states.get(symbol)
        if not state:
            return None, None
        return state.get("prev_smma20"), state.get("prev_smma120")
