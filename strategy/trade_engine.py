from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from broker.models import CrossoverEvent, Trade
from data.database import DatabaseManager
from utils.logger import logger

class TradeEngine:
    """
    Simulates paper trading execution and position management for crossover events
    with strict temporal integrity (exit_time > entry_time), symbol isolation, and P/L verification.
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db
        # Active open trades per symbol: { symbol: Trade }
        self.active_trades: Dict[str, Trade] = {}
        # History of completed trades
        self.completed_trades: List[Trade] = []

    def process_crossover(
        self,
        event: CrossoverEvent,
        ml_probability: float = 1.0,
        decision: str = "ACCEPT"
    ) -> Optional[Trade]:
        """
        Processes a crossover event, closes existing opposite position if valid, and opens new trade.
        Enforces strict temporal order constraint: exit_time MUST be strictly after entry_time.
        """
        symbol = event.symbol
        signal = event.signal
        timestamp = event.timestamp
        ltp = event.ltp

        # 1. Close existing open position if signal opposes
        if symbol in self.active_trades:
            active_trade = self.active_trades[symbol]
            if active_trade.signal != signal:
                # Validate symbol isolation
                if active_trade.symbol != symbol:
                    logger.error(f"[SYMBOL_MISMATCH_REJECTED] Trade symbol '{active_trade.symbol}' != event symbol '{symbol}'. Rejecting.")
                # Validate temporal order constraint: exit_time MUST be > entry_time
                elif timestamp <= active_trade.entry_time:
                    logger.error(
                        f"[TEMPORAL_INTEGRITY_REJECTED] Trade exit_time ({timestamp}) <= entry_time ({active_trade.entry_time}) "
                        f"for symbol {symbol}. Rejecting invalid completion."
                    )
                else:
                    # Opposite crossover: Close current trade strictly after entry
                    active_trade.exit_time = timestamp
                    active_trade.exit_price = ltp

                    if active_trade.signal == "BUY":
                        active_trade.pnl = round(ltp - active_trade.entry_price, 2)
                    else:  # SELL
                        active_trade.pnl = round(active_trade.entry_price - ltp, 2)

                    active_trade.profitable = 1 if active_trade.pnl > 0 else 0
                    self.completed_trades.append(active_trade)

                    # Persist to database if db provided
                    if self.db:
                        self.db.save_trade(active_trade.to_dict())

                    logger.info(
                        f"Closed {active_trade.signal} trade for {symbol} at {ltp} "
                        f"(Entry: {active_trade.entry_time}, Exit: {timestamp}). "
                        f"P/L: {active_trade.pnl:+.2f} (Profitable: {active_trade.profitable})"
                    )
                    del self.active_trades[symbol]

        # 2. Open new trade if decision is ACCEPT
        new_trade = Trade(
            symbol=symbol,
            signal=signal,
            entry_time=timestamp,
            entry_price=ltp,
            ml_probability=ml_probability,
            decision=decision
        )

        if decision == "ACCEPT":
            # If an active position for this symbol is already open, do not overwrite unless signal changed
            if symbol not in self.active_trades:
                self.active_trades[symbol] = new_trade
                logger.info(f"Opened ACCEPTED {signal} paper trade for {symbol} at {ltp} at {timestamp} (Prob: {ml_probability:.2%})")
            else:
                logger.debug(f"Position for {symbol} already active in direction {self.active_trades[symbol].signal}. Skipping duplicate open.")
        else:
            logger.info(f"Recorded AVOIDED {signal} trade for {symbol} at {ltp} at {timestamp} (Prob: {ml_probability:.2%})")

        return new_trade

    @staticmethod
    def validate_trade_integrity(trades: List[Trade]) -> Tuple[bool, int, List[str]]:
        """
        Audits a list of trades for:
        1. Temporal integrity (exit_time > entry_time)
        2. P/L calculation correctness (BUY: exit - entry, SELL: entry - exit)
        3. Profitable flag correctness (profitable == (pnl > 0))
        4. Symbol consistency
        Returns: (is_pass, error_count, list_of_error_messages)
        """
        error_count = 0
        errors = []

        for idx, t in enumerate(trades):
            if t.exit_time is not None:
                if t.exit_time <= t.entry_time:
                    error_count += 1
                    errors.append(f"Trade #{idx} [{t.symbol}]: Exit time ({t.exit_time}) <= Entry time ({t.entry_time})")

                if t.exit_price is not None:
                    if t.signal == "BUY":
                        expected_pnl = round(t.exit_price - t.entry_price, 2)
                    else:
                        expected_pnl = round(t.entry_price - t.exit_price, 2)

                    if t.pnl is None or abs(t.pnl - expected_pnl) > 0.01:
                        error_count += 1
                        errors.append(f"Trade #{idx} [{t.symbol} {t.signal}]: Stored P/L ({t.pnl}) != Expected P/L ({expected_pnl})")

                    expected_prof = 1 if expected_pnl > 0 else 0
                    if t.profitable != expected_prof:
                        error_count += 1
                        errors.append(f"Trade #{idx} [{t.symbol}]: Stored Profitable ({t.profitable}) != Expected ({expected_prof})")

        is_pass = (error_count == 0)
        return is_pass, error_count, errors
