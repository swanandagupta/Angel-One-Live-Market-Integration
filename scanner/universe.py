from typing import List
from broker.base import BaseBroker
from utils.logger import logger
from utils.helpers import normalize_symbol

class UniverseScanner:
    """Retrieves and normalizes the full available NSE equity stock universe."""

    def __init__(self, broker: BaseBroker):
        self.broker = broker

    def fetch_nse_universe(self) -> List[str]:
        """Fetch and normalize complete list of active NSE equity symbols."""
        try:
            raw_symbols = self.broker.get_instruments()
            normalized = [normalize_symbol(s) for s in raw_symbols if s]
            # De-duplicate while preserving order
            unique_symbols = list(dict.fromkeys(normalized))
            logger.info(f"Retrieved {len(unique_symbols)} NSE stock symbols from broker universe.")
            return unique_symbols
        except Exception as e:
            logger.error(f"Error retrieving NSE universe: {e}")
            return []
