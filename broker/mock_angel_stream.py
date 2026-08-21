import time
import random
import numpy as np
from datetime import datetime
from typing import Callable, List
from broker.models import MarketTick
from broker.angel_client import AngelClient
from utils.logger import logger

class MockAngelStreamer:
    """
    Simulates Angel One SmartWebSocketV2 binary SNAP_QUOTE payload data stream.
    Feeds raw parsed dictionaries into AngelClient.parse_websocket_message to verify
    the complete live pipeline without needing live credentials.
    """

    def __init__(self, client: AngelClient, symbols: List[str]):
        self.client = client
        self.symbols = symbols
        self.running = False

    def generate_mock_payload(self, symbol: str) -> dict:
        """Generates raw dictionary matching SmartWebSocketV2._parse_binary_data output."""
        info = self.client.symbol_master.get_info_by_symbol(symbol)
        token = info.token if info else "3045"

        base_price = 150.0 + random.uniform(-10, 10)
        # SmartAPI returns prices in paise (multiplied by 100)
        paise_price = int(base_price * 100)

        # Realistic bid/ask depth > 10 Lakhs (1,000,000) for liquidity qualified stocks
        bid_q = random.uniform(1_050_000, 3_000_000)
        ask_q = random.uniform(1_050_000, 3_000_000)

        return {
            "subscription_mode": 3,  # SNAP_QUOTE
            "exchange_type": 1,       # NSE_CM
            "token": token,
            "sequence_number": random.randint(1000, 99999),
            "exchange_timestamp": int(time.time() * 1000),
            "last_traded_price": paise_price,
            "last_traded_quantity": random.randint(50, 2000),
            "total_buy_quantity": bid_q,
            "total_sell_quantity": ask_q,
            "best_5_buy_data": [
                {"flag": 0, "quantity": int(bid_q / 5), "price": paise_price - 5, "no of orders": 10}
            ],
            "best_5_sell_data": [
                {"flag": 1, "quantity": int(ask_q / 5), "price": paise_price + 5, "no of orders": 12}
            ]
        }

    def emit_tick(self, symbol: str) -> MarketTick:
        """Simulates receiving a single websocket binary packet and parsing it."""
        payload = self.generate_mock_payload(symbol)
        tick = self.client.parse_websocket_message(payload)
        return tick
