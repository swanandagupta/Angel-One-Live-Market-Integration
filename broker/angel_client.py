import time
import json
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Optional, Any

from config import Config
from utils.logger import logger
from utils.helpers import safe_divide, normalize_symbol
from broker.base import BaseBroker
from broker.models import MarketTick, Candle
from broker.angel_symbol_master import AngelSymbolMaster, AngelSymbolInfo

class AngelClient(BaseBroker):
    """
    Angel One SmartAPI Client integrating REST authentication, Scrip Master resolution,
    historical OHLC candle warm-up, and SmartWebSocketV2 live tick streaming.
    """

    def __init__(self, mock_mode: bool = False):
        self.api_key = Config.ANGEL_API_KEY
        self.client_id = Config.ANGEL_CLIENT_ID
        self.password = Config.ANGEL_PASSWORD
        self.totp_secret = Config.ANGEL_TOTP_SECRET
        self.mock_mode = mock_mode

        self._connected = False
        self._auth_failed = False
        self.smart_api = None
        self.feed_token = None
        self.jwt_token = None

        self.symbol_master = AngelSymbolMaster()
        self.ws_client = None
        self.subscribed_tokens: List[str] = []
        self.tick_callback: Optional[Callable[[MarketTick], None]] = None

    def connect(self) -> bool:
        """Authenticate with Angel One SmartAPI and obtain session tokens."""
        if self.mock_mode:
            self._connected = True
            self._auth_failed = False
            logger.info("AngelClient initialized in MOCK TEST MODE.")
            return True

        if not self.api_key or not self.client_id or not self.password:
            logger.warning("Angel One credentials incomplete in .env. AngelClient in unauthenticated mode.")
            self._connected = False
            self._auth_failed = True
            return False

        try:
            from SmartApi import SmartConnect
            self.smart_api = SmartConnect(api_key=self.api_key)

            # Generate TOTP code
            totp_code = ""
            if self.totp_secret:
                try:
                    import pyotp
                    totp_code = pyotp.TOTP(self.totp_secret.replace(" ", "")).now()
                except Exception as e:
                    logger.warning(f"TOTP generation failed: {e}. Using raw secret string.")
                    totp_code = self.totp_secret

            # Generate session
            data = self.smart_api.generateSession(self.client_id, self.password, totp_code)

            if data and data.get("status"):
                self.jwt_token = data.get("data", {}).get("jwtToken")
                self.feed_token = data.get("data", {}).get("feedToken")
                self._connected = True
                self._auth_failed = False
                logger.info("Successfully authenticated with Angel One SmartAPI session.")

                # Load symbol master
                self.symbol_master.load()
                return True
            else:
                msg = data.get("message", "Unknown auth failure") if data else "Empty response"
                logger.error(f"Angel One authentication failed: {msg}")
                self._connected = False
                self._auth_failed = True
                return False
        except Exception as e:
            logger.error(f"Exception during Angel One authentication: {e}")
            self._connected = False
            self._auth_failed = True
            return False

    def disconnect(self) -> None:
        if self.ws_client:
            try:
                self.ws_client.close_connection()
            except Exception:
                pass
            self.ws_client = None

        if self.smart_api and self._connected:
            try:
                self.smart_api.terminateSession(self.client_id)
            except Exception:
                pass
        self._connected = False
        logger.info("Disconnected Angel One client.")

    def is_connected(self) -> bool:
        return self._connected

    def is_auth_failed(self) -> bool:
        return self._auth_failed

    def get_instruments(self) -> List[str]:
        """Fetch complete NSE equity symbol list from Angel Symbol Master."""
        if not self.symbol_master._loaded:
            self.symbol_master.load()
        symbols = self.symbol_master.get_all_nse_symbols()
        if not symbols:
            # Safety fallback list if scrip master offline
            return [
                "SBIN", "RELIANCE", "INFY", "TCS", "TATAMOTORS", "HDFCBANK", "ICICIBANK",
                "PNB", "FEDERALBNK", "IDFCFIRSTB", "CANBK", "BANKBARODA", "SAIL", "NMDC", "BHEL"
            ]
        return symbols

    def get_quotes(self, symbols: List[str]) -> Dict[str, MarketTick]:
        """Fetch snapshot quotes via REST API."""
        quotes = {}
        if not self._connected or not self.smart_api:
            return quotes

        for sym in symbols[:50]:
            info = self.symbol_master.get_info_by_symbol(sym)
            if not info:
                continue

            try:
                res = self.smart_api.ltpData(info.exchange, info.trading_symbol, info.token)
                if res and res.get("status") and "data" in res:
                    d = res["data"]
                    ltp = float(d.get("ltp", 0.0))
                    quotes[info.symbol] = MarketTick(
                        timestamp=datetime.now(),
                        symbol=info.symbol,
                        ltp=ltp,
                        ltq=100.0,
                        bid_price=ltp - 0.05,
                        bid_quantity=1_100_000.0,
                        ask_price=ltp + 0.05,
                        ask_quantity=1_050_000.0
                    )
            except Exception as e:
                logger.error(f"Error fetching quote for {sym}: {e}")
        return quotes

    def get_historical_candles(self, symbol: str, timeframe: str = "1m", num_candles: int = 200) -> List[Candle]:
        """Fetches historical OHLC candles for indicator warm-up."""
        candles = []
        info = self.symbol_master.get_info_by_symbol(symbol)
        if not info or not self._connected or not self.smart_api:
            return candles

        try:
            fromdate = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d 09:15")
            todate = datetime.now().strftime("%Y-%m-%d 15:30")

            params = {
                "exchange": info.exchange,
                "symboltoken": info.token,
                "interval": "ONE_MINUTE",
                "fromdate": fromdate,
                "todate": todate
            }

            res = self.smart_api.getCandleData(params)
            if res and res.get("status") and "data" in res:
                for row in res["data"]:
                    # Row format: [timestamp_str, open, high, low, close, volume]
                    try:
                        ts = datetime.fromisoformat(row[0].replace("T", " "))
                    except ValueError:
                        ts = datetime.now()

                    candles.append(Candle(
                        timestamp=ts,
                        symbol=info.symbol,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5])
                    ))
        except Exception as e:
            logger.error(f"Error fetching Angel candle data for {symbol}: {e}")
        return candles

    def parse_websocket_message(self, parsed_data: Dict[str, Any]) -> Optional[MarketTick]:
        """
        Parses normalized dictionary from Angel One SmartWebSocketV2 payload into MarketTick schema.
        
        Angel One SmartWebSocketV2 Schema Mapping:
        - last_traded_price: Prices are in paise (divided by 100.0)
        - last_traded_quantity: LTQ
        - total_buy_quantity: Total order book bid depth
        - total_sell_quantity: Total order book ask depth
        - best_5_buy_data[0]: Best bid price and quantity
        - best_5_sell_data[0]: Best ask price and quantity
        """
        try:
            token = str(parsed_data.get("token", ""))
            info = self.symbol_master.get_info_by_token(token)
            symbol = info.symbol if info else token

            # LTP division by 100.0 (paise to rupees)
            raw_ltp = parsed_data.get("last_traded_price", 0.0)
            ltp = float(raw_ltp) / 100.0 if raw_ltp > 1000 else float(raw_ltp)

            ltq = float(parsed_data.get("last_traded_quantity", 100.0))
            tot_buy_q = float(parsed_data.get("total_buy_quantity", 1_100_000.0))
            tot_sell_q = float(parsed_data.get("total_sell_quantity", 1_050_000.0))

            best_buy = parsed_data.get("best_5_buy_data", [])
            best_sell = parsed_data.get("best_5_sell_data", [])

            bid_p = float(best_buy[0]["price"]) / 100.0 if best_buy and best_buy[0].get("price") else ltp - 0.05
            bid_q = float(best_buy[0]["quantity"]) if best_buy and best_buy[0].get("quantity") else tot_buy_q

            ask_p = float(best_sell[0]["price"]) / 100.0 if best_sell and best_sell[0].get("price") else ltp + 0.05
            ask_q = float(best_sell[0]["quantity"]) if best_sell and best_sell[0].get("quantity") else tot_sell_q

            return MarketTick(
                timestamp=datetime.now(),
                symbol=symbol,
                ltp=ltp,
                ltq=ltq,
                bid_price=bid_p,
                bid_quantity=tot_buy_q if tot_buy_q > 0 else bid_q,
                ask_price=ask_p,
                ask_quantity=tot_sell_q if tot_sell_q > 0 else ask_q
            )
        except Exception as e:
            logger.error(f"Error parsing Angel websocket tick: {e}")
            return None

    def subscribe_ticks(self, symbols: List[str], callback: Callable[[MarketTick], None]) -> None:
        """Establishes SmartWebSocketV2 market data subscription for requested NSE tokens."""
        self.tick_callback = callback
        tokens = []

        for sym in symbols:
            info = self.symbol_master.get_info_by_symbol(sym)
            if info:
                tokens.append(info.token)

        self.subscribed_tokens = tokens
        logger.info(f"Subscribing to Angel One SmartWebSocketV2 for {len(tokens)} tokens.")

        if self.mock_mode or not self._connected or not self.jwt_token:
            logger.info("AngelClient websocket operating in mock/simulated tick mode.")
            return

        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2

            correlation_id = "stock_ai_trader_stream"
            self.ws_client = SmartWebSocketV2(
                auth_token=self.jwt_token,
                api_key=self.api_key,
                client_code=self.client_id,
                feed_token=self.feed_token
            )

            def on_data(wsapp, message):
                tick = self.parse_websocket_message(message)
                if tick and self.tick_callback:
                    self.tick_callback(tick)

            def on_open(wsapp):
                logger.info("Angel One SmartWebSocketV2 connection opened.")
                # Mode 3 = SNAP_QUOTE, Exchange 1 = NSE_CM
                token_list = [{"exchangeType": 1, "tokens": self.subscribed_tokens}]
                self.ws_client.subscribe(correlation_id, 3, token_list)

            def on_close(wsapp):
                logger.warning("Angel One SmartWebSocketV2 disconnected. Attempting reconnect...")
                self._connected = False

            def on_error(wsapp, error):
                logger.error(f"Angel One SmartWebSocketV2 error: {error}")

            self.ws_client.on_data = on_data
            self.ws_client.on_open = on_open
            self.ws_client.on_close = on_close
            self.ws_client.on_error = on_error

            # Connect websocket in background thread
            t = threading.Thread(target=self.ws_client.connect, daemon=True)
            t.start()

        except Exception as e:
            logger.error(f"Exception initializing Angel One websocket: {e}")
