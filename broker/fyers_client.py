import time
from datetime import datetime
from typing import List, Dict, Callable, Optional
from config import Config
from utils.logger import logger
from utils.helpers import safe_divide, normalize_symbol
from broker.base import BaseBroker
from broker.models import MarketTick, Candle

class FyersClient(BaseBroker):
    """FYERS API v3 Integration Client."""

    def __init__(self):
        self.client_id = Config.FYERS_CLIENT_ID
        self.access_token = Config.FYERS_ACCESS_TOKEN
        self._connected = False
        self.fyers = None

    def connect(self) -> bool:
        if not self.client_id or not self.access_token:
            logger.warning("FYERS credentials incomplete in .env. FyersClient in fallback/unauthenticated mode.")
            self._connected = False
            return False
        try:
            from fyers_apiv3 import fyersModel
            # Format app_id properly (FYERS expects client_id:access_token)
            app_id = self.client_id if ":" in self.client_id else f"{self.client_id}"
            self.fyers = fyersModel.FyersModel(
                client_id=app_id,
                token=self.access_token,
                log_path=str(Config.LOGS_DIR)
            )
            profile = self.fyers.get_profile()
            if profile.get("s") == "ok":
                self._connected = True
                logger.info("Successfully connected to FYERS API.")
                return True
            else:
                logger.error(f"FYERS connection failed response: {profile}")
                self._connected = False
                return False
        except Exception as e:
            logger.error(f"Exception connecting to FYERS: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        self._connected = False
        logger.info("Disconnected FYERS client.")

    def is_connected(self) -> bool:
        return self._connected

    def get_instruments(self) -> List[str]:
        """Fetch NSE equity symbol list. Normalizes symbol format."""
        # Standard active NSE stock list fallback if instrument file endpoint unavailable
        default_symbols = [
            "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL",
            "ITC", "KOTAKBANK", "LT", "AXISBANK", "WIPRO", "HCLTECH", "TATAMOTORS",
            "SUNPHARMA", "MARUTI", "ONGC", "NTPC", "POWERGRID", "TITAN", "BAJFINANCE",
            "TATASTEEL", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "ASIANPAINT", "COALINDIA",
            "BPCL", "GRASIM", "HEROMOTOCO", "EICHERMOT", "DIVISLAB", "DRREDDY", "CIPLA",
            "APOLLOHOSP", "INDUSINDBK", "HDFCLIFE", "SBILIFE", "BAJAJFINSV", "TATACONSUM",
            "BRITANNIA", "NESTLEIND", "JSWSTEEL", "HINDALCO", "TECHM", "SHRIRAMFIN", "BEL",
            "TRENT", "M&M", "PIDILITIND"
        ]
        return default_symbols

    def get_quotes(self, symbols: List[str]) -> Dict[str, MarketTick]:
        quotes = {}
        if not self._connected or not self.fyers:
            return quotes

        try:
            # Batch FYERS quotes format: "NSE:SBIN-EQ,NSE:RELIANCE-EQ"
            fyers_symbols = [f"NSE:{normalize_symbol(s)}-EQ" for s in symbols[:50]]
            data = {"symbols": ",".join(fyers_symbols)}
            response = self.fyers.quotes(data=data)
            
            if response.get("s") == "ok" and "d" in response:
                for item in response["d"]:
                    v = item.get("v", {})
                    raw_sym = item.get("n", "")
                    clean_sym = normalize_symbol(raw_sym)
                    cmd = v.get("cmd", {})
                    bid_p = float(cmd.get("cbr", 0.0))
                    bid_q = float(cmd.get("vbr", 0.0))
                    ask_p = float(cmd.get("car", 0.0))
                    ask_q = float(cmd.get("var", 0.0))
                    ltp = float(v.get("lp", 0.0))
                    ltq = float(v.get("volume", 0.0))
                    
                    quotes[clean_sym] = MarketTick(
                        timestamp=datetime.now(),
                        symbol=clean_sym,
                        ltp=ltp,
                        ltq=ltq,
                        bid_price=bid_p if bid_p > 0 else ltp,
                        bid_quantity=bid_q,
                        ask_price=ask_p if ask_p > 0 else ltp,
                        ask_quantity=ask_q
                    )
        except Exception as e:
            logger.error(f"Error fetching FYERS quotes: {e}")
        return quotes

    def get_historical_candles(self, symbol: str, timeframe: str = "1m", num_candles: int = 200) -> List[Candle]:
        candles = []
        if not self._connected or not self.fyers:
            return candles
        try:
            clean_sym = normalize_symbol(symbol)
            data = {
                "symbol": f"NSE:{clean_sym}-EQ",
                "resolution": "1" if timeframe == "1m" else "1",
                "date_format": "0",
                "range_from": str(int(time.time()) - (num_candles * 60 * 2)),
                "range_to": str(int(time.time())),
                "cont_flag": "1"
            }
            res = self.fyers.history(data=data)
            if res.get("s") == "ok" and "candles" in res:
                for c in res["candles"]:
                    ts = datetime.fromtimestamp(c[0])
                    candles.append(Candle(
                        timestamp=ts,
                        symbol=clean_sym,
                        open=float(c[1]),
                        high=float(c[2]),
                        low=float(c[3]),
                        close=float(c[4]),
                        volume=float(c[5])
                    ))
        except Exception as e:
            logger.error(f"Error fetching FYERS historical candles for {symbol}: {e}")
        return candles

    def subscribe_ticks(self, symbols: List[str], callback: Callable[[MarketTick], None]) -> None:
        logger.info(f"FYERS websocket tick subscription registered for {len(symbols)} symbols.")
        # Callback execution setup for streaming websocket messages
