import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config import Config
from utils.logger import logger
from broker.models import MarketTick, Candle
from broker.angel_symbol_master import AngelSymbolMaster

class HistoricalDataLoader:
    """
    Loads and manages historical 1-minute OHLC candle datasets for NSE equity stocks.
    Supports caching raw candle data to data/historical/SYMBOL/YYYY-MM-DD.csv.
    """

    CORE_50_NSE_SYMBOLS = [
        "SBIN", "RELIANCE", "INFY", "TCS", "TATAMOTORS", "HDFCBANK", "ICICIBANK",
        "PNB", "FEDERALBNK", "IDFCFIRSTB", "CANBK", "BANKBARODA", "SAIL", "NMDC", "BHEL",
        "ITC", "AXISBANK", "LT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
        "ASIANPAINT", "BAJFINANCE", "BHARTIARTL", "HCLTECH", "HEROMOTOCO", "HINDUNILVR",
        "INDUSINDBK", "JSWSTEEL", "KOTAKBANK", "M&M", "NESTLEIND", "NTPC", "ONGC",
        "POWERGRID", "TATASTEEL", "TECHM", "ADANIENT", "ADANIPORTS", "APOLLOHOSP",
        "BPCL", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM", "HDFCLIFE"
    ]

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (Config.DATA_STORAGE_DIR / "historical")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sample_file_path = Config.SAMPLE_DATA_PATH

    @staticmethod
    def get_market_start_datetime(days_back: int = 5) -> datetime:
        """Returns a valid NSE market open timestamp (09:15 AM IST) on recent trading day."""
        now = datetime.now()
        dt = now - timedelta(days=days_back)
        while dt.weekday() >= 5:  # Skip weekends (Sat/Sun)
            dt -= timedelta(days=1)
        return datetime(dt.year, dt.month, dt.day, 9, 15, 0)

    def fetch_angel_historical_candles(
        self,
        angel_client,
        symbols: Optional[List[str]] = None,
        days: int = 30
    ) -> Dict[str, List[Candle]]:
        """
        Fetches genuine historical 1-minute OHLC candles via Angel One REST API getCandleData().
        Caches downloaded candles to local CSV files under data/historical/SYMBOL/.
        """
        target_symbols = symbols or self.CORE_50_NSE_SYMBOLS
        all_candles: Dict[str, List[Candle]] = {}

        if not angel_client or not angel_client.is_connected():
            logger.warning("AngelClient not connected. Skipping REST API candle download.")
            return all_candles

        logger.info(f"Initiating Angel One historical candle collection for {len(target_symbols)} symbols over {days} days.")

        for sym in target_symbols:
            sym_dir = self.data_dir / sym
            sym_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Fetching Angel historical candles for {sym}...")
            # Request historical candles in batches or directly
            candles = angel_client.get_historical_candles(sym, timeframe="1m", num_candles=days * 375)
            
            if candles:
                all_candles[sym] = candles
                # Save to disk cache
                rows = []
                for c in candles:
                    rows.append({
                        "timestamp": c.timestamp.isoformat(),
                        "symbol": c.symbol,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume
                    })
                df_sym = pd.DataFrame(rows)
                cache_file = sym_dir / "candle_history.csv"
                df_sym.to_csv(cache_file, index=False)
                logger.info(f"Cached {len(candles)} candles for {sym} to {cache_file}")

            # Throttling to prevent API rate limit issues
            time.sleep(0.2)

        return all_candles

    def generate_sample_dataset(self, num_symbols: int = 50, num_candles: int = 1500) -> pd.DataFrame:
        """
        Generates multi-symbol 1-minute OHLC dataset across 50 NSE stocks.
        Constructs timestamps strictly within valid NSE trading hours (09:15 to 15:30 IST).
        Uses persistent momentum random walk to produce realistic crossover trade opportunities.
        """
        symbols = self.CORE_50_NSE_SYMBOLS[:num_symbols]
        base_start = self.get_market_start_datetime(days_back=10)
        
        # Precompute distinct trading days (skipping Sat/Sun)
        num_days_needed = (num_candles // 375) + 10
        market_days: List[datetime] = []
        curr_dt = base_start
        while len(market_days) < num_days_needed:
            if curr_dt.weekday() < 5:
                market_days.append(curr_dt)
            curr_dt += timedelta(days=1)

        rows = []
        np.random.seed(42)

        for sym_idx, sym in enumerate(symbols):
            base_price = np.random.uniform(40.0, 450.0)
            price = base_price
            momentum = np.random.normal(0, 0.5)

            for i in range(num_candles):
                minute_offset = i
                day_idx = minute_offset // 375
                min_in_day = minute_offset % 375

                ts_day = market_days[day_idx]
                ts = datetime(ts_day.year, ts_day.month, ts_day.day, 9, 15, 0) + timedelta(minutes=min_in_day)

                # Persistent momentum random walk
                momentum = 0.92 * momentum + np.random.normal(0, 0.40)
                price = max(30.0, min(500.0, price + momentum))

                high = price + abs(np.random.normal(0.25, 0.12))
                low = price - abs(np.random.normal(0.25, 0.12))
                open_p = price - np.random.normal(0, 0.12)
                close_p = price

                bid_q = np.random.uniform(1_050_000, 3_500_000)
                ask_q = np.random.uniform(1_050_000, 3_500_000)
                bid_p = close_p - 0.05
                ask_p = close_p + 0.05
                ltq = np.random.uniform(50, 5000)
                vol = ltq * 15.0

                rows.append({
                    "timestamp": ts.isoformat(),
                    "symbol": sym,
                    "open": round(open_p, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close_p, 2),
                    "ltp": round(close_p, 2),
                    "ltq": round(ltq, 2),
                    "volume": round(vol, 2),
                    "bid_price": round(bid_p, 2),
                    "bid_quantity": round(bid_q, 0),
                    "ask_price": round(ask_p, 2),
                    "ask_quantity": round(ask_q, 0)
                })

        df = pd.DataFrame(rows)
        self.sample_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.sample_file_path, index=False)
        logger.info(f"Generated historical dataset with {len(df)} candles across {len(symbols)} symbols at {self.sample_file_path}")
        return df

    def load_ticks_and_candles(self) -> Tuple[List[MarketTick], Dict[str, List[Candle]]]:
        """Loads dataset as MarketTicks and pre-aggregated Candle lists."""
        if not self.sample_file_path.exists():
            self.generate_sample_dataset()

        df = pd.read_csv(self.sample_file_path)
        ticks: List[MarketTick] = []
        candles_by_symbol: Dict[str, List[Candle]] = {}

        for _, row in df.iterrows():
            ts = datetime.fromisoformat(str(row["timestamp"]))
            sym = str(row["symbol"]).upper()

            tick = MarketTick(
                timestamp=ts,
                symbol=sym,
                ltp=float(row["ltp"]),
                ltq=float(row["ltq"]),
                bid_price=float(row["bid_price"]),
                bid_quantity=float(row["bid_quantity"]),
                ask_price=float(row["ask_price"]),
                ask_quantity=float(row["ask_quantity"])
            )
            ticks.append(tick)

            candle = Candle(
                timestamp=ts,
                symbol=sym,
                open=float(row.get("open", row["ltp"])),
                high=float(row.get("high", row["ltp"])),
                low=float(row.get("low", row["ltp"])),
                close=float(row.get("close", row["ltp"])),
                volume=float(row.get("volume", row["ltq"]))
            )

            if sym not in candles_by_symbol:
                candles_by_symbol[sym] = []
            candles_by_symbol[sym].append(candle)

        logger.info(f"Loaded {len(ticks)} ticks across {len(candles_by_symbol)} symbols from {self.sample_file_path}")
        return ticks, candles_by_symbol
