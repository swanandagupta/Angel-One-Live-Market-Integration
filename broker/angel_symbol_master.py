import json
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
from config import Config
from utils.logger import logger
from utils.helpers import normalize_symbol

@dataclass
class AngelSymbolInfo:
    symbol: str            # Normalized symbol, e.g. "SBIN"
    exchange: str          # "NSE"
    token: str             # Angel One scrip token string, e.g. "3045"
    trading_symbol: str    # Angel One trading symbol, e.g. "SBIN-EQ"

class AngelSymbolMaster:
    """Manages downloading, caching, and mapping for Angel One NSE scrip master instruments."""

    SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

    def __init__(self, cache_file: Optional[Path] = None):
        self.cache_file = cache_file or (Config.DATA_STORAGE_DIR / "angel_scrip_master.json")
        # Mappings: symbol -> AngelSymbolInfo, token -> AngelSymbolInfo
        self._symbol_map: Dict[str, AngelSymbolInfo] = {}
        self._token_map: Dict[str, AngelSymbolInfo] = {}
        self._loaded = False

    def load(self, force_refresh: bool = False) -> bool:
        """Loads scrip master from local disk cache or downloads fresh copy from Angel One API."""
        if self._loaded and not force_refresh:
            return True

        data = None

        # 1. Try loading from cache file
        if not force_refresh and self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"Loaded Angel One scrip master from local cache at {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to read local scrip master cache: {e}")

        # 2. Download from official endpoint if cache missing or empty
        if not data:
            try:
                logger.info(f"Downloading Angel One scrip master from {self.SCRIP_MASTER_URL}...")
                resp = requests.get(self.SCRIP_MASTER_URL, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    # Cache to disk
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    logger.info(f"Successfully downloaded and cached {len(data)} scrip master entries.")
                else:
                    logger.error(f"Failed to download scrip master, HTTP status: {resp.status_code}")
            except Exception as e:
                logger.error(f"Exception downloading Angel One scrip master: {e}")

        # 3. Process entries into memory maps for NSE equities
        if data:
            self._symbol_map.clear()
            self._token_map.clear()

            for item in data:
                exch = item.get("exch_seg", "").upper()
                tradingsymbol = item.get("symbol", "").upper()
                token = str(item.get("token", ""))

                # Filter for NSE equities
                if exch == "NSE" and (tradingsymbol.endswith("-EQ") or item.get("name")):
                    raw_name = item.get("name", "")
                    clean_sym = normalize_symbol(raw_name if raw_name else tradingsymbol)

                    info = AngelSymbolInfo(
                        symbol=clean_sym,
                        exchange="NSE",
                        token=token,
                        trading_symbol=tradingsymbol
                    )

                    if clean_sym and clean_sym not in self._symbol_map:
                        self._symbol_map[clean_sym] = info
                    if token and token not in self._token_map:
                        self._token_map[token] = info

            self._loaded = True
            logger.info(f"Processed {len(self._symbol_map)} NSE symbols into Angel symbol master lookup.")
            return True

        # Fallback to hardcoded core NSE symbols if download fails
        self._populate_fallback_symbols()
        self._loaded = True
        return False

    def _populate_fallback_symbols(self) -> None:
        """Populates common active NSE stock token mapping as safety fallback."""
        fallbacks = [
            ("SBIN", "3045", "SBIN-EQ"),
            ("RELIANCE", "2885", "RELIANCE-EQ"),
            ("INFY", "1594", "INFY-EQ"),
            ("TCS", "11536", "TCS-EQ"),
            ("TATAMOTORS", "3456", "TATAMOTORS-EQ"),
            ("HDFCBANK", "1333", "HDFCBANK-EQ"),
            ("ICICIBANK", "4963", "ICICIBANK-EQ"),
            ("PNB", "10666", "PNB-EQ"),
            ("FEDERALBNK", "1023", "FEDERALBNK-EQ"),
            ("IDFCFIRSTB", "11184", "IDFCFIRSTB-EQ"),
            ("CANBK", "10794", "CANBK-EQ"),
            ("BANKBARODA", "4668", "BANKBARODA-EQ"),
            ("SAIL", "2963", "SAIL-EQ"),
            ("NMDC", "15332", "NMDC-EQ"),
            ("BHEL", "438", "BHEL-EQ")
        ]
        for sym, token, tsym in fallbacks:
            info = AngelSymbolInfo(symbol=sym, exchange="NSE", token=token, trading_symbol=tsym)
            self._symbol_map[sym] = info
            self._token_map[token] = info
        self._loaded = True

    def get_info_by_symbol(self, symbol: str) -> Optional[AngelSymbolInfo]:
        """Resolve normalized symbol (e.g. 'SBIN') to AngelSymbolInfo."""
        clean = normalize_symbol(symbol)
        if not self._loaded and not self._symbol_map:
            self.load()
        return self._symbol_map.get(clean)

    def get_info_by_token(self, token: str) -> Optional[AngelSymbolInfo]:
        """Resolve Angel token string (e.g. '3045') to AngelSymbolInfo."""
        if not self._loaded and not self._token_map:
            self.load()
        return self._token_map.get(str(token))

    def get_all_nse_symbols(self) -> List[str]:
        """Return list of all available normalized NSE stock symbols."""
        if not self._loaded and not self._symbol_map:
            self.load()
        return list(self._symbol_map.keys())
