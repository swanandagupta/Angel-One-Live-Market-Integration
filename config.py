import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Config:
    # Broker Configuration
    BROKER = os.getenv("BROKER", "DEMO").upper()
    
    # FYERS Credentials
    FYERS_API_KEY = os.getenv("FYERS_API_KEY", "")
    FYERS_ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN", "")
    FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "")
    FYERS_SECRET_KEY = os.getenv("FYERS_SECRET_KEY", "")
    
    # Angel One Credentials
    ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
    ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
    ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD", "")
    ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", os.getenv("ANGEL_TOTP_KEY", ""))
    
    # ML & Trading Thresholds
    ML_THRESHOLD = float(os.getenv("ML_THRESHOLD", "0.70"))
    
    # Screening Constraints
    MIN_LTP = 30.0
    MAX_LTP = 500.0
    MIN_BID_QTY = 1_000_000
    MIN_ASK_QTY = 1_000_000
    
    # Technical Indicators
    SMMA_FAST = 20
    SMMA_SLOW = 120
    
    # Storage Paths
    LOGS_DIR = BASE_DIR / "logs"
    DATA_STORAGE_DIR = BASE_DIR / "data_storage"
    MODELS_DIR = BASE_DIR / "models"
    DB_PATH = DATA_STORAGE_DIR / "trader.db"
    MODEL_PATH = MODELS_DIR / "xgboost_crossover_model.joblib"
    MODEL_V2_PATH = MODELS_DIR / "xgboost_crossover_model_v2.joblib"
    SAMPLE_DATA_PATH = DATA_STORAGE_DIR / "sample_market_data.csv"
    
    # System & Dashboard
    REFRESH_INTERVAL_SEC = int(os.getenv("REFRESH_INTERVAL_SEC", "3"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure required directories exist
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
Config.DATA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
