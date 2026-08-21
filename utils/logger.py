import logging
import sys
from pathlib import Path
from config import Config

def setup_logger(name: str = "stock_ai_trader") -> logging.Logger:
    """Configures structured logger writing to console and logs/application.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    log_file = Config.LOGS_DIR / "application.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

logger = setup_logger()
