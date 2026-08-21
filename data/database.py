import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import Config
from utils.logger import logger

class DatabaseManager:
    """Manages persistent SQLite storage for crossover events, trades, ML predictions, and metadata."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create necessary schema tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Table: crossover_events
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS crossover_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        ltp REAL NOT NULL,
                        smma20 REAL NOT NULL,
                        smma120 REAL NOT NULL,
                        smma_gap REAL NOT NULL,
                        features_json TEXT
                    )
                """)
                
                # Table: trades
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        entry_time TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_time TEXT,
                        exit_price REAL,
                        pnl REAL,
                        profitable INTEGER,
                        ml_probability REAL,
                        decision TEXT NOT NULL
                    )
                """)

                # Table: predictions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        probability REAL NOT NULL,
                        decision TEXT NOT NULL,
                        explanation_json TEXT
                    )
                """)

                # Table: application_metadata
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS application_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TEXT
                    )
                """)
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")

    def save_crossover_event(self, event_data: Dict[str, Any]) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO crossover_events 
                    (timestamp, symbol, signal, ltp, smma20, smma120, smma_gap, features_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(event_data.get("timestamp")),
                    event_data.get("symbol"),
                    event_data.get("signal"),
                    float(event_data.get("ltp", 0.0)),
                    float(event_data.get("smma20", 0.0)),
                    float(event_data.get("smma120", 0.0)),
                    float(event_data.get("smma_gap", 0.0)),
                    json.dumps(event_data.get("features", {}))
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving crossover event: {e}")

    def save_trade(self, trade_data: Dict[str, Any]) -> int:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trades 
                    (symbol, signal, entry_time, entry_price, exit_time, exit_price, pnl, profitable, ml_probability, decision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get("symbol"),
                    trade_data.get("signal"),
                    str(trade_data.get("entry_time")),
                    float(trade_data.get("entry_price", 0.0)),
                    str(trade_data.get("exit_time")) if trade_data.get("exit_time") else None,
                    float(trade_data.get("exit_price")) if trade_data.get("exit_price") is not None else None,
                    float(trade_data.get("pnl")) if trade_data.get("pnl") is not None else None,
                    trade_data.get("profitable"),
                    float(trade_data.get("ml_probability", 0.0)),
                    trade_data.get("decision", "PENDING")
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return -1

    def update_trade(self, trade_id: int, exit_time: datetime, exit_price: float, pnl: float, profitable: int) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE trades 
                    SET exit_time = ?, exit_price = ?, pnl = ?, profitable = ?
                    WHERE id = ?
                """, (
                    exit_time.isoformat(),
                    exit_price,
                    pnl,
                    profitable,
                    trade_id
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating trade {trade_id}: {e}")

    def save_prediction(self, prediction_data: Dict[str, Any]) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO predictions
                    (timestamp, symbol, signal, probability, decision, explanation_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(prediction_data.get("timestamp")),
                    prediction_data.get("symbol"),
                    prediction_data.get("signal"),
                    float(prediction_data.get("probability", 0.0)),
                    prediction_data.get("decision"),
                    json.dumps(prediction_data.get("explanation", {}))
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

    def get_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching trades: {e}")
            return []

    def get_crossover_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM crossover_events ORDER BY id DESC LIMIT ?", (limit,))
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    r = dict(row)
                    if r.get("features_json"):
                        r["features"] = json.loads(r["features_json"])
                    results.append(r)
                return results
        except Exception as e:
            logger.error(f"Error fetching crossover events: {e}")
            return []
