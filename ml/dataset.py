import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from config import Config
from utils.logger import logger
from broker.models import Candle
from indicators.smma import calculate_smma_series
from features.feature_engineering import FeatureExtractor
from strategy.crossover import CrossoverDetector

class DatasetBuilder:
    """Converts historical candle data into crossover feature samples and P/L target labels."""

    V2_DATASET_PATH = Config.DATA_STORAGE_DIR / "ml_dataset_v2.csv"

    @staticmethod
    def build_dataset_v2(
        candles_by_symbol: Dict[str, List[Candle]],
        period_fast: int = Config.SMMA_FAST,
        period_slow: int = Config.SMMA_SLOW,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Builds genuine historical ML training dataset (v2) using strictly 17 candle-verifiable features.
        Calculates stateful SMMA crossovers and directional trade P/L until opposing crossover.
        Exports output to data/ml_dataset_v2.csv with 'target' as the final column.
        """
        output_file = output_path or DatasetBuilder.V2_DATASET_PATH
        rows = []

        for symbol, candles in candles_by_symbol.items():
            if len(candles) < period_slow + 5:
                continue

            close_prices = [c.close for c in candles]
            smma20_list = calculate_smma_series(close_prices, period_fast)
            smma120_list = calculate_smma_series(close_prices, period_slow)

            detector = CrossoverDetector()
            active_crossover = None

            for i in range(period_slow, len(candles)):
                candle = candles[i]
                c_close = candle.close
                s20 = smma20_list[i]
                s120 = smma120_list[i]

                if s20 is None or s120 is None:
                    continue

                prev_s20, prev_s120 = detector.get_prev_smma(symbol)

                # Evaluate crossover
                event = detector.update(
                    symbol=symbol,
                    timestamp=candle.timestamp,
                    ltp=c_close,
                    smma20=s20,
                    smma120=s120
                )

                if event:
                    # If an active crossover exists, close it and record sample
                    if active_crossover:
                        entry_p = active_crossover["entry_price"]
                        exit_p = c_close
                        sig = active_crossover["signal"]
                        pnl = (exit_p - entry_p) if sig == "BUY" else (entry_p - exit_p)
                        target = 1 if pnl > 0 else 0

                        sample = {
                            "timestamp": active_crossover["timestamp"],
                            "symbol": symbol,
                            "signal": sig,
                            "entry_ltp": entry_p,
                            "exit_ltp": exit_p,
                            "pnl": round(pnl, 2)
                        }

                        # Copy only the 17 candle-verifiable features
                        for feat_name in FeatureExtractor.HISTORICAL_CANDLE_FEATURE_NAMES:
                            sample[feat_name] = active_crossover["features"].get(feat_name, 0.0)

                        # Target as FINAL column
                        sample["target"] = target
                        rows.append(sample)

                    # Extract features for new crossover (using only candle close history)
                    past_closes = [c.close for c in candles[:i+1]]
                    full_feats = FeatureExtractor.extract_features(
                        signal=event.signal,
                        ltp=c_close,
                        smma20_curr=s20,
                        smma120_curr=s120,
                        smma20_prev=prev_s20,
                        smma120_prev=prev_s120,
                        close_prices=past_closes,
                        tick_metrics={}  # No tick/depth metrics in historical v2 dataset
                    )

                    active_crossover = {
                        "timestamp": candle.timestamp.isoformat(),
                        "symbol": symbol,
                        "signal": event.signal,
                        "entry_price": c_close,
                        "features": full_feats
                    }

        df = pd.DataFrame(rows)
        if not df.empty:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Successfully constructed and saved historical ML dataset v2 with {len(df)} crossover trade samples at {output_file}")
        else:
            logger.warning("No crossover trade samples generated for dataset v2.")

        return df

    @staticmethod
    def build_dataset_from_candles(
        candles_by_symbol: Dict[str, List[Candle]],
        period_fast: int = Config.SMMA_FAST,
        period_slow: int = Config.SMMA_SLOW
    ) -> pd.DataFrame:
        """Backward compatibility wrapper delegating to build_dataset_v2."""
        return DatasetBuilder.build_dataset_v2(candles_by_symbol, period_fast, period_slow)
