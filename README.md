# ⚡ QuantScreen Terminal: High-Frequency NSE Quantitative Market Screening & ML Signal Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20XGBoost%20%7C%20Plotly-orange.svg)](https://streamlit.io/)
[![Broker API](https://img.shields.io/badge/broker-Angel%20One%20SmartAPI%20V2-blue.svg)](https://smartapi.angelbroking.com/)
[![Test Suite](https://img.shields.io/badge/tests-39%2F39%20PASSED-green.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**QuantScreen Terminal** is an institutional-grade, real-time quantitative stock screening, market microstructure analysis, state-tracked SMMA crossover detector, and direction-specific XGBoost Machine Learning signal engine designed for National Stock Exchange (NSE) equities.

Powered by direct integration with **Angel One SmartAPI REST Services** and **SmartWebSocketV2 (Mode 3 SNAP_QUOTE)**, the platform ingests live level-2 market depth, tracks volume acceleration across rolling time windows, and evaluates technical crossovers through specialized Machine Learning inference models.

---

## 🏛️ Advanced System Architecture & Data Pipeline

```text
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │                            LIVE ANGEL ONE SMARTAPI GATEWAY                              │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                ┌─────────────────────────────┴─────────────────────────────┐
                ▼                                                           ▼
  [ SmartAPI REST Client (SmartConnect) ]                  [ SmartWebSocketV2 Streamer ]
  • PyOTP Automated TOTP Authentication                   • Mode 3: SNAP_QUOTE Feed Parser
  • JWT Session & Feed Token Refresh                      • Binary WebSocket Packet Decoder
  • Scrip Master Resolution (9,900+ Scrips)               • Sub-Millisecond Tick Emitter
                │                                                           │
                └─────────────────────────────┬─────────────────────────────┘
                                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │                   HIGH-THROUGHPUT IN-MEMORY STREAMING INGESTION LAYER                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────┤
 │  • Liquidity & Price Screener: LTP (₹30–₹500) & Depth Filter (Bid Qty > 10L & Ask Qty > 10L) │
 │  • Circular Tick Store: Sliding Window ETQ (5m / 20m / 60m) & Avg LTP (20m / 60m)      │
 │  • Order Book Depth Engine: Real-Time 5-Level Bid/Ask Volume Imbalance Metrics            │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │                           REAL-TIME 1M CANDLE & SMMA ENGINE                             │
 ├─────────────────────────────────────────────────────────────────────────────────────────┤
 │  • 1-Minute OHLC Candle Aggregator (Thread-Safe In-Memory Resampler)                    │
 │  • SMMA(20) & SMMA(120) Wilder's Exponential Smoothing Indicators                       │
 │  • Stateful Non-Repainting Crossover Detector (Golden Cross BUY / Death Cross SELL)     │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │                       DIRECTION-SPECIFIC XGBOOST ML INFERENCE ENGINE                    │
 ├─────────────────────────────────────────────────────────────────────────────────────────┤
 │  • 17 Candle-Derived Feature Engineering (Zero Lookahead Bias / Zero Hallucinated Depth)│
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                        │                                           │
         ┌──────────────┴──────────────┐             ┌──────────────┴──────────────┐
         ▼                             ▼             ▼                             ▼
  [ BUY Crossover Event ]      [ ML Model Inference ] [ SELL Crossover Event ]    [ ML Model Inference ]
  • Direction: LONG            • XGBoost BUY Model   • Direction: SHORT           • XGBoost SELL Model
  • Feature Matrix: 17 Vector  • Threshold: 0.55     • Feature Matrix: 17 Vector  • Threshold: 0.70
  • Probability Scoring        • ROC-AUC: 0.7216     • Probability Scoring        • ROC-AUC: 0.6083
         │                             │             │                             │
         └──────────────┬──────────────┘             └──────────────┬──────────────┘
                        │                                           │
                        └─────────────────────┬─────────────────────┘
                                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │                         QUANTITATIVE DECISION & REASON ENGINE                           │
 ├─────────────────────────────────────────────────────────────────────────────────────────┤
 │  • ACCEPT Signal: Probability >= Directional Model Threshold                             │
 │  • AVOID Signal:  Probability < Directional Model Threshold                             │
 │  • Quant Explainer: Human-Readable Feature Attribution Breakdown                        │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │                      PAPER TRADING ENGINE & REAL-TIME DASHBOARD UI                      │
 ├─────────────────────────────────────────────────────────────────────────────────────────┤
 │  • Strict Temporal Integrity Engine (Guarantees exit_time > entry_time)                 │
 │  • Position-Aware PnL Calculator (BUY: Exit-Entry | SELL: Entry-Exit)                   │
 │  • Full-Width Streamlit Custom Terminal UI with Real-Time Plotly Depth & Candle Charts  │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔌 Angel One SmartAPI Integration Details

QuantScreen integrates directly with Angel One's institutional trading infrastructure:

1. **Authentication Protocol**:
   - Automated Time-based One-Time Password (TOTP) generation using `pyotp` and secret seed injection.
   - Dual-token session maintenance (JWT `jwtToken` for REST endpoints & `feedToken` for WebSocket streaming).
   - Silent background token refresh cycle to ensure zero connection drops during market hours.

2. **Scrip Master Engine (`angel_symbol_master.py`)**:
   - Ingests and parses Angel One's official daily Open API instrument master file (~9,900+ equity symbols).
   - Dynamic mapping between NSE Stock Tickers (e.g. `RELIANCE`, `SBIN`, `TATAMOTORS`) and Angel One Symbol Tokens (e.g. `"2885"`, `"3045"`).
   - Fallback offline token resolver to guarantee uninterrupted offline/paper testing.

3. **High-Speed WebSocket Feed (`angel_websocket.py`)**:
   - Connects to `wss://smartapisocket.angelone.in/smart-stream` using `SmartWebSocketV2`.
   - Subscribes using **`Mode 3: SNAP_QUOTE`** to receive comprehensive 125-byte binary data packets.
   - Decodes real-time Last Traded Price (LTP), Last Traded Quantity (LTQ), Total Buy/Sell Quantities, Best 5 Bid/Ask Depth quotes, Average Traded Price (ATP), and Volume Traded.

---

## 🤖 Direction-Specific Production XGBoost Machine Learning Models

QuantScreen replaces generic symmetric classifiers with **two specialized direction-specific XGBoost models** trained on 634 genuine historical SMMA crossover trade outcomes across 50 core NSE stocks:

### Strategy Performance Benchmark (Unseen Historical Test Set, $N=96$)

Evaluating performance on a strictly chronological, un-shuffled unseen test dataset ($N=96$ crossover trades across 50 NSE stocks):

| Strategy / Model Configuration | Accepted Trades | Win Rate | Average P/L | Total Strategy P/L | Profit Factor | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A: Pure Technical SMMA Only** | 96 Trades | 27.08% | -₹2.56 | -₹246.22 | 0.8037 | ₹388.01 |
| **Strategy B: Unified Single XGBoost Model** | 28 Trades | 32.14% | -2.56 | -₹71.75 | 0.8292 | ₹161.65 |
| **Strategy C: Production Direction-Specific Models** | **21 Trades** | **33.33%** | **+₹9.60** | **+₹201.59 (NET PROFIT)** | **1.6615** | **₹88.47 (77.2% Reduction)** |

### Production Model Architecture & Hyperparameters
- **BUY Model (`models/xgboost_buy_model_v2.joblib`)**:
  - **Threshold**: **`0.55`** | **ROC-AUC**: **`0.7216`** | **Test Win Rate**: **`60.00%`** | **BUY Strategy P/L**: **`+₹355.86`** | **Profit Factor**: **`4.5353`**
  - Designed to capture bullish momentum breakouts where short-term SMMA(20) crosses above long-term SMMA(120).
- **SELL Model (`models/xgboost_sell_model_v2.joblib`)**:
  - **Threshold**: **`0.70`** | **ROC-AUC**: **`0.6083`** | **Test Win Rate**: **`9.09%`** | **SELL Strategy P/L**: **`-₹154.27`** (Strict risk filter)
  - Acts as an aggressive short-side risk filter, blocking weak breakdown signals during strong structural bull markets.

### 17 Candle-Derived Feature Engineering Matrix
To ensure 100% data integrity and eliminate synthetic data hallucination, the models consume **strictly 17 candle-derived features** computed from historical candles prior to signal trigger:
- Trend Indicators: `smma20`, `smma120`, `smma_gap`, `smma20_slope`, `smma120_slope`, `smma_gap_change`
- Price Momentum: `ltp`, `return_1m`, `return_5m`, `return_20m`
- Moving Averages: `avg_ltp_20m`, `avg_ltp_60m`, `dis` (Distance to SMMA20)
- Volatility Metrics: `hl_spread`, `std_20m`, `volatility_ratio`, `signal_dir`

---

## 🛡️ Core System Invariants & Risk Engine Rules

1. **Signal Equality Invariant**:
   $$\text{ACTIVE SIGNALS} = \text{ACCEPTED SIGNALS} + \text{AVOIDED SIGNALS}$$
   - Every technical crossover detected by the SMMA engine is passed to the ML layer. Signals above the directional threshold are `ACCEPTED`, while signals below are marked `AVOIDED`.
2. **Temporal Order Preservation**:
   $$\text{exit\_timestamp} > \text{entry\_timestamp}$$
   - Position tracking enforces strict chronological order. Non-monotonic trade attempts are rejected automatically to guarantee zero temporal data leakage.
3. **Directional PnL Formula**:
   - Long Positions (BUY): $\text{PnL} = \text{Exit Price} - \text{Entry Price}$
   - Short Positions (SELL): $\text{PnL} = \text{Entry Price} - \text{Exit Price}$
   - Profitable Flag: $\text{Profitable} = (\text{PnL} > 0)$

---

## 📁 Project Directory Structure

```text
QuantScreen/
├── .env.example             # Environment variable template with broker placeholders
├── .gitignore               # Excludes secrets, logs, binaries, and large media files
├── README.md                # Institutional documentation & system specification
├── app.py                   # Streamlit web application launcher
├── config.py                # System-wide configuration parameters & paths
├── requirements.txt         # Dependency manifest
├── run_app.py               # Standalone runner script
├── run_app.spec             # PyInstaller executable specification
├── backtest/                # Historical backtest engine & performance evaluator
├── broker/                  # Angel One SmartAPI, SmartWebSocketV2, & Mock Streamer
├── dashboard/               # Full-width custom Streamlit dashboard UI & Plotly charts
├── data/                    # Candle resampler, in-memory tick store, & historical loader
├── data_storage/            # Audited ML dataset & NSE symbol master file
├── features/                # 17-vector feature engineering pipeline
├── indicators/              # Incremental Wilder's SMMA calculator
├── ml/                      # XGBoost model training, inference, & probability explainer
├── models/                  # Production XGBoost joblib model artifacts
├── scanner/                 # Stock price (₹30-500) & liquidity depth (Bid/Ask > 10L) screener
├── strategy/                # SMMA crossover detector & paper trade PnL engine
└── tests/                   # Automated Pytest suite (39 unit tests)
```

---

## 🚀 Quickstart & Setup Guide

### 1. Clone & Setup Environment
```bash
git clone https://github.com/swanandagupta/Angel-One-Live-Market-Integration.git
cd Angel-One-Live-Market-Integration

# Create virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and fill in your Angel One credentials:
```bash
cp .env.example .env
```
```env
BROKER=ANGEL

ANGEL_API_KEY=your_api_key_here
ANGEL_CLIENT_ID=your_client_id_here
ANGEL_PASSWORD=your_password_here
ANGEL_TOTP_SECRET=your_totp_secret_here
```

### 3. Launch Live Dashboard
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

### 4. Run Automated Unit Tests
```bash
python -m pytest -v
```
Executes all **39 automated unit tests** covering Angel One integration, SMMA calculations, feature extraction, ML probability mapping, signal counter invariants, and trade engine integrity.

### 5. Build Standalone Executable
```bash
python -m PyInstaller --noconfirm run_app.spec
```
Generates standalone Windows binary in `dist/run_app/run_app.exe`.

---

## 🔒 Security & Confidentiality

- **Zero Credentials in Code**: All API keys, passwords, and TOTP secrets are dynamically loaded from local `.env` variables via `python-dotenv`.
- **Git Exclusions**: `.env`, `.env.*`, runtime logs, databases, PyInstaller build artifacts, and video recordings are strictly excluded via `.gitignore`.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.