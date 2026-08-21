import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from broker.models import Candle
from dashboard.styles import (
    COLOR_BG, COLOR_PANEL, COLOR_BORDER, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED, COLOR_GREEN, COLOR_RED, COLOR_ACCENT
)

TERMINAL_LAYOUT_CONFIG = dict(
    paper_bgcolor=COLOR_PANEL,
    plot_bgcolor=COLOR_PANEL,
    font=dict(family="Inter, sans-serif", size=11, color=COLOR_TEXT_PRIMARY),
    margin=dict(l=35, r=20, t=35, b=30),
    xaxis=dict(
        gridcolor="rgba(42, 48, 56, 0.4)",
        zerolinecolor=COLOR_BORDER,
        tickfont=dict(family="Roboto Mono", size=10, color=COLOR_TEXT_SECONDARY),
        showline=True,
        linecolor=COLOR_BORDER
    ),
    yaxis=dict(
        gridcolor="rgba(42, 48, 56, 0.4)",
        zerolinecolor=COLOR_BORDER,
        tickfont=dict(family="Roboto Mono", size=10, color=COLOR_TEXT_SECONDARY),
        showline=True,
        linecolor=COLOR_BORDER
    )
)

def create_smma_crossover_chart(
    candles: List[Candle],
    smma20: List[Optional[float]],
    smma120: List[Optional[float]],
    symbol: str,
    crossovers: Optional[List[Dict[str, Any]]] = None
) -> go.Figure:
    """Creates interactive Plotly OHLC price candlestick chart with SMMA20 and SMMA120 overlays."""
    if not candles:
        fig = go.Figure()
        fig.update_layout(**TERMINAL_LAYOUT_CONFIG, title="NO PRICE DATA AVAILABLE")
        return fig

    df = pd.DataFrame([c.to_dict() for c in candles])
    df["smma20"] = smma20[:len(df)]
    df["smma120"] = smma120[:len(df)]

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="LTP",
        increasing_line_color=COLOR_GREEN,
        decreasing_line_color=COLOR_RED,
        increasing_fillcolor=COLOR_GREEN,
        decreasing_fillcolor=COLOR_RED
    ))

    # SMMA 20 line
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["smma20"],
        mode="lines",
        name="SMMA (20)",
        line=dict(color="#3B82F6", width=1.5)
    ))

    # SMMA 120 line
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["smma120"],
        mode="lines",
        name="SMMA (120)",
        line=dict(color="#A855F7", width=1.5)
    ))

    # Add crossover signal markers if present
    if crossovers:
        for c in crossovers:
            sig = c.get("signal", "BUY")
            color = COLOR_GREEN if sig == "BUY" else COLOR_RED
            symbol_marker = "triangle-up" if sig == "BUY" else "triangle-down"
            fig.add_trace(go.Scatter(
                x=[c.get("timestamp")],
                y=[c.get("ltp")],
                mode="markers+text",
                name=f"{sig}",
                marker=dict(symbol=symbol_marker, size=12, color=color),
                text=[f"{sig} @ ₹{c.get('ltp'):.2f}"],
                textposition="top center" if sig == "BUY" else "bottom center",
                textfont=dict(family="Roboto Mono", size=10, color=color)
            ))

    layout_opts = TERMINAL_LAYOUT_CONFIG.copy()
    layout_opts.update(dict(
        title=dict(text=f"{symbol} — PRICE ACTION & SMMA CROSSOVER", font=dict(size=12, color=COLOR_TEXT_PRIMARY)),
        xaxis_rangeslider_visible=False,
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    ))
    fig.update_layout(**layout_opts)
    return fig

def create_depth_chart(bid_price: float, bid_qty: float, ask_price: float, ask_qty: float) -> go.Figure:
    """Creates a compact Order Book Market Depth comparison chart."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["BID DEPTH", "ASK DEPTH"],
        y=[bid_qty, ask_qty],
        marker_color=[COLOR_GREEN, COLOR_RED],
        text=[f"{bid_qty:,.0f} @ ₹{bid_price:.2f}", f"{ask_qty:,.0f} @ ₹{ask_price:.2f}"],
        textposition="auto",
        textfont=dict(family="Roboto Mono", size=11, color="#000000")
    ))

    layout_opts = TERMINAL_LAYOUT_CONFIG.copy()
    layout_opts.update(dict(
        title=dict(text="MARKET DEPTH", font=dict(size=12, color=COLOR_TEXT_PRIMARY)),
        height=260,
        margin=dict(l=20, r=20, t=35, b=20)
    ))
    fig.update_layout(**layout_opts)
    return fig

def create_cumulative_pnl_chart(
    df_trades: pd.DataFrame,
    title: str = "CUMULATIVE STRATEGY P/L (₹)"
) -> go.Figure:
    """Creates cumulative P/L curve for Strategy A (SMMA-only) vs Strategy B (SMMA+ML Filter)."""
    fig = go.Figure()

    if df_trades.empty or "pnl" not in df_trades.columns:
        fig.update_layout(**TERMINAL_LAYOUT_CONFIG, title="NO TRADE DATA FOR EQUITY CURVE")
        return fig

    df = df_trades.copy()
    df["cum_pnl_all"] = df["pnl"].cumsum()

    fig.add_trace(go.Scatter(
        x=np.arange(1, len(df) + 1),
        y=df["cum_pnl_all"],
        mode="lines+markers",
        name="Strategy A: SMMA-Only",
        line=dict(color="#3B82F6", width=2)
    ))

    if "ml_decision" in df.columns:
        df_ml = df[df["ml_decision"] == "ACCEPT"].copy()
        if not df_ml.empty:
            df_ml["cum_pnl_ml"] = df_ml["pnl"].cumsum()
            fig.add_trace(go.Scatter(
                x=np.arange(1, len(df_ml) + 1),
                y=df_ml["cum_pnl_ml"],
                mode="lines+markers",
                name="Strategy B: SMMA + ML Filter",
                line=dict(color=COLOR_GREEN, width=2)
            ))

    layout_opts = TERMINAL_LAYOUT_CONFIG.copy()
    layout_opts.update(dict(
        title=dict(text=title, font=dict(size=12, color=COLOR_TEXT_PRIMARY)),
        xaxis_title="Trade Index",
        yaxis_title="Cumulative P/L (₹)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10))
    ))
    fig.update_layout(**layout_opts)
    return fig

def create_feature_importance_chart(feature_importances: Dict[str, float]) -> go.Figure:
    """Creates horizontal bar chart for ML feature importances."""
    if not feature_importances:
        fig = go.Figure()
        fig.update_layout(**TERMINAL_LAYOUT_CONFIG, title="NO FEATURE IMPORTANCES AVAILABLE")
        return fig

    top_items = list(feature_importances.items())[:12]
    features = [item[0] for item in reversed(top_items)]
    importances = [item[1] for item in reversed(top_items)]

    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation="h",
        marker=dict(
            color=importances,
            colorscale="Viridis"
        )
    ))

    layout_opts = TERMINAL_LAYOUT_CONFIG.copy()
    layout_opts.update(dict(
        title=dict(text="QUANTITATIVE FEATURE IMPORTANCES", font=dict(size=12, color=COLOR_TEXT_PRIMARY)),
        xaxis_title="Importance Weight",
        height=380,
        margin=dict(l=120, r=20, t=35, b=30)
    ))
    fig.update_layout(**layout_opts)
    return fig
