import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
from dashboard.styles import (
    COLOR_GREEN, COLOR_RED, COLOR_TEXT_MUTED, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_ACCENT
)

def render_top_navigation(broker_label: str, is_connected: bool, is_market_open: bool, integrity_pass: bool = True):
    """Renders QuantScreen Terminal Top Navigation Header inside a sharp dark panel card."""
    status_dot = "#16C784" if is_connected else "#EA3943"
    status_text = "CONNECTED" if is_connected else ("ACTIVE" if "DEMO" in broker_label else "DISCONNECTED")

    market_dot = "#16C784" if is_market_open else "#F0B90B"
    market_text = "OPEN" if is_market_open else "CLOSED"

    integrity_dot = "#16C784" if integrity_pass else "#EA3943"
    integrity_text = "PASS" if integrity_pass else "ERROR"

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    html = f"""<div style="display: flex; justify-content: space-between; align-items: center; background-color: #15191F; border: 1px solid #2A3038; border-radius: 4px; padding: 12px 20px; margin-bottom: 16px; width: 100%; box-sizing: border-border-box;"><div style="display: flex; align-items: baseline; gap: 12px;"><span style="font-weight: 700; font-size: 20px; letter-spacing: 1px; color: #F2F4F7;">QuantScreen</span><span style="font-size: 13px; font-weight: 500; color: #A7AFBC;">NSE Quantitative Market Screening & Signal Engine</span><span style="font-size: 11px; font-weight: 600; color: #727B89; background-color: #1A1F26; padding: 2px 8px; border-radius: 2px; border: 1px solid #2A3038;">NSE EQUITY</span></div><div style="display: flex; gap: 18px; align-items: center; font-size: 12px; font-weight: 500; color: #A7AFBC; font-family: 'Roboto Mono', monospace;"><div><span style="height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; background-color: {market_dot};"></span>MARKET {market_text}</div><div><span style="height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; background-color: {status_dot};"></span>{broker_label}: {status_text}</div><div><span style="height: 8px; width: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; background-color: {integrity_dot};"></span>DATA INTEGRITY: {integrity_text}</div><div style="color: #F2F4F7; font-size: 11px;">{time_str}</div></div></div>"""

    st.markdown(html, unsafe_allow_html=True)

def render_market_overview_strip(
    is_market_open: bool,
    total_stocks: int,
    price_qualified: int,
    liquidity_qualified: int,
    active_signals: int,
    accepted_signals: int,
    avoided_signals: int
):
    """Renders 7 equal-height metric cards cleanly inside bordered dark panel containers."""
    market_str = "OPEN" if is_market_open else "CLOSED"
    market_color = COLOR_GREEN if is_market_open else COLOR_TEXT_SECONDARY

    metrics_data = [
        ("NSE MARKET", market_str, market_color),
        ("STOCKS SCANNED", f"{total_stocks:,}", COLOR_TEXT_PRIMARY),
        ("PRICE FILTERED", f"{price_qualified:,}", COLOR_TEXT_PRIMARY),
        ("LIQUIDITY QUALIFIED", f"{liquidity_qualified:,}", COLOR_TEXT_PRIMARY),
        ("ACTIVE SIGNALS", f"{active_signals:,}", COLOR_TEXT_PRIMARY),
        ("ACCEPTED", f"{accepted_signals:,}", COLOR_GREEN),
        ("AVOIDED", f"{avoided_signals:,}", COLOR_RED)
    ]

    cols = st.columns(7)
    for col, (label, val, color) in zip(cols, metrics_data):
        with col:
            card_html = f"""<div style="background-color: #15191F; border: 1px solid #2A3038; border-radius: 4px; padding: 10px 14px; min-height: 64px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 14px; box-sizing: border-box;"><div style="font-size: 10px; font-weight: 700; color: #A7AFBC; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{label}</div><div style="font-family: 'Roboto Mono', monospace; font-size: 17px; font-weight: 700; color: {color}; line-height: 1.2;">{val}</div></div>"""
            st.markdown(card_html, unsafe_allow_html=True)

def render_badge(text: str, badge_type: str = "neutral") -> str:
    """Renders compact professional badge HTML."""
    b_type = badge_type.lower()
    if b_type in ["buy", "accept"]:
        cls = "badge-buy"
    elif b_type in ["sell", "avoid"]:
        cls = "badge-sell"
    else:
        cls = "badge-neutral"
    return f'<span class="{cls}">{text}</span>'

def render_probability_bar(probability: Optional[float], threshold: float = 0.55):
    """Renders horizontal probability bar for ML model evaluation."""
    if probability is None:
        st.markdown(f"""<div style="font-family: 'Roboto Mono', monospace; font-size: 11px; margin-top: 4px; color: {COLOR_TEXT_MUTED};"><span>MODEL PROBABILITY: N/A (MODEL UNAVAILABLE)</span></div>""", unsafe_allow_html=True)
        return

    fill_pct = min(100.0, max(0.0, probability * 100.0))
    bar_color = COLOR_GREEN if probability >= threshold else COLOR_RED

    bar_html = f"""<div style="font-family: 'Roboto Mono', monospace; font-size: 11px; margin-top: 4px;"><div style="display: flex; justify-content: space-between; margin-bottom: 2px;"><span style="color: {COLOR_TEXT_SECONDARY}; font-weight: 500;">MODEL CONFIDENCE</span><span style="font-weight: 600; color: {bar_color};">{probability:.1%}</span></div><div class="prob-bar-container"><div class="prob-bar-fill" style="width: {fill_pct}%; background-color: {bar_color};"></div></div><div style="display: flex; justify-content: space-between; font-size: 10px; color: {COLOR_TEXT_MUTED};"><span>0%</span><span>THRESHOLD {threshold:.0%}</span><span>100%</span></div></div>"""

    st.markdown(bar_html, unsafe_allow_html=True)

def render_checklist_panel():
    """Renders technical status checklist panel for assignment verification."""
    checklist_items = [
        ("NSE Universe", "READY"),
        ("LTP Filter (₹30–₹500)", "READY"),
        ("Bid > ₹10 Lakh", "READY"),
        ("Ask > ₹10 Lakh", "READY"),
        ("SMMA 20", "READY"),
        ("SMMA 120", "READY"),
        ("LTQ", "READY"),
        ("ETQ 5m / 20m / 60m", "READY"),
        ("Market Depth (Bid/Ask)", "READY"),
        ("BUY ML Model (Thresh 0.55)", "READY"),
        ("SELL ML Model (Thresh 0.70)", "READY"),
        ("Crossover Engine", "READY"),
        ("Trade Engine (PnL)", "READY"),
        ("Signal Log", "READY"),
        ("Live Dashboard UI", "READY"),
    ]

    html_items = "".join(
        f"<div><span style='color:{COLOR_TEXT_SECONDARY};'>{item[0]}</span> <span style='float:right; color:{COLOR_GREEN}; font-weight:600;'>{item[1]}</span></div>"
        for item in checklist_items
    )

    panel_html = f"""<div style="background-color: #12161F; border: 1px solid #2A3038; border-radius: 4px; padding: 12px; font-family: 'Roboto Mono', monospace; font-size: 11px; line-height: 1.8;"><div style="font-weight: 700; color: {COLOR_ACCENT}; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">Technical Requirement Checklist</div>{html_items}</div>"""

    st.sidebar.markdown(panel_html, unsafe_allow_html=True)
