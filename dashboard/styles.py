"""
QuantScreen Quantitative Market Intelligence — Design System & CSS Styling
High-contrast, professional dark trading workstation design system.
"""

import streamlit as st

# Color Palette System
COLOR_BG = "#0B0E11"              # Very dark charcoal canvas
COLOR_PANEL = "#15191F"           # Panel / Card background
COLOR_ELEVATED = "#1A1F26"        # Elevated container background
COLOR_BORDER = "#2A3038"          # Subtle dark gray border
COLOR_TEXT_PRIMARY = "#F2F4F7"    # High-contrast off-white primary text
COLOR_TEXT_SECONDARY = "#A7AFBC"  # Muted secondary text
COLOR_TEXT_MUTED = "#727B89"      # Muted label text
COLOR_GREEN = "#16C784"           # Professional positive green
COLOR_RED = "#EA3943"             # Professional negative red
COLOR_ACCENT = "#F0B90B"          # Restrained gold accent

def inject_terminal_css():
    """Injects high-contrast custom CSS for QuantScreen quantitative trading workstation."""
    st.markdown(f"""
    <style>
        /* Import Professional Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;600;700&display=swap');

        /* Global Canvas & Reset */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: {COLOR_BG} !important;
            color: {COLOR_TEXT_PRIMARY} !important;
        }}

        .stApp {{
            background-color: {COLOR_BG} !important;
        }}

        /* Streamlit Core Overrides & Full-Width Container Spacing */
        header[data-testid="stHeader"] {{
            background-color: {COLOR_BG} !important;
            z-index: 99 !important;
        }}
        footer {{
            display: none !important;
        }}

        /* STRETCH FULL PAGE WIDTH - NO WASTED MARGINS */
        .stApp, .main, section.main,
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        div[data-testid="stMainBlockContainer"],
        .block-container {{
            max-width: 100% !important;
            width: 100% !important;
            padding-top: 1.2rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 1.0rem !important;
            padding-right: 1.0rem !important;
        }}

        /* Header Bar Container */
        .nexus-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: {COLOR_PANEL} !important;
            border: 1px solid {COLOR_BORDER} !important;
            padding: 12px 20px !important;
            margin-bottom: 16px !important;
            border-radius: 4px !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }}

        /* 7 Equal Height Metric Strip */
        .metric-box {{
            background-color: {COLOR_PANEL} !important;
            border: 1px solid {COLOR_BORDER} !important;
            border-radius: 4px !important;
            padding: 10px 14px !important;
            height: 64px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            box-sizing: border-box !important;
            margin-bottom: 14px !important;
        }}
        .metric-box-label {{
            font-size: 10px !important;
            font-weight: 700 !important;
            color: {COLOR_TEXT_SECONDARY} !important;
            text-transform: uppercase !important;
            letter-spacing: 0.6px !important;
            margin-bottom: 3px !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}
        .metric-box-value {{
            font-family: 'Roboto Mono', monospace !important;
            font-size: 17px !important;
            font-weight: 700 !important;
            color: {COLOR_TEXT_PRIMARY} !important;
            line-height: 1.2 !important;
        }}

        /* Section Headers */
        .section-header-row {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 12px;
            border-bottom: 1px solid {COLOR_BORDER};
            padding-bottom: 8px;
            width: 100%;
        }}
        .section-title {{
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.6px;
            color: {COLOR_TEXT_PRIMARY};
            text-transform: uppercase;
        }}
        .section-meta {{
            font-size: 12px;
            font-weight: 500;
            color: {COLOR_TEXT_SECONDARY};
        }}

        /* HIGH-VISIBILITY FULL-WIDTH TABLE STYLING */
        div[data-testid="stTable"] {{
            overflow-x: auto !important;
            width: 100% !important;
            max-width: 100% !important;
            display: block !important;
            border: 1px solid {COLOR_BORDER} !important;
            border-radius: 4px !important;
            background-color: {COLOR_PANEL} !important;
            margin-bottom: 18px !important;
        }}

        div[data-testid="stTable"] table {{
            font-family: 'Roboto Mono', monospace !important;
            font-size: 13px !important;
            color: {COLOR_TEXT_PRIMARY} !important;
            background-color: {COLOR_PANEL} !important;
            width: 100% !important;
            border-collapse: collapse !important;
        }}

        div[data-testid="stTable"] th {{
            background-color: #1E2329 !important;
            color: {COLOR_TEXT_SECONDARY} !important;
            font-weight: 700 !important;
            font-size: 11px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            border-bottom: 2px solid {COLOR_BORDER} !important;
            padding: 10px 14px !important;
            white-space: nowrap !important;
            text-align: right !important;
        }}

        div[data-testid="stTable"] td {{
            border-bottom: 1px solid {COLOR_BORDER} !important;
            padding: 9px 14px !important;
            white-space: nowrap !important;
            text-align: right !important;
            color: {COLOR_TEXT_PRIMARY} !important;
            font-size: 13px !important;
        }}

        div[data-testid="stTable"] th:first-child,
        div[data-testid="stTable"] td:first-child,
        div[data-testid="stTable"] th:nth-child(2),
        div[data-testid="stTable"] td:nth-child(2) {{
            text-align: left !important;
            color: {COLOR_TEXT_PRIMARY} !important;
            font-weight: 600 !important;
        }}

        div[data-testid="stTable"] tr:nth-child(even) td {{
            background-color: #1A1F26 !important;
        }}

        div[data-testid="stTable"] tr:hover td {{
            background-color: #222831 !important;
        }}

        /* Compact Professional Badges */
        .badge-buy {{
            color: {COLOR_GREEN};
            background-color: rgba(22, 199, 132, 0.15);
            padding: 3px 8px;
            border-radius: 2px;
            font-size: 11px;
            font-weight: 700;
            font-family: 'Roboto Mono', monospace;
            border: 1px solid rgba(22, 199, 132, 0.4);
        }}
        .badge-sell {{
            color: {COLOR_RED};
            background-color: rgba(234, 57, 67, 0.15);
            padding: 3px 8px;
            border-radius: 2px;
            font-size: 11px;
            font-weight: 700;
            font-family: 'Roboto Mono', monospace;
            border: 1px solid rgba(234, 57, 67, 0.4);
        }}
        .badge-neutral {{
            color: {COLOR_TEXT_SECONDARY};
            background-color: rgba(167, 175, 188, 0.15);
            padding: 3px 8px;
            border-radius: 2px;
            font-size: 11px;
            font-weight: 600;
            font-family: 'Roboto Mono', monospace;
            border: 1px solid rgba(167, 175, 188, 0.4);
        }}

        /* HIGH-CONTRAST SIDEBAR STYLING */
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_PANEL} !important;
            border-right: 1px solid {COLOR_BORDER} !important;
        }}

        .sidebar-section-header {{
            font-size: 11px;
            font-weight: 700;
            color: {COLOR_TEXT_SECONDARY} !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 14px;
            margin-bottom: 6px;
        }}

        /* High-Contrast Labels for Radio, Sliders, Selectboxes */
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {COLOR_TEXT_PRIMARY} !important;
            font-size: 12px !important;
            font-weight: 500 !important;
        }}

        /* Clean Sidebar Radio Items */
        div[data-testid="stSidebar"] div[role="radiogroup"] label {{
            background-color: {COLOR_PANEL} !important;
            border: 1px solid {COLOR_BORDER} !important;
            border-radius: 3px !important;
            padding: 8px 12px !important;
            margin-bottom: 4px !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            transition: all 0.15s ease !important;
        }}

        /* Suppress Streamlit Radio Circles */
        div[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
        div[data-testid="stSidebar"] div[role="radiogroup"] label svg,
        div[data-testid="stSidebar"] div[role="radiogroup"] label circle {{
            display: none !important;
            visibility: hidden !important;
            width: 0px !important;
            height: 0px !important;
        }}

        div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            border-color: {COLOR_TEXT_SECONDARY} !important;
            background-color: {COLOR_ELEVATED} !important;
        }}

        /* Active Navigation Item Highlight */
        div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
            border-left: 3px solid {COLOR_ACCENT} !important;
            border-top: 1px solid {COLOR_BORDER} !important;
            border-right: 1px solid {COLOR_BORDER} !important;
            border-bottom: 1px solid {COLOR_BORDER} !important;
            background-color: {COLOR_ELEVATED} !important;
        }}
        div[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] span {{
            color: {COLOR_TEXT_PRIMARY} !important;
            font-weight: 600 !important;
        }}

        /* Progress Bar */
        .prob-bar-container {{
            width: 100%;
            background-color: {COLOR_BG};
            border-radius: 2px;
            height: 8px;
            margin: 6px 0;
            overflow: hidden;
            border: 1px solid {COLOR_BORDER};
        }}
        .prob-bar-fill {{
            height: 100%;
            transition: width 0.3s ease;
        }}

        /* Scrollbars */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: {COLOR_BG}; }}
        ::-webkit-scrollbar-thumb {{ background: {COLOR_BORDER}; border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {COLOR_TEXT_MUTED}; }}
    </style>
    """, unsafe_allow_html=True)
