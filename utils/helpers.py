from typing import Union

def safe_divide(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    """Safely divide two numbers, returning fallback on ZeroDivisionError or NaN/Inf."""
    if denominator == 0.0 or denominator is None or numerator is None:
        return fallback
    try:
        res = float(numerator) / float(denominator)
        if res != res or res == float('inf') or res == float('-inf'):
            return fallback
        return res
    except (ZeroDivisionError, ValueError, TypeError):
        return fallback

def format_quantity(qty: Union[int, float]) -> str:
    """Format quantity in Lakhs/Crores for Indian equity market display."""
    if qty is None:
        return "0"
    qty = float(qty)
    if qty >= 10_000_000:
        return f"{qty / 10_000_000:.2f} Cr"
    elif qty >= 100_000:
        return f"{qty / 100_000:.2f} L"
    return f"{qty:,.0f}"

def format_currency(val: Union[int, float]) -> str:
    """Format price/currency value with Indian Rupee symbol."""
    if val is None:
        return "₹0.00"
    return f"₹{float(val):,.2f}"

def normalize_symbol(symbol: str) -> str:
    """Normalize symbol string to consistent uppercase standard (e.g., 'NSE:SBIN-EQ' -> 'SBIN')."""
    if not symbol:
        return ""
    sym = symbol.strip().upper()
    if ":" in sym:
        sym = sym.split(":")[-1]
    if sym.endswith("-EQ"):
        sym = sym[:-3]
    return sym
