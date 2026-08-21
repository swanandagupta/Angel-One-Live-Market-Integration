import pytest
from datetime import datetime
from broker.angel_symbol_master import AngelSymbolMaster, AngelSymbolInfo
from broker.angel_client import AngelClient
from broker.mock_angel_stream import MockAngelStreamer
from broker.models import MarketTick

def test_angel_symbol_master_fallback_resolution():
    master = AngelSymbolMaster()
    # Test fallback population without needing network download during fast unit test
    master._populate_fallback_symbols()
    master._loaded = True

    info_sbin = master.get_info_by_symbol("SBIN")
    assert info_sbin is not None
    assert info_sbin.symbol == "SBIN"
    assert info_sbin.exchange == "NSE"
    assert info_sbin.token == "3045"

    info_by_token = master.get_info_by_token("3045")
    assert info_by_token is not None
    assert info_by_token.symbol == "SBIN"

def test_angel_websocket_parser_normalization():
    client = AngelClient(mock_mode=True)
    client.symbol_master._populate_fallback_symbols()

    # Simulated SmartWebSocketV2 SNAP_QUOTE parsed output
    mock_payload = {
        "subscription_mode": 3,
        "exchange_type": 1,
        "token": "3045",  # SBIN
        "sequence_number": 12345,
        "exchange_timestamp": 1692500000000,
        "last_traded_price": 55025,  # 550.25 rupees (paise / 100)
        "last_traded_quantity": 250.0,
        "total_buy_quantity": 1_500_000.0,
        "total_sell_quantity": 1_200_000.0,
        "best_5_buy_data": [{"flag": 0, "quantity": 300000, "price": 55020}],
        "best_5_sell_data": [{"flag": 1, "quantity": 240000, "price": 55030}]
    }

    tick = client.parse_websocket_message(mock_payload)
    assert tick is not None
    assert isinstance(tick, MarketTick)
    assert tick.symbol == "SBIN"
    assert tick.ltp == pytest.approx(550.25)
    assert tick.ltq == 250.0
    assert tick.bid_price == pytest.approx(550.20)
    assert tick.ask_price == pytest.approx(550.30)
    assert tick.bid_quantity == 1_500_000.0
    assert tick.ask_quantity == 1_200_000.0

def test_mock_angel_streamer_emission():
    client = AngelClient(mock_mode=True)
    client.symbol_master._populate_fallback_symbols()

    streamer = MockAngelStreamer(client, ["SBIN", "RELIANCE"])
    tick = streamer.emit_tick("RELIANCE")

    assert tick is not None
    assert tick.symbol == "RELIANCE"
    assert tick.ltp > 0
    assert tick.bid_quantity > 1_000_000

def test_angel_client_mock_connect():
    client = AngelClient(mock_mode=True)
    assert client.connect() is True
    assert client.is_connected() is True
    assert client.is_auth_failed() is False
