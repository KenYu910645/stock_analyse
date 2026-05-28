from __future__ import annotations

from realtime.config import RealtimeConfig
from realtime.fubon_client import FubonRealtimeCollector


class RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def insert_raw_event(self, row):
        self.calls.append("raw")
        return 42

    def insert_trade(self, row):
        self.calls.append("trade")

    def insert_orderbook(self, row):
        self.calls.append("book")


def make_config() -> RealtimeConfig:
    return RealtimeConfig(
        fubon_id="id",
        fubon_password="password",
        fubon_cert_path="cert.pfx",
        fubon_cert_password="cert-password",
        fubon_mode="Speed",
        fubon_ws_url="",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="market_data",
        postgres_user="postgres",
        postgres_password="postgres",
        symbols=["2330"],
        channels=["trades", "books"],
        reconnect_delay_seconds=0,
    )


def test_on_message_inserts_raw_before_normalized_trade() -> None:
    writer = RecordingWriter()
    collector = FubonRealtimeCollector(make_config(), writer)

    collector.on_message(
        {
            "event": "data",
            "channel": "trades",
            "data": {"symbol": "2330", "price": 100, "size": 1},
        }
    )

    assert writer.calls == ["raw", "trade"]


def test_parse_failure_keeps_raw_event_saved(monkeypatch) -> None:
    writer = RecordingWriter()
    collector = FubonRealtimeCollector(make_config(), writer)

    def fail_parser(*_args, **_kwargs):
        raise ValueError("bad parser")

    monkeypatch.setattr("realtime.fubon_client.parse_trade_event", fail_parser)

    collector.on_message(
        {
            "event": "data",
            "channel": "trades",
            "data": {"symbol": "2330", "price": 100, "size": 1},
        }
    )

    assert writer.calls == ["raw"]


def test_disconnect_reconnects_when_running(monkeypatch) -> None:
    writer = RecordingWriter()
    collector = FubonRealtimeCollector(make_config(), writer)
    collector.running = True
    calls: list[str] = []

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr(collector, "init_realtime", lambda: calls.append("init"))
    monkeypatch.setattr(collector, "connect", lambda: calls.append("connect"))
    monkeypatch.setattr(collector, "subscribe", lambda: calls.append("subscribe"))

    collector.on_disconnect("closed")

    assert calls == ["init", "connect", "subscribe"]


def test_reconnect_retries_with_fresh_client_after_closed_socket(monkeypatch) -> None:
    writer = RecordingWriter()
    collector = FubonRealtimeCollector(make_config(), writer)
    collector.running = True
    calls: list[str] = []

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    monkeypatch.setattr(collector, "init_realtime", lambda: calls.append("init"))
    monkeypatch.setattr(collector, "connect", lambda: calls.append("connect"))

    attempts = {"count": 0}

    def subscribe() -> None:
        calls.append("subscribe")
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("Connection is already closed.")

    monkeypatch.setattr(collector, "subscribe", subscribe)

    collector.reconnect()

    assert calls == ["init", "connect", "subscribe", "init", "connect", "subscribe"]
