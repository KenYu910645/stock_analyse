from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal

from realtime.parser import (
    parse_book_event,
    parse_common_fields,
    parse_trade_event,
)


RECEIVED_AT = datetime(2026, 5, 17, 1, 2, 3, tzinfo=timezone.utc)


def test_parse_trade_event_with_documented_payload() -> None:
    event = {
        "event": "data",
        "channel": "trades",
        "data": {
            "symbol": "2330",
            "time": 1685338200000000,
            "price": 590,
            "size": 1,
            "volume": 12345,
            "bid": 589,
            "ask": 590,
        },
    }

    row = parse_trade_event(event, RECEIVED_AT, raw_event_id=7)

    assert row is not None
    assert row["symbol"] == "2330"
    assert row["price"] == Decimal("590")
    assert row["size"] == 1
    assert row["raw_event_id"] == 7
    assert row["ts"].tzinfo is not None


def test_parse_book_event_with_five_levels() -> None:
    event = {
        "event": "data",
        "channel": "books",
        "data": {
            "symbol": "2308",
            "time": 1685338200000000,
            "bids": [
                {"price": 100 + index, "size": 10 + index}
                for index in range(5)
            ],
            "asks": [
                {"price": 101 + index, "size": 20 + index}
                for index in range(5)
            ],
        },
    }

    row = parse_book_event(event, RECEIVED_AT)

    assert row is not None
    assert row["symbol"] == "2308"
    assert row["bid1_price"] == Decimal("100")
    assert row["bid5_size"] == 14
    assert row["ask1_price"] == Decimal("101")
    assert row["ask5_size"] == 24


def test_parse_book_event_fills_missing_levels_with_none() -> None:
    event = {
        "event": "data",
        "channel": "books",
        "data": {
            "symbol": "3105",
            "bids": [{"price": 50, "size": 3}],
            "asks": [],
        },
    }

    row = parse_book_event(event, RECEIVED_AT)

    assert row is not None
    assert row["ts"] == RECEIVED_AT
    assert row["bid1_price"] == Decimal("50")
    assert row["bid2_price"] is None
    assert row["ask1_price"] is None
    assert row["ask5_size"] is None


def test_parse_common_fields_accepts_json_string() -> None:
    event = {
        "event": "data",
        "channel": "trades",
        "data": {"symbol": "2330", "time": "2026-05-17T01:02:03Z"},
    }

    row = parse_common_fields(json.dumps(event), RECEIVED_AT)

    assert row["symbol"] == "2330"
    assert row["channel"] == "trades"
    assert row["event_type"] == "data"
    assert row["exchange_ts"] == RECEIVED_AT


def test_non_data_events_do_not_normalize() -> None:
    event = {"event": "heartbeat", "channel": "trades", "data": {"symbol": "2330"}}

    assert parse_trade_event(event, RECEIVED_AT) is None
    assert parse_book_event(event, RECEIVED_AT) is None
