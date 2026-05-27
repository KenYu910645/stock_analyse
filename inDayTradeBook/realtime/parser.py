"""Defensive parsers for Fubon realtime market data events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any


NON_DATA_EVENTS = {
    "authenticated",
    "heartbeat",
    "pong",
    "subscribed",
    "unsubscribed",
    "error",
}


def ensure_event_dict(message: dict[str, Any] | str | bytes) -> dict[str, Any]:
    if isinstance(message, dict):
        return message

    if isinstance(message, bytes):
        message = message.decode("utf-8")

    if isinstance(message, str):
        parsed = json.loads(message)
        if not isinstance(parsed, dict):
            raise ValueError("Fubon message JSON must decode to an object.")
        return parsed

    raise TypeError(f"Unsupported Fubon message type: {type(message)!r}")


def should_normalize_event(event: dict[str, Any]) -> bool:
    return event.get("event") == "data"


def _payload_data(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_exchange_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10**17:
            timestamp /= 1_000_000_000
        elif timestamp > 10**14:
            timestamp /= 1_000_000
        elif timestamp > 10**11:
            timestamp /= 1_000
        return datetime.fromtimestamp(timestamp, timezone.utc)

    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return parse_exchange_timestamp(int(text))
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    return None


def parse_common_fields(
    message: dict[str, Any] | str | bytes,
    received_at: datetime,
) -> dict[str, Any]:
    event = ensure_event_dict(message)
    data = _payload_data(event)
    exchange_ts = parse_exchange_timestamp(data.get("time"))

    return {
        "received_at": received_at,
        "exchange_ts": exchange_ts,
        "symbol": str(data.get("symbol") or event.get("symbol") or ""),
        "channel": str(event.get("channel") or data.get("channel") or ""),
        "event_type": event.get("event"),
        "payload": event,
    }


def parse_trade_event(
    message: dict[str, Any] | str | bytes,
    received_at: datetime,
    raw_event_id: int | None = None,
) -> dict[str, Any] | None:
    event = ensure_event_dict(message)
    if not should_normalize_event(event):
        return None

    data = _payload_data(event)
    ts = parse_exchange_timestamp(data.get("time")) or received_at

    return {
        "ts": ts,
        "received_at": received_at,
        "symbol": str(data.get("symbol") or ""),
        "price": _to_decimal(data.get("price")),
        "size": _to_int(data.get("size")),
        "volume": _to_int(data.get("volume")),
        "bid": _to_decimal(data.get("bid")),
        "ask": _to_decimal(data.get("ask")),
        "raw_event_id": raw_event_id,
        "payload": event,
    }


def _book_level_rows(levels: Any) -> list[dict[str, Any]]:
    if not isinstance(levels, list):
        return []
    return [level for level in levels[:5] if isinstance(level, dict)]


def _add_book_side(row: dict[str, Any], side: str, levels: Any) -> None:
    level_rows = _book_level_rows(levels)
    for index in range(5):
        level = level_rows[index] if index < len(level_rows) else {}
        row[f"{side}{index + 1}_price"] = _to_decimal(level.get("price"))
        row[f"{side}{index + 1}_size"] = _to_int(level.get("size"))


def parse_book_event(
    message: dict[str, Any] | str | bytes,
    received_at: datetime,
) -> dict[str, Any] | None:
    event = ensure_event_dict(message)
    if not should_normalize_event(event):
        return None

    data = _payload_data(event)
    ts = parse_exchange_timestamp(data.get("time")) or received_at
    row: dict[str, Any] = {
        "ts": ts,
        "received_at": received_at,
        "symbol": str(data.get("symbol") or ""),
        "payload": event,
    }
    _add_book_side(row, "bid", data.get("bids"))
    _add_book_side(row, "ask", data.get("asks"))
    return row
