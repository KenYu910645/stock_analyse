"""TimescaleDB writer for realtime market data."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Jsonb


class TimescaleWriter:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.conn: psycopg.Connection[Any] | None = None

    def connect(self) -> None:
        self.conn = psycopg.connect(self.dsn)
        self.conn.autocommit = True

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _connection(self) -> psycopg.Connection[Any]:
        if self.conn is None:
            raise RuntimeError("TimescaleWriter.connect() must be called first.")
        return self.conn

    def insert_raw_event(self, row: dict[str, Any]) -> int:
        sql = """
            INSERT INTO raw_market_events (
                received_at, exchange_ts, symbol, channel, event_type, payload
            )
            VALUES (
                %(received_at)s, %(exchange_ts)s, %(symbol)s, %(channel)s,
                %(event_type)s, %(payload)s
            )
            RETURNING id
        """
        payload = dict(row)
        payload["payload"] = Jsonb(payload["payload"])
        with self._connection().cursor() as cur:
            cur.execute(sql, payload)
            raw_id = cur.fetchone()

        if raw_id is None:
            raise RuntimeError("Raw event insert did not return an id.")
        return int(raw_id[0])

    def insert_trade(self, row: dict[str, Any]) -> None:
        sql = """
            INSERT INTO trades (
                ts, received_at, symbol, price, size, volume, bid, ask,
                raw_event_id, payload
            )
            VALUES (
                %(ts)s, %(received_at)s, %(symbol)s, %(price)s, %(size)s,
                %(volume)s, %(bid)s, %(ask)s, %(raw_event_id)s, %(payload)s
            )
        """
        payload = dict(row)
        payload["payload"] = Jsonb(payload["payload"])
        with self._connection().cursor() as cur:
            cur.execute(sql, payload)

    def insert_orderbook(self, row: dict[str, Any]) -> None:
        columns = [
            "ts",
            "received_at",
            "symbol",
            "bid1_price",
            "bid1_size",
            "bid2_price",
            "bid2_size",
            "bid3_price",
            "bid3_size",
            "bid4_price",
            "bid4_size",
            "bid5_price",
            "bid5_size",
            "ask1_price",
            "ask1_size",
            "ask2_price",
            "ask2_size",
            "ask3_price",
            "ask3_size",
            "ask4_price",
            "ask4_size",
            "ask5_price",
            "ask5_size",
            "payload",
        ]
        placeholders = ", ".join(f"%({column})s" for column in columns)
        sql = f"""
            INSERT INTO orderbook_5 ({", ".join(columns)})
            VALUES ({placeholders})
        """
        payload = dict(row)
        payload["payload"] = Jsonb(payload["payload"])
        with self._connection().cursor() as cur:
            cur.execute(sql, payload)
