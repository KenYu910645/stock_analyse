"""Fubon Neo realtime WebSocket collector."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from realtime.config import RealtimeConfig
from realtime.fubon_sdk import create_fubon_sdk
from realtime.logger import log_extra
from realtime.parser import (
    ensure_event_dict,
    parse_book_event,
    parse_common_fields,
    parse_trade_event,
    should_normalize_event,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from realtime.db import TimescaleWriter


class FubonRealtimeCollector:
    def __init__(self, config: RealtimeConfig, writer: "TimescaleWriter"):
        self.config = config
        self.writer = writer
        self.sdk: Any | None = None
        self.stock: Any | None = None
        self.running = False
        self.event_count = 0

    def login(self) -> None:
        self.sdk = create_fubon_sdk(self.config.fubon_ws_url)
        result = self.sdk.login(
            self.config.fubon_id,
            self.config.fubon_password,
            self.config.fubon_cert_path,
            self.config.fubon_cert_password_for_login,
        )
        logger.info("Fubon login success: %s", result, extra=log_extra())

    def init_realtime(self) -> None:
        if self.sdk is None:
            raise RuntimeError("login() must be called before init_realtime().")

        mode_name = self.config.fubon_mode.strip().lower()
        if mode_name in {"speed", "normal"}:
            try:
                from fubon_neo.sdk import Mode

                mode = Mode.Normal if mode_name == "normal" else Mode.Speed
                self.sdk.init_realtime(mode)
            except (ImportError, AttributeError):
                self.sdk.init_realtime()
        else:
            self.sdk.init_realtime()

        self.stock = self.sdk.marketdata.websocket_client.stock
        self.stock.on("message", self.on_message)
        self.stock.on("error", self.on_error)
        self.stock.on("disconnect", self.on_disconnect)
        self.stock.on("connect", self.on_connect)
        logger.info("Realtime client initialized", extra=log_extra())

    def connect(self) -> None:
        if self.stock is None:
            raise RuntimeError("init_realtime() must be called before connect().")
        self.stock.connect()

    def subscribe(self) -> None:
        if self.stock is None:
            raise RuntimeError("init_realtime() must be called before subscribe().")

        for channel in self.config.channels:
            self.stock.subscribe({"channel": channel, "symbols": self.config.symbols})
            logger.info(
                "Subscribed %s for %s",
                channel,
                ",".join(self.config.symbols),
                extra=log_extra(channel=channel),
            )

    def on_connect(self, *args: Any) -> None:
        logger.info("WebSocket connected: %s", args, extra=log_extra())

    def on_message(self, message: dict[str, Any] | str | bytes) -> None:
        received_at = datetime.now(timezone.utc)
        raw_id: int | None = None
        raw_row: dict[str, Any] | None = None

        try:
            event = ensure_event_dict(message)
            raw_row = parse_common_fields(event, received_at)
            raw_id = self.writer.insert_raw_event(raw_row)
            self.event_count += 1
        except Exception:
            logger.exception("Raw event insert failed", extra=log_extra())
            return

        symbol = raw_row.get("symbol") or None
        channel = raw_row.get("channel") or None

        try:
            if not should_normalize_event(event):
                logger.debug(
                    "Stored non-data event %s",
                    raw_row.get("event_type"),
                    extra=log_extra(symbol, channel),
                )
                return

            if channel == "trades":
                trade_row = parse_trade_event(event, received_at, raw_event_id=raw_id)
                if trade_row is not None:
                    self.writer.insert_trade(trade_row)
            elif channel == "books":
                book_row = parse_book_event(event, received_at)
                if book_row is not None:
                    self.writer.insert_orderbook(book_row)
        except Exception:
            logger.exception(
                "Parser or normalized insert failed",
                extra=log_extra(symbol, channel),
            )

    def on_error(self, error: Exception | str) -> None:
        logger.error("WebSocket error: %s", error, extra=log_extra())

    def on_disconnect(self, *args: Any) -> None:
        logger.warning("WebSocket disconnected: %s", args, extra=log_extra())
        if self.running:
            self.reconnect()

    def reconnect(self) -> None:
        time.sleep(self.config.reconnect_delay_seconds)
        try:
            self.connect()
            self.subscribe()
            logger.info("WebSocket reconnected and resubscribed", extra=log_extra())
        except Exception:
            logger.exception("Reconnect attempt failed", extra=log_extra())

    def run_forever(self) -> None:
        self.running = True
        logger.info("Collector running", extra=log_extra())
        try:
            while self.running:
                time.sleep(1)
        finally:
            self.close()

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        if self.stock is None:
            return
        for method_name in ("disconnect", "close"):
            method = getattr(self.stock, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    logger.exception("Failed to close websocket", extra=log_extra())
                break
