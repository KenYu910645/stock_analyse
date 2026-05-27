"""Logging setup for the realtime collector."""

from __future__ import annotations

import logging
import sys


class ContextDefaultsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "symbol"):
            record.symbol = "-"
        if not hasattr(record, "channel"):
            record.channel = "-"
        return True


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(ContextDefaultsFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s "
            "symbol=%(symbol)s channel=%(channel)s %(message)s"
        )
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )


def log_extra(symbol: str | None = None, channel: str | None = None) -> dict[str, str]:
    return {"symbol": symbol or "-", "channel": channel or "-"}
