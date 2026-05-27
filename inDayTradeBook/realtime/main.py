"""Command-line entry point for the realtime collector."""

from __future__ import annotations

import logging
import signal

from realtime.config import RealtimeConfig
from realtime.db import TimescaleWriter
from realtime.fubon_client import FubonRealtimeCollector
from realtime.logger import setup_logging


def main() -> None:
    setup_logging()
    config = RealtimeConfig.from_env()
    config.validate()

    writer = TimescaleWriter(config.postgres_dsn)
    writer.connect()

    collector = FubonRealtimeCollector(config=config, writer=writer)

    def stop(_signum: int, _frame: object) -> None:
        logging.getLogger(__name__).info(
            "Shutdown requested",
            extra={"symbol": "-", "channel": "-"},
        )
        collector.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        collector.login()
        collector.init_realtime()
        collector.connect()
        collector.subscribe()
        collector.run_forever()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
