"""Run sample realtime market-data validation queries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.config import RealtimeConfig


QUERIES = {
    "latest-trade": """
        SELECT *
        FROM trades
        WHERE symbol = %(symbol)s
        ORDER BY ts DESC
        LIMIT 20
    """,
    "latest-orderbook": """
        SELECT *
        FROM orderbook_5
        WHERE symbol = %(symbol)s
        ORDER BY ts DESC
        LIMIT 1
    """,
    "spread": """
        SELECT ts, ask1_price - bid1_price AS spread
        FROM orderbook_5
        WHERE symbol = %(symbol)s
          AND ts >= now() - INTERVAL '1 day'
        ORDER BY ts
    """,
    "imbalance": """
        SELECT
            ts,
            symbol,
            (
                COALESCE(bid1_size, 0) + COALESCE(bid2_size, 0) +
                COALESCE(bid3_size, 0) + COALESCE(bid4_size, 0) +
                COALESCE(bid5_size, 0) - COALESCE(ask1_size, 0) -
                COALESCE(ask2_size, 0) - COALESCE(ask3_size, 0) -
                COALESCE(ask4_size, 0) - COALESCE(ask5_size, 0)
            )::DOUBLE PRECISION
            /
            NULLIF(
                (
                    COALESCE(bid1_size, 0) + COALESCE(bid2_size, 0) +
                    COALESCE(bid3_size, 0) + COALESCE(bid4_size, 0) +
                    COALESCE(bid5_size, 0) + COALESCE(ask1_size, 0) +
                    COALESCE(ask2_size, 0) + COALESCE(ask3_size, 0) +
                    COALESCE(ask4_size, 0) + COALESCE(ask5_size, 0)
                ),
                0
            ) AS imbalance
        FROM orderbook_5
        WHERE symbol = %(symbol)s
        ORDER BY ts DESC
        LIMIT 100
    """,
    "event-counts": """
        SELECT symbol, channel, count(*) AS events
        FROM raw_market_events
        GROUP BY symbol, channel
        ORDER BY symbol, channel
    """,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="2330")
    parser.add_argument(
        "query",
        choices=sorted(QUERIES),
        nargs="?",
        default="latest-trade",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RealtimeConfig.from_env()

    with psycopg.connect(config.postgres_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(QUERIES[args.query], {"symbol": args.symbol})
            for row in cur.fetchall():
                print(dict(row))


if __name__ == "__main__":
    main()
