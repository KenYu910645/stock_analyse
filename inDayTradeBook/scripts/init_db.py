"""Initialize TimescaleDB schema for realtime market data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import psycopg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime.config import RealtimeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-compression",
        action="store_true",
        help="Also apply compression settings and policies.",
    )
    return parser.parse_args()


def migration_paths(include_compression: bool) -> list[Path]:
    paths = sorted((PROJECT_ROOT / "sql").glob("*.sql"))
    if include_compression:
        return paths
    return [path for path in paths if "compression" not in path.name]


def main() -> None:
    args = parse_args()
    config = RealtimeConfig.from_env()

    with psycopg.connect(config.postgres_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for path in migration_paths(args.with_compression):
                print(f"Applying {path.name}")
                cur.execute(path.read_text(encoding="utf-8"))

    print("Database initialization complete.")


if __name__ == "__main__":
    main()
