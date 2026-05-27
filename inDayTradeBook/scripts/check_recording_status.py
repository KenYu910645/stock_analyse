"""Print recorder status and current DB event counts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

FEATURE_ROOT = Path(__file__).resolve().parents[1]
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from realtime.config import RealtimeConfig


def latest_wrapper_log() -> Path | None:
    logs = sorted((FEATURE_ROOT / "log").glob("realtime_collector_*.wrapper.log"))
    return logs[-1] if logs else None


def task_status() -> str:
    try:
        result = subprocess.run(
            ["schtasks", "/Query", "/TN", "InDayTradeBookRealtime", "/FO", "LIST"],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        return f"schtasks unavailable: {exc}"

    if result.returncode != 0:
        return "task_not_found"

    stdout = result.stdout.decode("utf-8", errors="replace")
    for line in stdout.splitlines():
        if line.startswith("Status:") or line.startswith("Next Run Time:"):
            print(line)
    return "task_found"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"task={task_status()}")

    wrapper = latest_wrapper_log()
    if wrapper is not None:
        print(f"latest_wrapper_log={wrapper}")
        print("--- wrapper tail ---")
        print("\n".join(wrapper.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]))

    config = RealtimeConfig.from_env()
    with psycopg.connect(config.postgres_dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, channel, count(*) AS events
                FROM raw_market_events
                WHERE received_at >= date_trunc('day', now() AT TIME ZONE 'Asia/Taipei')
                                     AT TIME ZONE 'Asia/Taipei'
                GROUP BY symbol, channel
                ORDER BY symbol, channel
                """
            )
            print("--- today raw event counts ---")
            for row in cur.fetchall():
                print(dict(row))

            cur.execute(
                """
                SELECT max(received_at AT TIME ZONE 'Asia/Taipei') AS latest_event_tpe
                FROM raw_market_events
                """
            )
            print(dict(cur.fetchone()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
