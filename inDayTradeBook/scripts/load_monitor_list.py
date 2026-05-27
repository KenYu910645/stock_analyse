"""Load and validate stock symbols from monitor_list.txt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FEATURE_ROOT = Path(__file__).resolve().parents[1]
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from realtime.monitor_list import load_monitor_symbols, validate_subscription_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=FEATURE_ROOT / "monitor_list.txt",
        help="Monitor list path.",
    )
    parser.add_argument(
        "--channels",
        default="trades,books",
        help="Comma-separated channels used for subscription-count validation.",
    )
    parser.add_argument(
        "--max-subscriptions",
        type=int,
        default=200,
        help="Maximum subscriptions for one WebSocket connection.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "summary"],
        default="summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    channels = [channel.strip() for channel in args.channels.split(",") if channel.strip()]
    symbols = load_monitor_symbols(args.path)
    validate_subscription_limit(symbols, channels, args.max_subscriptions)

    if args.format == "csv":
        print(",".join(symbols))
    elif args.format == "json":
        print(json.dumps({"symbols": symbols, "channels": channels}, indent=2))
    else:
        print(f"monitor_path={args.path}")
        print(f"symbols={len(symbols)}")
        print(f"channels={','.join(channels)}")
        print(f"subscriptions={len(symbols) * len(channels)}")
        print(",".join(symbols))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
