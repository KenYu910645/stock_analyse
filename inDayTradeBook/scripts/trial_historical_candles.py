"""Trial Fubon historical candle request for 2330 recent 1-minute K data.

Fubon documents that minute K requests ignore from/to and return the recent
five days, so this script intentionally requests timeframe=1 without dates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

FEATURE_ROOT = Path(__file__).resolve().parents[1]
if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from realtime.config import RealtimeConfig
from realtime.fubon_sdk import create_fubon_sdk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="2330")
    parser.add_argument("--timeframe", default="1", help="1 means 1-minute K")
    parser.add_argument("--sort", default="asc", choices=["asc", "desc"])
    parser.add_argument(
        "--fields",
        default="open,high,low,close,volume",
        help="Minute K does not support turnover/change per Fubon docs.",
    )
    parser.add_argument(
        "--limit-print",
        type=int,
        default=5,
        help="Number of head/tail rows to print.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURE_ROOT / "data" / "historical" / "candles_2330_1m.csv",
        help="CSV output path for the returned candle rows.",
    )
    return parser.parse_args()


def login_and_get_reststock(config: RealtimeConfig) -> Any:
    sdk = create_fubon_sdk(config.fubon_ws_url)
    sdk.login(
        config.fubon_id,
        config.fubon_password,
        config.fubon_cert_path,
        config.fubon_cert_password_for_login,
    )
    sdk.init_realtime()
    return sdk.marketdata.rest_client.stock


def print_result(result: dict[str, Any], limit_print: int) -> None:
    data = result.get("data") or []
    print("HISTORICAL_CANDLES_OK")
    print(f"symbol={result.get('symbol')}")
    print(f"type={result.get('type')}")
    print(f"exchange={result.get('exchange')}")
    print(f"market={result.get('market')}")
    print(f"timeframe={result.get('timeframe')}")
    print(f"rows={len(data)}")

    if data:
        print(f"first={json.dumps(data[0], ensure_ascii=False, default=str)}")
        print(f"last={json.dumps(data[-1], ensure_ascii=False, default=str)}")

    if limit_print > 0:
        print("head:")
        for row in data[:limit_print]:
            print(json.dumps(row, ensure_ascii=False, default=str))
        if len(data) > limit_print:
            print("tail:")
            for row in data[-limit_print:]:
                print(json.dumps(row, ensure_ascii=False, default=str))


def save_csv(result: dict[str, Any], output_path: Path) -> None:
    rows = result.get("data") or []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["date", "open", "high", "low", "close", "volume"]

    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})

    print(f"saved_csv={output_path}")


def main() -> int:
    args = parse_args()
    config = RealtimeConfig.from_env()
    try:
        config.validate()
    except ValueError as exc:
        print("HISTORICAL_CANDLES_NOT_CONFIGURED")
        print(f"error={exc}")
        return 2

    try:
        from fubon_neo.fugle_marketdata.rest.base_rest import FugleAPIError
    except ImportError:
        FugleAPIError = Exception

    try:
        reststock = login_and_get_reststock(config)
    except Exception as exc:
        print("HISTORICAL_CANDLES_CONNECT_ERROR")
        print(f"error_type={type(exc).__name__}")
        print(f"error={exc}")
        return 1

    params = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "fields": args.fields,
        "sort": args.sort,
    }

    try:
        result = reststock.historical.candles(**params)
    except FugleAPIError as exc:
        print("HISTORICAL_CANDLES_API_ERROR")
        print(f"error={exc}")
        status_code = getattr(exc, "status_code", None)
        response_text = getattr(exc, "response_text", None)
        if status_code is not None:
            print(f"status_code={status_code}")
        if response_text is not None:
            print(f"response_text={response_text}")
        return 1

    save_csv(result, args.output)
    print_result(result, args.limit_print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
