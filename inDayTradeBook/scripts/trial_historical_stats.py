"""Trial Fubon historical stats request for 2330 52-week data."""

from __future__ import annotations

import argparse
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


def main() -> int:
    args = parse_args()
    config = RealtimeConfig.from_env()
    try:
        config.validate()
    except ValueError as exc:
        print("HISTORICAL_STATS_NOT_CONFIGURED")
        print(f"error={exc}")
        return 2

    try:
        from fubon_neo.fugle_marketdata.rest.base_rest import FugleAPIError
    except ImportError:
        FugleAPIError = Exception

    try:
        reststock = login_and_get_reststock(config)
    except Exception as exc:
        print("HISTORICAL_STATS_CONNECT_ERROR")
        print(f"error_type={type(exc).__name__}")
        print(f"error={exc}")
        return 1

    try:
        result = reststock.historical.stats(symbol=args.symbol)
    except FugleAPIError as exc:
        print("HISTORICAL_STATS_API_ERROR")
        print(f"error={exc}")
        status_code = getattr(exc, "status_code", None)
        response_text = getattr(exc, "response_text", None)
        if status_code is not None:
            print(f"status_code={status_code}")
        if response_text is not None:
            print(f"response_text={response_text}")
        return 1

    print("HISTORICAL_STATS_OK")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
