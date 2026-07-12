"""Scan and repair suspicious per-stock price rows against TWSE MI_INDEX.

The scan is intentionally conservative: it proposes rows with extreme
non-event jumps, inconsistent daily change references, or repeated full OHLCV
tuples, then only repairs a row after TWSE confirms a different official row
for the same code and date.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical, to_csv_storage
from downloader import price
from tools import apply_forward_adjustments_to_price as adjustments

PRICE_DIR = PROJECT_ROOT / "data" / "price"
DIVIDEND_DIR = PROJECT_ROOT / "data" / "dividend" / "ex_right_dividend"
LOG_DIR = PROJECT_ROOT / "logs" / "price_health"

RAW_COLUMNS = [
    "Capacity",
    "Turnover",
    "Open",
    "High",
    "Low",
    "Close",
    "Change",
    "Transaction",
]
OUTPUT_COLUMNS = price.PRICE_COLUMNS + adjustments.ADJUSTED_COLUMNS


def code_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    for column in RAW_COLUMNS + adjustments.ADJUSTED_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def ex_reference_by_date(code: str) -> dict[pd.Timestamp, float]:
    matches = sorted(DIVIDEND_DIR.glob(f"{code}_*.csv"))
    if not matches:
        return {}
    try:
        events = read_csv_canonical(
            matches[0],
            dtype={"stock_id": str},
            usecols=["ex_date", "ex_reference_price"],
        )
    except Exception:
        return {}
    events["ex_date"] = pd.to_datetime(events["ex_date"], errors="coerce")
    events["ex_reference_price"] = pd.to_numeric(events["ex_reference_price"], errors="coerce")
    events = events.dropna(subset=["ex_date", "ex_reference_price"])
    return {
        pd.Timestamp(row.ex_date).normalize(): float(row.ex_reference_price)
        for row in events.itertuples(index=False)
        if float(row.ex_reference_price) > 0
    }


def finite(value: Any) -> bool:
    try:
        return value is not None and not pd.isna(value) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def same_number(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    if not finite(left) and not finite(right):
        return True
    if not finite(left) or not finite(right):
        return False
    return abs(float(left) - float(right)) <= tolerance


def local_row_matches_official(local_row: pd.Series, official_row: dict[str, Any]) -> bool:
    return all(same_number(local_row[column], official_row[column]) for column in RAW_COLUMNS)


def raw_tuple_key(row: pd.Series) -> tuple[Any, ...]:
    key = []
    for column in RAW_COLUMNS:
        value = row[column]
        key.append(round(float(value), 6) if finite(value) else None)
    return tuple(key)


def detect_candidates(path: Path, jump_threshold: float, duplicate_min_count: int) -> list[dict[str, Any]]:
    code = code_from_path(path)
    if not code or not code[0].isdigit():
        return []
    try:
        df = numeric_frame(read_csv_canonical(path))
    except Exception as exc:
        return [{"code": code, "path": str(path), "error": f"read_failed: {exc}"}]
    if df.empty or not set(price.PRICE_COLUMNS).issubset(df.columns):
        return []

    ex_refs = ex_reference_by_date(code)
    df["previous_close"] = df["Close"].shift(1)
    df["next_close"] = df["Close"].shift(-1)
    df["pct_prev"] = (df["Close"] / df["previous_close"] - 1).replace([math.inf, -math.inf], pd.NA)
    df["pct_next_from_this"] = (df["next_close"] / df["Close"] - 1).replace([math.inf, -math.inf], pd.NA)
    df["change_reference"] = df["Close"] - df["Change"]
    df["reference_gap"] = (df["change_reference"] / df["previous_close"] - 1).replace([math.inf, -math.inf], pd.NA)
    df["_raw_key"] = df.apply(raw_tuple_key, axis=1)
    duplicate_counts = df["_raw_key"].value_counts(dropna=False).to_dict()

    candidates: dict[str, dict[str, Any]] = {}
    for index, row in df.iterrows():
        date_value = pd.Timestamp(row["Date"]).normalize()
        date_text = date_value.strftime("%Y-%m-%d")
        ex_ref = ex_refs.get(date_value)
        event_pct = None
        if ex_ref and finite(row["Close"]):
            event_pct = float(row["Close"]) / ex_ref - 1

        reasons = []
        pct_prev = row.get("pct_prev")
        if finite(pct_prev) and abs(float(pct_prev)) > jump_threshold:
            if event_pct is None or abs(event_pct) > jump_threshold:
                reasons.append(f"close_vs_previous={float(pct_prev):.4f}")

        reference_gap = row.get("reference_gap")
        if finite(reference_gap) and abs(float(reference_gap)) > jump_threshold:
            if event_pct is None or abs(event_pct) > jump_threshold:
                reasons.append(f"change_reference_gap={float(reference_gap):.4f}")

        duplicate_count = int(duplicate_counts.get(row["_raw_key"], 0))
        next_jump = row.get("pct_next_from_this")
        has_neighbor_jump = (
            (finite(pct_prev) and abs(float(pct_prev)) > 0.10)
            or (finite(next_jump) and abs(float(next_jump)) > 0.10)
        )
        duplicate_is_suspicious = (
            (duplicate_count >= duplicate_min_count and has_neighbor_jump)
            or (duplicate_count >= 5 and finite(row["Capacity"]) and float(row["Capacity"]) > 0)
        )
        if duplicate_is_suspicious:
            reasons.append(f"duplicate_full_ohlcv={duplicate_count}")

        if reasons:
            candidates[date_text] = {
                "code": code,
                "path": str(path),
                "date": date_text,
                "reasons": sorted(set(reasons)),
                "local": {column: None if pd.isna(row[column]) else float(row[column]) for column in RAW_COLUMNS},
            }

    return list(candidates.values())


def fetch_one_date(date_text: str, codes: set[str]) -> tuple[str, dict[str, dict[str, Any]], str | None]:
    query_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    try:
        with requests.Session() as session:
            rows = price.fetch_mi_stock_day(session, query_date, codes)
        return date_text, rows, None
    except Exception as exc:
        return date_text, {}, str(exc)


def fetch_official_rows(
    candidates: list[dict[str, Any]],
    pause_seconds: float,
    workers: int,
) -> dict[tuple[str, str], dict[str, Any] | None]:
    codes_by_date: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        if "date" in candidate and "code" in candidate:
            codes_by_date[candidate["date"]].add(candidate["code"])

    official: dict[tuple[str, str], dict[str, Any] | None] = {}
    items = sorted(codes_by_date.items())
    if workers <= 1:
        for index, (date_text, codes) in enumerate(items, start=1):
            fetched_date, rows, error = fetch_one_date(date_text, codes)
            if error:
                for code in codes:
                    official[(code, fetched_date)] = None
                print(f"[{index}/{len(items)}] {fetched_date}: fetch failed: {error}")
            else:
                for code in codes:
                    official[(code, fetched_date)] = rows.get(code)
                print(f"[{index}/{len(items)}] {fetched_date}: checked {len(codes)} codes")
            if pause_seconds:
                time.sleep(pause_seconds)
        return official

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_one_date, date_text, codes): (date_text, codes)
            for date_text, codes in items
        }
        for index, future in enumerate(as_completed(futures), start=1):
            date_text, codes = futures[future]
            fetched_date, rows, error = future.result()
            if error:
                for code in codes:
                    official[(code, fetched_date)] = None
                print(f"[{index}/{len(items)}] {fetched_date}: fetch failed: {error}")
            else:
                for code in codes:
                    official[(code, fetched_date)] = rows.get(code)
                print(f"[{index}/{len(items)}] {fetched_date}: checked {len(codes)} codes")
    return official


def fetch_stock_day_row(
    session: requests.Session,
    code: str,
    date_text: str,
    month_cache: dict[tuple[str, int, int], dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    query_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    if query_date < price.TWSE_STOCK_DAY_MIN_DATE:
        return None

    cache_key = (code, query_date.year, query_date.month)
    if cache_key not in month_cache:
        rows = price.fetch_twse_stock_month(session, code, query_date.year, query_date.month)
        month_cache[cache_key] = {row["Date"]: row for row in rows}
    return month_cache[cache_key].get(date_text)


def fill_stock_day_fallback(
    candidates: list[dict[str, Any]],
    official_rows: dict[tuple[str, str], dict[str, Any] | None],
    pause_seconds: float,
) -> None:
    missing = [
        candidate
        for candidate in candidates
        if official_rows.get((candidate["code"], candidate["date"])) is None
        and datetime.strptime(candidate["date"], "%Y-%m-%d").date() >= price.TWSE_STOCK_DAY_MIN_DATE
    ]
    if not missing:
        return

    month_cache: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    with requests.Session() as session:
        for index, candidate in enumerate(sorted(missing, key=lambda item: (item["code"], item["date"])), start=1):
            key = (candidate["code"], candidate["date"])
            try:
                row = fetch_stock_day_row(session, candidate["code"], candidate["date"], month_cache)
            except Exception as exc:
                candidate.setdefault("fetch_errors", []).append(f"STOCK_DAY: {exc}")
                row = None
            if row is not None:
                official_rows[key] = row
            if index % 100 == 0 or index == len(missing):
                print(f"STOCK_DAY fallback {index}/{len(missing)}; month_requests={len(month_cache)}")
            if pause_seconds and index < len(missing):
                time.sleep(pause_seconds)


def apply_official_repairs(
    candidates: list[dict[str, Any]],
    official_rows: dict[tuple[str, str], dict[str, Any] | None],
    apply: bool,
) -> tuple[list[dict[str, Any]], set[Path]]:
    repairs = []
    touched_paths: set[Path] = set()
    candidates_by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if "path" in candidate:
            candidates_by_path[Path(candidate["path"])].append(candidate)

    metadata = adjustments.load_metadata() if apply else pd.DataFrame()
    previous_map = adjustments.load_previous_trading_day_map() if apply else {}

    for path, path_candidates in sorted(candidates_by_path.items()):
        df = numeric_frame(read_csv_canonical(path))
        changed = False
        for candidate in path_candidates:
            official = official_rows.get((candidate["code"], candidate["date"]))
            if not official:
                candidate["status"] = "no_official_row"
                continue
            mask = df["Date"].dt.strftime("%Y-%m-%d").eq(candidate["date"])
            if not mask.any():
                candidate["status"] = "missing_local_row"
                continue
            idx = df.index[mask][0]
            if local_row_matches_official(df.loc[idx], official):
                candidate["status"] = "matches_official"
                continue

            before = {column: None if pd.isna(df.at[idx, column]) else float(df.at[idx, column]) for column in RAW_COLUMNS}
            after = {column: official[column] for column in RAW_COLUMNS}
            candidate["status"] = "repaired" if apply else "would_repair"
            candidate["official"] = after
            candidate["before"] = before
            repairs.append(candidate)
            if apply:
                for column in RAW_COLUMNS:
                    df.at[idx, column] = official[column]
                changed = True

        if apply and changed:
            code = code_from_path(path)
            instrument_type = ""
            if code in metadata.index and "Type" in metadata.columns:
                instrument_type = str(metadata.at[code, "Type"])
            events = adjustments.load_adjustment_events(code, metadata)
            adjusted, _merged, _inferred = adjustments.add_adjusted_columns(
                df,
                events,
                previous_map,
                allow_price_inferred_events=instrument_type.upper() == "ETF",
            )
            to_csv_storage(adjusted, path, index=False, encoding="utf-8-sig")
            touched_paths.add(path)

    return repairs, touched_paths


def write_report(report: dict[str, Any]) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"price_health_repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair suspicious price rows against TWSE MI_INDEX.")
    parser.add_argument("--apply", action="store_true", help="Write confirmed official-row repairs.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional first-N price files for testing.")
    parser.add_argument("--codes", default="", help="Comma-separated stock codes to scan.")
    parser.add_argument("--jump-threshold", type=float, default=0.30)
    parser.add_argument("--duplicate-min-count", type=int, default=3)
    parser.add_argument("--pause", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent TWSE date fetch workers.")
    parser.add_argument("--scan-only", action="store_true", help="Only scan local files and write candidate report.")
    parser.add_argument(
        "--universe",
        choices=["listed-common", "all"],
        default="listed-common",
        help="Scan TWSE listed common stocks by default; use all for every price CSV with a numeric code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_codes = {item.strip() for item in args.codes.split(",") if item.strip()}
    paths = sorted(path for path in PRICE_DIR.glob("*.csv") if not path.name.startswith("twse_price_"))
    if args.universe == "listed-common":
        metadata = adjustments.load_metadata()
        if {"Type", "Market"}.issubset(metadata.columns):
            eligible_codes = set(
                metadata[
                    metadata["Type"].astype(str).isin([price.COMMON_STOCK_TYPE, "STOCK"])
                    & metadata["Market"].astype(str).eq(price.TWSE_MARKET)
                ].index.astype(str)
            )
            paths = [path for path in paths if code_from_path(path) in eligible_codes]
    if requested_codes:
        paths = [path for path in paths if code_from_path(path) in requested_codes]
    if args.limit_files is not None:
        paths = paths[: args.limit_files]

    all_candidates = []
    read_errors = []
    for index, path in enumerate(paths, start=1):
        candidates = detect_candidates(path, args.jump_threshold, args.duplicate_min_count)
        for candidate in candidates:
            if "error" in candidate:
                read_errors.append(candidate)
            else:
                all_candidates.append(candidate)
        if index % 100 == 0:
            print(f"scanned {index}/{len(paths)} files; candidates={len(all_candidates)}")

    print(f"candidate_rows={len(all_candidates)} read_errors={len(read_errors)}")
    if args.scan_only:
        official_rows = {}
        repairs = []
        touched_paths = set()
        for candidate in all_candidates:
            candidate["status"] = "candidate_not_verified"
    else:
        official_rows = fetch_official_rows(all_candidates, args.pause, args.workers) if all_candidates else {}
        fill_stock_day_fallback(all_candidates, official_rows, args.pause)
        repairs, touched_paths = apply_official_repairs(all_candidates, official_rows, args.apply)

    statuses = defaultdict(int)
    for candidate in all_candidates:
        statuses[candidate.get("status", "unchecked")] += 1
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "apply": args.apply,
        "scan_only": args.scan_only,
        "scanned_files": len(paths),
        "candidate_rows": len(all_candidates),
        "read_errors": read_errors,
        "status_counts": dict(sorted(statuses.items())),
        "repair_count": len(repairs),
        "touched_paths": [str(path) for path in sorted(touched_paths)],
        "candidates": all_candidates if args.scan_only else [],
        "repairs": repairs,
    }
    report_path = write_report(report)
    print(json.dumps({k: report[k] for k in ["scanned_files", "candidate_rows", "status_counts", "repair_count"]}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
