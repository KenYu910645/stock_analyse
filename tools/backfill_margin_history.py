"""Backfill TWSE margin trading history into per-stock CSV files."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical, to_csv_storage  # noqa: E402
from downloader import margin_trading  # noqa: E402
from downloader import update_all_data  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data"
MARGIN_DIR = DATA_DIR / "margin"
METADATA_PATH = DATA_DIR / "metadata.csv"
TRADING_DAYS_PATH = DATA_DIR / "trading_days.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "margin_backfill"
DEFAULT_START_DATE = "2001-01-01"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backfill TWSE margin trading rows into per-stock CSVs."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--skip-feature-refresh", action="store_true")
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_listed_common_metadata() -> pd.DataFrame:
    metadata = read_csv_canonical(METADATA_PATH, dtype={"Code": str}).fillna("")
    metadata["Code"] = metadata["Code"].astype(str).str.strip()
    listed = metadata[
        metadata["Code"].str.match(r"^[0-9]{4}$", na=False)
        & ~metadata["Type"].isin(["ETF", "INDEX"])
        & ~metadata["Market"].isin(["INDEX"])
    ].copy()
    listed["Start"] = pd.to_datetime(listed["Start"], errors="coerce")
    listed = listed.drop_duplicates("Code").sort_values("Code")
    return listed


def load_trading_dates(start_date: date, end_date: date) -> list[date]:
    trading_days = pd.read_csv(TRADING_DAYS_PATH, dtype=str)
    column = "date" if "date" in trading_days.columns else trading_days.columns[0]
    dates = pd.to_datetime(trading_days[column], errors="coerce").dropna().dt.date
    return sorted(value for value in dates if start_date <= value <= end_date)


def margin_path(code: str, code_to_name: dict[str, str]) -> Path:
    return Path(
        update_all_data.stock_keyed_output_path(
            str(MARGIN_DIR),
            code,
            code_to_name.get(code, ""),
        )
    )


def existing_dates_for_path(path: Path) -> set[date]:
    if not path.exists():
        return set()
    try:
        df = read_csv_canonical(path, dtype=str, usecols=["Date"]).fillna("")
    except Exception:
        return set()
    return set(pd.to_datetime(df["Date"], errors="coerce").dropna().dt.date)


def compute_missing_dates(
    listed: pd.DataFrame,
    trading_dates: list[date],
    start_date: date,
) -> tuple[list[date], dict[str, Path]]:
    code_to_name = dict(zip(listed["Code"], listed["Name"]))
    code_to_path = {code: margin_path(code, code_to_name) for code in code_to_name}
    missing_dates: set[date] = set()

    for row in listed.itertuples(index=False):
        code = str(row.Code)
        metadata_start = row.Start.date() if not pd.isna(row.Start) else start_date
        expected_start = max(start_date, metadata_start)
        existing_dates = existing_dates_for_path(code_to_path[code])
        for trading_date in trading_dates:
            if trading_date >= expected_start and trading_date not in existing_dates:
                missing_dates.add(trading_date)

    return sorted(missing_dates), code_to_path


def fetch_one(query_date: date):
    session = requests.Session()
    payload = margin_trading.fetch_payload(session, query_date)
    rows = margin_trading.parse_payload_rows(payload, query_date)
    return query_date, rows


def fetch_batch(dates: list[date], workers: int):
    if workers <= 1:
        for query_date in dates:
            try:
                yield query_date, margin_trading.parse_payload_rows(
                    margin_trading.fetch_payload(requests.Session(), query_date),
                    query_date,
                ), ""
            except Exception as exc:
                yield query_date, [], str(exc)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_one, query_date): query_date for query_date in dates}
        for future in as_completed(futures):
            query_date = futures[future]
            try:
                fetched_date, rows = future.result()
                yield fetched_date, rows, ""
            except Exception as exc:
                yield query_date, [], str(exc)


def append_log(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def merge_rows(path: Path, rows: list[dict]) -> int:
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return 0
    if path.exists():
        existing = read_csv_canonical(path, dtype=str).fillna("")
        merged = pd.concat([existing, incoming], ignore_index=True)
    else:
        merged = incoming

    for column in margin_trading.OUTPUT_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
    extra_columns = [column for column in merged.columns if column not in margin_trading.OUTPUT_COLUMNS]
    merged = merged[margin_trading.OUTPUT_COLUMNS + extra_columns]
    before = len(merged)
    merged = (
        merged.drop_duplicates(subset=["Date", "Code"], keep="last")
        .sort_values(["Date", "Code"])
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    to_csv_storage(merged, path, index=False, encoding="utf-8-sig")
    return before - len(merged) + len(incoming)


def write_batch(rows_by_code: dict[str, list[dict]], code_to_path: dict[str, Path]) -> int:
    written = 0
    for code, rows in sorted(rows_by_code.items()):
        path = code_to_path.get(code)
        if path is None:
            continue
        written += merge_rows(path, rows)
    return written


def main():
    args = parse_args()
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    listed = load_listed_common_metadata()
    code_to_start = {
        str(row.Code): max(start_date, row.Start.date() if not pd.isna(row.Start) else start_date)
        for row in listed.itertuples(index=False)
    }
    trading_dates = load_trading_dates(start_date, end_date)
    missing_dates, code_to_path = compute_missing_dates(listed, trading_dates, start_date)
    if args.max_dates:
        missing_dates = missing_dates[: args.max_dates]

    print(
        f"listed={len(listed)} trading_dates={len(trading_dates)} "
        f"missing_dates={len(missing_dates)} range="
        f"{missing_dates[0] if missing_dates else ''}..{missing_dates[-1] if missing_dates else ''}",
        flush=True,
    )
    if not missing_dates:
        if not args.skip_feature_refresh:
            update_all_data.refresh_margin_features()
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    status_log = LOG_DIR / "backfill_status.csv"
    all_codes = set(code_to_path)
    total_rows = 0
    total_written = 0
    started_at = time.time()

    for batch_index in range(0, len(missing_dates), args.batch_size):
        batch_dates = missing_dates[batch_index : batch_index + args.batch_size]
        rows_by_code: dict[str, list[dict]] = defaultdict(list)
        log_rows = []
        for query_date, rows, error in fetch_batch(batch_dates, args.workers):
            if error:
                log_rows.append({
                    "Date": query_date.isoformat(),
                    "Status": "failed",
                    "Rows": 0,
                    "Error": error,
                    "FetchedAt": datetime.now().isoformat(timespec="seconds"),
                })
                print(f"failed {query_date}: {error}", flush=True)
                continue
            kept = 0
            for row in rows:
                code = str(row.get("Code", "")).strip()
                if code not in all_codes or query_date < code_to_start.get(code, start_date):
                    continue
                rows_by_code[code].append(row)
                kept += 1
            total_rows += kept
            log_rows.append({
                "Date": query_date.isoformat(),
                "Status": "ok",
                "Rows": kept,
                "Error": "",
                "FetchedAt": datetime.now().isoformat(timespec="seconds"),
            })

        written = write_batch(rows_by_code, code_to_path)
        total_written += written
        append_log(status_log, log_rows)
        done = min(batch_index + len(batch_dates), len(missing_dates))
        elapsed = time.time() - started_at
        print(
            f"batch={done}/{len(missing_dates)} dates "
            f"kept_rows={total_rows} written_rows={total_written} "
            f"elapsed_seconds={elapsed:.0f}",
            flush=True,
        )

    if not args.skip_feature_refresh:
        update_all_data.refresh_margin_features()


if __name__ == "__main__":
    main()
