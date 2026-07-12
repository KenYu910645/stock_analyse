"""Repair missing TWSE institutional-investor rows in per-stock CSVs.

The T86 source does not list every stock every day, so this tool first finds
date-level coverage anomalies, then confirms missing rows against TWSE before
writing anything.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical, to_csv_storage
from downloader import institutional_investors


DATA_DIR = PROJECT_ROOT / "data"
INSTITUTIONAL_DIR = DATA_DIR / "institutional"
METADATA_PATH = DATA_DIR / "metadata.csv"
TRADING_DAYS_PATH = DATA_DIR / "trading_days.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "institutional"
SOURCE_START = date(2012, 5, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair data/institutional gaps after TWSE T86 confirmation."
    )
    parser.add_argument(
        "--start-date",
        default=SOURCE_START.isoformat(),
        help="Earliest trading date to inspect. Default: 2012-05-02.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Latest trading date to inspect. Default: latest local institutional date.",
    )
    parser.add_argument(
        "--drop-threshold",
        type=int,
        default=80,
        help="Minimum local row-count drop versus nearby median for TWSE confirmation.",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        default=0.93,
        help="Maximum local/nearby median ratio for local-drop candidates.",
    )
    parser.add_argument(
        "--neighbor-window",
        type=int,
        default=5,
        help="Trading days on each side used for nearby median comparison.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.3,
        help="Polite pause between TWSE requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and report official gaps without writing CSV repairs.",
    )
    return parser.parse_args()


def parse_iso(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_listed_common_codes() -> set[str]:
    metadata = pd.read_csv(METADATA_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    if "CFI" in metadata.columns:
        return set(metadata.loc[metadata["CFI"].eq("ESVUFR"), "Code"].astype(str))

    canonical = read_csv_canonical(METADATA_PATH, dtype={"Code": str}).fillna("")
    mask = (
        canonical["Type"].eq(institutional_investors.COMMON_STOCK_TYPE)
        & canonical["Market"].eq(institutional_investors.TWSE_MARKET)
        & canonical["Code"].astype(str).str.match(r"^\d{4}$")
    )
    return set(canonical.loc[mask, "Code"].astype(str))


def load_trading_days(start: date, end: date) -> list[date]:
    trading = pd.read_csv(TRADING_DAYS_PATH, dtype=str, encoding="utf-8-sig")
    date_column = "date" if "date" in trading.columns else "Date"
    dates = pd.to_datetime(trading[date_column], errors="coerce").dropna().dt.date
    return [value for value in dates if start <= value <= end]


def institutional_paths() -> dict[str, Path]:
    return {
        path.name.split("_", 1)[0]: path
        for path in sorted(INSTITUTIONAL_DIR.glob("*.csv"))
    }


def local_date_codes(paths: dict[str, Path]) -> dict[date, set[str]]:
    by_date: dict[date, set[str]] = {}
    for code, path in paths.items():
        df = read_csv_canonical(path, dtype={"Code": str})
        dates = pd.to_datetime(df["Date"], errors="coerce").dt.date
        for value in dates.dropna():
            by_date.setdefault(value, set()).add(code)
    return by_date


def find_candidate_dates(
    trading_days: list[date],
    local_counts: Counter[date],
    drop_threshold: int,
    drop_ratio: float,
    neighbor_window: int,
) -> list[tuple[date, str, int, float | None]]:
    candidates: list[tuple[date, str, int, float | None]] = []
    for index, trading_date in enumerate(trading_days):
        count = local_counts[trading_date]
        if count == 0:
            candidates.append((trading_date, "whole_missing", count, None))
            continue

        neighbors = [
            local_counts[trading_days[neighbor_index]]
            for neighbor_index in range(
                max(0, index - neighbor_window),
                min(len(trading_days), index + neighbor_window + 1),
            )
            if neighbor_index != index and local_counts[trading_days[neighbor_index]] > 0
        ]
        if len(neighbors) < 4:
            continue
        median_count = float(statistics.median(neighbors))
        if median_count - count >= drop_threshold and count / median_count < drop_ratio:
            candidates.append((trading_date, "local_drop", count, median_count))
    return candidates


def fetch_official_rows(trading_date: date, listed_codes: set[str]) -> pd.DataFrame:
    payload = institutional_investors.get_json_response(trading_date)
    rows = institutional_investors.parse_twse_rows(payload, trading_date, listed_codes)
    if not rows:
        return pd.DataFrame(columns=institutional_investors.OUTPUT_COLUMNS)
    return institutional_investors.normalize_dataframe(rows)


def write_repaired_files(
    rows_by_code: dict[str, list[dict[str, object]]],
    paths_by_code: dict[str, Path],
) -> int:
    touched = 0
    for code, rows in sorted(rows_by_code.items()):
        if not rows:
            continue
        path = paths_by_code.get(code)
        if path is None:
            sample = rows[-1]
            name = str(sample.get("Name", "")).strip()
            safe_name = "".join("_" if char in '<>:"/\\|?*' or char.isspace() else char for char in name).strip("._")
            path = INSTITUTIONAL_DIR / (f"{code}_{safe_name}.csv" if safe_name else f"{code}.csv")

        existing = read_csv_canonical(path, dtype={"Code": str}) if path.exists() else pd.DataFrame()
        incoming = pd.DataFrame(rows, columns=institutional_investors.OUTPUT_COLUMNS)
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = institutional_investors.normalize_dataframe(combined.to_dict("records"))
        to_csv_storage(
            combined,
            path,
            index=False,
            encoding="utf-8-sig",
        )
        touched += 1
    return touched


def write_report(rows: list[dict[str, object]], run_id: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"institutional_gap_repair_{run_id}.csv"
    fieldnames = [
        "Date",
        "Reason",
        "LocalRowsBefore",
        "NearbyMedianRows",
        "OfficialRows",
        "MissingOfficialRows",
        "LocalExtraRows",
        "Action",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    args = parse_args()
    start = max(SOURCE_START, parse_iso(args.start_date))

    paths_by_code = institutional_paths()
    by_date = local_date_codes(paths_by_code)
    if not by_date:
        raise RuntimeError("No local institutional rows found.")

    latest_local = max(by_date)
    end = parse_iso(args.end_date) if args.end_date else latest_local
    trading_days = load_trading_days(start, end)
    local_counts = Counter({trading_date: len(by_date.get(trading_date, set())) for trading_date in trading_days})
    candidates = find_candidate_dates(
        trading_days,
        local_counts,
        args.drop_threshold,
        args.drop_ratio,
        args.neighbor_window,
    )
    listed_codes = load_listed_common_codes()

    print(f"candidate_dates={len(candidates)} range={start}..{end}")
    rows_to_write: dict[str, list[dict[str, object]]] = {}
    report_rows: list[dict[str, object]] = []

    for index, (trading_date, reason, local_count, nearby_median) in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] checking {trading_date} reason={reason} local={local_count}")
        official = fetch_official_rows(trading_date, listed_codes)
        official_codes = set(official["Code"].astype(str)) if not official.empty else set()
        local_codes = by_date.get(trading_date, set())
        missing_codes = official_codes - local_codes
        extra_codes = local_codes - official_codes
        action = "verified_no_missing"
        if missing_codes:
            missing_df = official[official["Code"].astype(str).isin(missing_codes)]
            for row in missing_df.to_dict("records"):
                rows_to_write.setdefault(str(row["Code"]), []).append(row)
            action = "would_append" if args.dry_run else "append"

        report_rows.append(
            {
                "Date": trading_date.isoformat(),
                "Reason": reason,
                "LocalRowsBefore": local_count,
                "NearbyMedianRows": "" if nearby_median is None else f"{nearby_median:.1f}",
                "OfficialRows": len(official_codes),
                "MissingOfficialRows": len(missing_codes),
                "LocalExtraRows": len(extra_codes),
                "Action": action,
            }
        )
        if args.pause_seconds and index < len(candidates):
            time.sleep(args.pause_seconds)

    touched_files = 0
    appended_rows = sum(len(rows) for rows in rows_to_write.values())
    if rows_to_write and not args.dry_run:
        touched_files = write_repaired_files(rows_to_write, paths_by_code)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = write_report(report_rows, run_id)
    print("Repair summary:")
    print(f"candidate_dates={len(candidates)}")
    print(f"official_missing_rows={appended_rows}")
    print(f"touched_files={touched_files}")
    print(f"report_path={report_path}")


if __name__ == "__main__":
    main()
