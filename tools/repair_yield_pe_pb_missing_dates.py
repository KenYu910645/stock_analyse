"""
Repair local yield/PER/PBR dates that TWSE has but data/yield_pe_pb missed.

The default input is the quality-audit CSV generated under
output/yield_pe_pb_quality/. Only dates classified as local download/update
misses are fetched and merged back into the canonical per-stock CSV files.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from downloader.yield_pe_pb import (  # noqa: E402
    get_json_response,
    load_listed_common_metadata,
    normalize_dataframe,
    parse_iso_date,
    parse_twse_rows,
    sleep_between_requests,
    write_per_stock_csvs,
)
from column_schema import read_csv_canonical  # noqa: E402


DEFAULT_AUDIT_CSV = (
    PROJECT_ROOT
    / "output"
    / "yield_pe_pb_quality"
    / "official_check_missing_all_market_dates.csv"
)
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "yield_pe_pb_repair"
PRICE_DIR = PROJECT_ROOT / "data" / "price"
REPAIR_CLASSIFICATION = "local_missing_download_or_update"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch TWSE valuation dates that official data contains but local "
            "data/yield_pe_pb missed."
        )
    )
    parser.add_argument(
        "--audit-csv",
        default=str(DEFAULT_AUDIT_CSV),
        help=(
            "CSV with Date and classification columns. Default: "
            "output/yield_pe_pb_quality/official_check_missing_all_market_dates.csv"
        ),
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Explicit YYYY-MM-DD dates to repair instead of reading --audit-csv.",
    )
    parser.add_argument(
        "--classification",
        default=REPAIR_CLASSIFICATION,
        help=(
            "Audit classification to repair. Default: "
            f"{REPAIR_CLASSIFICATION}."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of dates to process; 0 means no limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and parse rows but do not write data/yield_pe_pb CSVs.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory for the repair progress CSV log.",
    )
    return parser.parse_args()


def unique_sorted_dates(values: list[str]) -> list:
    dates = sorted({parse_iso_date(str(value).strip()) for value in values if str(value).strip()})
    return dates


def load_dates(args: argparse.Namespace) -> list:
    if args.dates:
        return unique_sorted_dates(args.dates)

    audit_csv = Path(args.audit_csv)
    if not audit_csv.exists():
        raise FileNotFoundError(f"Audit CSV not found: {audit_csv}")

    audit = pd.read_csv(audit_csv, dtype=str).fillna("")
    required = {"Date", "classification"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"{audit_csv} is missing columns: {sorted(missing)}")

    target = audit[audit["classification"].eq(args.classification)]
    return unique_sorted_dates(target["Date"].tolist())


def write_log(records: list[dict], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(log_path, index=False, encoding="utf-8-sig")


def blank_cell(value) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def price_path_for_code(code: str) -> Path | None:
    matches = sorted(PRICE_DIR.glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def fill_close_from_price(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty or not {"Date", "Code", "Close"}.issubset(frame.columns):
        return frame, 0

    result = frame.copy()
    filled = 0
    blank = result["Close"].map(blank_cell)
    if not blank.any():
        return result, 0

    for code, stock_df in result[blank].groupby("Code", sort=False):
        path = price_path_for_code(str(code).strip())
        if path is None:
            continue
        try:
            price = read_csv_canonical(path, dtype=str, usecols=["Date", "Close"]).fillna("")
        except ValueError:
            continue
        if price.empty:
            continue

        close_by_date = {
            str(row["Date"]): str(row["Close"]).strip()
            for _, row in price.iterrows()
            if str(row.get("Close", "")).strip()
        }
        for idx, row in stock_df.iterrows():
            close = close_by_date.get(str(row.get("Date", "")).strip(), "")
            if close:
                result.at[idx, "Close"] = close
                filled += 1

    return result, filled


def repair_dates(dates: list, args: argparse.Namespace) -> tuple[Path, list[dict]]:
    metadata = load_listed_common_metadata()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"yield_pe_pb_repair_{timestamp}.csv"
    records: list[dict] = []
    frames: list[pd.DataFrame] = []
    total = len(dates)

    for index, query_date in enumerate(dates, start=1):
        record = {
            "Date": query_date.isoformat(),
            "status": "",
            "official_status": "",
            "official_payload_date": "",
            "official_rows": 0,
            "listed_common_rows": 0,
            "filled_close_rows": 0,
            "written_files": 0,
            "rows_after_merge": 0,
            "error": "",
        }
        print(f"[{index}/{total}] Fetching {query_date}.")

        try:
            payload = get_json_response(query_date)
            record["official_status"] = str(payload.get("stat", ""))
            record["official_payload_date"] = str(payload.get("date", ""))

            rows = parse_twse_rows(payload, query_date)
            record["official_rows"] = len(rows)
            if not rows:
                record["status"] = "no_source_data"
            else:
                frame = normalize_dataframe(rows)
                frame = frame[frame["Code"].isin(metadata.index)].copy()
                record["listed_common_rows"] = len(frame)

                if frame.empty:
                    record["status"] = "no_listed_common_rows"
                elif args.dry_run:
                    record["status"] = "dry_run"
                else:
                    frame, filled_close = fill_close_from_price(frame)
                    record["filled_close_rows"] = filled_close
                    frames.append(frame)
                    record["status"] = "downloaded"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"Failed {query_date}: {exc}")

        records.append(record)
        write_log(records, log_path)
        print(
            f"{record['Date']} {record['status']} "
            f"official_rows={record['official_rows']} "
            f"listed_common_rows={record['listed_common_rows']} "
            f"filled_close_rows={record['filled_close_rows']} "
            f"written_files={record['written_files']}",
            flush=True,
        )

        if index < total:
            sleep_between_requests()

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        written_files, rows_after_merge = write_per_stock_csvs(combined, metadata)
        summary = {
            "Date": "ALL",
            "status": "batch_repaired",
            "official_status": "",
            "official_payload_date": "",
            "official_rows": int(sum(record["official_rows"] for record in records)),
            "listed_common_rows": int(len(combined)),
            "filled_close_rows": int(sum(record["filled_close_rows"] for record in records)),
            "written_files": int(written_files),
            "rows_after_merge": int(rows_after_merge),
            "error": "",
        }
        records.append(summary)
        write_log(records, log_path)
        print(
            "batch_repaired "
            f"listed_common_rows={summary['listed_common_rows']} "
            f"filled_close_rows={summary['filled_close_rows']} "
            f"written_files={written_files} "
            f"rows_after_merge={rows_after_merge}",
            flush=True,
        )

    return log_path, records


def main() -> None:
    args = parse_args()
    dates = load_dates(args)
    if args.limit and args.limit > 0:
        dates = dates[: args.limit]

    if not dates:
        print("No repair dates found.")
        return

    log_path, records = repair_dates(dates, args)
    status_counts = pd.Series([record["status"] for record in records]).value_counts()

    print("Repair summary:")
    print(status_counts.to_string())
    print(f"dates_processed={len(records)}")
    print(f"log_path={log_path}")

    failed = [record for record in records if record["status"] == "failed"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
