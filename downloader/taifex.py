"""Download TAIFEX futures, options, sentiment, and position CSV data.

The downloader writes cleaned source-style CSVs under ``data/taifex``. Dates are
normalized to Gregorian ISO ``YYYY-MM-DD`` and downloads are chunked to match
TAIFEX query limits.
"""

import argparse
import csv
import io
import os
import random
import re
import sys
import time
import zipfile
from datetime import date, datetime, timedelta
from itertools import chain
from pathlib import Path
from typing import Iterable

import requests

from column_schema import storage_columns, storage_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "taifex"
LOG_DIR = PROJECT_ROOT / "logs" / "taifex"
BASE_URL = "https://www.taifex.com.tw/enl/eng3"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DATASETS = {
    "futures_daily",
    "options_daily",
    "put_call_ratio",
    "institutional",
    "large_trader_oi",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Datasets to download. Default: all.",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Inclusive end date in YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite outputs even when they already exist.",
    )
    parser.add_argument(
        "--sleep-min",
        type=float,
        default=0.2,
        help="Minimum polite sleep seconds between requests.",
    )
    parser.add_argument(
        "--sleep-max",
        type=float,
        default=0.8,
        help="Maximum polite sleep seconds between requests.",
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def display_date(value: date) -> str:
    return value.strftime("%Y/%m/%d")


def chunk_dates(start: date, end: date, max_days: int) -> Iterable[tuple[date, date]]:
    current = start
    while current <= end:
        chunk_end = min(end, current + timedelta(days=max_days - 1))
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def sleep_between_requests(args: argparse.Namespace) -> None:
    time.sleep(random.uniform(args.sleep_min, args.sleep_max))


def decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "ms950", "big5", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def is_csv_response(response: requests.Response) -> bool:
    disposition = response.headers.get("content-disposition", "").lower()
    if ".csv" in disposition or ".zip" in disposition:
        return True
    prefix = response.content[:80].lower()
    return prefix.startswith(b"date,") or prefix.startswith(b"pk\x03\x04")


def request_with_retry(
    session: requests.Session,
    endpoint: str,
    data: dict[str, str],
    referer: str,
    args: argparse.Namespace,
    attempts: int = 5,
) -> requests.Response:
    url = f"{BASE_URL}/{endpoint}"
    headers = dict(HEADERS)
    headers["Referer"] = referer
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.post(url, data=data, headers=headers, timeout=120)
            response.raise_for_status()
            if is_csv_response(response):
                return response
            snippet = decode_text(response.content[:500]).replace("\n", " ")
            raise RuntimeError(f"TAIFEX returned non-CSV content: {snippet[:200]}")
        except Exception as exc:  # noqa: BLE001 - keep downloader resilient.
            last_error = exc
            if attempt == attempts:
                break
            wait_seconds = 5 * attempt
            print(f"Retry {attempt}/{attempts} for {endpoint} after {wait_seconds}s: {exc}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed TAIFEX request {endpoint} {data}: {last_error}")


def iter_csv_rows(content: bytes) -> Iterable[list[str]]:
    text = decode_text(content).replace("\ufeff", "")
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            continue
        yield [cell.strip() for cell in row]


def normalize_header(header: list[str]) -> list[str]:
    cleaned = [cell.strip() for cell in header]
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    if cleaned:
        cleaned[0] = "Date"
    return cleaned


def normalize_row(row: list[str], header_len: int) -> list[str]:
    if len(row) > header_len:
        row = row[:header_len]
    elif len(row) < header_len:
        row = row + [""] * (header_len - len(row))
    if row:
        row[0] = normalize_date_cell(row[0])
    return row


def normalize_date_cell(value: str) -> str:
    value = value.strip()
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value.replace("/", "-")


def write_csv_from_sources(output_path: Path, sources: Iterable[tuple[str, bytes]]) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total_rows = 0
    header: list[str] | None = None
    min_date: str | None = None
    max_date: str | None = None

    with tmp_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        for source_name, content in sources:
            source_rows = 0
            for raw_row in iter_csv_rows(content):
                if raw_row and raw_row[0].strip().lower() == "date":
                    row_header = normalize_header(raw_row)
                    if header is None:
                        header = row_header
                        writer.writerow(storage_columns(header))
                    continue
                if header is None:
                    raise RuntimeError(f"{source_name} has data rows before a Date header.")
                row = normalize_row(raw_row, len(header))
                if not row or not re.match(r"\d{4}-\d{2}-\d{2}$", row[0]):
                    continue
                writer.writerow(row)
                total_rows += 1
                source_rows += 1
                min_date = row[0] if min_date is None or row[0] < min_date else min_date
                max_date = row[0] if max_date is None or row[0] > max_date else max_date
            output.flush()
            print(f"Wrote {source_rows} rows from {source_name}; cumulative rows={total_rows}", flush=True)

    if header is None:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"No CSV header found for {output_path}")
    os.replace(tmp_path, output_path)
    return {"path": str(output_path), "rows": total_rows, "min_date": min_date, "max_date": max_date}


def annual_zip_sources(
    session: requests.Session,
    endpoint: str,
    referer: str,
    years: Iterable[int],
    args: argparse.Namespace,
) -> Iterable[tuple[str, bytes]]:
    for year in years:
        print(f"Downloading {endpoint} annual ZIP {year}", flush=True)
        response = request_with_retry(
            session,
            endpoint,
            {"down_type": "2", "his_year": str(year)},
            referer,
            args,
        )
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".csv"):
                    yield f"{year}:{name}", archive.read(name)
        sleep_between_requests(args)


def chunk_csv_sources(
    session: requests.Session,
    endpoint: str,
    referer: str,
    start: date,
    end: date,
    max_days: int,
    data_extra: dict[str, str],
    args: argparse.Namespace,
) -> Iterable[tuple[str, bytes]]:
    for chunk_start, chunk_end in chunk_dates(start, end, max_days):
        data = {
            "queryStartDate": display_date(chunk_start),
            "queryEndDate": display_date(chunk_end),
        }
        data.update(data_extra)
        print(f"Downloading {endpoint} {chunk_start} to {chunk_end}", flush=True)
        response = request_with_retry(session, endpoint, data, referer, args)
        yield f"{endpoint}:{chunk_start}:{chunk_end}", response.content
        sleep_between_requests(args)


def download_futures_daily(session: requests.Session, end_date: date, args: argparse.Namespace) -> dict[str, object]:
    output_path = DATA_ROOT / "futures_daily" / "futures_daily.csv"
    if output_path.exists() and not args.force:
        print(f"Skip existing {output_path}")
        return summarize_csv(output_path)
    end_year = min(end_date.year - 1, 2025)
    annual_years = range(1998, end_year + 1)
    current_start = date(2026, 1, 1) if end_date.year >= 2026 else end_date + timedelta(days=1)
    sources = annual_zip_sources(session, "futDataDown", f"{BASE_URL}/futDailyMarketView", annual_years, args)
    if current_start <= end_date:
        sources = chain(
            sources,
            chunk_csv_sources(
                session,
                "futDataDown",
                f"{BASE_URL}/futDailyMarketView",
                current_start,
                end_date,
                29,
                {"down_type": "1", "commodity_id": "all", "commodity_id2": ""},
                args,
            ),
        )
    return write_csv_from_sources(output_path, sources)


def download_options_daily(session: requests.Session, end_date: date, args: argparse.Namespace) -> dict[str, object]:
    output_path = DATA_ROOT / "options_daily" / "options_daily.csv"
    if output_path.exists() and not args.force:
        print(f"Skip existing {output_path}")
        return summarize_csv(output_path)
    end_year = min(end_date.year - 1, 2025)
    annual_years = range(2001, end_year + 1)
    current_start = date(2026, 1, 1) if end_date.year >= 2026 else end_date + timedelta(days=1)
    sources = annual_zip_sources(session, "optDataDown", f"{BASE_URL}/optDailyMarketView", annual_years, args)
    if current_start <= end_date:
        sources = chain(
            sources,
            chunk_csv_sources(
                session,
                "optDataDown",
                f"{BASE_URL}/optDailyMarketView",
                current_start,
                end_date,
                29,
                {"down_type": "1", "commodity_id": "all", "commodity_id2": ""},
                args,
            ),
        )
    return write_csv_from_sources(output_path, sources)


def download_put_call_ratio(session: requests.Session, end_date: date, args: argparse.Namespace) -> dict[str, object]:
    output_path = DATA_ROOT / "put_call_ratio" / "put_call_ratio.csv"
    if output_path.exists() and not args.force:
        print(f"Skip existing {output_path}")
        return summarize_csv(output_path)
    sources = chunk_csv_sources(
        session,
        "pcRatioDown",
        f"{BASE_URL}/pcRatio?menuid1=03",
        date(2001, 12, 24),
        end_date,
        31,
        {},
        args,
    )
    return write_csv_from_sources(output_path, sources)


def download_institutional(session: requests.Session, end_date: date, args: argparse.Namespace) -> list[dict[str, object]]:
    # The English CSV download endpoint exposes a rolling window advertised in
    # hidden firstDate/lastDate fields. Older records require TAIFEX historical
    # data application or another licensed source.
    start, source_last_date = fetch_institutional_window(session)
    end_date = min(end_date, source_last_date)
    outputs = [
        (
            DATA_ROOT / "institutional" / "total_table.csv",
            "totalTableDateDown",
            "totalTableDateView?menuid1=03",
        ),
        (
            DATA_ROOT / "institutional" / "futures_options.csv",
            "futAndOptDateDown",
            "futAndOptDateView",
        ),
    ]
    results = []
    for output_path, endpoint, page in outputs:
        if output_path.exists() and not args.force:
            print(f"Skip existing {output_path}")
            results.append(summarize_csv(output_path))
            continue
        sources = chunk_csv_sources(
            session,
            endpoint,
            f"{BASE_URL}/{page}",
            start,
            end_date,
            31,
            {},
            args,
        )
        results.append(write_csv_from_sources(output_path, sources))
    return results


def fetch_institutional_window(session: requests.Session) -> tuple[date, date]:
    response = session.get(
        f"{BASE_URL}/totalTableDateView?menuid1=03",
        headers=HEADERS,
        timeout=60,
    )
    response.raise_for_status()
    text = response.text
    first_match = re.search(r'id="firstDate"[^>]*value="(\d{4}/\d{2}/\d{2})', text)
    last_match = re.search(r'id="lastDate"[^>]*value="(\d{4}/\d{2}/\d{2})', text)
    if not first_match or not last_match:
        raise RuntimeError("Could not discover TAIFEX institutional firstDate/lastDate window.")
    return (
        datetime.strptime(first_match.group(1), "%Y/%m/%d").date(),
        datetime.strptime(last_match.group(1), "%Y/%m/%d").date(),
    )


def download_large_trader_oi(session: requests.Session, end_date: date, args: argparse.Namespace) -> list[dict[str, object]]:
    start = date(2004, 7, 1)
    outputs = [
        (
            DATA_ROOT / "large_trader_oi" / "futures.csv",
            "largeTraderFutDown",
            "largeTraderFutView",
        ),
        (
            DATA_ROOT / "large_trader_oi" / "options.csv",
            "largeTraderOptDown",
            "largeTraderOptView",
        ),
    ]
    results = []
    for output_path, endpoint, page in outputs:
        if output_path.exists() and not args.force:
            print(f"Skip existing {output_path}")
            results.append(summarize_csv(output_path))
            continue
        sources = chunk_csv_sources(
            session,
            endpoint,
            f"{BASE_URL}/{page}",
            start,
            end_date,
            90,
            {},
            args,
        )
        results.append(write_csv_from_sources(output_path, sources))
    return results


def summarize_csv(path: Path) -> dict[str, object]:
    rows = 0
    min_date: str | None = None
    max_date: str | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            value = (
                row.get("Date")
                or row.get(storage_name("Date"))
                or row.get("date")
                or row.get(storage_name("date"))
            )
            rows += 1
            if value:
                min_date = value if min_date is None or value < min_date else min_date
                max_date = value if max_date is None or value > max_date else max_date
    return {"path": str(path), "rows": rows, "min_date": min_date, "max_date": max_date}


def write_manifest(results: list[dict[str, object]], end_date: date) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = LOG_DIR / "taifex_download_manifest.csv"
    fieldnames = ["FetchedAt", "EndDate", "Path", "Rows", "MinDate", "MaxDate"]
    fetched_at = datetime.now().isoformat(timespec="seconds")
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "FetchedAt": fetched_at,
                    "EndDate": end_date.isoformat(),
                    "Path": result["path"],
                    "Rows": result["rows"],
                    "MinDate": result["min_date"],
                    "MaxDate": result["max_date"],
                }
            )
    print(f"Wrote manifest {manifest_path}")


def main() -> int:
    args = parse_args()
    if args.sleep_min < 0 or args.sleep_max < args.sleep_min:
        raise ValueError("--sleep-max must be >= --sleep-min and both must be non-negative.")
    end_date = parse_date(args.end_date)
    results: list[dict[str, object]] = []
    with requests.Session() as session:
        session.headers.update(HEADERS)
        if "futures_daily" in args.datasets:
            results.append(download_futures_daily(session, end_date, args))
        if "options_daily" in args.datasets:
            results.append(download_options_daily(session, end_date, args))
        if "put_call_ratio" in args.datasets:
            results.append(download_put_call_ratio(session, end_date, args))
        if "institutional" in args.datasets:
            results.extend(download_institutional(session, end_date, args))
        if "large_trader_oi" in args.datasets:
            results.extend(download_large_trader_oi(session, end_date, args))
    write_manifest(results, end_date)
    for result in results:
        print(
            f"{result['path']}: rows={result['rows']} "
            f"range={result['min_date']}..{result['max_date']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
