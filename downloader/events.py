"""
events.py

Download historical MOPS material events into per-stock CSV files.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from lxml import html


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "events"
METADATA_PATH = DATA_DIR / "metadata.csv"
LOG_DIR = PROJECT_ROOT / "logs" / "events"

MOPS_MATERIAL_EVENTS_URL = "https://mopsov.twse.com.tw/mops/web/ajax_t05st01"
MOPS_HISTORY_START_DATE = date(2011, 1, 1)
DEFAULT_END_DATE = date.today()
REQUEST_TIMEOUT_SECONDS = 40
DETAIL_REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 4
DETAIL_MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 20
THROTTLE_MIN_SECONDS = 0.4
THROTTLE_MAX_SECONDS = 1.0
DETAIL_THROTTLE_MIN_SECONDS = 0.2
DETAIL_THROTTLE_MAX_SECONDS = 0.6

COMMON_STOCK_TYPE = "\u80a1\u7968"
TWSE_MARKET = "\u4e0a\u5e02"

METADATA_CODE_COLUMN = "Code"
METADATA_NAME_COLUMN = "Name"
METADATA_TYPE_COLUMN = "\u985e\u578b"
METADATA_MARKET_COLUMN = "\u5e02\u5834"
METADATA_START_COLUMN = "\u8d77\u59cb\u65e5"

LABELS = {
    "\u767c\u8a00\u4eba": "Spokesperson",
    "\u767c\u8a00\u4eba\u8077\u7a31": "SpokespersonTitle",
    "\u767c\u8a00\u4eba\u96fb\u8a71": "SpokespersonPhone",
    "\u7b26\u5408\u689d\u6b3e": "Clause",
    "\u4e8b\u5be6\u767c\u751f\u65e5": "FactDate",
    "\u8aaa\u660e": "Description",
}

OUTPUT_COLUMNS = [
    "Date",
    "Time",
    "Code",
    "Name",
    "Subject",
    "FactDate",
    "Clause",
    "Description",
    "Spokesperson",
    "SpokespersonTitle",
    "SpokespersonPhone",
    "Source",
    "SourcePath",
    "SourceMarket",
    "DetailSeqNo",
    "DetailSpokeDate",
    "DetailSpokeTime",
    "FetchedAt",
]

DETAIL_COLUMNS = [
    "FactDate",
    "Clause",
    "Description",
    "Spokesperson",
    "SpokespersonTitle",
    "SpokespersonPhone",
]

DETAIL_FAILURE_COLUMNS = [
    "Code",
    "Date",
    "Time",
    "DetailSeqNo",
    "DetailSpokeDate",
    "DetailSpokeTime",
    "Error",
    "FailedAt",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 stock_analyse/1.0",
    "Referer": "https://mopsov.twse.com.tw/mops/web/t05st01",
    "Accept": "text/html,*/*",
    "Content-Type": "application/x-www-form-urlencoded",
}


@dataclass(frozen=True)
class Instrument:
    code: str
    name: str
    start_date: date | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MOPS historical material events into data/events."
    )
    parser.add_argument(
        "--metadata",
        default=str(METADATA_PATH),
        help="Metadata CSV path. Default: data/metadata.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Output directory. Default: data/events.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(LOG_DIR),
        help="Log directory. Default: logs/events.",
    )
    parser.add_argument(
        "--universe",
        choices=["listed-stocks", "all-metadata"],
        default="listed-stocks",
        help=(
            "Metadata universe to retain. all-metadata still fetches MOPS listed "
            "market rows, so non-MOPS instruments normally produce no rows."
        ),
    )
    parser.add_argument(
        "--codes",
        nargs="*",
        default=None,
        help="Optional stock codes to keep after fetching monthly MOPS rows.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "Start date YYYY-MM-DD. Default uses the later of the metadata start "
            "and the observed MOPS historical floor, 2011-01-01."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE.isoformat(),
        help="End date YYYY-MM-DD. Default: today.",
    )
    parser.add_argument(
        "--with-details",
        action="store_true",
        help="Fetch each detail page. This is much slower than list-only mode.",
    )
    parser.add_argument(
        "--enrich-existing-details",
        action="store_true",
        help=(
            "Read existing output CSVs and fill missing detail fields only. "
            "This is resumable and avoids refetching monthly list pages."
        ),
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Delete existing CSV files in the output directory before writing.",
    )
    parser.add_argument(
        "--max-months",
        type=int,
        default=None,
        help="Testing limit for number of monthly list pages to fetch.",
    )
    parser.add_argument(
        "--max-codes",
        type=int,
        default=None,
        help="Testing limit for number of metadata instruments to retain.",
    )
    parser.add_argument(
        "--sleep-min",
        type=float,
        default=THROTTLE_MIN_SECONDS,
        help="Minimum sleep between list requests.",
    )
    parser.add_argument(
        "--sleep-max",
        type=float,
        default=THROTTLE_MAX_SECONDS,
        help="Maximum sleep between list requests.",
    )
    parser.add_argument(
        "--detail-sleep-min",
        type=float,
        default=DETAIL_THROTTLE_MIN_SECONDS,
        help="Minimum sleep between detail requests.",
    )
    parser.add_argument(
        "--detail-sleep-max",
        type=float,
        default=DETAIL_THROTTLE_MAX_SECONDS,
        help="Maximum sleep between detail requests.",
    )
    parser.add_argument(
        "--max-detail-rows",
        type=int,
        default=None,
        help="Testing or chunking limit for detail rows to enrich.",
    )
    parser.add_argument(
        "--detail-save-every",
        type=int,
        default=100,
        help="Write the current CSV after this many enriched rows within a file.",
    )
    parser.add_argument(
        "--detail-retries",
        type=int,
        default=DETAIL_MAX_RETRIES,
        help="Retries per detail request. List requests still use the broader default.",
    )
    parser.add_argument(
        "--detail-timeout",
        type=float,
        default=DETAIL_REQUEST_TIMEOUT_SECONDS,
        help="Timeout seconds per detail request.",
    )
    parser.add_argument(
        "--max-consecutive-detail-failures",
        type=int,
        default=20,
        help="Stop the enrichment run after this many consecutive detail failures.",
    )
    parser.add_argument(
        "--detail-failure-log",
        default=None,
        help="CSV log for detail failures. Default: <log-dir>/events_detail_failures.csv.",
    )
    parser.add_argument(
        "--retry-known-detail-failures",
        action="store_true",
        help="Retry rows already listed in the detail failure log.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_start(value: date) -> date:
    return date(value.year, value.month, 1)


def month_end(value: date) -> date:
    if value.month == 12:
        return date(value.year, 12, 31)
    return date(value.year, value.month + 1, 1) - timedelta(days=1)


def iter_months(start_date: date, end_date: date) -> Iterable[date]:
    current = month_start(start_date)
    while current <= end_date:
        yield current
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def roc_year(value: date) -> int:
    return value.year - 1911


def clean_text(value: object) -> str:
    text = str(value or "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_roc_date(value: object) -> str:
    text = clean_text(value)
    compact_match = re.fullmatch(r"(\d{3})(\d{2})(\d{2})", text)
    if compact_match:
        return (
            f"{int(compact_match.group(1)) + 1911:04d}-"
            f"{int(compact_match.group(2)):02d}-"
            f"{int(compact_match.group(3)):02d}"
        )
    slash_match = re.fullmatch(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", text)
    if slash_match:
        return (
            f"{int(slash_match.group(1)) + 1911:04d}-"
            f"{int(slash_match.group(2)):02d}-"
            f"{int(slash_match.group(3)):02d}"
        )
    return text


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", str(value)).strip("._ ")
    return cleaned


def output_path_for(output_dir: Path, code: str, name: str) -> Path:
    safe_name = safe_filename_part(name)
    filename = f"{code}_{safe_name}.csv" if safe_name else f"{code}.csv"
    return output_dir / filename


def load_metadata(path: Path, universe: str, codes: list[str] | None) -> list[Instrument]:
    df = pd.read_csv(path, dtype={METADATA_CODE_COLUMN: str}, encoding="utf-8-sig").fillna("")
    df[METADATA_CODE_COLUMN] = df[METADATA_CODE_COLUMN].astype(str).str.strip()
    mask = df[METADATA_CODE_COLUMN].str.match(r"^\d{4}$")
    if universe == "listed-stocks":
        mask &= (
            (df[METADATA_MARKET_COLUMN] == TWSE_MARKET)
            & (df[METADATA_TYPE_COLUMN] == COMMON_STOCK_TYPE)
        )
    if codes:
        wanted = {str(code).strip() for code in codes}
        mask &= df[METADATA_CODE_COLUMN].isin(wanted)
    selected = df.loc[mask].drop_duplicates(METADATA_CODE_COLUMN)

    instruments: list[Instrument] = []
    for _, row in selected.sort_values(METADATA_CODE_COLUMN).iterrows():
        start = None
        start_text = str(row.get(METADATA_START_COLUMN, "")).strip()
        if start_text:
            try:
                start = parse_iso_date(start_text)
            except ValueError:
                start = None
        instruments.append(
            Instrument(
                code=str(row[METADATA_CODE_COLUMN]).strip(),
                name=str(row.get(METADATA_NAME_COLUMN, "")).strip(),
                start_date=start,
            )
        )
    return instruments


def build_list_payload(query_month: date, end_date: date) -> dict[str, str]:
    end_day = min(month_end(query_month), end_date).day
    return {
        "step": "1",
        "firstin": "1",
        "off": "1",
        "keyword4": "",
        "code1": "",
        "TYPEK2": "",
        "checkbtn": "",
        "queryName": "co_id",
        "inpuType": "co_id",
        "TYPEK": "sii",
        "co_id": "",
        "year": str(roc_year(query_month)),
        "month": f"{query_month.month:02d}",
        "b_date": "1",
        "e_date": str(end_day),
    }


def request_html(
    session: requests.Session,
    payload: dict[str, str],
    retries: int = MAX_RETRIES,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(
                MOPS_MATERIAL_EVENTS_URL,
                data=payload,
                headers=HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            response.encoding = "utf-8"
            text = response.text
            if "\u5b89\u5168\u6027\u8003\u91cf" in text:
                raise ValueError("MOPS returned a security warning page.")
            if "\u67e5\u8a62\u904e\u65bc\u983b\u7e41" in text:
                raise ValueError("MOPS returned a throttling page.")
            return text
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_seconds = RETRY_BACKOFF_SECONDS * attempt
            print(f"Request failed attempt {attempt}/{retries}: {exc}; sleep={sleep_seconds}s")
            time.sleep(sleep_seconds)
    raise last_error or RuntimeError("MOPS request failed")


def parse_detail_params(document: html.HtmlElement) -> list[dict[str, str]]:
    params: list[dict[str, str]] = []
    onclicks = document.xpath("//input/@onclick | //button/@onclick | //a/@onclick")
    pattern = re.compile(r"document\.t05st01_fm\.([A-Za-z_]+)\.value='([^']*)'")
    for onclick in onclicks:
        values = dict(pattern.findall(onclick))
        required = {"seq_no", "spoke_time", "spoke_date", "co_id"}
        if required.issubset(values):
            params.append(values)
    return params


def parse_list_rows(text: str, allowed_codes: set[str], fetched_at: str) -> list[dict[str, str]]:
    if "\u8cc7\u6599\u5eab\u4e2d\u67e5\u7121\u9700\u6c42\u8cc7\u6599" in text:
        return []

    document = html.fromstring(text)
    detail_params = parse_detail_params(document)
    detail_index = 0
    rows: list[dict[str, str]] = []

    for tr in document.xpath("//tr"):
        cells = [clean_text(" ".join(td.xpath(".//text()"))) for td in tr.xpath("./td")]
        if len(cells) < 5:
            continue
        code, name, roc_date, spoke_time, subject = cells[:5]
        if not re.fullmatch(r"\d{3}/\d{2}/\d{2}", roc_date):
            continue
        date_text = normalize_roc_date(roc_date)
        detail = detail_params[detail_index] if detail_index < len(detail_params) else {}
        detail_index += 1
        if code not in allowed_codes:
            continue
        rows.append(
            {
                "Date": date_text,
                "Time": spoke_time,
                "Code": code,
                "Name": name,
                "Subject": subject,
                "FactDate": "",
                "Clause": "",
                "Description": "",
                "Spokesperson": "",
                "SpokespersonTitle": "",
                "SpokespersonPhone": "",
                "Source": "MOPS",
                "SourcePath": "/mops/web/ajax_t05st01",
                "SourceMarket": "sii",
                "DetailSeqNo": detail.get("seq_no", ""),
                "DetailSpokeDate": detail.get("spoke_date", date_text.replace("-", "")),
                "DetailSpokeTime": detail.get("spoke_time", spoke_time.replace(":", "")),
                "FetchedAt": fetched_at,
            }
        )
    return rows


def build_detail_payload(row: dict[str, str]) -> dict[str, str]:
    return {
        "step": "2",
        "colorchg": "1",
        "seq_no": row["DetailSeqNo"],
        "spoke_time": row["DetailSpokeTime"],
        "spoke_date": row["DetailSpokeDate"],
        "co_id": row["Code"],
        "TYPEK": "sii",
        "off": "1",
        "firstin": "1",
        "year": str(int(row["Date"][:4]) - 1911),
        "month": row["DetailSpokeDate"][4:6],
        "b_date": "1",
        "e_date": "31",
    }


def parse_detail_row(text: str) -> dict[str, str]:
    document = html.fromstring(text)
    values = [clean_text(" ".join(td.xpath(".//text()"))) for td in document.xpath("//td")]
    detail = {
        "Spokesperson": "",
        "SpokespersonTitle": "",
        "SpokespersonPhone": "",
        "Clause": "",
        "FactDate": "",
        "Description": "",
    }
    for index, label in enumerate(values[:-1]):
        normalized = clean_text(label).replace("\uff1a", "").replace(":", "")
        key = LABELS.get(normalized)
        if not key:
            continue
        value = values[index + 1]
        detail[key] = normalize_roc_date(value) if key == "FactDate" else value
    return detail


def fetch_detail(
    session: requests.Session,
    row: dict[str, str],
    retries: int = DETAIL_MAX_RETRIES,
    timeout: float = DETAIL_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, str]:
    if not row.get("DetailSeqNo"):
        return {}
    text = request_html(session, build_detail_payload(row), retries=retries, timeout=timeout)
    return parse_detail_row(text)


def row_needs_detail(row: dict[str, str]) -> bool:
    if not row.get("DetailSeqNo"):
        return False
    return not any(row.get(column, "").strip() for column in DETAIL_COLUMNS)


def merge_detail(row: dict[str, str], detail: dict[str, str], fetched_at: str) -> bool:
    changed = False
    for column in DETAIL_COLUMNS:
        value = detail.get(column, "")
        if value and row.get(column, "") != value:
            row[column] = value
            changed = True
    if changed:
        row["FetchedAt"] = fetched_at
    return changed


def detail_failure_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("Code", ""),
        row.get("Date", ""),
        row.get("Time", ""),
        row.get("DetailSeqNo", ""),
        row.get("DetailSpokeDate", ""),
        row.get("DetailSpokeTime", ""),
    )


def read_detail_failure_keys(path: Path) -> set[tuple[str, str, str, str, str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (
                row.get("Code", ""),
                row.get("Date", ""),
                row.get("Time", ""),
                row.get("DetailSeqNo", ""),
                row.get("DetailSpokeDate", ""),
                row.get("DetailSpokeTime", ""),
            )
            for row in reader
        }


def append_detail_failure(path: Path, row: dict[str, str], error: str, failed_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_detail_failure_keys(path)
    key = detail_failure_key(row)
    if key in existing:
        return
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FAILURE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "Code": row.get("Code", ""),
                "Date": row.get("Date", ""),
                "Time": row.get("Time", ""),
                "DetailSeqNo": row.get("DetailSeqNo", ""),
                "DetailSpokeDate": row.get("DetailSpokeDate", ""),
                "DetailSpokeTime": row.get("DetailSpokeTime", ""),
                "Error": error,
                "FailedAt": failed_at,
            }
        )


def dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["Date"], row["Time"], row["Code"], row["DetailSeqNo"])
        existing = by_key.get(key)
        if not existing:
            by_key[key] = row
            continue
        if row.get("Description") and not existing.get("Description"):
            by_key[key] = row
    return sorted(by_key.values(), key=lambda r: (r["Date"], r["Time"], r["DetailSeqNo"]))


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            row = {column: raw.get(column, "") for column in OUTPUT_COLUMNS}
            rows.append(row)
        return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_stock_files(
    output_dir: Path,
    instruments: dict[str, Instrument],
    rows_by_code: dict[str, list[dict[str, str]]],
    replace_output: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if replace_output:
        for path in output_dir.glob("*.csv"):
            path.unlink()
    (output_dir / ".managed").touch()

    counts: dict[str, int] = {}
    for code, rows in sorted(rows_by_code.items()):
        instrument = instruments.get(code, Instrument(code=code, name=""))
        path = output_path_for(output_dir, code, instrument.name or rows[-1].get("Name", ""))
        combined = list(rows) if replace_output else read_existing_rows(path) + list(rows)
        deduped = dedupe_rows(combined)
        write_rows(path, deduped)
        counts[code] = len(deduped)
    return counts


def write_report(
    log_dir: Path,
    start_date: date,
    end_date: date,
    instruments: list[Instrument],
    month_count: int,
    total_rows: int,
    counts: dict[str, int],
    failures: list[str],
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"events_download_report_{timestamp}.md"
    nonempty = sum(1 for count in counts.values() if count > 0)
    lines = [
        "# MOPS Events Download Report",
        "",
        f"- StartedAt: {timestamp}",
        f"- Range: {start_date.isoformat()} to {end_date.isoformat()}",
        f"- Instruments requested: {len(instruments)}",
        f"- Months fetched: {month_count}",
        f"- Event rows parsed: {total_rows}",
        f"- Output files with rows: {nonempty}",
        f"- Failures: {len(failures)}",
        "",
        "## Top Row Counts",
        "",
    ]
    for code, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:20]:
        name = next((inst.name for inst in instruments if inst.code == code), "")
        lines.append(f"- {code} {name}: {count}")
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures[:200])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_enrichment_report(
    log_dir: Path,
    output_dir: Path,
    instruments: list[Instrument],
    files_scanned: int,
    rows_scanned: int,
    rows_needing_detail: int,
    rows_requested: int,
    rows_enriched: int,
    rows_unchanged: int,
    rows_skipped_known_failures: int,
    rows_skipped_malformed_dates: int,
    files_updated: int,
    failures: list[str],
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"events_detail_enrichment_report_{timestamp}.md"
    lines = [
        "# MOPS Events Detail Enrichment Report",
        "",
        f"- StartedAt: {timestamp}",
        f"- OutputDir: {output_dir}",
        f"- Instruments requested: {len(instruments)}",
        f"- Files scanned: {files_scanned}",
        f"- Rows scanned: {rows_scanned}",
        f"- Rows needing detail before this run: {rows_needing_detail}",
        f"- Detail requests attempted: {rows_requested}",
        f"- Rows enriched: {rows_enriched}",
        f"- Rows requested but unchanged: {rows_unchanged}",
        f"- Rows skipped from known failure log: {rows_skipped_known_failures}",
        f"- Rows skipped from malformed dates: {rows_skipped_malformed_dates}",
        f"- Files updated: {files_updated}",
        f"- Failures: {len(failures)}",
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures[:500])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def find_existing_event_file(output_dir: Path, instrument: Instrument) -> Path | None:
    direct = output_path_for(output_dir, instrument.code, instrument.name)
    if direct.exists():
        return direct
    code_only = output_dir / f"{instrument.code}.csv"
    if code_only.exists():
        return code_only
    matches = sorted(output_dir.glob(f"{instrument.code}_*.csv"))
    return matches[0] if matches else None


def enrich_existing_details(
    output_dir: Path,
    log_dir: Path,
    instruments: list[Instrument],
    start_date: date,
    end_date: date,
    detail_sleep_min: float,
    detail_sleep_max: float,
    max_detail_rows: int | None,
    detail_save_every: int,
    detail_retries: int,
    detail_timeout: float,
    max_consecutive_detail_failures: int,
    detail_failure_log: Path,
    retry_known_detail_failures: bool,
) -> Path:
    session = requests.Session()
    fetched_at = datetime.now().isoformat(timespec="seconds")
    failure_keys = set() if retry_known_detail_failures else read_detail_failure_keys(detail_failure_log)
    files_scanned = 0
    rows_scanned = 0
    rows_needing_detail = 0
    rows_requested = 0
    rows_enriched = 0
    rows_unchanged = 0
    rows_skipped_known_failures = 0
    rows_skipped_malformed_dates = 0
    files_updated = 0
    failures: list[str] = []
    exhausted = False
    consecutive_failures = 0

    print(
        f"Enriching existing MOPS details: instruments={len(instruments)} "
        f"range={start_date}..{end_date} max_detail_rows={max_detail_rows}"
        f" detail_save_every={detail_save_every} known_failures={len(failure_keys)}",
        flush=True,
    )
    for instrument in instruments:
        if exhausted:
            break
        path = find_existing_event_file(output_dir, instrument)
        if not path:
            continue

        rows = read_existing_rows(path)
        files_scanned += 1
        rows_scanned += len(rows)
        changed = False
        file_written = False
        changed_since_write = 0
        file_requested = 0
        file_enriched = 0

        for row in rows:
            if not row_needs_detail(row):
                continue
            try:
                row_date = parse_iso_date(row.get("Date", ""))
            except ValueError:
                rows_skipped_malformed_dates += 1
                continue
            if row_date < start_date or row_date > end_date:
                continue
            if detail_failure_key(row) in failure_keys:
                rows_needing_detail += 1
                rows_skipped_known_failures += 1
                continue
            rows_needing_detail += 1
            if max_detail_rows is not None and rows_requested >= max_detail_rows:
                exhausted = True
                break
            try:
                detail = fetch_detail(
                    session,
                    row,
                    retries=detail_retries,
                    timeout=detail_timeout,
                )
                rows_requested += 1
                file_requested += 1
                if merge_detail(row, detail, fetched_at):
                    rows_enriched += 1
                    file_enriched += 1
                    changed = True
                    changed_since_write += 1
                    if detail_save_every > 0 and changed_since_write >= detail_save_every:
                        write_rows(path, rows)
                        file_written = True
                        changed_since_write = 0
                else:
                    rows_unchanged += 1
                consecutive_failures = 0
                time.sleep(random.uniform(detail_sleep_min, detail_sleep_max))
            except Exception as exc:
                rows_requested += 1
                file_requested += 1
                consecutive_failures += 1
                session.close()
                session = requests.Session()
                message = f"detail {row['Code']} {row['Date']} {row['Time']}: {exc}"
                print(f"FAILED {message}")
                failures.append(message)
                append_detail_failure(detail_failure_log, row, str(exc), fetched_at)
                failure_keys.add(detail_failure_key(row))
                if (
                    max_consecutive_detail_failures > 0
                    and consecutive_failures >= max_consecutive_detail_failures
                ):
                    print(
                        "Stopping detail enrichment after "
                        f"{consecutive_failures} consecutive detail failures.",
                        flush=True,
                    )
                    exhausted = True
                    break
                time.sleep(random.uniform(detail_sleep_min, detail_sleep_max))

        if changed and changed_since_write:
            write_rows(path, rows)
            file_written = True
        if file_written:
            files_updated += 1
        if file_requested:
            print(
                f"{instrument.code} {instrument.name}: "
                f"requested={file_requested} enriched={file_enriched} "
                f"updated={file_written}",
                flush=True,
            )

    report_path = write_enrichment_report(
        log_dir,
        output_dir,
        instruments,
        files_scanned,
        rows_scanned,
        rows_needing_detail,
        rows_requested,
        rows_enriched,
        rows_unchanged,
        rows_skipped_known_failures,
        rows_skipped_malformed_dates,
        files_updated,
        failures,
    )
    print(
        f"Detail enrichment done. requested={rows_requested} enriched={rows_enriched} "
        f"failures={len(failures)} report={report_path}"
    )
    return report_path


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    detail_failure_log = (
        Path(args.detail_failure_log)
        if args.detail_failure_log
        else log_dir / "events_detail_failures.csv"
    )
    end_date = parse_iso_date(args.end_date)
    instruments = load_metadata(metadata_path, args.universe, args.codes)
    if args.max_codes is not None:
        instruments = instruments[: args.max_codes]
    if not instruments:
        raise ValueError("No metadata instruments selected.")

    metadata_start_dates = [instrument.start_date for instrument in instruments if instrument.start_date]
    default_start = max(min(metadata_start_dates or [MOPS_HISTORY_START_DATE]), MOPS_HISTORY_START_DATE)
    start_date = parse_iso_date(args.start_date) if args.start_date else default_start
    start_date = max(start_date, MOPS_HISTORY_START_DATE)
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is after end date {end_date}.")

    if args.enrich_existing_details:
        enrich_existing_details(
            output_dir=output_dir,
            log_dir=log_dir,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date,
            detail_sleep_min=args.detail_sleep_min,
            detail_sleep_max=args.detail_sleep_max,
            max_detail_rows=args.max_detail_rows,
            detail_save_every=args.detail_save_every,
            detail_retries=args.detail_retries,
            detail_timeout=args.detail_timeout,
            max_consecutive_detail_failures=args.max_consecutive_detail_failures,
            detail_failure_log=detail_failure_log,
            retry_known_detail_failures=args.retry_known_detail_failures,
        )
        return

    allowed_codes = {instrument.code for instrument in instruments}
    instrument_by_code = {instrument.code: instrument for instrument in instruments}
    rows_by_code: dict[str, list[dict[str, str]]] = defaultdict(list)
    failures: list[str] = []
    fetched_at = datetime.now().isoformat(timespec="seconds")

    session = requests.Session()
    months = list(iter_months(start_date, end_date))
    if args.max_months is not None:
        months = months[: args.max_months]

    print(
        f"Downloading MOPS events: instruments={len(instruments)} "
        f"months={len(months)} range={start_date}..{end_date}"
    )
    parsed_rows = 0
    for index, query_month in enumerate(months, start=1):
        try:
            text = request_html(session, build_list_payload(query_month, end_date))
            rows = parse_list_rows(text, allowed_codes, fetched_at)
            if args.with_details:
                detailed_rows = []
                for row in rows:
                    try:
                        row.update(
                            fetch_detail(
                                session,
                                row,
                                retries=args.detail_retries,
                                timeout=args.detail_timeout,
                            )
                        )
                    except Exception as exc:
                        failures.append(
                            f"detail {row['Code']} {row['Date']} {row['Time']}: {exc}"
                        )
                    detailed_rows.append(row)
                rows = detailed_rows
            for row in rows:
                rows_by_code[row["Code"]].append(row)
            parsed_rows += len(rows)
            print(
                f"{index}/{len(months)} {query_month:%Y-%m}: "
                f"rows={len(rows)} total={parsed_rows}"
            )
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))
        except Exception as exc:
            message = f"list {query_month:%Y-%m}: {exc}"
            print(f"FAILED {message}")
            failures.append(message)

    counts = write_stock_files(output_dir, instrument_by_code, rows_by_code, args.replace_output)
    report_path = write_report(
        log_dir,
        start_date,
        end_date,
        instruments,
        len(months),
        parsed_rows,
        counts,
        failures,
    )
    print(
        f"Done. files={len(counts)} rows={sum(counts.values())} "
        f"failures={len(failures)} report={report_path}"
    )


if __name__ == "__main__":
    main()
