from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical, to_csv_storage  # noqa: E402


BASE_URL = "https://bsr.twse.com.tw/bshtm/"
MENU_URL = urljoin(BASE_URL, "bsMenu.aspx")
CAPTCHA_RE = re.compile(r"CaptchaImage\.aspx\?guid=[0-9a-fA-F-]+")
CONTENT_RE = re.compile(r"bsContent\.aspx\?[^\"'<> ]*StkNo=(?P<stock>\d+)[^\"'<> ]*RecCount=\d+")
TEXT_RE = re.compile(r"[^A-Z0-9]")
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_METADATA_PATH = DATA_DIR / "metadata.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "broker" / "twse" / "by_stock"
DEFAULT_RAW_DIR = PROJECT_ROOT / "logs" / "broker" / "twse_captcha"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "broker"
LISTED_MARKET = "\u4e0a\u5e02"
COMMON_STOCK_TYPE = "\u80a1\u7968"
HAS_BROKER_COLUMN = "has_broker"
AVAILABLE_DATASET_COUNT_COLUMN = "available_dataset_count"
NO_DATA_MARKERS = (
    "\u67e5\u7121",
    "\u7121\u8cc7\u6599",
    "\u7121\u6b64",
    "\u672a\u6709\u8cc7\u6599",
)
_TEXT_RECOGNIZER: Any | None = None


@dataclass
class DownloadResult:
    stock: str
    captcha_text: str
    captcha_image_path: Path
    csv_path: Path
    content_url: str
    content_type: str
    byte_count: int
    attempt: int


class NoBrokerDataError(RuntimeError):
    """Raised when TWSE returns a clear no-data message for a stock."""


def clean_ocr_text(text: str) -> str:
    return TEXT_RE.sub("", "".join(text.split()).upper())[:5]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def today_text() -> str:
    return date.today().isoformat()


def file_date(value: str) -> str:
    return value.replace("-", "")


def extract_input_value(markup: str, name: str) -> str:
    patterns = [
        rf'id=["\']{re.escape(name)}["\'][^>]*value=["\']([^"\']*)["\']',
        rf'name=["\']{re.escape(name)}["\'][^>]*value=["\']([^"\']*)["\']',
        rf'value=["\']([^"\']*)["\'][^>]*(?:id|name)=["\']{re.escape(name)}["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, markup, flags=re.IGNORECASE)
        if match:
            return html.unescape(match.group(1))
    raise RuntimeError(f"Could not find required ASP.NET hidden field: {name}")


def fetch_menu(session: requests.Session) -> str:
    session.get(BASE_URL, timeout=30).raise_for_status()
    response = session.get(MENU_URL, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def extract_captcha_url(menu_html: str) -> str:
    match = CAPTCHA_RE.search(menu_html)
    if not match:
        raise RuntimeError("Could not find captcha image URL in bsMenu.aspx")
    return urljoin(MENU_URL, html.unescape(match.group(0)))


def save_captcha(session: requests.Session, captcha_url: str, raw_dir: Path) -> Path:
    response = session.get(captcha_url, timeout=30)
    response.raise_for_status()

    guid_match = re.search(r"guid=([0-9a-fA-F-]+)", captcha_url)
    guid = guid_match.group(1) if guid_match else "unknown-guid"

    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / f"bsr_twse_captcha_{timestamp()}_{guid}.jfif"
    output.write_bytes(response.content)
    return output


def recognize_captcha(captcha_path: Path) -> str:
    global _TEXT_RECOGNIZER

    try:
        from paddleocr import TextRecognition
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is required. Run this script with the PaddleOCR environment, for example: "
            r"C:\Users\spide\AppData\Local\Temp\pic_analyse_paddle_venv\Scripts\python.exe broker.py 2330"
        ) from exc

    png_path = Path(tempfile.gettempdir()) / f"{captcha_path.stem}.png"
    Image.open(captcha_path).convert("RGB").save(png_path)

    if _TEXT_RECOGNIZER is None:
        _TEXT_RECOGNIZER = TextRecognition()
    result = _TEXT_RECOGNIZER.predict(str(png_path))
    if not result:
        raise RuntimeError(f"PaddleOCR returned no result for {captcha_path}")

    text = clean_ocr_text(str(result[0].get("rec_text", "")))
    if len(text) != 5:
        raise RuntimeError(f"PaddleOCR result was not 5 characters: {text!r}")
    return text


def ensure_ocr_dependency_available() -> None:
    try:
        from paddleocr import TextRecognition  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR is required. Run this script with the PaddleOCR environment, for example: "
            r"C:\Users\spide\AppData\Local\Temp\pic_analyse_paddle_venv\Scripts\python.exe broker.py 2330"
        ) from exc


def build_query_payload(menu_html: str, stock: str, captcha_text: str) -> dict[str, str]:
    return {
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "__LASTFOCUS": "",
        "__VIEWSTATE": extract_input_value(menu_html, "__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": extract_input_value(menu_html, "__VIEWSTATEGENERATOR"),
        "__EVENTVALIDATION": extract_input_value(menu_html, "__EVENTVALIDATION"),
        "RadioButton_Normal": "RadioButton_Normal",
        "TextBox_Stkno": stock,
        "CaptchaControl1": captcha_text,
        "btnOK": "查詢",
    }


def is_no_data_message(message: str) -> bool:
    return any(marker in message for marker in NO_DATA_MARKERS)


def extract_content_url(response_html: str, stock: str) -> str:
    candidates = []
    for match in CONTENT_RE.finditer(response_html):
        candidate = html.unescape(match.group(0)).replace("&amp;", "&")
        if f"StkNo={stock}" in candidate:
            candidates.append(candidate)

    if not candidates:
        error = re.search(r'id=["\']Label_ErrorMsg["\'][^>]*>(.*?)</span>', response_html, re.IGNORECASE | re.DOTALL)
        message = re.sub(r"\s+", " ", html.unescape(error.group(1))).strip() if error else "unknown error"
        if is_no_data_message(message):
            raise NoBrokerDataError(message)
        raise RuntimeError(f"Query did not return a CSV content link. Site message: {message}")

    no_preview = [candidate for candidate in candidates if "v=t" not in candidate]
    selected = no_preview[0] if no_preview else candidates[0]
    return urljoin(MENU_URL, selected)


def submit_query(session: requests.Session, menu_html: str, stock: str, captcha_text: str) -> str:
    payload = build_query_payload(menu_html, stock, captcha_text)
    response = session.post(MENU_URL, data=payload, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    return extract_content_url(response.text, stock)


def save_csv(
    session: requests.Session,
    content_url: str,
    stock: str,
    output_dir: Path,
    query_date: str,
) -> tuple[Path, str, int]:
    response = session.get(content_url, timeout=30)
    response.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stock}_bsr_twse_{file_date(query_date)}_{timestamp()}.csv"
    csv_path.write_bytes(response.content)
    return csv_path, response.headers.get("Content-Type", ""), len(response.content)


def download_csv(
    stock: str,
    output_dir: Path,
    raw_dir: Path,
    max_attempts: int,
    query_date: str,
) -> DownloadResult:
    ensure_ocr_dependency_available()
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Referer": MENU_URL,
            }
        )

        try:
            menu_html = fetch_menu(session)
            captcha_url = extract_captcha_url(menu_html)
            captcha_path = save_captcha(session, captcha_url, raw_dir)
            captcha_text = recognize_captcha(captcha_path)
            content_url = submit_query(session, menu_html, stock, captcha_text)
            csv_path, content_type, byte_count = save_csv(
                session,
                content_url,
                stock,
                output_dir,
                query_date,
            )

            return DownloadResult(
                stock=stock,
                captcha_text=captcha_text,
                captcha_image_path=captcha_path,
                csv_path=csv_path,
                content_url=content_url,
                content_type=content_type,
                byte_count=byte_count,
                attempt=attempt,
            )
        except NoBrokerDataError:
            raise
        except Exception as exc:  # Retry because captcha recognition occasionally may fail.
            last_error = exc
            if attempt < max_attempts:
                time.sleep(0.5)

    raise RuntimeError(f"Failed after {max_attempts} attempts") from last_error


def validate_stock(value: str) -> str:
    if not re.fullmatch(r"\d{4,6}", value):
        raise argparse.ArgumentTypeError("stock must be a 4-6 digit numeric code, for example 2330")
    return value


def parse_iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def existing_csv_for_stock(output_dir: Path, stock: str, query_date: str) -> Path | None:
    candidates = sorted(output_dir.glob(f"{stock}_bsr_twse_{file_date(query_date)}_*.csv"))
    return candidates[-1] if candidates else None


def stock_codes_with_broker_files(output_dir: Path) -> set[str]:
    if not output_dir.exists():
        return set()

    codes: set[str] = set()
    patterns = [
        re.compile(r"^(?P<stock>\d{4,6})_bsr_twse_\d{8}_.*\.csv$"),
        re.compile(r"^bsr_twse_(?P<stock>\d{4,6})_.*\.csv$"),
    ]
    for path in output_dir.glob("*.csv"):
        if path.stat().st_size <= 0:
            continue
        for pattern in patterns:
            match = pattern.match(path.name)
            if match:
                codes.add(match.group("stock"))
                break
    return codes


def load_listed_common_stocks(
    metadata_path: Path,
    codes: list[str] | None = None,
    max_stocks: int | None = None,
) -> list[tuple[str, str]]:
    metadata = read_csv_canonical(metadata_path, dtype={"Code": str}).fillna("")
    required = {"Code", "Name", "Type", "Market"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_path} missing required columns: {sorted(missing)}")

    metadata["Code"] = metadata["Code"].astype(str).str.strip()
    mask = (
        metadata["Code"].str.fullmatch(r"\d{4}", na=False)
        & metadata["Type"].eq(COMMON_STOCK_TYPE)
        & metadata["Market"].eq(LISTED_MARKET)
    )
    if codes:
        requested = set(codes)
        mask = mask & metadata["Code"].isin(requested)

    selected = metadata.loc[mask, ["Code", "Name"]].drop_duplicates("Code").sort_values("Code")
    if max_stocks is not None:
        selected = selected.head(max_stocks)
    return [(str(row.Code), str(row.Name)) for row in selected.itertuples(index=False)]


def insert_column_before(df: pd.DataFrame, column: str, before: str) -> pd.DataFrame:
    if column not in df.columns:
        index = df.columns.get_loc(before) if before in df.columns else len(df.columns)
        df.insert(index, column, 0)
    return df


def update_metadata_has_broker(metadata_path: Path, output_dir: Path) -> dict[str, int]:
    metadata = read_csv_canonical(metadata_path, dtype={"Code": str}).fillna("")
    if "Code" not in metadata.columns:
        raise ValueError(f"{metadata_path} missing required column: Code")

    metadata = insert_column_before(metadata, HAS_BROKER_COLUMN, AVAILABLE_DATASET_COUNT_COLUMN)
    codes_with_data = stock_codes_with_broker_files(output_dir)
    metadata["Code"] = metadata["Code"].astype(str).str.strip()
    metadata[HAS_BROKER_COLUMN] = metadata["Code"].isin(codes_with_data).astype(int)

    if AVAILABLE_DATASET_COUNT_COLUMN in metadata.columns:
        dataset_columns = [column for column in metadata.columns if column.startswith("has_")]
        counts = metadata[dataset_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        metadata[AVAILABLE_DATASET_COUNT_COLUMN] = counts.sum(axis=1).astype(int)

    to_csv_storage(metadata, metadata_path, index=False, encoding="utf-8-sig")
    return {
        "metadata_rows": int(len(metadata)),
        "broker_codes_with_files": int(len(codes_with_data)),
        "metadata_has_broker": int(metadata[HAS_BROKER_COLUMN].sum()),
    }


def result_to_dict(result: DownloadResult) -> dict[str, Any]:
    return {
        "stock": result.stock,
        "captcha_text": result.captcha_text,
        "captcha_image_path": str(result.captcha_image_path),
        "csv_path": str(result.csv_path),
        "content_url": result.content_url,
        "content_type": result.content_type,
        "byte_count": result.byte_count,
        "attempt": result.attempt,
    }


def batch_log_paths(log_dir: Path, query_date: str) -> tuple[Path, Path]:
    run_id = timestamp()
    stem = f"broker_twse_download_{file_date(query_date)}_{run_id}"
    return log_dir / f"{stem}.csv", log_dir / f"{stem}.json"


def write_json_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def batch_record(
    stock: str,
    name: str,
    status: str,
    query_date: str,
    **values: Any,
) -> dict[str, Any]:
    record = {
        "Date": query_date,
        "Code": stock,
        "Name": name,
        "Status": status,
        "CsvPath": "",
        "CaptchaImagePath": "",
        "ContentUrl": "",
        "ContentType": "",
        "ByteCount": "",
        "Attempt": "",
        "Error": "",
        "FetchedAt": datetime.now().isoformat(timespec="seconds"),
    }
    record.update({key: "" if value is None else value for key, value in values.items()})
    return record


def print_progress(index: int, total: int, record: dict[str, Any], quiet: bool) -> None:
    if quiet and record["Status"] not in {"failed", "no_data"}:
        return
    bits = [f"[{index}/{total}]", str(record["Code"]), str(record["Status"])]
    if record.get("CsvPath"):
        bits.append(str(record["CsvPath"]))
    if record.get("Error"):
        bits.append(str(record["Error"]))
    print(" ".join(bits), flush=True)


def run_metadata_batch_summary(
    args: argparse.Namespace,
    *,
    print_summary: bool = True,
) -> dict[str, Any]:
    stocks = load_listed_common_stocks(args.metadata, args.codes, args.max_stocks)
    if not stocks:
        summary = {
            "date": args.date,
            "selected_stocks": 0,
            "status_counts": {},
            "output_dir": str(args.output_dir),
            "csv_log": "",
            "metadata_update": None,
        }
        result = {"summary": summary, "records": [], "exit_code": 1}
        if print_summary:
            print("No metadata stocks selected.", file=sys.stderr)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return result

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    needs_download = args.force or any(
        existing_csv_for_stock(args.output_dir, stock, args.date) is None
        for stock, _ in stocks
    )
    if needs_download:
        ensure_ocr_dependency_available()

    csv_log_path, json_log_path = batch_log_paths(args.log_dir, args.date)
    fieldnames = [
        "Date",
        "Code",
        "Name",
        "Status",
        "CsvPath",
        "CaptchaImagePath",
        "ContentUrl",
        "ContentType",
        "ByteCount",
        "Attempt",
        "Error",
        "FetchedAt",
    ]
    records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    print(f"Downloading TWSE BSR broker CSVs for {len(stocks)} metadata stocks into {args.output_dir}")
    print(f"Progress log: {csv_log_path}")
    with csv_log_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for index, (stock, name) in enumerate(stocks, start=1):
            existing = None if args.force else existing_csv_for_stock(args.output_dir, stock, args.date)
            if existing is not None:
                record = batch_record(
                    stock,
                    name,
                    "skipped_existing",
                    args.date,
                    CsvPath=str(existing),
                    ByteCount=existing.stat().st_size,
                )
            else:
                try:
                    result = download_csv(stock, args.output_dir, args.raw_dir, args.max_attempts, args.date)
                    record = batch_record(
                        stock,
                        name,
                        "success",
                        args.date,
                        CsvPath=str(result.csv_path),
                        CaptchaImagePath=str(result.captcha_image_path),
                        ContentUrl=result.content_url,
                        ContentType=result.content_type,
                        ByteCount=result.byte_count,
                        Attempt=result.attempt,
                    )
                except NoBrokerDataError as exc:
                    record = batch_record(stock, name, "no_data", args.date, Error=str(exc))
                except Exception as exc:
                    record = batch_record(stock, name, "failed", args.date, Error=str(exc))

            records.append(record)
            status = str(record["Status"])
            status_counts[status] = status_counts.get(status, 0) + 1
            writer.writerow(record)
            f.flush()
            print_progress(index, len(stocks), record, args.quiet)

            if index < len(stocks) and record["Status"] != "skipped_existing":
                time.sleep(random.uniform(args.throttle_min, args.throttle_max))

    metadata_update = None
    if args.update_metadata:
        metadata_update = update_metadata_has_broker(args.metadata, args.output_dir)

    summary = {
        "date": args.date,
        "selected_stocks": len(stocks),
        "status_counts": status_counts,
        "output_dir": str(args.output_dir),
        "csv_log": str(csv_log_path),
        "metadata_update": metadata_update,
    }
    write_json_log(json_log_path, {"summary": summary, "records": records})
    if print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    exit_code = 0 if status_counts.get("failed", 0) == 0 else 2
    return {"summary": summary, "records": records, "exit_code": exit_code}


def run_metadata_batch(args: argparse.Namespace) -> int:
    try:
        return int(run_metadata_batch_summary(args)["exit_code"])
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TWSE BSR broker trading CSV with PaddleOCR captcha solving.")
    parser.add_argument("stock", nargs="?", type=validate_stock, help="stock code, default: 2330 unless --all-metadata is set")
    parser.add_argument("--all-metadata", action="store_true", help="download all TWSE listed common stocks from data/metadata.csv")
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH, type=Path, help="metadata CSV path")
    parser.add_argument("--codes", nargs="+", type=validate_stock, help="optional code subset for --all-metadata")
    parser.add_argument("--max-stocks", type=int, default=None, help="optional test limit for --all-metadata")
    parser.add_argument("--date", default=today_text(), type=parse_iso_date, help="YYYY-MM-DD label for output filenames; default: today")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, type=Path)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR, type=Path)
    parser.add_argument("--max-attempts", default=3, type=int)
    parser.add_argument("--throttle-min", default=0.2, type=float)
    parser.add_argument("--throttle-max", default=0.8, type=float)
    parser.add_argument("--force", action="store_true", help="download even when this stock/date already has a CSV")
    parser.add_argument("--update-metadata", dest="update_metadata", action="store_true", default=None)
    parser.add_argument("--no-update-metadata", dest="update_metadata", action="store_false")
    parser.add_argument("--quiet", action="store_true", help="reduce batch progress output")
    parser.add_argument("--json", action="store_true", help="print machine-readable result JSON")
    args = parser.parse_args()
    if args.update_metadata is None:
        args.update_metadata = args.all_metadata
    if args.throttle_min < 0 or args.throttle_max < 0 or args.throttle_min > args.throttle_max:
        parser.error("--throttle-min/--throttle-max must be non-negative and min <= max")
    return args


def main() -> int:
    args = parse_args()
    if args.all_metadata:
        return run_metadata_batch(args)

    stock = args.stock or "2330"
    existing = None if args.force else existing_csv_for_stock(args.output_dir, stock, args.date)
    if existing is not None:
        data = {
            "stock": stock,
            "status": "skipped_existing",
            "csv_path": str(existing),
            "byte_count": existing.stat().st_size,
        }
        if args.json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(f"stock: {stock}")
            print("status: skipped_existing")
            print(f"csv: {existing}")
            print(f"bytes: {existing.stat().st_size}")
        if args.update_metadata:
            print(json.dumps(update_metadata_has_broker(args.metadata, args.output_dir), ensure_ascii=False, indent=2))
        return 0

    try:
        result = download_csv(stock, args.output_dir, args.raw_dir, args.max_attempts, args.date)
    except NoBrokerDataError as exc:
        print(f"NO_DATA: {exc}", file=sys.stderr)
        if args.update_metadata:
            print(json.dumps(update_metadata_has_broker(args.metadata, args.output_dir), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    data = result_to_dict(result)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(f"stock: {result.stock}")
        print(f"captcha: {result.captcha_text}")
        print(f"captcha_image: {result.captcha_image_path}")
        print(f"csv: {result.csv_path}")
        print(f"bytes: {result.byte_count}")
        print(f"content_type: {result.content_type}")
        print(f"attempt: {result.attempt}")
    if args.update_metadata:
        print(json.dumps(update_metadata_has_broker(args.metadata, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
