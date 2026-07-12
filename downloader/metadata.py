"""Build and refresh the canonical TWSE security catalog.

The full mode rebuilds stock, ETF, and index identity fields from official
TWSE sources. ``--availability-only`` leaves identity fields untouched and
recomputes local dataset flags from the current flat per-security layout.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from lxml import html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import canonical_name, read_csv_canonical, to_csv_storage  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_ROOT / "metadata.csv"

LISTED_MARKET = "上市"
COMMON_STOCK_TYPE = "股票"
ETF_TYPE = "ETF"
INDEX_TYPE = "INDEX"
GENERAL_BOARD = "一般"
INNOVATION_BOARD = "創新板"

TWSE_COMPANY_BASIC_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TWSE_ETF_LIST_URL = "https://www.twse.com.tw/rwd/zh/ETF/list"
TWSE_ISIN_URL = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
TWSE_INDEX_SERIES_URL = "https://www.twse.com.tw/zh/indices/indices/series.html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 stock_analyse/1.0",
    "Accept": "application/json, text/html, */*",
}

IDENTITY_COLUMNS = [
    "Code",
    "Name",
    "Type",
    "Market",
    "Group",
    "ISIN",
    "Start",
    "CFI",
    "Board",
    "CompanyName",
]
AVAILABILITY_COLUMNS = [
    "has_price",
    "has_adj_price",
    "has_institutional",
    "has_margin",
    "has_day_trading",
    "has_yield_pe_pb",
    "has_report",
    "has_broker",
]
ADJUSTED_PRICE_COLUMNS = ["open_adj", "close_adj", "high_adj", "low_adj", "AdjFactor"]
FLAT_DATASET_DIRS = {
    "has_institutional": DATA_ROOT / "institutional",
    "has_margin": DATA_ROOT / "margin",
    "has_day_trading": DATA_ROOT / "day_trading",
    "has_yield_pe_pb": DATA_ROOT / "yield_pe_pb",
    "has_report": DATA_ROOT / "report",
}
BROKER_DIR = DATA_ROOT / "broker" / "twse" / "by_stock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--availability-only",
        action="store_true",
        help="Refresh local dataset flags without querying identity sources.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def normalize_gregorian_date(value: str) -> str:
    text = str(value or "").strip().replace(".", "/")
    for pattern, has_day in (
        (r"(\d{4})/(\d{1,2})/(\d{1,2})", True),
        (r"(\d{2,3})/(\d{1,2})/(\d{1,2})", True),
        (r"(\d{4})/(\d{1,2})", False),
        (r"(\d{2,3})/(\d{1,2})", False),
    ):
        match = re.fullmatch(pattern, text)
        if not match:
            continue
        parts = [int(part) for part in match.groups()]
        if parts[0] < 1911:
            parts[0] += 1911
        if has_day:
            return f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
        return f"{parts[0]:04d}-{parts[1]:02d}"
    return text


def parse_listing_date(value: str) -> date:
    normalized = normalize_gregorian_date(value)
    return datetime.strptime(normalized, "%Y-%m-%d").date()


def split_code_name(value: str) -> tuple[str, str]:
    match = re.match(r"^([0-9A-Z]+)[\s　]*(.*)$", str(value).strip())
    return (match.group(1), match.group(2).strip()) if match else ("", "")


def split_twse_multiline(value: str) -> list[str]:
    return [part.strip() for part in str(value).split("<br>")]


def clean_suffix(value: str) -> str:
    return re.sub(r"\(.*?\)", "", str(value)).strip()


def get_json(url: str) -> object:
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_company_basic_map() -> dict[str, dict[str, str]]:
    rows = get_json(TWSE_COMPANY_BASIC_URL)
    return {str(row["公司代號"]): row for row in rows}


def fetch_etf_map() -> dict[str, dict[str, str]]:
    data = get_json(TWSE_ETF_LIST_URL)["data"]
    rows: dict[str, dict[str, str]] = {}
    for listing_dates, codes, names, issuer, index_name in data:
        for listing_date, code_part, name_part in zip(
            split_twse_multiline(listing_dates),
            split_twse_multiline(codes),
            split_twse_multiline(names),
        ):
            code = clean_suffix(code_part)
            rows[code] = {
                "Name": clean_suffix(name_part),
                "Issuer": issuer or "",
                "UnderlyingIndex": index_name or "",
                "Start": normalize_gregorian_date(listing_date),
            }
    return rows


def fetch_isin_rows() -> list[dict[str, str]]:
    response = requests.get(TWSE_ISIN_URL, headers=HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = "cp950"
    document = html.fromstring(response.text)
    rows: list[dict[str, str]] = []
    section = ""
    for row in document.xpath("//tr"):
        cells = [
            " ".join(text.strip() for text in cell.xpath(".//text()") if text.strip())
            for cell in row.xpath("./th|./td")
        ]
        if not cells:
            continue
        if len(cells) == 1:
            section = cells[0]
            continue
        if len(cells) < 6:
            continue
        code, name = split_code_name(cells[0])
        if code:
            rows.append(
                {
                    "Section": section,
                    "Code": code,
                    "Name": name,
                    "ISIN": cells[1],
                    "Start": normalize_gregorian_date(cells[2]),
                    "Market": cells[3],
                    "Group": cells[4],
                    "CFI": cells[5],
                }
            )
    return rows


def normalize_twse_url(url: str) -> str:
    if url.startswith("//"):
        return f"https:{url}"
    if url.startswith("/"):
        return f"https://www.twse.com.tw{url}"
    return url


def build_index_code(name: str, info_url: str) -> str:
    if name == "發行量加權股價指數":
        return "TAIEX"
    slug = normalize_twse_url(info_url).rstrip("/").split("/")[-1]
    if slug and slug.lower() not in {"multiple", "series"} and re.fullmatch(r"[A-Za-z0-9]+", slug):
        return slug.upper()
    return name


def fetch_index_rows() -> list[dict[str, str]]:
    table = pd.read_html(TWSE_INDEX_SERIES_URL, flavor="lxml")[0].fillna("")
    response = requests.get(TWSE_INDEX_SERIES_URL, headers=HEADERS, timeout=60)
    response.raise_for_status()
    response.encoding = "utf-8"
    document = html.fromstring(response.text)
    link_map: dict[str, str] = {}
    for row in document.xpath("//tr")[1:]:
        anchors = row.xpath(".//a")
        if anchors:
            name = re.sub(r"\s+", "", " ".join(anchors[0].xpath(".//text()")))
            link_map[name] = normalize_twse_url(anchors[0].get("href", ""))

    rows = []
    for record in table.to_dict(orient="records"):
        name = str(record.get("指數名稱", "")).replace("\n", "").strip()
        if not name:
            continue
        info_url = link_map.get(re.sub(r"\s+", "", name), "")
        rows.append(
            {
                "Code": build_index_code(name, info_url),
                "Name": name,
                "Group": str(record.get("類型.1", "")).strip(),
                "Start": normalize_gregorian_date(str(record.get("發布日", "")).strip()),
            }
        )
    return rows


def build_identity_catalog() -> pd.DataFrame:
    today = date.today()
    company_basic = fetch_company_basic_map()
    etfs = fetch_etf_map()
    rows: list[dict[str, str]] = []
    for source in sorted(fetch_isin_rows(), key=lambda item: item["Code"]):
        if not source["Market"].startswith(LISTED_MARKET):
            continue
        if source["Section"] not in {COMMON_STOCK_TYPE, INNOVATION_BOARD, ETF_TYPE}:
            continue
        if parse_listing_date(source["Start"]) > today:
            continue
        code = source["Code"]
        is_etf = source["Section"] == ETF_TYPE
        basic = company_basic.get(code, {})
        etf = etfs.get(code, {})
        rows.append(
            {
                "Code": code,
                "Name": etf.get("Name", source["Name"]) if is_etf else basic.get("公司簡稱", source["Name"]),
                "Type": ETF_TYPE if is_etf else COMMON_STOCK_TYPE,
                "Market": LISTED_MARKET,
                "Group": "ETF" if is_etf else source["Group"],
                "ISIN": source["ISIN"],
                "Start": source["Start"],
                "CFI": source["CFI"],
                "Board": ETF_TYPE if is_etf else (INNOVATION_BOARD if source["Section"] == INNOVATION_BOARD else GENERAL_BOARD),
                "CompanyName": "" if is_etf else basic.get("公司名稱", ""),
            }
        )
    for source in sorted(fetch_index_rows(), key=lambda item: item["Code"]):
        rows.append(
            {
                "Code": source["Code"],
                "Name": source["Name"],
                "Type": INDEX_TYPE,
                "Market": INDEX_TYPE,
                "Group": source["Group"],
                "ISIN": "",
                "Start": source["Start"],
                "CFI": "",
                "Board": INDEX_TYPE,
                "CompanyName": "",
            }
        )
    catalog = pd.DataFrame(rows, columns=IDENTITY_COLUMNS)
    if catalog["Code"].duplicated().any():
        duplicates = sorted(catalog.loc[catalog["Code"].duplicated(False), "Code"].unique())
        raise ValueError(f"Duplicate metadata codes from TWSE sources: {duplicates}")
    return catalog


def has_data_row(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            nonblank = (line for line in handle if line.strip())
            next(nonblank)
            next(nonblank)
        return True
    except (OSError, StopIteration):
        return False


def files_by_catalog_code(folder: Path, codes: set[str]) -> dict[str, list[Path]]:
    matches: dict[str, list[Path]] = {}
    if not folder.exists():
        return matches
    for path in folder.glob("*.csv"):
        code = path.stem.split("_", 1)[0]
        if code in codes and has_data_row(path):
            matches.setdefault(code, []).append(path)
    return matches


def has_adjusted_prices(paths: list[Path]) -> bool:
    for path in paths:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                canonical_fields = {
                    field: canonical_name(field)
                    for field in (reader.fieldnames or [])
                }
                if not set(ADJUSTED_PRICE_COLUMNS).issubset(canonical_fields.values()):
                    continue
                for row in reader:
                    canonical_row = {
                        canonical_fields[field]: value
                        for field, value in row.items()
                    }
                    if all(
                        str(canonical_row.get(column, "")).strip()
                        for column in ADJUSTED_PRICE_COLUMNS
                    ):
                        return True
        except (OSError, UnicodeError, csv.Error):
            continue
    return False


def refresh_availability(catalog: pd.DataFrame, data_root: Path = DATA_ROOT) -> pd.DataFrame:
    required = set(IDENTITY_COLUMNS)
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    result = catalog.copy()
    result["Code"] = result["Code"].astype(str)
    if result["Code"].duplicated().any():
        raise ValueError("Metadata contains duplicate Code values.")
    codes = set(result["Code"])

    price_files = files_by_catalog_code(data_root / "price", codes)
    folder_files = {
        flag: files_by_catalog_code(data_root / relative.relative_to(DATA_ROOT), codes)
        for flag, relative in FLAT_DATASET_DIRS.items()
    }
    broker_files = files_by_catalog_code(data_root / BROKER_DIR.relative_to(DATA_ROOT), codes)

    result["has_price"] = result["Code"].map(lambda code: int(code in price_files))
    result["has_adj_price"] = result["Code"].map(
        lambda code: int(has_adjusted_prices(price_files.get(code, [])))
    )
    for flag, matches in folder_files.items():
        result[flag] = result["Code"].map(lambda code, matches=matches: int(code in matches))
    result["has_broker"] = result["Code"].map(lambda code: int(code in broker_files))
    result["available_dataset_count"] = result[AVAILABILITY_COLUMNS].sum(axis=1).astype(int)
    return result[IDENTITY_COLUMNS + AVAILABILITY_COLUMNS + ["available_dataset_count"]]


def write_catalog(catalog: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_csv_storage(catalog, output_path, index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    if args.availability_only:
        if not args.output.exists():
            raise FileNotFoundError(f"Metadata catalog does not exist: {args.output}")
        catalog = read_csv_canonical(args.output, dtype={"Code": str}).fillna("")
    else:
        catalog = build_identity_catalog()
    catalog = refresh_availability(catalog, DATA_ROOT)
    write_catalog(catalog, args.output)
    print(f"Wrote {len(catalog)} securities to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
