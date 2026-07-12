from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BROKER_DIR = PROJECT_ROOT / "data" / "broker" / "by_broker"
PRICE_DIR = PROJECT_ROOT / "data" / "price"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
FOREIGN_BRANCH_SUMMARY = PROJECT_ROOT / "output" / "broker" / "foreign_branch_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "broker"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "broker"
LOG_DIR = PROJECT_ROOT / "logs" / "broker"
BROKER_LOCATION_CACHE = OUTPUT_DIR / "twse_broker_branch_locations.csv"
BROKER_CHURN_METRICS = OUTPUT_DIR / "broker_trading_style_churn_metrics.csv"
TWSE_BROKER_SERVICE_API = "https://www.twse.com.tw/rwd/zh/brokerService/brokerServiceAudit"

HORIZONS = (1, 3, 5, 10, 20)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOURCE_URL_DATE_RE = re.compile(r"[?&]e=(\d{4})-(\d{1,2})-(\d{1,2})(?:&|$)")
MIN_REASONABLE_DATE = "2000-01-01"
OTHER_BRANCH_CATEGORY = "其他分點"
MERGED_OTHER_CATEGORY_KEYWORDS = ("網路", "電子", "法人", "營業部")
CITY_EXCLUDED_CATEGORIES = {"外資分點（名稱推定）", "總公司/主分點或總號", "停業/舊分點"}
TAIWAN_CITIES = (
    "台北市",
    "新北市",
    "桃園市",
    "台中市",
    "台南市",
    "高雄市",
    "基隆市",
    "新竹市",
    "嘉義市",
    "新竹縣",
    "苗栗縣",
    "彰化縣",
    "南投縣",
    "雲林縣",
    "嘉義縣",
    "屏東縣",
    "宜蘭縣",
    "花蓮縣",
    "台東縣",
    "澎湖縣",
    "金門縣",
    "連江縣",
)
LEGACY_CITY_ALIASES = {
    "台北縣": "新北市",
    "臺北縣": "新北市",
    "台中縣": "台中市",
    "臺中縣": "台中市",
    "台南縣": "台南市",
    "臺南縣": "台南市",
    "高雄縣": "高雄市",
    "宜蘭市": "宜蘭縣",
    "花蓮市": "花蓮縣",
    "台東市": "台東縣",
    "臺東市": "台東縣",
    "苗栗市": "苗栗縣",
    "彰化市": "彰化縣",
    "南投市": "南投縣",
    "屏東市": "屏東縣",
}
BRANCH_CITY_KEYWORDS = {
    "台北市": ("台北", "臺北", "天母", "士林", "大安", "信義", "松山", "城中", "南京", "敦北", "敦南", "古亭", "民生", "忠孝", "內湖", "東湖", "西松", "大直", "復興", "萬華", "館前", "站前", "世貿", "仁愛", "大同"),
    "新北市": ("新北", "板橋", "新店", "新莊", "三重", "中和", "永和", "土城", "樹林", "汐止", "淡水", "林口", "蘆洲", "五股", "三峽", "雙和", "北新莊", "丹鳳"),
    "桃園市": ("桃園", "中壢", "北中壢", "內壢", "八德", "平鎮", "大園", "桃鶯", "桃盛"),
    "台中市": ("台中港", "台中", "臺中", "豐原", "烏日", "沙鹿", "大里", "潭子", "文心", "崇德", "市政", "中港"),
    "台南市": ("台南", "臺南", "新營", "麻豆", "佳里", "府城", "安南", "永康", "開元", "北門"),
    "高雄市": ("高雄", "岡山", "鳳山", "三民", "左楠", "大昌", "北高雄", "小港", "三多", "苓雅", "瑞豐", "高美館"),
    "基隆市": ("基隆",),
    "新竹市": ("新竹", "竹科", "科園"),
    "嘉義市": ("嘉義",),
    "新竹縣": ("竹北", "竹東", "湖口", "新豐"),
    "苗栗縣": ("苗栗", "頭份", "竹南"),
    "彰化縣": ("彰化", "員林", "鹿港"),
    "南投縣": ("南投", "埔里", "草屯"),
    "雲林縣": ("雲林", "斗六", "虎尾"),
    "嘉義縣": ("朴子", "民雄"),
    "屏東縣": ("屏東", "屏新", "東港", "潮州"),
    "宜蘭縣": ("宜蘭", "羅東"),
    "花蓮縣": ("花蓮",),
    "台東縣": ("台東", "臺東", "東昇"),
    "澎湖縣": ("澎湖",),
    "金門縣": ("金門",),
    "連江縣": ("馬祖", "連江"),
}


@dataclass
class Agg:
    count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    sum_return: float = 0.0
    weighted_sum_return: float = 0.0
    weight_sum: float = 0.0

    def add(self, decision_return: float, weight: float, direction: int) -> None:
        self.count += 1
        if direction > 0:
            self.buy_count += 1
        else:
            self.sell_count += 1
        if decision_return > 0:
            self.win_count += 1
        elif decision_return < 0:
            self.loss_count += 1
        self.sum_return += decision_return
        self.weighted_sum_return += decision_return * weight
        self.weight_sum += weight

    @property
    def avg_return(self) -> float:
        return self.sum_return / self.count if self.count else 0.0

    @property
    def weighted_return(self) -> float:
        return self.weighted_sum_return / self.weight_sum if self.weight_sum else 0.0

    @property
    def win_rate(self) -> float:
        return self.win_count / self.count if self.count else 0.0


@dataclass
class PriceSeries:
    dates: list[str]
    adj_close: list[float]
    close: list[float]
    date_to_index: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-branch broker decision performance report from Fubon rank data."
    )
    parser.add_argument("--broker-dir", type=Path, default=BROKER_DIR)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--foreign-branch-summary", type=Path, default=FOREIGN_BRANCH_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--broker-location-cache", type=Path, default=BROKER_LOCATION_CACHE)
    parser.add_argument(
        "--min-branch-events",
        type=int,
        default=1000,
        help="Exclude branches with fewer eligible deduplicated branch-date-stock events.",
    )
    parser.add_argument("--min-stock-events", type=int, default=50)
    return parser.parse_args()


def parse_int(value: str | None) -> int:
    if not value:
        return 0
    text = value.replace(",", "").strip()
    if not text or text == "-":
        return 0
    return int(float(text))


def parse_float(value: str | None) -> float:
    if not value:
        return math.nan
    text = value.replace(",", "").strip()
    if not text or text == "-":
        return math.nan
    return float(text)


def parse_source_url_date(source_url: str | None) -> str | None:
    if not source_url:
        return None
    match = SOURCE_URL_DATE_RE.search(source_url)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_date(raw_date: str, source_url: str | None) -> tuple[str | None, str | None]:
    if DATE_RE.match(raw_date) and raw_date >= MIN_REASONABLE_DATE:
        return raw_date, None
    source_date = parse_source_url_date(source_url)
    if source_date:
        return source_date, "來源網址日期修正"
    if DATE_RE.match(raw_date):
        return None, "日期早於合理範圍"
    return None, "日期格式異常"


def code_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def load_metadata(path: Path) -> tuple[set[str], dict[str, dict[str, str]]]:
    listed_common: set[str] = set()
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row.get("Code", "")
            if not code:
                continue
            metadata[code] = row
            if row.get("類型") == "股票" and row.get("市場") == "上市":
                listed_common.add(code)
    return listed_common, metadata


def load_foreign_branch_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row.get("分點名稱", "") for row in csv.DictReader(handle) if row.get("分點名稱")}


def normalize_branch_key(branch_name: str) -> str:
    return re.sub(r"\s+", "", branch_name).replace("臺", "台").strip()


def normalize_city_name(address: str) -> str:
    normalized = address.replace("臺", "台")
    for keyword, city in LEGACY_CITY_ALIASES.items():
        if keyword.replace("臺", "台") in normalized:
            return city
    for city in TAIWAN_CITIES:
        if city in normalized:
            return city
    return ""


def infer_city_from_branch_name(branch_name: str) -> str:
    normalized = normalize_branch_key(branch_name)
    suffix = normalized.split("-", 1)[1] if "-" in normalized else normalized
    candidates: list[tuple[int, str, str]] = []
    for city, keywords in BRANCH_CITY_KEYWORDS.items():
        for keyword in keywords:
            normalized_keyword = keyword.replace("臺", "台")
            if normalized_keyword and normalized_keyword in suffix:
                candidates.append((len(normalized_keyword), normalized_keyword, city))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    return candidates[0][2]


def twse_broker_service(**params: str) -> dict[str, object]:
    query = dict(params)
    query["response"] = "json"
    url = f"{TWSE_BROKER_SERVICE_API}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code not in {301, 302, 303, 307, 308}:
            raise
        location = exc.headers.get("Location")
        if not location:
            raise
        redirect_url = urllib.parse.urljoin(url, location)
        redirect_request = urllib.request.Request(
            redirect_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json,text/plain,*/*",
            },
        )
        with urllib.request.urlopen(redirect_request, timeout=30) as response:
            return json.loads(response.read())


def fetch_twse_broker_branch_locations() -> list[dict[str, str]]:
    main = twse_broker_service(showType="main")
    rows: list[dict[str, str]] = []
    for main_row in main.get("data", []):
        if len(main_row) < 2:
            continue
        head_code = str(main_row[0])
        head_name = str(main_row[1])
        detail = twse_broker_service(showType="list", stkNo=head_code)
        for row in detail.get("data", []):
            if len(row) < 5:
                continue
            branch_name = str(row[1])
            address = str(row[3])
            rows.append(
                {
                    "總公司代號": head_code,
                    "總公司名稱": head_name,
                    "分點代號": str(row[0]),
                    "分點名稱": branch_name,
                    "縣市": normalize_city_name(address),
                    "開業日": str(row[2]),
                    "地址": address,
                    "電話": str(row[4]),
                    "來源": TWSE_BROKER_SERVICE_API,
                }
            )
        time.sleep(0.02)
    return rows


def load_broker_branch_locations(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, object] = {"source": "twse_api", "path": str(path)}
    try:
        rows = fetch_twse_broker_branch_locations()
        if not rows:
            raise RuntimeError("TWSE broker branch location API returned no rows")
        write_csv(
            path,
            ["總公司代號", "總公司名稱", "分點代號", "分點名稱", "縣市", "開業日", "地址", "電話", "來源"],
            rows,
        )
        stats["fetched_rows"] = len(rows)
    except Exception as exc:
        stats["source"] = "cache"
        stats["fetch_error"] = f"{type(exc).__name__}: {exc}"
        if not path.exists():
            raise
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        stats["cached_rows"] = len(rows)
    for row in rows:
        row["縣市"] = row.get("縣市") or normalize_city_name(row.get("地址", ""))
    by_name = {normalize_branch_key(row.get("分點名稱", "")): row for row in rows if row.get("分點名稱")}
    stats["mapped_branch_names"] = len(by_name)
    stats["mapped_cities"] = len({row.get("縣市", "") for row in rows if row.get("縣市")})
    return by_name, stats


def load_price_paths(price_dir: Path) -> dict[str, Path]:
    return {
        code_from_path(path): path
        for path in sorted(price_dir.glob("*.csv"))
        if not path.name.startswith("twse_")
    }


def load_price_series(path: Path) -> PriceSeries:
    dates: list[str] = []
    adj_close: list[float] = []
    close: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date = (row.get("Date") or "").strip()
            if not date:
                continue
            close_value = parse_float(row.get("收盤價") or row.get("Close"))
            adj_value = parse_float(row.get("前復權收盤價") or row.get("close_adj"))
            if not math.isfinite(adj_value):
                adj_value = close_value
            if not math.isfinite(close_value):
                close_value = adj_value
            if not math.isfinite(adj_value) or adj_value <= 0:
                continue
            dates.append(date)
            adj_close.append(adj_value)
            close.append(close_value)
    return PriceSeries(dates, adj_close, close, {date: index for index, date in enumerate(dates)})


def branch_category(branch_name: str, foreign_branches: set[str]) -> str:
    if branch_name in foreign_branches:
        return "外資分點（名稱推定）"
    if any(keyword in branch_name for keyword in MERGED_OTHER_CATEGORY_KEYWORDS):
        return OTHER_BRANCH_CATEGORY
    if "停" in branch_name:
        return "停業/舊分點"
    if "-" not in branch_name:
        return "總公司/主分點或總號"
    return "一般分公司/地方分點"


def pct(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value * 100:.{digits}f}%"


def fmt_int(value: int | float) -> str:
    return f"{int(round(float(value))):,}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_churn_metrics(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    metrics: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            branch_name = row.get("分點名稱", "")
            if not branch_name:
                continue
            metrics[branch_name] = {
                "雙向換手率": parse_float(row.get("雙向換手率")),
                "買賣皆活躍事件率": parse_float(row.get("買賣皆活躍事件率")),
                "淨方向集中率": parse_float(row.get("淨方向集中率")),
                "平均每事件成交張數": parse_float(row.get("平均每事件成交張數")),
                "買賣合計張數": parse_int(row.get("買賣合計張數")),
                "雙向換手張數": parse_int(row.get("雙向換手張數")),
                "去重事件數": parse_int(row.get("去重事件數")),
            }
    return metrics


def agg_row(group: dict[str, object], agg: Agg) -> dict[str, object]:
    return {
        **group,
        "事件數": agg.count,
        "買超事件數": agg.buy_count,
        "賣超事件數": agg.sell_count,
        "正報酬事件數": agg.win_count,
        "負報酬事件數": agg.loss_count,
        "命中率": pct(agg.win_rate),
        "平均決策後報酬": pct(agg.avg_return),
        "淨買賣超金額權重報酬": pct(agg.weighted_return),
        "權重合計": agg.weight_sum,
    }


def add_horizon_event(
    event_date: str,
    code: str,
    branch_name: str,
    category: str,
    city: str | None,
    industry: str,
    net: int,
    price_paths: dict[str, Path],
    price_cache: dict[str, PriceSeries],
    overall: dict[int, Agg],
    by_category: dict[tuple[str, int], Agg],
    by_branch: dict[tuple[str, int], Agg],
    by_branch_side: dict[tuple[str, str, int], Agg],
    by_city: dict[tuple[str, int], Agg],
    by_industry: dict[tuple[str, int], Agg],
    by_stock20: dict[tuple[str, str], Agg],
    evaluation_stats: dict[str, int],
) -> None:
    price = price_cache.get(code)
    if price is None:
        price_path = price_paths.get(code)
        if not price_path:
            evaluation_stats["missing_price_file_events"] += 1
            return
        price = load_price_series(price_path)
        price_cache[code] = price
    if not price.dates:
        evaluation_stats["empty_price_file_events"] += 1
        return
    index = price.date_to_index.get(event_date)
    if index is None:
        evaluation_stats["missing_price_date_events"] += 1
        return
    entry_adj = price.adj_close[index]
    entry_close = price.close[index]
    if not math.isfinite(entry_adj) or entry_adj <= 0 or not math.isfinite(entry_close):
        evaluation_stats["invalid_entry_price_events"] += 1
        return
    direction = 1 if net > 0 else -1
    side = "買超" if direction > 0 else "賣超"
    weight = abs(net) * entry_close
    for horizon in HORIZONS:
        future_index = index + horizon
        if future_index >= len(price.adj_close):
            evaluation_stats[f"missing_future_{horizon}d"] += 1
            continue
        future_adj = price.adj_close[future_index]
        if not math.isfinite(future_adj) or future_adj <= 0:
            evaluation_stats[f"invalid_future_{horizon}d"] += 1
            continue
        raw_return = future_adj / entry_adj - 1.0
        decision_return = raw_return * direction
        overall[horizon].add(decision_return, weight, direction)
        by_category[(category, horizon)].add(decision_return, weight, direction)
        by_branch[(branch_name, horizon)].add(decision_return, weight, direction)
        by_branch_side[(branch_name, side, horizon)].add(decision_return, weight, direction)
        if city:
            by_city[(city, horizon)].add(decision_return, weight, direction)
        by_industry[(industry, horizon)].add(decision_return, weight, direction)
        if horizon == 20:
            by_stock20[(branch_name, code)].add(decision_return, weight, direction)
        evaluation_stats["evaluated_event_horizons"] += 1


def build_outputs(args: argparse.Namespace) -> dict[str, object]:
    listed_common, metadata = load_metadata(args.metadata)
    foreign_branches = load_foreign_branch_names(args.foreign_branch_summary)
    branch_locations, branch_location_stats = load_broker_branch_locations(args.broker_location_cache)
    price_paths = load_price_paths(args.price_dir)
    branch_files = sorted(args.broker_dir.glob("*.csv"))

    price_cache: dict[str, PriceSeries] = {}
    overall: dict[int, Agg] = defaultdict(Agg)
    by_category: dict[tuple[str, int], Agg] = defaultdict(Agg)
    by_branch: dict[tuple[str, int], Agg] = defaultdict(Agg)
    by_branch_side: dict[tuple[str, str, int], Agg] = defaultdict(Agg)
    by_city: dict[tuple[str, int], Agg] = defaultdict(Agg)
    by_industry: dict[tuple[str, int], Agg] = defaultdict(Agg)
    by_stock20: dict[tuple[str, str], Agg] = defaultdict(Agg)
    city_branch_names: dict[str, set[str]] = defaultdict(set)
    stock_names: dict[str, str] = {}

    load_stats = defaultdict(int)
    evaluation_stats = defaultdict(int)
    skipped_rows: list[dict[str, object]] = []
    corrected_dates: list[dict[str, object]] = []
    excluded_small_branches: list[dict[str, object]] = []
    city_branch_rows: list[dict[str, object]] = []
    city_excluded_branches: list[dict[str, object]] = []

    for file_index, path in enumerate(branch_files, 1):
        branch_name = path.stem
        category = branch_category(branch_name, foreign_branches)
        seen_in_branch: set[tuple[str, str]] = set()
        candidate_events: list[tuple[str, str, str, str, int]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                continue
            index = {name: i for i, name in enumerate(header)}
            required = ["Date", "Code", "Name", "買進", "賣出", "買賣超", "來源網址"]
            missing = [name for name in required if name not in index]
            if missing:
                load_stats["files_with_missing_columns"] += 1
                continue
            for row_number, row in enumerate(reader, start=2):
                load_stats["raw_rows"] += 1
                if len(row) <= max(index.values()):
                    load_stats["skipped_short_rows"] += 1
                    continue
                raw_date = row[index["Date"]].strip()
                source_url = row[index["來源網址"]].strip()
                date, date_note = normalize_date(raw_date, source_url)
                code = row[index["Code"]].strip()
                if not date or not code:
                    load_stats["skipped_bad_date_or_code"] += 1
                    if len(skipped_rows) < 1000:
                        skipped_rows.append(
                            {
                                "檔案": str(path.relative_to(PROJECT_ROOT)),
                                "列號": row_number,
                                "原因": date_note or "日期或股票代號異常",
                                "Date": raw_date,
                                "Code": code,
                            }
                        )
                    continue
                if date_note and len(corrected_dates) < 1000:
                    corrected_dates.append(
                        {
                            "檔案": str(path.relative_to(PROJECT_ROOT)),
                            "列號": row_number,
                            "原始Date": raw_date,
                            "修正Date": date,
                            "原因": date_note,
                            "來源網址": source_url,
                        }
                    )
                if code not in listed_common:
                    load_stats["skipped_non_listed_common"] += 1
                    continue
                try:
                    net = parse_int(row[index["買賣超"]])
                except ValueError:
                    load_stats["skipped_bad_number"] += 1
                    if len(skipped_rows) < 1000:
                        skipped_rows.append(
                            {
                                "檔案": str(path.relative_to(PROJECT_ROOT)),
                                "列號": row_number,
                                "原因": "買賣超欄位非數字",
                                "Date": raw_date,
                                "Code": code,
                            }
                        )
                    continue
                if net == 0:
                    load_stats["skipped_zero_net"] += 1
                    continue
                event_key = (date, code)
                if event_key in seen_in_branch:
                    load_stats["deduplicated_branch_date_stock"] += 1
                    continue
                seen_in_branch.add(event_key)

                stock_name = metadata.get(code, {}).get("Name", "") or row[index["Name"]].strip()
                stock_names[code] = stock_name
                industry = metadata.get(code, {}).get("產業群組", "未分類") or "未分類"
                load_stats["candidate_events"] += 1
                candidate_events.append((date, code, stock_name, industry, net))

        if candidate_events and len(candidate_events) < args.min_branch_events:
            load_stats["excluded_small_branch_files"] += 1
            load_stats["excluded_small_branch_events"] += len(candidate_events)
            excluded_small_branches.append(
                {
                    "分點名稱": branch_name,
                    "分點類別": category,
                    "可用事件數": len(candidate_events),
                    "事件數門檻": args.min_branch_events,
                    "來源檔案": str(path.relative_to(PROJECT_ROOT)),
                }
            )
        else:
            load_stats["included_branch_files"] += 1
            load_stats["events"] += len(candidate_events)
            location = branch_locations.get(normalize_branch_key(branch_name))
            city = ""
            city_source = ""
            city_exclusion_reason = ""
            if category in CITY_EXCLUDED_CATEGORIES:
                city_exclusion_reason = f"分類排除：{category}"
            else:
                if location and location.get("縣市"):
                    city = location["縣市"]
                    city_source = "TWSE地址"
                else:
                    city = infer_city_from_branch_name(branch_name)
                    if city:
                        city_source = "分點名稱推定"
                    elif not location:
                        city_exclusion_reason = "TWSE現行分公司名冊查無分點且分點名稱無法推定縣市"
                    else:
                        city_exclusion_reason = "TWSE地址無法解析縣市且分點名稱無法推定縣市"
            if city:
                location_row = location or {}
                city_branch_names[city].add(branch_name)
                city_branch_rows.append(
                    {
                        "縣市": city,
                        "縣市來源": city_source,
                        "分點名稱": branch_name,
                        "分點類別": category,
                        "可用事件數": len(candidate_events),
                        "TWSE分點代號": location_row.get("分點代號", ""),
                        "TWSE總公司代號": location_row.get("總公司代號", ""),
                        "TWSE總公司名稱": location_row.get("總公司名稱", ""),
                        "TWSE開業日": location_row.get("開業日", ""),
                        "TWSE地址": location_row.get("地址", ""),
                        "TWSE電話": location_row.get("電話", ""),
                    }
                )
            if not city:
                city_excluded_branches.append(
                    {
                        "分點名稱": branch_name,
                        "分點類別": category,
                        "可用事件數": len(candidate_events),
                        "原因": city_exclusion_reason,
                    }
                )
            for date, code, stock_name, industry, net in candidate_events:
                stock_names[code] = stock_name
                add_horizon_event(
                    date,
                    code,
                    branch_name,
                    category,
                    city,
                    industry,
                    net,
                    price_paths,
                    price_cache,
                    overall,
                    by_category,
                    by_branch,
                    by_branch_side,
                    by_city,
                    by_industry,
                    by_stock20,
                    evaluation_stats,
                )

        if file_index % 50 == 0 or file_index == len(branch_files):
            print(f"processed {file_index}/{len(branch_files)} branch files", flush=True)

    overall_rows = [agg_row({"觀察期交易日": horizon}, overall[horizon]) for horizon in HORIZONS]
    category_rows = [
        agg_row({"分點類別": category, "觀察期交易日": horizon}, agg)
        for (category, horizon), agg in sorted(
            by_category.items(), key=lambda item: (item[0][1], -item[1].weight_sum, item[0][0])
        )
    ]
    branch_rows = [
        agg_row(
            {
                "分點名稱": branch,
                "分點類別": branch_category(branch, foreign_branches),
                "觀察期交易日": horizon,
            },
            agg,
        )
        for (branch, horizon), agg in sorted(
            by_branch.items(), key=lambda item: (item[0][1], -item[1].weight_sum, item[0][0])
        )
    ]
    side_rows = [
        agg_row(
            {
                "分點名稱": branch,
                "分點類別": branch_category(branch, foreign_branches),
                "方向": side,
                "觀察期交易日": horizon,
            },
            agg,
        )
        for (branch, side, horizon), agg in sorted(
            by_branch_side.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
        )
    ]
    city_rows = [
        agg_row({"縣市": city, "分點數": len(city_branch_names.get(city, set())), "觀察期交易日": horizon}, agg)
        for (city, horizon), agg in sorted(
            by_city.items(), key=lambda item: (item[0][1], -item[1].weight_sum, item[0][0])
        )
    ]
    industry_rows = [
        agg_row({"產業群組": industry, "觀察期交易日": horizon}, agg)
        for (industry, horizon), agg in sorted(
            by_industry.items(), key=lambda item: (item[0][1], -item[1].weight_sum, item[0][0])
        )
    ]

    stock_rows: list[dict[str, object]] = []
    branch_stock_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (branch, code), agg in by_stock20.items():
        if agg.count < args.min_stock_events:
            continue
        row = agg_row(
            {
                "分點名稱": branch,
                "分點類別": branch_category(branch, foreign_branches),
                "股票代號": code,
                "股票名稱": stock_names.get(code, ""),
                "產業群組": metadata.get(code, {}).get("產業群組", "未分類") or "未分類",
                "觀察期交易日": 20,
            },
            agg,
        )
        branch_stock_rows[branch].append(row)

    for branch, rows in branch_stock_rows.items():
        best = sorted(rows, key=lambda row: float(row["淨買賣超金額權重報酬"]), reverse=True)[:5]
        worst = sorted(rows, key=lambda row: float(row["淨買賣超金額權重報酬"]))[:5]
        for label, selected_rows in [("績效最佳", best), ("績效最差", worst)]:
            for rank, row in enumerate(selected_rows, 1):
                stock_rows.append({"類別": label, "排名": rank, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    common_fields = [
        "事件數",
        "買超事件數",
        "賣超事件數",
        "正報酬事件數",
        "負報酬事件數",
        "命中率",
        "平均決策後報酬",
        "淨買賣超金額權重報酬",
        "權重合計",
    ]
    paths = {
        "overall": args.output_dir / "all_broker_performance_overall.csv",
        "category": args.output_dir / "all_broker_performance_by_category.csv",
        "branch": args.output_dir / "all_broker_performance_by_branch.csv",
        "side": args.output_dir / "all_broker_performance_by_branch_side.csv",
        "city": args.output_dir / "all_broker_performance_by_city.csv",
        "city_branch": args.output_dir / "all_broker_city_branch_locations.csv",
        "city_excluded": args.output_dir / "all_broker_city_excluded_branches.csv",
        "industry": args.output_dir / "all_broker_performance_by_industry.csv",
        "stock": args.output_dir / "all_broker_top_stocks_20d.csv",
        "excluded_small": args.output_dir / "all_broker_excluded_small_branches.csv",
        "metadata": args.output_dir / "all_broker_performance_metadata.json",
        "html": args.viz_dir / "all_broker_decision_performance.html",
        "city_html": args.viz_dir / "all_broker_city_decision_performance.html",
    }
    write_csv(paths["overall"], ["觀察期交易日", *common_fields], overall_rows)
    write_csv(paths["category"], ["分點類別", "觀察期交易日", *common_fields], category_rows)
    write_csv(paths["branch"], ["分點名稱", "分點類別", "觀察期交易日", *common_fields], branch_rows)
    write_csv(paths["side"], ["分點名稱", "分點類別", "方向", "觀察期交易日", *common_fields], side_rows)
    write_csv(paths["city"], ["縣市", "分點數", "觀察期交易日", *common_fields], city_rows)
    write_csv(
        paths["city_branch"],
        [
            "縣市",
            "縣市來源",
            "分點名稱",
            "分點類別",
            "可用事件數",
            "TWSE分點代號",
            "TWSE總公司代號",
            "TWSE總公司名稱",
            "TWSE開業日",
            "TWSE地址",
            "TWSE電話",
        ],
        sorted(city_branch_rows, key=lambda row: (str(row["縣市"]), str(row["分點名稱"]))),
    )
    write_csv(
        paths["city_excluded"],
        ["分點名稱", "分點類別", "可用事件數", "原因"],
        sorted(city_excluded_branches, key=lambda row: (str(row["原因"]), str(row["分點名稱"]))),
    )
    write_csv(paths["industry"], ["產業群組", "觀察期交易日", *common_fields], industry_rows)
    write_csv(
        paths["stock"],
        ["類別", "排名", "分點名稱", "分點類別", "股票代號", "股票名稱", "產業群組", "觀察期交易日", *common_fields],
        stock_rows,
    )
    write_csv(
        paths["excluded_small"],
        ["分點名稱", "分點類別", "可用事件數", "事件數門檻", "來源檔案"],
        sorted(excluded_small_branches, key=lambda row: (int(row["可用事件數"]), str(row["分點名稱"]))),
    )
    if skipped_rows:
        write_csv(
            args.log_dir / "all_broker_performance_skipped_rows.csv",
            ["檔案", "列號", "原因", "Date", "Code"],
            skipped_rows,
        )
    if corrected_dates:
        write_csv(
            args.log_dir / "all_broker_performance_corrected_dates.csv",
            ["檔案", "列號", "原始Date", "修正Date", "原因", "來源網址"],
            corrected_dates,
        )

    report_metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "branch_files": len(branch_files),
        "horizons": list(HORIZONS),
        "listed_common_codes": len(listed_common),
        "price_files": len(price_paths),
        "loaded_price_series": len(price_cache),
        "foreign_branch_count": len(foreign_branches),
        "min_branch_events": args.min_branch_events,
        "min_stock_events": args.min_stock_events,
        "excluded_small_branch_count": len(excluded_small_branches),
        "city_report": {
            "location_source_url": TWSE_BROKER_SERVICE_API,
            "location_cache": str(args.broker_location_cache),
            "location_stats": branch_location_stats,
            "included_branch_count": len(city_branch_rows),
            "twse_address_branch_count": sum(1 for row in city_branch_rows if row.get("縣市來源") == "TWSE地址"),
            "name_inferred_branch_count": sum(1 for row in city_branch_rows if row.get("縣市來源") == "分點名稱推定"),
            "excluded_branch_count": len(city_excluded_branches),
            "city_count": len(city_branch_names),
            "excluded_categories": sorted(CITY_EXCLUDED_CATEGORIES),
        },
        "merged_categories_into_other": [
            "明示法人分點",
            "營業部/可能法人櫃台",
            "網路/電子分點",
        ],
        "broker_load_stats": dict(load_stats),
        "evaluation_stats": dict(evaluation_stats),
        "outputs": [
            str(paths[key])
            for key in [
                "overall",
                "category",
                "branch",
                "side",
                "city",
                "city_branch",
                "city_excluded",
                "industry",
                "stock",
                "excluded_small",
                "html",
                "city_html",
            ]
        ],
    }
    paths["metadata"].write_text(json.dumps(report_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["html"].write_text(
        render_html(
            overall_rows,
            category_rows,
            branch_rows,
            side_rows,
            industry_rows,
            stock_rows,
            load_churn_metrics(args.output_dir / BROKER_CHURN_METRICS.name),
            report_metadata,
        ),
        encoding="utf-8",
    )
    paths["city_html"].write_text(
        render_city_html(city_rows, city_branch_rows, city_excluded_branches, report_metadata),
        encoding="utf-8",
    )
    return report_metadata


def h(value: object) -> str:
    return html.escape(str(value), quote=True)


def display_value(key: str, value: object) -> str:
    if value in {"", None}:
        return ""
    if key in {"事件數", "買超事件數", "賣超事件數", "正報酬事件數", "負報酬事件數", "分點數", "可用事件數"}:
        return fmt_int(float(value))
    if key in {"權重合計", "買賣合計張數", "雙向換手張數", "去重事件數"}:
        return fmt_int(float(value))
    if key in {"命中率", "平均決策後報酬", "淨買賣超金額權重報酬", "雙向換手率", "買賣皆活躍事件率", "淨方向集中率"}:
        return fmt_pct(float(value))
    if key == "平均每事件成交張數":
        return f"{float(value):,.1f}"
    return str(value)


def render_table(headers: list[str], rows: list[dict[str, object]], limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    header_html = "".join(f"<th>{h(header)}</th>" for header in headers)
    body = []
    for row in selected:
        cells = "".join(f"<td>{h(display_value(header, row.get(header, '')))}</td>" for header in headers)
        body.append(f"<tr>{cells}</tr>")
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    """


def render_branch_chart(rows: list[dict[str, object]], title_key: str) -> str:
    values = [float(row["淨買賣超金額權重報酬"]) for row in rows]
    max_abs = max((abs(value) for value in values), default=1.0) or 1.0
    parts = []
    for row in rows:
        value = float(row["淨買賣超金額權重報酬"])
        width = abs(value) / max_abs * 50
        if value >= 0:
            bar = f"<div class='bar neg'></div><div class='bar pos' style='width:{width:.2f}%'></div>"
        else:
            bar = f"<div class='bar neg' style='width:{width:.2f}%'></div><div class='bar pos'></div>"
        parts.append(
            f"""
            <div class="chart-row">
              <div class="chart-label">{h(row[title_key])}</div>
              <div class="chart-track">{bar}</div>
              <div class="chart-value">{fmt_pct(value)}</div>
            </div>
            """
        )
    return f"<div class='branch-chart'>{''.join(parts)}</div>"


def render_html(
    overall_rows: list[dict[str, object]],
    category_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
    side_rows: list[dict[str, object]],
    industry_rows: list[dict[str, object]],
    stock_rows: list[dict[str, object]],
    churn_metrics: dict[str, dict[str, object]],
    metadata: dict[str, object],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    horizon20 = next((row for row in overall_rows if int(row["觀察期交易日"]) == 20), overall_rows[-1])
    def branch_rows_for_horizon(horizon: int) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in branch_rows:
            if int(row["觀察期交易日"]) != horizon:
                continue
            merged = dict(row)
            merged.update(churn_metrics.get(str(row["分點名稱"]), {}))
            rows.append(merged)
        return rows

    branch1 = branch_rows_for_horizon(1)
    branch5 = branch_rows_for_horizon(5)
    branch20 = branch_rows_for_horizon(20)
    best1 = sorted(branch1, key=lambda row: float(row["淨買賣超金額權重報酬"]), reverse=True)
    best5 = sorted(branch5, key=lambda row: float(row["淨買賣超金額權重報酬"]), reverse=True)
    best20 = sorted(branch20, key=lambda row: float(row["淨買賣超金額權重報酬"]), reverse=True)
    worst20 = sorted(branch20, key=lambda row: float(row["淨買賣超金額權重報酬"]))
    high_churn20 = sorted(
        [row for row in branch20 if math.isfinite(float(row.get("雙向換手率", math.nan)))],
        key=lambda row: (float(row.get("雙向換手率", 0)), float(row.get("買賣皆活躍事件率", 0))),
        reverse=True,
    )
    category20 = sorted(
        [row for row in category_rows if int(row["觀察期交易日"]) == 20],
        key=lambda row: float(row["權重合計"]),
        reverse=True,
    )
    side20 = sorted(
        [row for row in side_rows if int(row["觀察期交易日"]) == 20],
        key=lambda row: float(row["權重合計"]),
        reverse=True,
    )
    industry20 = sorted(
        [row for row in industry_rows if int(row["觀察期交易日"]) == 20],
        key=lambda row: float(row["權重合計"]),
        reverse=True,
    )
    best_stocks = [row for row in stock_rows if row["類別"] == "績效最佳"][:50]
    worst_stocks = [row for row in stock_rows if row["類別"] == "績效最差"][:50]

    common_headers = [
        "觀察期交易日",
        "事件數",
        "買超事件數",
        "賣超事件數",
        "命中率",
        "平均決策後報酬",
        "淨買賣超金額權重報酬",
    ]
    category_headers = ["分點類別", *common_headers]
    branch_headers = [
        "分點名稱",
        "分點類別",
        "雙向換手率",
        "買賣皆活躍事件率",
        "平均每事件成交張數",
        *common_headers,
    ]
    churn_headers = [
        "分點名稱",
        "分點類別",
        "雙向換手率",
        "買賣皆活躍事件率",
        "淨方向集中率",
        "平均每事件成交張數",
        "買賣合計張數",
        "去重事件數",
        "命中率",
        "淨買賣超金額權重報酬",
    ]
    side_headers = ["分點名稱", "分點類別", "方向", *common_headers]
    industry_headers = ["產業群組", *common_headers]
    stock_headers = [
        "類別",
        "排名",
        "分點名稱",
        "分點類別",
        "股票代號",
        "股票名稱",
        "產業群組",
        "事件數",
        "命中率",
        "淨買賣超金額權重報酬",
    ]

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>全分點決策後績效報告</title>
  <style>
    :root {{
      --ink: #18212a;
      --muted: #5f6d7a;
      --line: #d9e0e7;
      --bg: #f6f7f9;
      --panel: #fff;
      --pos: #0f766e;
      --neg: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }}
    header {{
      padding: 28px 36px 20px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    main {{ padding: 24px 36px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 18px 0 10px; font-size: 15px; letter-spacing: 0; }}
    p {{ margin: 6px 0; color: var(--muted); }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .pill {{
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
    }}
    .section {{
      margin-bottom: 22px;
      padding: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .note {{
      padding: 12px 14px;
      background: #fff7ed;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      color: #7c2d12;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .card {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .card strong {{ display:block; font-size: 22px; margin-bottom: 4px; }}
    .card span {{ color: var(--muted); font-size: 13px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    table {{ width: 100%; min-width: 980px; border-collapse: collapse; }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
      font-size: 13px;
    }}
    th {{ position: sticky; top: 0; background: #f1f4f7; text-align: right; }}
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2), td:nth-child(3), th:nth-child(3), td:nth-child(4), th:nth-child(4) {{ text-align: left; }}
    .branch-chart {{ display: grid; gap: 8px; }}
    .chart-row {{ display: grid; grid-template-columns: 220px 1fr 90px; gap: 10px; align-items: center; }}
    .chart-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .chart-track {{ display: grid; grid-template-columns: 1fr 1fr; align-items: center; height: 18px; background: #eef2f5; border-radius: 999px; overflow: hidden; }}
    .bar {{ height: 100%; }}
    .bar.neg {{ justify-self: end; background: var(--neg); }}
    .bar.pos {{ justify-self: start; background: var(--pos); }}
    .chart-value {{ text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }}
    .split {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .cards, .split {{ grid-template-columns: 1fr; }}
      .chart-row {{ grid-template-columns: 1fr; }}
      .chart-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>全分點決策後績效報告</h1>
    <p>用 Fubon 全部分點每日每股淨買賣超，對照 `data/price/` 復權收盤價，估算後續 1/3/5/10/20 個交易日的決策後績效。</p>
    <div class="meta">
      <span class="pill">產生時間：{h(generated_at)}</span>
      <span class="pill">分點檔案：{fmt_int(metadata['branch_files'])}</span>
      <span class="pill">納入分點：{fmt_int(metadata['broker_load_stats'].get('included_branch_files', 0))}</span>
      <span class="pill">排除小分點：{fmt_int(metadata.get('excluded_small_branch_count', 0))}</span>
      <span class="pill">分點事件門檻：{fmt_int(metadata.get('min_branch_events', 0))}</span>
      <span class="pill">去重後事件：{fmt_int(metadata['broker_load_stats'].get('events', 0))}</span>
      <span class="pill">換手率分點：{fmt_int(len(churn_metrics))}</span>
      <span class="pill">已載入價格序列：{fmt_int(metadata['loaded_price_series'])}</span>
    </div>
  </header>
  <main>
    <div class="note">注意：這是決策後績效，不是真實交易損益。買超事件用後續報酬，賣超事件用後續報酬反向；未扣交易成本，未估實際成交價，且 Fubon 排行資料只涵蓋排名列。本版先排除事件數低於 {fmt_int(metadata.get('min_branch_events', 0))} 的極小分點，並將明示法人、營業部/可能法人櫃台、網路/電子分點併入其他分點。</div>
    <section class="section">
      <h2>整體 20 日結果</h2>
      <div class="cards">
        <div class="card"><strong>{fmt_int(horizon20['事件數'])}</strong><span>可評估事件數</span></div>
        <div class="card"><strong>{fmt_pct(float(horizon20['命中率']))}</strong><span>方向命中率</span></div>
        <div class="card"><strong>{fmt_pct(float(horizon20['平均決策後報酬']))}</strong><span>平均決策後報酬</span></div>
        <div class="card"><strong>{fmt_pct(float(horizon20['淨買賣超金額權重報酬']))}</strong><span>淨買賣超金額權重報酬</span></div>
      </div>
    </section>
    <section class="section">
      <h2>整體期間比較</h2>
      {render_table(common_headers, overall_rows)}
    </section>
    <section class="section">
      <h2>分點類別比較（20 日）</h2>
      {render_table(category_headers, category20)}
    </section>
    <section class="section">
      <h2>20 日績效最佳分點</h2>
      {render_branch_chart(best20[:50], '分點名稱')}
    </section>
    <section class="section">
      <h2>20 日績效最差分點</h2>
      {render_branch_chart(worst20[:50], '分點名稱')}
    </section>
    <section class="section">
      <h2>分點績效表（1 日，前 50 名）</h2>
      {render_table(branch_headers, best1, limit=50)}
    </section>
    <section class="section">
      <h2>分點績效表（5 日，前 50 名）</h2>
      {render_table(branch_headers, best5, limit=50)}
    </section>
    <section class="section">
      <h2>分點績效表（20 日，前 50 名）</h2>
      {render_table(branch_headers, best20, limit=50)}
    </section>
    <section class="section">
      <h2>換手率最高分點（20 日分點樣本，依雙向換手率排序）</h2>
      <p>雙向換手率 = sum(2 * min(買進, 賣出)) / sum(買進 + 賣出)，代表分點在同日同股買賣兩邊都活躍的程度；買賣皆活躍事件率則是 min(買進, 賣出) / max(買進, 賣出) ≥ 25% 的事件比例。</p>
      {render_table(churn_headers, high_churn20, limit=50)}
    </section>
    <section class="section">
      <h2>買超與賣超拆開看（20 日，依權重排序前 300）</h2>
      {render_table(side_headers, side20, limit=300)}
    </section>
    <section class="section">
      <h2>產業群組（20 日，依權重排序）</h2>
      {render_table(industry_headers, industry20)}
    </section>
    <section class="section">
      <h2>股票層級：20 日績效最佳與最差</h2>
      <div class="split">
        <div>
          <h3>績效最佳</h3>
          {render_table(stock_headers, best_stocks)}
        </div>
        <div>
          <h3>績效最差</h3>
          {render_table(stock_headers, worst_stocks)}
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


def render_city_html(
    city_rows: list[dict[str, object]],
    city_branch_rows: list[dict[str, object]],
    city_excluded_branches: list[dict[str, object]],
    metadata: dict[str, object],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    horizon20_rows = [row for row in city_rows if int(row["觀察期交易日"]) == 20]
    horizon20 = Agg()
    for row in horizon20_rows:
        row_agg = Agg(
            count=int(row["事件數"]),
            buy_count=int(row["買超事件數"]),
            sell_count=int(row["賣超事件數"]),
            win_count=int(row["正報酬事件數"]),
            loss_count=int(row["負報酬事件數"]),
            sum_return=float(row["平均決策後報酬"]) * int(row["事件數"]),
            weighted_sum_return=float(row["淨買賣超金額權重報酬"]) * float(row["權重合計"]),
            weight_sum=float(row["權重合計"]),
        )
        horizon20.count += row_agg.count
        horizon20.buy_count += row_agg.buy_count
        horizon20.sell_count += row_agg.sell_count
        horizon20.win_count += row_agg.win_count
        horizon20.loss_count += row_agg.loss_count
        horizon20.sum_return += row_agg.sum_return
        horizon20.weighted_sum_return += row_agg.weighted_sum_return
        horizon20.weight_sum += row_agg.weight_sum

    city20_by_return = sorted(
        horizon20_rows,
        key=lambda row: float(row["淨買賣超金額權重報酬"]),
        reverse=True,
    )
    city20_by_weight = sorted(
        horizon20_rows,
        key=lambda row: float(row["權重合計"]),
        reverse=True,
    )
    all_city_rows = sorted(
        city_rows,
        key=lambda row: (int(row["觀察期交易日"]), -float(row["權重合計"]), str(row["縣市"])),
    )
    branch_rows = sorted(city_branch_rows, key=lambda row: (str(row["縣市"]), str(row["分點名稱"])))
    excluded_rows = sorted(city_excluded_branches, key=lambda row: (str(row["原因"]), str(row["分點名稱"])))
    city_report = metadata.get("city_report", {})

    city_headers = [
        "縣市",
        "分點數",
        "觀察期交易日",
        "事件數",
        "買超事件數",
        "賣超事件數",
        "命中率",
        "平均決策後報酬",
        "淨買賣超金額權重報酬",
    ]
    branch_headers = ["縣市", "縣市來源", "分點名稱", "分點類別", "可用事件數", "TWSE分點代號", "TWSE地址", "TWSE電話"]
    excluded_headers = ["分點名稱", "分點類別", "可用事件數", "原因"]

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>分點縣市決策後績效報告</title>
  <style>
    :root {{
      --ink: #18212a;
      --muted: #5f6d7a;
      --line: #d9e0e7;
      --bg: #f6f7f9;
      --panel: #fff;
      --pos: #0f766e;
      --neg: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
    }}
    header {{
      padding: 28px 36px 20px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    main {{ padding: 24px 36px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; letter-spacing: 0; }}
    p {{ margin: 6px 0; color: var(--muted); }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }}
    .pill {{
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
    }}
    .section {{
      margin-bottom: 22px;
      padding: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .note {{
      padding: 12px 14px;
      background: #eef6ff;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      color: #1e3a8a;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .card {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .card strong {{ display:block; font-size: 22px; margin-bottom: 4px; }}
    .card span {{ color: var(--muted); font-size: 13px; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    table {{ width: 100%; min-width: 980px; border-collapse: collapse; }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
      font-size: 13px;
    }}
    th {{ position: sticky; top: 0; background: #f1f4f7; text-align: right; }}
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2), td:nth-child(3), th:nth-child(3), td:nth-child(4), th:nth-child(4) {{ text-align: left; }}
    .branch-chart {{ display: grid; gap: 8px; }}
    .chart-row {{ display: grid; grid-template-columns: 160px 1fr 90px; gap: 10px; align-items: center; }}
    .chart-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .chart-track {{ display: grid; grid-template-columns: 1fr 1fr; align-items: center; height: 18px; background: #eef2f5; border-radius: 999px; overflow: hidden; }}
    .bar {{ height: 100%; }}
    .bar.neg {{ justify-self: end; background: var(--neg); }}
    .bar.pos {{ justify-self: start; background: var(--pos); }}
    .chart-value {{ text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .cards {{ grid-template-columns: 1fr; }}
      .chart-row {{ grid-template-columns: 1fr; }}
      .chart-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>分點縣市決策後績效報告</h1>
    <p>優先用 TWSE 現行證券商分公司地址將 Fubon 分點訊號歸到台灣縣市；地址解析不到時，再用分點名稱地名推定縣市，並統計後續 1/3/5/10/20 個交易日的決策後績效。</p>
    <div class="meta">
      <span class="pill">產生時間：{h(generated_at)}</span>
      <span class="pill">縣市數：{fmt_int(city_report.get('city_count', 0))}</span>
      <span class="pill">納入分點：{fmt_int(city_report.get('included_branch_count', 0))}</span>
      <span class="pill">TWSE地址判斷：{fmt_int(city_report.get('twse_address_branch_count', 0))}</span>
      <span class="pill">名稱推定：{fmt_int(city_report.get('name_inferred_branch_count', 0))}</span>
      <span class="pill">排除分點：{fmt_int(city_report.get('excluded_branch_count', 0))}</span>
      <span class="pill">位置來源：TWSE 現行分公司名冊 + 分點名稱</span>
    </div>
  </header>
  <main>
    <div class="note">本報告納入可由 TWSE 現行分公司地址或分點名稱地名歸到縣市的分點；外資推定分點、總公司/主分點或總號、停業/舊分點，以及地址與名稱都無法推定縣市者都剔除。績效仍是決策後報酬，不是真實庫存損益。</div>
    <section class="section">
      <h2>縣市分組 20 日總覽</h2>
      <div class="cards">
        <div class="card"><strong>{fmt_int(horizon20.count)}</strong><span>可評估事件數</span></div>
        <div class="card"><strong>{fmt_int(city_report.get('included_branch_count', 0))}</strong><span>納入分點數</span></div>
        <div class="card"><strong>{fmt_pct(horizon20.win_rate)}</strong><span>方向命中率</span></div>
        <div class="card"><strong>{fmt_pct(horizon20.avg_return)}</strong><span>平均決策後報酬</span></div>
        <div class="card"><strong>{fmt_pct(horizon20.weighted_return)}</strong><span>金額權重報酬</span></div>
      </div>
    </section>
    <section class="section">
      <h2>20 日縣市金額權重報酬</h2>
      {render_branch_chart(city20_by_return, '縣市')}
    </section>
    <section class="section">
      <h2>20 日縣市比較</h2>
      {render_table(city_headers, city20_by_weight)}
    </section>
    <section class="section">
      <h2>所有觀察期縣市表</h2>
      {render_table(city_headers, all_city_rows)}
    </section>
    <section class="section">
      <h2>納入分點位置對照</h2>
      {render_table(branch_headers, branch_rows)}
    </section>
    <section class="section">
      <h2>剔除分點與原因</h2>
      {render_table(excluded_headers, excluded_rows)}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    metadata = build_outputs(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
