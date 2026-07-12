from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BROKER_DIR = PROJECT_ROOT / "data" / "broker" / "by_broker"
PRICE_DIR = PROJECT_ROOT / "data" / "price"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "broker"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "broker"

BRANCH_CSV = OUTPUT_DIR / "broker_volume_share_by_branch.csv"
DATE_CSV = OUTPUT_DIR / "broker_volume_share_by_date.csv"
BRANCH_DATE_CSV = OUTPUT_DIR / "broker_volume_share_by_branch_date.csv"
STOCKDAY_CSV = OUTPUT_DIR / "broker_volume_share_by_stockday_coverage.csv"
METADATA_JSON = OUTPUT_DIR / "broker_volume_share_metadata.json"
REPORT_HTML = VIZ_DIR / "broker_volume_share_report.html"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOURCE_URL_DATE_RE = re.compile(r"[?&]e=(\d{4})-(\d{1,2})-(\d{1,2})(?:&|$)")
MIN_REASONABLE_DATE = "2000-01-01"
LOTS_PER_SHARE = 1 / 1000


@dataclass
class BranchStats:
    branch: str
    raw_rows: int = 0
    duplicate_rows: int = 0
    matched_events: int = 0
    unmatched_price_events: int = 0
    out_of_universe_events: int = 0
    buy_lots: int = 0
    sell_lots: int = 0
    dates: set[str] = field(default_factory=set)
    stocks: set[str] = field(default_factory=set)
    first_date: str | None = None
    last_date: str | None = None

    @property
    def gross_lots(self) -> int:
        return self.buy_lots + self.sell_lots

    @property
    def net_lots(self) -> int:
        return self.buy_lots - self.sell_lots

    def add_event(self, date: str, code: str, buy_lots: int, sell_lots: int) -> None:
        self.matched_events += 1
        self.buy_lots += buy_lots
        self.sell_lots += sell_lots
        self.dates.add(date)
        self.stocks.add(code)
        if self.first_date is None or date < self.first_date:
            self.first_date = date
        if self.last_date is None or date > self.last_date:
            self.last_date = date


@dataclass
class AggStats:
    buy_lots: int = 0
    sell_lots: int = 0
    events: int = 0

    @property
    def gross_lots(self) -> int:
        return self.buy_lots + self.sell_lots

    @property
    def net_lots(self) -> int:
        return self.buy_lots - self.sell_lots

    def add(self, buy_lots: int, sell_lots: int) -> None:
        self.buy_lots += buy_lots
        self.sell_lots += sell_lots
        self.events += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Fubon broker branch volume shares against official daily volume.")
    parser.add_argument("--broker-dir", type=Path, default=BROKER_DIR)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    parser.add_argument("--top-branches", type=int, default=80)
    parser.add_argument("--top-dates", type=int, default=30)
    return parser.parse_args()


def parse_int(value: str | None) -> int:
    if value is None:
        return 0
    text = str(value).replace(",", "").strip()
    if not text or text == "-":
        return 0
    return int(float(text))


def parse_lot_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text or text == "-":
        return 0
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    text = str(value).replace(",", "").strip()
    if not text or text == "-":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_source_url_date(source_url: str | None) -> str | None:
    if not source_url:
        return None
    match = SOURCE_URL_DATE_RE.search(source_url)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def normalize_date(raw_date: str, source_url: str | None) -> str | None:
    text = (raw_date or "").strip()
    if DATE_RE.match(text) and text >= MIN_REASONABLE_DATE:
        return text
    return parse_source_url_date(source_url)


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def fmt_int(value: object) -> str:
    number = parse_float(str(value))
    return "" if not math.isfinite(number) else f"{int(round(number)):,}"


def fmt_pct(value: object, digits: int = 2) -> str:
    number = parse_float(str(value))
    return "" if not math.isfinite(number) else f"{number * 100:.{digits}f}%"


def h(value: object) -> str:
    return html.escape(str(value), quote=True)


def load_listed_common_codes(path: Path) -> tuple[set[str], dict[str, dict[str, str]]]:
    rows_by_code: dict[str, dict[str, str]] = {}
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = (row.get("Code") or "").strip()
            if not code:
                continue
            rows_by_code[code] = row
            if row.get("市場") == "上市" and row.get("類型") == "股票" and row.get("板別") == "一般":
                allowed.add(code)
    return allowed, rows_by_code


def price_paths_by_code(price_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in price_dir.glob("*.csv"):
        code = path.name.split("_", 1)[0]
        paths[code] = path
    return paths


def load_price_lots(price_dir: Path, allowed_codes: set[str]) -> tuple[dict[tuple[str, str], float], dict[str, float], dict[str, int]]:
    by_stockday: dict[tuple[str, str], float] = {}
    daily_lots: dict[str, float] = defaultdict(float)
    daily_stock_count: dict[str, int] = defaultdict(int)
    for code, path in sorted(price_paths_by_code(price_dir).items()):
        if code not in allowed_codes:
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            volume_field = "成交股數" if reader.fieldnames and "成交股數" in reader.fieldnames else "Capacity"
            for row in reader:
                date = (row.get("Date") or "").strip()
                if not date:
                    continue
                shares = parse_float(row.get(volume_field))
                if not math.isfinite(shares) or shares <= 0:
                    continue
                lots = shares * LOTS_PER_SHARE
                by_stockday[(date, code)] = lots
                daily_lots[date] += lots
                daily_stock_count[date] += 1
    return by_stockday, dict(daily_lots), dict(daily_stock_count)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_table(headers: list[str], rows: list[dict[str, object]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    if not selected:
        return "<div class='empty'>沒有資料</div>"
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + "".join(f"<th>{h(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{h(display_value(header, row.get(header, '')))}</td>" for header in headers) + "</tr>"
            for row in selected
        )
        + "</tbody></table></div>"
    )


def display_value(header: str, value: object) -> str:
    if value in ("", None):
        return ""
    if header.endswith("佔比") or header.endswith("覆蓋率") or header.endswith("市佔估算") or header in {
        "買方市佔估算",
        "賣方市佔估算",
        "雙邊市佔估算",
        "Fubon揭露量佔比",
    }:
        return fmt_pct(value)
    if header.endswith("張") or header.endswith("數") or header in {"排名", "活躍天數", "股票數", "事件數", "重複列數"}:
        return fmt_int(value)
    return str(value)


def bar_svg(rows: list[dict[str, object]], value_key: str, label_key: str, title: str, limit: int = 30) -> str:
    selected = rows[:limit]
    if not selected:
        return "<div class='empty'>沒有圖表資料</div>"
    width = 1000
    left = 175
    right = 96
    top = 34
    row_h = 28
    height = top + row_h * len(selected) + 24
    values = [parse_float(str(row[value_key])) for row in selected]
    max_v = max([value for value in values if math.isfinite(value)] or [0.0])
    if max_v <= 0:
        max_v = 1.0
    plot_w = width - left - right
    parts = [f'<text x="{left}" y="18">{h(title)}</text>']
    for index, row in enumerate(selected):
        value = parse_float(str(row[value_key]))
        value = value if math.isfinite(value) else 0.0
        y = top + index * row_h
        bar_w = max(1.0, value / max_v * plot_w)
        parts.append(f'<text x="8" y="{y + 15}">{h(row[label_key])}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="18"/>')
        parts.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 14}">{h(fmt_pct(value))}</text>')
    return f'<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{h(title)}">' + "".join(parts) + "</svg>"


def build_report(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir
    viz_dir = args.viz_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    allowed_codes, metadata = load_listed_common_codes(args.metadata)
    price_lots, daily_market_lots, daily_stock_count = load_price_lots(args.price_dir, allowed_codes)

    branch_stats: dict[str, BranchStats] = {}
    branch_date_stats: dict[tuple[str, str], AggStats] = defaultdict(AggStats)
    date_stats: dict[str, AggStats] = defaultdict(AggStats)
    stockday_stats: dict[tuple[str, str], AggStats] = defaultdict(AggStats)
    active_branches_by_date: dict[str, set[str]] = defaultdict(set)

    raw_rows = 0
    duplicate_rows = 0
    out_of_universe_events = 0
    unmatched_price_events = 0
    invalid_numeric_events = 0
    matched_events = 0
    corrected_date_rows = 0

    files = sorted(args.broker_dir.glob("*.csv"))
    for file_index, path in enumerate(files, start=1):
        branch = path.stem
        stats = branch_stats.setdefault(branch, BranchStats(branch=branch))
        seen: set[tuple[str, str]] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw_rows += 1
                stats.raw_rows += 1
                date = normalize_date(row.get("Date", ""), row.get("來源網址"))
                if not date:
                    unmatched_price_events += 1
                    stats.unmatched_price_events += 1
                    continue
                if date != (row.get("Date") or "").strip():
                    corrected_date_rows += 1
                code = (row.get("Code") or "").strip()
                key = (date, code)
                if key in seen:
                    duplicate_rows += 1
                    stats.duplicate_rows += 1
                    continue
                seen.add(key)
                if code not in allowed_codes:
                    out_of_universe_events += 1
                    stats.out_of_universe_events += 1
                    continue
                market_lots = price_lots.get((date, code))
                if not market_lots:
                    unmatched_price_events += 1
                    stats.unmatched_price_events += 1
                    continue
                buy_lots = parse_lot_int(row.get("買進"))
                sell_lots = parse_lot_int(row.get("賣出"))
                if buy_lots is None or sell_lots is None:
                    invalid_numeric_events += 1
                    continue
                if buy_lots + sell_lots <= 0:
                    continue

                matched_events += 1
                stats.add_event(date, code, buy_lots, sell_lots)
                branch_date_stats[(branch, date)].add(buy_lots, sell_lots)
                date_stats[date].add(buy_lots, sell_lots)
                stockday_stats[(date, code)].add(buy_lots, sell_lots)
                active_branches_by_date[date].add(branch)
        if file_index % 100 == 0:
            print(f"processed {file_index}/{len(files)} broker files", flush=True)

    covered_dates = set(date_stats)
    market_one_side_lots_all_dates = sum(daily_market_lots.get(date, 0.0) for date in covered_dates)
    market_two_side_lots_all_dates = market_one_side_lots_all_dates * 2
    matched_stockday_market_one_side_lots = sum(price_lots[key] for key in stockday_stats)
    matched_stockday_market_two_side_lots = matched_stockday_market_one_side_lots * 2

    total_disclosed_buy_lots = sum(stats.buy_lots for stats in branch_stats.values())
    total_disclosed_sell_lots = sum(stats.sell_lots for stats in branch_stats.values())
    total_disclosed_gross_lots = total_disclosed_buy_lots + total_disclosed_sell_lots

    branch_rows: list[dict[str, object]] = []
    for stats in branch_stats.values():
        if stats.gross_lots <= 0:
            continue
        branch_rows.append(
            {
                "排名": 0,
                "分點名稱": stats.branch,
                "買進張": stats.buy_lots,
                "賣出張": stats.sell_lots,
                "雙邊成交張": stats.gross_lots,
                "買賣超張": stats.net_lots,
                "Fubon揭露量佔比": safe_ratio(stats.gross_lots, total_disclosed_gross_lots),
                "雙邊市佔估算": safe_ratio(stats.gross_lots, market_two_side_lots_all_dates),
                "買方市佔估算": safe_ratio(stats.buy_lots, market_one_side_lots_all_dates),
                "賣方市佔估算": safe_ratio(stats.sell_lots, market_one_side_lots_all_dates),
                "匹配股票日雙邊市佔估算": safe_ratio(stats.gross_lots, matched_stockday_market_two_side_lots),
                "活躍天數": len(stats.dates),
                "股票數": len(stats.stocks),
                "事件數": stats.matched_events,
                "重複列數": stats.duplicate_rows,
                "價格未匹配事件數": stats.unmatched_price_events,
                "非上市普通股事件數": stats.out_of_universe_events,
                "起始日期": stats.first_date or "",
                "結束日期": stats.last_date or "",
            }
        )
    branch_rows.sort(key=lambda row: float(row["雙邊成交張"]), reverse=True)
    for rank, row in enumerate(branch_rows, start=1):
        row["排名"] = rank

    date_rows: list[dict[str, object]] = []
    for date, stats in sorted(date_stats.items()):
        market_one_side = daily_market_lots.get(date, 0.0)
        date_rows.append(
            {
                "Date": date,
                "上市普通股市場成交張": market_one_side,
                "上市普通股雙邊成交張": market_one_side * 2,
                "Fubon揭露買進張": stats.buy_lots,
                "Fubon揭露賣出張": stats.sell_lots,
                "Fubon揭露雙邊成交張": stats.gross_lots,
                "買方覆蓋率": safe_ratio(stats.buy_lots, market_one_side),
                "賣方覆蓋率": safe_ratio(stats.sell_lots, market_one_side),
                "雙邊覆蓋率": safe_ratio(stats.gross_lots, market_one_side * 2),
                "活躍分點數": len(active_branches_by_date[date]),
                "價格股票數": daily_stock_count.get(date, 0),
                "Fubon匹配事件數": stats.events,
            }
        )
    date_rows.sort(key=lambda row: str(row["Date"]))

    branch_date_rows: list[dict[str, object]] = []
    for (branch, date), stats in branch_date_stats.items():
        market_one_side = daily_market_lots.get(date, 0.0)
        branch_date_rows.append(
            {
                "Date": date,
                "分點名稱": branch,
                "買進張": stats.buy_lots,
                "賣出張": stats.sell_lots,
                "雙邊成交張": stats.gross_lots,
                "買賣超張": stats.net_lots,
                "雙邊市佔估算": safe_ratio(stats.gross_lots, market_one_side * 2),
                "買方市佔估算": safe_ratio(stats.buy_lots, market_one_side),
                "賣方市佔估算": safe_ratio(stats.sell_lots, market_one_side),
                "事件數": stats.events,
            }
        )
    branch_date_rows.sort(key=lambda row: (str(row["Date"]), -float(row["雙邊成交張"]), str(row["分點名稱"])))

    stockday_rows: list[dict[str, object]] = []
    for (date, code), stats in stockday_stats.items():
        market_lots = price_lots.get((date, code), 0.0)
        meta = metadata.get(code, {})
        stockday_rows.append(
            {
                "Date": date,
                "Code": code,
                "Name": meta.get("Name", ""),
                "市場成交張": market_lots,
                "市場雙邊成交張": market_lots * 2,
                "Fubon揭露買進張": stats.buy_lots,
                "Fubon揭露賣出張": stats.sell_lots,
                "Fubon揭露雙邊成交張": stats.gross_lots,
                "買方覆蓋率": safe_ratio(stats.buy_lots, market_lots),
                "賣方覆蓋率": safe_ratio(stats.sell_lots, market_lots),
                "雙邊覆蓋率": safe_ratio(stats.gross_lots, market_lots * 2),
                "分點事件數": stats.events,
            }
        )
    stockday_rows.sort(key=lambda row: float(row["雙邊覆蓋率"]) if math.isfinite(float(row["雙邊覆蓋率"])) else -1.0, reverse=True)

    write_csv(
        output_dir / BRANCH_CSV.name,
        [
            "排名",
            "分點名稱",
            "買進張",
            "賣出張",
            "雙邊成交張",
            "買賣超張",
            "Fubon揭露量佔比",
            "雙邊市佔估算",
            "買方市佔估算",
            "賣方市佔估算",
            "匹配股票日雙邊市佔估算",
            "活躍天數",
            "股票數",
            "事件數",
            "重複列數",
            "價格未匹配事件數",
            "非上市普通股事件數",
            "起始日期",
            "結束日期",
        ],
        branch_rows,
    )
    write_csv(
        output_dir / DATE_CSV.name,
        [
            "Date",
            "上市普通股市場成交張",
            "上市普通股雙邊成交張",
            "Fubon揭露買進張",
            "Fubon揭露賣出張",
            "Fubon揭露雙邊成交張",
            "買方覆蓋率",
            "賣方覆蓋率",
            "雙邊覆蓋率",
            "活躍分點數",
            "價格股票數",
            "Fubon匹配事件數",
        ],
        date_rows,
    )
    write_csv(
        output_dir / BRANCH_DATE_CSV.name,
        [
            "Date",
            "分點名稱",
            "買進張",
            "賣出張",
            "雙邊成交張",
            "買賣超張",
            "雙邊市佔估算",
            "買方市佔估算",
            "賣方市佔估算",
            "事件數",
        ],
        branch_date_rows,
    )
    write_csv(
        output_dir / STOCKDAY_CSV.name,
        [
            "Date",
            "Code",
            "Name",
            "市場成交張",
            "市場雙邊成交張",
            "Fubon揭露買進張",
            "Fubon揭露賣出張",
            "Fubon揭露雙邊成交張",
            "買方覆蓋率",
            "賣方覆蓋率",
            "雙邊覆蓋率",
            "分點事件數",
        ],
        stockday_rows,
    )

    coverage_summary = {
        "market_one_side_lots_all_covered_dates": market_one_side_lots_all_dates,
        "market_two_side_lots_all_covered_dates": market_two_side_lots_all_dates,
        "matched_stockday_market_one_side_lots": matched_stockday_market_one_side_lots,
        "matched_stockday_market_two_side_lots": matched_stockday_market_two_side_lots,
        "total_disclosed_buy_lots": total_disclosed_buy_lots,
        "total_disclosed_sell_lots": total_disclosed_sell_lots,
        "total_disclosed_gross_lots": total_disclosed_gross_lots,
        "buy_coverage_all_listed_dates": safe_ratio(total_disclosed_buy_lots, market_one_side_lots_all_dates),
        "sell_coverage_all_listed_dates": safe_ratio(total_disclosed_sell_lots, market_one_side_lots_all_dates),
        "two_side_coverage_all_listed_dates": safe_ratio(total_disclosed_gross_lots, market_two_side_lots_all_dates),
        "two_side_coverage_matched_stockdays": safe_ratio(total_disclosed_gross_lots, matched_stockday_market_two_side_lots),
    }

    metadata_json = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "broker_files": len(files),
        "listed_common_codes": len(allowed_codes),
        "price_stockdays": len(price_lots),
        "covered_dates": len(covered_dates),
        "matched_stockdays": len(stockday_stats),
        "raw_rows": raw_rows,
        "duplicate_rows": duplicate_rows,
        "matched_events": matched_events,
        "out_of_universe_events": out_of_universe_events,
        "unmatched_price_events": unmatched_price_events,
        "invalid_numeric_events": invalid_numeric_events,
        "corrected_date_rows": corrected_date_rows,
        **coverage_summary,
        "outputs": [
            str(output_dir / BRANCH_CSV.name),
            str(output_dir / DATE_CSV.name),
            str(output_dir / BRANCH_DATE_CSV.name),
            str(output_dir / STOCKDAY_CSV.name),
            str(viz_dir / REPORT_HTML.name),
        ],
    }
    (output_dir / METADATA_JSON.name).write_text(json.dumps(metadata_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (viz_dir / REPORT_HTML.name).write_text(
        render_html(branch_rows, date_rows, stockday_rows, metadata_json, args),
        encoding="utf-8",
    )
    return metadata_json


def render_html(
    branch_rows: list[dict[str, object]],
    date_rows: list[dict[str, object]],
    stockday_rows: list[dict[str, object]],
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> str:
    branch_headers = [
        "排名",
        "分點名稱",
        "雙邊成交張",
        "Fubon揭露量佔比",
        "雙邊市佔估算",
        "買方市佔估算",
        "賣方市佔估算",
        "活躍天數",
        "股票數",
        "事件數",
        "起始日期",
        "結束日期",
    ]
    date_headers = [
        "Date",
        "上市普通股市場成交張",
        "Fubon揭露雙邊成交張",
        "買方覆蓋率",
        "賣方覆蓋率",
        "雙邊覆蓋率",
        "活躍分點數",
        "Fubon匹配事件數",
    ]
    stockday_headers = [
        "Date",
        "Code",
        "Name",
        "市場成交張",
        "Fubon揭露雙邊成交張",
        "買方覆蓋率",
        "賣方覆蓋率",
        "雙邊覆蓋率",
        "分點事件數",
    ]
    date_desc = sorted(date_rows, key=lambda row: str(row["Date"]), reverse=True)
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>分點成交量佔比估算報告</title>
  <style>
    :root {{ --ink:#17212b; --muted:#5f6b76; --line:#d8dee6; --panel:#fff; --bg:#f6f7f9; --bar:#2563eb; --warn:#7c2d12; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Noto Sans TC","Microsoft JhengHei",Arial,sans-serif; color:var(--ink); background:var(--bg); line-height:1.55; }}
    header {{ padding:28px 36px 20px; background:#fff; border-bottom:1px solid var(--line); }}
    main {{ padding:24px 36px 48px; }}
    h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:0 0 14px; font-size:20px; letter-spacing:0; }}
    p {{ color:var(--muted); margin:6px 0; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }}
    .pill {{ padding:6px 10px; border:1px solid var(--line); border-radius:999px; background:#fff; color:var(--muted); font-size:13px; }}
    .section {{ margin-bottom:22px; padding:20px; background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    .note {{ margin-bottom:18px; padding:14px 16px; background:#fff7ed; border:1px solid #fed7aa; border-radius:8px; color:var(--warn); }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
    .card {{ padding:14px; border:1px solid var(--line); border-radius:8px; background:#fff; }}
    .card strong {{ display:block; font-size:21px; }}
    .card span {{ color:var(--muted); font-size:13px; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ border-collapse:collapse; width:100%; min-width:1040px; font-size:13px; }}
    th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ background:#eef2f7; color:#334155; position:sticky; top:0; }}
    tbody tr:hover {{ background:#f8fafc; }}
    .bar-chart {{ width:100%; height:auto; background:#fff; border:1px solid var(--line); border-radius:8px; }}
    .bar-chart rect {{ fill:var(--bar); }}
    .bar-chart text {{ font-size:12px; fill:var(--muted); }}
    code {{ color:#334155; }}
    @media(max-width:900px) {{ header,main {{ padding-left:16px; padding-right:16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>分點成交量佔比估算報告</h1>
    <p>用 Fubon 分點排名資料的買進/賣出張數，對齊上市普通股官方成交股數，估算每個分點在已揭露排名量與全市場雙邊成交量中的佔比。</p>
    <div class="meta">
      <span class="pill">產生時間：{h(metadata["generated_at"])}</span>
      <span class="pill">分點檔案：{fmt_int(metadata["broker_files"])}</span>
      <span class="pill">涵蓋交易日：{fmt_int(metadata["covered_dates"])}</span>
      <span class="pill">匹配事件：{fmt_int(metadata["matched_events"])}</span>
    </div>
  </header>
  <main>
    <div class="note">重點限制：如果資料是完整逐分點交易，所有分點買方加總會等於市場成交量、賣方加總也會等於市場成交量；買進+賣出則會等於雙邊成交量，也就是 2 倍市場成交量。本報告的 Fubon 來源是排名資料，不是完整逐分點全量，所以「市佔估算」應視為已揭露排名資料下限，不是官方市佔。</div>

    <section class="section">
      <h2>Coverage 摘要</h2>
      <div class="cards">
        <div class="card"><strong>{fmt_pct(metadata["two_side_coverage_all_listed_dates"])}</strong><span>相對上市普通股雙邊成交量的揭露覆蓋率</span></div>
        <div class="card"><strong>{fmt_pct(metadata["two_side_coverage_matched_stockdays"])}</strong><span>相對有 Fubon 排名股票日的揭露覆蓋率</span></div>
        <div class="card"><strong>{fmt_int(metadata["total_disclosed_gross_lots"])}</strong><span>Fubon 揭露雙邊成交張</span></div>
        <div class="card"><strong>{fmt_int(metadata["market_two_side_lots_all_covered_dates"])}</strong><span>上市普通股雙邊成交張</span></div>
      </div>
    </section>

    <section class="section">
      <h2>分點總佔比排行</h2>
      {bar_svg(branch_rows, "雙邊市佔估算", "分點名稱", "分點雙邊市佔估算排行", args.top_branches)}
      {render_table(branch_headers, branch_rows, args.top_branches)}
      <p>完整分點總表：<code>output/broker/{h(BRANCH_CSV.name)}</code></p>
    </section>

    <section class="section">
      <h2>最近交易日 Coverage</h2>
      {render_table(date_headers, date_desc, args.top_dates)}
      <p>完整每日 coverage：<code>output/broker/{h(DATE_CSV.name)}</code></p>
    </section>

    <section class="section">
      <h2>Coverage 最高的股票日</h2>
      {render_table(stockday_headers, stockday_rows, args.top_dates)}
      <p>股票日 coverage 用單一股票單日的官方成交量比較 Fubon 已揭露分點排名量；完整表：<code>output/broker/{h(STOCKDAY_CSV.name)}</code></p>
    </section>

    <section class="section">
      <h2>每日分點佔比明細</h2>
      <p>已輸出每個分點每天的買方、賣方、雙邊市佔估算，可用來驗證同日所有分點加總的 coverage。檔案：<code>output/broker/{h(BRANCH_DATE_CSV.name)}</code></p>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    metadata = build_report(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
