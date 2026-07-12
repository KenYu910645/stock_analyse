from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BROKER_DIR = PROJECT_ROOT / "data" / "broker" / "by_broker"
PRICE_DIR = PROJECT_ROOT / "data" / "price"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
BRANCH_SUMMARY_PATH = PROJECT_ROOT / "output" / "broker" / "foreign_branch_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "broker"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "broker"
LOG_DIR = PROJECT_ROOT / "logs" / "broker"

HORIZONS = (1, 3, 5, 10, 20)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SOURCE_URL_DATE_RE = re.compile(r"[?&]e=(\d{4})-(\d{1,2})-(\d{1,2})(?:&|$)")
MIN_REASONABLE_DATE = "2000-01-01"


@dataclass(frozen=True)
class BrokerEvent:
    branch: str
    date: str
    code: str
    name: str
    buy: int
    sell: int
    net: int


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build foreign broker branch decision-payoff report from Fubon rank data."
    )
    parser.add_argument("--broker-dir", type=Path, default=BROKER_DIR)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--foreign-branch-summary", type=Path, default=BRANCH_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--min-stock-events", type=int, default=30)
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_foreign_branches(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row["分點名稱"] for row in csv.DictReader(handle)]


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


def load_price_paths(price_dir: Path) -> dict[str, Path]:
    return {
        code_from_path(path): path
        for path in sorted(price_dir.glob("*.csv"))
        if not path.name.startswith("twse_")
    }


def load_price_series(path: Path) -> tuple[list[str], list[float], list[float]]:
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
    return dates, adj_close, close


def load_broker_events(
    broker_dir: Path,
    branches: list[str],
    listed_common: set[str],
    project_root: Path,
) -> tuple[dict[str, list[BrokerEvent]], dict[str, int], list[dict[str, object]], list[dict[str, object]]]:
    events_by_code: dict[str, list[BrokerEvent]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    stats = defaultdict(int)
    skipped_rows: list[dict[str, object]] = []
    corrected_dates: list[dict[str, object]] = []

    for branch in branches:
        path = broker_dir / f"{branch}.csv"
        if not path.exists():
            stats["missing_branch_files"] += 1
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=2):
                raw_date = (row.get("Date") or "").strip()
                source_url = (row.get("來源網址") or "").strip()
                date, date_note = normalize_date(raw_date, source_url)
                code = (row.get("Code") or "").strip()
                if not date or not code:
                    stats["skipped_bad_date_or_code"] += 1
                    if len(skipped_rows) < 1000:
                        skipped_rows.append(
                            {
                                "檔案": str(path.relative_to(project_root)),
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
                            "檔案": str(path.relative_to(project_root)),
                            "列號": row_number,
                            "原始Date": raw_date,
                            "修正Date": date,
                            "原因": date_note,
                            "來源網址": source_url,
                        }
                    )
                if code not in listed_common:
                    stats["skipped_non_listed_common"] += 1
                    continue
                try:
                    buy = parse_int(row.get("買進"))
                    sell = parse_int(row.get("賣出"))
                    net = parse_int(row.get("買賣超"))
                except ValueError:
                    stats["skipped_bad_number"] += 1
                    if len(skipped_rows) < 1000:
                        skipped_rows.append(
                            {
                                "檔案": str(path.relative_to(project_root)),
                                "列號": row_number,
                                "原因": "買進賣出欄位非數字",
                                "Date": raw_date,
                                "Code": code,
                            }
                        )
                    continue
                if net == 0:
                    stats["skipped_zero_net"] += 1
                    continue
                key = (branch, date, code)
                if key in seen:
                    stats["deduplicated_branch_date_stock"] += 1
                    continue
                seen.add(key)
                events_by_code[code].append(
                    BrokerEvent(
                        branch=branch,
                        date=date,
                        code=code,
                        name=(row.get("Name") or "").strip(),
                        buy=buy,
                        sell=sell,
                        net=net,
                    )
                )
                stats["events"] += 1
    return events_by_code, dict(stats), skipped_rows, corrected_dates


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


def build_outputs(args: argparse.Namespace) -> dict[str, object]:
    branches = load_foreign_branches(args.foreign_branch_summary)
    listed_common, metadata = load_metadata(args.metadata)
    price_paths = load_price_paths(args.price_dir)
    events_by_code, load_stats, skipped_rows, corrected_dates = load_broker_events(
        args.broker_dir, branches, listed_common, PROJECT_ROOT
    )

    overall: dict[int, Agg] = defaultdict(Agg)
    by_branch: dict[tuple[str, int], Agg] = defaultdict(Agg)
    by_branch_side: dict[tuple[str, str, int], Agg] = defaultdict(Agg)
    by_industry: dict[tuple[str, int], Agg] = defaultdict(Agg)
    by_stock: dict[tuple[str, str, int], Agg] = defaultdict(Agg)

    evaluation_stats = defaultdict(int)
    stock_names: dict[str, str] = {}

    for code, events in events_by_code.items():
        price_path = price_paths.get(code)
        if not price_path:
            evaluation_stats["missing_price_file_events"] += len(events)
            continue
        dates, adj_close, close = load_price_series(price_path)
        if not dates:
            evaluation_stats["empty_price_file_events"] += len(events)
            continue
        date_to_index = {date: index for index, date in enumerate(dates)}
        industry = metadata.get(code, {}).get("產業群組", "未分類") or "未分類"
        stock_name = metadata.get(code, {}).get("Name", "") or events[0].name
        stock_names[code] = stock_name

        for event in events:
            index = date_to_index.get(event.date)
            if index is None:
                evaluation_stats["missing_price_date_events"] += 1
                continue
            entry_adj = adj_close[index]
            entry_close = close[index]
            if not math.isfinite(entry_adj) or entry_adj <= 0 or not math.isfinite(entry_close):
                evaluation_stats["invalid_entry_price_events"] += 1
                continue
            direction = 1 if event.net > 0 else -1
            side = "買超" if direction > 0 else "賣超"
            weight = abs(event.net) * entry_close
            for horizon in HORIZONS:
                future_index = index + horizon
                if future_index >= len(adj_close):
                    evaluation_stats[f"missing_future_{horizon}d"] += 1
                    continue
                future_adj = adj_close[future_index]
                if not math.isfinite(future_adj) or future_adj <= 0:
                    evaluation_stats[f"invalid_future_{horizon}d"] += 1
                    continue
                raw_return = future_adj / entry_adj - 1.0
                decision_return = raw_return * direction
                overall[horizon].add(decision_return, weight, direction)
                by_branch[(event.branch, horizon)].add(decision_return, weight, direction)
                by_branch_side[(event.branch, side, horizon)].add(decision_return, weight, direction)
                by_industry[(industry, horizon)].add(decision_return, weight, direction)
                by_stock[(event.branch, code, horizon)].add(decision_return, weight, direction)
                evaluation_stats["evaluated_event_horizons"] += 1

    overall_rows = [
        agg_row({"觀察期交易日": horizon}, overall[horizon])
        for horizon in HORIZONS
    ]
    branch_rows = [
        agg_row({"分點名稱": branch, "觀察期交易日": horizon}, agg)
        for (branch, horizon), agg in sorted(
            by_branch.items(), key=lambda item: (item[0][1], -item[1].weight_sum, item[0][0])
        )
    ]
    side_rows = [
        agg_row({"分點名稱": branch, "方向": side, "觀察期交易日": horizon}, agg)
        for (branch, side, horizon), agg in sorted(
            by_branch_side.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
        )
    ]
    industry_rows = [
        agg_row({"產業群組": industry, "觀察期交易日": horizon}, agg)
        for (industry, horizon), agg in sorted(
            by_industry.items(), key=lambda item: (item[0][1], -item[1].weight_sum, item[0][0])
        )
    ]

    stock_rows: list[dict[str, object]] = []
    horizon_for_stock = 20
    branch_stock_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (branch, code, horizon), agg in by_stock.items():
        if horizon != horizon_for_stock or agg.count < args.min_stock_events:
            continue
        row = agg_row(
            {
                "分點名稱": branch,
                "股票代號": code,
                "股票名稱": stock_names.get(code, ""),
                "產業群組": metadata.get(code, {}).get("產業群組", "未分類") or "未分類",
                "觀察期交易日": horizon,
            },
            agg,
        )
        stock_rows.append(row)
        branch_stock_rows[branch].append(row)

    top_stock_rows: list[dict[str, object]] = []
    for branch, rows in branch_stock_rows.items():
        best = sorted(rows, key=lambda row: float(row["淨買賣超金額權重報酬"]), reverse=True)[:10]
        worst = sorted(rows, key=lambda row: float(row["淨買賣超金額權重報酬"]))[:10]
        for direction, selected_rows in [("績效最佳", best), ("績效最差", worst)]:
            for rank, row in enumerate(selected_rows, 1):
                top_stock_rows.append({"類別": direction, "排名": rank, **row})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    overall_path = args.output_dir / "foreign_broker_performance_overall.csv"
    branch_path = args.output_dir / "foreign_broker_performance_by_branch.csv"
    side_path = args.output_dir / "foreign_broker_performance_by_branch_side.csv"
    industry_path = args.output_dir / "foreign_broker_performance_by_industry.csv"
    stock_path = args.output_dir / "foreign_broker_top_stocks_20d.csv"
    metadata_path = args.output_dir / "foreign_broker_performance_metadata.json"
    html_path = args.viz_dir / "foreign_broker_decision_performance.html"

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
    write_csv(overall_path, ["觀察期交易日", *common_fields], overall_rows)
    write_csv(branch_path, ["分點名稱", "觀察期交易日", *common_fields], branch_rows)
    write_csv(side_path, ["分點名稱", "方向", "觀察期交易日", *common_fields], side_rows)
    write_csv(industry_path, ["產業群組", "觀察期交易日", *common_fields], industry_rows)
    write_csv(
        stock_path,
        ["類別", "排名", "分點名稱", "股票代號", "股票名稱", "產業群組", "觀察期交易日", *common_fields],
        top_stock_rows,
    )
    if skipped_rows:
        write_csv(
            args.log_dir / "foreign_broker_performance_skipped_rows.csv",
            ["檔案", "列號", "原因", "Date", "Code"],
            skipped_rows,
        )
    if corrected_dates:
        write_csv(
            args.log_dir / "foreign_broker_performance_corrected_dates.csv",
            ["檔案", "列號", "原始Date", "修正Date", "原因", "來源網址"],
            corrected_dates,
        )

    report_metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "branches": branches,
        "horizons": list(HORIZONS),
        "listed_common_codes": len(listed_common),
        "price_files": len(price_paths),
        "broker_load_stats": dict(load_stats),
        "evaluation_stats": dict(evaluation_stats),
        "outputs": [
            str(overall_path),
            str(branch_path),
            str(side_path),
            str(industry_path),
            str(stock_path),
            str(html_path),
        ],
    }
    metadata_path.write_text(json.dumps(report_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(
        render_html(overall_rows, branch_rows, side_rows, industry_rows, top_stock_rows, report_metadata),
        encoding="utf-8",
    )
    return report_metadata


def h(value: object) -> str:
    return html.escape(str(value), quote=True)


def display_value(key: str, value: object) -> str:
    if key in {"事件數", "買超事件數", "賣超事件數", "正報酬事件數", "負報酬事件數"}:
        return fmt_int(float(value))
    if key == "權重合計":
        return fmt_int(float(value))
    if key in {"命中率", "平均決策後報酬", "淨買賣超金額權重報酬"}:
        return fmt_pct(float(value))
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


def render_branch_chart(branch_rows: list[dict[str, object]], horizon: int) -> str:
    rows = [row for row in branch_rows if int(row["觀察期交易日"]) == horizon]
    rows = sorted(rows, key=lambda row: float(row["淨買賣超金額權重報酬"]), reverse=True)
    if not rows:
        return ""
    values = [float(row["淨買賣超金額權重報酬"]) for row in rows]
    max_abs = max(abs(value) for value in values) or 1.0
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
              <div class="chart-label">{h(row['分點名稱'])}</div>
              <div class="chart-track">{bar}</div>
              <div class="chart-value">{fmt_pct(value)}</div>
            </div>
            """
        )
    return f"<div class='branch-chart'>{''.join(parts)}</div>"


def render_html(
    overall_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
    side_rows: list[dict[str, object]],
    industry_rows: list[dict[str, object]],
    stock_rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    horizon20 = next((row for row in overall_rows if int(row["觀察期交易日"]) == 20), overall_rows[-1])
    best20 = sorted(
        [row for row in branch_rows if int(row["觀察期交易日"]) == 20],
        key=lambda row: float(row["淨買賣超金額權重報酬"]),
        reverse=True,
    )
    side20 = [row for row in side_rows if int(row["觀察期交易日"]) == 20]
    industry20 = sorted(
        [row for row in industry_rows if int(row["觀察期交易日"]) == 20],
        key=lambda row: float(row["權重合計"]),
        reverse=True,
    )
    best_stocks = [row for row in stock_rows if row["類別"] == "績效最佳"][:40]
    worst_stocks = [row for row in stock_rows if row["類別"] == "績效最差"][:40]

    common_headers = [
        "觀察期交易日",
        "事件數",
        "買超事件數",
        "賣超事件數",
        "命中率",
        "平均決策後報酬",
        "淨買賣超金額權重報酬",
    ]
    branch_headers = ["分點名稱", *common_headers]
    side_headers = ["分點名稱", "方向", *common_headers]
    industry_headers = ["產業群組", *common_headers]
    stock_headers = [
        "類別",
        "排名",
        "分點名稱",
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
  <title>外資分點決策後績效初報告</title>
  <style>
    :root {{
      --ink: #18212a;
      --muted: #5f6d7a;
      --line: #d9e0e7;
      --bg: #f6f7f9;
      --panel: #fff;
      --pos: #0f766e;
      --neg: #b42318;
      --soft: #edf7f5;
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
    table {{ width: 100%; min-width: 900px; border-collapse: collapse; }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
      font-size: 13px;
    }}
    th {{ position: sticky; top: 0; background: #f1f4f7; text-align: right; }}
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2), td:nth-child(3), th:nth-child(3) {{ text-align: left; }}
    .branch-chart {{ display: grid; gap: 8px; }}
    .chart-row {{ display: grid; grid-template-columns: 190px 1fr 90px; gap: 10px; align-items: center; }}
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
    <h1>外資分點決策後績效初報告</h1>
    <p>用 Fubon 外資分點每日每股淨買賣超，對照 `data/price/` 復權收盤價，估算後續 1/3/5/10/20 個交易日的決策後績效。</p>
    <div class="meta">
      <span class="pill">產生時間：{h(generated_at)}</span>
      <span class="pill">外資分點：{fmt_int(len(metadata['branches']))}</span>
      <span class="pill">去重後事件：{fmt_int(metadata['broker_load_stats'].get('events', 0))}</span>
      <span class="pill">上市普通股：{fmt_int(metadata['listed_common_codes'])}</span>
    </div>
  </header>
  <main>
    <div class="note">注意：這是決策後績效，不是真實交易損益。買超事件用後續報酬，賣超事件用後續報酬反向；未扣交易成本，未估實際成交價，且 Fubon 排行資料只涵蓋排名列。</div>
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
      <h2>各外資分點 20 日加權報酬</h2>
      {render_branch_chart(branch_rows, 20)}
    </section>
    <section class="section">
      <h2>各分點績效表</h2>
      {render_table(branch_headers, best20)}
    </section>
    <section class="section">
      <h2>買超與賣超拆開看（20 日）</h2>
      {render_table(side_headers, side20)}
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


def main() -> None:
    args = parse_args()
    metadata = build_outputs(args)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
