from __future__ import annotations

import argparse
import csv
import html
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BROKER_DIR = PROJECT_ROOT / "data" / "broker" / "by_broker"
OUTPUT_DIR = PROJECT_ROOT / "output" / "broker"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "broker"

BRANCH_PERFORMANCE_CSV = OUTPUT_DIR / "all_broker_performance_by_branch.csv"
BRANCH_SIDE_PERFORMANCE_CSV = OUTPUT_DIR / "all_broker_performance_by_branch_side.csv"
EXCLUDED_SMALL_BRANCHES_CSV = OUTPUT_DIR / "all_broker_excluded_small_branches.csv"
CHURN_METRICS_CSV = OUTPUT_DIR / "broker_trading_style_churn_metrics.csv"
STYLE_BY_BRANCH_CSV = OUTPUT_DIR / "broker_trading_style_by_branch.csv"
STYLE_SUMMARY_CSV = OUTPUT_DIR / "broker_trading_style_summary.csv"
STYLE_TOP_BRANCHES_CSV = OUTPUT_DIR / "broker_trading_style_top_branches.csv"
STYLE_OVERLAP_CSV = OUTPUT_DIR / "broker_trading_style_overlap.csv"
STYLE_HTML = VIZ_DIR / "broker_trading_style_report.html"

HORIZONS = (1, 3, 5, 10, 20)
STYLE_SCORE_THRESHOLD = 85.0
MIN_EVENTS = 1000


@dataclass
class BranchPerformance:
    branch: str
    category: str
    events: dict[int, int] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)
    avg_return: dict[int, float] = field(default_factory=dict)
    weighted_return: dict[int, float] = field(default_factory=dict)
    weight_sum: dict[int, float] = field(default_factory=dict)
    side_weighted_return: dict[tuple[str, int], float] = field(default_factory=dict)


@dataclass
class ChurnMetrics:
    branch: str
    raw_rows: int = 0
    unique_events: int = 0
    gross_volume: int = 0
    abs_net_volume: int = 0
    two_way_volume: int = 0
    balanced_events: int = 0
    buy_dominant_events: int = 0
    sell_dominant_events: int = 0

    @property
    def two_way_ratio(self) -> float:
        return self.two_way_volume / self.gross_volume if self.gross_volume else 0.0

    @property
    def balanced_event_ratio(self) -> float:
        return self.balanced_events / self.unique_events if self.unique_events else 0.0

    @property
    def net_direction_ratio(self) -> float:
        return self.abs_net_volume / self.gross_volume if self.gross_volume else 0.0

    @property
    def avg_gross_per_event(self) -> float:
        return self.gross_volume / self.unique_events if self.unique_events else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Fubon broker branches into day-trade, overnight, swing, and longer-horizon styles."
    )
    parser.add_argument("--broker-dir", type=Path, default=BROKER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    parser.add_argument("--branch-performance", type=Path, default=BRANCH_PERFORMANCE_CSV)
    parser.add_argument("--branch-side-performance", type=Path, default=BRANCH_SIDE_PERFORMANCE_CSV)
    parser.add_argument("--excluded-small-branches", type=Path, default=EXCLUDED_SMALL_BRANCHES_CSV)
    parser.add_argument("--churn-metrics", type=Path, default=CHURN_METRICS_CSV)
    parser.add_argument("--min-events", type=int, default=MIN_EVENTS)
    parser.add_argument("--style-score-threshold", type=float, default=STYLE_SCORE_THRESHOLD)
    parser.add_argument(
        "--reuse-churn-cache",
        action="store_true",
        help="Reuse existing broker_trading_style_churn_metrics.csv instead of scanning raw broker files.",
    )
    return parser.parse_args()


def parse_int(value: str | None) -> int:
    if not value:
        return 0
    text = value.replace(",", "").strip()
    if not text or text == "-":
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def parse_float(value: str | None) -> float:
    if not value:
        return math.nan
    text = value.replace(",", "").strip()
    if not text or text == "-":
        return math.nan
    return float(text)


def pct(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def fmt_pct(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value * 100:.{digits}f}%"


def fmt_score(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.1f}"


def fmt_num(value: int | float) -> str:
    return f"{int(round(float(value))):,}"


def percent_ranks(values: dict[str, float]) -> dict[str, float]:
    finite_items = [(key, value) for key, value in values.items() if math.isfinite(value)]
    if not finite_items:
        return {key: 0.0 for key in values}
    finite_items.sort(key=lambda item: item[1])
    denominator = max(len(finite_items) - 1, 1)
    ranks: dict[str, float] = {}
    for index, (key, _) in enumerate(finite_items):
        ranks[key] = index / denominator * 100.0
    return {key: ranks.get(key, 0.0) for key in values}


def load_branch_performance(path: Path) -> dict[str, BranchPerformance]:
    branches: dict[str, BranchPerformance] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            branch = row["分點名稱"]
            horizon = parse_int(row["觀察期交易日"])
            if horizon not in HORIZONS:
                continue
            perf = branches.setdefault(branch, BranchPerformance(branch=branch, category=row["分點類別"]))
            perf.events[horizon] = parse_int(row["事件數"])
            perf.hit_rate[horizon] = parse_float(row["命中率"])
            perf.avg_return[horizon] = parse_float(row["平均決策後報酬"])
            perf.weighted_return[horizon] = parse_float(row["淨買賣超金額權重報酬"])
            perf.weight_sum[horizon] = parse_float(row["權重合計"])
    return branches


def load_branch_side_performance(path: Path, branches: dict[str, BranchPerformance]) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            branch = row["分點名稱"]
            perf = branches.get(branch)
            if not perf:
                continue
            horizon = parse_int(row["觀察期交易日"])
            side = row["方向"]
            perf.side_weighted_return[(side, horizon)] = parse_float(row["淨買賣超金額權重報酬"])


def read_churn_cache(path: Path) -> dict[str, ChurnMetrics]:
    rows: dict[str, ChurnMetrics] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            metric = ChurnMetrics(
                branch=row["分點名稱"],
                raw_rows=parse_int(row["原始列數"]),
                unique_events=parse_int(row["去重事件數"]),
                gross_volume=parse_int(row["買賣合計張數"]),
                abs_net_volume=parse_int(row["絕對買賣超張數"]),
                two_way_volume=parse_int(row["雙向換手張數"]),
                balanced_events=parse_int(row["買賣皆活躍事件數"]),
                buy_dominant_events=parse_int(row["買方主導事件數"]),
                sell_dominant_events=parse_int(row["賣方主導事件數"]),
            )
            rows[metric.branch] = metric
    return rows


def scan_churn_metrics(broker_dir: Path) -> dict[str, ChurnMetrics]:
    metrics: dict[str, ChurnMetrics] = {}
    files = sorted(broker_dir.glob("*.csv"))
    for file_index, path in enumerate(files, 1):
        metric = ChurnMetrics(branch=path.stem)
        seen: set[tuple[str, str]] = set()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                metrics[metric.branch] = metric
                continue
            index = {name: i for i, name in enumerate(header)}
            required = ["Date", "Code", "買進", "賣出"]
            if any(name not in index for name in required):
                metrics[metric.branch] = metric
                continue
            max_index = max(index[name] for name in required)
            for row in reader:
                metric.raw_rows += 1
                if len(row) <= max_index:
                    continue
                key = (row[index["Date"]].strip(), row[index["Code"]].strip())
                if key in seen:
                    continue
                seen.add(key)
                buy = parse_int(row[index["買進"]])
                sell = parse_int(row[index["賣出"]])
                gross = buy + sell
                if gross <= 0:
                    continue
                metric.unique_events += 1
                metric.gross_volume += gross
                metric.abs_net_volume += abs(buy - sell)
                metric.two_way_volume += 2 * min(buy, sell)
                larger = max(buy, sell)
                if larger > 0 and min(buy, sell) / larger >= 0.25:
                    metric.balanced_events += 1
                if buy > sell:
                    metric.buy_dominant_events += 1
                elif sell > buy:
                    metric.sell_dominant_events += 1
        metrics[metric.branch] = metric
        if file_index % 100 == 0:
            print(f"processed {file_index}/{len(files)} broker files", flush=True)
    return metrics


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_churn_metrics(path: Path, rows: dict[str, ChurnMetrics]) -> None:
    write_csv(
        path,
        [
            "分點名稱",
            "原始列數",
            "去重事件數",
            "買賣合計張數",
            "絕對買賣超張數",
            "雙向換手張數",
            "買賣皆活躍事件數",
            "買方主導事件數",
            "賣方主導事件數",
            "雙向換手率",
            "買賣皆活躍事件率",
            "淨方向集中率",
            "平均每事件成交張數",
        ],
        [
            {
                "分點名稱": metric.branch,
                "原始列數": metric.raw_rows,
                "去重事件數": metric.unique_events,
                "買賣合計張數": metric.gross_volume,
                "絕對買賣超張數": metric.abs_net_volume,
                "雙向換手張數": metric.two_way_volume,
                "買賣皆活躍事件數": metric.balanced_events,
                "買方主導事件數": metric.buy_dominant_events,
                "賣方主導事件數": metric.sell_dominant_events,
                "雙向換手率": pct(metric.two_way_ratio),
                "買賣皆活躍事件率": pct(metric.balanced_event_ratio),
                "淨方向集中率": pct(metric.net_direction_ratio),
                "平均每事件成交張數": pct(metric.avg_gross_per_event),
            }
            for metric in sorted(rows.values(), key=lambda item: item.branch)
        ],
    )


def load_excluded_small_branches(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_small_branch_rows(
    rows: list[dict[str, object]],
    excluded_small: list[dict[str, str]],
    churn_metrics: dict[str, ChurnMetrics],
) -> None:
    existing = {str(row["分點名稱"]) for row in rows}
    for excluded in excluded_small:
        branch = excluded.get("分點名稱", "")
        if not branch or branch in existing:
            continue
        churn = churn_metrics.get(branch, ChurnMetrics(branch))
        rows.append(
            {
                "分點名稱": branch,
                "分點類別": excluded.get("分點類別", ""),
                "樣本狀態": "樣本不足",
                "風格標籤": "樣本不足",
                "當沖傾向": "0",
                "隔日衝": "0",
                "波段分點": "0",
                "長線傾向": "0",
                "當沖分數": "",
                "隔日衝分數": "",
                "波段分數": "",
                "長線分數": "",
                "可用事件數": excluded.get("可用事件數", "0"),
                "雙向換手率": pct(churn.two_way_ratio),
                "買賣皆活躍事件率": pct(churn.balanced_event_ratio),
                "淨方向集中率": pct(churn.net_direction_ratio),
                "平均每事件成交張數": pct(churn.avg_gross_per_event),
                "1日金額權重報酬": "",
                "3日金額權重報酬": "",
                "5日金額權重報酬": "",
                "10日金額權重報酬": "",
                "20日金額權重報酬": "",
                "20日命中率": "",
                "20日買超報酬": "",
                "20日賣超報酬": "",
                "20日較佳方向": "",
                "20日較佳方向報酬": "",
            }
        )


def build_style_rows(
    branches: dict[str, BranchPerformance],
    churn_metrics: dict[str, ChurnMetrics],
    min_events: int,
    score_threshold: float,
) -> list[dict[str, object]]:
    eligible = {
        branch: perf
        for branch, perf in branches.items()
        if perf.events.get(20, 0) >= min_events and all(horizon in perf.weighted_return for horizon in HORIZONS)
    }

    same_day_inputs = {
        branch: 0.60 * churn_metrics.get(branch, ChurnMetrics(branch)).two_way_ratio
        + 0.25 * churn_metrics.get(branch, ChurnMetrics(branch)).balanced_event_ratio
        + 0.15 * math.log1p(churn_metrics.get(branch, ChurnMetrics(branch)).avg_gross_per_event)
        for branch in eligible
    }
    overnight_inputs = {
        branch: 0.70 * perf.weighted_return[1]
        + 0.30 * (perf.weighted_return[1] - max(perf.weighted_return[5], perf.weighted_return[10], perf.weighted_return[20]))
        for branch, perf in eligible.items()
    }
    swing_inputs = {
        branch: 0.70 * max(perf.weighted_return[5], perf.weighted_return[10])
        + 0.30
        * (max(perf.weighted_return[5], perf.weighted_return[10]) - max(perf.weighted_return[1], perf.weighted_return[20]))
        for branch, perf in eligible.items()
    }
    long_inputs = {
        branch: 0.70 * perf.weighted_return[20]
        + 0.30 * (perf.weighted_return[20] - max(perf.weighted_return[1], perf.weighted_return[3], perf.weighted_return[5]))
        for branch, perf in eligible.items()
    }

    same_day_scores = percent_ranks(same_day_inputs)
    overnight_scores = percent_ranks(overnight_inputs)
    swing_scores = percent_ranks(swing_inputs)
    long_scores = percent_ranks(long_inputs)

    rows: list[dict[str, object]] = []
    for branch, perf in sorted(branches.items()):
        churn = churn_metrics.get(branch, ChurnMetrics(branch))
        is_eligible = branch in eligible
        wr = {horizon: perf.weighted_return.get(horizon, math.nan) for horizon in HORIZONS}
        hit = {horizon: perf.hit_rate.get(horizon, math.nan) for horizon in HORIZONS}
        same_day_score = same_day_scores.get(branch, 0.0)
        overnight_score = overnight_scores.get(branch, 0.0)
        swing_score = swing_scores.get(branch, 0.0)
        long_score = long_scores.get(branch, 0.0)
        mid_return = max(wr[5], wr[10]) if math.isfinite(wr[5]) and math.isfinite(wr[10]) else math.nan

        day_trade = is_eligible and same_day_score >= score_threshold and churn.two_way_ratio >= 0.0
        overnight = is_eligible and overnight_score >= score_threshold and wr[1] > 0
        swing = is_eligible and swing_score >= score_threshold and mid_return > 0
        long_term = is_eligible and long_score >= score_threshold and wr[20] > 0 and hit[20] >= 0.5

        labels = []
        if day_trade:
            labels.append("當沖傾向")
        if overnight:
            labels.append("隔日衝")
        if swing:
            labels.append("波段分點")
        if long_term:
            labels.append("長線傾向")

        buy20 = perf.side_weighted_return.get(("買超", 20), math.nan)
        sell20 = perf.side_weighted_return.get(("賣超", 20), math.nan)
        if math.isfinite(buy20) and math.isfinite(sell20):
            better_side = "買超" if buy20 >= sell20 else "賣超"
            better_side_return = max(buy20, sell20)
        else:
            better_side = ""
            better_side_return = math.nan

        rows.append(
            {
                "分點名稱": branch,
                "分點類別": perf.category,
                "樣本狀態": "可分類" if is_eligible else "樣本不足",
                "風格標籤": "、".join(labels) if labels else ("未明顯歸類" if is_eligible else "樣本不足"),
                "當沖傾向": "1" if day_trade else "0",
                "隔日衝": "1" if overnight else "0",
                "波段分點": "1" if swing else "0",
                "長線傾向": "1" if long_term else "0",
                "當沖分數": pct(same_day_score),
                "隔日衝分數": pct(overnight_score),
                "波段分數": pct(swing_score),
                "長線分數": pct(long_score),
                "可用事件數": perf.events.get(20, 0),
                "雙向換手率": pct(churn.two_way_ratio),
                "買賣皆活躍事件率": pct(churn.balanced_event_ratio),
                "淨方向集中率": pct(churn.net_direction_ratio),
                "平均每事件成交張數": pct(churn.avg_gross_per_event),
                "1日金額權重報酬": pct(wr[1]),
                "3日金額權重報酬": pct(wr[3]),
                "5日金額權重報酬": pct(wr[5]),
                "10日金額權重報酬": pct(wr[10]),
                "20日金額權重報酬": pct(wr[20]),
                "20日命中率": pct(hit[20]),
                "20日買超報酬": pct(buy20),
                "20日賣超報酬": pct(sell20),
                "20日較佳方向": better_side,
                "20日較佳方向報酬": pct(better_side_return),
            }
        )
    return rows


def summarize_styles(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for style, score_col, return_col in [
        ("當沖傾向", "當沖分數", "雙向換手率"),
        ("隔日衝", "隔日衝分數", "1日金額權重報酬"),
        ("波段分點", "波段分數", "10日金額權重報酬"),
        ("長線傾向", "長線分數", "20日金額權重報酬"),
    ]:
        style_rows = [row for row in rows if row.get(style) == "1"]
        if not style_rows:
            summary.append({"風格": style, "分點數": 0})
            continue
        summary.append(
            {
                "風格": style,
                "分點數": len(style_rows),
                "平均可用事件數": pct(sum(parse_float(str(row["可用事件數"])) for row in style_rows) / len(style_rows)),
                "平均風格分數": pct(sum(parse_float(str(row[score_col])) for row in style_rows) / len(style_rows)),
                "平均雙向換手率": pct(sum(parse_float(str(row["雙向換手率"])) for row in style_rows) / len(style_rows)),
                "平均1日金額權重報酬": pct(sum(parse_float(str(row["1日金額權重報酬"])) for row in style_rows) / len(style_rows)),
                "平均5日金額權重報酬": pct(sum(parse_float(str(row["5日金額權重報酬"])) for row in style_rows) / len(style_rows)),
                "平均10日金額權重報酬": pct(sum(parse_float(str(row["10日金額權重報酬"])) for row in style_rows) / len(style_rows)),
                "平均20日金額權重報酬": pct(sum(parse_float(str(row["20日金額權重報酬"])) for row in style_rows) / len(style_rows)),
                "代表排序指標": return_col,
            }
        )
    eligible_rows = [row for row in rows if row["樣本狀態"] == "可分類"]
    unclassified = [row for row in eligible_rows if row["風格標籤"] == "未明顯歸類"]
    small = [row for row in rows if row["樣本狀態"] == "樣本不足"]
    summary.append({"風格": "未明顯歸類", "分點數": len(unclassified)})
    summary.append({"風格": "樣本不足", "分點數": len(small)})
    return summary


def top_style_rows(rows: list[dict[str, object]], limit: int = 50) -> list[dict[str, object]]:
    ranking_specs = [
        ("當沖傾向", "當沖分數"),
        ("隔日衝", "隔日衝分數"),
        ("波段分點", "波段分數"),
        ("長線傾向", "長線分數"),
    ]
    output: list[dict[str, object]] = []
    for style, score_col in ranking_specs:
        style_rows = [row for row in rows if row.get(style) == "1"]
        style_rows.sort(
            key=lambda row: (
                parse_float(str(row.get(score_col, ""))),
                parse_float(str(row.get("20日金額權重報酬", ""))),
                parse_int(str(row.get("可用事件數", ""))),
            ),
            reverse=True,
        )
        for rank, row in enumerate(style_rows[:limit], 1):
            output.append(
                {
                    "風格": style,
                    "排名": rank,
                    "分點名稱": row["分點名稱"],
                    "分點類別": row["分點類別"],
                    "風格標籤": row["風格標籤"],
                    "風格分數": row[score_col],
                    "可用事件數": row["可用事件數"],
                    "雙向換手率": row["雙向換手率"],
                    "買賣皆活躍事件率": row["買賣皆活躍事件率"],
                    "1日金額權重報酬": row["1日金額權重報酬"],
                    "5日金額權重報酬": row["5日金額權重報酬"],
                    "10日金額權重報酬": row["10日金額權重報酬"],
                    "20日金額權重報酬": row["20日金額權重報酬"],
                    "20日命中率": row["20日命中率"],
                    "20日較佳方向": row["20日較佳方向"],
                    "20日較佳方向報酬": row["20日較佳方向報酬"],
                }
            )
    return output


def overlap_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    styles = ("當沖傾向", "隔日衝", "波段分點", "長線傾向")
    output: list[dict[str, object]] = []
    for style_a in styles:
        for style_b in styles:
            output.append(
                {
                    "風格A": style_a,
                    "風格B": style_b,
                    "重疊分點數": sum(1 for row in rows if row.get(style_a) == "1" and row.get(style_b) == "1"),
                }
            )
    return output


def table(headers: list[str], rows: list[dict[str, object]], numeric_cols: set[str] | None = None) -> str:
    numeric_cols = numeric_cols or set()
    body = []
    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header, "")
            cls = ' class="num"' if header in numeric_cols else ""
            cells.append(f"<td{cls}>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def format_display_row(row: dict[str, object]) -> dict[str, object]:
    display = dict(row)
    percent_fields = [
        "雙向換手率",
        "買賣皆活躍事件率",
        "淨方向集中率",
        "1日金額權重報酬",
        "3日金額權重報酬",
        "5日金額權重報酬",
        "10日金額權重報酬",
        "20日金額權重報酬",
        "20日命中率",
        "20日買超報酬",
        "20日賣超報酬",
        "20日較佳方向報酬",
        "平均雙向換手率",
        "平均1日金額權重報酬",
        "平均5日金額權重報酬",
        "平均10日金額權重報酬",
        "平均20日金額權重報酬",
    ]
    score_fields = ["當沖分數", "隔日衝分數", "波段分數", "長線分數", "平均風格分數", "風格分數"]
    count_fields = ["可用事件數", "平均可用事件數", "分點數", "重疊分點數"]
    for field_name in percent_fields:
        if field_name in display:
            display[field_name] = fmt_pct(parse_float(str(display[field_name])))
    for field_name in score_fields:
        if field_name in display:
            display[field_name] = fmt_score(parse_float(str(display[field_name])))
    for field_name in count_fields:
        if field_name in display:
            display[field_name] = fmt_num(parse_float(str(display[field_name])))
    if "平均每事件成交張數" in display:
        display["平均每事件成交張數"] = fmt_num(parse_float(str(display["平均每事件成交張數"])))
    return display


def render_html(
    rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    top_rows: list[dict[str, object]],
    overlap: list[dict[str, object]],
    metadata: dict[str, object],
    output_path: Path,
) -> None:
    eligible_rows = [row for row in rows if row["樣本狀態"] == "可分類"]
    unclassified_count = sum(1 for row in eligible_rows if row["風格標籤"] == "未明顯歸類")
    style_counts = {row["風格"]: parse_int(str(row.get("分點數", 0))) for row in summary_rows}
    top_by_style = {
        style: [format_display_row(row) for row in top_rows if row["風格"] == style][:30]
        for style in ("當沖傾向", "隔日衝", "波段分點", "長線傾向")
    }
    complete_rows = [format_display_row(row) for row in sorted(rows, key=lambda item: item["分點名稱"])]
    summary_display = [format_display_row(row) for row in summary_rows]
    overlap_display = [format_display_row(row) for row in overlap]

    summary_headers = [
        "風格",
        "分點數",
        "平均可用事件數",
        "平均風格分數",
        "平均雙向換手率",
        "平均1日金額權重報酬",
        "平均5日金額權重報酬",
        "平均10日金額權重報酬",
        "平均20日金額權重報酬",
        "代表排序指標",
    ]
    top_headers = [
        "排名",
        "分點名稱",
        "分點類別",
        "風格標籤",
        "風格分數",
        "可用事件數",
        "雙向換手率",
        "買賣皆活躍事件率",
        "1日金額權重報酬",
        "5日金額權重報酬",
        "10日金額權重報酬",
        "20日金額權重報酬",
        "20日命中率",
        "20日較佳方向",
        "20日較佳方向報酬",
    ]
    complete_headers = [
        "分點名稱",
        "分點類別",
        "樣本狀態",
        "風格標籤",
        "當沖分數",
        "隔日衝分數",
        "波段分數",
        "長線分數",
        "可用事件數",
        "雙向換手率",
        "買賣皆活躍事件率",
        "1日金額權重報酬",
        "5日金額權重報酬",
        "10日金額權重報酬",
        "20日金額權重報酬",
        "20日命中率",
        "20日較佳方向",
    ]
    numeric_cols = {
        "排名",
        "分點數",
        "平均可用事件數",
        "平均風格分數",
        "平均雙向換手率",
        "平均1日金額權重報酬",
        "平均5日金額權重報酬",
        "平均10日金額權重報酬",
        "平均20日金額權重報酬",
        "風格分數",
        "可用事件數",
        "雙向換手率",
        "買賣皆活躍事件率",
        "1日金額權重報酬",
        "5日金額權重報酬",
        "10日金額權重報酬",
        "20日金額權重報酬",
        "20日命中率",
        "20日較佳方向報酬",
        "當沖分數",
        "隔日衝分數",
        "波段分數",
        "長線分數",
        "重疊分點數",
    }

    bars = []
    max_count = max(style_counts.get(style, 0) for style in ("當沖傾向", "隔日衝", "波段分點", "長線傾向")) or 1
    for style in ("當沖傾向", "隔日衝", "波段分點", "長線傾向"):
        count = style_counts.get(style, 0)
        width = count / max_count * 100
        bars.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{html.escape(style)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
              <div class="bar-value">{fmt_num(count)}</div>
            </div>
            """
        )

    sections = []
    for style, style_rows in top_by_style.items():
        sections.append(
            f"""
            <section>
              <h2>{html.escape(style)}代表分點</h2>
              {table(top_headers, style_rows, numeric_cols)}
            </section>
            """
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>分點交易風格分類報告</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #5c6672;
      --line: #d8dee7;
      --soft: #f5f7fa;
      --accent: #2563eb;
      --accent-2: #0f766e;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
      line-height: 1.5;
    }}
    header {{
      padding: 28px 32px 20px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }}
    main {{ padding: 24px 32px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 20px 0 10px; font-size: 16px; letter-spacing: 0; }}
    p, li {{ color: var(--muted); }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
    }}
    .kpis {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin: 18px 0 18px;
    }}
    .kpi {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .kpi .label {{ color: var(--muted); font-size: 13px; }}
    .kpi .value {{ font-size: 24px; font-weight: 700; margin-top: 2px; }}
    .note {{
      border-left: 4px solid var(--accent-2);
      background: #f0fdfa;
      padding: 12px 14px;
      margin: 14px 0;
      color: #115e59;
    }}
    .rules {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin-top: 10px;
    }}
    .rule {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--soft);
    }}
    .rule strong {{ display: block; margin-bottom: 4px; }}
    .bars {{ max-width: 760px; margin: 10px 0 20px; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 88px minmax(160px, 1fr) 70px;
      gap: 10px;
      align-items: center;
      margin: 8px 0;
    }}
    .bar-label {{ font-size: 14px; color: var(--muted); }}
    .bar-track {{ height: 14px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
    .bar-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 980px;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eef2f7;
      z-index: 1;
      color: #334155;
    }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tbody tr:hover {{ background: #f8fafc; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
    }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .grid-2 {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 78px minmax(120px, 1fr) 58px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>分點交易風格分類報告</h1>
    <p>以 Fubon 分點排名資料的分點日淨買賣訊號，對照 1/3/5/10/20 日後調整收盤價變化，並加入同日買賣雙向換手特徵。這是分點層級的訊號分類，不是單一帳戶交易行為或真實庫存損益。</p>
    <div class="meta">
      <span class="pill">產出時間：{html.escape(str(metadata["generated_at"]))}</span>
      <span class="pill">可分類門檻：{fmt_num(metadata["min_events"])} 事件</span>
      <span class="pill">風格分數門檻：{fmt_score(float(metadata["score_threshold"]))}</span>
      <span class="pill">長線以 20 日代理</span>
    </div>
  </header>
  <main>
    <section>
      <div class="kpis">
        <div class="kpi"><div class="label">總分點數</div><div class="value">{fmt_num(metadata["total_branches"])}</div></div>
        <div class="kpi"><div class="label">可分類分點</div><div class="value">{fmt_num(metadata["eligible_branches"])}</div></div>
        <div class="kpi"><div class="label">樣本不足</div><div class="value">{fmt_num(metadata["small_branches"])}</div></div>
        <div class="kpi"><div class="label">未明顯歸類</div><div class="value">{fmt_num(unclassified_count)}</div></div>
      </div>
      <div class="bars">{"".join(bars)}</div>
      <div class="note">同一分點可以同時符合多個風格；例如短線雙向換手高，同時 10/20 日決策後報酬也排名前段，就會同時被標成當沖傾向與波段或長線傾向。</div>
    </section>

    <section>
      <h2>分類規則</h2>
      <div class="rules">
        <div class="rule"><strong>當沖傾向</strong><span>雙向換手率、買賣皆活躍事件率、平均每事件成交張數的綜合分數達門檻。代表分點內同日買賣都活躍，不能直接等同同一帳戶當沖。</span></div>
        <div class="rule"><strong>隔日衝</strong><span>1 日金額權重決策報酬與短期優勢分數達門檻，且 1 日報酬為正。</span></div>
        <div class="rule"><strong>波段分點</strong><span>5 或 10 日決策報酬與相對 1/20 日的中期優勢分數達門檻，且中期報酬為正。</span></div>
        <div class="rule"><strong>長線傾向</strong><span>20 日決策報酬與延伸優勢分數達門檻，20 日報酬為正且命中率至少 50%。目前資料只評估到 20 個交易日。</span></div>
      </div>
    </section>

    <section>
      <h2>風格摘要</h2>
      <div class="table-wrap">{table(summary_headers, summary_display, numeric_cols)}</div>
    </section>

    {"".join(sections)}

    <section class="grid-2">
      <div>
        <h2>風格重疊</h2>
        <div class="table-wrap">{table(["風格A", "風格B", "重疊分點數"], overlap_display, numeric_cols)}</div>
      </div>
      <div>
        <h2>讀法提醒</h2>
        <p>金額權重報酬是用分點該筆淨買賣超金額當權重，買超後上漲為正，賣超後下跌也為正。它衡量的是分點日訊號後的方向性結果，不是實際持倉成本、稅費或成交可得性。</p>
        <p>當沖傾向使用同日買進與賣出同時出現的程度估算。分點資料是分點彙總，不能拆到同一自然人或同一法人帳戶，因此應視為「短線換手風格」的代理指標。</p>
      </div>
    </section>

    <section>
      <h2>完整分點分類</h2>
      <div class="table-wrap">{table(complete_headers, complete_rows, numeric_cols)}</div>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    branch_performance = load_branch_performance(args.branch_performance)
    load_branch_side_performance(args.branch_side_performance, branch_performance)

    if args.reuse_churn_cache and args.churn_metrics.exists():
        churn_metrics = read_churn_cache(args.churn_metrics)
        churn_source = "cache"
    else:
        churn_metrics = scan_churn_metrics(args.broker_dir)
        write_churn_metrics(args.churn_metrics, churn_metrics)
        churn_source = "raw_scan"

    excluded_small = load_excluded_small_branches(args.excluded_small_branches)
    style_rows = build_style_rows(branch_performance, churn_metrics, args.min_events, args.style_score_threshold)
    append_small_branch_rows(style_rows, excluded_small, churn_metrics)
    summary = summarize_styles(style_rows)
    top_rows = top_style_rows(style_rows)
    overlap = overlap_rows(style_rows)

    by_branch_fields = [
        "分點名稱",
        "分點類別",
        "樣本狀態",
        "風格標籤",
        "當沖傾向",
        "隔日衝",
        "波段分點",
        "長線傾向",
        "當沖分數",
        "隔日衝分數",
        "波段分數",
        "長線分數",
        "可用事件數",
        "雙向換手率",
        "買賣皆活躍事件率",
        "淨方向集中率",
        "平均每事件成交張數",
        "1日金額權重報酬",
        "3日金額權重報酬",
        "5日金額權重報酬",
        "10日金額權重報酬",
        "20日金額權重報酬",
        "20日命中率",
        "20日買超報酬",
        "20日賣超報酬",
        "20日較佳方向",
        "20日較佳方向報酬",
    ]
    summary_fields = [
        "風格",
        "分點數",
        "平均可用事件數",
        "平均風格分數",
        "平均雙向換手率",
        "平均1日金額權重報酬",
        "平均5日金額權重報酬",
        "平均10日金額權重報酬",
        "平均20日金額權重報酬",
        "代表排序指標",
    ]
    top_fields = [
        "風格",
        "排名",
        "分點名稱",
        "分點類別",
        "風格標籤",
        "風格分數",
        "可用事件數",
        "雙向換手率",
        "買賣皆活躍事件率",
        "1日金額權重報酬",
        "5日金額權重報酬",
        "10日金額權重報酬",
        "20日金額權重報酬",
        "20日命中率",
        "20日較佳方向",
        "20日較佳方向報酬",
    ]

    write_csv(STYLE_BY_BRANCH_CSV, by_branch_fields, style_rows)
    write_csv(STYLE_SUMMARY_CSV, summary_fields, summary)
    write_csv(STYLE_TOP_BRANCHES_CSV, top_fields, top_rows)
    write_csv(STYLE_OVERLAP_CSV, ["風格A", "風格B", "重疊分點數"], overlap)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_branches": len(style_rows),
        "eligible_branches": sum(1 for row in style_rows if row["樣本狀態"] == "可分類"),
        "small_branches": sum(1 for row in style_rows if row["樣本狀態"] == "樣本不足"),
        "min_events": args.min_events,
        "score_threshold": args.style_score_threshold,
        "churn_source": churn_source,
        "outputs": [
            str(STYLE_BY_BRANCH_CSV),
            str(STYLE_SUMMARY_CSV),
            str(STYLE_TOP_BRANCHES_CSV),
            str(STYLE_OVERLAP_CSV),
            str(CHURN_METRICS_CSV),
            str(STYLE_HTML),
        ],
    }
    render_html(style_rows, summary, top_rows, overlap, metadata, STYLE_HTML)
    print(metadata)


if __name__ == "__main__":
    main()
