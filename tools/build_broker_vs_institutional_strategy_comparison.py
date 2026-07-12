from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from backtest_top_broker_following_strategy import (
    BRANCH_PERFORMANCE_CSV,
    BROKER_DIR,
    METADATA_PATH,
    OUTPUT_DIR,
    PRICE_DIR,
    ROUND_TRIP_COST_RATE,
    VIZ_DIR,
    enrich_signals,
    forward_open_return,
    load_branch_signals,
    load_listed_common_codes,
    load_price_series,
    parse_float,
    parse_int,
    pct,
    price_paths_by_code,
    select_daily_signals,
    select_top_branches,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTITUTIONAL_METRICS_CSV = PROJECT_ROOT / "output" / "institutional_flow_backtest" / "strategy_metrics.csv"
INSTITUTIONAL_REPORT = PROJECT_ROOT / "data_viz" / "institutional_flow_backtest" / "summary.html"
BROKER_REPORT = PROJECT_ROOT / "data_viz" / "broker" / "all_broker_decision_performance.html"

REPORT_HTML = VIZ_DIR / "broker_vs_institutional_following_comparison.html"
COMPARISON_CSV = OUTPUT_DIR / "broker_vs_institutional_following_comparison.csv"
BEST_CSV = OUTPUT_DIR / "broker_vs_institutional_following_best.csv"
METADATA_JSON = OUTPUT_DIR / "broker_vs_institutional_following_metadata.json"
EXCLUDED_BRANCHES_CSV = OUTPUT_DIR / "broker_vs_institutional_following_excluded_branches.csv"
EXCLUDED_SMALL_BRANCHES_CSV = OUTPUT_DIR / "all_broker_excluded_small_branches.csv"
CITY_EXCLUDED_BRANCHES_CSV = OUTPUT_DIR / "all_broker_city_excluded_branches.csv"

HOLDING_DAYS = (1, 5, 10, 20, 60)
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BrokerTrade:
    branch: str
    signal_date: str
    holding_days: int
    gross_return: float
    net_return: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare broker-branch following results with prior institutional-flow backtests.")
    parser.add_argument("--broker-dir", type=Path, default=BROKER_DIR)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--branch-performance", type=Path, default=BRANCH_PERFORMANCE_CSV)
    parser.add_argument("--institutional-metrics", type=Path, default=INSTITUTIONAL_METRICS_CSV)
    parser.add_argument(
        "--top-branches",
        type=int,
        default=0,
        help="Use 0 to test all valid broker branches after the all-broker small-branch filter.",
    )
    parser.add_argument("--selection-horizon", type=int, default=20)
    parser.add_argument("--broker-top-n", type=int, default=30)
    parser.add_argument("--chart-top-n", type=int, default=50)
    parser.add_argument("--min-broker-signal-days", type=int, default=250)
    parser.add_argument("--min-broker-trades", type=int, default=5000)
    parser.add_argument(
        "--include-small-branches",
        action="store_true",
        help="Bypass the all-broker small-branch filter and test every broker branch CSV.",
    )
    parser.add_argument(
        "--include-stopped-branches",
        action="store_true",
        help="Include branches marked as stopped/old in names or all-broker city exclusions.",
    )
    parser.add_argument(
        "--include-main-branches",
        action="store_true",
        help="Include head-office/main-code branches excluded by the all-broker city report.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else math.nan


def median(values: list[float]) -> float:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return math.nan
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2


def positive_ratio(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(1 for value in clean if value > 0) / len(clean) if clean else math.nan


def profit_factor(values: list[float]) -> float:
    gains = sum(value for value in values if math.isfinite(value) and value > 0)
    losses = sum(value for value in values if math.isfinite(value) and value < 0)
    return gains / abs(losses) if losses < 0 else math.nan


def annualized_basket_return(avg_basket_return: float, holding_days: int) -> float:
    if not math.isfinite(avg_basket_return) or avg_basket_return <= -1:
        return math.nan
    return (1.0 + avg_basket_return) ** (TRADING_DAYS_PER_YEAR / holding_days) - 1.0


def fmt_pct(value: object, digits: int = 2) -> str:
    number = parse_float(value)
    return "" if not math.isfinite(number) else f"{number * 100:.{digits}f}%"


def fmt_num(value: object, digits: int = 2) -> str:
    number = parse_float(value)
    return "" if not math.isfinite(number) else f"{number:,.{digits}f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_institutional_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("ParticipantKey") not in {"foreign", "trust", "dealer"}:
                continue
            if parse_int(row.get("HoldingDays")) not in HOLDING_DAYS:
                continue
            rows.append(
                {
                    "來源": "法人舊回測",
                    "群組": row.get("Participant", ""),
                    "策略說明": "三大法人買超強度 long-only top30，沿用 institutional_flow_backtest",
                    "訊號窗": parse_int(row.get("SignalWindow")),
                    "持有天數": parse_int(row.get("HoldingDays")),
                    "交易筆數": parse_int(row.get("TradeCount")),
                    "訊號日數": parse_int(row.get("SignalDayCount")),
                    "每日平均檔數": pct(parse_float(row.get("AvgPicksPerSignalDay"))),
                    "平均單筆毛報酬": pct(parse_float(row.get("AvgGrossReturn"))),
                    "平均單筆淨報酬": pct(parse_float(row.get("AvgNetReturn"))),
                    "單筆勝率": pct(parse_float(row.get("WinRate"))),
                    "ProfitFactor": pct(parse_float(row.get("ProfitFactor"))),
                    "Basket平均淨報酬": pct(parse_float(row.get("BasketAvgNetReturn"))),
                    "Basket中位數淨報酬": pct(parse_float(row.get("BasketMedianNetReturn"))),
                    "Basket勝率": pct(parse_float(row.get("BasketWinRate"))),
                    "年化Basket平均報酬": pct(parse_float(row.get("AnnualizedAvgBasketReturn"))),
                }
            )
    return rows


def load_valid_branch_names(path: Path, horizon: int) -> list[str]:
    branches: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if parse_int(row.get("觀察期交易日")) != horizon:
                continue
            branch = str(row.get("分點名稱") or "").strip()
            if branch and branch not in seen:
                seen.add(branch)
                branches.append(branch)
    return branches


def load_excluded_small_branch_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"broker_excluded_small_branch_file_count": 0, "broker_excluded_small_branch_threshold": math.nan}
    names: set[str] = set()
    thresholds: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            branch = str(row.get("分點名稱") or "").strip()
            if branch:
                names.add(branch)
            threshold = parse_int(row.get("事件數門檻"))
            if threshold:
                thresholds.add(threshold)
    return {
        "broker_excluded_small_branch_file_count": len(names),
        "broker_excluded_small_branch_threshold": min(thresholds) if thresholds else math.nan,
    }


def load_city_excluded_branch_reasons(path: Path) -> dict[str, str]:
    reasons: dict[str, str] = {}
    if not path.exists():
        return reasons
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            branch = str(row.get("分點名稱") or "").strip()
            reason = str(row.get("原因") or "").strip()
            if branch and reason:
                reasons[branch] = reason
    return reasons


def is_stopped_branch_name(branch: str) -> bool:
    return "(停)" in branch or "（停）" in branch


def apply_static_branch_filters(branches: list[str], args: argparse.Namespace) -> tuple[list[str], list[dict[str, object]]]:
    city_excluded_reasons = load_city_excluded_branch_reasons(CITY_EXCLUDED_BRANCHES_CSV)
    filtered: list[str] = []
    excluded: list[dict[str, object]] = []
    for branch in branches:
        city_reason = city_excluded_reasons.get(branch, "")
        if not args.include_stopped_branches and (is_stopped_branch_name(branch) or "停業/舊分點" in city_reason):
            excluded.append({"分點名稱": branch, "排除原因": "停業/舊分點", "訊號日數": "", "交易筆數": "", "來源": "static"})
            continue
        if not args.include_main_branches and "總公司/主分點或總號" in city_reason:
            excluded.append({"分點名稱": branch, "排除原因": "總公司/主分點或總號", "訊號日數": "", "交易筆數": "", "來源": "static"})
            continue
        filtered.append(branch)
    return filtered, excluded


def select_broker_branches(args: argparse.Namespace) -> tuple[list[str], str, dict[str, object], list[dict[str, object]]]:
    all_branch_csvs = [path.stem for path in sorted(args.broker_dir.glob("*.csv"))]
    all_branch_set = set(all_branch_csvs)
    excluded_summary = load_excluded_small_branch_summary(EXCLUDED_SMALL_BRANCHES_CSV)

    if args.top_branches > 0:
        selected = select_top_branches(args.branch_performance, args.top_branches, args.selection_horizon)
        branches = [str(row["分點名稱"]) for row in selected]
        branches, static_excluded = apply_static_branch_filters(branches, args)
        return branches, f"Top{args.top_branches} 分點之一", {
            "broker_branch_csv_count": len(all_branch_csvs),
            "broker_branch_filter": f"Top{args.top_branches} by {args.selection_horizon}D all-broker performance",
            "broker_excluded_small_branches": max(0, len(all_branch_csvs) - len(branches)),
            "broker_excluded_static_branches": len(static_excluded),
            "broker_excluded_stopped_or_old_branches": sum(1 for row in static_excluded if row["排除原因"] == "停業/舊分點"),
            "broker_excluded_main_branches": sum(1 for row in static_excluded if row["排除原因"] == "總公司/主分點或總號"),
            **excluded_summary,
        }, static_excluded
    if args.include_small_branches:
        branches, static_excluded = apply_static_branch_filters(all_branch_csvs, args)
        return branches, "全部分點（含小樣本）", {
            "broker_branch_csv_count": len(all_branch_csvs),
            "broker_branch_filter": "none",
            "broker_excluded_small_branches": 0,
            "broker_excluded_static_branches": len(static_excluded),
            "broker_excluded_stopped_or_old_branches": sum(1 for row in static_excluded if row["排除原因"] == "停業/舊分點"),
            "broker_excluded_main_branches": sum(1 for row in static_excluded if row["排除原因"] == "總公司/主分點或總號"),
            **excluded_summary,
        }, static_excluded

    valid_branches = [branch for branch in load_valid_branch_names(args.branch_performance, args.selection_horizon) if branch in all_branch_set]
    excluded_branches = sorted(all_branch_set - set(valid_branches))
    branches, static_excluded = apply_static_branch_filters(valid_branches, args)
    return branches, "全部有效分點（沿用舊報告小樣本剔除、停業/總號剔除）", {
        "broker_branch_csv_count": len(all_branch_csvs),
        "broker_branch_filter": f"all_broker_performance valid branches at {args.selection_horizon}D",
        "broker_excluded_small_branches": len(excluded_branches),
        "broker_excluded_small_branch_names": excluded_branches,
        "broker_excluded_static_branches": len(static_excluded),
        "broker_excluded_stopped_or_old_branches": sum(1 for row in static_excluded if row["排除原因"] == "停業/舊分點"),
        "broker_excluded_main_branches": sum(1 for row in static_excluded if row["排除原因"] == "總公司/主分點或總號"),
        **excluded_summary,
    }, static_excluded


def build_broker_trades(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[str], dict[str, object]]:
    allowed_codes, _metadata = load_listed_common_codes(args.metadata)
    price_paths = price_paths_by_code(args.price_dir)
    branches, branch_selection_label, selection_metadata, excluded_branch_rows = select_broker_branches(args)
    price_cache: dict[str, object | None] = {}

    rows: list[dict[str, object]] = []
    included_branches: list[str] = []
    raw_count = 0
    enriched_count = 0
    buy_signal_count = 0
    selected_count = 0
    branches_with_buy_signals = 0
    branches_with_trades = 0
    for branch_index, branch in enumerate(branches, 1):
        raw_signals = load_branch_signals(args.broker_dir, branch, allowed_codes)
        raw_count += len(raw_signals)
        for code in {signal.code for signal in raw_signals}:
            if code not in price_cache:
                path = price_paths.get(code)
                price_cache[code] = load_price_series(path) if path else None
        prices = {code: series for code, series in price_cache.items() if series is not None}
        enriched = enrich_signals(raw_signals, prices)
        enriched_count += len(enriched)
        buy_signals = [signal for signal in enriched if signal.signal_notional > 0]
        buy_signal_count += len(buy_signals)
        if buy_signals:
            branches_with_buy_signals += 1
        selected_signals = select_daily_signals(buy_signals, args.broker_top_n)
        selected_count += len(selected_signals)
        selected_signal_days = len({signal.date for signal in selected_signals})
        if selected_signal_days < args.min_broker_signal_days or len(selected_signals) < args.min_broker_trades:
            reasons: list[str] = []
            if selected_signal_days < args.min_broker_signal_days:
                reasons.append(f"訊號日數 {selected_signal_days} < {args.min_broker_signal_days}")
            if len(selected_signals) < args.min_broker_trades:
                reasons.append(f"交易筆數 {len(selected_signals)} < {args.min_broker_trades}")
            excluded_branch_rows.append(
                {
                    "分點名稱": branch,
                    "排除原因": "；".join(reasons),
                    "訊號日數": selected_signal_days,
                    "交易筆數": len(selected_signals),
                    "來源": "activity",
                }
            )
            if branch_index % 50 == 0 or branch_index == len(branches):
                print(f"processed {branch_index}/{len(branches)} broker branches", flush=True)
            continue

        included_branches.append(branch)
        branch_has_trades = False
        for holding_days in HOLDING_DAYS:
            trades: list[BrokerTrade] = []
            for signal in selected_signals:
                price = prices.get(signal.code)
                if not price:
                    continue
                forward = forward_open_return(price, signal.date, holding_days)
                if not forward:
                    continue
                _entry_date, _exit_date, raw_return = forward
                gross_return = raw_return
                trades.append(
                    BrokerTrade(
                        branch=branch,
                        signal_date=signal.date,
                        holding_days=holding_days,
                        gross_return=gross_return,
                        net_return=gross_return - ROUND_TRIP_COST_RATE,
                    )
                )
            if trades:
                branch_has_trades = True
            rows.append(summarize_broker_trades(branch, holding_days, trades, args.broker_top_n, branch_selection_label))
        if branch_has_trades:
            branches_with_trades += 1
        if branch_index % 50 == 0 or branch_index == len(branches):
            print(f"processed {branch_index}/{len(branches)} broker branches", flush=True)

    metadata = {
        "broker_allowed_codes": len(allowed_codes),
        "broker_price_series": sum(1 for series in price_cache.values() if series is not None),
        "broker_raw_signals": raw_count,
        "broker_enriched_signals": enriched_count,
        "broker_buy_signals": buy_signal_count,
        "broker_selected_signals": selected_count,
        "broker_branch_selection": branch_selection_label,
        "broker_branch_candidates": len(branches),
        "broker_branches_after_activity_filter": len(included_branches),
        "broker_branches_with_buy_signals": branches_with_buy_signals,
        "broker_branches_with_trades": branches_with_trades,
        "broker_min_signal_days": args.min_broker_signal_days,
        "broker_min_trades": args.min_broker_trades,
        "broker_excluded_low_activity_branches": sum(1 for row in excluded_branch_rows if row["來源"] == "activity"),
        "broker_excluded_branch_rows": excluded_branch_rows,
        **selection_metadata,
    }
    return rows, included_branches, metadata


def summarize_broker_trades(branch: str, holding_days: int, trades: list[BrokerTrade], top_n: int, branch_selection_label: str) -> dict[str, object]:
    gross_returns = [trade.gross_return for trade in trades]
    net_returns = [trade.net_return for trade in trades]
    by_date: dict[str, list[BrokerTrade]] = defaultdict(list)
    for trade in trades:
        by_date[trade.signal_date].append(trade)
    basket_net = [mean([trade.net_return for trade in day_trades]) for day_trades in by_date.values()]

    return {
        "來源": "分點跟單",
        "群組": branch,
        "策略說明": f"{branch_selection_label}；每分點每日買超金額 top{top_n}，只做多、不做空",
        "訊號窗": 1,
        "持有天數": holding_days,
        "交易筆數": len(trades),
        "訊號日數": len(by_date),
        "每日平均檔數": pct(len(trades) / len(by_date) if by_date else math.nan),
        "平均單筆毛報酬": pct(mean(gross_returns)),
        "平均單筆淨報酬": pct(mean(net_returns)),
        "單筆勝率": pct(positive_ratio(net_returns)),
        "ProfitFactor": pct(profit_factor(net_returns)),
        "Basket平均淨報酬": pct(mean(basket_net)),
        "Basket中位數淨報酬": pct(median(basket_net)),
        "Basket勝率": pct(positive_ratio(basket_net)),
        "年化Basket平均報酬": pct(annualized_basket_return(mean(basket_net), holding_days)),
    }


def best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["來源"]), str(row["群組"]))].append(row)
    best: list[dict[str, object]] = []
    for (_source, _group), items in grouped.items():
        best.append(max(items, key=sort_return_value))
    return sorted(best, key=sort_return_value, reverse=True)


def sort_return_value(row: dict[str, object]) -> float:
    value = parse_float(row.get("Basket平均淨報酬"))
    return value if math.isfinite(value) else -math.inf


def ranked_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for rank, row in enumerate(sorted(rows, key=sort_return_value, reverse=True), start=1):
        copied = dict(row)
        copied["排名"] = rank
        ranked.append(copied)
    return ranked


def h(value: object) -> str:
    return html.escape(str(value), quote=True)


def display_value(key: str, value: object) -> str:
    if value in (None, ""):
        return ""
    if key in {"排名", "訊號窗", "持有天數", "交易筆數", "訊號日數"}:
        return f"{parse_int(value):,}"
    if key == "每日平均檔數":
        return fmt_num(parse_float(value), 1)
    if key in {
        "平均單筆毛報酬",
        "平均單筆淨報酬",
        "單筆勝率",
        "Basket平均淨報酬",
        "Basket中位數淨報酬",
        "Basket勝率",
        "年化Basket平均報酬",
    }:
        return fmt_pct(value)
    if key == "ProfitFactor":
        return fmt_num(value, 2)
    return str(value)


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


def bar_svg(rows: list[dict[str, object]], title: str, limit: int) -> str:
    rows = rows[:limit]
    if not rows:
        return "<div class='empty'>沒有圖表資料</div>"
    values = [parse_float(row["Basket平均淨報酬"]) for row in rows]
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return "<div class='empty'>沒有圖表資料</div>"
    width = 980
    row_h = 28
    left = 150
    right = 88
    top = 34
    height = top + row_h * len(rows) + 28
    min_v = min(min(clean), 0.0)
    max_v = max(max(clean), 0.0)
    if max_v == min_v:
        max_v += 0.01
        min_v -= 0.01
    plot_w = width - left - right
    zero_x = left + (0.0 - min_v) / (max_v - min_v) * plot_w
    parts = [
        f'<text x="{left}" y="18">{h(title)}</text>',
        f'<line x1="{zero_x:.1f}" y1="{top-8}" x2="{zero_x:.1f}" y2="{height-20}" class="base"/>',
    ]
    for idx, row in enumerate(rows):
        value = parse_float(row["Basket平均淨報酬"])
        y = top + idx * row_h
        x = left + (min(0.0, value) - min_v) / (max_v - min_v) * plot_w
        bar_w = abs(value) / (max_v - min_v) * plot_w
        klass = "pos" if value >= 0 else "neg"
        label = f"{row['群組']} S{row['訊號窗']}/{row['持有天數']}D"
        parts.append(f'<text x="8" y="{y+15}">{h(label)}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="18" class="{klass}"/>')
        label_x = x + bar_w + 5 if value >= 0 else max(4, x - 58)
        parts.append(f'<text x="{label_x:.1f}" y="{y+14}">{h(fmt_pct(value))}</text>')
    return f'<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{h(title)}">' + "".join(parts) + "</svg>"


def horizon_sections(rows: list[dict[str, object]], headers: list[str], chart_top_n: int) -> str:
    sections: list[str] = []
    for holding_days in HOLDING_DAYS:
        horizon_rows = ranked_rows([row for row in rows if parse_int(row["持有天數"]) == holding_days])
        sections.append(
            f"""
    <section class="section">
      <h2>{holding_days}D 持有期排名</h2>
      {bar_svg(horizon_rows, f"{holding_days}D Basket 平均淨報酬 Top {chart_top_n}", chart_top_n)}
      {render_table(headers, horizon_rows, chart_top_n)}
    </section>
"""
        )
    return "".join(sections)


def render_html(rows: list[dict[str, object]], best: list[dict[str, object]], metadata: dict[str, object]) -> str:
    headers = [
        "排名",
        "來源",
        "群組",
        "訊號窗",
        "持有天數",
        "交易筆數",
        "訊號日數",
        "每日平均檔數",
        "平均單筆淨報酬",
        "單筆勝率",
        "Basket平均淨報酬",
        "Basket勝率",
        "年化Basket平均報酬",
        "策略說明",
    ]
    overall_best = best[0] if best else {}
    broker_best = next((row for row in best if row["來源"] == "分點跟單"), {})
    inst_best = next((row for row in best if row["來源"] == "法人舊回測"), {})
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>全有效分點跟單 vs 三大法人回測排名</title>
  <style>
    :root {{ --ink:#17212b; --muted:#5f6b76; --line:#d8dee6; --panel:#fff; --bg:#f6f7f9; --pos:#0f766e; --neg:#b42318; }}
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
    .note {{ margin-bottom:18px; padding:14px 16px; background:#fff7ed; border:1px solid #fed7aa; border-radius:8px; color:#7c2d12; }}
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
    .bar-chart .pos {{ fill:var(--pos); }}
    .bar-chart .neg {{ fill:var(--neg); }}
    .bar-chart .base {{ stroke:#94a3b8; stroke-dasharray:4 4; }}
    .bar-chart text {{ font-size:12px; fill:var(--muted); }}
    @media(max-width:900px) {{ header,main {{ padding-left:16px; padding-right:16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>全有效分點跟單 vs 三大法人回測排名</h1>
    <p>法人沿用既有 institutional_flow_backtest 的三大法人 long-only top30 買超籃子；分點改為沿用舊全分點績效報告的小樣本剔除後，逐一測試 Fubon by_broker 有效分點，且只跟買超、不跟賣超。</p>
    <div class="meta">
      <span class="pill">產生時間：{h(metadata["generated_at"])}</span>
      <span class="pill">原始分點CSV：{fmt_num(metadata.get("broker_branch_csv_count", metadata["broker_branch_candidates"]), 0)} 個</span>
      <span class="pill">剔除小樣本：{fmt_num(metadata.get("broker_excluded_small_branches", 0), 0)} 個</span>
      <span class="pill">小樣本門檻：可用事件數 {fmt_num(metadata.get("broker_excluded_small_branch_threshold", math.nan), 0)}</span>
      <span class="pill">剔除停業/總號：{fmt_num(metadata.get("broker_excluded_static_branches", 0), 0)} 個</span>
      <span class="pill">低活躍門檻：{fmt_num(metadata.get("broker_min_signal_days", 0), 0)} 日 / {fmt_num(metadata.get("broker_min_trades", 0), 0)} 筆</span>
      <span class="pill">剔除低活躍：{fmt_num(metadata.get("broker_excluded_low_activity_branches", 0), 0)} 個</span>
      <span class="pill">有效分點候選：{fmt_num(metadata["broker_branch_candidates"], 0)} 個</span>
      <span class="pill">納入回測分點：{fmt_num(metadata.get("broker_branches_after_activity_filter", metadata["broker_branch_candidates"]), 0)} 個</span>
      <span class="pill">有交易分點：{fmt_num(metadata["broker_branches_with_trades"], 0)} 個</span>
      <span class="pill">分點每日 topN：{metadata["broker_top_n"]}</span>
      <span class="pill">圖表 TopN：{metadata["chart_top_n"]}</span>
      <span class="pill">費稅 round-trip：{fmt_pct(ROUND_TRIP_COST_RATE)}</span>
      <span class="pill">法人來源：{h(INSTITUTIONAL_METRICS_CSV.relative_to(PROJECT_ROOT))}</span>
    </div>
  </header>
  <main>
    <div class="note">注意：兩邊交易時間與費稅口徑相近，都是訊號日後用 next-open 到固定持有期 exit-open，且現在都只做 long-only。差異在於三大法人用全市場買超強度排序，分點用單一分點買超金額排序。本版不再先挑前 50 名分點，而是沿用舊全分點績效報告的小樣本剔除，再排除停業/舊分點、總公司/主分點或總號，以及訊號日數或交易筆數不足的低活躍分點；畫面上的圖表為各週期 Top {metadata["chart_top_n"]}，完整排名在 CSV。</div>

    <section class="section">
      <h2>摘要</h2>
      <div class="cards">
        <div class="card"><strong>{h(overall_best.get("群組", ""))}</strong><span>全表最佳 Basket 平均淨報酬</span></div>
        <div class="card"><strong>{display_value("Basket平均淨報酬", overall_best.get("Basket平均淨報酬", ""))}</strong><span>最佳 Basket 平均淨報酬</span></div>
        <div class="card"><strong>{h(inst_best.get("群組", ""))}</strong><span>最佳法人組合</span></div>
        <div class="card"><strong>{h(broker_best.get("群組", ""))}</strong><span>最佳分點組合</span></div>
      </div>
    </section>

    {horizon_sections(rows, headers, int(metadata["chart_top_n"]))}

    <section class="section">
      <h2>各分點/法人最佳參數排名</h2>
      {render_table(headers, best)}
    </section>

    <section class="section">
      <h2>全部策略列排名</h2>
      {render_table(headers, rows)}
    </section>

    <section class="section">
      <h2>參考報告</h2>
      <p>法人舊回測：<code>{h(INSTITUTIONAL_REPORT.relative_to(PROJECT_ROOT))}</code><br>
      分點來源：<code>{h(BROKER_DIR.relative_to(PROJECT_ROOT))}</code></p>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    institutional_rows = load_institutional_rows(args.institutional_metrics)
    broker_rows, branches, broker_meta = build_broker_trades(args)
    excluded_branch_rows = broker_meta.pop("broker_excluded_branch_rows", [])
    rows = ranked_rows(institutional_rows + broker_rows)
    best = ranked_rows(best_rows(rows))

    headers = [
        "排名",
        "來源",
        "群組",
        "策略說明",
        "訊號窗",
        "持有天數",
        "交易筆數",
        "訊號日數",
        "每日平均檔數",
        "平均單筆毛報酬",
        "平均單筆淨報酬",
        "單筆勝率",
        "ProfitFactor",
        "Basket平均淨報酬",
        "Basket中位數淨報酬",
        "Basket勝率",
        "年化Basket平均報酬",
    ]
    write_csv(COMPARISON_CSV, headers, rows)
    write_csv(BEST_CSV, headers, best)
    write_csv(EXCLUDED_BRANCHES_CSV, ["分點名稱", "排除原因", "訊號日數", "交易筆數", "來源"], excluded_branch_rows)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "institutional_metrics": str(args.institutional_metrics),
        "institutional_rows": len(institutional_rows),
        "institutional_participants": ["foreign", "trust", "dealer"],
        "broker_rows": len(broker_rows),
        "broker_branches": branches,
        "broker_top_branches": args.top_branches,
        "broker_top_n": args.broker_top_n,
        "chart_top_n": args.chart_top_n,
        "holding_days": list(HOLDING_DAYS),
        "round_trip_cost_rate": ROUND_TRIP_COST_RATE,
        **broker_meta,
        "outputs": [str(COMPARISON_CSV), str(BEST_CSV), str(EXCLUDED_BRANCHES_CSV), str(REPORT_HTML)],
    }
    METADATA_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_HTML.write_text(render_html(rows, best, metadata), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
