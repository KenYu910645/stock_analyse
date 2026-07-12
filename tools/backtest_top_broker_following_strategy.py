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
OUTPUT_DIR = PROJECT_ROOT / "output" / "broker"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "broker"
BRANCH_PERFORMANCE_CSV = OUTPUT_DIR / "all_broker_performance_by_branch.csv"

REPORT_PREFIX = "top10_broker_independent_following"
REPORT_HTML = VIZ_DIR / f"{REPORT_PREFIX}_strategy_report.html"
SELECTED_BRANCHES_CSV = OUTPUT_DIR / f"{REPORT_PREFIX}_selected_branches.csv"
IC_TIMESERIES_CSV = OUTPUT_DIR / f"{REPORT_PREFIX}_ic_timeseries.csv"
IC_SUMMARY_CSV = OUTPUT_DIR / f"{REPORT_PREFIX}_ic_summary.csv"
TRADE_SUMMARY_CSV = OUTPUT_DIR / f"{REPORT_PREFIX}_trade_summary.csv"
TRADES_CSV = OUTPUT_DIR / f"{REPORT_PREFIX}_trades.csv"
DAILY_RETURNS_CSV = OUTPUT_DIR / f"{REPORT_PREFIX}_daily_returns.csv"
METADATA_JSON = OUTPUT_DIR / f"{REPORT_PREFIX}_strategy_metadata.json"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HORIZONS = (1, 5, 10, 20)
ROUND_TRIP_COST_RATE = 0.001425 * 2 + 0.003

COL_GROUP = "訊號組"
COL_BRANCH = "分點名稱"
COL_HOLDING = "持有天數"
COL_DATE = "Date"
COL_SIGNAL_DATE = "訊號日"
COL_ENTRY_DATE = "進場日"
COL_EXIT_DATE = "出場日"
COL_CODE = "股票代號"
COL_NAME = "股票名稱"
COL_DIRECTION = "方向"
COL_NET = "淨買賣超張數"
COL_SIGNAL_NOTIONAL = "訊號金額"
COL_GROSS_RETURN = "毛報酬"
COL_NET_RETURN = "費稅後報酬"
COL_ACTIVE_POSITIONS = "活躍部位數"
COL_DAILY_GROSS_RETURN = "日毛報酬"
COL_DAILY_NET_RETURN = "日費稅後報酬"
COL_GROSS_EQUITY = "毛權益曲線"
COL_NET_EQUITY = "費稅後權益曲線"


@dataclass
class PriceSeries:
    code: str
    name: str
    dates: list[str]
    open_adj: list[float]
    close_adj: list[float]
    close: list[float]
    date_to_index: dict[str, int]


@dataclass
class RawSignal:
    branch: str
    date: str
    code: str
    name: str
    net: int
    buy: int
    sell: int


@dataclass
class EnrichedSignal:
    branch: str
    date: str
    code: str
    name: str
    net: int
    buy: int
    sell: int
    signal_notional: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and backtest independent next-open following strategies for top 20D Fubon broker branches."
    )
    parser.add_argument("--broker-dir", type=Path, default=BROKER_DIR)
    parser.add_argument("--price-dir", type=Path, default=PRICE_DIR)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--branch-performance", type=Path, default=BRANCH_PERFORMANCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    parser.add_argument("--top-branches", type=int, default=10)
    parser.add_argument("--selection-horizon", type=int, default=20)
    parser.add_argument("--max-daily-signals", type=int, default=50)
    parser.add_argument("--min-ic-sample", type=int, default=5)
    parser.add_argument("--benchmark-code", default="0050")
    return parser.parse_args()


def parse_int(value: object) -> int:
    if value is None:
        return 0
    text = str(value).replace(",", "").strip()
    if not text or text == "-":
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).replace(",", "").strip()
    if not text or text == "-":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def first_row_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def pct(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def fmt_pct(value: object, digits: int = 2) -> str:
    number = parse_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number * 100:.{digits}f}%"


def fmt_num(value: object, digits: int = 0) -> str:
    number = parse_float(value)
    if not math.isfinite(number):
        return ""
    return f"{number:,.{digits}f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_listed_common_codes(path: Path) -> tuple[set[str], dict[str, dict[str, str]]]:
    allowed: set[str] = set()
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            code = (row.get("Code") or "").strip()
            if not code:
                continue
            metadata[code] = row
            type_value = first_row_value(row, "Type", "類型").strip()
            market_value = first_row_value(row, "Market", "市場").strip()
            cfi = (row.get("CFI") or "").strip()
            if code.isdigit() and len(code) == 4 and market_value == "上市" and (type_value == "股票" or cfi == "ESVUFR"):
                allowed.add(code)
    return allowed, metadata


def price_paths_by_code(price_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in price_dir.glob("*.csv"):
        code = path.stem.split("_", 1)[0]
        if code and code not in paths:
            paths[code] = path
    return paths


def load_price_series(path: Path) -> PriceSeries:
    dates: list[str] = []
    open_adj: list[float] = []
    close_adj: list[float] = []
    close: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            date = (row.get("Date") or "").strip()
            if not DATE_RE.match(date):
                continue
            raw_open = parse_float(first_row_value(row, "open_adj", "前復權開盤價", "Open", "開盤價"))
            raw_close_adj = parse_float(first_row_value(row, "close_adj", "前復權收盤價", "Close", "收盤價"))
            raw_close = parse_float(first_row_value(row, "Close", "收盤價", "close_adj", "前復權收盤價"))
            if not (math.isfinite(raw_open) and raw_open > 0 and math.isfinite(raw_close_adj) and raw_close_adj > 0):
                continue
            dates.append(date)
            open_adj.append(raw_open)
            close_adj.append(raw_close_adj)
            close.append(raw_close if math.isfinite(raw_close) and raw_close > 0 else raw_close_adj)
    code = path.stem.split("_", 1)[0]
    name = path.stem.split("_", 1)[1] if "_" in path.stem else code
    return PriceSeries(code, name, dates, open_adj, close_adj, close, {date: idx for idx, date in enumerate(dates)})


def select_top_branches(path: Path, top_n: int, horizon: int) -> list[dict[str, object]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if parse_int(row.get("觀察期交易日")) == horizon:
                rows.append(row)
    rows.sort(key=lambda row: parse_float(row.get("淨買賣超金額權重報酬")), reverse=True)

    selected: list[dict[str, object]] = []
    for rank, row in enumerate(rows[:top_n], 1):
        selected.append(
            {
                "排名": rank,
                "分點名稱": row["分點名稱"],
                "分點類別": row["分點類別"],
                "選擇依據觀察期": horizon,
                "事件數": parse_int(row.get("事件數")),
                "命中率": pct(parse_float(row.get("命中率"))),
                "20日金額權重報酬": pct(parse_float(row.get("淨買賣超金額權重報酬"))),
            }
        )
    return selected


def load_branch_signals(broker_dir: Path, branch_name: str, allowed_codes: set[str]) -> list[RawSignal]:
    path = broker_dir / f"{branch_name}.csv"
    if not path.exists():
        return []
    signals: list[RawSignal] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Date", "Code", "Name", "買進", "賣出", "買賣超"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            return signals
        for row in reader:
            date = (row.get("Date") or "").strip()
            code = (row.get("Code") or "").strip()
            if not DATE_RE.match(date) or code not in allowed_codes:
                continue
            key = (date, code)
            if key in seen:
                continue
            seen.add(key)
            net = parse_int(row.get("買賣超"))
            if net == 0:
                continue
            signals.append(
                RawSignal(
                    branch=branch_name,
                    date=date,
                    code=code,
                    name=(row.get("Name") or "").strip(),
                    net=net,
                    buy=parse_int(row.get("買進")),
                    sell=parse_int(row.get("賣出")),
                )
            )
    return signals


def enrich_signals(raw_signals: list[RawSignal], prices: dict[str, PriceSeries]) -> list[EnrichedSignal]:
    enriched: list[EnrichedSignal] = []
    for signal in raw_signals:
        price = prices.get(signal.code)
        if not price:
            continue
        idx = price.date_to_index.get(signal.date)
        if idx is None:
            continue
        close_price = price.close[idx]
        if not math.isfinite(close_price) or close_price <= 0:
            continue
        enriched.append(
            EnrichedSignal(
                branch=signal.branch,
                date=signal.date,
                code=signal.code,
                name=signal.name or price.name,
                net=signal.net,
                buy=signal.buy,
                sell=signal.sell,
                signal_notional=signal.net * close_price,
            )
        )
    return enriched


def select_daily_signals(signals: list[EnrichedSignal], max_daily_signals: int) -> list[EnrichedSignal]:
    by_date: dict[str, list[EnrichedSignal]] = defaultdict(list)
    for signal in signals:
        by_date[signal.date].append(signal)
    selected: list[EnrichedSignal] = []
    for date in sorted(by_date):
        day_signals = sorted(by_date[date], key=lambda signal: abs(signal.signal_notional), reverse=True)
        selected.extend(day_signals[:max_daily_signals])
    return selected


def forward_open_return(price: PriceSeries, signal_date: str, holding_days: int) -> tuple[str, str, float] | None:
    signal_idx = price.date_to_index.get(signal_date)
    if signal_idx is None:
        return None
    entry_idx = signal_idx + 1
    exit_idx = entry_idx + holding_days
    if exit_idx >= len(price.dates):
        return None
    entry = price.open_adj[entry_idx]
    exit_price = price.open_adj[exit_idx]
    if not (math.isfinite(entry) and entry > 0 and math.isfinite(exit_price) and exit_price > 0):
        return None
    return price.dates[entry_idx], price.dates[exit_idx], exit_price / entry - 1.0


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for original_idx, _ in indexed[idx:end]:
            result[original_idx] = rank
        idx = end
    return result


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return math.nan
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return math.nan
    return num / den_x / den_y


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def summarize_values(values: list[float]) -> dict[str, float]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return {"mean": math.nan, "median": math.nan, "positive_ratio": math.nan}
    sorted_values = sorted(clean)
    mid = len(sorted_values) // 2
    median = sorted_values[mid] if len(sorted_values) % 2 else (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return {
        "mean": sum(clean) / len(clean),
        "median": median,
        "positive_ratio": sum(1 for value in clean if value > 0) / len(clean),
    }


def compute_ic_rows(
    branch_signals: dict[str, list[EnrichedSignal]],
    prices: dict[str, PriceSeries],
    min_sample: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    timeseries: list[dict[str, object]] = []
    for branch, signals in branch_signals.items():
        records = [(signal.date, signal.code, signal.signal_notional) for signal in signals]
        for horizon in HORIZONS:
            by_date: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for date, code, score in records:
                price = prices.get(code)
                if not price:
                    continue
                forward = forward_open_return(price, date, horizon)
                if not forward:
                    continue
                _, _, future_return = forward
                by_date[date].append((score, future_return))
            for date, pairs in sorted(by_date.items()):
                if len(pairs) < min_sample:
                    continue
                xs = [pair[0] for pair in pairs]
                ys = [pair[1] for pair in pairs]
                timeseries.append(
                    {
                        COL_GROUP: branch,
                        "觀察期交易日": horizon,
                        COL_DATE: date,
                        "樣本數": len(pairs),
                        "IC": pct(pearson(xs, ys)),
                        "RankIC": pct(spearman(xs, ys)),
                    }
                )

    summary: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in timeseries:
        grouped[(str(row[COL_GROUP]), int(row["觀察期交易日"]))].append(row)
    for (branch, horizon), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ic_values = [parse_float(row["IC"]) for row in rows]
        rank_values = [parse_float(row["RankIC"]) for row in rows]
        sample_counts = [parse_int(row["樣本數"]) for row in rows]
        ic_stats = summarize_values(ic_values)
        rank_stats = summarize_values(rank_values)
        summary.append(
            {
                COL_GROUP: branch,
                "觀察期交易日": horizon,
                "IC日期數": len(rows),
                "平均每日樣本數": pct(sum(sample_counts) / len(sample_counts) if sample_counts else math.nan),
                "IC平均": pct(ic_stats["mean"]),
                "IC中位數": pct(ic_stats["median"]),
                "IC為正比例": pct(ic_stats["positive_ratio"]),
                "RankIC平均": pct(rank_stats["mean"]),
                "RankIC中位數": pct(rank_stats["median"]),
                "RankIC為正比例": pct(rank_stats["positive_ratio"]),
            }
        )
    return timeseries, summary


def build_trades(signals: list[EnrichedSignal], prices: dict[str, PriceSeries], holding_days: int) -> list[dict[str, object]]:
    trades: list[dict[str, object]] = []
    for signal in signals:
        price = prices.get(signal.code)
        if not price:
            continue
        forward = forward_open_return(price, signal.date, holding_days)
        if not forward:
            continue
        entry_date, exit_date, raw_return = forward
        direction = 1 if signal.signal_notional > 0 else -1
        gross_return = raw_return * direction
        trades.append(
            {
                COL_GROUP: signal.branch,
                COL_HOLDING: holding_days,
                COL_SIGNAL_DATE: signal.date,
                COL_ENTRY_DATE: entry_date,
                COL_EXIT_DATE: exit_date,
                COL_CODE: signal.code,
                COL_NAME: signal.name,
                COL_DIRECTION: "做多" if direction > 0 else "做空",
                COL_NET: signal.net,
                COL_SIGNAL_NOTIONAL: pct(signal.signal_notional),
                COL_GROSS_RETURN: pct(gross_return),
                COL_NET_RETURN: pct(gross_return - ROUND_TRIP_COST_RATE),
            }
        )
    return trades


def max_drawdown(equity: list[float]) -> float:
    peak = -math.inf
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, value / peak - 1.0)
    return max_dd


def portfolio_daily_returns(trades: list[dict[str, object]], prices: dict[str, PriceSeries], holding_days: int) -> list[dict[str, object]]:
    returns_by_date: dict[str, list[float]] = defaultdict(list)
    gross_by_date: dict[str, list[float]] = defaultdict(list)
    active_by_date: dict[str, int] = defaultdict(int)
    daily_cost = ROUND_TRIP_COST_RATE / holding_days

    for trade in trades:
        code = str(trade[COL_CODE])
        price = prices.get(code)
        if not price:
            continue
        entry_idx = price.date_to_index.get(str(trade[COL_ENTRY_DATE]))
        exit_idx = price.date_to_index.get(str(trade[COL_EXIT_DATE]))
        if entry_idx is None or exit_idx is None or exit_idx <= entry_idx:
            continue
        direction = 1 if trade[COL_DIRECTION] == "做多" else -1
        for idx in range(entry_idx, exit_idx):
            start = price.open_adj[idx]
            end = price.open_adj[idx + 1]
            if not (math.isfinite(start) and start > 0 and math.isfinite(end) and end > 0):
                continue
            date = price.dates[idx + 1]
            gross = direction * (end / start - 1.0)
            gross_by_date[date].append(gross)
            returns_by_date[date].append(gross - daily_cost)
            active_by_date[date] += 1

    equity = 1.0
    gross_equity = 1.0
    rows: list[dict[str, object]] = []
    for date in sorted(returns_by_date):
        daily_returns = returns_by_date[date]
        gross_returns = gross_by_date[date]
        net_ret = sum(daily_returns) / len(daily_returns)
        gross_ret = sum(gross_returns) / len(gross_returns)
        equity *= 1.0 + net_ret
        gross_equity *= 1.0 + gross_ret
        rows.append(
            {
                COL_HOLDING: holding_days,
                COL_DATE: date,
                COL_ACTIVE_POSITIONS: active_by_date[date],
                COL_DAILY_GROSS_RETURN: pct(gross_ret),
                COL_DAILY_NET_RETURN: pct(net_ret),
                COL_GROSS_EQUITY: pct(gross_equity),
                COL_NET_EQUITY: pct(equity),
            }
        )
    return rows


def benchmark_daily_returns(price: PriceSeries, dates: list[str]) -> tuple[float, list[float]]:
    if not dates:
        return math.nan, []
    returns: list[float] = []
    equity = 1.0
    for date in dates:
        idx = price.date_to_index.get(date)
        if idx is None or idx <= 0:
            continue
        previous = price.open_adj[idx - 1]
        current = price.open_adj[idx]
        if previous > 0 and current > 0:
            daily = current / previous - 1.0
            returns.append(daily)
            equity *= 1.0 + daily
    return equity - 1.0, returns


def summarize_strategy(
    branch_names: list[str],
    trades_by_branch_horizon: dict[tuple[str, int], list[dict[str, object]]],
    daily_by_branch_horizon: dict[tuple[str, int], list[dict[str, object]]],
    benchmark: PriceSeries | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for branch in branch_names:
        for horizon in HORIZONS:
            trades = trades_by_branch_horizon.get((branch, horizon), [])
            daily = daily_by_branch_horizon.get((branch, horizon), [])
            gross_returns = [parse_float(trade[COL_GROSS_RETURN]) for trade in trades]
            net_returns = [parse_float(trade[COL_NET_RETURN]) for trade in trades]
            gross_stats = summarize_values(gross_returns)
            net_stats = summarize_values(net_returns)
            daily_returns = [parse_float(row[COL_DAILY_NET_RETURN]) for row in daily]
            equity = [parse_float(row[COL_NET_EQUITY]) for row in daily]
            total_return = equity[-1] - 1.0 if equity else math.nan
            day_count = len(daily_returns)
            avg_daily = sum(daily_returns) / day_count if day_count else math.nan
            if day_count > 1:
                variance = sum((value - avg_daily) ** 2 for value in daily_returns) / (day_count - 1)
                daily_std = math.sqrt(variance)
            else:
                daily_std = math.nan
            annual_return = (1.0 + total_return) ** (252 / day_count) - 1.0 if day_count and total_return > -1 else math.nan
            annual_vol = daily_std * math.sqrt(252) if math.isfinite(daily_std) else math.nan
            sharpe = annual_return / annual_vol if math.isfinite(annual_return) and math.isfinite(annual_vol) and annual_vol else math.nan
            benchmark_return, benchmark_returns = (math.nan, [])
            if benchmark:
                benchmark_return, benchmark_returns = benchmark_daily_returns(benchmark, [str(row[COL_DATE]) for row in daily])
            benchmark_sharpe = math.nan
            if benchmark_returns:
                bench_avg = sum(benchmark_returns) / len(benchmark_returns)
                if len(benchmark_returns) > 1:
                    bench_std = math.sqrt(sum((value - bench_avg) ** 2 for value in benchmark_returns) / (len(benchmark_returns) - 1))
                    bench_annual_return = (1.0 + benchmark_return) ** (252 / len(benchmark_returns)) - 1.0 if benchmark_return > -1 else math.nan
                    benchmark_sharpe = bench_annual_return / (bench_std * math.sqrt(252)) if bench_std else math.nan
            rows.append(
                {
                    COL_GROUP: branch,
                    COL_HOLDING: horizon,
                    "交易筆數": len(trades),
                    "做多筆數": sum(1 for trade in trades if trade[COL_DIRECTION] == "做多"),
                    "做空筆數": sum(1 for trade in trades if trade[COL_DIRECTION] == "做空"),
                    "平均毛報酬": pct(gross_stats["mean"]),
                    "毛勝率": pct(gross_stats["positive_ratio"]),
                    "平均費稅後報酬": pct(net_stats["mean"]),
                    "費稅後勝率": pct(net_stats["positive_ratio"]),
                    "日數": day_count,
                    "策略費稅後總報酬": pct(total_return),
                    "策略年化報酬": pct(annual_return),
                    "策略年化波動": pct(annual_vol),
                    "策略Sharpe": pct(sharpe),
                    "最大回撤": pct(max_drawdown(equity)),
                    "0050同期報酬": pct(benchmark_return),
                    "0050同期Sharpe": pct(benchmark_sharpe),
                    "平均活躍部位數": pct(
                        sum(parse_int(row[COL_ACTIVE_POSITIONS]) for row in daily) / len(daily) if daily else math.nan
                    ),
                }
            )
    return rows


def h(text: object) -> str:
    return html.escape(str(text), quote=True)


def display_value(key: str, value: object) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    if key in {"排名", "事件數", "IC日期數", "樣本數", "交易筆數", "做多筆數", "做空筆數", "日數", COL_ACTIVE_POSITIONS}:
        return fmt_num(parse_int(text), 0)
    if key in {
        "命中率",
        "20日金額權重報酬",
        "IC平均",
        "IC中位數",
        "IC為正比例",
        "RankIC平均",
        "RankIC中位數",
        "RankIC為正比例",
        "平均毛報酬",
        "毛勝率",
        "平均費稅後報酬",
        "費稅後勝率",
        "策略費稅後總報酬",
        "策略年化報酬",
        "策略年化波動",
        "最大回撤",
        "0050同期報酬",
    }:
        return fmt_pct(text)
    if key in {"策略Sharpe", "0050同期Sharpe"}:
        return fmt_num(parse_float(text), 2)
    if key in {COL_SIGNAL_NOTIONAL}:
        return fmt_num(parse_float(text), 0)
    if key == "平均活躍部位數":
        return fmt_num(parse_float(text), 1)
    return text


def render_table(headers: list[str], rows: list[dict[str, object]], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    if not selected:
        return "<div class='empty'>沒有可顯示資料</div>"
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


def bar_svg(rows: list[dict[str, object]], value_key: str, title: str) -> str:
    if not rows:
        return "<div class='empty'>沒有圖表資料</div>"
    values = [parse_float(row[value_key]) for row in rows]
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return "<div class='empty'>沒有圖表資料</div>"
    width = 980
    row_h = 28
    left = 132
    right = 80
    top = 32
    height = top + row_h * len(rows) + 28
    min_v = min(min(clean), 0.0)
    max_v = max(max(clean), 0.0)
    if max_v == min_v:
        max_v += 0.01
        min_v -= 0.01
    plot_w = width - left - right
    zero_x = left + (0.0 - min_v) / (max_v - min_v) * plot_w
    lines = [f'<text x="{left}" y="18">{h(title)}</text>', f'<line x1="{zero_x:.1f}" y1="{top-8}" x2="{zero_x:.1f}" y2="{height-20}" class="base"/>']
    for idx, row in enumerate(rows):
        value = parse_float(row[value_key])
        y = top + idx * row_h
        x = left + (min(0.0, value) - min_v) / (max_v - min_v) * plot_w
        bar_w = abs(value) / (max_v - min_v) * plot_w
        klass = "pos" if value >= 0 else "neg"
        lines.append(f'<text x="8" y="{y+16}">{h(row[COL_GROUP])}</text>')
        lines.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="18" class="{klass}"/>')
        label_x = x + bar_w + 5 if value >= 0 else max(4, x - 58)
        lines.append(f'<text x="{label_x:.1f}" y="{y+14}">{h(fmt_pct(value))}</text>')
    return f'<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{h(title)}">' + "".join(lines) + "</svg>"


def render_html(
    selected_branches: list[dict[str, object]],
    ic_summary: list[dict[str, object]],
    strategy_summary: list[dict[str, object]],
    all_trades: list[dict[str, object]],
    metadata: dict[str, object],
) -> str:
    branch_names = [str(row[COL_BRANCH]) for row in selected_branches]
    selected_names = "、".join(branch_names)
    summary_20 = [row for row in strategy_summary if int(row[COL_HOLDING]) == 20]
    summary_20_sorted = sorted(summary_20, key=lambda row: parse_float(row["策略費稅後總報酬"]), reverse=True)
    best_20 = summary_20_sorted[0] if summary_20_sorted else {}
    median_20 = math.nan
    total_returns = sorted(parse_float(row["策略費稅後總報酬"]) for row in summary_20 if math.isfinite(parse_float(row["策略費稅後總報酬"])))
    if total_returns:
        mid = len(total_returns) // 2
        median_20 = total_returns[mid] if len(total_returns) % 2 else (total_returns[mid - 1] + total_returns[mid]) / 2

    ic_20 = [row for row in ic_summary if int(row["觀察期交易日"]) == 20]
    ic_20_sorted = sorted(ic_20, key=lambda row: parse_float(row["RankIC平均"]), reverse=True)
    trade_20_sample = [trade for trade in all_trades if int(trade[COL_HOLDING]) == 20][:120]

    branch_headers = ["排名", COL_BRANCH, "分點類別", "事件數", "命中率", "20日金額權重報酬"]
    ic_headers = [COL_GROUP, "觀察期交易日", "IC日期數", "平均每日樣本數", "IC平均", "IC為正比例", "RankIC平均", "RankIC為正比例"]
    summary_headers = [
        COL_GROUP,
        COL_HOLDING,
        "交易筆數",
        "做多筆數",
        "做空筆數",
        "平均毛報酬",
        "毛勝率",
        "平均費稅後報酬",
        "費稅後勝率",
        "策略費稅後總報酬",
        "策略Sharpe",
        "最大回撤",
        "0050同期報酬",
        "平均活躍部位數",
    ]
    trade_headers = [
        COL_GROUP,
        COL_HOLDING,
        COL_SIGNAL_DATE,
        COL_ENTRY_DATE,
        COL_EXIT_DATE,
        COL_CODE,
        COL_NAME,
        COL_DIRECTION,
        COL_NET,
        COL_SIGNAL_NOTIONAL,
        COL_GROSS_RETURN,
        COL_NET_RETURN,
    ]

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>20D 前十分點獨立跟單策略 IC 與回測報告</title>
  <style>
    :root {{
      --ink:#17212b; --muted:#5f6b76; --line:#d8dee6; --panel:#fff; --bg:#f6f7f9;
      --accent:#0f766e; --neg:#b42318; --warn:#b45309;
    }}
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
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }}
    .card {{ padding:14px; border:1px solid var(--line); border-radius:8px; background:#fff; }}
    .card strong {{ display:block; font-size:22px; }}
    .card span {{ color:var(--muted); font-size:13px; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
    table {{ border-collapse:collapse; width:100%; min-width:980px; font-size:13px; }}
    th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ background:#eef2f7; color:#334155; position:sticky; top:0; }}
    tbody tr:hover {{ background:#f8fafc; }}
    .bar-chart {{ width:100%; height:auto; background:#fff; border:1px solid var(--line); border-radius:8px; }}
    .bar-chart .pos {{ fill:var(--accent); }}
    .bar-chart .neg {{ fill:var(--neg); }}
    .bar-chart .base {{ stroke:#94a3b8; stroke-dasharray:4 4; }}
    .bar-chart text {{ font-size:12px; fill:var(--muted); }}
    .empty {{ color:var(--muted); padding:16px; }}
    @media(max-width:900px) {{ header,main {{ padding-left:16px; padding-right:16px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>20D 前十分點獨立跟單策略 IC 與回測報告</h1>
    <p>分點：{h(selected_names)}。每個分點各自產生訊號、各自回測，不做跨分點合併。買超隔日開盤做多，賣超隔日開盤做空。</p>
    <div class="meta">
      <span class="pill">產生時間：{h(metadata["generated_at"])}</span>
      <span class="pill">選擇方式：20D 績效前 {metadata["top_branches"]} 名</span>
      <span class="pill">每分點每日最多 {metadata["max_daily_signals"]} 檔</span>
      <span class="pill">進出場：隔日開盤到固定天數後開盤</span>
      <span class="pill">Round-trip 費稅：{fmt_pct(ROUND_TRIP_COST_RATE)}</span>
    </div>
  </header>
  <main>
    <div class="note">這是 in-sample 初版驗證：分點是用全歷史 20D 績效挑出，再回測同一段歷史，會有選擇偏誤。策略也假設可做空、可在隔日開盤成交，未檢查融券/借券、漲跌停與實際滑價。</div>

    <section class="section">
      <h2>20 日獨立跟單摘要</h2>
      <div class="cards">
        <div class="card"><strong>{h(best_20.get(COL_GROUP, ""))}</strong><span>20D 費稅後最佳分點</span></div>
        <div class="card"><strong>{display_value("策略費稅後總報酬", best_20.get("策略費稅後總報酬", ""))}</strong><span>最佳分點總報酬</span></div>
        <div class="card"><strong>{fmt_pct(median_20)}</strong><span>前十分點 20D 總報酬中位數</span></div>
        <div class="card"><strong>{fmt_num(metadata["enriched_signals"], 0)}</strong><span>成功對齊價格訊號</span></div>
      </div>
    </section>

    <section class="section">
      <h2>20 日費稅後總報酬排名</h2>
      {bar_svg(summary_20_sorted, "策略費稅後總報酬", "各分點獨立跟單 20D 費稅後總報酬")}
      {render_table(summary_headers, summary_20_sorted)}
    </section>

    <section class="section">
      <h2>選出分點</h2>
      {render_table(branch_headers, selected_branches)}
    </section>

    <section class="section">
      <h2>20 日 IC 摘要</h2>
      <p>IC 是訊號金額與未來報酬的每日相關；RankIC 是排名相關。這裡每個分點獨立計算，沒有把不同分點的訊號合併。</p>
      {render_table(ic_headers, ic_20_sorted)}
    </section>

    <section class="section">
      <h2>全部週期策略摘要</h2>
      {render_table(summary_headers, strategy_summary)}
    </section>

    <section class="section">
      <h2>全部週期 IC 摘要</h2>
      {render_table(ic_headers, ic_summary)}
    </section>

    <section class="section">
      <h2>20 日交易明細樣本</h2>
      {render_table(trade_headers, trade_20_sample)}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.viz_dir.mkdir(parents=True, exist_ok=True)

    allowed_codes, _metadata_by_code = load_listed_common_codes(args.metadata)
    selected_branches = select_top_branches(args.branch_performance, args.top_branches, args.selection_horizon)
    branch_names = [str(row[COL_BRANCH]) for row in selected_branches]

    raw_by_branch: dict[str, list[RawSignal]] = {}
    all_raw: list[RawSignal] = []
    for branch in branch_names:
        signals = load_branch_signals(args.broker_dir, branch, allowed_codes)
        raw_by_branch[branch] = signals
        all_raw.extend(signals)

    codes = sorted({signal.code for signal in all_raw})
    price_paths = price_paths_by_code(args.price_dir)
    prices: dict[str, PriceSeries] = {}
    for code in codes:
        path = price_paths.get(code)
        if path:
            prices[code] = load_price_series(path)
    benchmark = load_price_series(price_paths[args.benchmark_code]) if args.benchmark_code in price_paths else None

    branch_signals: dict[str, list[EnrichedSignal]] = {}
    branch_strategy_signals: dict[str, list[EnrichedSignal]] = {}
    all_enriched: list[EnrichedSignal] = []
    all_strategy_signals: list[EnrichedSignal] = []
    for branch, signals in raw_by_branch.items():
        enriched = enrich_signals(signals, prices)
        strategy_signals = select_daily_signals(enriched, args.max_daily_signals)
        branch_signals[branch] = enriched
        branch_strategy_signals[branch] = strategy_signals
        all_enriched.extend(enriched)
        all_strategy_signals.extend(strategy_signals)

    ic_timeseries, ic_summary = compute_ic_rows(branch_signals, prices, args.min_ic_sample)

    trades_by_branch_horizon: dict[tuple[str, int], list[dict[str, object]]] = {}
    daily_by_branch_horizon: dict[tuple[str, int], list[dict[str, object]]] = {}
    all_trades: list[dict[str, object]] = []
    all_daily: list[dict[str, object]] = []
    for branch in branch_names:
        for horizon in HORIZONS:
            trades = build_trades(branch_strategy_signals[branch], prices, horizon)
            daily = portfolio_daily_returns(trades, prices, horizon)
            for row in daily:
                row[COL_GROUP] = branch
            trades_by_branch_horizon[(branch, horizon)] = trades
            daily_by_branch_horizon[(branch, horizon)] = daily
            all_trades.extend(trades)
            all_daily.extend(daily)

    strategy_summary = summarize_strategy(branch_names, trades_by_branch_horizon, daily_by_branch_horizon, benchmark)

    write_csv(SELECTED_BRANCHES_CSV, ["排名", COL_BRANCH, "分點類別", "選擇依據觀察期", "事件數", "命中率", "20日金額權重報酬"], selected_branches)
    write_csv(IC_TIMESERIES_CSV, [COL_GROUP, "觀察期交易日", COL_DATE, "樣本數", "IC", "RankIC"], ic_timeseries)
    write_csv(
        IC_SUMMARY_CSV,
        [COL_GROUP, "觀察期交易日", "IC日期數", "平均每日樣本數", "IC平均", "IC中位數", "IC為正比例", "RankIC平均", "RankIC中位數", "RankIC為正比例"],
        ic_summary,
    )
    write_csv(
        TRADE_SUMMARY_CSV,
        [
            COL_GROUP,
            COL_HOLDING,
            "交易筆數",
            "做多筆數",
            "做空筆數",
            "平均毛報酬",
            "毛勝率",
            "平均費稅後報酬",
            "費稅後勝率",
            "日數",
            "策略費稅後總報酬",
            "策略年化報酬",
            "策略年化波動",
            "策略Sharpe",
            "最大回撤",
            "0050同期報酬",
            "0050同期Sharpe",
            "平均活躍部位數",
        ],
        strategy_summary,
    )
    write_csv(
        TRADES_CSV,
        [COL_GROUP, COL_HOLDING, COL_SIGNAL_DATE, COL_ENTRY_DATE, COL_EXIT_DATE, COL_CODE, COL_NAME, COL_DIRECTION, COL_NET, COL_SIGNAL_NOTIONAL, COL_GROSS_RETURN, COL_NET_RETURN],
        all_trades,
    )
    write_csv(
        DAILY_RETURNS_CSV,
        [COL_GROUP, COL_HOLDING, COL_DATE, COL_ACTIVE_POSITIONS, COL_DAILY_GROSS_RETURN, COL_DAILY_NET_RETURN, COL_GROSS_EQUITY, COL_NET_EQUITY],
        all_daily,
    )

    report_metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_branches": args.top_branches,
        "selection_horizon": args.selection_horizon,
        "branch_names": branch_names,
        "allowed_codes": len(allowed_codes),
        "raw_signals": len(all_raw),
        "enriched_signals": len(all_enriched),
        "strategy_signals": len(all_strategy_signals),
        "price_series": len(prices),
        "max_daily_signals": args.max_daily_signals,
        "round_trip_cost_rate": ROUND_TRIP_COST_RATE,
        "benchmark_code": args.benchmark_code,
        "outputs": [
            str(SELECTED_BRANCHES_CSV),
            str(IC_TIMESERIES_CSV),
            str(IC_SUMMARY_CSV),
            str(TRADE_SUMMARY_CSV),
            str(TRADES_CSV),
            str(DAILY_RETURNS_CSV),
            str(REPORT_HTML),
        ],
    }
    METADATA_JSON.write_text(json.dumps(report_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_HTML.write_text(render_html(selected_branches, ic_summary, strategy_summary, all_trades, report_metadata), encoding="utf-8")
    print(json.dumps(report_metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
