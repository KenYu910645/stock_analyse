"""Backtest a margin-financing contrarian strategy across listed common stocks."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import csv_columns_canonical, read_csv_canonical

SIGNAL_COLUMN = "MarginBalance20DayChangeRate"
OUTPUT_DIR = PROJECT_ROOT / "output" / "margin_patterns" / "contrarian_backtest"
VIZ_PATH = PROJECT_ROOT / "data_viz" / "margin_patterns" / "margin_contrarian_backtest.html"

MARGIN_COLUMNS = [
    "Date",
    "Code",
    "Name",
    "MarginCurrentBalance",
    "MarginBalance20DayChangeRate",
]


@dataclass
class Config:
    window: int
    change_top_quantile: float
    change_bottom_quantile: float
    min_rows: int
    round_trip_cost_bps: float
    max_holding_days: int | None
    side_mode: str
    lookback_years: int | None
    start_date: pd.Timestamp | None
    benchmark_code: str | None
    benchmark_name: str | None
    output_dir: Path
    viz_path: Path
    codes: set[str] | None
    max_stocks: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the margin contrarian strategy.")
    parser.add_argument("--codes", nargs="*", help="Optional stock-code subset.")
    parser.add_argument("--max-stocks", type=int)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--change-top-quantile", type=float, default=0.90)
    parser.add_argument("--change-bottom-quantile", type=float, default=0.10)
    parser.add_argument("--min-rows", type=int, default=180)
    parser.add_argument("--round-trip-cost-bps", type=float, default=60.0)
    parser.add_argument("--max-holding-days", type=int, default=0, help="0 means no holding cap.")
    parser.add_argument("--side-mode", choices=["both", "long-only", "short-only"], default="both")
    parser.add_argument("--lookback-years", type=int, default=0, help="0 means use the full available period.")
    parser.add_argument("--start-date", help="Optional ISO date. Overrides --lookback-years.")
    parser.add_argument("--benchmark-code", default="0050", help="Benchmark price code. Empty string disables it.")
    parser.add_argument("--benchmark-name", default="0050 買進持有")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-path", type=Path, default=VIZ_PATH)
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def json_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if pd.isna(value):
        return None
    return value


def fmt_pct(value: Any, digits: int = 2) -> str:
    number = json_float(value)
    if number is None:
        return ""
    return f"{number * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    number = json_float(value)
    if number is None:
        return ""
    if abs(number) >= 100_000_000:
        return f"{number / 100_000_000:.{digits}f}億"
    if abs(number) >= 10_000:
        return f"{number / 10_000:.{digits}f}萬"
    if digits == 0:
        return f"{number:.0f}"
    return f"{number:.{digits}f}"


def find_stock_file(folder: Path, code: str) -> Path | None:
    matches = sorted(folder.glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def stock_universe(config: Config) -> pd.DataFrame:
    metadata = read_csv_canonical(PROJECT_ROOT / "data" / "metadata.csv", dtype={"Code": str}).fillna("")
    required = {"Code", "Name", "Type", "Market"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")
    universe = metadata[metadata["Type"].eq("股票") & metadata["Market"].eq("上市")].copy()
    universe["Code"] = universe["Code"].astype(str)
    if config.codes:
        universe = universe[universe["Code"].isin(config.codes)].copy()
    universe = universe.sort_values("Code").reset_index(drop=True)
    if config.max_stocks is not None:
        universe = universe.head(config.max_stocks)
    return universe


def latest_price_date(universe: pd.DataFrame) -> pd.Timestamp:
    latest: pd.Timestamp | None = None
    for code in universe["Code"].astype(str):
        price_path = find_stock_file(PROJECT_ROOT / "data" / "price", code)
        if price_path is None:
            continue
        try:
            price = read_csv_canonical(price_path, usecols=["Date"])
        except Exception:
            continue
        dates = pd.to_datetime(price["Date"], errors="coerce").dropna()
        if dates.empty:
            continue
        current = dates.max()
        latest = current if latest is None else max(latest, current)
    if latest is None:
        raise ValueError("no_price_dates_available")
    return pd.Timestamp(latest)


def load_panel(code: str, metadata_name: str, config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    margin_path = find_stock_file(PROJECT_ROOT / "data" / "margin", code)
    price_path = find_stock_file(PROJECT_ROOT / "data" / "price", code)
    if margin_path is None:
        raise FileNotFoundError("missing_margin_csv")
    if price_path is None:
        raise FileNotFoundError("missing_price_csv")

    margin_columns = csv_columns_canonical(margin_path)
    margin_usecols = [column for column in MARGIN_COLUMNS if column in margin_columns]
    if not {"Date", "MarginCurrentBalance"}.issubset(margin_usecols):
        raise ValueError("margin_csv_missing_required_columns")
    margin = read_csv_canonical(margin_path, usecols=margin_usecols, dtype={"Code": str})
    if "Name" not in margin.columns:
        margin["Name"] = metadata_name
    margin["Date"] = pd.to_datetime(margin["Date"], errors="coerce")
    margin = margin.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    for column in margin.columns:
        if column not in {"Date", "Code", "Name"}:
            margin[column] = pd.to_numeric(margin[column], errors="coerce")
    if SIGNAL_COLUMN not in margin.columns:
        margin[SIGNAL_COLUMN] = np.nan

    price_columns = csv_columns_canonical(price_path)
    open_column = "open_adj" if "open_adj" in price_columns else ("Open" if "Open" in price_columns else None)
    close_column = "close_adj" if "close_adj" in price_columns else ("Close" if "Close" in price_columns else None)
    if "Date" not in price_columns or open_column is None or close_column is None:
        raise ValueError("price_csv_missing_required_columns")
    price = read_csv_canonical(price_path, usecols=["Date", open_column, close_column])
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price["OpenExec"] = pd.to_numeric(price[open_column], errors="coerce")
    price["CloseExec"] = pd.to_numeric(price[close_column], errors="coerce")
    price = price.dropna(subset=["Date", "OpenExec", "CloseExec"])
    price = price[price["OpenExec"].gt(0) & price["CloseExec"].gt(0)]
    price = price[["Date", "OpenExec", "CloseExec"]].sort_values("Date").drop_duplicates("Date", keep="last")

    panel = margin.merge(price, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    panel = panel.dropna(subset=["MarginCurrentBalance", "OpenExec", "CloseExec"])
    panel = panel[panel["MarginCurrentBalance"].gt(0)].reset_index(drop=True)
    if len(panel) < config.min_rows:
        raise ValueError("insufficient_joined_rows")

    previous_balance = panel["MarginCurrentBalance"].shift(config.window)
    computed_signal = panel["MarginCurrentBalance"] / previous_balance - 1
    computed_signal = computed_signal.where(previous_balance.gt(0), np.nan)
    panel[SIGNAL_COLUMN] = panel[SIGNAL_COLUMN].where(panel[SIGNAL_COLUMN].notna(), computed_signal)
    panel[SIGNAL_COLUMN] = pd.to_numeric(panel[SIGNAL_COLUMN], errors="coerce").replace([np.inf, -np.inf], np.nan)

    valid_signal = panel[SIGNAL_COLUMN].dropna()
    if len(valid_signal) < config.min_rows // 2:
        raise ValueError("insufficient_valid_signal_rows")
    thresholds = {
        "change_top": float(valid_signal.quantile(config.change_top_quantile)),
        "change_bottom": float(valid_signal.quantile(config.change_bottom_quantile)),
    }
    meta = {
        "name": str(panel["Name"].dropna().iloc[-1]) if panel["Name"].notna().any() else metadata_name,
        "margin_path": str(margin_path.relative_to(PROJECT_ROOT)),
        "price_path": str(price_path.relative_to(PROJECT_ROOT)),
        **thresholds,
    }
    return panel, meta


def build_trade(
    *,
    code: str,
    name: str,
    panel: pd.DataFrame,
    direction: int,
    entry_signal_index: int,
    entry_exec_index: int,
    exit_signal_index: int,
    exit_exec_index: int,
    exit_price: float,
    exit_reason: str,
    cost_rate: float,
) -> dict[str, Any]:
    entry_price = float(panel.at[entry_exec_index, "OpenExec"])
    gross_return = direction * (float(exit_price) / entry_price - 1)
    net_return = gross_return - cost_rate
    side = "Long" if direction > 0 else "Short"
    return {
        "Code": code,
        "Name": name,
        "Side": side,
        "Direction": direction,
        "EntrySignalDate": panel.at[entry_signal_index, "Date"].date().isoformat(),
        "EntryExecDate": panel.at[entry_exec_index, "Date"].date().isoformat(),
        "ExitSignalDate": panel.at[exit_signal_index, "Date"].date().isoformat(),
        "ExitExecDate": panel.at[exit_exec_index, "Date"].date().isoformat(),
        "EntrySignalIndex": entry_signal_index,
        "EntryExecIndex": entry_exec_index,
        "ExitSignalIndex": exit_signal_index,
        "ExitExecIndex": exit_exec_index,
        "EntryPrice": entry_price,
        "ExitPrice": float(exit_price),
        "GrossReturn": gross_return,
        "NetReturn": net_return,
        "HoldingDays": int(exit_exec_index - entry_exec_index),
        "ExitReason": exit_reason,
        "EntryMarginChange20D": json_float(panel.at[entry_signal_index, SIGNAL_COLUMN]),
        "ExitMarginChange20D": json_float(panel.at[exit_signal_index, SIGNAL_COLUMN]),
        "EntryMarginBalance": json_float(panel.at[entry_signal_index, "MarginCurrentBalance"]),
        "ExitMarginBalance": json_float(panel.at[exit_signal_index, "MarginCurrentBalance"]),
    }


def backtest_stock(code: str, metadata_name: str, config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel, meta = load_panel(code, metadata_name, config)
    top = meta["change_top"]
    bottom = meta["change_bottom"]
    cost_rate = config.round_trip_cost_bps / 10_000

    trades: list[dict[str, Any]] = []
    position = 0
    entry_signal_index: int | None = None
    entry_exec_index: int | None = None

    def open_position(signal_index: int, direction: int) -> None:
        nonlocal position, entry_signal_index, entry_exec_index
        position = direction
        entry_signal_index = signal_index
        entry_exec_index = signal_index + 1

    def close_position(signal_index: int, reason: str) -> None:
        nonlocal position, entry_signal_index, entry_exec_index
        if entry_signal_index is None or entry_exec_index is None:
            raise RuntimeError("close_position_without_entry")
        trades.append(
            build_trade(
                code=code,
                name=meta["name"],
                panel=panel,
                direction=position,
                entry_signal_index=entry_signal_index,
                entry_exec_index=entry_exec_index,
                exit_signal_index=signal_index,
                exit_exec_index=signal_index + 1,
                exit_price=float(panel.at[signal_index + 1, "OpenExec"]),
                exit_reason=reason,
                cost_rate=cost_rate,
            )
        )
        position = 0
        entry_signal_index = None
        entry_exec_index = None

    for index in range(0, len(panel) - 1):
        signal = panel.at[index, SIGNAL_COLUMN]
        if pd.isna(signal):
            continue
        signal_date = pd.Timestamp(panel.at[index, "Date"])

        if position != 0 and entry_signal_index is not None and index > entry_signal_index:
            exit_reason: str | None = None
            if position < 0:
                if signal <= bottom:
                    exit_reason = "opposite_margin_drop_signal"
                elif signal <= 0:
                    exit_reason = "margin_change_back_to_nonpositive"
            else:
                if signal >= top:
                    exit_reason = "opposite_margin_surge_signal"
                elif signal >= 0:
                    exit_reason = "margin_change_back_to_nonnegative"
            if (
                exit_reason is None
                and config.max_holding_days is not None
                and entry_exec_index is not None
                and index + 1 - entry_exec_index >= config.max_holding_days
            ):
                exit_reason = "max_holding_days"
            if exit_reason is not None:
                close_position(index, exit_reason)

        if position == 0:
            if config.start_date is not None and signal_date < config.start_date:
                continue
            if config.side_mode in {"both", "short-only"} and signal >= top:
                open_position(index, -1)
            elif config.side_mode in {"both", "long-only"} and signal <= bottom:
                open_position(index, 1)

    if position != 0 and entry_signal_index is not None and entry_exec_index is not None:
        last_index = len(panel) - 1
        trades.append(
            build_trade(
                code=code,
                name=meta["name"],
                panel=panel,
                direction=position,
                entry_signal_index=entry_signal_index,
                entry_exec_index=entry_exec_index,
                exit_signal_index=last_index,
                exit_exec_index=last_index,
                exit_price=float(panel.at[last_index, "CloseExec"]),
                exit_reason="forced_last_close",
                cost_rate=cost_rate,
            )
        )

    trade_df = pd.DataFrame(trades)
    if not trade_df.empty:
        trade_df["MarginSurgeThreshold"] = top
        trade_df["MarginDropThreshold"] = bottom
        trade_df["MarginPath"] = meta["margin_path"]
        trade_df["PricePath"] = meta["price_path"]
    return trade_df, panel, meta


def portfolio_segments(trades: pd.DataFrame, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades.empty:
        return pd.DataFrame()
    for _, trade in trades.iterrows():
        code = str(trade["Code"])
        panel = panels[code]
        direction = int(trade["Direction"])
        side = str(trade["Side"])
        start = int(trade["EntryExecIndex"])
        end = int(trade["ExitExecIndex"])
        for index in range(start, min(end, len(panel) - 1)):
            start_price = float(panel.at[index, "OpenExec"])
            end_price = float(panel.at[index + 1, "OpenExec"])
            rows.append(
                {
                    "Date": panel.at[index + 1, "Date"].date().isoformat(),
                    "Code": code,
                    "Name": trade["Name"],
                    "Side": side,
                    "GrossReturn": direction * (end_price / start_price - 1),
                }
            )
        if str(trade["ExitReason"]) == "forced_last_close" and end >= start and end < len(panel):
            start_price = float(panel.at[end, "OpenExec"])
            end_price = float(panel.at[end, "CloseExec"])
            rows.append(
                {
                    "Date": panel.at[end, "Date"].date().isoformat(),
                    "Code": code,
                    "Name": trade["Name"],
                    "Side": side,
                    "GrossReturn": direction * (end_price / start_price - 1),
                }
            )
    return pd.DataFrame(rows)


def equity_metrics(returns: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {
            "TradingDays": 0,
            "TotalReturn": None,
            "AnnualizedReturn": None,
            "AnnualizedVolatility": None,
            "Sharpe": None,
            "MaxDrawdown": None,
        }
    equity = (1 + clean).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    annualized = float((1 + total_return) ** (252 / len(clean)) - 1) if total_return > -1 else -1.0
    volatility = float(clean.std(ddof=0) * math.sqrt(252))
    sharpe = float(clean.mean() / clean.std(ddof=0) * math.sqrt(252)) if clean.std(ddof=0) > 0 else None
    drawdown = equity / equity.cummax() - 1
    return {
        "TradingDays": int(len(clean)),
        "TotalReturn": total_return,
        "AnnualizedReturn": annualized,
        "AnnualizedVolatility": volatility,
        "Sharpe": sharpe,
        "MaxDrawdown": float(drawdown.min()),
    }


def split_continuous_prices(prices: pd.Series) -> tuple[pd.Series, list[dict[str, Any]]]:
    adjusted: list[float] = []
    events: list[dict[str, Any]] = []
    multiplier = 1.0
    previous: float | None = None
    for index, value in prices.items():
        price = float(value)
        candidate = price * multiplier
        if previous is not None and previous > 0 and candidate > 0:
            ratio = candidate / previous
            if ratio < 0.45:
                split_ratio = round(1 / ratio)
                if split_ratio >= 2 and abs(ratio * split_ratio - 1) <= 0.18:
                    multiplier *= split_ratio
                    candidate = price * multiplier
                    events.append(
                        {
                            "date": index.date().isoformat() if hasattr(index, "date") else str(index),
                            "type": "split_down",
                            "ratio": split_ratio,
                        }
                    )
            elif ratio > 2.2:
                split_ratio = round(ratio)
                if split_ratio >= 2 and abs(ratio / split_ratio - 1) <= 0.18:
                    multiplier /= split_ratio
                    candidate = price * multiplier
                    events.append(
                        {
                            "date": index.date().isoformat() if hasattr(index, "date") else str(index),
                            "type": "reverse_split",
                            "ratio": split_ratio,
                        }
                    )
        adjusted.append(candidate)
        previous = candidate
    return pd.Series(adjusted, index=prices.index, dtype=float), events


def build_daily_portfolio(trades: pd.DataFrame, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    segments = portfolio_segments(trades, panels)
    if segments.empty:
        return pd.DataFrame()
    segments["Date"] = pd.to_datetime(segments["Date"])
    combined = segments.groupby("Date")["GrossReturn"].mean()
    long_return = segments[segments["Side"].eq("Long")].groupby("Date")["GrossReturn"].mean()
    short_return = segments[segments["Side"].eq("Short")].groupby("Date")["GrossReturn"].mean()
    daily = pd.DataFrame(index=combined.index)
    daily["GrossReturn"] = combined
    daily["LongGrossReturn"] = long_return.reindex(daily.index).fillna(0.0)
    daily["ShortGrossReturn"] = short_return.reindex(daily.index).fillna(0.0)
    daily["ActivePositions"] = segments.groupby("Date")["Code"].count().reindex(daily.index).fillna(0).astype(int)
    daily["LongPositions"] = (
        segments[segments["Side"].eq("Long")].groupby("Date")["Code"].count().reindex(daily.index).fillna(0).astype(int)
    )
    daily["ShortPositions"] = (
        segments[segments["Side"].eq("Short")].groupby("Date")["Code"].count().reindex(daily.index).fillna(0).astype(int)
    )
    daily["Equity"] = (1 + daily["GrossReturn"]).cumprod()
    daily["LongEquity"] = (1 + daily["LongGrossReturn"]).cumprod()
    daily["ShortEquity"] = (1 + daily["ShortGrossReturn"]).cumprod()
    daily["Drawdown"] = daily["Equity"] / daily["Equity"].cummax() - 1
    daily = daily.reset_index()
    daily["Date"] = daily["Date"].dt.date.astype(str)
    return daily


def add_benchmark_to_daily(daily: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if daily.empty or not config.benchmark_code:
        return daily, None
    benchmark_path = find_stock_file(PROJECT_ROOT / "data" / "price", config.benchmark_code)
    if benchmark_path is None:
        raise FileNotFoundError(f"missing_benchmark_price_csv:{config.benchmark_code}")
    columns = csv_columns_canonical(benchmark_path)
    close_column = "close_adj" if "close_adj" in columns else ("Close" if "Close" in columns else None)
    if "Date" not in columns or close_column is None:
        raise ValueError(f"benchmark_price_csv_missing_required_columns:{config.benchmark_code}")
    price = read_csv_canonical(benchmark_path, usecols=["Date", close_column])
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price["BenchmarkClose"] = pd.to_numeric(price[close_column], errors="coerce")
    price = price.dropna(subset=["Date", "BenchmarkClose"])
    price = price[price["BenchmarkClose"].gt(0)].sort_values("Date").drop_duplicates("Date", keep="last")
    daily_dates = pd.to_datetime(daily["Date"], errors="coerce")
    start = daily_dates.min()
    end = daily_dates.max()
    benchmark_start = config.start_date if config.start_date is not None else start
    price = price[price["Date"].between(benchmark_start, end)].copy()
    if price.empty:
        raise ValueError(f"benchmark_has_no_overlap:{config.benchmark_code}")
    price = price.set_index("Date")
    price["BenchmarkClose"], split_events = split_continuous_prices(price["BenchmarkClose"])
    price = price.reset_index()
    price["BenchmarkReturn"] = price["BenchmarkClose"].pct_change()
    benchmark_metrics = equity_metrics(price["BenchmarkReturn"].dropna())
    benchmark_metrics["TotalReturn"] = float(price["BenchmarkClose"].iloc[-1] / price["BenchmarkClose"].iloc[0] - 1)
    benchmark_metrics["MaxDrawdown"] = float(
        (price["BenchmarkClose"] / price["BenchmarkClose"].cummax() - 1).min()
    )
    aligned = price.set_index("Date").reindex(daily_dates).ffill()
    if aligned["BenchmarkClose"].isna().all():
        raise ValueError(f"benchmark_has_no_overlap:{config.benchmark_code}")
    aligned["BenchmarkClose"] = aligned["BenchmarkClose"].ffill().bfill()
    base = float(price["BenchmarkClose"].iloc[0])
    daily = daily.copy()
    daily["BenchmarkReturn"] = aligned["BenchmarkClose"].pct_change().fillna(0.0).to_numpy()
    daily["BenchmarkEquity"] = (aligned["BenchmarkClose"] / base).to_numpy()
    name = config.benchmark_name or f"{config.benchmark_code} 買進持有"
    return daily, {
        "benchmark_code": config.benchmark_code,
        "benchmark_name": name,
        "benchmark_path": str(benchmark_path.relative_to(PROJECT_ROOT)),
        "benchmark_start": str(price["Date"].iloc[0].date()),
        "benchmark_end": str(price["Date"].iloc[-1].date()),
        "split_adjustments": split_events,
        "benchmark_metrics": benchmark_metrics,
    }


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("All", trades)]
    if not trades.empty:
        groups.extend((side, group) for side, group in trades.groupby("Side"))
    for label, group in groups:
        if group.empty:
            continue
        positive = group.loc[group["NetReturn"].gt(0), "NetReturn"].sum()
        negative = group.loc[group["NetReturn"].lt(0), "NetReturn"].sum()
        rows.append(
            {
                "Side": label,
                "Trades": int(len(group)),
                "WinRateGross": float(group["GrossReturn"].gt(0).mean()),
                "WinRateNet": float(group["NetReturn"].gt(0).mean()),
                "AverageGrossReturn": float(group["GrossReturn"].mean()),
                "MedianGrossReturn": float(group["GrossReturn"].median()),
                "AverageNetReturn": float(group["NetReturn"].mean()),
                "MedianNetReturn": float(group["NetReturn"].median()),
                "TotalNetReturnSum": float(group["NetReturn"].sum()),
                "AverageHoldingDays": float(group["HoldingDays"].mean()),
                "MedianHoldingDays": float(group["HoldingDays"].median()),
                "ProfitFactorNet": float(positive / abs(negative)) if negative < 0 else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_by_stock(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby(["Code", "Name"], as_index=False).agg(
        Trades=("NetReturn", "size"),
        LongTrades=("Side", lambda values: int((values == "Long").sum())),
        ShortTrades=("Side", lambda values: int((values == "Short").sum())),
        AverageNetReturn=("NetReturn", "mean"),
        MedianNetReturn=("NetReturn", "median"),
        TotalNetReturnSum=("NetReturn", "sum"),
        WinRateNet=("NetReturn", lambda values: float((values > 0).mean())),
        AverageHoldingDays=("HoldingDays", "mean"),
    )
    return grouped.sort_values("TotalNetReturnSum", ascending=False).reset_index(drop=True)


def summarize_years(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["Year"] = pd.to_datetime(frame["Date"]).dt.year
    rows = []
    for year, group in frame.groupby("Year"):
        rows.append(
            {
                "Year": int(year),
                "GrossReturn": float((1 + group["GrossReturn"]).prod() - 1),
                "LongGrossReturn": float((1 + group["LongGrossReturn"]).prod() - 1),
                "ShortGrossReturn": float((1 + group["ShortGrossReturn"]).prod() - 1),
                "BenchmarkReturn": (
                    float((1 + group["BenchmarkReturn"]).prod() - 1) if "BenchmarkReturn" in group.columns else None
                ),
                "AverageActivePositions": float(group["ActivePositions"].mean()),
                "MaxDrawdown": float((group["Equity"] / group["Equity"].cummax() - 1).min()),
            }
        )
    return pd.DataFrame(rows)


def portfolio_metric_table(
    daily: pd.DataFrame,
    side_mode: str,
    benchmark_meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    specs = [("Combined", "GrossReturn")]
    if side_mode in {"both", "long-only"}:
        specs.append(("LongOnly", "LongGrossReturn"))
    if side_mode in {"both", "short-only"}:
        specs.append(("ShortOnly", "ShortGrossReturn"))
    metrics = []
    for label, column in specs:
        row = {"Portfolio": label, **equity_metrics(daily[column])}
        row["AverageActivePositions"] = float(daily["ActivePositions"].mean()) if label == "Combined" else None
        metrics.append(row)
    if benchmark_meta and benchmark_meta.get("benchmark_metrics"):
        metrics.append(
            {
                "Portfolio": "Benchmark0050",
                **benchmark_meta["benchmark_metrics"],
                "AverageActivePositions": None,
            }
        )
    return pd.DataFrame(metrics)


def pct_columns() -> set[str]:
    return {
        "GrossReturn",
        "NetReturn",
        "WinRateGross",
        "WinRateNet",
        "AverageGrossReturn",
        "MedianGrossReturn",
        "AverageNetReturn",
        "MedianNetReturn",
        "TotalNetReturnSum",
        "TotalReturn",
        "AnnualizedReturn",
        "AnnualizedVolatility",
        "MaxDrawdown",
        "MarginSurgeThreshold",
        "MarginDropThreshold",
        "EntryMarginChange20D",
        "ExitMarginChange20D",
        "AverageNetReturn",
        "MedianNetReturn",
        "GrossReturn",
        "LongGrossReturn",
        "ShortGrossReturn",
    }


DISPLAY_LABELS = {
    "Portfolio": "投資組合",
    "TradingDays": "交易日數",
    "TotalReturn": "累積報酬",
    "AnnualizedReturn": "年化報酬",
    "AnnualizedVolatility": "年化波動",
    "Sharpe": "夏普比率",
    "MaxDrawdown": "最大回撤",
    "AverageActivePositions": "平均持倉檔數",
    "Side": "方向",
    "Trades": "交易筆數",
    "WinRateGross": "未扣成本勝率",
    "WinRateNet": "扣成本勝率",
    "AverageGrossReturn": "平均未扣成本報酬",
    "MedianGrossReturn": "中位未扣成本報酬",
    "AverageNetReturn": "平均淨報酬",
    "MedianNetReturn": "中位淨報酬",
    "TotalNetReturnSum": "淨報酬加總",
    "AverageHoldingDays": "平均持有交易日",
    "MedianHoldingDays": "中位持有交易日",
    "ProfitFactorNet": "淨利潤因子",
    "Year": "年度",
    "GrossReturn": "未扣成本報酬",
    "LongGrossReturn": "做多報酬",
    "ShortGrossReturn": "做空報酬",
    "BenchmarkReturn": "0050 買進持有報酬",
    "BenchmarkEquity": "0050 買進持有淨值",
    "Code": "代碼",
    "Name": "名稱",
    "LongTrades": "做多筆數",
    "ShortTrades": "做空筆數",
    "EntrySignalDate": "進場訊號日",
    "EntryExecDate": "進場成交日",
    "ExitSignalDate": "出場訊號日",
    "ExitExecDate": "出場成交日",
    "EntryPrice": "進場價",
    "ExitPrice": "出場價",
    "NetReturn": "淨報酬",
    "HoldingDays": "持有交易日",
    "ExitReason": "出場原因",
    "EntryMarginChange20D": "進場融資20日變化",
    "ExitMarginChange20D": "出場融資20日變化",
    "EntryMarginBalance": "進場融資餘額",
    "ExitMarginBalance": "出場融資餘額",
    "MarginSurgeThreshold": "融資大漲門檻",
    "MarginDropThreshold": "融資大跌門檻",
    "Reason": "跳過原因",
}


VALUE_TRANSLATIONS = {
    "Portfolio": {
        "Combined": "策略整體",
        "LongOnly": "只做多",
        "ShortOnly": "只做空",
        "Benchmark0050": "0050 買進持有",
    },
    "Side": {
        "All": "全部",
        "Long": "做多",
        "Short": "做空",
    },
    "ExitReason": {
        "opposite_margin_drop_signal": "出現融資大跌訊號",
        "margin_change_back_to_nonpositive": "融資變化回到零以下",
        "opposite_margin_surge_signal": "出現融資大漲訊號",
        "margin_change_back_to_nonnegative": "融資變化回到零以上",
        "max_holding_days": "達到最大持有天數",
        "forced_last_close": "資料最後一日強制平倉",
    },
    "Reason": {
        "missing_margin_csv": "缺融資資料",
        "missing_price_csv": "缺價格資料",
        "margin_csv_missing_required_columns": "融資欄位不足",
        "price_csv_missing_required_columns": "價格欄位不足",
        "insufficient_joined_rows": "價格與融資可合併資料不足",
        "insufficient_valid_signal_rows": "有效訊號資料不足",
    },
}


def display_label(column: str) -> str:
    return DISPLAY_LABELS.get(column, column)


def display_value(column: str, value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return VALUE_TRANSLATIONS.get(column, {}).get(text, text)


def table_html(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 30) -> str:
    if df.empty:
        return '<p class="muted">沒有資料。</p>'
    data = df.copy()
    if columns is not None:
        data = data[[column for column in columns if column in data.columns]]
    data = data.head(max_rows)
    headers = "".join(f"<th>{html.escape(display_label(str(column)))}</th>" for column in data.columns)
    rows = []
    pct = pct_columns()
    for _, row in data.iterrows():
        cells = []
        for column, value in row.items():
            if column in pct:
                text = fmt_pct(value)
            elif isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                text = fmt_num(value, 2)
            elif pd.isna(value):
                text = ""
            else:
                text = display_value(str(column), value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def equity_svg(daily: pd.DataFrame, side_mode: str) -> str:
    if daily.empty:
        return ""
    width = 980
    height = 300
    pad_left = 56
    pad_right = 22
    pad_top = 22
    pad_bottom = 34
    series_specs = [("Equity", "#111827", "策略整體")]
    if side_mode == "both":
        series_specs.extend(
            [
                ("LongEquity", "#047857", "只做多"),
                ("ShortEquity", "#b91c1c", "只做空"),
            ]
        )
    if "BenchmarkEquity" in daily.columns:
        series_specs.append(("BenchmarkEquity", "#2563eb", "0050 買進持有"))
    values = pd.concat([daily[column] for column, _, _ in series_specs]).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return ""
    ymin = float(values.min())
    ymax = float(values.max())
    if math.isclose(ymin, ymax):
        ymin *= 0.95
        ymax *= 1.05
    ypad = (ymax - ymin) * 0.08
    ymin -= ypad
    ymax += ypad
    n = len(daily)

    def x_pos(index: int) -> float:
        if n <= 1:
            return pad_left
        return pad_left + index / (n - 1) * (width - pad_left - pad_right)

    def y_pos(value: float) -> float:
        return pad_top + (ymax - value) / (ymax - ymin) * (height - pad_top - pad_bottom)

    paths = []
    for column, color, _ in series_specs:
        points = " ".join(f"{x_pos(index):.1f},{y_pos(float(value)):.1f}" for index, value in enumerate(daily[column]))
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />')
    y_ticks = []
    for fraction in [0, 0.25, 0.5, 0.75, 1]:
        value = ymin + (ymax - ymin) * fraction
        y = y_pos(value)
        y_ticks.append(
            f'<line x1="{pad_left}" x2="{width - pad_right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb" />'
            f'<text x="{pad_left - 8}" y="{y + 4:.1f}" text-anchor="end">{html.escape(fmt_num(value, 2))}</text>'
        )
    first_date = html.escape(str(daily["Date"].iloc[0]))
    last_date = html.escape(str(daily["Date"].iloc[-1]))
    legend = "".join(
        f'<span><i style="background:{color}"></i>{html.escape(label)}</span>' for _, color, label in series_specs
    )
    return f"""
<div class="legend">{legend}</div>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="策略淨值曲線">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white" />
  {''.join(y_ticks)}
  <line x1="{pad_left}" x2="{width - pad_right}" y1="{height - pad_bottom}" y2="{height - pad_bottom}" stroke="#9ca3af" />
  {''.join(paths)}
  <text x="{pad_left}" y="{height - 8}" text-anchor="start">{first_date}</text>
  <text x="{width - pad_right}" y="{height - 8}" text-anchor="end">{last_date}</text>
</svg>
"""


def build_report(
    config: Config,
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    trade_summary: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    by_stock: pd.DataFrame,
    skipped: pd.DataFrame,
    benchmark_meta: dict[str, Any] | None,
) -> str:
    all_summary = trade_summary[trade_summary["Side"].eq("All")].iloc[0] if not trade_summary.empty else {}
    combined_metrics = (
        portfolio_metrics[portfolio_metrics["Portfolio"].eq("Combined")].iloc[0] if not portfolio_metrics.empty else {}
    )
    benchmark_metrics = (
        portfolio_metrics[portfolio_metrics["Portfolio"].eq("Benchmark0050")].iloc[0]
        if not portfolio_metrics.empty and portfolio_metrics["Portfolio"].eq("Benchmark0050").any()
        else {}
    )
    start = daily["Date"].iloc[0] if not daily.empty else ""
    end = daily["Date"].iloc[-1] if not daily.empty else ""
    side_label = {
        "both": "融資大漲做空、融資大跌做多",
        "long-only": "只在融資大跌後做多",
        "short-only": "只在融資大漲後做空",
    }[config.side_mode]
    start_rule = (
        f"本次只允許 {config.start_date.date().isoformat()} 之後的新進場訊號。"
        if config.start_date is not None
        else "本次使用完整可用期間的新進場訊號。"
    )
    cards = [
        ("策略方向", side_label),
        ("回測期間", f"{start} 至 {end}"),
        ("交易筆數", fmt_num(all_summary.get("Trades"), 0) if len(all_summary) else ""),
        ("平均淨報酬", fmt_pct(all_summary.get("AverageNetReturn")) if len(all_summary) else ""),
        ("扣成本勝率", fmt_pct(all_summary.get("WinRateNet")) if len(all_summary) else ""),
        ("策略累積報酬", fmt_pct(combined_metrics.get("TotalReturn")) if len(combined_metrics) else ""),
        ("0050累積報酬", fmt_pct(benchmark_metrics.get("TotalReturn")) if len(benchmark_metrics) else ""),
        ("最大回撤", fmt_pct(combined_metrics.get("MaxDrawdown")) if len(combined_metrics) else ""),
    ]
    card_html = "".join(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>' for label, value in cards)
    entry_rules = []
    if config.side_mode in {"both", "short-only"}:
        entry_rules.append(
            "<li>做空進場：融資大漲訊號出現後，隔一個交易日用復權開盤價放空。</li>"
        )
    if config.side_mode in {"both", "long-only"}:
        entry_rules.append(
            "<li>做多進場：融資大跌訊號出現後，隔一個交易日用復權開盤價買進。</li>"
        )
    exit_rules = []
    if config.side_mode in {"both", "short-only"}:
        exit_rules.append(
            f"<li>做空出場：{config.window} 日融資變化率回到零以下，或出現融資大跌訊號，隔一個交易日用復權開盤價回補。</li>"
        )
    if config.side_mode in {"both", "long-only"}:
        exit_rules.append(
            f"<li>做多出場：{config.window} 日融資變化率回到零以上，或出現融資大漲訊號，隔一個交易日用復權開盤價賣出。</li>"
        )
    benchmark_line = "比較基準：未啟用。"
    if benchmark_meta:
        split_events = benchmark_meta.get("split_adjustments") or []
        split_text = ""
        if split_events:
            split_text = "；已偵測並連續化分割事件：" + "、".join(
                f"{event['date']} 約 {event['ratio']}:1" for event in split_events
            )
        benchmark_line = (
            f"比較基準：{html.escape(benchmark_meta['benchmark_name'])}，使用同一段日期的復權收盤價買進持有{split_text}。"
        )
    assumptions = f"""
<ul>
  <li>融資大漲：每檔股票自己的 {config.window} 日融資餘額變化率大於等於歷史 {config.change_top_quantile:.0%} 分位。</li>
  <li>融資大跌：每檔股票自己的 {config.window} 日融資餘額變化率小於等於歷史 {config.change_bottom_quantile:.0%} 分位。</li>
  {''.join(entry_rules)}
  {''.join(exit_rules)}
  <li>{start_rule}</li>
  <li>{benchmark_line}</li>
  <li>逐筆淨報酬扣除來回 {config.round_trip_cost_bps:.0f} 基點交易成本，不含融券借券利息與借券可得性限制。</li>
  <li>投資組合淨值曲線是所有當日有效持倉等權平均的未扣成本日報酬，主要用來觀察時間序列穩定性。</li>
</ul>
"""
    portfolio_columns = [
        "Portfolio",
        "TradingDays",
        "TotalReturn",
        "AnnualizedReturn",
        "AnnualizedVolatility",
        "Sharpe",
        "MaxDrawdown",
        "AverageActivePositions",
    ]
    trade_summary_columns = [
        "Side",
        "Trades",
        "WinRateGross",
        "WinRateNet",
        "AverageGrossReturn",
        "MedianGrossReturn",
        "AverageNetReturn",
        "MedianNetReturn",
        "AverageHoldingDays",
        "ProfitFactorNet",
    ]
    yearly_columns = [
        "Year",
        "GrossReturn",
        "LongGrossReturn",
        "ShortGrossReturn",
        "BenchmarkReturn",
        "AverageActivePositions",
        "MaxDrawdown",
    ]
    stock_columns = [
        "Code",
        "Name",
        "Trades",
        "AverageNetReturn",
        "MedianNetReturn",
        "TotalNetReturnSum",
        "WinRateNet",
        "AverageHoldingDays",
    ]
    trade_columns = [
        "Code",
        "Name",
        "EntrySignalDate",
        "EntryExecDate",
        "ExitSignalDate",
        "ExitExecDate",
        "EntryPrice",
        "ExitPrice",
        "NetReturn",
        "HoldingDays",
        "ExitReason",
        "EntryMarginChange20D",
        "ExitMarginChange20D",
    ]
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>融資反向策略回測報告</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f8fafc; }}
    header {{ padding: 24px 32px 12px; background: white; border-bottom: 1px solid #e5e7eb; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    main {{ padding: 20px 32px 48px; }}
    .muted {{ color: #64748b; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-top: 16px; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px; }}
    .label {{ color: #64748b; font-size: 12px; }}
    .value {{ margin-top: 6px; font-size: 20px; font-weight: 700; }}
    section {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f1f5f9; color: #334155; position: sticky; top: 0; }}
    .scroll {{ overflow-x: auto; }}
    svg {{ width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 8px; }}
    svg text {{ fill: #475569; font-size: 12px; }}
    .legend {{ display: flex; gap: 18px; align-items: center; margin-bottom: 8px; color: #475569; font-size: 13px; }}
    .legend i {{ display: inline-block; width: 18px; height: 3px; margin-right: 6px; vertical-align: middle; }}
    ul {{ margin: 0; padding-left: 20px; line-height: 1.7; color: #334155; }}
  </style>
</head>
<body>
  <header>
    <h1>融資反向策略回測報告</h1>
    <div class="muted">{side_label}；資料來源為價格資料與融資資料，股票範圍為上市普通股。</div>
    <div class="cards">{card_html}</div>
  </header>
  <main>
    <section>
      <h2>回測規則</h2>
      {assumptions}
    </section>
    <section>
      <h2>投資組合淨值曲線</h2>
      {equity_svg(daily, config.side_mode)}
    </section>
    <section>
      <h2>投資組合績效</h2>
      <div class="scroll">{table_html(portfolio_metrics, portfolio_columns)}</div>
    </section>
    <section>
      <h2>交易摘要</h2>
      <div class="scroll">{table_html(trade_summary, trade_summary_columns)}</div>
    </section>
    <section>
      <h2>年度報酬</h2>
      <div class="scroll">{table_html(yearly, yearly_columns, max_rows=80)}</div>
    </section>
    <section>
      <h2>淨報酬加總最高股票</h2>
      <div class="scroll">{table_html(by_stock, stock_columns, max_rows=30)}</div>
    </section>
    <section>
      <h2>淨報酬加總最低股票</h2>
      <div class="scroll">{table_html(by_stock.sort_values("TotalNetReturnSum"), stock_columns, max_rows=30)}</div>
    </section>
    <section>
      <h2>最差交易</h2>
      <div class="scroll">{table_html(trades.sort_values("NetReturn"), trade_columns, max_rows=30)}</div>
    </section>
    <section>
      <h2>跳過股票</h2>
      <div class="scroll">{table_html(skipped, ["Code", "Name", "Reason"], max_rows=80)}</div>
    </section>
  </main>
</body>
</html>
"""


REPORT_ENGLISH_TERMS = [
    "Portfolio",
    "Trade Summary",
    "Yearly Return",
    "Best Stocks",
    "Worst Stocks",
    "Worst Trades",
    "Skipped",
    "Combined",
    "LongOnly",
    "ShortOnly",
    "GrossReturn",
    "NetReturn",
    "WinRate",
    "ProfitFactor",
    "AverageActivePositions",
    "ExitReason",
    "Benchmark",
]


def assert_chinese_report_text(report: str) -> None:
    visible = re.sub(r"<style\b.*?</style>", " ", report, flags=re.IGNORECASE | re.DOTALL)
    visible = re.sub(r"<script\b.*?</script>", " ", visible, flags=re.IGNORECASE | re.DOTALL)
    visible = re.sub(r"<[^>]+>", " ", visible)
    leftovers = [term for term in REPORT_ENGLISH_TERMS if term in visible]
    if leftovers:
        raise ValueError(f"report_contains_english_ui_text: {', '.join(leftovers)}")


def write_outputs(
    config: Config,
    universe_count: int,
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    trade_summary: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    by_stock: pd.DataFrame,
    skipped: pd.DataFrame,
    benchmark_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.viz_path.parent.mkdir(parents=True, exist_ok=True)
    csv_outputs = {
        "trades": config.output_dir / "trades.csv",
        "daily_portfolio_returns": config.output_dir / "daily_portfolio_returns.csv",
        "trade_summary": config.output_dir / "trade_summary.csv",
        "portfolio_metrics": config.output_dir / "portfolio_metrics.csv",
        "yearly_returns": config.output_dir / "yearly_returns.csv",
        "per_stock_summary": config.output_dir / "per_stock_summary.csv",
        "skipped": config.output_dir / "skipped.csv",
    }
    trades.to_csv(csv_outputs["trades"], index=False, encoding="utf-8-sig")
    daily.to_csv(csv_outputs["daily_portfolio_returns"], index=False, encoding="utf-8-sig")
    trade_summary.to_csv(csv_outputs["trade_summary"], index=False, encoding="utf-8-sig")
    portfolio_metrics.to_csv(csv_outputs["portfolio_metrics"], index=False, encoding="utf-8-sig")
    yearly.to_csv(csv_outputs["yearly_returns"], index=False, encoding="utf-8-sig")
    by_stock.to_csv(csv_outputs["per_stock_summary"], index=False, encoding="utf-8-sig")
    skipped.to_csv(csv_outputs["skipped"], index=False, encoding="utf-8-sig")

    report = build_report(config, trades, daily, trade_summary, portfolio_metrics, yearly, by_stock, skipped, benchmark_meta)
    assert_chinese_report_text(report)
    config.viz_path.write_text(report, encoding="utf-8-sig")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "universe_stocks_attempted": int(universe_count),
        "stocks_backtested": int(universe_count - len(skipped)),
        "stocks_with_trades": int(len(trades["Code"].drop_duplicates())) if not trades.empty else 0,
        "trades": int(len(trades)),
        "long_trades": int(trades["Side"].eq("Long").sum()) if not trades.empty else 0,
        "short_trades": int(trades["Side"].eq("Short").sum()) if not trades.empty else 0,
        "skipped": int(len(skipped)),
        "round_trip_cost_bps": config.round_trip_cost_bps,
        "window": config.window,
        "change_top_quantile": config.change_top_quantile,
        "change_bottom_quantile": config.change_bottom_quantile,
        "side_mode": config.side_mode,
        "lookback_years": config.lookback_years,
        "start_date": config.start_date.date().isoformat() if config.start_date is not None else None,
        "benchmark": benchmark_meta,
        "report_chinese_text_check": "passed",
        "output_dir": project_relative(config.output_dir),
        "report": project_relative(config.viz_path),
    }
    if not portfolio_metrics.empty:
        summary["portfolio_metrics"] = portfolio_metrics.to_dict("records")
    if not trade_summary.empty:
        summary["trade_summary"] = trade_summary.to_dict("records")
    summary = json_safe(summary)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    return summary


def main() -> None:
    args = parse_args()
    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    config = Config(
        window=args.window,
        change_top_quantile=args.change_top_quantile,
        change_bottom_quantile=args.change_bottom_quantile,
        min_rows=args.min_rows,
        round_trip_cost_bps=args.round_trip_cost_bps,
        max_holding_days=args.max_holding_days if args.max_holding_days > 0 else None,
        side_mode=args.side_mode,
        lookback_years=args.lookback_years if args.lookback_years > 0 else None,
        start_date=start_date,
        benchmark_code=args.benchmark_code.strip() or None,
        benchmark_name=args.benchmark_name.strip() or None,
        output_dir=resolve_project_path(args.output_dir),
        viz_path=resolve_project_path(args.viz_path),
        codes=set(args.codes) if args.codes else None,
        max_stocks=args.max_stocks,
    )
    universe = stock_universe(config)
    if config.start_date is None and config.lookback_years is not None:
        config.start_date = latest_price_date(universe) - pd.DateOffset(years=config.lookback_years)
    all_trades: list[pd.DataFrame] = []
    panels: dict[str, pd.DataFrame] = {}
    skipped: list[dict[str, str]] = []
    for _, stock in universe.iterrows():
        code = str(stock["Code"])
        name = str(stock["Name"])
        try:
            trade_df, panel, _ = backtest_stock(code, name, config)
            panels[code] = panel
            if not trade_df.empty:
                all_trades.append(trade_df)
        except Exception as exc:  # noqa: BLE001 - keep batch report complete.
            skipped.append({"Code": code, "Name": name, "Reason": str(exc)})

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    daily = build_daily_portfolio(trades, panels)
    daily, benchmark_meta = add_benchmark_to_daily(daily, config)
    trade_summary = summarize_trades(trades)
    portfolio_metrics = portfolio_metric_table(daily, config.side_mode, benchmark_meta)
    yearly = summarize_years(daily)
    by_stock = summarize_by_stock(trades)
    skipped_df = pd.DataFrame(skipped)
    summary = write_outputs(
        config,
        len(universe),
        trades,
        daily,
        trade_summary,
        portfolio_metrics,
        yearly,
        by_stock,
        skipped_df,
        benchmark_meta,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
