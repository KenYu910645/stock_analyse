"""Backtest participant-flow long-only signal baskets over recent history."""

from __future__ import annotations

import argparse
import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from column_schema import read_csv_canonical
from build_institutional_participation_report import compute_metrics
from strategies.trade_cost import SELL_TAX_RATE, TRANSACTION_FEE_RATE


DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "price"
INSTITUTIONAL_DIR = DATA_DIR / "institutional"
METADATA_PATH = DATA_DIR / "metadata.csv"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "institutional_flow_backtest"
DATA_VIZ_ROOT = PROJECT_ROOT / "data_viz" / "institutional_flow_backtest"

SIGNAL_WINDOWS = (1, 3, 5)
HOLDING_DAYS = (1, 5, 10, 20, 30, 60)
DEFAULT_TOP_N = 30
DEFAULT_MIN_DAILY_TURNOVER = 20_000_000
DEFAULT_STOP_LOSS_RATE = 0.10
DEFAULT_TRUST_ENTRY_STREAK_DAYS = 3
DEFAULT_MIN_TRUST_ENTRY_VALUE = 100_000_000
ROUND_TRIP_COST_RATE = TRANSACTION_FEE_RATE * 2 + SELL_TAX_RATE
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class ParticipantSpec:
    key: str
    label: str
    report_slug: str
    score_source_key: str
    inverse: bool
    color: str
    description: str


PARTICIPANTS = [
    ParticipantSpec(
        key="foreign",
        label="\u5916\u8cc7",
        report_slug="foreign",
        score_source_key="foreign",
        inverse=False,
        color="#2563eb",
        description="\u8ffd\u8e64\u5916\u8cc7\u8cb7\u8d85\u5f37\u5ea6\uff1a\u5916\u8cc7\u8cb7\u8d85\u8d8a\u5f37\uff0c\u6392\u540d\u8d8a\u524d\u9762\u3002",
    ),
    ParticipantSpec(
        key="trust",
        label="\u6295\u4fe1",
        report_slug="trust",
        score_source_key="trust",
        inverse=False,
        color="#d97706",
        description="\u8ffd\u8e64\u6295\u4fe1\u8cb7\u8d85\u5f37\u5ea6\uff1a\u6295\u4fe1\u8cb7\u8d85\u8d8a\u5f37\uff0c\u6392\u540d\u8d8a\u524d\u9762\u3002",
    ),
    ParticipantSpec(
        key="dealer",
        label="\u81ea\u71df\u5546",
        report_slug="dealer",
        score_source_key="dealer",
        inverse=False,
        color="#7c3aed",
        description="\u8ffd\u8e64\u81ea\u71df\u5546\u8cb7\u8d85\u5f37\u5ea6\uff1b\u672a\u62c6\u5206\u81ea\u884c\u8207\u907f\u96aa\u90e8\u4f4d\u3002",
    ),
    ParticipantSpec(
        key="other_inverse",
        label="\u53cd\u505a\u5176\u4ed6",
        report_slug="other_inverse",
        score_source_key="other",
        inverse=True,
        color="#64748b",
        description="\u53cd\u505a\u5176\u4ed6\uff1a\u5176\u4ed6\u8ce3\u8d85\u8d8a\u5f37\uff0c\u4ee3\u8868\u4e09\u5927\u6cd5\u4eba\u5408\u8a08\u8cb7\u8d85\u8d8a\u5f37\u3002",
    ),
]
PARTICIPANT_BY_KEY = {item.key: item for item in PARTICIPANTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest institutional-flow signal baskets.")
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-daily-turnover", type=float, default=DEFAULT_MIN_DAILY_TURNOVER)
    parser.add_argument("--stop-loss-rate", type=float, default=DEFAULT_STOP_LOSS_RATE)
    parser.add_argument("--trust-entry-streak-days", type=int, default=DEFAULT_TRUST_ENTRY_STREAK_DAYS)
    parser.add_argument("--min-trust-entry-value", type=float, default=DEFAULT_MIN_TRUST_ENTRY_VALUE)
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N stock limit for testing.")
    return parser.parse_args()


def code_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def stock_name_from_path(path: Path, fallback: str = "") -> str:
    return path.stem.split("_", 1)[1] if "_" in path.stem else fallback or code_from_path(path)


def path_by_code(directory: Path) -> dict[str, Path]:
    return {
        code_from_path(path): path
        for path in sorted(directory.glob("*.csv"))
        if not path.name.startswith("twse_")
    }


def fmt_pct(value: Any, digits: int = 2) -> str:
    number = to_float(value)
    return "" if number is None else f"{number * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if abs(number) >= 100_000_000:
        return f"{number / 100_000_000:,.{digits}f}\u5104"
    if abs(number) >= 10_000:
        return f"{number / 10_000:,.{digits}f}\u842c"
    return f"{number:,.{digits}f}"


def to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def listed_common_codes(limit: int | None) -> set[str]:
    metadata = pd.read_csv(METADATA_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    universe = metadata[metadata["\u985e\u578b"].eq("\u80a1\u7968") & metadata["\u5e02\u5834"].eq("\u4e0a\u5e02")]
    codes = sorted(universe["Code"].astype(str))
    if limit is not None:
        codes = codes[:limit]
    return set(codes)


def latest_price_date(price_paths: dict[str, Path], codes: set[str]) -> pd.Timestamp:
    latest: pd.Timestamp | None = None
    for code in sorted(codes):
        path = price_paths.get(code)
        if path is None:
            continue
        try:
            frame = read_csv_canonical(path, usecols=["Date"])
        except Exception:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
        if dates.empty:
            continue
        current = pd.Timestamp(dates.max())
        latest = current if latest is None else max(latest, current)
    if latest is None:
        raise ValueError("no_price_dates_found")
    return latest


def number_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def iso_date_strings(frame: pd.DataFrame, column: str = "Date") -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")


def load_stock_panel(code: str, price_path: Path, institutional_path: Path, start_buffer: pd.Timestamp) -> pd.DataFrame:
    price = read_csv_canonical(price_path, dtype={"Code": str})
    institutional = read_csv_canonical(institutional_path, dtype={"Code": str})
    institutional_dates = set(iso_date_strings(institutional).dropna())
    if not institutional_dates:
        return pd.DataFrame()

    price_dates = iso_date_strings(price)
    price = price[price_dates.isin(institutional_dates)].copy()
    if price.empty:
        return pd.DataFrame()

    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price = price.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    price = price[price["Date"].ge(start_buffer)].copy()
    if price.empty:
        return pd.DataFrame()

    for column in ["Capacity", "Turnover", "Open", "Close", "open_adj", "close_adj"]:
        if column in price.columns:
            price[column] = pd.to_numeric(price[column], errors="coerce")
    price["OpenExec"] = price["open_adj"] if "open_adj" in price.columns else price["Open"]
    price["CloseExec"] = price["close_adj"] if "close_adj" in price.columns else price["Close"]
    price_extra = price[["Date", "Turnover", "OpenExec", "CloseExec"]].copy()
    price_extra["DateText"] = price_extra["Date"].dt.strftime("%Y-%m-%d")

    metrics = compute_metrics(price, institutional)
    if metrics.empty:
        return pd.DataFrame()
    metrics = metrics[metrics["Date"].isin(institutional_dates)].copy()
    metrics["Date"] = pd.to_datetime(metrics["Date"], errors="coerce")
    metrics = metrics.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    metrics = metrics[metrics["Date"].ge(start_buffer)].copy()
    if metrics.empty:
        return pd.DataFrame()

    frame = metrics.merge(
        price_extra[["DateText", "Turnover", "OpenExec", "CloseExec"]],
        left_on=metrics["Date"].dt.strftime("%Y-%m-%d"),
        right_on="DateText",
        how="left",
        suffixes=("", "_price"),
    ).drop(columns=["key_0", "DateText"], errors="ignore")
    for column in ["Capacity", "Turnover", "OpenExec", "CloseExec"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Date", "Capacity", "Turnover", "OpenExec"])
    frame = frame[frame["Capacity"].gt(0) & frame["Turnover"].gt(0) & frame["OpenExec"].gt(0)].copy()
    if frame.empty:
        return frame

    average_price = (frame["Turnover"] / frame["Capacity"]).where(frame["Capacity"].gt(0), frame["OpenExec"])
    frame["foreign_net"] = number_series(frame, "foreign_buy") - number_series(frame, "foreign_sell")
    frame["trust_net"] = number_series(frame, "trust_buy") - number_series(frame, "trust_sell")
    frame["dealer_net"] = number_series(frame, "dealer_buy") - number_series(frame, "dealer_sell")
    frame["other_net"] = -(frame["foreign_net"] + frame["trust_net"] + frame["dealer_net"])
    for key in ["foreign", "trust", "dealer", "other"]:
        frame[f"{key}_net_value"] = frame[f"{key}_net"] * average_price
    frame["Code"] = code
    return frame.reset_index(drop=True)


def add_signals_and_returns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("Date").reset_index(drop=True)
    turnover = frame["Turnover"]
    for window in SIGNAL_WINDOWS:
        rolling_turnover = turnover.rolling(window, min_periods=window).sum()
        frame[f"turnover_{window}d"] = rolling_turnover
        for participant in PARTICIPANTS:
            source = frame[f"{participant.score_source_key}_net_value"]
            if participant.inverse:
                source = -source
            frame[f"score_{participant.key}_{window}d"] = source.rolling(window, min_periods=window).sum() / rolling_turnover
    for hold in HOLDING_DAYS:
        exit_open = frame["OpenExec"].shift(-(hold + 1))
        entry_open = frame["OpenExec"].shift(-1)
        frame[f"entry_date_{hold}d"] = frame["Date"].shift(-1)
        frame[f"exit_date_{hold}d"] = frame["Date"].shift(-(hold + 1))
        frame[f"gross_return_{hold}d"] = exit_open / entry_open - 1.0
        frame[f"net_return_{hold}d"] = frame[f"gross_return_{hold}d"] - ROUND_TRIP_COST_RATE
    return frame


def build_signal_panel(
    *,
    lookback_years: int,
    limit: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    price_paths = path_by_code(PRICE_DIR)
    institutional_paths = path_by_code(INSTITUTIONAL_DIR)
    allowed = listed_common_codes(limit)
    codes = sorted((set(price_paths) & set(institutional_paths)) & allowed)
    latest = latest_price_date(price_paths, codes)
    start = latest - pd.DateOffset(years=lookback_years)
    start_buffer = start - pd.Timedelta(days=120)

    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for index, code in enumerate(codes, start=1):
        try:
            frame = load_stock_panel(code, price_paths[code], institutional_paths[code], start_buffer)
            if frame.empty:
                skipped.append({"Code": code, "Reason": "empty_panel"})
                continue
            frame["Name"] = stock_name_from_path(institutional_paths[code], stock_name_from_path(price_paths[code], code))
            frame = add_signals_and_returns(frame)
            frame = frame[frame["Date"].ge(start)].copy()
            if frame.empty:
                skipped.append({"Code": code, "Reason": "no_rows_after_start"})
                continue
            rows.append(frame)
        except Exception as exc:
            skipped.append({"Code": code, "Reason": str(exc)})
        if index % 100 == 0 or index == len(codes):
            print(f"processed {index}/{len(codes)}")

    if not rows:
        raise SystemExit("no_signal_rows_built")
    panel = pd.concat(rows, ignore_index=True)
    meta = {
        "latest_date": latest.strftime("%Y-%m-%d"),
        "start_date": start.strftime("%Y-%m-%d"),
        "stock_count": len(codes),
        "loaded_stock_count": int(panel["Code"].nunique()),
        "skipped": skipped,
    }
    return panel, meta


def select_trades(
    panel: pd.DataFrame,
    *,
    participant: ParticipantSpec,
    signal_window: int,
    holding_days: int,
    top_n: int,
    min_daily_turnover: float,
) -> pd.DataFrame:
    score_col = f"score_{participant.key}_{signal_window}d"
    turnover_col = f"turnover_{signal_window}d"
    gross_col = f"gross_return_{holding_days}d"
    net_col = f"net_return_{holding_days}d"
    entry_col = f"entry_date_{holding_days}d"
    exit_col = f"exit_date_{holding_days}d"
    needed = [
        "Date",
        "Code",
        "Name",
        "Turnover",
        score_col,
        turnover_col,
        gross_col,
        net_col,
        entry_col,
        exit_col,
    ]
    candidates = panel[needed].copy()
    candidates = candidates.rename(
        columns={
            "Date": "SignalDate",
            score_col: "Score",
            turnover_col: "SignalWindowTurnover",
            gross_col: "GrossReturn",
            net_col: "NetReturn",
            entry_col: "EntryDate",
            exit_col: "ExitDate",
        }
    )
    min_window_turnover = min_daily_turnover * signal_window
    candidates = candidates[
        candidates["Score"].gt(0)
        & candidates["SignalWindowTurnover"].ge(min_window_turnover)
        & candidates["GrossReturn"].map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
        & candidates["NetReturn"].map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    ].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(["SignalDate", "Score", "Code"], ascending=[True, False, True])
    selected = candidates.groupby("SignalDate", group_keys=False).head(top_n).copy()
    selected["ParticipantKey"] = participant.key
    selected["Participant"] = participant.label
    selected["SignalWindow"] = signal_window
    selected["HoldingDays"] = holding_days
    selected["Rank"] = selected.groupby("SignalDate")["Score"].rank(ascending=False, method="first").astype(int)
    for column in ["SignalDate", "EntryDate", "ExitDate"]:
        selected[column] = pd.to_datetime(selected[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return selected


def basket_metrics(returns: pd.Series, holding_days: int) -> dict[str, float | int | None]:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return {
            "BasketAvgNetReturn": None,
            "BasketMedianNetReturn": None,
            "BasketWinRate": None,
            "AnnualizedAvgBasketReturn": None,
            "SignalDayCount": 0,
        }
    average = float(clean.mean())
    annualized = (1 + average) ** (TRADING_DAYS_PER_YEAR / holding_days) - 1 if average > -1 else None
    return {
        "BasketAvgNetReturn": average,
        "BasketMedianNetReturn": float(clean.median()),
        "BasketWinRate": float(clean.gt(0).mean()),
        "AnnualizedAvgBasketReturn": float(annualized) if annualized is not None else None,
        "SignalDayCount": int(len(clean)),
    }


def summarize_trades(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    basket_returns = (
        trades.groupby(["ParticipantKey", "Participant", "SignalWindow", "HoldingDays", "SignalDate"], dropna=False)
        .agg(BasketNetReturn=("NetReturn", "mean"), BasketGrossReturn=("GrossReturn", "mean"), PickCount=("Code", "count"))
        .reset_index()
        .sort_values(["ParticipantKey", "SignalWindow", "HoldingDays", "SignalDate"])
    )
    rows: list[dict[str, Any]] = []
    grouped = trades.groupby(["ParticipantKey", "Participant", "SignalWindow", "HoldingDays"], dropna=False)
    for keys, group in grouped:
        participant_key, participant, signal_window, holding_days = keys
        basket = basket_returns[
            basket_returns["ParticipantKey"].eq(participant_key)
            & basket_returns["SignalWindow"].eq(signal_window)
            & basket_returns["HoldingDays"].eq(holding_days)
        ]
        gains = group.loc[group["NetReturn"].gt(0), "NetReturn"].sum()
        losses = group.loc[group["NetReturn"].lt(0), "NetReturn"].sum()
        metrics = basket_metrics(basket["BasketNetReturn"], int(holding_days))
        rows.append(
            {
                "ParticipantKey": participant_key,
                "Participant": participant,
                "SignalWindow": int(signal_window),
                "HoldingDays": int(holding_days),
                "TradeCount": int(len(group)),
                "SignalDayCount": int(group["SignalDate"].nunique()),
                "AvgPicksPerSignalDay": float(len(group) / max(group["SignalDate"].nunique(), 1)),
                "AvgScore": float(group["Score"].mean()),
                "AvgGrossReturn": float(group["GrossReturn"].mean()),
                "AvgNetReturn": float(group["NetReturn"].mean()),
                "MedianNetReturn": float(group["NetReturn"].median()),
                "WinRate": float(group["NetReturn"].gt(0).mean()),
                "ProfitFactor": float(gains / abs(losses)) if losses < 0 else None,
                **metrics,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["ParticipantKey", "SignalWindow", "HoldingDays"]).reset_index(drop=True)
    return summary, basket_returns


def table_rows(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        cells = []
        for column, _label, kind in columns:
            value = getattr(row, column)
            css = ""
            if kind in {"pct", "return"}:
                numeric = to_float(value)
                if numeric is not None:
                    css = "pos" if numeric >= 0 else "neg"
            if kind == "pct":
                text = fmt_pct(value)
            elif kind == "num":
                text = fmt_num(value, 2)
            elif kind == "int":
                text = f"{int(value):,}" if pd.notna(value) else ""
            else:
                text = html.escape(str(value))
            cells.append(f'<td class="{css}">{text}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def metrics_table(frame: pd.DataFrame, include_participant: bool = False) -> str:
    columns = []
    if include_participant:
        columns.append(("Participant", "\u7fa4\u7d44", "text"))
    columns.extend([
        ("SignalWindow", "\u8a0a\u865f\u65e5\u6578", "int"),
        ("HoldingDays", "\u6301\u6709\u65e5\u6578", "int"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("SignalDayCount", "\u8a0a\u865f\u65e5\u6578", "int"),
        ("AvgNetReturn", "\u5e73\u5747\u55ae\u7b46\u5831\u916c", "pct"),
        ("WinRate", "\u52dd\u7387", "pct"),
        ("BasketAvgNetReturn", "\u8a0a\u865f\u7c43\u5b50\u5e73\u5747", "pct"),
        ("BasketWinRate", "\u7c43\u5b50\u52dd\u7387", "pct"),
        ("AnnualizedAvgBasketReturn", "\u7c43\u5b50\u5e73\u5747\u5e74\u5316\u53c3\u8003", "pct"),
        ("ProfitFactor", "\u7372\u5229\u56e0\u5b50", "num"),
    ])
    heads = "".join(f"<th>{label}</th>" for _column, label, _kind in columns)
    return f"<table><thead><tr>{heads}</tr></thead><tbody>{table_rows(frame, columns)}</tbody></table>"


def bar_chart_svg(frame: pd.DataFrame, metric: str, title: str, percent: bool = True) -> str:
    width = 980
    height = 360
    left = 62
    right = 24
    top = 28
    bottom = 72
    chart_width = width - left - right
    chart_height = height - top - bottom
    subset = frame.copy().sort_values(["SignalWindow", "HoldingDays"])
    labels = [f"{int(row.SignalWindow)}d/{int(row.HoldingDays)}d" for row in subset.itertuples(index=False)]
    values = [float(value) if pd.notna(value) and math.isfinite(float(value)) else 0.0 for value in subset[metric]]
    if not values:
        return ""
    max_abs = max(max(abs(value) for value in values), 0.001)
    zero_y = top + chart_height / 2
    bar_gap = 3
    bar_width = max(8, (chart_width - bar_gap * (len(values) - 1)) / len(values))
    bars = []
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = abs(value) / max_abs * (chart_height / 2 - 18)
        x = left + index * (bar_width + bar_gap)
        y = zero_y - bar_height if value >= 0 else zero_y
        text_y = y - 4 if value >= 0 else y + bar_height + 13
        color = "#0f766e" if value >= 0 else "#b91c1c"
        text = fmt_pct(value, 1) if percent else fmt_num(value, 1)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="2" fill="{color}"/>'
            f'<text x="{x + bar_width / 2:.1f}" y="{text_y:.1f}" text-anchor="middle">{text}</text>'
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 34}" text-anchor="middle" transform="rotate(-45 {x + bar_width / 2:.1f} {height - 34})">{html.escape(label)}</text>'
        )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <style>
    .axis {{ stroke: #334155; stroke-width: 1; }}
    .baseline {{ stroke: #475569; stroke-width: 1; stroke-dasharray: 5 5; }}
    text {{ fill: #475569; font-size: 10px; font-family: "Microsoft JhengHei", Arial, sans-serif; }}
  </style>
  <line x1="{left}" y1="{zero_y:.1f}" x2="{left + chart_width}" y2="{zero_y:.1f}" class="baseline"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>
  {''.join(bars)}
  <text x="{left + chart_width / 2}" y="16" text-anchor="middle">{html.escape(title)}</text>
</svg>
"""


def grouped_participant_bar_chart_svg(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    *,
    baseline: float = 0.0,
    percent: bool = True,
) -> str:
    width = 1180
    height = 470
    left = 70
    right = 28
    top = 54
    bottom = 98
    chart_width = width - left - right
    chart_height = height - top - bottom
    combos = sorted(
        {
            (int(row.SignalWindow), int(row.HoldingDays))
            for row in frame.itertuples(index=False)
            if pd.notna(row.SignalWindow) and pd.notna(row.HoldingDays)
        }
    )
    if not combos:
        return ""

    values: dict[tuple[str, int, int], float] = {}
    clean_values = [baseline]
    for row in frame.itertuples(index=False):
        value = to_float(getattr(row, metric))
        if value is None or not math.isfinite(value):
            continue
        signal_window = int(row.SignalWindow)
        holding_days = int(row.HoldingDays)
        values[(row.ParticipantKey, signal_window, holding_days)] = value
        clean_values.append(value)

    floor = min(clean_values)
    ceiling = max(clean_values)
    if math.isclose(floor, ceiling):
        floor -= 0.001
        ceiling += 0.001
    padding = max((ceiling - floor) * 0.12, 0.002 if percent else 1.0)
    floor -= padding
    ceiling += padding

    def y_for(value: float) -> float:
        return top + (ceiling - value) / (ceiling - floor) * chart_height

    def format_value(value: float) -> str:
        return fmt_pct(value, 1) if percent else fmt_num(value, 1)

    baseline_y = y_for(baseline)
    group_width = chart_width / len(combos)
    inner_gap = 3
    bar_width = max(7, min(16, (group_width - 16 - inner_gap * (len(PARTICIPANTS) - 1)) / len(PARTICIPANTS)))
    bar_pack_width = bar_width * len(PARTICIPANTS) + inner_gap * (len(PARTICIPANTS) - 1)

    grid_lines = []
    tick_count = 5
    for index in range(tick_count + 1):
        value = floor + (ceiling - floor) * index / tick_count
        y = y_for(value)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left - 8}" y="{y + 3:.1f}" text-anchor="end">{format_value(value)}</text>'
        )

    bars = []
    for combo_index, (signal_window, holding_days) in enumerate(combos):
        group_x = left + combo_index * group_width
        bar_start = group_x + (group_width - bar_pack_width) / 2
        label_x = group_x + group_width / 2
        bars.append(
            f'<text x="{label_x:.1f}" y="{height - 38}" text-anchor="middle" '
            f'transform="rotate(-45 {label_x:.1f} {height - 38})">{signal_window}/{holding_days}</text>'
        )
        for participant_index, participant in enumerate(PARTICIPANTS):
            value = values.get((participant.key, signal_window, holding_days))
            if value is None:
                continue
            y = y_for(value)
            rect_y = min(y, baseline_y)
            rect_height = max(abs(y - baseline_y), 1.5)
            x = bar_start + participant_index * (bar_width + inner_gap)
            tooltip = (
                f"{participant.label} {signal_window}\u65e5\u8a0a\u865f/"
                f"{holding_days}\u65e5\u6301\u6709 {format_value(value)}"
            )
            bars.append(
                f'<rect x="{x:.1f}" y="{rect_y:.1f}" width="{bar_width:.1f}" height="{rect_height:.1f}" '
                f'rx="1.5" fill="{participant.color}"><title>{html.escape(tooltip)}</title></rect>'
            )

    legend = []
    legend_x = left
    for participant in PARTICIPANTS:
        legend.append(
            f'<rect x="{legend_x}" y="24" width="12" height="12" rx="2" fill="{participant.color}"/>'
            f'<text x="{legend_x + 18}" y="34">{html.escape(participant.label)}</text>'
        )
        legend_x += 92

    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <style>
    .axis {{ stroke: #334155; stroke-width: 1; }}
    .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    .baseline {{ stroke: #334155; stroke-width: 1.2; stroke-dasharray: 5 5; }}
    text {{ fill: #475569; font-size: 10px; font-family: "Microsoft JhengHei", Arial, sans-serif; }}
    .title {{ fill: #172033; font-size: 15px; font-weight: 700; }}
  </style>
  <text x="{left + chart_width / 2}" y="17" text-anchor="middle" class="title">{html.escape(title)}</text>
  {''.join(legend)}
  {''.join(grid_lines)}
  <line x1="{left}" y1="{baseline_y:.1f}" x2="{left + chart_width}" y2="{baseline_y:.1f}" class="baseline"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>
  <line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" class="axis"/>
  {''.join(bars)}
  <text x="{left + chart_width / 2}" y="{height - 12}" text-anchor="middle">\u8a0a\u865f\u65e5\u6578 / \u6301\u6709\u65e5\u6578</text>
</svg>
"""


def histogram_svg(frame: pd.DataFrame, metric: str, title: str, *, percent: bool = True, bins: int = 14) -> str:
    width = 980
    height = 340
    left = 62
    right = 24
    top = 34
    bottom = 76
    chart_width = width - left - right
    chart_height = height - top - bottom
    values = pd.to_numeric(frame[metric], errors="coerce").dropna()
    values = values[np.isfinite(values)]
    if values.empty:
        return ""
    counts, edges = np.histogram(values.to_numpy(), bins=bins)
    max_count = max(int(counts.max()), 1)
    bar_gap = 4
    bar_width = max(10, (chart_width - bar_gap * (len(counts) - 1)) / len(counts))
    bars = []
    for index, count in enumerate(counts):
        x = left + index * (bar_width + bar_gap)
        bar_height = count / max_count * (chart_height - 18)
        y = top + chart_height - bar_height
        low = edges[index]
        high = edges[index + 1]
        if percent:
            label = f"{low * 100:.1f}%~{high * 100:.1f}%"
        else:
            label = f"{low:.0f}~{high:.0f}"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="2" fill="#d97706">'
            f"<title>{html.escape(label)}: {int(count):,}</title></rect>"
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle">{int(count):,}</text>'
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 34}" text-anchor="middle" '
            f'transform="rotate(-45 {x + bar_width / 2:.1f} {height - 34})">{html.escape(label)}</text>'
        )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
  <style>
    .axis {{ stroke: #334155; stroke-width: 1; }}
    .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    text {{ fill: #475569; font-size: 10px; font-family: "Microsoft JhengHei", Arial, sans-serif; }}
    .title {{ fill: #172033; font-size: 15px; font-weight: 700; }}
  </style>
  <text x="{left + chart_width / 2}" y="18" text-anchor="middle" class="title">{html.escape(title)}</text>
  <line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" class="axis"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>
  {''.join(bars)}
</svg>
"""


def next_stock_row(stock_frames: dict[str, pd.DataFrame], code: str, signal_date: pd.Timestamp) -> tuple[int, pd.Series] | None:
    frame = stock_frames.get(code)
    if frame is None or frame.empty:
        return None
    dates = frame["Date"].to_numpy(dtype="datetime64[ns]")
    position = int(np.searchsorted(dates, pd.Timestamp(signal_date).to_datetime64(), side="right"))
    if position >= len(frame):
        return None
    return position, frame.iloc[position]


def close_dynamic_trade(
    trade: dict[str, Any],
    *,
    exit_row: pd.Series,
    exit_position: int,
    exit_reason: str,
    exit_signal: dict[str, Any] | None,
) -> dict[str, Any]:
    exit_open = float(exit_row["OpenExec"])
    gross_return = exit_open / float(trade["EntryOpen"]) - 1.0
    exit_date = pd.Timestamp(exit_row["Date"])
    exit_signal_date = ""
    sell_score = None
    sell_strength = None
    sell_rank = None
    if exit_signal is not None:
        exit_signal_date = pd.Timestamp(exit_signal["SignalDate"]).strftime("%Y-%m-%d")
        sell_score = float(exit_signal["Score"])
        sell_strength = float(exit_signal["SellStrength"])
        sell_rank = int(exit_signal["SellRank"])
    return {
        "EntrySignalDate": pd.Timestamp(trade["EntrySignalDate"]).strftime("%Y-%m-%d"),
        "EntryDate": pd.Timestamp(trade["EntryDate"]).strftime("%Y-%m-%d"),
        "ExitSignalDate": exit_signal_date,
        "ExitDate": exit_date.strftime("%Y-%m-%d"),
        "Code": trade["Code"],
        "Name": trade["Name"],
        "EntryOpen": float(trade["EntryOpen"]),
        "ExitOpen": exit_open,
        "BuyScore": float(trade["BuyScore"]),
        "BuyRank": int(trade["BuyRank"]),
        "SellScore": sell_score,
        "SellStrength": sell_strength,
        "SellRank": sell_rank,
        "GrossReturn": gross_return,
        "NetReturn": gross_return - ROUND_TRIP_COST_RATE,
        "HoldingTradingDays": int(exit_position - trade["EntryPosition"]),
        "HoldingCalendarDays": int((exit_date - pd.Timestamp(trade["EntryDate"])).days),
        "ExitReason": exit_reason,
    }


def select_trust_dynamic_exit_trades(
    panel: pd.DataFrame,
    *,
    top_n: int,
    min_daily_turnover: float,
) -> pd.DataFrame:
    score_col = "score_trust_1d"
    turnover_col = "turnover_1d"
    needed = ["Date", "Code", "Name", "OpenExec", "Turnover", score_col, turnover_col]
    candidates = panel[needed].copy().rename(columns={"Date": "SignalDate", score_col: "Score", turnover_col: "SignalTurnover"})
    for column in ["OpenExec", "Turnover", "Score", "SignalTurnover"]:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates = candidates.dropna(subset=["SignalDate", "OpenExec", "Turnover", "Score", "SignalTurnover"])
    candidates = candidates[candidates["SignalTurnover"].ge(min_daily_turnover)].copy()
    if candidates.empty:
        return pd.DataFrame()

    buy_signals = candidates[candidates["Score"].gt(0)].sort_values(
        ["SignalDate", "Score", "Code"],
        ascending=[True, False, True],
    )
    buy_signals = buy_signals.groupby("SignalDate", group_keys=False).head(top_n).copy()
    buy_signals["BuyRank"] = buy_signals.groupby("SignalDate")["Score"].rank(ascending=False, method="first").astype(int)

    sell_signals = candidates[candidates["Score"].lt(0)].copy()
    sell_signals["SellStrength"] = -sell_signals["Score"]
    sell_signals = sell_signals.sort_values(["SignalDate", "SellStrength", "Code"], ascending=[True, False, True])
    sell_signals = sell_signals.groupby("SignalDate", group_keys=False).head(top_n).copy()
    sell_signals["SellRank"] = sell_signals.groupby("SignalDate")["SellStrength"].rank(ascending=False, method="first").astype(int)

    stock_frames = {
        str(code): group.sort_values("Date").reset_index(drop=True)
        for code, group in panel.groupby("Code", dropna=False)
    }
    buy_by_date = {date: group.to_dict("records") for date, group in buy_signals.groupby("SignalDate", sort=True)}
    sell_by_date = {
        date: {str(row["Code"]): row for row in group.to_dict("records")}
        for date, group in sell_signals.groupby("SignalDate", sort=True)
    }

    holdings: dict[str, dict[str, Any]] = {}
    closed: list[dict[str, Any]] = []
    for signal_date in sorted(set(buy_by_date) | set(sell_by_date)):
        for code, sell_signal in sell_by_date.get(signal_date, {}).items():
            if code not in holdings:
                continue
            exit_info = next_stock_row(stock_frames, code, signal_date)
            if exit_info is None:
                continue
            exit_position, exit_row = exit_info
            closed.append(
                close_dynamic_trade(
                    holdings.pop(code),
                    exit_row=exit_row,
                    exit_position=exit_position,
                    exit_reason="sell_signal",
                    exit_signal=sell_signal,
                )
            )

        for buy_signal in buy_by_date.get(signal_date, []):
            code = str(buy_signal["Code"])
            if code in holdings:
                continue
            entry_info = next_stock_row(stock_frames, code, signal_date)
            if entry_info is None:
                continue
            entry_position, entry_row = entry_info
            entry_open = float(entry_row["OpenExec"])
            if not math.isfinite(entry_open) or entry_open <= 0:
                continue
            holdings[code] = {
                "EntrySignalDate": signal_date,
                "EntryDate": pd.Timestamp(entry_row["Date"]),
                "EntryPosition": entry_position,
                "EntryOpen": entry_open,
                "Code": code,
                "Name": buy_signal["Name"],
                "BuyScore": float(buy_signal["Score"]),
                "BuyRank": int(buy_signal["BuyRank"]),
            }

    for code, trade in holdings.items():
        frame = stock_frames.get(code)
        if frame is None or frame.empty:
            continue
        exit_position = len(frame) - 1
        if exit_position <= trade["EntryPosition"]:
            continue
        closed.append(
            close_dynamic_trade(
                trade,
                exit_row=frame.iloc[exit_position],
                exit_position=exit_position,
                exit_reason="data_end_mark",
                exit_signal=None,
            )
        )

    trades = pd.DataFrame(closed)
    if trades.empty:
        return trades
    trades = trades.sort_values(["EntryDate", "BuyRank", "Code"]).reset_index(drop=True)
    return trades


def active_position_counts(trades: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(panel["Date"], errors="coerce").dropna().unique())
    events: dict[pd.Timestamp, int] = {}
    for row in trades.itertuples(index=False):
        entry_date = pd.Timestamp(row.EntryDate)
        exit_date = pd.Timestamp(row.ExitDate)
        events[entry_date] = events.get(entry_date, 0) + 1
        events[exit_date] = events.get(exit_date, 0) - 1
    active = 0
    rows = []
    for date in dates:
        timestamp = pd.Timestamp(date)
        active += events.get(timestamp, 0)
        rows.append({"Date": timestamp.strftime("%Y-%m-%d"), "ActivePositions": int(active)})
    return pd.DataFrame(rows)


def summarize_trust_dynamic_exit(
    trades: pd.DataFrame,
    active_counts: pd.DataFrame,
    *,
    meta: dict[str, Any],
    top_n: int,
    min_daily_turnover: float,
) -> pd.DataFrame:
    clean = pd.to_numeric(trades["NetReturn"], errors="coerce").dropna()
    clean = clean[np.isfinite(clean)]
    gains = clean[clean.gt(0)].sum()
    losses = clean[clean.lt(0)].sum()
    holding_days = pd.to_numeric(trades["HoldingTradingDays"], errors="coerce").dropna()
    sell_signal_count = int(trades["ExitReason"].eq("sell_signal").sum())
    active_series = pd.to_numeric(active_counts["ActivePositions"], errors="coerce") if not active_counts.empty else pd.Series(dtype=float)
    average = float(clean.mean()) if not clean.empty else None
    avg_holding = float(holding_days.mean()) if not holding_days.empty else None
    annualized = None
    if average is not None and avg_holding and average > -1:
        annualized = (1 + average) ** (TRADING_DAYS_PER_YEAR / avg_holding) - 1
    row = {
        "Strategy": "trust_dynamic_exit",
        "Participant": "\u6295\u4fe1",
        "SignalWindow": 1,
        "EntryTopN": int(top_n),
        "ExitTopN": int(top_n),
        "MinDailyTurnover": float(min_daily_turnover),
        "StartDate": meta.get("start_date"),
        "LatestDate": meta.get("latest_date"),
        "StockCount": meta.get("loaded_stock_count"),
        "TradeCount": int(len(trades)),
        "SellSignalExitCount": sell_signal_count,
        "DataEndExitCount": int(trades["ExitReason"].eq("data_end_mark").sum()),
        "SellSignalExitRate": sell_signal_count / len(trades) if len(trades) else None,
        "AvgGrossReturn": float(trades["GrossReturn"].mean()) if len(trades) else None,
        "AvgNetReturn": average,
        "MedianNetReturn": float(clean.median()) if not clean.empty else None,
        "WinRate": float(clean.gt(0).mean()) if not clean.empty else None,
        "ProfitFactor": float(gains / abs(losses)) if losses < 0 else None,
        "AvgHoldingTradingDays": avg_holding,
        "MedianHoldingTradingDays": float(holding_days.median()) if not holding_days.empty else None,
        "AvgHoldingCalendarDays": float(pd.to_numeric(trades["HoldingCalendarDays"], errors="coerce").mean()) if len(trades) else None,
        "AnnualizedAvgTradeReturn": float(annualized) if annualized is not None else None,
        "AverageActivePositions": float(active_series.mean()) if not active_series.empty else None,
        "MaxActivePositions": int(active_series.max()) if not active_series.empty else None,
        "MarketExposureRate": float(active_series.gt(0).mean()) if not active_series.empty else None,
    }
    return pd.DataFrame([row])


def monthly_dynamic_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    frame = trades.copy()
    frame["ExitMonth"] = pd.to_datetime(frame["ExitDate"], errors="coerce").dt.to_period("M").astype(str)
    grouped = frame.groupby("ExitMonth", dropna=False)
    monthly = grouped.agg(
        TradeCount=("Code", "count"),
        AvgNetReturn=("NetReturn", "mean"),
        MedianNetReturn=("NetReturn", "median"),
        WinRate=("NetReturn", lambda values: float(pd.to_numeric(values, errors="coerce").gt(0).mean())),
        AvgHoldingTradingDays=("HoldingTradingDays", "mean"),
    ).reset_index()
    return monthly.sort_values("ExitMonth", ascending=False)


def dynamic_comparison_table(dynamic_summary: pd.DataFrame) -> str:
    rows = []
    row = dynamic_summary.iloc[0]
    rows.append(
        {
            "StrategyName": "\u52d5\u614b\u51fa\u5834\uff1a\u6295\u4fe1\u5927\u8ce3\u9694\u65e5\u51fa\u5834",
            "TradeCount": row.TradeCount,
            "HoldingRule": f"{fmt_num(row.AvgHoldingTradingDays, 1)}\u65e5\u5e73\u5747",
            "AvgNetReturn": row.AvgNetReturn,
            "WinRate": row.WinRate,
            "ProfitFactor": row.ProfitFactor,
        }
    )
    metrics_path = OUTPUT_ROOT / "strategy_metrics.csv"
    if metrics_path.exists():
        fixed = pd.read_csv(metrics_path, encoding="utf-8-sig")
        fixed = fixed[fixed["ParticipantKey"].eq("trust") & fixed["SignalWindow"].eq(1)].copy()
        fixed = fixed.sort_values("HoldingDays")
        for fixed_row in fixed.itertuples(index=False):
            rows.append(
                {
                    "StrategyName": f"\u56fa\u5b9a\u6301\u6709 {int(fixed_row.HoldingDays)} \u65e5",
                    "TradeCount": fixed_row.TradeCount,
                    "HoldingRule": f"{int(fixed_row.HoldingDays)}\u65e5",
                    "AvgNetReturn": fixed_row.AvgNetReturn,
                    "WinRate": fixed_row.WinRate,
                    "ProfitFactor": fixed_row.ProfitFactor,
                }
            )
    columns = [
        ("StrategyName", "\u7b56\u7565", "text"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("HoldingRule", "\u51fa\u5834/\u6301\u6709", "text"),
        ("AvgNetReturn", "\u5e73\u5747\u6de8\u5831\u916c", "pct"),
        ("WinRate", "\u52dd\u7387", "pct"),
        ("ProfitFactor", "\u7372\u5229\u56e0\u5b50", "num"),
    ]
    frame = pd.DataFrame(rows)
    heads = "".join(f"<th>{label}</th>" for _column, label, _kind in columns)
    return f"<table><thead><tr>{heads}</tr></thead><tbody>{table_rows(frame, columns)}</tbody></table>"


def write_trust_dynamic_exit_report(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    top_n: int,
    min_daily_turnover: float,
) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / "trust_dynamic_exit_strategy_report.html"
    row = summary.iloc[0]
    recent = trades.sort_values("ExitDate", ascending=False).head(100)
    recent = recent.copy()
    recent["ExitReason"] = recent["ExitReason"].map(
        {
            "sell_signal": "\u6295\u4fe1\u5927\u8ce3",
            "data_end_mark": "\u8cc7\u6599\u7d50\u675f",
        }
    ).fillna(recent["ExitReason"])
    recent_columns = [
        ("EntrySignalDate", "\u8cb7\u8a0a\u65e5", "text"),
        ("EntryDate", "\u9032\u5834\u65e5", "text"),
        ("ExitSignalDate", "\u8ce3\u8a0a\u65e5", "text"),
        ("ExitDate", "\u51fa\u5834\u65e5", "text"),
        ("Code", "\u4ee3\u865f", "text"),
        ("Name", "\u540d\u7a31", "text"),
        ("NetReturn", "\u6de8\u5831\u916c", "pct"),
        ("HoldingTradingDays", "\u6301\u6709\u4ea4\u6613\u65e5", "int"),
        ("ExitReason", "\u51fa\u5834\u539f\u56e0", "text"),
    ]
    monthly_columns = [
        ("ExitMonth", "\u51fa\u5834\u6708\u4efd", "text"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("AvgNetReturn", "\u5e73\u5747\u6de8\u5831\u916c", "pct"),
        ("MedianNetReturn", "\u4e2d\u4f4d\u6578", "pct"),
        ("WinRate", "\u52dd\u7387", "pct"),
        ("AvgHoldingTradingDays", "\u5e73\u5747\u6301\u6709\u65e5", "num"),
    ]
    recent_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in recent_columns)
    monthly_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in monthly_columns)
    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>\u6295\u4fe1\u52d5\u614b\u51fa\u5834\u7b56\u7565\u56de\u6e2c</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin: 24px 0 10px; font-size: 18px; }}
p {{ line-height: 1.65; }}
.meta {{ color: #64748b; font-size: 13px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }}
.metric {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 10px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 19px; font-weight: 700; margin-top: 4px; }}
.panel {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 14px; margin: 14px 0; }}
.chart {{ width: 100%; height: auto; display: block; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee9; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: right; font-size: 13px; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
td:nth-child(1), td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6),
th:nth-child(1), th:nth-child(2), th:nth-child(3), th:nth-child(4), th:nth-child(5), th:nth-child(6) {{ text-align: left; }}
.pos {{ color: #047857; font-weight: 700; }}
.neg {{ color: #b91c1c; font-weight: 700; }}
a {{ color: #1d4ed8; text-decoration: none; }}
</style>
</head>
<body>
<main>
<h1>\u6295\u4fe1\u8cb7\u9032\uff0f\u5927\u8ce3\u52d5\u614b\u51fa\u5834\u56de\u6e2c</h1>
<div class="meta">\u8fd1\u4e94\u5e74\uff1b1 \u65e5\u6295\u4fe1\u8cb7\u8d85\u5f37\u5ea6\u9032\u5834\uff1b\u6bcf\u65e5\u524d {top_n} \u6a94\uff1b\u6700\u4f4e\u65e5\u6210\u4ea4\u91d1\u984d {fmt_num(min_daily_turnover, 0)}\uff1b\u6295\u4fe1\u8ce3\u8d85\u5f37\u5ea6\u9032\u5165\u524d {top_n} \u6a94\u5f8c\u9694\u5929\u958b\u76e4\u51fa\u5834</div>
<p>\u672c\u5831\u544a\u6aa2\u67e5\u300c\u8ffd\u8e64\u6295\u4fe1\u5927\u8cb7\uff0c\u9047\u5230\u6295\u4fe1\u5927\u8ce3\u518d\u51fa\u5834\u300d\u662f\u5426\u512a\u65bc\u56fa\u5b9a\u6301\u6709\u5230\u671f\u3002\u540c\u4e00\u6a94\u80a1\u7968\u540c\u6642\u53ea\u6301\u6709\u4e00\u7b46\uff1b\u672a\u7b49\u5230\u8ce3\u8a0a\u865f\u4f46\u8cc7\u6599\u7d50\u675f\u7684\u90e8\u4f4d\uff0c\u4ee5\u6700\u5f8c\u53ef\u7528\u958b\u76e4\u50f9\u6a19\u8a18\u51fa\u5834\u3002</p>
<section class="summary">
<div class="metric"><div class="label">\u4ea4\u6613\u6578</div><div class="value">{int(row.TradeCount):,}</div></div>
<div class="metric"><div class="label">\u5e73\u5747\u6de8\u5831\u916c</div><div class="value">{fmt_pct(row.AvgNetReturn)}</div></div>
<div class="metric"><div class="label">\u52dd\u7387</div><div class="value">{fmt_pct(row.WinRate)}</div></div>
<div class="metric"><div class="label">\u5e73\u5747\u6301\u6709\u4ea4\u6613\u65e5</div><div class="value">{fmt_num(row.AvgHoldingTradingDays, 1)}</div></div>
<div class="metric"><div class="label">\u5927\u8ce3\u8a0a\u865f\u51fa\u5834\u7387</div><div class="value">{fmt_pct(row.SellSignalExitRate)}</div></div>
<div class="metric"><div class="label">\u7372\u5229\u56e0\u5b50</div><div class="value">{fmt_num(row.ProfitFactor, 2)}</div></div>
<div class="metric"><div class="label">\u5e73\u5747\u6d3b\u8e8d\u6301\u80a1\u6578</div><div class="value">{fmt_num(row.AverageActivePositions, 1)}</div></div>
<div class="metric"><div class="label">\u6700\u5927\u6d3b\u8e8d\u6301\u80a1\u6578</div><div class="value">{fmt_num(row.MaxActivePositions, 0)}</div></div>
</section>
<section class="panel">
<h2>\u8207\u56fa\u5b9a\u6301\u6709\u7248\u672c\u5c0d\u7167</h2>
{dynamic_comparison_table(summary)}
</section>
<section class="panel">
<h2>\u55ae\u7b46\u5831\u916c\u5206\u5e03</h2>
{histogram_svg(trades, "NetReturn", "\u55ae\u7b46\u6de8\u5831\u916c\u5206\u5e03")}
</section>
<section class="panel">
<h2>\u6301\u5009\u4ea4\u6613\u65e5\u5206\u5e03</h2>
{histogram_svg(trades, "HoldingTradingDays", "\u6301\u5009\u4ea4\u6613\u65e5\u5206\u5e03", percent=False)}
</section>
<section class="panel">
<h2>\u6708\u7d71\u8a08</h2>
<table><thead><tr>{monthly_heads}</tr></thead><tbody>{table_rows(monthly, monthly_columns)}</tbody></table>
</section>
<section class="panel">
<h2>\u6700\u8fd1\u51fa\u5834\u4ea4\u6613</h2>
<table><thead><tr>{recent_heads}</tr></thead><tbody>{table_rows(recent, recent_columns)}</tbody></table>
</section>
<p><a href="summary.html">\u56de\u5230\u7b56\u7565\u7d71\u6574\u5831\u544a</a></p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def build_trust_dynamic_exit_report(
    panel: pd.DataFrame,
    meta: dict[str, Any],
    *,
    top_n: int,
    min_daily_turnover: float,
) -> dict[str, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    trades = select_trust_dynamic_exit_trades(panel, top_n=top_n, min_daily_turnover=min_daily_turnover)
    if trades.empty:
        raise SystemExit("no_trust_dynamic_exit_trades")
    active_counts = active_position_counts(trades, panel)
    summary = summarize_trust_dynamic_exit(
        trades,
        active_counts,
        meta=meta,
        top_n=top_n,
        min_daily_turnover=min_daily_turnover,
    )
    monthly = monthly_dynamic_summary(trades)
    paths = {
        "trust_dynamic_trades": OUTPUT_ROOT / "trust_dynamic_exit_trades.csv",
        "trust_dynamic_summary": OUTPUT_ROOT / "trust_dynamic_exit_summary.csv",
        "trust_dynamic_monthly": OUTPUT_ROOT / "trust_dynamic_exit_monthly.csv",
        "trust_dynamic_active_positions": OUTPUT_ROOT / "trust_dynamic_exit_active_positions.csv",
    }
    trades.to_csv(paths["trust_dynamic_trades"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["trust_dynamic_summary"], index=False, encoding="utf-8-sig")
    monthly.to_csv(paths["trust_dynamic_monthly"], index=False, encoding="utf-8-sig")
    active_counts.to_csv(paths["trust_dynamic_active_positions"], index=False, encoding="utf-8-sig")
    paths["trust_dynamic_report"] = write_trust_dynamic_exit_report(
        trades,
        summary,
        monthly,
        top_n=top_n,
        min_daily_turnover=min_daily_turnover,
    )
    return paths


def close_cumulative_sell_trade(
    trade: dict[str, Any],
    *,
    trigger_date: pd.Timestamp,
    trigger_row: pd.Series,
    exit_row: pd.Series,
    exit_position: int,
    exit_reason: str,
    stop_loss_rate: float,
) -> dict[str, Any]:
    exit_open = float(exit_row["OpenExec"])
    trigger_close = to_float(trigger_row.get("CloseExec"))
    trigger_close_return = None
    if trigger_close is not None and trade["EntryOpen"] > 0:
        trigger_close_return = trigger_close / float(trade["EntryOpen"]) - 1.0
    gross_return = exit_open / float(trade["EntryOpen"]) - 1.0
    exit_date = pd.Timestamp(exit_row["Date"])
    entry_buy = float(trade["EntryBuyNetShares"])
    entry_buy_value = to_float(trade.get("EntryBuyNetValue"))
    quota = float(trade.get("TrustQuotaShares", entry_buy))
    trigger_net = to_float(trigger_row.get("trust_net"))
    return {
        "EntrySignalStartDate": pd.Timestamp(trade.get("EntrySignalStartDate", trade["EntrySignalDate"])).strftime("%Y-%m-%d"),
        "EntrySignalDate": pd.Timestamp(trade["EntrySignalDate"]).strftime("%Y-%m-%d"),
        "EntrySignalDays": int(trade.get("EntrySignalDays", 1)),
        "EntryDate": pd.Timestamp(trade["EntryDate"]).strftime("%Y-%m-%d"),
        "ExitTriggerDate": pd.Timestamp(trigger_date).strftime("%Y-%m-%d"),
        "ExitDate": exit_date.strftime("%Y-%m-%d"),
        "Code": trade["Code"],
        "Name": trade["Name"],
        "EntryOpen": float(trade["EntryOpen"]),
        "ExitOpen": exit_open,
        "EntryBuyNetShares": entry_buy,
        "EntryBuyNetValue": entry_buy_value,
        "InitialTrustQuotaShares": entry_buy,
        "TriggerTrustQuotaShares": quota,
        "TrustQuotaToInitialBuyRatio": quota / entry_buy if entry_buy > 0 else None,
        "TrustNetFlowAfterSignalShares": quota - entry_buy,
        "TriggerTrustNetShares": trigger_net,
        "TriggerCloseReturn": trigger_close_return,
        "StopLossRate": float(stop_loss_rate),
        "BuyScore": float(trade["BuyScore"]),
        "BuyRank": int(trade["BuyRank"]),
        "GrossReturn": gross_return,
        "NetReturn": gross_return - ROUND_TRIP_COST_RATE,
        "HoldingTradingDays": int(exit_position - trade["EntryPosition"]),
        "HoldingCalendarDays": int((exit_date - pd.Timestamp(trade["EntryDate"])).days),
        "ExitReason": exit_reason,
    }


def select_trust_cumulative_sell_stop_trades(
    panel: pd.DataFrame,
    *,
    top_n: int,
    min_daily_turnover: float,
    stop_loss_rate: float,
    trust_entry_streak_days: int = 1,
    min_trust_entry_value: float = 0.0,
) -> pd.DataFrame:
    if trust_entry_streak_days < 1:
        raise ValueError("trust_entry_streak_days_must_be_positive")
    needed = [
        "Date",
        "Code",
        "Name",
        "OpenExec",
        "CloseExec",
        "Turnover",
        "trust_net",
        "trust_net_value",
    ]
    candidates = panel[needed].copy().rename(columns={"Date": "SignalDate"})
    for column in ["OpenExec", "CloseExec", "Turnover", "trust_net", "trust_net_value"]:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    candidates = candidates.dropna(subset=["SignalDate", "OpenExec", "CloseExec", "Turnover", "trust_net", "trust_net_value"])
    candidates = candidates.sort_values(["Code", "SignalDate"]).reset_index(drop=True)
    by_code = candidates.groupby("Code", group_keys=False)
    candidates["SignalTrustPositiveDays"] = by_code["trust_net"].transform(
        lambda values: values.gt(0).astype(int).rolling(trust_entry_streak_days, min_periods=trust_entry_streak_days).sum()
    )
    candidates["SignalTrustNetShares"] = by_code["trust_net"].transform(
        lambda values: values.rolling(trust_entry_streak_days, min_periods=trust_entry_streak_days).sum()
    )
    candidates["SignalTrustNetValue"] = by_code["trust_net_value"].transform(
        lambda values: values.rolling(trust_entry_streak_days, min_periods=trust_entry_streak_days).sum()
    )
    candidates["SignalTurnover"] = by_code["Turnover"].transform(
        lambda values: values.rolling(trust_entry_streak_days, min_periods=trust_entry_streak_days).sum()
    )
    candidates["SignalMinDailyTurnover"] = by_code["Turnover"].transform(
        lambda values: values.rolling(trust_entry_streak_days, min_periods=trust_entry_streak_days).min()
    )
    candidates["SignalStartDate"] = by_code["SignalDate"].shift(trust_entry_streak_days - 1)
    candidates["Score"] = candidates["SignalTrustNetValue"] / candidates["SignalTurnover"].replace(0, np.nan)
    candidates = candidates.dropna(
        subset=[
            "SignalStartDate",
            "SignalTrustPositiveDays",
            "SignalTrustNetShares",
            "SignalTrustNetValue",
            "SignalTurnover",
            "SignalMinDailyTurnover",
            "Score",
        ]
    )
    candidates = candidates[
        candidates["SignalTrustPositiveDays"].eq(trust_entry_streak_days)
        & candidates["SignalMinDailyTurnover"].ge(min_daily_turnover)
        & candidates["SignalTrustNetValue"].ge(min_trust_entry_value)
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    buy_signals = candidates.sort_values(
        ["SignalDate", "SignalTrustNetValue", "Code"],
        ascending=[True, False, True],
    )
    buy_signals = buy_signals.groupby("SignalDate", group_keys=False).head(top_n).copy()
    buy_signals["BuyRank"] = buy_signals.groupby("SignalDate")["SignalTrustNetValue"].rank(ascending=False, method="first").astype(int)

    stock_frames = {
        str(code): group.sort_values("Date").reset_index(drop=True)
        for code, group in panel.groupby("Code", dropna=False)
    }
    position_by_code_date = {
        code: {pd.Timestamp(row.Date): int(index) for index, row in frame.iterrows()}
        for code, frame in stock_frames.items()
    }
    buy_by_date = {pd.Timestamp(date): group.to_dict("records") for date, group in buy_signals.groupby("SignalDate", sort=True)}
    all_dates = sorted(pd.to_datetime(panel["Date"], errors="coerce").dropna().unique())

    holdings: dict[str, dict[str, Any]] = {}
    closed: list[dict[str, Any]] = []
    for current in all_dates:
        current_date = pd.Timestamp(current)
        closed_codes_today: set[str] = set()
        for code, trade in list(holdings.items()):
            if current_date < pd.Timestamp(trade["EntryDate"]):
                continue
            frame = stock_frames.get(code)
            position = position_by_code_date.get(code, {}).get(current_date)
            if frame is None or position is None:
                continue
            row = frame.iloc[position]
            trust_net = to_float(row.get("trust_net")) or 0.0
            trade["TrustQuotaShares"] = float(trade.get("TrustQuotaShares", trade["EntryBuyNetShares"])) + trust_net

            close_exec = to_float(row.get("CloseExec"))
            close_return = close_exec / float(trade["EntryOpen"]) - 1.0 if close_exec is not None and trade["EntryOpen"] > 0 else None
            quota_hit = float(trade.get("TrustQuotaShares", 0.0)) <= 0
            stop_hit = close_return is not None and close_return <= -stop_loss_rate
            if not quota_hit and not stop_hit:
                continue
            exit_position = position + 1
            if exit_position >= len(frame):
                continue
            if stop_hit and quota_hit:
                exit_reason = "stop_loss_and_quota_depleted"
            elif stop_hit:
                exit_reason = "stop_loss"
            else:
                exit_reason = "quota_depleted"
            closed.append(
                close_cumulative_sell_trade(
                    trade,
                    trigger_date=current_date,
                    trigger_row=row,
                    exit_row=frame.iloc[exit_position],
                    exit_position=exit_position,
                    exit_reason=exit_reason,
                    stop_loss_rate=stop_loss_rate,
                )
            )
            holdings.pop(code)
            closed_codes_today.add(code)

        for buy_signal in buy_by_date.get(current_date, []):
            code = str(buy_signal["Code"])
            if code in holdings or code in closed_codes_today:
                continue
            entry_info = next_stock_row(stock_frames, code, current_date)
            if entry_info is None:
                continue
            entry_position, entry_row = entry_info
            entry_open = to_float(entry_row.get("OpenExec"))
            entry_buy_net = to_float(buy_signal.get("SignalTrustNetShares"))
            entry_buy_value = to_float(buy_signal.get("SignalTrustNetValue"))
            if entry_open is None or entry_open <= 0 or entry_buy_net is None or entry_buy_net <= 0:
                continue
            holdings[code] = {
                "EntrySignalStartDate": pd.Timestamp(buy_signal["SignalStartDate"]),
                "EntrySignalDate": current_date,
                "EntrySignalDays": int(trust_entry_streak_days),
                "EntryDate": pd.Timestamp(entry_row["Date"]),
                "EntryPosition": entry_position,
                "EntryOpen": entry_open,
                "EntryBuyNetShares": entry_buy_net,
                "EntryBuyNetValue": entry_buy_value,
                "TrustQuotaShares": entry_buy_net,
                "Code": code,
                "Name": buy_signal["Name"],
                "BuyScore": float(buy_signal["Score"]),
                "BuyRank": int(buy_signal["BuyRank"]),
            }

    for code, trade in holdings.items():
        frame = stock_frames.get(code)
        if frame is None or frame.empty:
            continue
        exit_position = len(frame) - 1
        if exit_position <= trade["EntryPosition"]:
            continue
        closed.append(
            close_cumulative_sell_trade(
                trade,
                trigger_date=pd.Timestamp(frame.iloc[exit_position]["Date"]),
                trigger_row=frame.iloc[exit_position],
                exit_row=frame.iloc[exit_position],
                exit_position=exit_position,
                exit_reason="data_end_mark",
                stop_loss_rate=stop_loss_rate,
            )
        )

    trades = pd.DataFrame(closed)
    if trades.empty:
        return trades
    return trades.sort_values(["EntryDate", "BuyRank", "Code"]).reset_index(drop=True)


def summarize_trust_cumulative_sell_stop(
    trades: pd.DataFrame,
    active_counts: pd.DataFrame,
    *,
    meta: dict[str, Any],
    top_n: int,
    min_daily_turnover: float,
    stop_loss_rate: float,
    trust_entry_streak_days: int = 1,
    min_trust_entry_value: float = 0.0,
) -> pd.DataFrame:
    clean = pd.to_numeric(trades["NetReturn"], errors="coerce").dropna()
    clean = clean[np.isfinite(clean)]
    gains = clean[clean.gt(0)].sum()
    losses = clean[clean.lt(0)].sum()
    holding_days = pd.to_numeric(trades["HoldingTradingDays"], errors="coerce").dropna()
    active_series = pd.to_numeric(active_counts["ActivePositions"], errors="coerce") if not active_counts.empty else pd.Series(dtype=float)
    stop_mask = trades["ExitReason"].isin(["stop_loss", "stop_loss_and_quota_depleted", "stop_loss_and_cumulative_sell"])
    quota_mask = trades["ExitReason"].isin(["quota_depleted", "stop_loss_and_quota_depleted", "cumulative_sell", "stop_loss_and_cumulative_sell"])
    average = float(clean.mean()) if not clean.empty else None
    avg_holding = float(holding_days.mean()) if not holding_days.empty else None
    annualized = None
    if average is not None and avg_holding and average > -1:
        annualized = (1 + average) ** (TRADING_DAYS_PER_YEAR / avg_holding) - 1
    return pd.DataFrame(
        [
            {
                "Strategy": "trust_cumulative_sell_stop",
                "Participant": "\u6295\u4fe1",
                "SignalWindow": int(trust_entry_streak_days),
                "TrustEntryStreakDays": int(trust_entry_streak_days),
                "EntryTopN": int(top_n),
                "MinDailyTurnover": float(min_daily_turnover),
                "MinTrustEntryValue": float(min_trust_entry_value),
                "StopLossRate": float(stop_loss_rate),
                "StartDate": meta.get("start_date"),
                "LatestDate": meta.get("latest_date"),
                "StockCount": meta.get("loaded_stock_count"),
                "TradeCount": int(len(trades)),
                "QuotaDepletedExitCount": int(quota_mask.sum()),
                "CumulativeSellExitCount": int(quota_mask.sum()),
                "StopLossExitCount": int(stop_mask.sum()),
                "DataEndExitCount": int(trades["ExitReason"].eq("data_end_mark").sum()),
                "QuotaDepletedExitRate": float(quota_mask.mean()) if len(trades) else None,
                "CumulativeSellExitRate": float(quota_mask.mean()) if len(trades) else None,
                "StopLossExitRate": float(stop_mask.mean()) if len(trades) else None,
                "DataEndExitRate": float(trades["ExitReason"].eq("data_end_mark").mean()) if len(trades) else None,
                "AvgGrossReturn": float(trades["GrossReturn"].mean()) if len(trades) else None,
                "AvgNetReturn": average,
                "MedianNetReturn": float(clean.median()) if not clean.empty else None,
                "WinRate": float(clean.gt(0).mean()) if not clean.empty else None,
                "ProfitFactor": float(gains / abs(losses)) if losses < 0 else None,
                "AvgHoldingTradingDays": avg_holding,
                "MedianHoldingTradingDays": float(holding_days.median()) if not holding_days.empty else None,
                "AvgHoldingCalendarDays": float(pd.to_numeric(trades["HoldingCalendarDays"], errors="coerce").mean()) if len(trades) else None,
                "AnnualizedAvgTradeReturn": float(annualized) if annualized is not None else None,
                "AverageActivePositions": float(active_series.mean()) if not active_series.empty else None,
                "MaxActivePositions": int(active_series.max()) if not active_series.empty else None,
                "MarketExposureRate": float(active_series.gt(0).mean()) if not active_series.empty else None,
                "AvgTrustQuotaToInitialBuyRatio": float(pd.to_numeric(trades["TrustQuotaToInitialBuyRatio"], errors="coerce").mean()) if len(trades) else None,
                "AvgTrustNetFlowAfterSignalShares": float(pd.to_numeric(trades["TrustNetFlowAfterSignalShares"], errors="coerce").mean()) if len(trades) else None,
            }
        ]
    )


def trust_cumulative_comparison_table(summary: pd.DataFrame) -> str:
    rows = []
    row = summary.iloc[0]
    stop_loss_label = fmt_pct(row.StopLossRate, 0)
    entry_days = int(getattr(row, "TrustEntryStreakDays", getattr(row, "SignalWindow", 1)))
    min_entry_value = to_float(getattr(row, "MinTrustEntryValue", 0.0)) or 0.0
    entry_label = f"\u9023\u8cb7{entry_days}\u65e5"
    if min_entry_value > 0:
        entry_label += f"\u3001\u8cb7\u8d85\u91d1\u984d\u2265{fmt_num(min_entry_value, 0)}"
    rows.append(
        {
            "StrategyName": f"{entry_label}\uff0bquota \u6b78\u96f6\uff0b{stop_loss_label}\u505c\u640d",
            "TradeCount": row.TradeCount,
            "ExitRule": "\u521d\u59cb quota \u70ba\u8a0a\u865f\u5340\u9593\u5408\u8a08\u8cb7\u8d85\u80a1\u6578\uff0c\u5f8c\u7e8c\u6bcf\u65e5\u96a8\u6de8\u8cb7\u8ce3\u589e\u6e1b\uff0cquota \u5c0f\u65bc\u7b49\u65bc 0",
            "StopLossRate": row.StopLossRate,
            "AvgNetReturn": row.AvgNetReturn,
            "MedianNetReturn": row.MedianNetReturn,
            "WinRate": row.WinRate,
            "ProfitFactor": row.ProfitFactor,
            "AvgHoldingTradingDays": row.AvgHoldingTradingDays,
        }
    )
    dynamic_path = OUTPUT_ROOT / "trust_dynamic_exit_summary.csv"
    if dynamic_path.exists():
        dynamic = pd.read_csv(dynamic_path, encoding="utf-8-sig").iloc[0]
        rows.append(
            {
                "StrategyName": "\u524d\u7248\uff1a\u6295\u4fe1\u5927\u8ce3\u6392\u540d\u51fa\u5834",
                "TradeCount": dynamic.TradeCount,
                "ExitRule": "\u6295\u4fe1\u5927\u8ce3\u524dN\u6a94",
                "StopLossRate": None,
                "AvgNetReturn": dynamic.AvgNetReturn,
                "MedianNetReturn": dynamic.MedianNetReturn,
                "WinRate": dynamic.WinRate,
                "ProfitFactor": dynamic.ProfitFactor,
                "AvgHoldingTradingDays": dynamic.AvgHoldingTradingDays,
            }
        )
    metrics_path = OUTPUT_ROOT / "strategy_metrics.csv"
    if metrics_path.exists():
        fixed = pd.read_csv(metrics_path, encoding="utf-8-sig")
        fixed = fixed[fixed["ParticipantKey"].eq("trust") & fixed["SignalWindow"].eq(1)].sort_values("HoldingDays")
        for fixed_row in fixed.itertuples(index=False):
            rows.append(
                {
                    "StrategyName": f"\u56fa\u5b9a\u6301\u6709 {int(fixed_row.HoldingDays)} \u65e5",
                    "TradeCount": fixed_row.TradeCount,
                    "ExitRule": f"{int(fixed_row.HoldingDays)}\u65e5",
                    "StopLossRate": None,
                    "AvgNetReturn": fixed_row.AvgNetReturn,
                    "MedianNetReturn": fixed_row.MedianNetReturn,
                    "WinRate": fixed_row.WinRate,
                    "ProfitFactor": fixed_row.ProfitFactor,
                    "AvgHoldingTradingDays": fixed_row.HoldingDays,
                }
            )
    columns = [
        ("StrategyName", "\u7b56\u7565", "text"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("ExitRule", "\u51fa\u5834\u898f\u5247", "text"),
        ("StopLossRate", "\u505c\u640d", "pct"),
        ("AvgNetReturn", "\u5e73\u5747\u6de8\u5831\u916c", "pct"),
        ("MedianNetReturn", "\u4e2d\u4f4d\u6578", "pct"),
        ("WinRate", "\u52dd\u7387", "pct"),
        ("ProfitFactor", "\u7372\u5229\u56e0\u5b50", "num"),
        ("AvgHoldingTradingDays", "\u5e73\u5747\u6301\u6709\u65e5", "num"),
    ]
    heads = "".join(f"<th>{label}</th>" for _column, label, _kind in columns)
    return f"<table><thead><tr>{heads}</tr></thead><tbody>{table_rows(pd.DataFrame(rows), columns)}</tbody></table>"


def write_trust_cumulative_sell_stop_report(
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    top_n: int,
    min_daily_turnover: float,
    stop_loss_rate: float,
) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / "trust_cumulative_sell_stop_strategy_report.html"
    row = summary.iloc[0]
    entry_days = int(getattr(row, "TrustEntryStreakDays", getattr(row, "SignalWindow", 1)))
    min_entry_value = to_float(getattr(row, "MinTrustEntryValue", 0.0)) or 0.0
    entry_rule_text = f"\u6295\u4fe1\u9023\u7e8c {entry_days} \u5929\u8cb7\u8d85"
    if min_entry_value > 0:
        entry_rule_text += f"\uff0c{entry_days}\u5929\u5408\u8a08\u8cb7\u8d85\u91d1\u984d\u81f3\u5c11 {fmt_num(min_entry_value, 0)}"
    reason_labels = {
        "quota_depleted": "\u6de8\u8cb7\u8ce3 quota \u6b78\u96f6",
        "cumulative_sell": "\u6de8\u8cb7\u8ce3 quota \u6b78\u96f6",
        "stop_loss": "\u505c\u640d",
        "stop_loss_and_quota_depleted": "\u505c\u640d\u4e14 quota \u6b78\u96f6",
        "stop_loss_and_cumulative_sell": "\u505c\u640d\u4e14 quota \u6b78\u96f6",
        "data_end_mark": "\u8cc7\u6599\u7d50\u675f",
    }
    reason_summary = (
        trades.groupby("ExitReason", dropna=False)
        .agg(
            TradeCount=("Code", "count"),
            AvgNetReturn=("NetReturn", "mean"),
            AvgHoldingTradingDays=("HoldingTradingDays", "mean"),
        )
        .reset_index()
    )
    reason_summary["ExitReasonName"] = reason_summary["ExitReason"].map(reason_labels).fillna(reason_summary["ExitReason"])
    reason_summary["ExitRate"] = reason_summary["TradeCount"] / max(len(trades), 1)
    reason_summary = reason_summary[["ExitReasonName", "TradeCount", "ExitRate", "AvgNetReturn", "AvgHoldingTradingDays"]]
    reason_columns = [
        ("ExitReasonName", "\u51fa\u5834\u539f\u56e0", "text"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("ExitRate", "\u5360\u6bd4", "pct"),
        ("AvgNetReturn", "\u5e73\u5747\u6de8\u5831\u916c", "pct"),
        ("AvgHoldingTradingDays", "\u5e73\u5747\u6301\u6709\u65e5", "num"),
    ]
    monthly_columns = [
        ("ExitMonth", "\u51fa\u5834\u6708\u4efd", "text"),
        ("TradeCount", "\u4ea4\u6613\u6578", "int"),
        ("AvgNetReturn", "\u5e73\u5747\u6de8\u5831\u916c", "pct"),
        ("MedianNetReturn", "\u4e2d\u4f4d\u6578", "pct"),
        ("WinRate", "\u52dd\u7387", "pct"),
        ("AvgHoldingTradingDays", "\u5e73\u5747\u6301\u6709\u65e5", "num"),
    ]
    recent = trades.sort_values("ExitDate", ascending=False).head(100).copy()
    recent["ExitReason"] = recent["ExitReason"].map(reason_labels).fillna(recent["ExitReason"])
    recent_columns = [
        ("EntrySignalStartDate", "\u8a0a\u865f\u8d77\u65e5", "text"),
        ("EntrySignalDate", "\u8cb7\u8a0a\u65e5", "text"),
        ("EntrySignalDays", "\u9023\u8cb7\u5929\u6578", "int"),
        ("EntryDate", "\u9032\u5834\u65e5", "text"),
        ("ExitTriggerDate", "\u89f8\u767c\u65e5", "text"),
        ("ExitDate", "\u51fa\u5834\u65e5", "text"),
        ("Code", "\u4ee3\u865f", "text"),
        ("Name", "\u540d\u7a31", "text"),
        ("NetReturn", "\u6de8\u5831\u916c", "pct"),
        ("HoldingTradingDays", "\u6301\u6709\u4ea4\u6613\u65e5", "int"),
        ("EntryBuyNetValue", "\u8a0a\u865f\u8cb7\u8d85\u91d1\u984d", "num"),
        ("InitialTrustQuotaShares", "\u521d\u59cb quota", "int"),
        ("TriggerTrustQuotaShares", "\u89f8\u767c quota", "int"),
        ("TrustQuotaToInitialBuyRatio", "quota/\u521d\u59cb\u8cb7\u8d85", "pct"),
        ("TrustNetFlowAfterSignalShares", "\u8a0a\u865f\u5f8c\u6de8\u8cb7\u8ce3", "int"),
        ("TriggerCloseReturn", "\u89f8\u767c\u6536\u76e4\u5831\u916c", "pct"),
        ("ExitReason", "\u51fa\u5834\u539f\u56e0", "text"),
    ]
    reason_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in reason_columns)
    monthly_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in monthly_columns)
    recent_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in recent_columns)
    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>\u6295\u4fe1\u6de8\u8cb7\u8ce3 quota \u505c\u640d\u7b56\u7565\u56de\u6e2c</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin: 24px 0 10px; font-size: 18px; }}
p {{ line-height: 1.65; }}
.meta {{ color: #64748b; font-size: 13px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }}
.metric {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 10px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 19px; font-weight: 700; margin-top: 4px; }}
.panel {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 14px; margin: 14px 0; }}
.chart {{ width: 100%; height: auto; display: block; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee9; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: right; font-size: 13px; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
td:nth-child(1), td:nth-child(2), td:nth-child(3), td:nth-child(4), td:nth-child(5), td:nth-child(6),
th:nth-child(1), th:nth-child(2), th:nth-child(3), th:nth-child(4), th:nth-child(5), th:nth-child(6) {{ text-align: left; }}
.pos {{ color: #047857; font-weight: 700; }}
.neg {{ color: #b91c1c; font-weight: 700; }}
a {{ color: #1d4ed8; text-decoration: none; }}
</style>
</head>
<body>
<main>
<h1>\u6295\u4fe1\u6de8\u8cb7\u8ce3 quota \u52a0\u505c\u640d\u7b56\u7565\u56de\u6e2c</h1>
<div class="meta">\u8fd1\u4e94\u5e74\uff1b{entry_rule_text}\uff1b\u6bcf\u65e5\u6700\u591a\u524d {top_n} \u6a94\uff1b\u6bcf\u500b\u8a0a\u865f\u65e5\u6700\u4f4e\u65e5\u6210\u4ea4\u91d1\u984d {fmt_num(min_daily_turnover, 0)}\uff1b\u6295\u4fe1\u6301\u5009 quota \u5c0f\u65bc\u7b49\u65bc 0 \u9694\u5929\u958b\u76e4\u51fa\u5834\uff1b\u505c\u640d {fmt_pct(stop_loss_rate, 0)}</div>
<p>\u8cb7\u9032\u8a0a\u865f\u9700\u8981\u6295\u4fe1\u9023\u7e8c {entry_days} \u500b\u4ea4\u6613\u65e5\u6de8\u8cb7\u8d85\uff0c\u4e14\u8a72\u5340\u9593\u5408\u8a08\u8cb7\u8d85\u91d1\u984d\u9054\u5230\u9580\u6abb\u3002\u521d\u59cb quota \u4f7f\u7528\u9019 {entry_days} \u5929\u7684\u6295\u4fe1\u6de8\u8cb7\u8d85\u80a1\u6578\u5408\u8a08\u3002\u9032\u5834\u5f8c\u6bcf\u65e5\u7528\u7576\u5929\u6295\u4fe1\u6de8\u8cb7\u8ce3\u80a1\u6578\u8abf\u6574 quota\uff1a\u6de8\u8cb7\u589e\u52a0 quota\uff0c\u6de8\u8ce3\u964d\u4f4e quota\u3002\u82e5 quota \u5c0f\u65bc\u7b49\u65bc 0\uff0c\u9694\u5929\u958b\u76e4\u51fa\u5834\u3002\u82e5\u4efb\u4e00\u5929\u6536\u76e4\u76f8\u5c0d\u9032\u5834\u958b\u76e4\u8dcc\u7834 {fmt_pct(stop_loss_rate, 0)}\uff0c\u4e5f\u5728\u9694\u5929\u958b\u76e4\u51fa\u5834\u3002</p>
<section class="summary">
<div class="metric"><div class="label">\u8cb7\u9032\u8a0a\u865f</div><div class="value">\u9023\u8cb7 {entry_days} \u65e5</div></div>
<div class="metric"><div class="label">\u8cb7\u8d85\u91d1\u984d\u9580\u6abb</div><div class="value">{fmt_num(min_entry_value, 0)}</div></div>
<div class="metric"><div class="label">\u4ea4\u6613\u6578</div><div class="value">{int(row.TradeCount):,}</div></div>
<div class="metric"><div class="label">\u5e73\u5747\u6de8\u5831\u916c</div><div class="value">{fmt_pct(row.AvgNetReturn)}</div></div>
<div class="metric"><div class="label">\u4e2d\u4f4d\u6578\u6de8\u5831\u916c</div><div class="value">{fmt_pct(row.MedianNetReturn)}</div></div>
<div class="metric"><div class="label">\u52dd\u7387</div><div class="value">{fmt_pct(row.WinRate)}</div></div>
<div class="metric"><div class="label">\u505c\u640d\u6bd4\u4f8b</div><div class="value">{fmt_pct(row.StopLossExitRate)}</div></div>
<div class="metric"><div class="label">quota \u6b78\u96f6\u51fa\u5834\u6bd4\u4f8b</div><div class="value">{fmt_pct(row.QuotaDepletedExitRate)}</div></div>
<div class="metric"><div class="label">\u5e73\u5747\u6301\u6709\u4ea4\u6613\u65e5</div><div class="value">{fmt_num(row.AvgHoldingTradingDays, 1)}</div></div>
<div class="metric"><div class="label">\u7372\u5229\u56e0\u5b50</div><div class="value">{fmt_num(row.ProfitFactor, 2)}</div></div>
</section>
<section class="panel">
<h2>\u8207\u5176\u4ed6\u6295\u4fe1\u7248\u672c\u5c0d\u7167</h2>
{trust_cumulative_comparison_table(summary)}
</section>
<section class="panel">
<h2>\u51fa\u5834\u539f\u56e0\u7d71\u8a08</h2>
<table><thead><tr>{reason_heads}</tr></thead><tbody>{table_rows(reason_summary, reason_columns)}</tbody></table>
</section>
<section class="panel">
<h2>\u55ae\u7b46\u5831\u916c\u5206\u5e03</h2>
{histogram_svg(trades, "NetReturn", "\u55ae\u7b46\u6de8\u5831\u916c\u5206\u5e03")}
</section>
<section class="panel">
<h2>\u6301\u5009\u4ea4\u6613\u65e5\u5206\u5e03</h2>
{histogram_svg(trades, "HoldingTradingDays", "\u6301\u5009\u4ea4\u6613\u65e5\u5206\u5e03", percent=False)}
</section>
<section class="panel">
<h2>\u6708\u7d71\u8a08</h2>
<table><thead><tr>{monthly_heads}</tr></thead><tbody>{table_rows(monthly, monthly_columns)}</tbody></table>
</section>
<section class="panel">
<h2>\u6700\u8fd1\u51fa\u5834\u4ea4\u6613</h2>
<table><thead><tr>{recent_heads}</tr></thead><tbody>{table_rows(recent, recent_columns)}</tbody></table>
</section>
<p><a href="summary.html">\u56de\u5230\u7b56\u7565\u7d71\u6574\u5831\u544a</a></p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def build_trust_cumulative_sell_stop_report(
    panel: pd.DataFrame,
    meta: dict[str, Any],
    *,
    top_n: int,
    min_daily_turnover: float,
    stop_loss_rate: float = DEFAULT_STOP_LOSS_RATE,
    trust_entry_streak_days: int = DEFAULT_TRUST_ENTRY_STREAK_DAYS,
    min_trust_entry_value: float = DEFAULT_MIN_TRUST_ENTRY_VALUE,
) -> dict[str, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    trades = select_trust_cumulative_sell_stop_trades(
        panel,
        top_n=top_n,
        min_daily_turnover=min_daily_turnover,
        stop_loss_rate=stop_loss_rate,
        trust_entry_streak_days=trust_entry_streak_days,
        min_trust_entry_value=min_trust_entry_value,
    )
    if trades.empty:
        raise SystemExit("no_trust_cumulative_sell_stop_trades")
    active_counts = active_position_counts(trades, panel)
    summary = summarize_trust_cumulative_sell_stop(
        trades,
        active_counts,
        meta=meta,
        top_n=top_n,
        min_daily_turnover=min_daily_turnover,
        stop_loss_rate=stop_loss_rate,
        trust_entry_streak_days=trust_entry_streak_days,
        min_trust_entry_value=min_trust_entry_value,
    )
    monthly = monthly_dynamic_summary(trades)
    paths = {
        "trust_cumulative_trades": OUTPUT_ROOT / "trust_cumulative_sell_stop_trades.csv",
        "trust_cumulative_summary": OUTPUT_ROOT / "trust_cumulative_sell_stop_summary.csv",
        "trust_cumulative_monthly": OUTPUT_ROOT / "trust_cumulative_sell_stop_monthly.csv",
        "trust_cumulative_active_positions": OUTPUT_ROOT / "trust_cumulative_sell_stop_active_positions.csv",
    }
    trades.to_csv(paths["trust_cumulative_trades"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["trust_cumulative_summary"], index=False, encoding="utf-8-sig")
    monthly.to_csv(paths["trust_cumulative_monthly"], index=False, encoding="utf-8-sig")
    active_counts.to_csv(paths["trust_cumulative_active_positions"], index=False, encoding="utf-8-sig")
    paths["trust_cumulative_report"] = write_trust_cumulative_sell_stop_report(
        trades,
        summary,
        monthly,
        top_n=top_n,
        min_daily_turnover=min_daily_turnover,
        stop_loss_rate=stop_loss_rate,
    )
    return paths


def write_participant_report(participant: ParticipantSpec, summary: pd.DataFrame, trades: pd.DataFrame) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / f"{participant.report_slug}_strategy_report.html"
    participant_summary = summary[summary["ParticipantKey"].eq(participant.key)].copy()
    participant_trades = trades[trades["ParticipantKey"].eq(participant.key)].copy()
    best = participant_summary.sort_values("AvgNetReturn", ascending=False).head(1).iloc[0]
    best_basket = participant_summary.sort_values("BasketAvgNetReturn", ascending=False).head(1).iloc[0]
    table = metrics_table(participant_summary)
    recent = participant_trades.sort_values("SignalDate", ascending=False).head(80)
    recent_columns = [
        ("SignalDate", "\u8a0a\u865f\u65e5", "text"),
        ("Code", "\u4ee3\u865f", "text"),
        ("Name", "\u540d\u7a31", "text"),
        ("SignalWindow", "\u8a0a\u865f", "int"),
        ("HoldingDays", "\u6301\u6709", "int"),
        ("Score", "\u5f37\u5ea6", "pct"),
        ("NetReturn", "\u6de8\u5831\u916c", "pct"),
    ]
    recent_heads = "".join(f"<th>{label}</th>" for _col, label, _kind in recent_columns)
    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(participant.label)}\u6cd5\u4eba\u6d41\u5411\u7b56\u7565\u56de\u6e2c</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin: 24px 0 10px; font-size: 18px; }}
p {{ line-height: 1.65; }}
.meta {{ color: #64748b; font-size: 13px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }}
.metric {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 10px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 19px; font-weight: 700; margin-top: 4px; }}
.panel {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 14px; margin: 14px 0; }}
.chart {{ width: 100%; height: auto; display: block; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee9; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: right; font-size: 13px; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
td:nth-child(1), td:nth-child(2), td:nth-child(3), th:nth-child(1), th:nth-child(2), th:nth-child(3) {{ text-align: left; }}
.pos {{ color: #047857; font-weight: 700; }}
.neg {{ color: #b91c1c; font-weight: 700; }}
a {{ color: #1d4ed8; text-decoration: none; }}
</style>
</head>
<body>
<main>
<h1>{html.escape(participant.label)}\u6cd5\u4eba\u6d41\u5411\u7b56\u7565\u56de\u6e2c</h1>
<div class="meta">\u8fd1\u4e94\u5e74\uff1b\u76e4\u5f8c\u8a0a\u865f\uff0c\u9694\u5929\u958b\u76e4\u9032\u5834\uff1b\u56fa\u5b9a\u6301\u6709\u5230\u671f\u51fa\u5834\uff1b\u6bcf\u65e5\u9078\u524d {DEFAULT_TOP_N} \u6a94</div>
<p>{html.escape(participant.description)}</p>
<p>\u8a0a\u865f\u5f37\u5ea6 = \u8cb7\u8ce3\u8d85\u4f30\u7b97\u91d1\u984d / \u6210\u4ea4\u91d1\u984d\u3002\u7b56\u7565\u6bcf\u500b\u8a0a\u865f\u65e5\u9078\u5f37\u5ea6\u6700\u9ad8\u7684\u80a1\u7968\u7b49\u6b0a\u8cb7\u9032\uff0c\u56de\u6e2c\u5df2\u6263\u8cb7\u9032\u624b\u7e8c\u8cbb\u3001\u8ce3\u51fa\u624b\u7e8c\u8cbb\u8207\u8ce3\u51fa\u4ea4\u6613\u7a05\u3002\u9019\u662f\u4e8b\u4ef6\u578b\u8a0a\u865f\u56de\u6e2c\uff0c\u4e0d\u5c07\u6bcf\u65e5\u91cd\u758a\u8a0a\u865f\u5f37\u5236\u8907\u5229\u6210\u5be6\u76e4\u8cc7\u91d1\u66f2\u7dda\u3002</p>
<section class="summary">
<div class="metric"><div class="label">\u6700\u4f73\u5e73\u5747\u55ae\u7b46</div><div class="value">{int(best.SignalWindow)}\u65e5\u8a0a\u865f / {int(best.HoldingDays)}\u65e5\u6301\u6709 {fmt_pct(best.AvgNetReturn)}</div></div>
<div class="metric"><div class="label">\u6700\u4f73\u8a0a\u865f\u7c43\u5b50</div><div class="value">{int(best_basket.SignalWindow)}\u65e5\u8a0a\u865f / {int(best_basket.HoldingDays)}\u65e5\u6301\u6709 {fmt_pct(best_basket.BasketAvgNetReturn)}</div></div>
<div class="metric"><div class="label">\u4ea4\u6613\u6578</div><div class="value">{int(participant_summary.TradeCount.sum()):,}</div></div>
<div class="metric"><div class="label">\u53c3\u6578\u7d44\u5408</div><div class="value">15</div></div>
</section>
<section class="panel">
<h2>\u5e73\u5747\u55ae\u7b46\u6de8\u5831\u916c</h2>
{bar_chart_svg(participant_summary, "AvgNetReturn", "\u5e73\u5747\u55ae\u7b46\u6de8\u5831\u916c")}
</section>
<section class="panel">
<h2>\u7b56\u7565\u7d44\u5408\u7e3d\u8868</h2>
{table}
</section>
<section class="panel">
<h2>\u6700\u8fd1\u5165\u9078\u4ea4\u6613\u7bc4\u4f8b</h2>
<table><thead><tr>{recent_heads}</tr></thead><tbody>{table_rows(recent, recent_columns)}</tbody></table>
</section>
<p><a href="summary.html">\u56de\u5230\u7b56\u7565\u7d71\u6574\u5831\u544a</a></p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def write_summary_report(summary: pd.DataFrame) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / "summary.html"
    best_by_participant = (
        summary.sort_values("AvgNetReturn", ascending=False)
        .groupby("ParticipantKey", group_keys=False)
        .head(1)
        .copy()
    )
    table = metrics_table(summary.sort_values(["ParticipantKey", "SignalWindow", "HoldingDays"]), include_participant=True)
    best_table = metrics_table(best_by_participant, include_participant=True)
    avg_return_chart = grouped_participant_bar_chart_svg(
        summary,
        "AvgNetReturn",
        "\u540c\u53c3\u6578\u56db\u7fa4\u5e73\u5747\u55ae\u7b46\u6de8\u5831\u916c",
        baseline=0.0,
    )
    win_rate_chart = grouped_participant_bar_chart_svg(
        summary,
        "WinRate",
        "\u540c\u53c3\u6578\u56db\u7fa4\u52dd\u7387",
        baseline=0.5,
    )
    links = "\n".join(
        f'<li><a href="{participant.report_slug}_strategy_report.html">{html.escape(participant.label)}\u7b56\u7565\u5831\u544a</a></li>'
        for participant in PARTICIPANTS
    )
    links += '\n<li><a href="trust_dynamic_exit_strategy_report.html">\u6295\u4fe1\u52d5\u614b\u51fa\u5834\u7b56\u7565\u5831\u544a</a></li>'
    links += '\n<li><a href="trust_cumulative_sell_stop_strategy_report.html">\u6295\u4fe1\u6de8\u8cb7\u8ce3 quota \u505c\u640d\u7b56\u7565\u5831\u544a</a></li>'
    links += '\n<li><a href="trust_top50_cumulative_sell_stop_strategy_report.html">\u6295\u4fe1\u5168\u6b77\u53f2\u5f37\u80a1\u524d50\u6a94 quota \u7b56\u7565\u5831\u544a</a></li>'
    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>\u6cd5\u4eba\u6d41\u5411\u7b56\u7565\u56de\u6e2c\u7d71\u6574</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 22px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin: 24px 0 10px; font-size: 18px; }}
p {{ line-height: 1.65; }}
.panel {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 14px; margin: 14px 0; }}
.chart {{ width: 100%; height: auto; display: block; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee9; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: right; font-size: 13px; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
td:nth-child(1), td:nth-child(2), th:nth-child(1), th:nth-child(2) {{ text-align: left; }}
.pos {{ color: #047857; font-weight: 700; }}
.neg {{ color: #b91c1c; font-weight: 700; }}
a {{ color: #1d4ed8; text-decoration: none; }}
</style>
</head>
<body>
<main>
<h1>\u6cd5\u4eba\u6d41\u5411\u7b56\u7565\u56de\u6e2c\u7d71\u6574</h1>
<p>\u672c\u5831\u544a\u6bd4\u8f03 1/3/5 \u65e5\u8a0a\u865f\u8207 5/10/20/30/60 \u65e5\u56fa\u5b9a\u6301\u6709\u671f\u3002\u6bcf\u65e5\u76e4\u5f8c\u9078\u8a0a\u865f\u5f37\u5ea6\u524d {DEFAULT_TOP_N} \u6a94\uff0c\u9694\u5929\u958b\u76e4\u8cb7\u9032\uff0c\u6301\u6709\u5230\u671f\u958b\u76e4\u8ce3\u51fa\u3002\u672c\u7248\u5148\u505a\u4e8b\u4ef6\u578b\u8a0a\u865f\u56de\u6e2c\uff1a\u91cd\u9ede\u770b\u5e73\u5747\u55ae\u7b46\u5831\u916c\u3001\u52dd\u7387\u8207\u8a0a\u865f\u65e5\u7c43\u5b50\u5e73\u5747\u5831\u916c\uff0c\u4e0d\u628a\u6bcf\u65e5\u91cd\u758a\u90e8\u4f4d\u8907\u5229\u6210\u4e00\u689d\u5be6\u76e4\u8cc7\u91d1\u66f2\u7dda\u3002</p>
<section class="panel">
<h2>\u5206\u5831\u544a</h2>
<ul>{links}</ul>
</section>
<section class="panel">
<h2>\u5404\u7fa4\u6700\u4f73\u5e73\u5747\u55ae\u7b46\u7d44\u5408</h2>
{best_table}
</section>
<section class="panel">
<h2>\u540c\u53c3\u6578\u56db\u7fa4\u7e3e\u6548\u76f4\u65b9\u5716</h2>
<p>\u6bcf\u500b\u6a6b\u8ef8\u6a19\u7c64\u662f\u300c\u8a0a\u865f\u65e5\u6578 / \u6301\u6709\u65e5\u6578\u300d\uff0c\u540c\u4e00\u7d44\u53c3\u6578\u4e0b\u4e26\u6392\u6bd4\u8f03\u5916\u8cc7\u3001\u6295\u4fe1\u3001\u81ea\u71df\u5546\u3001\u53cd\u505a\u5176\u4ed6\u3002\u52dd\u7387\u5716\u7684\u865b\u7dda\u662f 50% \u57fa\u6e96\u3002</p>
{avg_return_chart}
{win_rate_chart}
</section>
<section class="panel">
<h2>\u5168\u53c3\u6578\u7d44\u5408</h2>
{table}
</section>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def build_reports(args: argparse.Namespace) -> dict[str, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    panel, meta = build_signal_panel(lookback_years=args.lookback_years, limit=args.limit)

    all_trades: list[pd.DataFrame] = []
    for participant in PARTICIPANTS:
        for signal_window in SIGNAL_WINDOWS:
            for holding_days in HOLDING_DAYS:
                trades = select_trades(
                    panel,
                    participant=participant,
                    signal_window=signal_window,
                    holding_days=holding_days,
                    top_n=args.top_n,
                    min_daily_turnover=args.min_daily_turnover,
                )
                if not trades.empty:
                    all_trades.append(trades)
    if not all_trades:
        raise SystemExit("no_trades_selected")
    trades = pd.concat(all_trades, ignore_index=True)
    summary, basket_returns = summarize_trades(trades)

    paths = {
        "trades": OUTPUT_ROOT / "selected_trades.csv",
        "summary": OUTPUT_ROOT / "strategy_metrics.csv",
        "basket_returns": OUTPUT_ROOT / "signal_basket_returns.csv",
        "skipped": OUTPUT_ROOT / "skipped_stocks.csv",
        "summary_report": DATA_VIZ_ROOT / "summary.html",
    }
    trades.to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    basket_returns.to_csv(paths["basket_returns"], index=False, encoding="utf-8-sig")
    pd.DataFrame(meta["skipped"]).to_csv(paths["skipped"], index=False, encoding="utf-8-sig")

    report_paths: dict[str, Path] = {}
    for participant in PARTICIPANTS:
        report_paths[participant.key] = write_participant_report(participant, summary, trades)
    report_paths["summary_report"] = write_summary_report(summary)
    report_paths.update(
        build_trust_dynamic_exit_report(
            panel,
            meta,
            top_n=args.top_n,
            min_daily_turnover=args.min_daily_turnover,
        )
    )
    report_paths.update(
        build_trust_cumulative_sell_stop_report(
            panel,
            meta,
            top_n=args.top_n,
            min_daily_turnover=args.min_daily_turnover,
            stop_loss_rate=args.stop_loss_rate,
            trust_entry_streak_days=args.trust_entry_streak_days,
            min_trust_entry_value=args.min_trust_entry_value,
        )
    )
    paths.update(report_paths)
    return paths


def main() -> None:
    args = parse_args()
    paths = build_reports(args)
    print(f"summary_report={paths['summary_report']}")
    print(f"metrics={paths['summary']}")
    for participant in PARTICIPANTS:
        print(f"{participant.key}_report={paths[participant.key]}")
    print(f"trust_dynamic_report={paths['trust_dynamic_report']}")
    print(f"trust_dynamic_summary={paths['trust_dynamic_summary']}")
    print(f"trust_cumulative_report={paths['trust_cumulative_report']}")
    print(f"trust_cumulative_summary={paths['trust_cumulative_summary']}")


if __name__ == "__main__":
    main()
