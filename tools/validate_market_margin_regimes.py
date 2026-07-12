"""Validate all-market price behavior during aggregate margin-financing regimes."""

from __future__ import annotations

import argparse
import html
import json
import math
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

OUTPUT_DIR = PROJECT_ROOT / "output" / "margin_patterns" / "market"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "margin_patterns"
TAIEX_CODE = "TAIEX"
SIGNAL_COLUMN = "MarginBalance20DayChangeRate"


@dataclass
class MarketRegimeConfig:
    price_mode: str
    min_price_stock_count: int
    window: int
    long_horizon: int
    change_top_quantile: float
    change_bottom_quantile: float
    level_high_quantile: float
    level_low_quantile: float
    flat_band: float
    min_interval_days: int
    output_dir: Path
    viz_dir: Path


MARGIN_COLUMNS = [
    "Date",
    "Code",
    "Name",
    "MarginPurchase",
    "MarginSale",
    "MarginCashRepayment",
    "MarginPreviousBalance",
    "MarginCurrentBalance",
    "MarginMarketValue",
    "ShortPurchase",
    "ShortSale",
    "ShortStockRepayment",
    "ShortPreviousBalance",
    "ShortCurrentBalance",
    "Offsetting",
]

SUM_COLUMNS = [
    "MarginPurchase",
    "MarginSale",
    "MarginCashRepayment",
    "MarginPreviousBalance",
    "MarginCurrentBalance",
    "MarginMarketValue",
    "ShortPurchase",
    "ShortSale",
    "ShortStockRepayment",
    "ShortPreviousBalance",
    "ShortCurrentBalance",
    "Offsetting",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate market price behavior during aggregate margin regimes.")
    parser.add_argument("--price-mode", choices=["average", "taiex"], default="average")
    parser.add_argument("--min-price-stock-count", type=int, default=100)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--long-horizon", type=int, default=60)
    parser.add_argument("--change-top-quantile", type=float, default=0.90)
    parser.add_argument("--change-bottom-quantile", type=float, default=0.10)
    parser.add_argument("--level-high-quantile", type=float, default=0.80)
    parser.add_argument("--level-low-quantile", type=float, default=0.20)
    parser.add_argument("--flat-band", type=float, default=0.03)
    parser.add_argument("--min-interval-days", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    return parser.parse_args()


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    number = float(value)
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.{digits}f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.{digits}f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.{digits}f}K"
    return f"{number:.{digits}f}"


def json_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def listed_common_stock_universe() -> pd.DataFrame:
    metadata = read_csv_canonical(PROJECT_ROOT / "data" / "metadata.csv", dtype={"Code": str})
    required = {"Code", "Name", "Type", "Market"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {sorted(missing)}")
    mask = metadata["Type"].eq("股票") & metadata["Market"].eq("上市")
    universe = metadata.loc[mask].copy()
    universe["Code"] = universe["Code"].astype(str)
    return universe


def aggregate_margin(universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed_codes = set(universe["Code"].astype(str))
    coverage_rows: list[dict[str, Any]] = []
    grouped_frames: list[pd.DataFrame] = []
    margin_dir = PROJECT_ROOT / "data" / "margin"
    for path in sorted(margin_dir.glob("*.csv")):
        columns = csv_columns_canonical(path)
        usecols = [column for column in MARGIN_COLUMNS if column in columns]
        if not {"Date", "Code", "MarginCurrentBalance"}.issubset(usecols):
            coverage_rows.append({"Code": path.stem.split("_", 1)[0], "File": str(path), "Loaded": False, "Reason": "missing_required_columns"})
            continue
        df = read_csv_canonical(path, usecols=usecols, dtype={"Code": str})
        df["Code"] = df["Code"].astype(str)
        df = df[df["Code"].isin(allowed_codes)].copy()
        if df.empty:
            coverage_rows.append({"Code": path.stem.split("_", 1)[0], "File": str(path), "Loaded": False, "Reason": "not_in_universe"})
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        if df.empty:
            coverage_rows.append({"Code": path.stem.split("_", 1)[0], "File": str(path), "Loaded": False, "Reason": "no_valid_dates"})
            continue
        for column in SUM_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
            df[column] = pd.to_numeric(df[column], errors="coerce")
        sums = df.groupby("Date", as_index=True)[SUM_COLUMNS].sum(min_count=1)
        sums["StockCount"] = df.groupby("Date")["Code"].nunique()
        sums["MarginMarketValueNonNullCount"] = df.groupby("Date")["MarginMarketValue"].count()
        grouped_frames.append(sums.reset_index())
        coverage_rows.append(
            {
                "Code": df["Code"].iloc[0],
                "Name": df["Name"].iloc[0] if "Name" in df.columns and not df["Name"].empty else "",
                "File": str(path.relative_to(PROJECT_ROOT)),
                "Loaded": True,
                "Reason": "",
                "Rows": int(len(df)),
                "Start": df["Date"].min().strftime("%Y-%m-%d"),
                "End": df["Date"].max().strftime("%Y-%m-%d"),
            }
        )
    if not grouped_frames:
        raise ValueError("No margin CSV files were loaded for the listed common-stock universe.")
    combined = pd.concat(grouped_frames, ignore_index=True)
    aggregate_columns = SUM_COLUMNS + ["StockCount", "MarginMarketValueNonNullCount"]
    market = combined.groupby("Date", as_index=False)[aggregate_columns].sum(min_count=1)
    market.loc[market["MarginMarketValueNonNullCount"].eq(0), "MarginMarketValue"] = np.nan
    market = market.rename(
        columns={
            "MarginPurchase": "TotalMarginPurchase",
            "MarginSale": "TotalMarginSale",
            "MarginCashRepayment": "TotalMarginCashRepayment",
            "MarginPreviousBalance": "TotalMarginPreviousBalance",
            "MarginCurrentBalance": "TotalMarginBalance",
            "MarginMarketValue": "TotalMarginMarketValue",
            "ShortPurchase": "TotalShortPurchase",
            "ShortSale": "TotalShortSale",
            "ShortStockRepayment": "TotalShortStockRepayment",
            "ShortPreviousBalance": "TotalShortPreviousBalance",
            "ShortCurrentBalance": "TotalShortBalance",
            "Offsetting": "TotalOffsetting",
        }
    )
    market["AverageMarginBalancePerStock"] = market["TotalMarginBalance"] / market["StockCount"].replace(0, np.nan)
    market["ShortMarginBalanceRatio"] = market["TotalShortBalance"] / market["TotalMarginBalance"].replace(0, np.nan)
    return market.sort_values("Date").reset_index(drop=True), pd.DataFrame(coverage_rows)


def load_taiex_price() -> tuple[pd.DataFrame, Path]:
    matches = sorted((PROJECT_ROOT / "data" / "price").glob(f"{TAIEX_CODE}_*.csv"))
    if not matches:
        raise FileNotFoundError("TAIEX price CSV not found under data/price.")
    path = matches[0]
    price = read_csv_canonical(path, dtype={"Code": str})
    close_column = "Close" if "Close" in price.columns else "close_adj"
    if close_column not in price.columns:
        raise ValueError(f"{path} has no Close or close_adj column.")
    price = price[["Date", close_column]].copy()
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price[close_column] = pd.to_numeric(price[close_column], errors="coerce")
    price = price.dropna(subset=["Date", close_column])
    price = price[price[close_column].gt(0)]
    price = price.sort_values("Date").drop_duplicates("Date", keep="last")
    return price.rename(columns={close_column: "PriceClose"}), path


def load_equal_weighted_stock_price(universe: pd.DataFrame, config: MarketRegimeConfig) -> tuple[pd.DataFrame, Path]:
    price_dir = PROJECT_ROOT / "data" / "price"
    frames: list[pd.DataFrame] = []
    for code in sorted(universe["Code"].astype(str).unique()):
        matches = sorted(price_dir.glob(f"{code}_*.csv"))
        if not matches:
            continue
        path = matches[0]
        columns = csv_columns_canonical(path)
        close_column = "close_adj" if "close_adj" in columns else "Close"
        if "Date" not in columns or close_column not in columns:
            continue
        df = read_csv_canonical(path, usecols=["Date", close_column])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df[close_column] = pd.to_numeric(df[close_column], errors="coerce")
        df = df.dropna(subset=["Date", close_column])
        df = df[df[close_column].gt(0)]
        df = df.sort_values("Date").drop_duplicates("Date", keep="last")
        df["DailyReturn"] = df[close_column].pct_change()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["DailyReturn"])
        if df.empty:
            continue
        frames.append(df[["Date", "DailyReturn"]])
    if not frames:
        raise ValueError("No stock price files were available to build the all-stock average price index.")
    returns = pd.concat(frames, ignore_index=True)
    average = (
        returns.groupby("Date", as_index=False)
        .agg(AverageStockDailyReturn=("DailyReturn", "mean"), PriceStockCount=("DailyReturn", "count"))
        .sort_values("Date")
    )
    average = average[average["PriceStockCount"].ge(config.min_price_stock_count)].copy()
    if average.empty:
        raise ValueError("All-stock average price index is empty after applying min stock-count filter.")
    average["PriceClose"] = (1 + average["AverageStockDailyReturn"]).cumprod() * 100.0
    return average[["Date", "PriceClose", "AverageStockDailyReturn", "PriceStockCount"]], price_dir


def build_panel(config: MarketRegimeConfig) -> tuple[pd.DataFrame, pd.DataFrame, Path, pd.DataFrame]:
    universe = listed_common_stock_universe()
    margin, coverage = aggregate_margin(universe)
    if config.price_mode == "taiex":
        price, price_path = load_taiex_price()
    else:
        price, price_path = load_equal_weighted_stock_price(universe, config)
    panel = margin.merge(price, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    panel = panel.dropna(subset=["TotalMarginBalance", "PriceClose"])
    panel = panel[panel["TotalMarginBalance"].gt(0)]
    panel[SIGNAL_COLUMN] = panel["TotalMarginBalance"] / panel["TotalMarginBalance"].shift(config.window) - 1
    if "AverageStockDailyReturn" not in panel.columns:
        panel["AverageStockDailyReturn"] = panel["PriceClose"].pct_change()
    panel["DailyReturn"] = panel["AverageStockDailyReturn"].where(
        panel["AverageStockDailyReturn"].notna(),
        panel["PriceClose"].pct_change(),
    )
    panel["PriceReturn20D"] = panel["PriceClose"] / panel["PriceClose"].shift(config.window) - 1
    panel["PriceReturn60D"] = panel["PriceClose"] / panel["PriceClose"].shift(config.long_horizon) - 1
    future_price_returns = pd.concat(
        [panel["PriceClose"].shift(-offset) / panel["PriceClose"] - 1 for offset in range(1, config.window + 1)],
        axis=1,
    )
    future_daily_returns = pd.concat(
        [panel["DailyReturn"].shift(-offset) for offset in range(1, config.window + 1)],
        axis=1,
    )
    panel["FutureAvgReturn20DFromClose"] = future_price_returns.mean(axis=1)
    panel["FutureEndReturn20DFromClose"] = panel["PriceClose"].shift(-config.window) / panel["PriceClose"] - 1
    panel["FutureEndReturn60DFromClose"] = panel["PriceClose"].shift(-config.long_horizon) / panel["PriceClose"] - 1
    panel["FutureMaxReturn20DFromClose"] = future_price_returns.max(axis=1)
    panel["FutureMinReturn20DFromClose"] = future_price_returns.min(axis=1)
    panel["FutureVolatility20D"] = future_daily_returns.std(axis=1) * math.sqrt(252)
    panel["FutureAvgAbsDailyReturn20D"] = future_daily_returns.abs().mean(axis=1)
    return panel, coverage, price_path, universe


def thresholds(panel: pd.DataFrame, config: MarketRegimeConfig) -> dict[str, float]:
    clean_signal = panel[SIGNAL_COLUMN].replace([np.inf, -np.inf], np.nan).dropna()
    clean_level = panel["TotalMarginBalance"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "change_top": float(clean_signal.quantile(config.change_top_quantile)),
        "change_bottom": float(clean_signal.quantile(config.change_bottom_quantile)),
        "level_high": float(clean_level.quantile(config.level_high_quantile)),
        "level_low": float(clean_level.quantile(config.level_low_quantile)),
    }


def regime_masks(panel: pd.DataFrame, marks: dict[str, float]) -> dict[str, pd.Series]:
    surge = panel[SIGNAL_COLUMN].ge(marks["change_top"])
    drop = panel[SIGNAL_COLUMN].le(marks["change_bottom"])
    high = panel["TotalMarginBalance"].ge(marks["level_high"])
    low = panel["TotalMarginBalance"].le(marks["level_low"])
    return {
        "全樣本": pd.Series(True, index=panel.index),
        "全市場融資大漲": surge,
        "全市場融資大跌": drop,
        "全市場融資高水位": high,
        "全市場融資低水位": low,
        "融資大漲且高水位": surge & high,
        "融資高水位但未大漲": high & ~surge,
        "融資低水位且非大漲": low & ~surge,
    }


def contiguous_ranges(mask: pd.Series) -> list[tuple[int, int]]:
    values = mask.fillna(False).astype(bool).tolist()
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ranges.append((start, index - 1))
            start = None
    if start is not None:
        ranges.append((start, len(values) - 1))
    return ranges


def max_drawdown(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    running_high = clean.cummax()
    return float((clean / running_high - 1).min())


def classify_return(value: float, flat_band: float) -> str:
    if pd.isna(value):
        return ""
    if value <= -flat_band:
        return "下行"
    if value >= flat_band:
        return "上行"
    return "盤整"


def summarize_daily(panel: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = panel.dropna(subset=["FutureAvgReturn20DFromClose", "FutureEndReturn20DFromClose", "FutureVolatility20D"])
    for name, mask in masks.items():
        data = base[mask.reindex(base.index).fillna(False)]
        if data.empty:
            continue
        rows.append(
            {
                "狀態": name,
                "日數": int(len(data)),
                "日數占比": float(len(data) / len(base)),
                "後20日平均價格報酬": float(data["FutureAvgReturn20DFromClose"].mean()),
                "後20日終點報酬": float(data["FutureEndReturn20DFromClose"].mean()),
                "後60日終點報酬": float(data["FutureEndReturn60DFromClose"].mean()),
                "後20日最大報酬": float(data["FutureMaxReturn20DFromClose"].mean()),
                "後20日最小報酬": float(data["FutureMinReturn20DFromClose"].mean()),
                "後20日盤整率": float(data["FutureAvgReturn20DFromClose"].abs().le(0.02).mean()),
                "後20日不突破率": float(data["FutureMaxReturn20DFromClose"].le(0.03).mean()),
                "後20日正報酬率": float(data["FutureEndReturn20DFromClose"].gt(0).mean()),
                "後20日年化波動": float(data["FutureVolatility20D"].mean()),
                "後20日平均絕對日報酬": float(data["FutureAvgAbsDailyReturn20D"].mean()),
                "當下前20日報酬": float(data["PriceReturn20D"].mean()),
                "當下前60日報酬": float(data["PriceReturn60D"].mean()),
                "前20日下跌占比": float(data["PriceReturn20D"].le(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def interval_records(panel: pd.DataFrame, masks: dict[str, pd.Series], config: MarketRegimeConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    interval_names = ["全市場融資大漲", "全市場融資大跌", "全市場融資高水位", "全市場融資低水位", "融資大漲且高水位"]
    for name in interval_names:
        for ordinal, (start, end) in enumerate(contiguous_ranges(masks[name]), start=1):
            block = panel.iloc[start : end + 1].copy()
            if block.empty:
                continue
            start_close = float(block["PriceClose"].iloc[0])
            end_close = float(block["PriceClose"].iloc[-1])
            interval_return = end_close / start_close - 1 if start_close else float("nan")
            daily_return = block["DailyReturn"].dropna()
            vol = float(daily_return.std() * math.sqrt(252)) if len(daily_return) >= 2 else float("nan")
            rows.append(
                {
                    "狀態": name,
                    "區間序號": ordinal,
                    "開始日": block["Date"].iloc[0].strftime("%Y-%m-%d"),
                    "結束日": block["Date"].iloc[-1].strftime("%Y-%m-%d"),
                    "交易日數": int(len(block)),
                    "開始價格指標": start_close,
                    "結束價格指標": end_close,
                    "區間報酬": interval_return,
                    "區間類型": classify_return(interval_return, config.flat_band),
                    "區間最大回撤": max_drawdown(block["PriceClose"]),
                    "區間年化波動": vol,
                    "區間平均日報酬": float(daily_return.mean()) if len(daily_return) else float("nan"),
                    "開始日後20日平均價格報酬": json_float(panel.at[start, "FutureAvgReturn20DFromClose"]),
                    "開始日後20日終點報酬": json_float(panel.at[start, "FutureEndReturn20DFromClose"]),
                    "結束日後20日終點報酬": json_float(panel.at[end, "FutureEndReturn20DFromClose"]),
                    "結束日後60日終點報酬": json_float(panel.at[end, "FutureEndReturn60DFromClose"]),
                    "平均融資20日變化率": float(block[SIGNAL_COLUMN].mean()),
                    "平均融資總餘額": float(block["TotalMarginBalance"].mean()),
                    "平均券資比": float(block["ShortMarginBalanceRatio"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_intervals(intervals: pd.DataFrame, *, min_days: int) -> pd.DataFrame:
    if intervals.empty:
        return pd.DataFrame()
    data = intervals[intervals["交易日數"].ge(min_days)].copy()
    if data.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for name, group in data.groupby("狀態", sort=False):
        weights = group["交易日數"].astype(float)
        rows.append(
            {
                "狀態": name,
                "最小區間日數": min_days,
                "區間數": int(len(group)),
                "總交易日數": int(group["交易日數"].sum()),
                "平均區間日數": float(group["交易日數"].mean()),
                "區間平均報酬": float(group["區間報酬"].mean()),
                "交易日加權區間報酬": float(np.average(group["區間報酬"], weights=weights)),
                "下行區間率": float(group["區間類型"].eq("下行").mean()),
                "盤整區間率": float(group["區間類型"].eq("盤整").mean()),
                "上行區間率": float(group["區間類型"].eq("上行").mean()),
                "下行或盤整區間率": float(group["區間類型"].isin(["下行", "盤整"]).mean()),
                "平均最大回撤": float(group["區間最大回撤"].mean()),
                "平均區間波動": float(group["區間年化波動"].mean()),
                "結束日後20日報酬": float(group["結束日後20日終點報酬"].mean()),
                "結束日後60日報酬": float(group["結束日後60日終點報酬"].mean()),
            }
        )
    return pd.DataFrame(rows)


def pct_columns() -> set[str]:
    return {
        "日數占比",
        "後20日平均價格報酬",
        "後20日終點報酬",
        "後60日終點報酬",
        "後20日最大報酬",
        "後20日最小報酬",
        "後20日盤整率",
        "後20日不突破率",
        "後20日正報酬率",
        "後20日年化波動",
        "後20日平均絕對日報酬",
        "當下前20日報酬",
        "當下前60日報酬",
        "前20日下跌占比",
        "區間平均報酬",
        "交易日加權區間報酬",
        "下行區間率",
        "盤整區間率",
        "上行區間率",
        "下行或盤整區間率",
        "平均最大回撤",
        "平均區間波動",
        "結束日後20日報酬",
        "結束日後60日報酬",
    }


def table_html(df: pd.DataFrame, *, max_rows: int = 20) -> str:
    if df.empty:
        return "<p class=\"muted\">沒有符合條件的資料。</p>"
    percent_columns = pct_columns()
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in df.columns)
    body = []
    for _, row in df.head(max_rows).iterrows():
        cells = []
        for column, value in row.items():
            if column in percent_columns:
                text = fmt_pct(value)
            elif isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                text = fmt_num(value, 2)
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def bar_svg(df: pd.DataFrame, value_column: str, *, title: str, positive_good: bool = True) -> str:
    if df.empty or value_column not in df.columns:
        return ""
    labels = df["狀態"].astype(str).tolist()
    values = pd.to_numeric(df[value_column], errors="coerce").fillna(0).tolist()
    width = 920
    row_h = 34
    height = 64 + row_h * len(values)
    left = 190
    right = 48
    center = left + (width - left - right) / 2
    max_abs = max([abs(v) for v in values] + [0.001])
    scale = (width - left - right) / 2 / max_abs
    parts = [f"<svg viewBox=\"0 0 {width} {height}\" class=\"chart\" role=\"img\">"]
    parts.append(f"<text x=\"10\" y=\"24\" class=\"chart-title\">{html.escape(title)}</text>")
    parts.append(f"<line x1=\"{center:.1f}\" y1=\"40\" x2=\"{center:.1f}\" y2=\"{height - 12}\" class=\"axis\"/>")
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 52 + index * row_h
        bar_width = abs(value) * scale
        x = center if value >= 0 else center - bar_width
        color_class = "good" if (value >= 0) == positive_good else "bad"
        parts.append(f"<text x=\"10\" y=\"{y + 13}\" class=\"label\">{html.escape(label)}</text>")
        parts.append(f"<rect x=\"{x:.1f}\" y=\"{y}\" width=\"{bar_width:.1f}\" height=\"18\" class=\"bar {color_class}\"/>")
        text_x = x + bar_width + 6 if value >= 0 else x - 58
        parts.append(f"<text x=\"{text_x:.1f}\" y=\"{y + 14}\" class=\"value\">{fmt_pct(value)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def stacked_interval_svg(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    width = 920
    row_h = 36
    height = 64 + row_h * len(df)
    left = 190
    chart_w = 640
    colors = {"下行區間率": "#c2410c", "盤整區間率": "#64748b", "上行區間率": "#047857"}
    parts = [f"<svg viewBox=\"0 0 {width} {height}\" class=\"chart\" role=\"img\">"]
    parts.append("<text x=\"10\" y=\"24\" class=\"chart-title\">區間類型分布</text>")
    for index, (_, row) in enumerate(df.iterrows()):
        y = 52 + index * row_h
        parts.append(f"<text x=\"10\" y=\"{y + 14}\" class=\"label\">{html.escape(str(row['狀態']))}</text>")
        x = left
        for column, color in colors.items():
            value = float(row.get(column, 0) or 0)
            w = chart_w * value
            parts.append(f"<rect x=\"{x:.1f}\" y=\"{y}\" width=\"{w:.1f}\" height=\"20\" fill=\"{color}\"/>")
            if w > 48:
                parts.append(f"<text x=\"{x + w / 2:.1f}\" y=\"{y + 15}\" class=\"stack-text\">{fmt_pct(value, 0)}</text>")
            x += w
    parts.append("</svg>")
    return "".join(parts)


def line_svg(panel: pd.DataFrame) -> str:
    data = panel.dropna(subset=["PriceClose", "TotalMarginBalance"]).copy()
    if data.empty:
        return ""
    if len(data) > 900:
        data = data.iloc[np.linspace(0, len(data) - 1, 900).round().astype(int)].copy()
    width = 920
    height = 280
    left = 54
    right = 32
    top = 42
    bottom = 34
    chart_w = width - left - right
    chart_h = height - top - bottom

    def points(series: pd.Series) -> str:
        values = pd.to_numeric(series, errors="coerce")
        min_v = float(values.min())
        max_v = float(values.max())
        span = max(max_v - min_v, 1e-9)
        pts = []
        for index, value in enumerate(values):
            x = left + chart_w * index / max(len(values) - 1, 1)
            y = top + chart_h * (1 - (float(value) - min_v) / span)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    start = data["Date"].iloc[0].strftime("%Y-%m-%d")
    end = data["Date"].iloc[-1].strftime("%Y-%m-%d")
    return (
        f"<svg viewBox=\"0 0 {width} {height}\" class=\"chart\" role=\"img\">"
        "<text x=\"10\" y=\"24\" class=\"chart-title\">價格指標與全市場融資餘額，標準化視覺比較</text>"
        f"<text x=\"{left}\" y=\"{height - 10}\" class=\"muted-svg\">{start}</text>"
        f"<text x=\"{width - 120}\" y=\"{height - 10}\" class=\"muted-svg\">{end}</text>"
        f"<rect x=\"{left}\" y=\"{top}\" width=\"{chart_w}\" height=\"{chart_h}\" class=\"plot-bg\"/>"
        f"<polyline points=\"{points(data['PriceClose'])}\" class=\"line price\"/>"
        f"<polyline points=\"{points(data['TotalMarginBalance'])}\" class=\"line margin\"/>"
        "<text x=\"70\" y=\"56\" class=\"legend price-text\">價格指標</text>"
        "<text x=\"140\" y=\"56\" class=\"legend margin-text\">全市場融資餘額</text>"
        "</svg>"
    )


def write_report(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    interval_summary: pd.DataFrame,
    intervals: pd.DataFrame,
    marks: dict[str, float],
    config: MarketRegimeConfig,
    coverage: pd.DataFrame,
    price_path: Path,
    universe: pd.DataFrame,
) -> Path:
    report_path = config.viz_dir / "market_margin_regime_validation.html"
    loaded = coverage[coverage["Loaded"].eq(True)]
    start = panel["Date"].min().strftime("%Y-%m-%d")
    end = panel["Date"].max().strftime("%Y-%m-%d")
    if config.price_mode == "taiex":
        price_label = f"TAIEX：{price_path.name}"
        method_note = "價格端使用加權指數收盤價。"
    else:
        price_label = f"上市普通股等權平均前復權報酬指數，最低每日股票數 {config.min_price_stock_count}"
        method_note = "價格端使用所有上市普通股前復權日報酬的每日平均，累積成等權平均指數。這比目前本地 TAIEX CSV 覆蓋更完整，也更貼近所有股票平均。"
    daily_focus = daily[daily["狀態"].isin(["全樣本", "全市場融資大漲", "全市場融資高水位", "全市場融資低水位", "融資大漲且高水位"])]
    interval_focus = interval_summary[
        interval_summary["狀態"].isin(["全市場融資大漲", "全市場融資高水位", "全市場融資低水位", "融資大漲且高水位"])
    ]
    worst = intervals.sort_values("區間報酬").head(8)
    best_low = intervals[intervals["狀態"].eq("全市場融資低水位")].sort_values("區間報酬", ascending=False).head(8)
    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>全市場融資狀態與價格特性驗證</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans TC", Arial, sans-serif; color: #172033; background: #f7f9fc; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px 48px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
h2 {{ margin: 28px 0 12px; font-size: 21px; }}
p {{ line-height: 1.65; }}
.muted {{ color: #617086; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 18px 0 22px; }}
.card {{ background: white; border: 1px solid #d9e2ef; border-radius: 8px; padding: 14px 16px; }}
.card .label {{ color: #617086; font-size: 13px; }}
.card .value {{ display: block; margin-top: 7px; font-size: 22px; font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; background: white; border: 1px solid #d9e2ef; margin: 10px 0 20px; }}
th, td {{ border-bottom: 1px solid #e6edf5; padding: 8px 9px; text-align: right; white-space: nowrap; font-size: 13px; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eef4fb; color: #1f2a3d; }}
.chart {{ width: 100%; height: auto; background: white; border: 1px solid #d9e2ef; border-radius: 8px; margin: 10px 0 16px; }}
.chart-title {{ font-size: 16px; font-weight: 700; fill: #1f2a3d; }}
.axis {{ stroke: #94a3b8; stroke-width: 1; }}
.label {{ font-size: 13px; fill: #26364d; }}
.value {{ font-size: 12px; fill: #26364d; }}
.bar.good {{ fill: #047857; }}
.bar.bad {{ fill: #c2410c; }}
.stack-text {{ font-size: 11px; fill: white; text-anchor: middle; }}
.plot-bg {{ fill: #fbfdff; stroke: #e2e8f0; }}
.line {{ fill: none; stroke-width: 2; }}
.line.price {{ stroke: #0f766e; }}
.line.margin {{ stroke: #7c3aed; }}
.legend {{ font-size: 12px; }}
.price-text {{ fill: #0f766e; }}
.margin-text {{ fill: #7c3aed; }}
.muted-svg {{ font-size: 11px; fill: #617086; }}
</style>
</head>
<body>
<main>
<h1>全市場融資狀態與 TAIEX 特性驗證</h1>
<p class="muted">方法：從 data/margin per-stock CSV 加總上市普通股融資融券資料，價格使用 {html.escape(price_label)}。分析期間為 {start} 到 {end}。</p>
<div class="cards">
<div class="card"><span class="label">metadata 上市普通股</span><span class="value">{len(universe):,}</span></div>
<div class="card"><span class="label">成功載入 margin 檔</span><span class="value">{len(loaded):,}</span></div>
<div class="card"><span class="label">共同交易日數</span><span class="value">{len(panel):,}</span></div>
<div class="card"><span class="label">融資大漲門檻</span><span class="value">{fmt_pct(marks["change_top"])}</span></div>
<div class="card"><span class="label">融資高水位門檻</span><span class="value">{fmt_num(marks["level_high"], 0)}</span></div>
<div class="card"><span class="label">融資低水位門檻</span><span class="value">{fmt_num(marks["level_low"], 0)}</span></div>
</div>
{line_svg(panel)}
<h2>重點解讀</h2>
<p>這份市場版不是看單一股票，而是把所有上市普通股的融資餘額加總，再觀察市場價格指標在該狀態下的表現。{html.escape(method_note)} 這回答的是「整個市場槓桿水位高低」與大盤/全股票平均價格的關係，不是個股橫斷面選股訊號。</p>
<p>判讀時我會優先看連續區間，因為你的直覺是在看一段融資狀態內，價格是否走不動、下行，或波動變大；單日 forward return 則用來檢查該狀態後續 20/60 日的平均結果。</p>
<h2>每日狀態統計</h2>
{bar_svg(daily_focus, "後20日平均價格報酬", title="各狀態後20日平均價格報酬")}
{bar_svg(daily_focus, "後20日年化波動", title="各狀態後20日年化波動", positive_good=False)}
{table_html(daily_focus)}
<h2>連續區間統計，至少 {config.min_interval_days} 個交易日</h2>
{stacked_interval_svg(interval_focus)}
{bar_svg(interval_focus, "區間平均報酬", title="連續區間平均報酬")}
{table_html(interval_focus)}
<h2>最大下跌區間</h2>
{table_html(worst)}
<h2>融資低水位中的強勢區間</h2>
{table_html(best_low)}
</main>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def write_outputs(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    interval_summary_all: pd.DataFrame,
    interval_summary_min: pd.DataFrame,
    intervals: pd.DataFrame,
    marks: dict[str, float],
    config: MarketRegimeConfig,
    coverage: pd.DataFrame,
    price_path: Path,
    universe: pd.DataFrame,
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.viz_dir.mkdir(parents=True, exist_ok=True)
    panel_out = panel.copy()
    panel_out["Date"] = panel_out["Date"].dt.strftime("%Y-%m-%d")
    panel_out.to_csv(config.output_dir / "market_margin_price_panel.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(config.output_dir / "regime_daily_summary.csv", index=False, encoding="utf-8-sig")
    interval_summary_all.to_csv(config.output_dir / "regime_interval_summary_all.csv", index=False, encoding="utf-8-sig")
    interval_summary_min.to_csv(config.output_dir / "regime_interval_summary_min5.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(config.output_dir / "regime_intervals.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(config.output_dir / "margin_universe_coverage.csv", index=False, encoding="utf-8-sig")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "TWSE listed common stocks from data/metadata.csv",
        "price_mode": config.price_mode,
        "min_price_stock_count": config.min_price_stock_count,
        "price_source": str(price_path.relative_to(PROJECT_ROOT)) if price_path.is_file() else str(price_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(panel)),
        "start": panel["Date"].min().strftime("%Y-%m-%d"),
        "end": panel["Date"].max().strftime("%Y-%m-%d"),
        "universe_count": int(len(universe)),
        "loaded_margin_files": int(coverage["Loaded"].eq(True).sum()),
        "flat_band": config.flat_band,
        "thresholds": marks,
    }
    (config.output_dir / "regime_validation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return write_report(panel, daily, interval_summary_min, intervals, marks, config, coverage, price_path, universe)


def main() -> None:
    args = parse_args()
    config = MarketRegimeConfig(
        price_mode=args.price_mode,
        min_price_stock_count=args.min_price_stock_count,
        window=args.window,
        long_horizon=args.long_horizon,
        change_top_quantile=args.change_top_quantile,
        change_bottom_quantile=args.change_bottom_quantile,
        level_high_quantile=args.level_high_quantile,
        level_low_quantile=args.level_low_quantile,
        flat_band=args.flat_band,
        min_interval_days=args.min_interval_days,
        output_dir=args.output_dir,
        viz_dir=args.viz_dir,
    )
    panel, coverage, price_path, universe = build_panel(config)
    marks = thresholds(panel, config)
    masks = regime_masks(panel, marks)
    daily = summarize_daily(panel, masks)
    intervals = interval_records(panel, masks, config)
    interval_summary_all = summarize_intervals(intervals, min_days=1)
    interval_summary_min = summarize_intervals(intervals, min_days=config.min_interval_days)
    report = write_outputs(
        panel,
        daily,
        interval_summary_all,
        interval_summary_min,
        intervals,
        marks,
        config,
        coverage,
        price_path,
        universe,
    )
    print(
        json.dumps(
            {
                "price_mode": config.price_mode,
                "rows": int(len(panel)),
                "start": panel["Date"].min().strftime("%Y-%m-%d"),
                "end": panel["Date"].max().strftime("%Y-%m-%d"),
                "universe_count": int(len(universe)),
                "loaded_margin_files": int(coverage["Loaded"].eq(True).sum()),
                "report": str(report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
