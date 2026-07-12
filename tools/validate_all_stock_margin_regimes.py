"""Generate per-stock margin-regime validation reports for TWSE listed common stocks."""

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

OUTPUT_DIR = PROJECT_ROOT / "output" / "margin_patterns" / "by_stock"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "margin_patterns" / "by_stock"
SIGNAL_COLUMN = "MarginBalance20DayChangeRate"


@dataclass
class Config:
    window: int
    long_horizon: int
    change_top_quantile: float
    change_bottom_quantile: float
    level_high_quantile: float
    level_low_quantile: float
    flat_band: float
    min_rows: int
    min_interval_days: int
    output_dir: Path
    viz_dir: Path
    codes: set[str] | None
    max_stocks: int | None


MARGIN_COLUMNS = [
    "Date",
    "Code",
    "Name",
    "MarginCurrentBalance",
    "MarginBalance20DayChangeRate",
    "MarginFinancingUsageRate",
    "MarginMarketValue",
    "MarginMarketValueTo20DayAvgTurnover",
    "ShortCurrentBalance",
    "ShortMarginBalanceRatio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-stock margin-regime validation reports.")
    parser.add_argument("--codes", nargs="*", help="Optional stock-code subset.")
    parser.add_argument("--max-stocks", type=int)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--long-horizon", type=int, default=60)
    parser.add_argument("--change-top-quantile", type=float, default=0.90)
    parser.add_argument("--change-bottom-quantile", type=float, default=0.10)
    parser.add_argument("--level-high-quantile", type=float, default=0.80)
    parser.add_argument("--level-low-quantile", type=float, default=0.20)
    parser.add_argument("--flat-band", type=float, default=0.03)
    parser.add_argument("--min-rows", type=int, default=180)
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


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip()
    return cleaned or "unknown"


def stock_universe(config: Config) -> pd.DataFrame:
    metadata = read_csv_canonical(PROJECT_ROOT / "data" / "metadata.csv", dtype={"Code": str})
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


def find_stock_file(folder: Path, code: str) -> Path | None:
    matches = sorted(folder.glob(f"{code}_*.csv"))
    return matches[0] if matches else None


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
    margin = margin.dropna(subset=["Date"]).copy()
    for column in margin.columns:
        if column not in {"Date", "Code", "Name"}:
            margin[column] = pd.to_numeric(margin[column], errors="coerce")
    if SIGNAL_COLUMN not in margin.columns:
        margin[SIGNAL_COLUMN] = np.nan

    price_columns = csv_columns_canonical(price_path)
    close_column = "close_adj" if "close_adj" in price_columns else "Close"
    if "Date" not in price_columns or close_column not in price_columns:
        raise ValueError("price_csv_missing_required_columns")
    price = read_csv_canonical(price_path, usecols=["Date", close_column])
    price["Date"] = pd.to_datetime(price["Date"], errors="coerce")
    price[close_column] = pd.to_numeric(price[close_column], errors="coerce")
    price = price.dropna(subset=["Date", close_column])
    price = price[price[close_column].gt(0)]
    price = price.sort_values("Date").drop_duplicates("Date", keep="last")
    price = price.rename(columns={close_column: "PriceClose"})

    panel = margin.merge(price, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    panel = panel.dropna(subset=["MarginCurrentBalance", "PriceClose"])
    panel = panel[panel["MarginCurrentBalance"].gt(0)].copy()
    if len(panel) < config.min_rows:
        raise ValueError("insufficient_joined_rows")
    computed_signal = panel["MarginCurrentBalance"] / panel["MarginCurrentBalance"].shift(config.window) - 1
    panel[SIGNAL_COLUMN] = panel[SIGNAL_COLUMN].where(panel[SIGNAL_COLUMN].notna(), computed_signal)
    panel["DailyReturn"] = panel["PriceClose"].pct_change()
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
    meta = {
        "margin_path": str(margin_path.relative_to(PROJECT_ROOT)),
        "price_path": str(price_path.relative_to(PROJECT_ROOT)),
        "name": str(panel["Name"].dropna().iloc[-1]) if panel["Name"].notna().any() else metadata_name,
    }
    return panel, meta


def thresholds(panel: pd.DataFrame, config: Config) -> dict[str, float]:
    signal = panel[SIGNAL_COLUMN].replace([np.inf, -np.inf], np.nan).dropna()
    level = panel["MarginCurrentBalance"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(signal) < config.min_rows // 2:
        raise ValueError("insufficient_valid_signal_rows")
    return {
        "change_top": float(signal.quantile(config.change_top_quantile)),
        "change_bottom": float(signal.quantile(config.change_bottom_quantile)),
        "level_high": float(level.quantile(config.level_high_quantile)),
        "level_low": float(level.quantile(config.level_low_quantile)),
    }


def regime_masks(panel: pd.DataFrame, marks: dict[str, float]) -> dict[str, pd.Series]:
    surge = panel[SIGNAL_COLUMN].ge(marks["change_top"])
    drop = panel[SIGNAL_COLUMN].le(marks["change_bottom"])
    high = panel["MarginCurrentBalance"].ge(marks["level_high"])
    low = panel["MarginCurrentBalance"].le(marks["level_low"])
    return {
        "全樣本": pd.Series(True, index=panel.index),
        "融資大漲": surge,
        "融資大跌": drop,
        "融資高水位": high,
        "融資低水位": low,
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
    if base.empty:
        return pd.DataFrame()
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


def interval_records(panel: pd.DataFrame, masks: dict[str, pd.Series], config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in ["融資大漲", "融資大跌", "融資高水位", "融資低水位", "融資大漲且高水位"]:
        for ordinal, (start, end) in enumerate(contiguous_ranges(masks[name]), start=1):
            block = panel.iloc[start : end + 1].copy()
            if block.empty:
                continue
            start_close = float(block["PriceClose"].iloc[0])
            end_close = float(block["PriceClose"].iloc[-1])
            interval_return = end_close / start_close - 1 if start_close else float("nan")
            daily_return = block["DailyReturn"].dropna()
            vol = float(daily_return.std() * math.sqrt(252)) if len(daily_return) >= 2 else float("nan")
            start_row = panel.iloc[start]
            end_row = panel.iloc[end]
            rows.append(
                {
                    "狀態": name,
                    "區間序號": ordinal,
                    "開始日": block["Date"].iloc[0].strftime("%Y-%m-%d"),
                    "結束日": block["Date"].iloc[-1].strftime("%Y-%m-%d"),
                    "交易日數": int(len(block)),
                    "開始復權收盤": start_close,
                    "結束復權收盤": end_close,
                    "區間報酬": interval_return,
                    "區間類型": classify_return(interval_return, config.flat_band),
                    "區間最大回撤": max_drawdown(block["PriceClose"]),
                    "區間年化波動": vol,
                    "區間平均日報酬": float(daily_return.mean()) if len(daily_return) else float("nan"),
                    "開始日後20日平均價格報酬": json_float(start_row["FutureAvgReturn20DFromClose"]),
                    "開始日後20日終點報酬": json_float(start_row["FutureEndReturn20DFromClose"]),
                    "結束日後20日終點報酬": json_float(end_row["FutureEndReturn20DFromClose"]),
                    "結束日後60日終點報酬": json_float(end_row["FutureEndReturn60DFromClose"]),
                    "平均融資20日變化率": float(block[SIGNAL_COLUMN].mean()),
                    "平均融資餘額": float(block["MarginCurrentBalance"].mean()),
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
    width = 900
    row_h = 34
    height = 62 + row_h * len(values)
    left = 170
    right = 50
    center = left + (width - left - right) / 2
    max_abs = max([abs(v) for v in values] + [0.001])
    scale = (width - left - right) / 2 / max_abs
    parts = [f"<svg viewBox=\"0 0 {width} {height}\" class=\"chart\" role=\"img\">"]
    parts.append(f"<text x=\"10\" y=\"24\" class=\"chart-title\">{html.escape(title)}</text>")
    parts.append(f"<line x1=\"{center:.1f}\" y1=\"40\" x2=\"{center:.1f}\" y2=\"{height - 12}\" class=\"axis\"/>")
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 50 + index * row_h
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
    width = 900
    row_h = 36
    height = 62 + row_h * len(df)
    left = 170
    chart_w = 630
    colors = {"下行區間率": "#c2410c", "盤整區間率": "#64748b", "上行區間率": "#047857"}
    parts = [f"<svg viewBox=\"0 0 {width} {height}\" class=\"chart\" role=\"img\">"]
    parts.append("<text x=\"10\" y=\"24\" class=\"chart-title\">區間類型分布</text>")
    for index, (_, row) in enumerate(df.iterrows()):
        y = 50 + index * row_h
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


def css() -> str:
    return """
body { margin: 0; font-family: "Microsoft JhengHei", "Noto Sans TC", Arial, sans-serif; color: #172033; background: #f7f9fc; }
main { max-width: 1160px; margin: 0 auto; padding: 28px 22px 48px; }
h1 { margin: 0 0 8px; font-size: 30px; }
h2 { margin: 28px 0 12px; font-size: 21px; }
p { line-height: 1.65; }
a { color: #0f766e; text-decoration: none; }
.muted { color: #617086; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 18px 0 22px; }
.card { background: white; border: 1px solid #d9e2ef; border-radius: 8px; padding: 14px 16px; }
.card .label { color: #617086; font-size: 13px; }
.card .value { display: block; margin-top: 7px; font-size: 21px; font-weight: 700; }
table { border-collapse: collapse; width: 100%; background: white; border: 1px solid #d9e2ef; margin: 10px 0 20px; }
th, td { border-bottom: 1px solid #e6edf5; padding: 8px 9px; text-align: right; white-space: nowrap; font-size: 13px; }
th:first-child, td:first-child { text-align: left; }
th { background: #eef4fb; color: #1f2a3d; }
.chart { width: 100%; height: auto; background: white; border: 1px solid #d9e2ef; border-radius: 8px; margin: 10px 0 16px; }
.chart-title { font-size: 16px; font-weight: 700; fill: #1f2a3d; }
.axis { stroke: #94a3b8; stroke-width: 1; }
.label { font-size: 13px; fill: #26364d; }
.value { font-size: 12px; fill: #26364d; }
.bar.good { fill: #047857; }
.bar.bad { fill: #c2410c; }
.stack-text { font-size: 11px; fill: white; text-anchor: middle; }
"""


def focus_rows(df: pd.DataFrame) -> pd.DataFrame:
    focus = ["全樣本", "融資大漲", "融資高水位", "融資低水位", "融資大漲且高水位"]
    return df[df["狀態"].isin(focus)].copy()


def write_stock_report(
    code: str,
    name: str,
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    interval_summary: pd.DataFrame,
    intervals: pd.DataFrame,
    marks: dict[str, float],
    meta: dict[str, Any],
    config: Config,
) -> str:
    file_name = f"{code}_{safe_filename(name)}.html"
    report_path = config.viz_dir / file_name
    start = panel["Date"].min().strftime("%Y-%m-%d")
    end = panel["Date"].max().strftime("%Y-%m-%d")
    daily_focus = focus_rows(daily)
    interval_focus = interval_summary[
        interval_summary["狀態"].isin(["融資大漲", "融資高水位", "融資低水位", "融資大漲且高水位"])
    ].copy()
    worst = intervals.sort_values("區間報酬").head(8)
    best_low = intervals[intervals["狀態"].eq("融資低水位")].sort_values("區間報酬", ascending=False).head(8)
    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>{html.escape(code)} {html.escape(name)} 融資狀態驗證</title>
<style>{css()}</style>
</head>
<body>
<main>
<p><a href="index.html">回總覽</a></p>
<h1>{html.escape(code)} {html.escape(name)} 融資狀態驗證</h1>
<p class="muted">來源：{html.escape(meta["price_path"])} × {html.escape(meta["margin_path"])}。使用前復權收盤價計算報酬，分析期間 {start} 到 {end}。</p>
<div class="cards">
<div class="card"><span class="label">共同交易日數</span><span class="value">{len(panel):,}</span></div>
<div class="card"><span class="label">融資大漲門檻</span><span class="value">{fmt_pct(marks["change_top"])}</span></div>
<div class="card"><span class="label">融資大跌門檻</span><span class="value">{fmt_pct(marks["change_bottom"])}</span></div>
<div class="card"><span class="label">融資高水位門檻</span><span class="value">{fmt_num(marks["level_high"], 0)}</span></div>
<div class="card"><span class="label">融資低水位門檻</span><span class="value">{fmt_num(marks["level_low"], 0)}</span></div>
</div>
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
    return file_name


def row_by_state(df: pd.DataFrame, state: str) -> dict[str, Any]:
    if df.empty or "狀態" not in df.columns:
        return {}
    rows = df[df["狀態"].eq(state)]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def summary_record(
    code: str,
    name: str,
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    interval_summary: pd.DataFrame,
    marks: dict[str, float],
    report_file: str,
) -> dict[str, Any]:
    all_row = row_by_state(daily, "全樣本")
    surge = row_by_state(daily, "融資大漲")
    high = row_by_state(daily, "融資高水位")
    low = row_by_state(daily, "融資低水位")
    surge_high = row_by_state(daily, "融資大漲且高水位")
    high_i = row_by_state(interval_summary, "融資高水位")
    low_i = row_by_state(interval_summary, "融資低水位")
    surge_high_i = row_by_state(interval_summary, "融資大漲且高水位")
    return {
        "Code": code,
        "Name": name,
        "Rows": int(len(panel)),
        "Start": panel["Date"].min().strftime("%Y-%m-%d"),
        "End": panel["Date"].max().strftime("%Y-%m-%d"),
        "Report": report_file,
        "MarginSurgeThreshold": marks["change_top"],
        "MarginDropThreshold": marks["change_bottom"],
        "MarginHighLevelThreshold": marks["level_high"],
        "MarginLowLevelThreshold": marks["level_low"],
        "AllFutureAvgReturn20D": all_row.get("後20日平均價格報酬"),
        "SurgeFutureAvgReturn20D": surge.get("後20日平均價格報酬"),
        "HighFutureAvgReturn20D": high.get("後20日平均價格報酬"),
        "LowFutureAvgReturn20D": low.get("後20日平均價格報酬"),
        "SurgeHighFutureAvgReturn20D": surge_high.get("後20日平均價格報酬"),
        "HighFutureVolatility20D": high.get("後20日年化波動"),
        "LowFutureVolatility20D": low.get("後20日年化波動"),
        "HighIntervalAvgReturn": high_i.get("區間平均報酬"),
        "HighIntervalDownOrFlatRate": high_i.get("下行或盤整區間率"),
        "LowIntervalAvgReturn": low_i.get("區間平均報酬"),
        "LowIntervalUpRate": low_i.get("上行區間率"),
        "SurgeHighIntervalAvgReturn": surge_high_i.get("區間平均報酬"),
        "SurgeHighIntervalDownOrFlatRate": surge_high_i.get("下行或盤整區間率"),
    }


def write_index(summary: pd.DataFrame, skipped: pd.DataFrame, config: Config) -> None:
    rows = []
    display = summary.sort_values("HighFutureAvgReturn20D", na_position="last").copy()
    for _, row in display.iterrows():
        rows.append(
            {
                "Code": row["Code"],
                "Name": row["Name"],
                "報告": f'<a href="{html.escape(str(row["Report"]))}">open</a>',
                "融資高水位後20日平均": fmt_pct(row["HighFutureAvgReturn20D"]),
                "融資低水位後20日平均": fmt_pct(row["LowFutureAvgReturn20D"]),
                "大漲且高水位後20日平均": fmt_pct(row["SurgeHighFutureAvgReturn20D"]),
                "高水位區間報酬": fmt_pct(row["HighIntervalAvgReturn"]),
                "低水位區間報酬": fmt_pct(row["LowIntervalAvgReturn"]),
            }
        )
    header = "".join(f"<th>{html.escape(column)}</th>" for column in rows[0].keys()) if rows else ""
    body = []
    for row in rows:
        cells = []
        for column, value in row.items():
            if column == "報告":
                cells.append(f"<td>{value}</td>")
            else:
                cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    generated = datetime.now().isoformat(timespec="seconds")
    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>全股票融資狀態驗證總覽</title>
<style>{css()}</style>
</head>
<body>
<main>
<h1>全股票融資狀態驗證總覽</h1>
<p class="muted">產生時間 {generated}。每檔股票使用自己的融資 20 日變化率 top/bottom 10% 與融資餘額 top/bottom 20% 門檻。</p>
<div class="cards">
<div class="card"><span class="label">成功產生報告</span><span class="value">{len(summary):,}</span></div>
<div class="card"><span class="label">略過股票</span><span class="value">{len(skipped):,}</span></div>
<div class="card"><span class="label">高水位後20日平均，中位數</span><span class="value">{fmt_pct(summary["HighFutureAvgReturn20D"].median())}</span></div>
<div class="card"><span class="label">低水位後20日平均，中位數</span><span class="value">{fmt_pct(summary["LowFutureAvgReturn20D"].median())}</span></div>
</div>
<h2>股票清單</h2>
<table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>
</main>
</body>
</html>
"""
    (config.viz_dir / "index.html").write_text(html_text, encoding="utf-8")


def process_stock(row: pd.Series, config: Config) -> tuple[dict[str, Any] | None, pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    code = str(row["Code"])
    metadata_name = str(row["Name"])
    try:
        panel, meta = load_panel(code, metadata_name, config)
        name = str(meta["name"])
        marks = thresholds(panel, config)
        masks = regime_masks(panel, marks)
        daily = summarize_daily(panel, masks)
        intervals = interval_records(panel, masks, config)
        interval_summary = summarize_intervals(intervals, min_days=config.min_interval_days)
        if daily.empty:
            raise ValueError("empty_daily_summary")
        report_file = write_stock_report(code, name, panel, daily, interval_summary, intervals, marks, meta, config)
        record = summary_record(code, name, panel, daily, interval_summary, marks, report_file)
        daily_out = daily.copy()
        daily_out.insert(0, "Code", code)
        daily_out.insert(1, "Name", name)
        interval_out = interval_summary.copy()
        interval_out.insert(0, "Code", code)
        interval_out.insert(1, "Name", name)
        return record, daily_out, interval_out, None
    except Exception as exc:  # noqa: BLE001 - batch report should keep going per stock.
        return None, pd.DataFrame(), pd.DataFrame(), {"Code": code, "Name": metadata_name, "Reason": str(exc)}


def main() -> None:
    args = parse_args()
    config = Config(
        window=args.window,
        long_horizon=args.long_horizon,
        change_top_quantile=args.change_top_quantile,
        change_bottom_quantile=args.change_bottom_quantile,
        level_high_quantile=args.level_high_quantile,
        level_low_quantile=args.level_low_quantile,
        flat_band=args.flat_band,
        min_rows=args.min_rows,
        min_interval_days=args.min_interval_days,
        output_dir=args.output_dir,
        viz_dir=args.viz_dir,
        codes=set(args.codes) if args.codes else None,
        max_stocks=args.max_stocks,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.viz_dir.mkdir(parents=True, exist_ok=True)
    universe = stock_universe(config)
    summary_records: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    interval_frames: list[pd.DataFrame] = []
    skipped_records: list[dict[str, Any]] = []
    for index, row in universe.iterrows():
        record, daily, interval_summary, skipped = process_stock(row, config)
        if record is not None:
            summary_records.append(record)
            daily_frames.append(daily)
            interval_frames.append(interval_summary)
        if skipped is not None:
            skipped_records.append(skipped)
        if (index + 1) % 100 == 0:
            print(f"processed {index + 1}/{len(universe)} stocks; reports={len(summary_records)} skipped={len(skipped_records)}", flush=True)

    summary = pd.DataFrame(summary_records)
    skipped = pd.DataFrame(skipped_records)
    all_daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    all_intervals = pd.concat(interval_frames, ignore_index=True) if interval_frames else pd.DataFrame()
    summary.to_csv(config.output_dir / "all_stock_regime_summary.csv", index=False, encoding="utf-8-sig")
    all_daily.to_csv(config.output_dir / "all_stock_regime_daily_summary.csv", index=False, encoding="utf-8-sig")
    all_intervals.to_csv(config.output_dir / "all_stock_regime_interval_summary_min5.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(config.output_dir / "all_stock_skipped.csv", index=False, encoding="utf-8-sig")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "universe_count": int(len(universe)),
        "reports": int(len(summary)),
        "skipped": int(len(skipped)),
        "output_dir": str(config.output_dir.relative_to(PROJECT_ROOT)),
        "viz_dir": str(config.viz_dir.relative_to(PROJECT_ROOT)),
        "window": config.window,
        "long_horizon": config.long_horizon,
        "thresholds": {
            "change_top_quantile": config.change_top_quantile,
            "change_bottom_quantile": config.change_bottom_quantile,
            "level_high_quantile": config.level_high_quantile,
            "level_low_quantile": config.level_low_quantile,
        },
    }
    (config.output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not summary.empty:
        write_index(summary, skipped, config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
