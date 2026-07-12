"""Explore 2330 price and margin patterns with hypothesis checks."""

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

from column_schema import read_csv_canonical

PRICE_DIR = PROJECT_ROOT / "data" / "price"
MARGIN_DIR = PROJECT_ROOT / "data" / "margin"
OUTPUT_DIR = PROJECT_ROOT / "output" / "margin_patterns" / "2330"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "margin_patterns"

CODE = "2330"
SIGNAL_COLUMN = "MarginBalance20DayChangeRate"


@dataclass
class Config:
    code: str
    window: int
    top_quantile: float
    bottom_quantile: float
    near_high_band: float
    breakout_threshold: float
    plateau_band: float
    output_dir: Path
    viz_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze single-stock price/margin patterns.")
    parser.add_argument("--code", default=CODE)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--top-quantile", type=float, default=0.90)
    parser.add_argument("--bottom-quantile", type=float, default=0.10)
    parser.add_argument("--near-high-band", type=float, default=0.05)
    parser.add_argument("--breakout-threshold", type=float, default=0.03)
    parser.add_argument("--plateau-band", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    return parser.parse_args()


def stock_path(folder: Path, code: str) -> Path:
    matches = sorted(folder.glob(f"{code}_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {code}_*.csv under {folder}")
    return matches[0]


def load_panel(config: Config) -> tuple[pd.DataFrame, Path, Path]:
    price_path = stock_path(PRICE_DIR, config.code)
    margin_path = stock_path(MARGIN_DIR, config.code)

    price_df = read_csv_canonical(price_path, dtype=str).fillna("")
    margin_df = read_csv_canonical(margin_path, dtype=str).fillna("")

    price_columns = ["Date", "Open", "High", "Low", "Close", "Turnover", "close_adj"]
    margin_columns = [
        "Date",
        "MarginCurrentBalance",
        "MarginFinancingUsageRate",
        SIGNAL_COLUMN,
        "MarginMarketValueTo20DayAvgTurnover",
        "ShortCurrentBalance",
        "ShortMarginBalanceRatio",
    ]
    missing_price = [column for column in price_columns if column not in price_df.columns]
    missing_margin = [column for column in margin_columns if column not in margin_df.columns]
    if missing_price or missing_margin:
        raise ValueError(f"missing columns price={missing_price} margin={missing_margin}")

    price_df = price_df[price_columns].copy()
    margin_df = margin_df[margin_columns].copy()
    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
    margin_df["Date"] = pd.to_datetime(margin_df["Date"], errors="coerce")
    for column in price_columns:
        if column != "Date":
            price_df[column] = pd.to_numeric(price_df[column], errors="coerce")
    for column in margin_columns:
        if column != "Date":
            margin_df[column] = pd.to_numeric(margin_df[column], errors="coerce")

    price_df = (
        price_df.dropna(subset=["Date", "close_adj"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    margin_df = (
        margin_df.dropna(subset=["Date", "MarginCurrentBalance"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    panel = margin_df.merge(price_df, on="Date", how="inner").sort_values("Date").reset_index(drop=True)

    window = config.window
    entry = panel["close_adj"].shift(-1)
    future_prices = pd.concat([panel["close_adj"].shift(-offset) for offset in range(1, window + 1)], axis=1)
    future_returns = future_prices.divide(entry, axis=0) - 1
    panel[f"FutureAverageReturn{window}D"] = future_returns.mean(axis=1)
    panel[f"FutureEndReturn{window}D"] = future_returns.iloc[:, -1]
    panel[f"FutureMaxReturn{window}D"] = future_returns.max(axis=1)
    panel[f"FutureMinReturn{window}D"] = future_returns.min(axis=1)
    panel[f"PositiveDayRatio{window}D"] = future_returns.gt(0).mean(axis=1)

    for horizon in [5, 20, 60]:
        panel[f"ForwardReturn{horizon}D"] = panel["close_adj"].shift(-horizon) / entry - 1
    for lookback in [20, 60, 120]:
        panel[f"PriceReturn{lookback}D"] = panel["close_adj"] / panel["close_adj"].shift(lookback) - 1
        panel[f"MarginBalanceChange{lookback}D"] = panel["MarginCurrentBalance"] / panel["MarginCurrentBalance"].shift(lookback) - 1

    panel["Previous120DHigh"] = panel["close_adj"].shift(1).rolling(120, min_periods=60).max()
    panel["DistanceTo120DHigh"] = panel["close_adj"] / panel["Previous120DHigh"] - 1
    panel["FutureBreaks120DHigh20D"] = future_prices.max(axis=1) / panel["Previous120DHigh"] - 1
    panel[f"Plateau{window}D"] = panel[f"FutureAverageReturn{window}D"].abs().le(config.plateau_band)
    panel[f"NoBreakout{window}D"] = panel[f"FutureMaxReturn{window}D"].le(config.breakout_threshold)
    return panel, price_path, margin_path


def metric_stats(df: pd.DataFrame, config: Config) -> dict[str, Any]:
    window = config.window
    metric = f"FutureAverageReturn{window}D"
    data = df.dropna(subset=[metric]).copy()
    return {
        "Rows": int(len(data)),
        "AverageReturn20D": float(data[metric].mean()) if len(data) else None,
        "MedianAverageReturn20D": float(data[metric].median()) if len(data) else None,
        "EndReturn20D": float(data[f"FutureEndReturn{window}D"].mean()) if len(data) else None,
        "MaxReturn20D": float(data[f"FutureMaxReturn{window}D"].mean()) if len(data) else None,
        "MinReturn20D": float(data[f"FutureMinReturn{window}D"].mean()) if len(data) else None,
        "PlateauRate": float(data[f"Plateau{window}D"].mean()) if len(data) else None,
        "NoBreakoutRate": float(data[f"NoBreakout{window}D"].mean()) if len(data) else None,
        "PositiveAverageRate": float(data[metric].gt(0).mean()) if len(data) else None,
        "ForwardReturn60D": float(data["ForwardReturn60D"].mean()) if len(data) else None,
    }


def add_stat_row(rows: list[dict[str, Any]], hypothesis: str, group: str, df: pd.DataFrame, config: Config) -> None:
    row = {"Hypothesis": hypothesis, "Group": group}
    row.update(metric_stats(df, config))
    rows.append(row)


def run_hypotheses(panel: pd.DataFrame, config: Config) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = panel.dropna(subset=[SIGNAL_COLUMN, f"FutureAverageReturn{config.window}D"]).copy()
    m_top = valid[SIGNAL_COLUMN].quantile(config.top_quantile)
    m_bottom = valid[SIGNAL_COLUMN].quantile(config.bottom_quantile)
    m_mid_low = valid[SIGNAL_COLUMN].quantile(0.40)
    m_mid_high = valid[SIGNAL_COLUMN].quantile(0.60)
    pressure_top = valid["MarginMarketValueTo20DayAvgTurnover"].quantile(0.80)
    pressure_bottom = valid["MarginMarketValueTo20DayAvgTurnover"].quantile(0.20)
    short_top = valid["ShortMarginBalanceRatio"].quantile(0.90)

    add_stat_row(rows, "H1 融資20日大增是否壓抑後續表現", "融資大增 top 10%", valid[valid[SIGNAL_COLUMN].ge(m_top)], config)
    add_stat_row(rows, "H1 融資20日大增是否壓抑後續表現", "融資大減 bottom 10%", valid[valid[SIGNAL_COLUMN].le(m_bottom)], config)
    add_stat_row(
        rows,
        "H1 融資20日大增是否壓抑後續表現",
        "融資中性 40-60%",
        valid[valid[SIGNAL_COLUMN].between(m_mid_low, m_mid_high)],
        config,
    )

    add_stat_row(
        rows,
        "H2 融資市值相對成交量過重是否形成壓力",
        "融資壓力高 top 20%",
        valid[valid["MarginMarketValueTo20DayAvgTurnover"].ge(pressure_top)],
        config,
    )
    add_stat_row(
        rows,
        "H2 融資市值相對成交量過重是否形成壓力",
        "融資壓力低 bottom 20%",
        valid[valid["MarginMarketValueTo20DayAvgTurnover"].le(pressure_bottom)],
        config,
    )

    near_high = valid["DistanceTo120DHigh"].ge(-config.near_high_band)
    add_stat_row(
        rows,
        "H3 接近120日高點且融資大增是否更難突破",
        "近前高 + 融資大增",
        valid[near_high & valid[SIGNAL_COLUMN].ge(valid[SIGNAL_COLUMN].quantile(0.80))],
        config,
    )
    add_stat_row(
        rows,
        "H3 接近120日高點且融資大增是否更難突破",
        "所有近前高",
        valid[near_high],
        config,
    )

    pullback = valid["PriceReturn60D"].le(-0.05)
    add_stat_row(
        rows,
        "H4 回檔後融資下降是否更容易反彈",
        "60日跌逾5% + 融資大減",
        valid[pullback & valid[SIGNAL_COLUMN].le(valid[SIGNAL_COLUMN].quantile(0.20))],
        config,
    )
    add_stat_row(rows, "H4 回檔後融資下降是否更容易反彈", "所有60日跌逾5%", valid[pullback], config)

    margin_surge = valid[SIGNAL_COLUMN].ge(valid[SIGNAL_COLUMN].quantile(0.80))
    add_stat_row(
        rows,
        "H5 融資大增但股價未同步上漲是否較弱",
        "融資大增 + 20日股價不漲",
        valid[margin_surge & valid["PriceReturn20D"].le(0)],
        config,
    )
    add_stat_row(
        rows,
        "H5 融資大增但股價未同步上漲是否較弱",
        "融資大增 + 20日股價已漲",
        valid[margin_surge & valid["PriceReturn20D"].gt(0)],
        config,
    )

    add_stat_row(rows, "H6 券資比偏高是否暗示後續壓力", "券資比 top 10%", valid[valid["ShortMarginBalanceRatio"].ge(short_top)], config)
    add_stat_row(rows, "H6 券資比偏高是否暗示後續壓力", "其他期間", valid[valid["ShortMarginBalanceRatio"].lt(short_top)], config)

    result = pd.DataFrame(rows)
    return result


def format_pct(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def format_num(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.{digits}f}"


def svg_line_chart(
    series: list[dict[str, Any]],
    width: int = 980,
    height: int = 320,
    title: str = "",
    y_format: str = "number",
) -> str:
    left, right, top, bottom = 72, 28, 34, 42
    all_points = [point for item in series for point in item["points"] if pd.notna(point[1])]
    if not all_points:
        return "<div>no data</div>"
    dates = [point[0] for point in all_points]
    values = [float(point[1]) for point in all_points]
    min_date, max_date = min(dates), max(dates)
    min_value, max_value = min(values), max(values)
    if math.isclose(min_value, max_value):
        min_value -= 1
        max_value += 1
    pad = (max_value - min_value) * 0.12
    min_value -= pad
    max_value += pad
    span = max((max_date - min_date).days, 1)

    def x_pos(date: pd.Timestamp) -> float:
        return left + ((date - min_date).days / span) * (width - left - right)

    def y_pos(value: float) -> float:
        return top + (max_value - value) / (max_value - min_value) * (height - top - bottom)

    def label(value: float) -> str:
        return format_pct(value, 1) if y_format == "pct" else format_num(value, 1)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">',
        f'<text x="{left}" y="20" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" class="axis"/>',
        f'<text x="8" y="{top+5}" class="tick">{html.escape(label(max_value))}</text>',
        f'<text x="8" y="{height-bottom}" class="tick">{html.escape(label(min_value))}</text>',
        f'<text x="{left}" y="{height-14}" class="tick">{min_date.strftime("%Y-%m-%d")}</text>',
        f'<text x="{width-right-78}" y="{height-14}" class="tick">{max_date.strftime("%Y-%m-%d")}</text>',
    ]
    legend_x = left
    for item in series:
        points = [(point[0], point[1]) for point in item["points"] if pd.notna(point[1])]
        if not points:
            continue
        polyline = " ".join(f"{x_pos(date):.1f},{y_pos(float(value)):.1f}" for date, value in points)
        color = item.get("color", "#3157d5")
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{legend_x}" y="{top+4}" width="10" height="10" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+14}" y="{top+14}" class="legend">{html.escape(item["label"])}</text>')
        legend_x += 140
    parts.append("</svg>")
    return "".join(parts)


def svg_bar_chart(rows: pd.DataFrame, metric: str, width: int = 980, height: int = 320, title: str = "") -> str:
    data = rows.dropna(subset=[metric]).copy()
    if data.empty:
        return "<div>no data</div>"
    left, right, top, bottom = 78, 20, 34, 86
    values = data[metric].astype(float).tolist()
    min_value, max_value = min(0, min(values)), max(0, max(values))
    pad = max(0.01, (max_value - min_value) * 0.18)
    min_value -= pad
    max_value += pad
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_width = max(12, min(48, plot_width / len(data) * 0.58))

    def y_pos(value: float) -> float:
        return top + (max_value - value) / (max_value - min_value) * plot_height

    zero_y = y_pos(0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">',
        f'<text x="{left}" y="20" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" class="axis"/>',
        f'<text x="8" y="{top+5}" class="tick">{format_pct(max_value, 1)}</text>',
        f'<text x="8" y="{height-bottom}" class="tick">{format_pct(min_value, 1)}</text>',
    ]
    step = plot_width / len(data)
    for index, row in enumerate(data.itertuples(index=False)):
        value = float(getattr(row, metric))
        x = left + index * step + step / 2 - bar_width / 2
        y = y_pos(value)
        bar_y = min(y, zero_y)
        bar_h = max(2, abs(zero_y - y))
        color = "#d94b4b" if value >= 0 else "#1b8a5a"
        label = str(getattr(row, "Group", getattr(row, "Hypothesis", "")))
        parts.append(f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width/2:.1f}" y="{bar_y - 5:.1f}" text-anchor="middle" class="bar-label">{format_pct(value)}</text>')
        parts.append(
            f'<text transform="translate({x + bar_width/2:.1f},{height-16}) rotate(-35)" text-anchor="end" class="tick">{html.escape(label[:18])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_scatter(panel: pd.DataFrame, config: Config, width: int = 980, height: int = 360) -> str:
    metric = f"FutureAverageReturn{config.window}D"
    data = panel[[SIGNAL_COLUMN, metric]].dropna().copy()
    if data.empty:
        return "<div>no data</div>"
    x_low, x_high = data[SIGNAL_COLUMN].quantile([0.01, 0.99])
    y_low, y_high = data[metric].quantile([0.01, 0.99])
    data = data[data[SIGNAL_COLUMN].between(x_low, x_high) & data[metric].between(y_low, y_high)]
    left, right, top, bottom = 78, 28, 34, 48
    x_pad = (x_high - x_low) * 0.05
    y_pad = (y_high - y_low) * 0.10
    x_low -= x_pad
    x_high += x_pad
    y_low -= y_pad
    y_high += y_pad

    def x_pos(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * (width - left - right)

    def y_pos(value: float) -> float:
        return top + (y_high - value) / (y_high - y_low) * (height - top - bottom)

    corr = data[SIGNAL_COLUMN].corr(data[metric])
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">',
        f'<text x="{left}" y="20" class="chart-title">融資20日變化率 vs 後20日平均報酬</text>',
        f'<line x1="{left}" y1="{y_pos(0):.1f}" x2="{width-right}" y2="{y_pos(0):.1f}" class="axis"/>',
        f'<line x1="{x_pos(0):.1f}" y1="{top}" x2="{x_pos(0):.1f}" y2="{height-bottom}" class="axis"/>',
        f'<text x="{left}" y="{height-14}" class="tick">融資20日變化率</text>',
        f'<text x="8" y="{top+5}" class="tick">{format_pct(y_high, 1)}</text>',
        f'<text x="8" y="{height-bottom}" class="tick">{format_pct(y_low, 1)}</text>',
        f'<text x="{width-right-150}" y="20" class="legend">corr={corr:.3f}</text>',
    ]
    sample = data.sample(min(1800, len(data)), random_state=2330) if len(data) > 1800 else data
    for row in sample.itertuples(index=False):
        x_value = float(getattr(row, SIGNAL_COLUMN))
        y_value = float(getattr(row, metric))
        color = "#d94b4b" if x_value >= 0 else "#1b8a5a"
        parts.append(f'<circle cx="{x_pos(x_value):.1f}" cy="{y_pos(y_value):.1f}" r="2.1" fill="{color}" opacity="0.28"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_table(df: pd.DataFrame, columns: list[str], pct_columns: set[str] | None = None) -> str:
    pct_columns = pct_columns or set()
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = []
    for record in df[columns].to_dict("records"):
        cells = []
        for column in columns:
            value = record[column]
            text = format_pct(value) if column in pct_columns else format_num(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_report(
    panel: pd.DataFrame,
    hypothesis: pd.DataFrame,
    top_events: pd.DataFrame,
    bottom_events: pd.DataFrame,
    config: Config,
    price_path: Path,
    margin_path: Path,
) -> str:
    window = config.window
    start_date = panel["Date"].min().strftime("%Y-%m-%d")
    end_date = panel["Date"].max().strftime("%Y-%m-%d")
    rows = len(panel)

    norm = panel[["Date", "close_adj", "MarginCurrentBalance"]].dropna().copy()
    norm["PriceIndex"] = norm["close_adj"] / norm["close_adj"].iloc[0] * 100
    norm["MarginIndex"] = norm["MarginCurrentBalance"] / norm["MarginCurrentBalance"].iloc[0] * 100
    long_chart = svg_line_chart(
        [
            {"label": "復權股價 index", "color": "#3157d5", "points": list(zip(norm["Date"], norm["PriceIndex"]))},
            {"label": "融資餘額 index", "color": "#d94b4b", "points": list(zip(norm["Date"], norm["MarginIndex"]))},
        ],
        title="2330 復權股價與融資餘額長期相對走勢",
    )
    margin_change_chart = svg_line_chart(
        [{"label": "融資餘額20日變化率", "color": "#d94b4b", "points": list(zip(panel["Date"], panel[SIGNAL_COLUMN]))}],
        title="融資餘額20日變化率",
        y_format="pct",
    )
    scatter = svg_scatter(panel, config)

    bar_data = hypothesis[hypothesis["Hypothesis"].isin([
        "H1 融資20日大增是否壓抑後續表現",
        "H2 融資市值相對成交量過重是否形成壓力",
        "H3 接近120日高點且融資大增是否更難突破",
    ])].copy()
    bar_data["Group"] = bar_data["Group"]
    bar_chart = svg_bar_chart(bar_data, "AverageReturn20D", title="各假設群組後20日平均報酬")

    pct_columns = {
        "AverageReturn20D",
        "MedianAverageReturn20D",
        "EndReturn20D",
        "MaxReturn20D",
        "MinReturn20D",
        "PlateauRate",
        "NoBreakoutRate",
        "PositiveAverageRate",
        "ForwardReturn60D",
        SIGNAL_COLUMN,
        f"FutureAverageReturn{window}D",
        f"FutureEndReturn{window}D",
        f"FutureMaxReturn{window}D",
        f"FutureMinReturn{window}D",
        "PriceReturn20D",
        "DistanceTo120DHigh",
    }
    hypothesis_columns = [
        "Hypothesis",
        "Group",
        "Rows",
        "AverageReturn20D",
        "EndReturn20D",
        "MaxReturn20D",
        "MinReturn20D",
        "PlateauRate",
        "NoBreakoutRate",
        "ForwardReturn60D",
    ]
    event_columns = [
        "Date",
        SIGNAL_COLUMN,
        "close_adj",
        "MarginCurrentBalance",
        "PriceReturn20D",
        f"FutureAverageReturn{window}D",
        f"FutureEndReturn{window}D",
        f"FutureMaxReturn{window}D",
        f"FutureMinReturn{window}D",
        "DistanceTo120DHigh",
    ]
    hypothesis_table = render_table(hypothesis, hypothesis_columns, pct_columns)
    top_table = render_table(top_events, event_columns, pct_columns)
    bottom_table = render_table(bottom_events, event_columns, pct_columns)

    def stat_value(hyp: str, group: str, column: str) -> float | None:
        data = hypothesis[(hypothesis["Hypothesis"].eq(hyp)) & (hypothesis["Group"].eq(group))]
        if data.empty:
            return None
        return float(data.iloc[0][column])

    h1_top = stat_value("H1 融資20日大增是否壓抑後續表現", "融資大增 top 10%", "AverageReturn20D")
    h1_bottom = stat_value("H1 融資20日大增是否壓抑後續表現", "融資大減 bottom 10%", "AverageReturn20D")
    h3_near_surge = stat_value("H3 接近120日高點且融資大增是否更難突破", "近前高 + 融資大增", "MaxReturn20D")
    h3_near_all = stat_value("H3 接近120日高點且融資大增是否更難突破", "所有近前高", "MaxReturn20D")
    h4_delev = stat_value("H4 回檔後融資下降是否更容易反彈", "60日跌逾5% + 融資大減", "AverageReturn20D")
    h4_all = stat_value("H4 回檔後融資下降是否更容易反彈", "所有60日跌逾5%", "AverageReturn20D")

    findings = [
        f"H1: 單純融資20日大增後20日平均報酬 {format_pct(h1_top)}，融資大減為 {format_pct(h1_bottom)}。如果要證明上方壓力，這個差距本身不強。",
        f"H3: 接近120日高點且融資大增的後20日最高報酬 {format_pct(h3_near_surge)}，所有近前高樣本為 {format_pct(h3_near_all)}。",
        f"H4: 60日回檔逾5%且融資下降後的後20日平均報酬 {format_pct(h4_delev)}，所有回檔樣本為 {format_pct(h4_all)}。",
    ]

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>2330 台積電 融資與股價型態探索</title>
<style>
body {{ margin: 0; background: #f7f8fb; color: #172033; font-family: Arial, "Microsoft JhengHei", sans-serif; }}
header {{ background: #172033; color: white; padding: 24px 32px 18px; }}
h1 {{ margin: 0 0 8px; font-size: 25px; }}
.meta {{ color: #cbd5e1; font-size: 13px; line-height: 1.55; }}
main {{ padding: 24px 32px 42px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.card, section {{ background: white; border: 1px solid #dfe5ef; border-radius: 6px; }}
.card {{ padding: 14px 16px; }}
.label {{ color: #59677c; font-size: 12px; }}
.value {{ margin-top: 5px; font-weight: 700; font-size: 21px; }}
section {{ padding: 18px; margin: 16px 0; }}
h2 {{ margin: 0 0 12px; font-size: 18px; }}
.note, li {{ color: #4b5870; font-size: 14px; line-height: 1.65; }}
svg {{ display: block; max-width: 100%; }}
.axis {{ stroke: #94a3b8; stroke-width: 1; }}
.tick {{ fill: #59677c; font-size: 12px; }}
.legend {{ fill: #334155; font-size: 12px; }}
.chart-title {{ fill: #243044; font-size: 14px; font-weight: 700; }}
.bar-label {{ fill: #334155; font-size: 11px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
th, td {{ border-bottom: 1px solid #e5eaf2; padding: 7px 8px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #f2f5f9; color: #334155; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 960px) {{ .cards, .grid2 {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} }}
</style>
</head>
<body>
<header>
<h1>2330 台積電：融資與股價型態探索</h1>
<div class="meta">資料：{html.escape(str(price_path.relative_to(PROJECT_ROOT)))} + {html.escape(str(margin_path.relative_to(PROJECT_ROOT)))}；共同區間 {start_date} 到 {end_date}；主 metric: 後{window}日平均報酬 = mean(close_adj[t+1:t+{window}]) / close_adj[t+1] - 1。</div>
</header>
<main>
<div class="cards">
  <div class="card"><div class="label">共同樣本列數</div><div class="value">{rows:,}</div></div>
  <div class="card"><div class="label">最新復權收盤</div><div class="value">{format_num(panel['close_adj'].iloc[-1], 0)}</div></div>
  <div class="card"><div class="label">最新融資餘額</div><div class="value">{format_num(panel['MarginCurrentBalance'].iloc[-1], 0)}</div></div>
  <div class="card"><div class="label">最新融資20日變化</div><div class="value">{format_pct(panel[SIGNAL_COLUMN].iloc[-1])}</div></div>
</div>

<section>
<h2>先看長期結構</h2>
{long_chart}
<div class="note">兩條線都重設為 2004 年共同起點 = 100。這張圖用來看融資餘額是否只是跟著長期股價趨勢走，或在特定階段明顯背離。</div>
</section>

<section>
<h2>融資變化率本身</h2>
{margin_change_chart}
</section>

<section>
<h2>核心散點：融資20日變化 vs 後20日平均報酬</h2>
{scatter}
<div class="note">如果「融資大增造成上方壓力」是很強的單因子，這裡應該看到右側點雲明顯往下。實際結果要看 corr 與分組表。</div>
</section>

<section>
<h2>假設驗證摘要</h2>
<ul>{"".join(f"<li>{html.escape(item)}</li>" for item in findings)}</ul>
{bar_chart}
</section>

<section>
<h2>所有假設分組統計</h2>
{hypothesis_table}
</section>

<div class="grid2">
<section>
<h2>歷史融資20日大增事件 top 15</h2>
{top_table}
</section>
<section>
<h2>歷史融資20日大減事件 bottom 15</h2>
{bottom_table}
</section>
</div>
</main>
</body>
</html>
"""


def write_outputs(
    panel: pd.DataFrame,
    hypothesis: pd.DataFrame,
    config: Config,
    price_path: Path,
    margin_path: Path,
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.viz_dir.mkdir(parents=True, exist_ok=True)
    window = config.window
    top_events = (
        panel.dropna(subset=[SIGNAL_COLUMN, f"FutureAverageReturn{window}D"])
        .sort_values(SIGNAL_COLUMN, ascending=False)
        .head(15)
        .copy()
    )
    bottom_events = (
        panel.dropna(subset=[SIGNAL_COLUMN, f"FutureAverageReturn{window}D"])
        .sort_values(SIGNAL_COLUMN, ascending=True)
        .head(15)
        .copy()
    )
    for df in [top_events, bottom_events]:
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    hypothesis.to_csv(config.output_dir / "hypothesis_summary.csv", index=False, encoding="utf-8-sig")
    top_events.to_csv(config.output_dir / "top_margin_surge_events.csv", index=False, encoding="utf-8-sig")
    bottom_events.to_csv(config.output_dir / "bottom_margin_drop_events.csv", index=False, encoding="utf-8-sig")

    panel_export = panel.copy()
    panel_export["Date"] = panel_export["Date"].dt.strftime("%Y-%m-%d")
    panel_export.to_csv(config.output_dir / "panel_2330_margin_price.csv", index=False, encoding="utf-8-sig")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "code": config.code,
        "window": config.window,
        "top_quantile": config.top_quantile,
        "bottom_quantile": config.bottom_quantile,
        "near_high_band": config.near_high_band,
        "breakout_threshold": config.breakout_threshold,
        "plateau_band": config.plateau_band,
        "price_path": str(price_path.relative_to(PROJECT_ROOT)),
        "margin_path": str(margin_path.relative_to(PROJECT_ROOT)),
    }
    (config.output_dir / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = build_report(panel, hypothesis, top_events, bottom_events, config, price_path, margin_path)
    report_path = config.viz_dir / f"{config.code}_margin_pattern_report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    config = Config(
        code=str(args.code),
        window=args.window,
        top_quantile=args.top_quantile,
        bottom_quantile=args.bottom_quantile,
        near_high_band=args.near_high_band,
        breakout_threshold=args.breakout_threshold,
        plateau_band=args.plateau_band,
        output_dir=args.output_dir,
        viz_dir=args.viz_dir,
    )
    panel, price_path, margin_path = load_panel(config)
    hypothesis = run_hypotheses(panel, config)
    report_path = write_outputs(panel, hypothesis, config, price_path, margin_path)
    print(
        json.dumps(
            {
                "code": config.code,
                "rows": int(len(panel)),
                "start": panel["Date"].min().strftime("%Y-%m-%d"),
                "end": panel["Date"].max().strftime("%Y-%m-%d"),
                "hypothesis_rows": int(len(hypothesis)),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
