"""Generate per-CSV visualizations mirroring selected data/ folders.

The renderer writes self-contained HTML files under data_viz/<dataset>/,
preserving the same relative folder structure as data/. Price charts reuse
the existing stock_viz.py K-plot builder when Plotly is available; other
datasets use inline SVG charts with only pandas required.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical

DATA_ROOT = PROJECT_ROOT / "data"
DATA_VIZ_ROOT = PROJECT_ROOT / "data_viz"
DEFAULT_DATASETS = [
    "day_trading",
    "dividend",
    "yield_pe_pb",
    "institutional",
    "margin",
    "price",
    "shareholding",
]
COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#7c3aed",
    "#f59e0b",
    "#0f766e",
    "#64748b",
    "#be185d",
]


@dataclass
class VizResult:
    source_path: Path
    output_path: Path
    status: str
    note: str = ""


def safe_title_from_path(csv_path: Path) -> str:
    return csv_path.stem.replace("_", " ")


def parse_source_date(value: Any) -> pd.Timestamp:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return pd.NaT
    digits = re.sub(r"\D", "", text)
    if len(digits) == 8:
        year = int(digits[:4])
        if year >= 1900:
            return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    if len(digits) == 7:
        year = int(digits[:3]) + 1911
        return pd.to_datetime(f"{year}{digits[3:]}", format="%Y%m%d", errors="coerce")
    if len(digits) == 3:
        return pd.Timestamp(year=int(digits) + 1911, month=1, day=1)
    if len(digits) == 4:
        year = int(digits)
        if year >= 1900:
            return pd.Timestamp(year=year, month=1, day=1)
    return pd.to_datetime(text, errors="coerce")


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def normalize_date_column(df: pd.DataFrame, candidates: list[str]) -> tuple[pd.DataFrame, str]:
    for column in candidates:
        if column in df.columns:
            working = df.copy()
            working["_viz_date"] = working[column].map(parse_source_date)
            working = working.dropna(subset=["_viz_date"]).sort_values("_viz_date")
            return working, "_viz_date"
    raise ValueError(f"No date column found from candidates: {candidates}")


def stock_label(df: pd.DataFrame, csv_path: Path) -> str:
    code = csv_path.stem.split("_", 1)[0]
    for column in ["Name", "stock_name", "公司簡稱", "證券名稱"]:
        if column in df.columns:
            value = df[column].dropna().astype(str)
            if not value.empty and value.iloc[0]:
                return f"{code} {value.iloc[0]}"
    return safe_title_from_path(csv_path)


def downsample_xy(x_values: list[Any], y_values: list[float], max_points: int = 1200) -> tuple[list[Any], list[float]]:
    if len(x_values) <= max_points:
        return x_values, y_values
    step = max(1, math.ceil(len(x_values) / max_points))
    return x_values[::step], y_values[::step]


def finite_values(series_list: list[dict[str, Any]]) -> list[float]:
    values = []
    for series in series_list:
        values.extend(
            float(value)
            for value in series["y"]
            if value is not None and pd.notna(value) and math.isfinite(float(value))
        )
    return values


def nice_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        pad = abs(high) * 0.05 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def format_viz_number(value: float) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    for suffix, divisor in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs_value >= divisor:
            scaled = abs_value / divisor
            text = f"{scaled:.2f}".rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"
    if abs_value >= 100:
        return f"{value:,.0f}"
    if abs_value >= 1:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if abs_value == 0:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def render_time_panel(panel: dict[str, Any], width: int = 1120, height: int = 260) -> str:
    title = html.escape(panel["title"])
    series_list = panel["series"]
    dates = panel["x"]
    show_point_labels = bool(panel.get("show_point_labels"))
    values = finite_values(series_list)
    y_low, y_high = nice_range(values)
    plot_left, plot_top, plot_right, plot_bottom = 72, 36, width - 24, height - 42
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    n = max(1, len(dates) - 1)

    def x_pos(index: int) -> float:
        return plot_left + (index / n) * plot_width if n else plot_left

    def y_pos(value: float) -> float:
        return plot_bottom - ((value - y_low) / (y_high - y_low)) * plot_height

    elements = [
        f'<svg viewBox="0 0 {width} {height}" class="panel" role="img" aria-label="{title}">',
        f'<text x="{plot_left}" y="22" class="panel-title">{title}</text>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis"/>',
        f'<text x="8" y="{plot_top + 4}" class="tick">{format_viz_number(y_high)}</text>',
        f'<text x="8" y="{plot_bottom}" class="tick">{format_viz_number(y_low)}</text>',
    ]
    if dates:
        elements.append(f'<text x="{plot_left}" y="{height - 14}" class="tick">{html.escape(str(dates[0])[:10])}</text>')
        elements.append(f'<text x="{plot_right - 78}" y="{height - 14}" class="tick">{html.escape(str(dates[-1])[:10])}</text>')

    legend_x = plot_left
    legend_y = 34
    for index, series in enumerate(series_list):
        color = COLORS[index % len(COLORS)]
        label = html.escape(series["label"])
        xs, ys = downsample_xy(dates, series["y"])
        path_parts = []
        for point_index, value in enumerate(ys):
            if value is None or pd.isna(value):
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            x = x_pos(point_index if len(xs) == len(dates) else min(point_index * math.ceil(len(dates) / max(len(xs), 1)), len(dates) - 1))
            y = y_pos(value)
            path_parts.append(("M" if not path_parts else "L") + f"{x:.1f},{y:.1f}")
        if path_parts:
            elements.append(f'<path d="{" ".join(path_parts)}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        if show_point_labels:
            label_offset = -8 - (index % 3) * 12
            for point_index, value in enumerate(series["y"]):
                if value is None or pd.isna(value):
                    continue
                value = float(value)
                if not math.isfinite(value):
                    continue
                x = x_pos(point_index)
                y = y_pos(value)
                value_text = html.escape(format_viz_number(value))
                elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{color}"/>')
                elements.append(
                    f'<text x="{x:.1f}" y="{y + label_offset:.1f}" class="point-label" '
                    f'text-anchor="middle" fill="{color}">{value_text}</text>'
                )
        elements.append(f'<rect x="{legend_x}" y="{legend_y}" width="10" height="10" fill="{color}"/>')
        elements.append(f'<text x="{legend_x + 14}" y="{legend_y + 10}" class="legend">{label}</text>')
        legend_x += min(260, 24 + len(label) * 8)
    elements.append("</svg>")
    return "\n".join(elements)


def render_bar_panel(panel: dict[str, Any], width: int = 1120, height: int = 300) -> str:
    title = html.escape(panel["title"])
    labels = [str(item) for item in panel["x"]]
    values = [float(value) if pd.notna(value) else 0.0 for value in panel["y"]]
    value_labels = panel.get("text") or [f"{value:,.0f}" for value in values]
    y_low, y_high = nice_range(values + [0.0])
    plot_left, plot_top, plot_right, plot_bottom = 82, 44, width - 24, height - 74
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    count = max(1, len(labels))
    bar_width = max(6, plot_width / count * 0.72)

    def y_pos(value: float) -> float:
        return plot_bottom - ((value - y_low) / (y_high - y_low)) * plot_height

    zero_y = y_pos(0.0)
    elements = [
        f'<svg viewBox="0 0 {width} {height}" class="panel" role="img" aria-label="{title}">',
        f'<text x="{plot_left}" y="22" class="panel-title">{title}</text>',
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis"/>',
        f'<text x="8" y="{plot_top + 4}" class="tick">{format_viz_number(y_high)}</text>',
        f'<text x="8" y="{plot_bottom}" class="tick">{format_viz_number(y_low)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        center = plot_left + (index + 0.5) / count * plot_width
        y = y_pos(value)
        rect_y = min(y, zero_y)
        rect_h = abs(zero_y - y)
        elements.append(
            f'<rect x="{center - bar_width / 2:.1f}" y="{rect_y:.1f}" width="{bar_width:.1f}" height="{rect_h:.1f}" fill="#2563eb"/>'
        )
        label_text = html.escape(str(value_labels[index]))
        label_y = rect_y - 6 if value >= 0 else rect_y + rect_h + 14
        if count <= 35:
            elements.append(
                f'<text x="{center:.1f}" y="{label_y:.1f}" class="bar-label" text-anchor="middle">{label_text}</text>'
            )
        if count <= 35:
            elements.append(
                f'<text transform="translate({center:.1f},{height - 18}) rotate(-35)" class="tick">{html.escape(label[:18])}</text>'
            )
    elements.append("</svg>")
    return "\n".join(elements)


def render_pie_panel(panel: dict[str, Any], width: int = 1120, height: int = 360) -> str:
    title = html.escape(panel["title"])
    labels = [str(item) for item in panel["x"]]
    values = [max(0.0, float(value) if pd.notna(value) else 0.0) for value in panel["y"]]
    value_labels = panel.get("text") or [f"{value:.2f}%" for value in values]
    callouts = panel.get("callouts") or labels
    total = sum(values)
    cx, cy, radius = 520, 205, 118

    def point(angle: float) -> tuple[float, float]:
        return cx + radius * math.cos(angle), cy + radius * math.sin(angle)

    def text_lines(value: Any) -> list[str]:
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return str(value).split("\n")

    elements = [
        f'<svg viewBox="0 0 {width} {height}" class="panel" role="img" aria-label="{title}">',
        f'<text x="82" y="28" class="panel-title">{title}</text>',
    ]
    note = panel.get("note")
    if note:
        elements.append(f'<text x="82" y="52" class="legend">{html.escape(str(note))}</text>')

    if total <= 0:
        elements.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="#e2e8f0"/>')
        elements.append(f'<text x="{cx}" y="{cy}" class="bar-label" text-anchor="middle">0%</text>')
    else:
        angle = -math.pi / 2
        callout_items = []
        for index, (label, value, label_text, callout) in enumerate(zip(labels, values, value_labels, callouts)):
            color = COLORS[index % len(COLORS)]
            fraction = value / total
            next_angle = angle + fraction * math.tau
            if math.isclose(fraction, 1.0):
                elements.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}"/>')
                mid_angle = angle + math.pi
            else:
                x1, y1 = point(angle)
                x2, y2 = point(next_angle)
                large_arc = 1 if fraction > 0.5 else 0
                elements.append(
                    f'<path d="M {cx:.1f},{cy:.1f} L {x1:.1f},{y1:.1f} '
                    f'A {radius},{radius} 0 {large_arc} 1 {x2:.1f},{y2:.1f} Z" fill="{color}"/>'
                )
                mid_angle = angle + (next_angle - angle) / 2
            line_start_x, line_start_y = point(mid_angle)
            line_mid_x = cx + (radius + 26) * math.cos(mid_angle)
            ideal_y = cy + (radius + 26) * math.sin(mid_angle)
            right_side = math.cos(mid_angle) >= 0
            callout_items.append(
                {
                    "color": color,
                    "line_start_x": line_start_x,
                    "line_start_y": line_start_y,
                    "line_mid_x": line_mid_x,
                    "ideal_y": ideal_y,
                    "right_side": right_side,
                    "lines": [f"{label}：{label_text}", *text_lines(callout)],
                }
            )
            angle = next_angle

        def assign_callout_y(items: list[dict[str, Any]]) -> None:
            if not items:
                return
            top_y = 96.0
            bottom_y = height - 46.0
            min_gap = 64.0
            ordered = sorted(items, key=lambda item: item["ideal_y"])
            previous_y = top_y - min_gap
            for item in ordered:
                item["line_y"] = max(float(item["ideal_y"]), previous_y + min_gap, top_y)
                previous_y = item["line_y"]
            overflow = ordered[-1]["line_y"] - bottom_y
            if overflow > 0:
                for item in ordered:
                    item["line_y"] -= overflow
            underflow = top_y - ordered[0]["line_y"]
            if underflow > 0:
                for item in ordered:
                    item["line_y"] += underflow

        assign_callout_y([item for item in callout_items if item["right_side"]])
        assign_callout_y([item for item in callout_items if not item["right_side"]])

        for item in callout_items:
            line_y = item["line_y"]
            line_end_x = 990 if item["right_side"] else 82
            text_x = line_end_x - 4 if item["right_side"] else line_end_x + 4
            text_anchor = "end" if item["right_side"] else "start"
            elements.append(
                f'<path d="M {item["line_start_x"]:.1f},{item["line_start_y"]:.1f} '
                f'L {item["line_mid_x"]:.1f},{line_y:.1f} L {line_end_x:.1f},{line_y:.1f}" '
                f'fill="none" stroke="{item["color"]}" stroke-width="1.4"/>'
            )
            elements.append(
                f'<text x="{text_x:.1f}" y="{line_y - 6:.1f}" class="legend" '
                f'text-anchor="{text_anchor}" fill="#0f172a">'
            )
            for line_index, line in enumerate(item["lines"]):
                dy = 0 if line_index == 0 else (20 if line_index == 1 else 15)
                weight = ' font-weight="700"' if line_index == 0 else ""
                elements.append(
                    f'<tspan x="{text_x:.1f}" dy="{dy}"{weight}>{html.escape(line)}</tspan>'
                )
            elements.append("</text>")
    elements.append("</svg>")
    return "\n".join(elements)


def write_svg_page(output_path: Path, title: str, panels: list[dict[str, Any]], source_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = []
    for panel in panels:
        if panel.get("kind") == "bar":
            body.append(render_bar_panel(panel))
        elif panel.get("kind") == "pie":
            body.append(render_pie_panel(panel))
        else:
            body.append(render_time_panel(panel))
    source_abs = source_path.resolve()
    try:
        source_rel = source_abs.relative_to(PROJECT_ROOT)
    except ValueError:
        source_rel = source_path
    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ margin: 20px; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
.meta {{ color: #64748b; font-size: 13px; margin-bottom: 16px; }}
.panel {{ width: 100%; max-width: 1160px; display: block; background: white; border: 1px solid #e2e8f0; margin: 12px 0; }}
.axis {{ stroke: #94a3b8; stroke-width: 1; }}
.tick {{ fill: #475569; font-size: 11px; }}
.legend {{ fill: #334155; font-size: 12px; }}
.panel-title {{ fill: #0f172a; font-size: 15px; font-weight: 700; }}
.bar-label {{ fill: #0f172a; font-size: 11px; font-weight: 700; }}
.point-label {{ font-size: 10px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 3px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="meta">來源：{html.escape(str(source_rel))}</div>
{chr(10).join(body)}
</body>
</html>
""",
        encoding="utf-8",
    )


def write_empty_page(output_path: Path, title: str, message: str, source_path: Path) -> None:
    write_svg_page(
        output_path,
        title,
        [{"title": "No visualization", "x": ["message"], "y": [0], "kind": "bar"}],
        source_path,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n<!-- {html.escape(message)} -->\n")


def output_path_for(csv_path: Path, dataset: str) -> Path:
    rel = csv_path.relative_to(DATA_ROOT / dataset)
    return DATA_VIZ_ROOT / dataset / rel.with_suffix(".html")


def line_series(
    df: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    return [
        {"label": (labels or {}).get(column, column), "y": numeric_series(df, column).tolist()}
        for column in columns
        if column in df.columns
    ]


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


def price_records(df: pd.DataFrame, adjusted: bool = False) -> list[dict[str, Any]]:
    if adjusted:
        columns = {
            "open": "open_adj",
            "high": "high_adj",
            "low": "low_adj",
            "close": "close_adj",
        }
    else:
        columns = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        }
    required = ["_viz_date", "Capacity", *columns.values()]
    if not all(column in df.columns for column in required):
        return []
    include_change = not adjusted and "Change" in df.columns
    records = []
    iter_columns = required + (["Change"] if include_change else [])
    for row in df[iter_columns].itertuples(index=False, name=None):
        date, capacity, open_, high, low, close = row[:6]
        values = [json_float(open_), json_float(high), json_float(low), json_float(close)]
        if any(value is None for value in values):
            continue
        record = {
            "t": date.strftime("%Y-%m-%d"),
            "o": values[0],
            "h": values[1],
            "l": values[2],
            "c": values[3],
            "v": json_float(capacity) or 0.0,
        }
        if include_change:
            change = json_float(row[6])
            if change is not None:
                record["chg"] = change
        records.append(record)
    return records


MARGIN_WEBGL_METRICS = [
    ("MarginCurrentBalance", "margin_balance", "融資今日餘額"),
    ("ShortCurrentBalance", "short_balance", "融券今日餘額"),
    ("MarginFinancingUsageRate", "margin_usage", "融資使用率"),
    ("MarginBalance20DayChangeRate", "margin_20d_change", "融資餘額20日變化率"),
    ("MarginMarketValueTo20DayAvgTurnover", "margin_value_turnover", "融資市值20日均成交值比"),
    ("ShortMarginBalanceRatio", "short_margin_ratio", "券資比"),
    ("MarginMarketValue", "margin_market_value", "融資市值"),
]

DAY_TRADING_WEBGL_METRICS = [
    ("DayTradingVolumeRatio", "day_volume_ratio", "當沖成交股數占比"),
    ("DayTradingTurnoverRatio", "day_turnover_ratio", "當沖成交值占比"),
    ("DayTradingTurnover", "day_turnover", "當沖成交值"),
    ("DayTradingAvgSpreadRate", "day_avg_spread_rate", "當沖平均價差率"),
    ("DayTradingAmountImbalanceRatio", "day_amount_imbalance", "當沖買賣金額差率"),
    ("DayTradingVolumeRatio20DayZScore", "day_volume_ratio_z20", "當沖成交股數占比20日ZScore"),
    ("DayTradingTurnover20DayZScore", "day_turnover_z20", "當沖成交值20日ZScore"),
    ("IntradayRangeRate", "intraday_range", "日內振幅"),
    ("OpenCloseReturn", "open_close_return", "開收報酬率"),
]

INSTITUTIONAL_WEBGL_METRICS = [
    ("InstitutionalNet", "institutional_net", "三大法人買賣超"),
    ("ForeignNetExDealer", "foreign_net", "外資買賣超"),
    ("InvestmentTrustNet", "investment_trust_net", "投信買賣超"),
    ("DealerNet", "dealer_net", "自營商買賣超"),
    ("ForeignBuyExDealer", "foreign_buy", "外資買進"),
    ("ForeignSellExDealer", "foreign_sell", "外資賣出"),
]

YIELD_PE_PB_WEBGL_METRICS = [
    ("DividendYield", "dividend_yield", "殖利率"),
    ("PEratio", "pe_ratio", "本益比"),
    ("PBratio", "pb_ratio", "股價淨值比"),
]

DIVIDEND_WEBGL_METRICS = [
    ("cash_dividend", "cash_dividend", "現金股利"),
    ("dividend_value", "dividend_value", "權值息值"),
    ("stock_dividend_rate", "stock_dividend_rate", "股票股利率"),
    ("ex_reference_price", "ex_reference_price", "除權息參考價"),
    ("opening_reference_price", "opening_reference_price", "開盤參考價"),
]


EX_RIGHT_EVENT_FIELDS = [
    ("right_or_dividend", "type", "權息別", "text"),
    ("previous_close", "previous_close", "除權息前收盤價", "number"),
    ("ex_reference_price", "reference_price", "除權息參考價", "number"),
    ("opening_reference_price", "opening_reference_price", "開盤參考價", "number"),
    ("opening_auction_base", "opening_auction_base", "開盤競價基準", "number"),
    ("dividend_value", "dividend_value", "權值息值", "number"),
    ("cash_dividend", "cash_dividend", "現金股利", "number"),
    ("stock_dividend_rate", "stock_dividend_rate", "股票股利率", "number"),
    ("cash_capital_increase_price", "cash_capital_increase_price", "現金增資認購價", "number"),
    ("cash_capital_increase_rate", "cash_capital_increase_rate", "現金增資配股率", "number"),
    ("deducted_dividend_reference_price", "deducted_dividend_reference_price", "減除股利參考價", "number"),
]


def overlay_payload_by_date(
    df: pd.DataFrame,
    metric_specs: list[tuple[str, str, str]],
    date_candidates: list[str] | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    df, date_col = normalize_date_column(df, date_candidates or ["Date"])
    available = [(column, key, label) for column, key, label in metric_specs if column in df.columns]
    if not available:
        return {}, []
    payload: dict[str, dict[str, float]] = {}
    for row in df[[date_col, *[column for column, _, _ in available]]].itertuples(index=False, name=None):
        date = row[0]
        values: dict[str, float] = {}
        for (_, key, _), value in zip(available, row[1:]):
            number = json_float(value)
            if number is not None:
                values[key] = number
        if values:
            payload[date.strftime("%Y-%m-%d")] = values
    metrics = [{"key": key, "label": label} for _, key, label in available]
    return payload, metrics


def margin_payload_by_date(df: pd.DataFrame) -> tuple[dict[str, dict[str, float]], list[dict[str, str]]]:
    return overlay_payload_by_date(df, MARGIN_WEBGL_METRICS, ["Date"])


def is_nonzero_event_value(value: Any) -> bool:
    number = json_float(value)
    return number is not None and abs(number) > 1e-9


def is_cash_capital_increase_event(values: dict[str, Any]) -> bool:
    if is_nonzero_event_value(values.get("cash_capital_increase_rate")):
        return True
    if is_nonzero_event_value(values.get("cash_capital_increase_price")):
        return True
    event_type = str(values.get("type") or "")
    if "權" not in event_type:
        return False
    previous_close = json_float(values.get("previous_close"))
    reference_price = json_float(values.get("reference_price"))
    deducted_reference = json_float(values.get("deducted_dividend_reference_price"))
    opening_base = json_float(values.get("opening_auction_base"))
    if previous_close is None or reference_price is None:
        return False
    if abs(previous_close - reference_price) <= 1e-9:
        return False
    for cash_capital_base in [deducted_reference, opening_base]:
        if cash_capital_base is not None and abs(cash_capital_base - previous_close) <= 1e-9:
            return True
    return False


def ex_right_events_by_date(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty:
        return {}
    df, date_col = normalize_date_column(df, ["ex_date"])
    available = [(column, key, label, kind) for column, key, label, kind in EX_RIGHT_EVENT_FIELDS if column in df.columns]
    events: dict[str, dict[str, Any]] = {}
    for row in df[[date_col, *[column for column, _, _, _ in available]]].itertuples(index=False, name=None):
        date = row[0]
        values: dict[str, Any] = {"label": "除權息"}
        detail_parts = []
        for (_, key, label, kind), value in zip(available, row[1:]):
            if pd.isna(value):
                continue
            if kind == "number":
                number = json_float(value)
                if number is None:
                    continue
                values[key] = number
                detail_parts.append(f"{label} {number:g}")
            else:
                text = str(value).strip()
                if json_float(text) is not None:
                    continue
                if text and text.lower() != "nan":
                    values[key] = text
                    detail_parts.append(f"{label} {text}")
        if "reference_price" in values or detail_parts:
            if is_cash_capital_increase_event(values):
                values["label"] = "現金增資"
            values["detail"] = "、".join(detail_parts)
            events[date.strftime("%Y-%m-%d")] = values
    return events


def find_ex_right_csv_for_stock(csv_path: Path, df: pd.DataFrame) -> Path | None:
    code = ""
    for column in ["Code", "stock_id"]:
        if column in df.columns and not df[column].dropna().empty:
            code = str(df[column].dropna().iloc[0]).strip()
            break
    if not code:
        code = csv_path.stem.split("_", 1)[0]
    matches = sorted((DATA_ROOT / "dividend" / "ex_right_dividend").glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def load_ex_right_events_for_stock(csv_path: Path, df: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], Path | None]:
    event_csv = find_ex_right_csv_for_stock(csv_path, df)
    if event_csv is None:
        return {}, None
    event_df = read_csv_canonical(event_csv, dtype={"stock_id": str})
    return ex_right_events_by_date(event_df), event_csv


def attach_margin_payload(records: list[dict[str, Any]], margin_by_date: dict[str, dict[str, float]]) -> None:
    if not margin_by_date:
        return
    for record in records:
        margin_values = margin_by_date.get(record["t"])
        if margin_values:
            record["m"] = margin_values


def attach_event_payload(records: list[dict[str, Any]], events_by_date: dict[str, dict[str, Any]]) -> None:
    if not events_by_date:
        return
    for record in records:
        event = events_by_date.get(record["t"])
        if event:
            record["e"] = event


def attach_volume_segments_payload(
    records: list[dict[str, Any]],
    volume_segments_by_date: dict[str, list[dict[str, Any]]],
) -> None:
    if not volume_segments_by_date:
        return
    for record in records:
        segments = volume_segments_by_date.get(record["t"])
        if segments:
            record["vs"] = segments


def find_price_csv_for_stock(csv_path: Path, df: pd.DataFrame) -> Path | None:
    code = ""
    if "Code" in df.columns and not df["Code"].dropna().empty:
        code = str(df["Code"].dropna().iloc[0]).strip()
    if not code:
        code = csv_path.stem.split("_", 1)[0]
    matches = sorted((DATA_ROOT / "price").glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def write_price_webgl_page(
    csv_path: Path,
    output_path: Path,
    title: str,
    df: pd.DataFrame,
    *,
    source_paths: list[Path] | None = None,
    margin_by_date: dict[str, dict[str, float]] | None = None,
    margin_metrics: list[dict[str, str]] | None = None,
    volume_segments_by_date: dict[str, list[dict[str, Any]]] | None = None,
    volume_segment_groups: list[dict[str, str]] | None = None,
    duplicate_auxiliary_payload_to_adjusted: bool = True,
    events_by_date: dict[str, dict[str, Any]] | None = None,
    page_suffix: str = "WebGL price",
    metric_control_label: str = "疊圖指標",
    highlight_rules: list[dict[str, Any]] | None = None,
    extra_body_before_chart: str = "",
    extra_body_after_chart: str = "",
    extra_styles: str = "",
) -> bool:
    df, _ = normalize_date_column(df, ["Date"])
    raw_records = price_records(df, adjusted=False)
    adjusted_records = price_records(df, adjusted=True)
    if not raw_records:
        return False
    margin_by_date = margin_by_date or {}
    margin_metrics = margin_metrics or []
    volume_segments_by_date = volume_segments_by_date or {}
    volume_segment_groups = volume_segment_groups or []
    events_by_date = events_by_date or {}
    highlight_rules = highlight_rules or []
    attach_margin_payload(raw_records, margin_by_date)
    attach_volume_segments_payload(raw_records, volume_segments_by_date)
    attach_event_payload(raw_records, events_by_date)
    if duplicate_auxiliary_payload_to_adjusted:
        attach_margin_payload(adjusted_records, margin_by_date)
        attach_volume_segments_payload(adjusted_records, volume_segments_by_date)
        attach_event_payload(adjusted_records, events_by_date)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_labels = []
    for source_path in source_paths or [csv_path]:
        source_abs = source_path.resolve()
        try:
            source_labels.append(str(source_abs.relative_to(PROJECT_ROOT)))
        except ValueError:
            source_labels.append(str(source_path))
    payload = json.dumps(
        {
            "raw": raw_records,
            "adjusted": adjusted_records,
            "marginMetrics": margin_metrics,
            "volumeSegmentGroups": volume_segment_groups,
            "highlightRules": highlight_rules,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    margin_control = ""
    if margin_metrics:
        options = "\n".join(
            f'<option value="{html.escape(metric["key"])}">{html.escape(metric["label"])}</option>'
            for metric in margin_metrics
        )
        margin_control = (
            f'<label class="metric-control">{html.escape(metric_control_label)} '
            f'<select id="marginMetric">{options}</select>'
            '</label>'
        )
    output_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} {html.escape(page_suffix)}</title>
<style>
body {{ margin: 20px; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
.meta {{ color: #64748b; font-size: 13px; margin-bottom: 12px; }}
.toolbar {{ display: flex; align-items: center; gap: 8px; margin: 12px 0; flex-wrap: wrap; }}
button {{ border: 1px solid #cbd5e1; background: white; color: #172033; padding: 6px 10px; cursor: pointer; }}
button.active {{ background: #172033; color: white; border-color: #172033; }}
select {{ border: 1px solid #cbd5e1; background: white; color: #172033; padding: 6px 8px; }}
.metric-control {{ color: #334155; font-size: 13px; }}
.chart-wrap {{ position: relative; height: min(78vh, 820px); min-height: 520px; border: 1px solid #d7dee9; background: white; }}
canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; display: block; }}
.readout {{ min-width: 420px; color: #334155; font-size: 13px; }}
.hint {{ color: #64748b; font-size: 12px; }}
{extra_styles}
</style>
</head>
<body>
<h1>{html.escape(title)} {html.escape(page_suffix)}</h1>
<div class="meta">來源：{html.escape('、'.join(source_labels))}</div>
{extra_body_before_chart}
<div class="toolbar">
  <button id="rawBtn" class="active" type="button">原始K線</button>
  <button id="adjBtn" type="button"{' disabled' if not adjusted_records else ''}>復權K線</button>
  {margin_control}
  <button id="resetBtn" type="button">重設</button>
  <span id="readout" class="readout"></span>
  <span class="hint">拖曳平移 / 滾輪縮放 / 移動游標 / 左右鍵逐日切換</span>
</div>
<div class="chart-wrap">
  <canvas id="glCanvas"></canvas>
  <canvas id="overlayCanvas"></canvas>
</div>
{extra_body_after_chart}
<script>
const PRICE_DATA = {payload};
const state = {{
  mode: "raw",
  start: 0,
  end: 1,
  dragging: false,
  dragX: 0,
  cursorIndex: null,
  needsDraw: true
}};
const glCanvas = document.getElementById("glCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const readout = document.getElementById("readout");
const rawBtn = document.getElementById("rawBtn");
const adjBtn = document.getElementById("adjBtn");
const resetBtn = document.getElementById("resetBtn");
const marginMetricSelect = document.getElementById("marginMetric");
const gl = glCanvas.getContext("webgl", {{antialias: true, preserveDrawingBuffer: false}});
if (!gl) {{
  document.querySelector(".chart-wrap").innerHTML = "<p style='padding:20px'>WebGL is not available in this browser.</p>";
  throw new Error("WebGL unavailable");
}}
const overlay = overlayCanvas.getContext("2d");
const vertexShaderSource = `
attribute vec2 a_position;
attribute vec4 a_color;
varying vec4 v_color;
void main() {{
  gl_Position = vec4(a_position, 0.0, 1.0);
  v_color = a_color;
}}`;
const fragmentShaderSource = `
precision mediump float;
varying vec4 v_color;
void main() {{
  gl_FragColor = v_color;
}}`;
function compileShader(type, source) {{
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
  return shader;
}}
const program = gl.createProgram();
gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexShaderSource));
gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
const posLoc = gl.getAttribLocation(program, "a_position");
const colorLoc = gl.getAttribLocation(program, "a_color");
const buffer = gl.createBuffer();
const X_LEFT = -0.96;
const X_SPAN = 1.84;
const rawByDate = new Map((PRICE_DATA.raw || []).map((row) => [row.t, row]));

function activeData() {{ return PRICE_DATA[state.mode] || PRICE_DATA.raw; }}
function auxiliaryRow(row) {{
  if (!row) return null;
  if (row.m || row.vs || row.e) return row;
  return rawByDate.get(row.t) || row;
}}
function metricValue(row, key) {{
  const source = auxiliaryRow(row);
  const value = source && source.m ? Number(source.m[key]) : null;
  return Number.isFinite(value) ? value : null;
}}
function volumeSegments(row) {{
  const source = auxiliaryRow(row);
  return source && source.vs ? source.vs : null;
}}
function eventPayload(row) {{
  const source = auxiliaryRow(row);
  return source && source.e ? source.e : null;
}}
function activeMarginMetric() {{
  if (!PRICE_DATA.marginMetrics.length) return null;
  const key = marginMetricSelect ? marginMetricSelect.value : PRICE_DATA.marginMetrics[0].key;
  return PRICE_DATA.marginMetrics.find((metric) => metric.key === key) || PRICE_DATA.marginMetrics[0];
}}
function formatNumber(value) {{
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  if (abs >= 1_000_000_000) return (value / 1_000_000_000).toFixed(2).replace(/\\.00$/, "") + "B";
  if (abs >= 1_000_000) return (value / 1_000_000).toFixed(2).replace(/\\.00$/, "") + "M";
  if (abs >= 1_000) return (value / 1_000).toFixed(2).replace(/\\.00$/, "") + "K";
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(2).replace(/0+$/, "").replace(/\\.$/, "");
  return value.toFixed(4).replace(/0+$/, "").replace(/\\.$/, "");
}}
function colorWithOpacity(hex, opacity) {{
  const match = /^#?([0-9a-fA-F]{{6}})$/.exec(String(hex || "#64748b").trim());
  opacity = Number.isFinite(Number(opacity)) ? Number(opacity) : 0.12;
  if (!match) return `rgba(100,116,139,${{opacity}})`;
  const n = parseInt(match[1], 16);
  return `rgba(${{(n >> 16) & 255}},${{(n >> 8) & 255}},${{n & 255}},${{opacity}})`;
}}
function hasHighlight(row, key) {{
  const value = metricValue(row, key);
  return Number.isFinite(value) && Math.abs(value) > 0.5;
}}
function resetView() {{
  const data = activeData();
  state.start = 0;
  state.end = Math.max(1, data.length - 1);
  state.cursorIndex = data.length ? data.length - 1 : null;
  state.needsDraw = true;
}}
function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = glCanvas.getBoundingClientRect();
  for (const canvas of [glCanvas, overlayCanvas]) {{
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  }}
  state.needsDraw = true;
}}
function colorFor(row) {{
  return row.c >= row.o ? [0.86, 0.16, 0.16, 1.0] : [0.05, 0.55, 0.31, 1.0];
}}
function colorArray(hex, alpha) {{
  const match = /^#?([0-9a-fA-F]{{6}})$/.exec(String(hex || "#64748b").trim());
  const opacity = Number.isFinite(Number(alpha)) ? Number(alpha) : 0.65;
  if (!match) return [0.39, 0.45, 0.55, opacity];
  const n = parseInt(match[1], 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255, opacity];
}}
function segmentValue(segment, field) {{
  if (Array.isArray(segment)) {{
    const position = field === "p" ? 1 : 0;
    const value = Number(segment[position]);
    return Number.isFinite(value) ? value : null;
  }}
  const value = Number(segment ? segment[field] : null);
  return Number.isFinite(value) ? value : null;
}}
function segmentGroup(index) {{
  const groups = PRICE_DATA.volumeSegmentGroups || [];
  return groups[index] || {{}};
}}
function segmentColor(segment, index) {{
  if (segment && !Array.isArray(segment) && segment.color) return segment.color;
  return segmentGroup(index).color || "#64748b";
}}
function segmentLabel(segment, index) {{
  if (segment && !Array.isArray(segment) && segment.label) return segment.label;
  return segmentGroup(index).label || "";
}}
function pctChangeFor(index) {{
  const data = activeData();
  const row = data[index];
  const previous = data[index - 1];
  if (!row || !Number.isFinite(row.c)) return null;
  const event = eventPayload(row);
  const eventReference = state.mode === "raw" && event && Number.isFinite(event.reference_price) ? event.reference_price : null;
  const changeReference = state.mode === "raw" && Number.isFinite(row.chg) ? row.c - row.chg : null;
  const base = eventReference || (changeReference && changeReference > 0 ? changeReference : null) || (previous && Number.isFinite(previous.c) ? previous.c : null);
  if (!base) return null;
  return {{
    value: ((row.c - base) / base) * 100,
    base,
    adjustedForEvent: eventReference !== null,
  }};
}}
function addRect(vertices, x0, y0, x1, y1, color) {{
  const left = Math.min(x0, x1), right = Math.max(x0, x1);
  const bottom = Math.min(y0, y1), top = Math.max(y0, y1);
  const points = [left,bottom, right,bottom, right,top, left,bottom, right,top, left,top];
  for (let i = 0; i < points.length; i += 2) vertices.push(points[i], points[i+1], ...color);
}}
function drawWebGL() {{
  const data = activeData();
  const metric = activeMarginMetric();
  const n = data.length;
  const width = Math.max(1, state.end - state.start);
  const from = Math.max(0, Math.floor(state.start) - 2);
  const to = Math.min(n - 1, Math.ceil(state.end) + 2);
  let minPrice = Infinity, maxPrice = -Infinity, maxVol = 0, minMargin = Infinity, maxMargin = -Infinity;
  for (let i = from; i <= to; i++) {{
    const row = data[i];
    if (!row) continue;
    minPrice = Math.min(minPrice, row.l);
    maxPrice = Math.max(maxPrice, row.h);
    maxVol = Math.max(maxVol, row.v || 0);
    const marginValue = metric ? metricValue(row, metric.key) : null;
    if (Number.isFinite(marginValue)) {{
      minMargin = Math.min(minMargin, marginValue);
      maxMargin = Math.max(maxMargin, marginValue);
    }}
  }}
  if (!Number.isFinite(minPrice) || !Number.isFinite(maxPrice) || maxPrice <= minPrice) {{
    minPrice = 0; maxPrice = 1;
  }}
  if (!Number.isFinite(minMargin) || !Number.isFinite(maxMargin) || maxMargin <= minMargin) {{
    minMargin = 0; maxMargin = 1;
  }}
  const pad = (maxPrice - minPrice) * 0.05;
  minPrice -= pad; maxPrice += pad;
  const marginPad = (maxMargin - minMargin) * 0.08 || 1;
  minMargin -= marginPad; maxMargin += marginPad;
  const layout = metric
    ? {{priceBottom: -0.02, priceTop: 0.90, marginBottom: -0.50, marginTop: -0.18, volBottom: -0.92, volTop: -0.62}}
    : {{priceBottom: -0.24, priceTop: 0.90, marginBottom: null, marginTop: null, volBottom: -0.92, volTop: -0.45}};
  const xOf = (i) => X_LEFT + ((i - state.start) / width) * X_SPAN;
  const yPrice = (v) => layout.priceBottom + ((v - minPrice) / (maxPrice - minPrice)) * (layout.priceTop - layout.priceBottom);
  const yVol = (v) => layout.volBottom + (maxVol ? (v / maxVol) : 0) * (layout.volTop - layout.volBottom);
  const candleW = Math.max(2 / glCanvas.width, Math.min(0.018, 1.20 / width));
  const wickW = Math.max(2 / glCanvas.width, candleW * 0.16);
  const vertices = [];
  for (let i = from; i <= to; i++) {{
    const row = data[i];
    if (!row) continue;
    const x = xOf(i);
    const color = colorFor(row);
    const openY = yPrice(row.o);
    const closeY = yPrice(row.c);
    const bodyCenter = (openY + closeY) / 2;
    const minBodyH = Math.max(2 / glCanvas.height, 3 * 2 / glCanvas.height);
    const bodyH = Math.max(Math.abs(closeY - openY), minBodyH);
    addRect(vertices, x - wickW / 2, yPrice(row.l), x + wickW / 2, yPrice(row.h), color);
    addRect(vertices, x - candleW / 2, bodyCenter - bodyH / 2, x + candleW / 2, bodyCenter + bodyH / 2, color);
    const segments = volumeSegments(row);
    if (segments && segments.length) {{
      const volumeTop = yVol(row.v || 0);
      const segmentTotal = segments.reduce((total, segment) => total + Math.max(0, segmentValue(segment, "v") || 0), 0);
      const denominator = segmentTotal > 0 ? segmentTotal : (row.v || 1);
      let stackBottom = layout.volBottom;
      for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex++) {{
        const segment = segments[segmentIndex];
        const share = Math.max(0, segmentValue(segment, "v") || 0) / denominator;
        const stackTop = stackBottom + (volumeTop - layout.volBottom) * share;
        addRect(vertices, x - candleW / 2, stackBottom, x + candleW / 2, stackTop, colorArray(segmentColor(segment, segmentIndex), 0.72));
        stackBottom = stackTop;
      }}
    }} else {{
      addRect(vertices, x - candleW / 2, layout.volBottom, x + candleW / 2, yVol(row.v || 0), [color[0], color[1], color[2], 0.50]);
    }}
  }}
  gl.viewport(0, 0, glCanvas.width, glCanvas.height);
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(colorLoc);
  gl.vertexAttribPointer(colorLoc, 4, gl.FLOAT, false, 24, 8);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 6);
  drawOverlay(minPrice, maxPrice, maxVol, from, to, xOf, layout, metric, minMargin, maxMargin);
}}
function drawOverlay(minPrice, maxPrice, maxVol, from, to, xOf, layout, metric, minMargin, maxMargin) {{
  const ctx = overlay;
  const w = overlayCanvas.width, h = overlayCanvas.height;
  const dpr = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.scale(dpr, dpr);
  const cssW = w / dpr, cssH = h / dpr;
  ctx.strokeStyle = "#e2e8f0";
  ctx.fillStyle = "#475569";
  ctx.font = "12px Arial";
  ctx.lineWidth = 1;
  function pxX(ndc) {{ return ((ndc + 1) / 2) * cssW; }}
  function pxY(ndc) {{ return ((1 - ndc) / 2) * cssH; }}
  const data = activeData();
  function drawHighlightRules() {{
    const rules = PRICE_DATA.highlightRules || [];
    if (!rules.length) return;
    const visibleRules = [];
    for (const rule of rules) {{
      if (rule.panel === "margin" && !metric) continue;
      const topNdc = rule.panel === "margin" ? layout.marginTop : layout.priceTop;
      const bottomNdc = rule.panel === "margin" ? layout.marginBottom : layout.priceBottom;
      if (topNdc === null || bottomNdc === null) continue;
      visibleRules.push(rule);
      const topY = Math.min(pxY(topNdc), pxY(bottomNdc));
      const bottomY = Math.max(pxY(topNdc), pxY(bottomNdc));
      ctx.fillStyle = colorWithOpacity(rule.color, rule.opacity);
      let rangeStart = null;
      const closeRange = (endIndex) => {{
        if (rangeStart === null) return;
        const x0 = pxX(xOf(rangeStart - 0.5));
        const x1 = pxX(xOf(endIndex + 0.5));
        const left = Math.max(0, Math.min(x0, x1));
        const right = Math.min(cssW, Math.max(x0, x1));
        ctx.fillRect(left, topY, Math.max(1, right - left), bottomY - topY);
        if (rule.marker === "top" || rule.marker === "bottom") {{
          const midX = (left + right) / 2;
          const markerOpacity = Math.min(0.95, (Number(rule.opacity) || 0.12) * 4.5);
          ctx.fillStyle = colorWithOpacity(rule.color, markerOpacity);
          ctx.beginPath();
          if (rule.marker === "top") {{
            ctx.moveTo(midX, topY + 6);
            ctx.lineTo(midX - 5, topY + 18);
            ctx.lineTo(midX + 5, topY + 18);
          }} else {{
            ctx.moveTo(midX, bottomY - 6);
            ctx.lineTo(midX - 5, bottomY - 18);
            ctx.lineTo(midX + 5, bottomY - 18);
          }}
          ctx.closePath();
          ctx.fill();
          ctx.fillStyle = colorWithOpacity(rule.color, rule.opacity);
        }}
        rangeStart = null;
      }};
      for (let i = from; i <= to; i++) {{
        const row = data[i];
        if (hasHighlight(row, rule.key)) {{
          if (rangeStart === null) rangeStart = i;
        }} else {{
          closeRange(i - 1);
        }}
      }}
      closeRange(to);
    }}
    if (visibleRules.length) {{
      let legendX = 8;
      const legendY = 18;
      ctx.font = "12px Arial";
      for (const rule of visibleRules.slice(0, 6)) {{
        ctx.fillStyle = colorWithOpacity(rule.color, Math.min(0.55, (Number(rule.opacity) || 0.12) * 2.8));
        ctx.fillRect(legendX, legendY - 10, 12, 12);
        ctx.fillStyle = "#334155";
        const label = rule.label || rule.key;
        ctx.fillText(label, legendX + 16, legendY);
        legendX += Math.min(220, ctx.measureText(label).width + 34);
      }}
    }}
  }}
  drawHighlightRules();
  for (let g = 0; g <= 5; g++) {{
    const t = g / 5;
    const ndc = layout.priceBottom + t * (layout.priceTop - layout.priceBottom);
    const y = pxY(ndc);
    const price = minPrice + t * (maxPrice - minPrice);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cssW, y); ctx.stroke();
    ctx.fillText(formatNumber(price), cssW - 76, y - 4);
  }}
  ctx.beginPath(); ctx.moveTo(0, pxY(layout.volTop)); ctx.lineTo(cssW, pxY(layout.volTop)); ctx.stroke();
  if (metric) {{
    const marginTopY = pxY(layout.marginTop);
    const marginBottomY = pxY(layout.marginBottom);
    const yMargin = (value) => marginBottomY - ((value - minMargin) / (maxMargin - minMargin)) * (marginBottomY - marginTopY);
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 1.7;
    ctx.beginPath();
    let started = false;
    for (let i = from; i <= to; i++) {{
      const row = data[i];
      const value = metric ? metricValue(row, metric.key) : null;
      if (!Number.isFinite(value)) {{
        started = false;
        continue;
      }}
      const x = pxX(xOf(i));
      const y = yMargin(value);
      if (!started) {{
        ctx.moveTo(x, y);
        started = true;
      }} else {{
        ctx.lineTo(x, y);
      }}
    }}
    ctx.stroke();
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, marginTopY); ctx.lineTo(cssW, marginTopY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, marginBottomY); ctx.lineTo(cssW, marginBottomY); ctx.stroke();
    ctx.fillStyle = "#2563eb";
    ctx.fillText(metric.label, 8, marginTopY + 14);
    ctx.fillStyle = "#475569";
    ctx.fillText(formatNumber(maxMargin), cssW - 76, marginTopY + 12);
    ctx.fillText(formatNumber(minMargin), cssW - 76, marginBottomY - 4);
  }}
  ctx.strokeStyle = "#f59e0b";
  ctx.fillStyle = "#b45309";
  ctx.lineWidth = 1;
  for (let i = from; i <= to; i++) {{
    const row = data[i];
    const event = eventPayload(row);
    if (!row || !event) continue;
    const x = pxX(xOf(i));
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x, pxY(layout.priceTop)); ctx.lineTo(x, pxY(layout.volBottom)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText(event.label || "除權息", Math.max(4, Math.min(cssW - 52, x + 4)), pxY(layout.priceTop) + 14);
  }}
  const ticks = 6;
  for (let t = 0; t <= ticks; t++) {{
    const idx = Math.round(state.start + (t / ticks) * (state.end - state.start));
    if (!data[idx]) continue;
    ctx.fillText(data[idx].t, Math.max(4, Math.min(cssW - 80, pxX(xOf(idx)) - 30)), cssH - 12);
  }}
  if (state.cursorIndex !== null) {{
    const idx = Math.max(0, Math.min(data.length - 1, state.cursorIndex));
    const row = data[idx];
    if (row) {{
      const x = pxX(xOf(idx));
      const pctChange = pctChangeFor(idx);
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cssH); ctx.stroke();
      const lines = [
        row.t,
        `開 ${{row.o.toFixed(2)}}  收 ${{row.c.toFixed(2)}}`,
        `高 ${{row.h.toFixed(2)}}  低 ${{row.l.toFixed(2)}}`,
        `漲跌幅 ${{pctChange === null ? "N/A" : (pctChange.value >= 0 ? "+" : "") + pctChange.value.toFixed(2) + "%"}}${{pctChange && pctChange.adjustedForEvent ? " (除權息參考價)" : ""}}`,
        `成交量 ${{Math.round(row.v).toLocaleString()}}`
      ];
      const segments = volumeSegments(row);
      if (segments && segments.length) {{
        const segmentTotal = segments.reduce((total, segment) => total + Math.max(0, segmentValue(segment, "v") || 0), 0);
        const segmentLabels = segments.map((segment, segmentIndex) => {{
          const participation = segmentValue(segment, "p");
          const denominator = row.v || segmentTotal;
          const ratio = participation !== null ? participation : (denominator ? Math.max(0, segmentValue(segment, "v") || 0) / denominator : 0);
          return `${{segmentLabel(segment, segmentIndex)}} ${{(ratio * 100).toFixed(1)}}%`;
        }}).join("  ");
        if (segmentLabels) lines.push(`\u6210\u4ea4\u53c3\u8207 ${{segmentLabels}}`);
      }}
      const event = eventPayload(row);
      if (event) {{
        lines.push(`${{event.label || "除權息"}}${{event.reference_price ? " 參考價 " + formatNumber(event.reference_price) : ""}}`);
        if (event.dividend_value) lines.push(`權值息值 ${{formatNumber(event.dividend_value)}}`);
        if (event.detail) lines.push(event.detail);
      }}
      const metricReadout = metric ? metricValue(row, metric.key) : null;
      if (metric && Number.isFinite(metricReadout)) {{
        lines.push(`${{metric.label}} ${{formatNumber(metricReadout)}}`);
      }}
      readout.textContent = lines.join("  ");
      ctx.font = "12px Arial";
      const boxW = Math.max(...lines.map(line => ctx.measureText(line).width)) + 18;
      const boxH = lines.length * 17 + 12;
      const boxX = x + boxW + 16 > cssW ? x - boxW - 12 : x + 12;
      const boxY = 18;
      ctx.fillStyle = "rgba(255,255,255,0.94)";
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1;
      ctx.fillRect(boxX, boxY, boxW, boxH);
      ctx.strokeRect(boxX, boxY, boxW, boxH);
      ctx.fillStyle = "#111827";
      lines.forEach((line, i) => ctx.fillText(line, boxX + 9, boxY + 20 + i * 17));
    }}
  }}
  ctx.restore();
}}
function requestDraw() {{
  if (!state.needsDraw) return;
  state.needsDraw = false;
  drawWebGL();
}}
function clampView() {{
  const data = activeData();
  const minWidth = Math.min(20, Math.max(1, data.length - 1));
  const maxWidth = Math.max(1, data.length - 1);
  let width = Math.max(minWidth, Math.min(maxWidth, state.end - state.start));
  if (state.start < 0) {{ state.start = 0; state.end = width; }}
  if (state.end > maxWidth) {{ state.end = maxWidth; state.start = maxWidth - width; }}
}}
function clampCursor() {{
  const data = activeData();
  if (!data.length) {{ state.cursorIndex = null; return; }}
  if (state.cursorIndex === null) state.cursorIndex = Math.round(state.end);
  state.cursorIndex = Math.max(0, Math.min(data.length - 1, state.cursorIndex));
}}
function ensureCursorVisible() {{
  clampCursor();
  if (state.cursorIndex === null) return;
  const width = state.end - state.start;
  if (state.cursorIndex < state.start) {{
    state.start = state.cursorIndex;
    state.end = state.start + width;
  }} else if (state.cursorIndex > state.end) {{
    state.end = state.cursorIndex;
    state.start = state.end - width;
  }}
  clampView();
}}
function setCursorIndex(index, keepVisible = false) {{
  state.cursorIndex = Math.round(index);
  if (keepVisible) ensureCursorVisible();
  else clampCursor();
  state.needsDraw = true;
}}
function cursorIndexFromClientX(clientX) {{
  const rect = overlayCanvas.getBoundingClientRect();
  const chartLeft = ((X_LEFT + 1) / 2) * rect.width;
  const chartWidth = (X_SPAN / 2) * rect.width;
  const ratio = Math.max(0, Math.min(1, (clientX - rect.left - chartLeft) / Math.max(1, chartWidth)));
  return Math.round(state.start + ratio * (state.end - state.start));
}}
overlayCanvas.addEventListener("pointerdown", (event) => {{
  state.dragging = true;
  state.dragX = event.clientX;
  overlayCanvas.setPointerCapture(event.pointerId);
}});
overlayCanvas.addEventListener("pointermove", (event) => {{
  const rect = overlayCanvas.getBoundingClientRect();
  setCursorIndex(cursorIndexFromClientX(event.clientX), false);
  if (state.dragging) {{
    const width = state.end - state.start;
    const dx = event.clientX - state.dragX;
    const shift = -dx / Math.max(1, rect.width) * width;
    state.start += shift;
    state.end += shift;
    state.dragX = event.clientX;
    clampView();
  }}
  state.needsDraw = true;
}});
overlayCanvas.addEventListener("pointerup", () => {{ state.dragging = false; }});
overlayCanvas.addEventListener("pointerleave", () => {{ state.dragging = false; state.needsDraw = true; }});
overlayCanvas.addEventListener("wheel", (event) => {{
  event.preventDefault();
  const rect = overlayCanvas.getBoundingClientRect();
  const data = activeData();
  const focus = state.start + ((event.clientX - rect.left) / Math.max(1, rect.width)) * (state.end - state.start);
  const factor = event.deltaY < 0 ? 0.82 : 1.22;
  const width = Math.max(10, Math.min(data.length - 1, (state.end - state.start) * factor));
  const leftFrac = (focus - state.start) / Math.max(1, state.end - state.start);
  state.start = focus - width * leftFrac;
  state.end = state.start + width;
  clampView();
  ensureCursorVisible();
  state.needsDraw = true;
}}, {{passive: false}});
window.addEventListener("keydown", (event) => {{
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  const tag = event.target && event.target.tagName ? event.target.tagName.toLowerCase() : "";
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  event.preventDefault();
  clampCursor();
  if (state.cursorIndex === null) return;
  const step = event.key === "ArrowRight" ? 1 : -1;
  setCursorIndex(state.cursorIndex + step, true);
}});
rawBtn.addEventListener("click", () => {{
  state.mode = "raw"; rawBtn.classList.add("active"); adjBtn.classList.remove("active"); resetView();
}});
adjBtn.addEventListener("click", () => {{
  if (!PRICE_DATA.adjusted.length) return;
  state.mode = "adjusted"; adjBtn.classList.add("active"); rawBtn.classList.remove("active"); resetView();
}});
if (marginMetricSelect) {{
  marginMetricSelect.addEventListener("change", () => {{ state.needsDraw = true; }});
}}
resetBtn.addEventListener("click", resetView);
window.addEventListener("resize", resize);
resize();
resetView();
function loop() {{
  if (state.needsDraw) requestDraw();
  requestAnimationFrame(loop);
}}
loop();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return True


def price_viz(csv_path: Path, output_path: Path) -> None:
    df = read_csv_canonical(csv_path)
    if df.empty:
        write_empty_page(output_path, safe_title_from_path(csv_path), "No rows in source CSV.", csv_path)
        return

    if {"Code", "Name", "Close"}.issubset(df.columns) and df["Code"].nunique(dropna=True) > 1:
        working = df.copy()
        label = working["Code"].astype(str) + " " + working["Name"].astype(str)
        working["_label"] = label
        turnover_col = "TradeValue" if "TradeValue" in working.columns else None
        sort_col = turnover_col or "Close"
        top = working.sort_values(sort_col, key=lambda values: pd.to_numeric(values, errors="coerce"), ascending=False).head(50)
        panels = [
            {
                "title": "Top 50 close prices",
                "x": top["_label"].tolist(),
                "y": numeric_series(top, "Close").tolist(),
                "kind": "bar",
            }
        ]
        if turnover_col:
            panels.insert(
                0,
                {
                    "title": "Top 50 turnover",
                    "x": top["_label"].tolist(),
                    "y": numeric_series(top, turnover_col).tolist(),
                    "kind": "bar",
                },
            )
        write_svg_page(output_path, f"{safe_title_from_path(csv_path)} market snapshot", panels, csv_path)
        return

    events_by_date, event_csv = load_ex_right_events_for_stock(csv_path, df)
    source_paths = [csv_path]
    if event_csv is not None:
        source_paths.append(event_csv)
    if write_price_webgl_page(
        csv_path,
        output_path,
        f"{safe_title_from_path(csv_path)} price",
        df,
        source_paths=source_paths,
        events_by_date=events_by_date,
    ):
        return

    try:
        import plotly.io as pio
        from stock_viz import build_stock_figure, get_autorange_script, get_stock_title

        config = {
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ],
        }
        raw_fig, raw_clean = build_stock_figure(df, f"{get_stock_title(csv_path)} - raw OHLC")
        raw_html = pio.to_html(
            raw_fig,
            include_plotlyjs="cdn",
            full_html=False,
            config=config,
            post_script=get_autorange_script(raw_clean),
        )

        adjusted_html = ""
        if {"open_adj", "high_adj", "low_adj", "close_adj"}.issubset(df.columns):
            adjusted_df = df[["Date", "Capacity", "open_adj", "high_adj", "low_adj", "close_adj"]].rename(
                columns={
                    "open_adj": "Open",
                    "high_adj": "High",
                    "low_adj": "Low",
                    "close_adj": "Close",
                }
            )
            adjusted_fig, adjusted_clean = build_stock_figure(
                adjusted_df,
                f"{get_stock_title(csv_path)} - adjusted OHLC",
            )
            adjusted_html = pio.to_html(
                adjusted_fig,
                include_plotlyjs=False,
                full_html=False,
                config=config,
                post_script=get_autorange_script(adjusted_clean),
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        csv_abs = csv_path.resolve()
        try:
            source_rel = csv_abs.relative_to(PROJECT_ROOT)
        except ValueError:
            source_rel = csv_path
        output_path.write_text(
            f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(get_stock_title(csv_path))} price</title>
<style>
body {{ margin: 20px; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; margin: 22px 0 8px; }}
.meta {{ color: #64748b; font-size: 13px; margin-bottom: 12px; }}
.chart {{ background: white; border: 1px solid #e2e8f0; padding: 8px; margin-bottom: 18px; }}
</style>
</head>
<body>
<h1>{html.escape(get_stock_title(csv_path))} price</h1>
<div class="meta">來源：{html.escape(str(source_rel))}</div>
<h2>Raw OHLC K-plot</h2>
<div class="chart">{raw_html}</div>
{('<h2>Adjusted OHLC K-plot</h2><div class="chart">' + adjusted_html + '</div>') if adjusted_html else ''}
</body>
</html>
""",
            encoding="utf-8",
        )
        return
    except Exception:
        pass

    df, date_col = normalize_date_column(df, ["Date"])
    dates = df[date_col].dt.strftime("%Y-%m-%d").tolist()
    title = f"{safe_title_from_path(csv_path)} price"
    panels = [
        {"title": "Raw OHLC", "x": dates, "series": line_series(df, ["Close", "Open", "High", "Low"])},
        {"title": "Adjusted OHLC", "x": dates, "series": line_series(df, ["close_adj", "open_adj", "high_adj", "low_adj"])},
        {"title": "Capacity", "x": dates, "series": line_series(df, ["Capacity"])},
    ]
    write_svg_page(output_path, title, [panel for panel in panels if panel["series"]], csv_path)


def price_overlay_viz(
    csv_path: Path,
    output_path: Path,
    *,
    metric_specs: list[tuple[str, str, str]],
    date_candidates: list[str] | None = None,
    page_suffix: str,
    metric_control_label: str,
) -> None:
    overlay_df = read_csv_canonical(csv_path, dtype=str)
    price_csv = find_price_csv_for_stock(csv_path, overlay_df)
    if price_csv is None:
        write_empty_page(output_path, safe_title_from_path(csv_path), "Matching price CSV was not found.", csv_path)
        return
    price_df = read_csv_canonical(price_csv)
    overlay_by_date, overlay_metrics = overlay_payload_by_date(overlay_df, metric_specs, date_candidates or ["Date"])
    events_by_date, event_csv = load_ex_right_events_for_stock(price_csv, price_df)
    source_paths = [price_csv, csv_path]
    if event_csv is not None and event_csv != csv_path:
        source_paths.append(event_csv)
    if write_price_webgl_page(
        price_csv,
        output_path,
        stock_label(overlay_df, csv_path),
        price_df,
        source_paths=source_paths,
        margin_by_date=overlay_by_date,
        margin_metrics=overlay_metrics,
        events_by_date=events_by_date,
        page_suffix=page_suffix,
        metric_control_label=metric_control_label,
    ):
        return
    write_empty_page(output_path, safe_title_from_path(csv_path), "Price CSV did not contain renderable OHLC rows.", csv_path)


def yield_pe_pb_viz(csv_path: Path, output_path: Path) -> None:
    price_overlay_viz(
        csv_path,
        output_path,
        metric_specs=YIELD_PE_PB_WEBGL_METRICS,
        date_candidates=["Date"],
        page_suffix="價格與殖利率本益比股淨比",
        metric_control_label="估值指標",
    )


def institutional_viz(csv_path: Path, output_path: Path) -> None:
    price_overlay_viz(
        csv_path,
        output_path,
        metric_specs=INSTITUTIONAL_WEBGL_METRICS,
        date_candidates=["Date"],
        page_suffix="價格與法人買賣超",
        metric_control_label="法人指標",
    )


def day_trading_viz(csv_path: Path, output_path: Path) -> None:
    price_overlay_viz(
        csv_path,
        output_path,
        metric_specs=DAY_TRADING_WEBGL_METRICS,
        date_candidates=["Date"],
        page_suffix="價格與當沖",
        metric_control_label="當沖指標",
    )


def margin_viz(csv_path: Path, output_path: Path) -> None:
    margin_df = read_csv_canonical(csv_path, dtype={"Code": str})
    price_csv = find_price_csv_for_stock(csv_path, margin_df)
    if price_csv is None:
        write_empty_page(output_path, safe_title_from_path(csv_path), "Matching price CSV was not found.", csv_path)
        return
    price_df = read_csv_canonical(price_csv)
    margin_by_date, margin_metrics = margin_payload_by_date(margin_df)
    events_by_date, event_csv = load_ex_right_events_for_stock(price_csv, price_df)
    source_paths = [price_csv, csv_path]
    if event_csv is not None:
        source_paths.append(event_csv)
    if write_price_webgl_page(
        price_csv,
        output_path,
        stock_label(margin_df, csv_path),
        price_df,
        source_paths=source_paths,
        margin_by_date=margin_by_date,
        margin_metrics=margin_metrics,
        events_by_date=events_by_date,
        page_suffix="價格與融資融券",
        metric_control_label="融資融券指標",
    ):
        return
    write_empty_page(output_path, safe_title_from_path(csv_path), "Price CSV did not contain renderable OHLC rows.", csv_path)


def compact_shareholding_bucket(label: Any) -> str:
    text = "" if pd.isna(label) else str(label).strip()
    if not text:
        return ""
    if "\u5dee\u7570" in text:
        return "\u5dee\u7570"
    if "\u5408\u8a08" in text:
        return "\u5408\u8a08"
    numbers = [int(item.replace(",", "")) for item in re.findall(r"\d[\d,]*", text)]
    if not numbers:
        return text

    def compact(value: int) -> str:
        if value >= 1_000_000:
            return f"{value // 1_000_000}M"
        if value >= 1_000:
            return f"{value // 1_000}k"
        return str(value)

    if len(numbers) >= 2:
        return f"{compact(numbers[0])}-{compact(numbers[1])}"
    if "\u4ee5\u4e0a" in text or "+" in text:
        return f"{compact(numbers[0])}+"
    return compact(numbers[0])


def parse_shareholding_bucket_range(label: Any) -> tuple[int | None, int | None]:
    text = "" if pd.isna(label) else str(label).strip()
    if not text or "\u5dee\u7570" in text or "\u5408\u8a08" in text:
        return None, None
    numbers = [int(item.replace(",", "")) for item in re.findall(r"\d[\d,]*", text)]
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    if len(numbers) == 1:
        return numbers[0], None
    return None, None


def nearest_shareholding_boundary(threshold: float, ranges: list[tuple[int, int | None]]) -> int:
    boundaries = {lower for lower, _upper in ranges if lower > 0}
    for _lower, upper in ranges:
        if upper is not None:
            boundaries.add(upper + 1)
    if not boundaries:
        return max(1, math.ceil(threshold))
    return min(sorted(boundaries), key=lambda value: (abs(value - threshold), value))


def latest_close_for_shareholding(csv_path: Path, latest_date: pd.Timestamp) -> float | None:
    price_path = DATA_ROOT / "price" / csv_path.name
    if not price_path.exists():
        return None
    price_df = read_csv_canonical(price_path, dtype=str)
    price_df, price_date_col = normalize_date_column(price_df, ["Date", "date"])
    close_col = next((col for col in ["Close", "close", "\u6536\u76e4\u50f9"] if col in price_df.columns), None)
    if not close_col:
        return None
    price_df = price_df[price_df[price_date_col].le(latest_date)].copy()
    price_df["_viz_close"] = pd.to_numeric(price_df[close_col], errors="coerce")
    price_df = price_df.dropna(subset=["_viz_close"])
    if price_df.empty:
        return None
    return float(price_df.iloc[-1]["_viz_close"])


def format_shareholding_threshold(shares: int, close_price: float | None) -> str:
    if close_price and math.isfinite(close_price):
        value = shares * close_price
        return f"{shares:,.0f}\u80a1/\u7d04{format_twd_wan(value)}"
    return f"{shares:,.0f}\u80a1"


def format_close_price(value: float | None) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "\u7121"
    text = f"{float(value):,.2f}".rstrip("0").rstrip(".")
    return text


def format_twd_wan(value: float | None) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "\u7121\u6536\u76e4\u50f9"
    value = float(value)
    if value >= 100_000_000:
        amount = f"{value / 100_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{amount}\u5104"
    return f"{value / 10_000:.0f}\u842c"


def percentage_labels_sum_100(values: list[float]) -> list[str]:
    rounded = [round(float(value), 2) for value in values]
    diff_cents = int(round((100.0 - sum(rounded)) * 100))
    if rounded and diff_cents:
        index = max(range(len(rounded)), key=lambda item: rounded[item])
        rounded[index] = round(rounded[index] + diff_cents / 100, 2)
    return [f"{value:.2f}%" for value in rounded]


def shareholding_type_pie_panel(
    latest: pd.DataFrame,
    csv_path: Path,
    latest_date: pd.Timestamp,
    label_col: str | None,
    shares_col: str | None,
    ratio_col: str | None,
    level_col: str | None,
) -> dict[str, Any] | None:
    if not label_col:
        return None
    working = latest.copy()
    if level_col:
        working[level_col] = pd.to_numeric(working[level_col], errors="coerce")
    working["_bucket_range"] = working[label_col].map(parse_shareholding_bucket_range)
    working["_bucket_lower"] = working["_bucket_range"].map(lambda value: value[0])
    working["_bucket_upper"] = working["_bucket_range"].map(lambda value: value[1])
    detail = working.dropna(subset=["_bucket_lower"]).copy()
    if detail.empty:
        return None
    detail["_bucket_lower"] = detail["_bucket_lower"].astype(int)
    detail["_bucket_upper"] = detail["_bucket_upper"].where(detail["_bucket_upper"].notna(), None)
    if level_col:
        total_candidates = working[working[level_col].eq(17)]
        detail = detail[detail[level_col].ne(17)]
    else:
        total_candidates = working[working[label_col].astype(str).str.contains("\u5408\u8a08", na=False)]
    if level_col:
        detail = detail[detail[level_col].ne(16)]
    else:
        detail = detail[~detail[label_col].astype(str).str.contains("\u5dee\u7570", na=False)]
    if detail.empty:
        return None

    if shares_col:
        detail["_weight"] = numeric_series(detail, shares_col).fillna(0)
        total_shares = (
            numeric_series(total_candidates, shares_col).dropna().iloc[0]
            if not total_candidates.empty
            else detail["_weight"].sum()
        )
    elif ratio_col:
        detail["_weight"] = numeric_series(detail, ratio_col).fillna(0)
        total_shares = detail["_weight"].sum()
    else:
        return None
    total_weight = float(detail["_weight"].sum())
    if total_weight <= 0:
        return None

    close_price = latest_close_for_shareholding(csv_path, latest_date)
    bucket_ranges = [
        (int(row["_bucket_lower"]), None if pd.isna(row["_bucket_upper"]) else int(row["_bucket_upper"]))
        for _, row in detail.iterrows()
    ]
    retail_raw = 10_000_000 / close_price if close_price and close_price > 0 else 0
    strategic_raw = float(total_shares) * 0.01 if total_shares and pd.notna(total_shares) else 0
    retail_cutoff = nearest_shareholding_boundary(retail_raw, bucket_ranges)
    strategic_cutoff = max(retail_cutoff, nearest_shareholding_boundary(strategic_raw, bucket_ranges))
    retail_max_shares = max(0, retail_cutoff - 1)
    middle_max_shares = max(retail_max_shares, strategic_cutoff - 1)
    retail_max_value = retail_max_shares * close_price if close_price and math.isfinite(close_price) else None
    middle_min_value = retail_cutoff * close_price if close_price and math.isfinite(close_price) else None
    middle_max_value = middle_max_shares * close_price if close_price and math.isfinite(close_price) else None
    strategic_min_value = strategic_cutoff * close_price if close_price and math.isfinite(close_price) else None

    buckets = {
        f"\u6563\u6236(\u6301\u6709\u91d1\u984d{format_twd_wan(retail_max_value)}\u4ee5\u4e0b)": 0.0,
        f"\u5927\u578b\u6563\u6236(\u6301\u6709\u91d1\u984d{format_twd_wan(middle_max_value)}\u4ee5\u4e0b)": 0.0,
        "1%\u4ee5\u4e0a\u80a1\u6771": 0.0,
    }
    middle_label = list(buckets)[1]
    for _, row in detail.iterrows():
        lower = int(row["_bucket_lower"])
        weight = float(row["_weight"])
        if lower >= strategic_cutoff:
            buckets["1%\u4ee5\u4e0a\u80a1\u6771"] += weight
        elif lower >= retail_cutoff:
            buckets[middle_label] += weight
        else:
            first_key = next(iter(buckets))
            buckets[first_key] += weight

    labels = list(buckets)
    percentages = [value / total_weight * 100 for value in buckets.values()]
    if percentages:
        percentages[-1] += 100.0 - sum(percentages)
    close_text = format_close_price(close_price)
    note = (
        "\u9580\u6abb\u5df2\u5438\u9644\u5230\u6700\u8fd1\u7684TDCC\u6301\u80a1\u7d1a\u8ddd\uff1a"
        f"\u6536\u76e4\u50f9={close_text}\u5143\uff1b"
        f"\u6563\u6236\u4e0a\u9650={retail_max_shares:,.0f}\u80a1/{format_twd_wan(retail_max_value)}\uff1b"
        f"1%\u80a1\u672c={format_shareholding_threshold(strategic_cutoff, close_price)}"
    )
    callouts = (
        [
            f"\u6536\u76e4\u50f9 {close_text}\u5143\uff1b\u4e0a\u9650{retail_max_shares:,.0f}\u80a1",
            f"\u6301\u6709\u90e8\u4f4d\u5e02\u503c\uff1a{format_twd_wan(retail_max_value)}\u4ee5\u4e0b",
        ],
        [
            f"{retail_cutoff:,.0f}\u80a1\u5230{middle_max_shares:,.0f}\u80a1",
            f"\u6301\u6709\u90e8\u4f4d\u5e02\u503c\uff1a{format_twd_wan(middle_min_value)}\u81f3{format_twd_wan(middle_max_value)}",
        ],
        [
            f"\u6301\u80a1\u9054\u516c\u53f8\u80a1\u672c1%\uff1a{strategic_cutoff:,.0f}\u80a1\u4ee5\u4e0a",
            f"\u6301\u6709\u90e8\u4f4d\u5e02\u503c\uff1a{format_twd_wan(strategic_min_value)}\u4ee5\u4e0a",
        ],
    )
    return {
        "title": "\u6301\u80a1\u578b\u614b\u5206\u985e\u5713\u9905\u5716",
        "x": labels,
        "y": percentages,
        "text": percentage_labels_sum_100(percentages),
        "kind": "pie",
        "note": note,
        "callouts": callouts,
    }


def shareholding_viz(csv_path: Path, output_path: Path) -> None:
    df = read_csv_canonical(csv_path, dtype=str)
    df, date_col = normalize_date_column(df, ["DataDate", "\u8cc7\u6599\u65e5\u671f", "date", "Date"])
    if df.empty:
        write_empty_page(
            output_path,
            safe_title_from_path(csv_path),
            "\u6c92\u6709\u53ef\u7528\u7684\u80a1\u6b0a\u5206\u6563\u8cc7\u6599\u3002",
            csv_path,
        )
        return
    latest_date = df[date_col].max()
    latest_all = df[df[date_col].eq(latest_date)].copy()
    latest = latest_all.copy()
    level_col = next((col for col in ["\u6301\u80a1\u5206\u7d1a", "holding_level", "HoldingLevel", "level"] if col in latest.columns), None)
    label_col = next((col for col in ["\u6301\u80a1/\u55ae\u4f4d\u6578\u5206\u7d1a", "holding_level_name", "HoldingLevelName", "LevelName"] if col in latest.columns), None)
    ratio_col = next((col for col in ["\u5360\u96c6\u4fdd\u5eab\u5b58\u6578\u6bd4\u4f8b%", "holding_ratio", "HoldingRatio", "ratio"] if col in latest.columns), None)
    holders_col = next((col for col in ["\u4eba\u6578", "holders", "Holders", "holder_count"] if col in latest.columns), None)
    shares_col = next((col for col in ["\u80a1\u6578", "shares", "Shares", "share_count"] if col in latest.columns), None)
    if level_col:
        latest[level_col] = pd.to_numeric(latest[level_col], errors="coerce")
        latest = latest[latest[level_col].ne(17)].sort_values(level_col)
    labels = (
        latest[label_col].map(compact_shareholding_bucket).tolist()
        if label_col
        else latest.index.astype(str).tolist()
    )
    ratios = numeric_series(latest, ratio_col).fillna(0) if ratio_col else pd.Series([0] * len(latest))
    holders = numeric_series(latest, holders_col).fillna(0) if holders_col else pd.Series([0] * len(latest))
    title = f"{safe_title_from_path(csv_path)} \u80a1\u6b0a\u5206\u6563\u5716 {latest_date.date()}"
    panels = [
        {
            "title": "\u5404\u6301\u80a1\u7d1a\u8ddd\u5360\u96c6\u4fdd\u5eab\u5b58\u6bd4\u4f8b",
            "x": labels,
            "y": ratios.tolist(),
            "text": [f"{value:.2f}%" for value in ratios],
            "kind": "bar",
        },
        {
            "title": "\u5404\u6301\u80a1\u7d1a\u8ddd\u80a1\u6771\u4eba\u6578",
            "x": labels,
            "y": holders.tolist(),
            "text": [f"{value:,.0f}" for value in holders],
            "kind": "bar",
        },
    ]
    pie_panel = shareholding_type_pie_panel(latest_all, csv_path, latest_date, label_col, shares_col, ratio_col, level_col)
    if pie_panel:
        panels.append(pie_panel)
    write_svg_page(output_path, title, panels, csv_path)


def ex_right_dividend_viz(csv_path: Path, output_path: Path) -> None:
    price_overlay_viz(
        csv_path,
        output_path,
        metric_specs=DIVIDEND_WEBGL_METRICS,
        date_candidates=["ex_date"],
        page_suffix="價格與除權息",
        metric_control_label="除權息指標",
    )


def dividend_viz(csv_path: Path, output_path: Path) -> None:
    subdataset = csv_path.parent.name
    if subdataset == "ex_right_dividend":
        ex_right_dividend_viz(csv_path, output_path)
    else:
        raise ValueError(f"Dividend visualization is disabled for subdataset: {subdataset}")


VIZ_BY_DATASET: dict[str, Callable[[Path, Path], None]] = {
    "day_trading": day_trading_viz,
    "dividend": dividend_viz,
    "yield_pe_pb": yield_pe_pb_viz,
    "institutional": institutional_viz,
    "margin": margin_viz,
    "price": price_viz,
    "shareholding": shareholding_viz,
}

DATASET_ALIASES = {
    "yeild_pe_pb": "yield_pe_pb",
    "dividend_pe_pb": "yield_pe_pb",
}


def csv_files_for_dataset(dataset: str) -> list[Path]:
    root = DATA_ROOT / dataset
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if dataset == "dividend":
        root = root / "ex_right_dividend"

    metadata_path = DATA_ROOT / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata catalog is required: {metadata_path}")
    metadata = read_csv_canonical(metadata_path, dtype={"Code": str})
    if "Code" not in metadata.columns:
        raise ValueError(f"{metadata_path} is missing the Code column.")
    catalog_codes = set(metadata["Code"].dropna().astype(str))
    return sorted(
        csv_path
        for csv_path in root.glob("*.csv")
        if csv_path.stem.split("_", 1)[0] in catalog_codes
    )


def generate_dataset(dataset: str, limit: int | None = None, force: bool = False) -> list[VizResult]:
    dataset = DATASET_ALIASES.get(dataset, dataset)
    if dataset not in VIZ_BY_DATASET:
        raise ValueError(f"Unsupported dataset: {dataset}")
    files = csv_files_for_dataset(dataset)
    if limit is not None:
        files = files[:limit]
    renderer = VIZ_BY_DATASET[dataset]
    results = []
    for index, csv_path in enumerate(files, start=1):
        output_path = output_path_for(csv_path, dataset)
        if output_path.exists() and not force:
            if dataset != "price" or output_has_price_viz(output_path):
                results.append(VizResult(csv_path, output_path, "skipped", "exists"))
                continue
        try:
            renderer(csv_path, output_path)
            status = "written" if output_path.exists() else "failed"
            note = "" if output_path.exists() else "renderer did not create output"
        except Exception as exc:
            status = "failed"
            note = str(exc)
            write_empty_page(output_path, safe_title_from_path(csv_path), f"Visualization failed: {exc}", csv_path)
        results.append(VizResult(csv_path, output_path, status, note))
        if index % 100 == 0:
            print(f"{dataset}: processed {index}/{len(files)}")
    return results


def output_has_price_viz(output_path: Path) -> bool:
    try:
        with output_path.open("r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read(16384)
            return "WebGL price" in text or "Raw OHLC K-plot" in text or "market snapshot" in text
    except OSError:
        return False


def write_manifest(results_by_dataset: dict[str, list[VizResult]]) -> Path:
    rows = []
    for dataset, results in results_by_dataset.items():
        for result in results:
            rows.append(
                {
                    "dataset": dataset,
                    "source_path": str(result.source_path.relative_to(PROJECT_ROOT)),
                    "output_path": str(result.output_path.relative_to(PROJECT_ROOT)),
                    "status": result.status,
                    "note": result.note,
                }
            )
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_VIZ_ROOT / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data_viz outputs mirroring selected data/ folders."
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset folders to render.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset file limit.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = [DATASET_ALIASES.get(item.strip(), item.strip()) for item in args.datasets.split(",") if item.strip()]
    results_by_dataset = {}
    for dataset in datasets:
        print(f"Generating {dataset} visualizations...")
        results = generate_dataset(dataset, limit=args.limit, force=args.force)
        results_by_dataset[dataset] = results
        counts = pd.Series([result.status for result in results]).value_counts().to_dict()
        print(f"{dataset}: {counts}")
    manifest_path = write_manifest(results_by_dataset)
    print(f"Manifest: {manifest_path}")
    failed = sum(
        result.status == "failed"
        for results in results_by_dataset.values()
        for result in results
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
