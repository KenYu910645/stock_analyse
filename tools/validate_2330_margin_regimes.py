"""Validate 2330 price behavior during margin-financing regimes."""

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

from tools.analyze_2330_margin_patterns import Config as PatternConfig
from tools.analyze_2330_margin_patterns import load_panel

OUTPUT_DIR = PROJECT_ROOT / "output" / "margin_patterns" / "2330"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "margin_patterns"
SIGNAL_COLUMN = "MarginBalance20DayChangeRate"


@dataclass
class RegimeConfig:
    code: str
    window: int
    long_horizon: int
    change_top_quantile: float
    change_bottom_quantile: float
    level_high_quantile: float
    level_low_quantile: float
    flat_band: float
    output_dir: Path
    viz_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate 2330 price behavior during margin regimes.")
    parser.add_argument("--code", default="2330")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--long-horizon", type=int, default=60)
    parser.add_argument("--change-top-quantile", type=float, default=0.90)
    parser.add_argument("--change-bottom-quantile", type=float, default=0.10)
    parser.add_argument("--level-high-quantile", type=float, default=0.80)
    parser.add_argument("--level-low-quantile", type=float, default=0.20)
    parser.add_argument("--flat-band", type=float, default=0.03)
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
    if digits == 0:
        return f"{number:,.0f}"
    return f"{number:,.{digits}f}"


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
    drawdown = clean / running_high - 1
    return float(drawdown.min())


def classify_return(value: float, flat_band: float) -> str:
    if pd.isna(value):
        return ""
    if value <= -flat_band:
        return "下行"
    if value >= flat_band:
        return "上行"
    return "盤整"


def add_forward_metrics(panel: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
    panel = panel.copy()
    close = panel["close_adj"]
    panel["DailyReturn"] = close.pct_change()
    future_price_returns = pd.concat(
        [close.shift(-offset) / close - 1 for offset in range(1, config.window + 1)],
        axis=1,
    )
    future_daily_returns = pd.concat(
        [panel["DailyReturn"].shift(-offset) for offset in range(1, config.window + 1)],
        axis=1,
    )
    panel["FutureAvgReturn20DFromClose"] = future_price_returns.mean(axis=1)
    panel["FutureEndReturn20DFromClose"] = close.shift(-config.window) / close - 1
    panel["FutureEndReturn60DFromClose"] = close.shift(-config.long_horizon) / close - 1
    panel["FutureMaxReturn20DFromClose"] = future_price_returns.max(axis=1)
    panel["FutureMinReturn20DFromClose"] = future_price_returns.min(axis=1)
    panel["FutureVolatility20D"] = future_daily_returns.std(axis=1) * math.sqrt(252)
    panel["FutureAvgAbsDailyReturn20D"] = future_daily_returns.abs().mean(axis=1)
    panel["PastVolatility20D"] = panel["DailyReturn"].rolling(config.window, min_periods=10).std() * math.sqrt(252)
    return panel


def regime_masks(panel: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, pd.Series]:
    surge = panel[SIGNAL_COLUMN].ge(thresholds["change_top"])
    drop = panel[SIGNAL_COLUMN].le(thresholds["change_bottom"])
    high = panel["MarginCurrentBalance"].ge(thresholds["level_high"])
    low = panel["MarginCurrentBalance"].le(thresholds["level_low"])
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


def summarize_daily(panel: pd.DataFrame, masks: dict[str, pd.Series], config: RegimeConfig) -> pd.DataFrame:
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


def interval_records(panel: pd.DataFrame, masks: dict[str, pd.Series], config: RegimeConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in ["融資大漲", "融資大跌", "融資高水位", "融資低水位", "融資大漲且高水位"]:
        mask = masks[name]
        for ordinal, (start, end) in enumerate(contiguous_ranges(mask), start=1):
            block = panel.iloc[start : end + 1].copy()
            if block.empty:
                continue
            start_close = float(block["close_adj"].iloc[0])
            end_close = float(block["close_adj"].iloc[-1])
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
                    "開始復權收盤": start_close,
                    "結束復權收盤": end_close,
                    "區間報酬": interval_return,
                    "區間分類": classify_return(interval_return, config.flat_band),
                    "區間最大回撤": max_drawdown(block["close_adj"]),
                    "區間年化波動": vol,
                    "區間平均日報酬": float(daily_return.mean()) if len(daily_return) else float("nan"),
                    "開始日後20日平均價格報酬": json_float(panel.at[start, "FutureAvgReturn20DFromClose"]),
                    "開始日後20日終點報酬": json_float(panel.at[start, "FutureEndReturn20DFromClose"]),
                    "結束日後20日終點報酬": json_float(panel.at[end, "FutureEndReturn20DFromClose"]),
                    "結束日後60日終點報酬": json_float(panel.at[end, "FutureEndReturn60DFromClose"]),
                    "平均融資20日變化率": float(block[SIGNAL_COLUMN].mean()),
                    "平均融資餘額": float(block["MarginCurrentBalance"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_intervals(intervals: pd.DataFrame, config: RegimeConfig, *, min_days: int = 1) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = intervals[intervals["交易日數"].ge(min_days)].copy()
    if data.empty:
        return pd.DataFrame()
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
                "下行區間率": float(group["區間分類"].eq("下行").mean()),
                "盤整區間率": float(group["區間分類"].eq("盤整").mean()),
                "上行區間率": float(group["區間分類"].eq("上行").mean()),
                "下行或盤整區間率": float(group["區間分類"].isin(["下行", "盤整"]).mean()),
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
        "區間報酬",
        "區間最大回撤",
        "區間年化波動",
        "區間平均日報酬",
        "開始日後20日平均價格報酬",
        "開始日後20日終點報酬",
        "結束日後20日終點報酬",
        "結束日後60日終點報酬",
        "平均融資20日變化率",
    }


def render_table(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if df.empty:
        return "<div class=\"note\">沒有資料</div>"
    columns = columns or df.columns.tolist()
    data = df[columns].head(max_rows) if max_rows else df[columns]
    pct_cols = pct_columns()
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for record in data.to_dict("records"):
        cells = []
        for column in columns:
            value = record[column]
            if column in pct_cols:
                text = fmt_pct(value)
            elif isinstance(value, (int, np.integer)):
                text = f"{int(value):,}"
            elif isinstance(value, (float, np.floating)):
                text = fmt_num(value, 2)
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def svg_grouped_bar(df: pd.DataFrame, value_col: str, title: str, *, pct: bool = True, width: int = 980, height: int = 330) -> str:
    data = df.copy()
    if data.empty or value_col not in data:
        return "<div class=\"note\">沒有資料</div>"
    left, right, top, bottom = 82, 24, 42, 78
    labels = data["狀態"].astype(str).tolist()
    values = data[value_col].astype(float).fillna(0).tolist()
    low = min(0.0, min(values))
    high = max(0.0, max(values))
    if math.isclose(low, high):
        low -= 0.01
        high += 0.01
    pad = (high - low) * 0.18
    low -= pad
    high += pad
    plot_w = width - left - right
    plot_h = height - top - bottom
    step = plot_w / max(1, len(values))
    bar_w = max(18, min(54, step * 0.48))

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_h

    zero_y = y(0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">',
        f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" class="axis"/>',
        f'<text x="10" y="{top+5}" class="tick">{fmt_pct(high, 1) if pct else fmt_num(high, 1)}</text>',
        f'<text x="10" y="{height-bottom}" class="tick">{fmt_pct(low, 1) if pct else fmt_num(low, 1)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * step + step / 2 - bar_w / 2
        y_value = y(value)
        bar_y = min(y_value, zero_y)
        bar_h = max(2, abs(zero_y - y_value))
        color = "#d94b4b" if value >= 0 else "#1b8a5a"
        parts.append(f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.86"/>')
        label_text = fmt_pct(value, 1) if pct else fmt_num(value, 1)
        text_y = bar_y - 6 if value >= 0 else bar_y + bar_h + 14
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{text_y:.1f}" text-anchor="middle" class="bar-label">{html.escape(label_text)}</text>')
        parts.append(f'<text transform="translate({x + bar_w/2:.1f},{height-18}) rotate(-32)" text-anchor="end" class="tick">{html.escape(label[:12])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_stacked_interval(summary: pd.DataFrame, width: int = 980, height: int = 320) -> str:
    data = summary.copy()
    if data.empty:
        return "<div class=\"note\">沒有資料</div>"
    left, right, top, bottom = 138, 30, 42, 44
    row_h = (height - top - bottom) / max(1, len(data))
    plot_w = width - left - right
    colors = {"下行區間率": "#16a34a", "盤整區間率": "#94a3b8", "上行區間率": "#dc2626"}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">',
        f'<text x="{left}" y="24" class="chart-title">連續區間分類比例：下行 / 盤整 / 上行</text>',
    ]
    for row_index, row in enumerate(data.itertuples(index=False)):
        y = top + row_index * row_h + row_h * 0.22
        x = left
        name = getattr(row, "狀態")
        parts.append(f'<text x="10" y="{y + row_h*0.35:.1f}" class="tick">{html.escape(str(name))}</text>')
        for col in ["下行區間率", "盤整區間率", "上行區間率"]:
            value = float(getattr(row, col))
            w = value * plot_w
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{row_h*0.56:.1f}" fill="{colors[col]}" opacity="0.82"/>')
            if w > 45:
                parts.append(f'<text x="{x + w/2:.1f}" y="{y + row_h*0.36:.1f}" text-anchor="middle" class="bar-label light">{fmt_pct(value, 0)}</text>')
            x += w
    legend_x = left
    for label, color in [("下行", colors["下行區間率"]), ("盤整", colors["盤整區間率"]), ("上行", colors["上行區間率"])]:
        parts.append(f'<rect x="{legend_x}" y="{height-22}" width="12" height="12" fill="{color}" opacity="0.82"/>')
        parts.append(f'<text x="{legend_x+18}" y="{height-12}" class="legend">{label}</text>')
        legend_x += 76
    parts.append("</svg>")
    return "".join(parts)


def build_findings(daily: pd.DataFrame, intervals5: pd.DataFrame) -> list[str]:
    def row(df: pd.DataFrame, name: str) -> pd.Series:
        return df[df["狀態"].eq(name)].iloc[0]

    all_daily = row(daily, "全樣本")
    surge_daily = row(daily, "融資大漲")
    high_daily = row(daily, "融資高水位")
    low_daily = row(daily, "融資低水位")
    surge_int = row(intervals5, "融資大漲") if "融資大漲" in set(intervals5["狀態"]) else None
    high_int = row(intervals5, "融資高水位") if "融資高水位" in set(intervals5["狀態"]) else None
    low_int = row(intervals5, "融資低水位") if "融資低水位" in set(intervals5["狀態"]) else None
    findings = [
        (
            "融資大漲後的平均表現沒有明顯輸給全樣本："
            f"後20日平均價格報酬 {fmt_pct(surge_daily['後20日平均價格報酬'])}，"
            f"全樣本 {fmt_pct(all_daily['後20日平均價格報酬'])}。"
        ),
        (
            "融資大漲確實伴隨較高波動："
            f"後20日年化波動 {fmt_pct(surge_daily['後20日年化波動'])}，"
            f"全樣本 {fmt_pct(all_daily['後20日年化波動'])}。"
        ),
        (
            "融資高水位比較接近你的直覺："
            f"後20日平均價格報酬 {fmt_pct(high_daily['後20日平均價格報酬'])}，"
            f"低於全樣本 {fmt_pct(all_daily['後20日平均價格報酬'])}。"
        ),
        (
            "融資低水位本身不是立即上漲訊號："
            f"後20日平均價格報酬 {fmt_pct(low_daily['後20日平均價格報酬'])}，"
            f"後20日正報酬率 {fmt_pct(low_daily['後20日正報酬率'])}。"
        ),
    ]
    if surge_int is not None:
        findings.append(
            "只看連續區間且至少5日，融資大漲區間下行或盤整占 "
            f"{fmt_pct(surge_int['下行或盤整區間率'])}，上行占 {fmt_pct(surge_int['上行區間率'])}。"
        )
    if high_int is not None:
        findings.append(
            "高水位長區間的區間平均報酬為 "
            f"{fmt_pct(high_int['區間平均報酬'])}，下行或盤整占 {fmt_pct(high_int['下行或盤整區間率'])}。"
        )
    if low_int is not None:
        findings.append(
            "低水位長區間平均報酬為 "
            f"{fmt_pct(low_int['區間平均報酬'])}，上行區間占 {fmt_pct(low_int['上行區間率'])}。"
        )
    return findings


def build_report(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    interval_summary: pd.DataFrame,
    interval_summary5: pd.DataFrame,
    intervals: pd.DataFrame,
    thresholds: dict[str, float],
    config: RegimeConfig,
    price_path: Path,
    margin_path: Path,
) -> str:
    start_date = panel["Date"].min().strftime("%Y-%m-%d")
    end_date = panel["Date"].max().strftime("%Y-%m-%d")
    findings = build_findings(daily, interval_summary5)
    daily_cols = [
        "狀態",
        "日數",
        "後20日平均價格報酬",
        "後20日終點報酬",
        "後60日終點報酬",
        "後20日正報酬率",
        "後20日盤整率",
        "後20日年化波動",
        "當下前20日報酬",
        "前20日下跌占比",
    ]
    summary_cols = [
        "狀態",
        "區間數",
        "總交易日數",
        "平均區間日數",
        "區間平均報酬",
        "下行區間率",
        "盤整區間率",
        "上行區間率",
        "下行或盤整區間率",
        "平均最大回撤",
        "平均區間波動",
        "結束日後20日報酬",
    ]
    interval_cols = [
        "狀態",
        "開始日",
        "結束日",
        "交易日數",
        "區間報酬",
        "區間分類",
        "區間最大回撤",
        "結束日後20日終點報酬",
        "平均融資20日變化率",
        "平均融資餘額",
    ]
    long_intervals = intervals[intervals["交易日數"].ge(10)].sort_values(["狀態", "開始日"])
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>2330 融資狀態與股價特性驗證</title>
<style>
body {{ margin: 0; background: #f7f8fb; color: #172033; font-family: Arial, "Microsoft JhengHei", sans-serif; }}
header {{ background: #172033; color: white; padding: 24px 32px 18px; }}
h1 {{ margin: 0 0 8px; font-size: 24px; }}
.meta {{ color: #cbd5e1; font-size: 13px; line-height: 1.55; }}
main {{ padding: 24px 32px 42px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.card, section {{ background: white; border: 1px solid #dfe5ef; border-radius: 6px; }}
.card {{ padding: 14px 16px; }}
.label {{ color: #59677c; font-size: 12px; }}
.value {{ margin-top: 5px; font-weight: 700; font-size: 20px; }}
section {{ padding: 18px; margin: 16px 0; overflow-x: auto; }}
h2 {{ margin: 0 0 12px; font-size: 18px; }}
.note, li {{ color: #4b5870; font-size: 14px; line-height: 1.65; }}
svg {{ display: block; max-width: 100%; }}
.axis {{ stroke: #94a3b8; stroke-width: 1; }}
.tick {{ fill: #59677c; font-size: 12px; }}
.legend {{ fill: #334155; font-size: 12px; }}
.chart-title {{ fill: #243044; font-size: 14px; font-weight: 700; }}
.bar-label {{ fill: #334155; font-size: 11px; }}
.bar-label.light {{ fill: white; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
th, td {{ border-bottom: 1px solid #e5eaf2; padding: 7px 8px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child, th:nth-child(6), td:nth-child(6) {{ text-align: left; }}
th {{ background: #f2f5f9; color: #334155; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 960px) {{ .cards, .grid2 {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} }}
</style>
</head>
<body>
<header>
<h1>2330 融資狀態與股價特性驗證</h1>
<div class="meta">資料：{html.escape(str(price_path.relative_to(PROJECT_ROOT)))} + {html.escape(str(margin_path.relative_to(PROJECT_ROOT)))}。區間 {start_date} 到 {end_date}。報酬使用復權收盤價 close_adj。</div>
</header>
<main>
<div class="cards">
  <div class="card"><div class="label">融資大漲門檻</div><div class="value">>= {fmt_pct(thresholds["change_top"])}</div></div>
  <div class="card"><div class="label">融資大跌門檻</div><div class="value">&lt;= {fmt_pct(thresholds["change_bottom"])}</div></div>
  <div class="card"><div class="label">融資高水位門檻</div><div class="value">>= {fmt_num(thresholds["level_high"], 0)}</div></div>
  <div class="card"><div class="label">融資低水位門檻</div><div class="value">&lt;= {fmt_num(thresholds["level_low"], 0)}</div></div>
</div>

<section>
<h2>結論摘要</h2>
<ul>{"".join(f"<li>{html.escape(item)}</li>" for item in findings)}</ul>
<div class="note">判斷「下行/盤整/上行」時，用整段區間報酬小於 -{fmt_pct(config.flat_band)}、介於正負 {fmt_pct(config.flat_band)}、大於 +{fmt_pct(config.flat_band)} 分類。下面同時列出所有區間與至少5日的較長區間，避免單日雜訊主導結論。</div>
</section>

<div class="grid2">
<section>
{svg_grouped_bar(daily[daily["狀態"].isin(["全樣本", "融資大漲", "融資高水位", "融資低水位"])], "後20日平均價格報酬", "每日狀態：後20日平均價格報酬")}
</section>
<section>
{svg_grouped_bar(daily[daily["狀態"].isin(["全樣本", "融資大漲", "融資高水位", "融資低水位"])], "後20日年化波動", "每日狀態：後20日年化波動")}
</section>
</div>

<section>
{svg_stacked_interval(interval_summary5)}
</section>

<section>
<h2>每日狀態統計</h2>
{render_table(daily, daily_cols)}
</section>

<section>
<h2>連續區間彙總：全部區間</h2>
{render_table(interval_summary, summary_cols)}
</section>

<section>
<h2>連續區間彙總：至少5個交易日</h2>
{render_table(interval_summary5, summary_cols)}
</section>

<section>
<h2>較長連續區間明細：至少10個交易日</h2>
{render_table(long_intervals, interval_cols, max_rows=80)}
</section>
</main>
</body>
</html>
"""


def write_outputs(
    panel: pd.DataFrame,
    daily: pd.DataFrame,
    interval_summary: pd.DataFrame,
    interval_summary5: pd.DataFrame,
    intervals: pd.DataFrame,
    thresholds: dict[str, float],
    config: RegimeConfig,
    price_path: Path,
    margin_path: Path,
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.viz_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(config.output_dir / "regime_daily_summary.csv", index=False, encoding="utf-8-sig")
    interval_summary.to_csv(config.output_dir / "regime_interval_summary_all.csv", index=False, encoding="utf-8-sig")
    interval_summary5.to_csv(config.output_dir / "regime_interval_summary_min5.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(config.output_dir / "regime_intervals.csv", index=False, encoding="utf-8-sig")
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "code": config.code,
        "rows": int(len(panel)),
        "start": panel["Date"].min().strftime("%Y-%m-%d"),
        "end": panel["Date"].max().strftime("%Y-%m-%d"),
        "flat_band": config.flat_band,
        "thresholds": thresholds,
        "price_path": str(price_path.relative_to(PROJECT_ROOT)),
        "margin_path": str(margin_path.relative_to(PROJECT_ROOT)),
    }
    (config.output_dir / "regime_validation_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(panel, daily, interval_summary, interval_summary5, intervals, thresholds, config, price_path, margin_path)
    report_path = config.viz_dir / f"{config.code}_margin_regime_validation.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    config = RegimeConfig(
        code=str(args.code),
        window=args.window,
        long_horizon=args.long_horizon,
        change_top_quantile=args.change_top_quantile,
        change_bottom_quantile=args.change_bottom_quantile,
        level_high_quantile=args.level_high_quantile,
        level_low_quantile=args.level_low_quantile,
        flat_band=args.flat_band,
        output_dir=args.output_dir,
        viz_dir=args.viz_dir,
    )
    pattern_config = PatternConfig(
        code=config.code,
        window=config.window,
        top_quantile=config.change_top_quantile,
        bottom_quantile=config.change_bottom_quantile,
        near_high_band=0.05,
        breakout_threshold=0.03,
        plateau_band=0.02,
        output_dir=config.output_dir,
        viz_dir=config.viz_dir,
    )
    panel, price_path, margin_path = load_panel(pattern_config)
    panel = add_forward_metrics(panel.reset_index(drop=True), config)
    thresholds = {
        "change_top": float(panel[SIGNAL_COLUMN].quantile(config.change_top_quantile)),
        "change_bottom": float(panel[SIGNAL_COLUMN].quantile(config.change_bottom_quantile)),
        "level_high": float(panel["MarginCurrentBalance"].quantile(config.level_high_quantile)),
        "level_low": float(panel["MarginCurrentBalance"].quantile(config.level_low_quantile)),
    }
    masks = regime_masks(panel, thresholds)
    daily = summarize_daily(panel, masks, config)
    intervals = interval_records(panel, masks, config)
    interval_summary = summarize_intervals(intervals, config, min_days=1)
    interval_summary5 = summarize_intervals(intervals, config, min_days=5)
    report_path = write_outputs(
        panel,
        daily,
        interval_summary,
        interval_summary5,
        intervals,
        thresholds,
        config,
        price_path,
        margin_path,
    )
    print(
        json.dumps(
            {
                "code": config.code,
                "rows": int(len(panel)),
                "start": panel["Date"].min().strftime("%Y-%m-%d"),
                "end": panel["Date"].max().strftime("%Y-%m-%d"),
                "daily_rows": int(len(daily)),
                "interval_rows": int(len(intervals)),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
