"""Run a margin-financing event study across TWSE listed common stocks.

Signals are selected cross-sectionally each date from
MarginBalance20DayChangeRate.  Returns use adjusted closes and start from the
next trading day's close to avoid using a same-close fill after the signal is
known.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical
from downloader import price as price_downloader

PRICE_DIR = PROJECT_ROOT / "data" / "price"
MARGIN_DIR = PROJECT_ROOT / "data" / "margin"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "margin_event_study"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "margin_event_study"

SIGNAL_COLUMN = "MarginBalance20DayChangeRate"
GROUP_TOP = "融資大增 top 10%"
GROUP_BOTTOM = "融資大減 bottom 10%"


@dataclass
class StudyConfig:
    horizons: list[int]
    top_pct: float
    bottom_pct: float
    min_stocks_per_date: int


def parse_horizons(value: str) -> list[int]:
    horizons = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        horizon = int(text)
        if horizon <= 0:
            raise argparse.ArgumentTypeError("horizons must be positive integers")
        horizons.append(horizon)
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    return sorted(set(horizons))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Margin financing event study.")
    parser.add_argument("--horizons", type=parse_horizons, default=parse_horizons("5,10,20,60"))
    parser.add_argument("--top-pct", type=float, default=0.10)
    parser.add_argument("--bottom-pct", type=float, default=0.10)
    parser.add_argument("--min-stocks-per-date", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    return parser.parse_args()


def price_path_for_code(code: str) -> Path | None:
    matches = sorted(PRICE_DIR.glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def margin_path_for_code(code: str) -> Path | None:
    matches = sorted(MARGIN_DIR.glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def load_universe() -> pd.DataFrame:
    metadata = read_csv_canonical(METADATA_PATH, dtype={"Code": str}).fillna("")
    required = {"Code", "Name", "Type", "Market"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    mask = (
        metadata["Type"].astype(str).isin([price_downloader.COMMON_STOCK_TYPE, "STOCK"])
        & metadata["Market"].astype(str).eq(price_downloader.TWSE_MARKET)
    )
    universe = metadata.loc[mask, ["Code", "Name"]].copy()
    universe["Code"] = universe["Code"].astype(str).str.strip()
    universe["Name"] = universe["Name"].astype(str).str.strip()
    return universe[universe["Code"].ne("")].drop_duplicates("Code")


def load_stock_panel(code: str, name: str, config: StudyConfig) -> pd.DataFrame | None:
    price_path = price_path_for_code(code)
    margin_path = margin_path_for_code(code)
    if not price_path or not margin_path:
        return None

    try:
        price_df = read_csv_canonical(price_path, dtype=str, usecols=["Date", "close_adj"]).fillna("")
        margin_df = read_csv_canonical(
            margin_path,
            dtype=str,
            usecols=["Date", SIGNAL_COLUMN, "MarginCurrentBalance"],
        ).fillna("")
    except Exception as exc:
        print(f"skip {code}: {exc}")
        return None

    if price_df.empty or margin_df.empty:
        return None

    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
    price_df["close_adj"] = pd.to_numeric(price_df["close_adj"], errors="coerce")
    price_df = (
        price_df.dropna(subset=["Date", "close_adj"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    price_df = price_df[price_df["close_adj"].gt(0)].copy()
    if price_df.empty:
        return None

    for horizon in config.horizons:
        entry = price_df["close_adj"].shift(-1)
        exit_ = price_df["close_adj"].shift(-(horizon + 1))
        price_df[f"ForwardReturn{horizon}D"] = exit_ / entry - 1

    margin_df["Date"] = pd.to_datetime(margin_df["Date"], errors="coerce")
    margin_df[SIGNAL_COLUMN] = pd.to_numeric(margin_df[SIGNAL_COLUMN], errors="coerce")
    margin_df["MarginCurrentBalance"] = pd.to_numeric(margin_df["MarginCurrentBalance"], errors="coerce")
    margin_df = (
        margin_df.dropna(subset=["Date", SIGNAL_COLUMN])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if margin_df.empty:
        return None

    merged = margin_df.merge(price_df, on="Date", how="inner")
    if merged.empty:
        return None

    merged.insert(0, "Code", code)
    merged.insert(1, "Name", name)
    return merged


def assign_signal_groups(panel: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    result = panel.copy()
    result["SignalGroup"] = ""
    signal = result[SIGNAL_COLUMN].to_numpy(dtype=float)

    for _date, index_values in result.groupby("Date", sort=True).indices.items():
        idx = np.array(index_values, dtype=int)
        values = signal[idx]
        valid_mask = np.isfinite(values)
        idx = idx[valid_mask]
        values = values[valid_mask]
        if len(idx) < config.min_stocks_per_date:
            continue

        order = np.argsort(values, kind="mergesort")
        bottom_count = max(1, int(math.floor(len(idx) * config.bottom_pct)))
        top_count = max(1, int(math.floor(len(idx) * config.top_pct)))
        bottom_idx = idx[order[:bottom_count]]
        top_idx = idx[order[-top_count:]]
        result.loc[bottom_idx, "SignalGroup"] = GROUP_BOTTOM
        result.loc[top_idx, "SignalGroup"] = GROUP_TOP

    return result


def summarize_returns(panel: pd.DataFrame, config: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    daily_frames = []
    yearly_frames = []
    selected = panel[panel["SignalGroup"].ne("")].copy()

    for horizon in config.horizons:
        column = f"ForwardReturn{horizon}D"
        events = selected.dropna(subset=[column]).copy()
        if events.empty:
            continue

        daily = (
            events.groupby(["Date", "SignalGroup"], as_index=False)
            .agg(
                DailyMeanReturn=(column, "mean"),
                EventCount=("Code", "count"),
                MeanSignal=(SIGNAL_COLUMN, "mean"),
                MedianSignal=(SIGNAL_COLUMN, "median"),
            )
        )
        daily["Horizon"] = horizon
        daily_frames.append(daily)

        for group_name, group in events.groupby("SignalGroup"):
            returns = group[column].dropna()
            daily_group = daily[daily["SignalGroup"].eq(group_name)]["DailyMeanReturn"].dropna()
            summary_rows.append(
                {
                    "SignalGroup": group_name,
                    "Horizon": horizon,
                    "EventCount": int(len(returns)),
                    "SignalDateCount": int(daily_group.size),
                    "EventWeightedMeanReturn": float(returns.mean()),
                    "DateWeightedMeanReturn": float(daily_group.mean()),
                    "MedianReturn": float(returns.median()),
                    "WinRate": float((returns > 0).mean()),
                    "P10": float(returns.quantile(0.10)),
                    "P25": float(returns.quantile(0.25)),
                    "P75": float(returns.quantile(0.75)),
                    "P90": float(returns.quantile(0.90)),
                }
            )

        yearly = events.assign(Year=events["Date"].dt.year)
        yearly = (
            yearly.groupby(["Year", "SignalGroup"], as_index=False)
            .agg(MeanReturn=(column, "mean"), MedianReturn=(column, "median"), EventCount=("Code", "count"))
        )
        yearly["Horizon"] = horizon
        yearly_frames.append(yearly)

    summary = pd.DataFrame(summary_rows)
    daily_returns = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    yearly_returns = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()

    spread_rows = []
    if not daily_returns.empty:
        for horizon, data in daily_returns.groupby("Horizon"):
            pivot = data.pivot(index="Date", columns="SignalGroup", values="DailyMeanReturn")
            if {GROUP_TOP, GROUP_BOTTOM}.issubset(pivot.columns):
                spread = pivot[GROUP_TOP] - pivot[GROUP_BOTTOM]
                spread_rows.append(
                    {
                        "SignalGroup": "融資大增 - 融資大減",
                        "Horizon": int(horizon),
                        "EventCount": None,
                        "SignalDateCount": int(spread.dropna().size),
                        "EventWeightedMeanReturn": None,
                        "DateWeightedMeanReturn": float(spread.mean()),
                        "MedianReturn": float(spread.median()),
                        "WinRate": float((spread > 0).mean()),
                        "P10": float(spread.quantile(0.10)),
                        "P25": float(spread.quantile(0.25)),
                        "P75": float(spread.quantile(0.75)),
                        "P90": float(spread.quantile(0.90)),
                    }
                )
    if spread_rows:
        summary = pd.concat([summary, pd.DataFrame(spread_rows)], ignore_index=True)

    return summary, daily_returns, yearly_returns


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{float(value):,.{digits}f}"


def safe_json_records(df: pd.DataFrame) -> str:
    clean = df.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    records = clean.replace([np.inf, -np.inf], np.nan).where(pd.notna(clean), None).to_dict("records")
    return json.dumps(records, ensure_ascii=False)


def html_table(df: pd.DataFrame, columns: Iterable[str], percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    rows = []
    for row in df[list(columns)].to_dict("records"):
        cells = []
        for column in columns:
            value = row[column]
            text = pct(value) if column in percent_columns else num(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_html_report(
    summary: pd.DataFrame,
    daily_returns: pd.DataFrame,
    yearly_returns: pd.DataFrame,
    panel: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path,
    viz_dir: Path,
) -> Path:
    viz_dir.mkdir(parents=True, exist_ok=True)

    summary_for_chart = summary[summary["SignalGroup"].isin([GROUP_TOP, GROUP_BOTTOM])].copy()
    spread_for_chart = summary[summary["SignalGroup"].eq("融資大增 - 融資大減")].copy()
    display_summary = summary.copy()
    display_summary = display_summary.sort_values(["Horizon", "SignalGroup"]).reset_index(drop=True)

    start_date = panel["Date"].min().strftime("%Y-%m-%d") if not panel.empty else ""
    end_date = panel["Date"].max().strftime("%Y-%m-%d") if not panel.empty else ""
    selected_events = int(panel["SignalGroup"].ne("").sum())
    stock_count = int(panel["Code"].nunique())
    signal_dates = int(panel.loc[panel["SignalGroup"].ne(""), "Date"].nunique())

    table_columns = [
        "SignalGroup",
        "Horizon",
        "EventCount",
        "SignalDateCount",
        "DateWeightedMeanReturn",
        "EventWeightedMeanReturn",
        "MedianReturn",
        "WinRate",
        "P25",
        "P75",
    ]
    percent_columns = {
        "DateWeightedMeanReturn",
        "EventWeightedMeanReturn",
        "MedianReturn",
        "WinRate",
        "P25",
        "P75",
    }

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>融資餘額20日變化率 Event Study</title>
<style>
body {{ font-family: Arial, "Microsoft JhengHei", sans-serif; margin: 0; color: #182033; background: #f7f8fb; }}
header {{ padding: 24px 32px 18px; background: #172033; color: white; }}
h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 700; }}
.meta {{ color: #cbd5e1; font-size: 13px; }}
main {{ padding: 24px 32px 40px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin-bottom: 22px; }}
.card {{ background: white; border: 1px solid #dfe5ef; border-radius: 6px; padding: 14px 16px; }}
.card .label {{ color: #59677c; font-size: 12px; }}
.card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
section {{ background: white; border: 1px solid #dfe5ef; border-radius: 6px; margin: 16px 0; padding: 18px; }}
h2 {{ font-size: 18px; margin: 0 0 12px; }}
.chart {{ width: 100%; height: 360px; }}
.note {{ color: #59677c; font-size: 13px; line-height: 1.6; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5eaf2; padding: 8px 10px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f2f5f9; color: #334155; }}
.legend {{ display: flex; gap: 16px; font-size: 13px; margin: 8px 0 0; }}
.legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 6px; vertical-align: -1px; border-radius: 2px; }}
.top::before {{ background: #d94b4b; }}
.bottom::before {{ background: #1b8a5a; }}
.spread::before {{ background: #3157d5; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }} main {{ padding: 18px; }} }}
</style>
</head>
<body>
<header>
<h1>融資餘額20日變化率 Event Study</h1>
<div class="meta">Signal: 每日全市場 top {config.top_pct:.0%} / bottom {config.bottom_pct:.0%}；Return: t+1 adjusted close 進場，持有 N 個交易日</div>
</header>
<main>
<div class="cards">
  <div class="card"><div class="label">股票數</div><div class="value">{stock_count:,}</div></div>
  <div class="card"><div class="label">樣本日期</div><div class="value">{html.escape(start_date)} - {html.escape(end_date)}</div></div>
  <div class="card"><div class="label">Signal 日期數</div><div class="value">{signal_dates:,}</div></div>
  <div class="card"><div class="label">Signal events</div><div class="value">{selected_events:,}</div></div>
</div>

<section>
<h2>平均未來報酬：融資大增 vs 融資大減</h2>
<div id="meanChart" class="chart"></div>
<div class="legend"><span class="top">融資大增</span><span class="bottom">融資大減</span><span class="spread">大增 - 大減</span></div>
</section>

<section>
<h2>勝率比較</h2>
<div id="winChart" class="chart"></div>
</section>

<section>
<h2>20 日 forward return 的一年滾動平均</h2>
<div id="rollingChart" class="chart"></div>
<div class="note">使用每日 cohort 平均報酬再做 252 個 signal days 滾動平均；線圖用來看效果是否集中在少數年份。</div>
</section>

<section>
<h2>統計表</h2>
{html_table(display_summary, table_columns, percent_columns)}
</section>

<section>
<h2>輸出檔</h2>
<div class="note">
Summary CSV: {html.escape(str((output_dir / "summary.csv").resolve()))}<br>
Daily cohort CSV: {html.escape(str((output_dir / "daily_cohort_returns.csv").resolve()))}<br>
Yearly CSV: {html.escape(str((output_dir / "yearly_returns.csv").resolve()))}
</div>
</section>
</main>
<script>
const summary = {safe_json_records(summary_for_chart)};
const spread = {safe_json_records(spread_for_chart)};
const daily = {safe_json_records(daily_returns)};

function fmtPct(v) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  return (Number(v) * 100).toFixed(2) + "%";
}}
function svgBarChart(targetId, metric, yLabel) {{
  const root = document.getElementById(targetId);
  const horizons = Array.from(new Set(summary.map(d => d.Horizon))).sort((a,b) => a-b);
  const groups = ["{GROUP_TOP}", "{GROUP_BOTTOM}"];
  const values = [];
  for (const h of horizons) {{
    for (const g of groups) {{
      const row = summary.find(d => d.Horizon === h && d.SignalGroup === g);
      if (row) values.push(Number(row[metric]));
    }}
    const srow = spread.find(d => d.Horizon === h);
    if (srow && metric === "DateWeightedMeanReturn") values.push(Number(srow[metric]));
  }}
  const minV = Math.min(0, ...values);
  const maxV = Math.max(0, ...values);
  const pad = Math.max(0.01, (maxV - minV) * 0.15);
  const yMin = minV - pad;
  const yMax = maxV + pad;
  const width = root.clientWidth || 900;
  const height = 340;
  const left = 72, right = 24, top = 24, bottom = 48;
  const plotW = width - left - right;
  const plotH = height - top - bottom;
  const xStep = plotW / horizons.length;
  const barW = Math.min(34, xStep / 5);
  const y = v => top + (yMax - v) / (yMax - yMin) * plotH;
  const zeroY = y(0);
  let parts = [`<svg viewBox="0 0 ${{width}} ${{height}}" width="100%" height="100%">`];
  parts.push(`<line x1="${{left}}" y1="${{zeroY}}" x2="${{width-right}}" y2="${{zeroY}}" stroke="#6b7280" stroke-width="1"/>`);
  parts.push(`<text x="8" y="${{top+8}}" fill="#64748b" font-size="12">${{fmtPct(yMax)}}</text>`);
  parts.push(`<text x="8" y="${{height-bottom}}" fill="#64748b" font-size="12">${{fmtPct(yMin)}}</text>`);
  horizons.forEach((h, i) => {{
    const center = left + i * xStep + xStep / 2;
    const rows = [
      [summary.find(d => d.Horizon === h && d.SignalGroup === "{GROUP_TOP}"), "#d94b4b", -barW*1.2],
      [summary.find(d => d.Horizon === h && d.SignalGroup === "{GROUP_BOTTOM}"), "#1b8a5a", 0],
    ];
    if (metric === "DateWeightedMeanReturn") rows.push([spread.find(d => d.Horizon === h), "#3157d5", barW*1.2]);
    rows.forEach(([row, color, offset]) => {{
      if (!row) return;
      const value = Number(row[metric]);
      const yy = y(value);
      const barY = Math.min(yy, zeroY);
      const barH = Math.max(2, Math.abs(zeroY - yy));
      parts.push(`<rect x="${{center + offset - barW/2}}" y="${{barY}}" width="${{barW}}" height="${{barH}}" fill="${{color}}"/>`);
      parts.push(`<text x="${{center + offset}}" y="${{barY - 5}}" text-anchor="middle" fill="#334155" font-size="11">${{fmtPct(value)}}</text>`);
    }});
    parts.push(`<text x="${{center}}" y="${{height-16}}" text-anchor="middle" fill="#334155" font-size="12">${{h}}日</text>`);
  }});
  parts.push(`<text x="${{left}}" y="14" fill="#334155" font-size="13">${{yLabel}}</text></svg>`);
  root.innerHTML = parts.join("");
}}

function rollingMean(values, windowSize) {{
  const out = [];
  const queue = [];
  let sum = 0;
  for (const item of values) {{
    const v = Number(item.value);
    if (!Number.isFinite(v)) continue;
    queue.push(v);
    sum += v;
    if (queue.length > windowSize) sum -= queue.shift();
    out.push({{date: item.date, value: sum / queue.length}});
  }}
  return out;
}}
function svgRollingChart() {{
  const root = document.getElementById("rollingChart");
  const horizon = 20;
  const rows = daily.filter(d => d.Horizon === horizon);
  const groups = ["{GROUP_TOP}", "{GROUP_BOTTOM}"];
  const series = groups.map(group => {{
    const values = rows.filter(d => d.SignalGroup === group).sort((a,b) => String(a.Date).localeCompare(String(b.Date))).map(d => ({{date: String(d.Date).slice(0,10), value: d.DailyMeanReturn}}));
    return {{group, values: rollingMean(values, 252)}};
  }});
  const spreadRows = [];
  const byDate = new Map();
  rows.forEach(d => {{
    const key = String(d.Date).slice(0,10);
    if (!byDate.has(key)) byDate.set(key, {{}});
    byDate.get(key)[d.SignalGroup] = Number(d.DailyMeanReturn);
  }});
  Array.from(byDate.keys()).sort().forEach(date => {{
    const row = byDate.get(date);
    if (Number.isFinite(row["{GROUP_TOP}"]) && Number.isFinite(row["{GROUP_BOTTOM}"])) {{
      spreadRows.push({{date, value: row["{GROUP_TOP}"] - row["{GROUP_BOTTOM}"]}});
    }}
  }});
  series.push({{group: "融資大增 - 融資大減", values: rollingMean(spreadRows, 252)}});
  const all = series.flatMap(s => s.values);
  if (!all.length) return;
  const width = root.clientWidth || 900, height = 340;
  const left = 72, right = 24, top = 24, bottom = 48;
  const yVals = all.map(d => d.value);
  const yMin = Math.min(0, ...yVals) - 0.01;
  const yMax = Math.max(0, ...yVals) + 0.01;
  const dates = Array.from(new Set(all.map(d => d.date))).sort();
  const dateIndex = new Map(dates.map((d, i) => [d, i]));
  const x = date => left + (dateIndex.get(date) || 0) / Math.max(1, dates.length - 1) * (width - left - right);
  const y = v => top + (yMax - v) / (yMax - yMin) * (height - top - bottom);
  const colors = new Map([["{GROUP_TOP}", "#d94b4b"], ["{GROUP_BOTTOM}", "#1b8a5a"], ["融資大增 - 融資大減", "#3157d5"]]);
  let parts = [`<svg viewBox="0 0 ${{width}} ${{height}}" width="100%" height="100%">`];
  parts.push(`<line x1="${{left}}" y1="${{y(0)}}" x2="${{width-right}}" y2="${{y(0)}}" stroke="#6b7280"/>`);
  parts.push(`<text x="8" y="${{top+8}}" fill="#64748b" font-size="12">${{fmtPct(yMax)}}</text>`);
  parts.push(`<text x="8" y="${{height-bottom}}" fill="#64748b" font-size="12">${{fmtPct(yMin)}}</text>`);
  for (const s of series) {{
    const points = s.values.map(d => `${{x(d.date).toFixed(1)}},${{y(d.value).toFixed(1)}}`).join(" ");
    parts.push(`<polyline points="${{points}}" fill="none" stroke="${{colors.get(s.group)}}" stroke-width="2"/>`);
  }}
  if (dates.length) {{
    parts.push(`<text x="${{left}}" y="${{height-16}}" fill="#334155" font-size="12">${{dates[0]}}</text>`);
    parts.push(`<text x="${{width-right-72}}" y="${{height-16}}" fill="#334155" font-size="12">${{dates[dates.length-1]}}</text>`);
  }}
  parts.push(`</svg>`);
  root.innerHTML = parts.join("");
}}

svgBarChart("meanChart", "DateWeightedMeanReturn", "每日 cohort 平均報酬");
svgBarChart("winChart", "WinRate", "勝率");
svgRollingChart();
window.addEventListener("resize", () => {{
  svgBarChart("meanChart", "DateWeightedMeanReturn", "每日 cohort 平均報酬");
  svgBarChart("winChart", "WinRate", "勝率");
  svgRollingChart();
}});
</script>
</body>
</html>
"""
    report_path = viz_dir / "index.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def write_outputs(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    daily_returns: pd.DataFrame,
    yearly_returns: pd.DataFrame,
    config: StudyConfig,
    output_dir: Path,
    viz_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    daily_returns.to_csv(output_dir / "daily_cohort_returns.csv", index=False, encoding="utf-8-sig")
    yearly_returns.to_csv(output_dir / "yearly_returns.csv", index=False, encoding="utf-8-sig")

    coverage = (
        panel.groupby("Date", as_index=False)
        .agg(StockCount=("Code", "nunique"), SignalCount=(SIGNAL_COLUMN, "count"), SelectedCount=("SignalGroup", lambda x: int(x.ne("").sum())))
    )
    coverage.to_csv(output_dir / "coverage_by_date.csv", index=False, encoding="utf-8-sig")

    config_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "horizons": config.horizons,
        "top_pct": config.top_pct,
        "bottom_pct": config.bottom_pct,
        "min_stocks_per_date": config.min_stocks_per_date,
        "return_definition": "close_adj[t+1+horizon] / close_adj[t+1] - 1",
        "signal_column": SIGNAL_COLUMN,
    }
    (output_dir / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return write_html_report(summary, daily_returns, yearly_returns, panel, config, output_dir, viz_dir)


def main() -> None:
    args = parse_args()
    config = StudyConfig(
        horizons=args.horizons,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        min_stocks_per_date=args.min_stocks_per_date,
    )

    universe = load_universe()
    frames = []
    skipped = 0
    for index, row in enumerate(universe.itertuples(index=False), start=1):
        frame = load_stock_panel(str(row.Code), str(row.Name), config)
        if frame is None:
            skipped += 1
        else:
            frames.append(frame)
        if index % 100 == 0:
            print(f"loaded {index}/{len(universe)} stocks; usable={len(frames)} skipped={skipped}")

    if not frames:
        raise SystemExit("No usable stock panels found.")

    panel = pd.concat(frames, ignore_index=True)
    panel = assign_signal_groups(panel, config)
    summary, daily_returns, yearly_returns = summarize_returns(panel, config)
    report_path = write_outputs(panel, summary, daily_returns, yearly_returns, config, args.output_dir, args.viz_dir)

    print(
        json.dumps(
            {
                "universe_stocks": int(len(universe)),
                "usable_stocks": int(panel["Code"].nunique()),
                "panel_rows": int(len(panel)),
                "selected_events": int(panel["SignalGroup"].ne("").sum()),
                "summary_rows": int(len(summary)),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
