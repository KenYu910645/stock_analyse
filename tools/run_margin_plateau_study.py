"""Study whether margin-financing surges are followed by flat price action."""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical
from tools.run_margin_event_study import (
    GROUP_BOTTOM,
    GROUP_TOP,
    SIGNAL_COLUMN,
    StudyConfig,
    assign_signal_groups,
    load_universe,
    margin_path_for_code,
    price_path_for_code,
    safe_json_records,
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "margin_plateau_study"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "margin_plateau_study"


@dataclass
class PlateauConfig:
    window: int
    top_pct: float
    bottom_pct: float
    plateau_band: float
    breakout_threshold: float
    min_stocks_per_date: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Margin financing plateau event study.")
    parser.add_argument("--window", type=int, default=20, help="Future trading-day window.")
    parser.add_argument("--top-pct", type=float, default=0.10)
    parser.add_argument("--bottom-pct", type=float, default=0.10)
    parser.add_argument("--plateau-band", type=float, default=0.02)
    parser.add_argument("--breakout-threshold", type=float, default=0.05)
    parser.add_argument("--min-stocks-per-date", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    return parser.parse_args()


def load_stock_panel(code: str, name: str, config: PlateauConfig) -> pd.DataFrame | None:
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

    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
    price_df["close_adj"] = pd.to_numeric(price_df["close_adj"], errors="coerce")
    price_df = (
        price_df.dropna(subset=["Date", "close_adj"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    price_df = price_df[price_df["close_adj"].gt(0)].copy()
    if len(price_df) <= config.window + 1:
        return None

    entry = price_df["close_adj"].shift(-1)
    future_prices = pd.concat(
        [price_df["close_adj"].shift(-offset) for offset in range(1, config.window + 1)],
        axis=1,
    )
    future_returns = future_prices.divide(entry, axis=0) - 1
    price_df[f"AverageReturn{config.window}D"] = future_returns.mean(axis=1)
    price_df[f"EndReturn{config.window}D"] = future_returns.iloc[:, -1]
    price_df[f"MaxReturn{config.window}D"] = future_returns.max(axis=1)
    price_df[f"MinReturn{config.window}D"] = future_returns.min(axis=1)
    price_df[f"PositiveDayRatio{config.window}D"] = future_returns.gt(0).mean(axis=1)

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


def summarize(panel: pd.DataFrame, config: PlateauConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = panel[panel["SignalGroup"].ne("")].copy()
    window = config.window
    selected = selected.dropna(subset=[f"AverageReturn{window}D"])
    selected[f"Plateau{window}D"] = selected[f"AverageReturn{window}D"].abs().le(config.plateau_band)
    selected[f"NoBreakout{window}D"] = selected[f"MaxReturn{window}D"].le(config.breakout_threshold)

    summary_rows: list[dict[str, Any]] = []
    for group_name, group in selected.groupby("SignalGroup"):
        daily = (
            group.groupby("Date", as_index=False)
            .agg(
                AverageReturn=(f"AverageReturn{window}D", "mean"),
                EndReturn=(f"EndReturn{window}D", "mean"),
                MaxReturn=(f"MaxReturn{window}D", "mean"),
                MinReturn=(f"MinReturn{window}D", "mean"),
                PositiveDayRatio=(f"PositiveDayRatio{window}D", "mean"),
                PlateauRate=(f"Plateau{window}D", "mean"),
                NoBreakoutRate=(f"NoBreakout{window}D", "mean"),
                EventCount=("Code", "count"),
                MeanSignal=(SIGNAL_COLUMN, "mean"),
                MedianSignal=(SIGNAL_COLUMN, "median"),
            )
        )
        summary_rows.append(
            {
                "SignalGroup": group_name,
                "EventCount": int(len(group)),
                "SignalDateCount": int(daily["Date"].nunique()),
                "EventWeightedAverageReturn": float(group[f"AverageReturn{window}D"].mean()),
                "DateWeightedAverageReturn": float(daily["AverageReturn"].mean()),
                "MedianAverageReturn": float(group[f"AverageReturn{window}D"].median()),
                "EventWeightedEndReturn": float(group[f"EndReturn{window}D"].mean()),
                "DateWeightedEndReturn": float(daily["EndReturn"].mean()),
                "EventWeightedMaxReturn": float(group[f"MaxReturn{window}D"].mean()),
                "DateWeightedMaxReturn": float(daily["MaxReturn"].mean()),
                "EventWeightedMinReturn": float(group[f"MinReturn{window}D"].mean()),
                "DateWeightedMinReturn": float(daily["MinReturn"].mean()),
                "PositiveAverageRate": float(group[f"AverageReturn{window}D"].gt(0).mean()),
                "PositiveDayRatio": float(group[f"PositiveDayRatio{window}D"].mean()),
                "PlateauRate": float(group[f"Plateau{window}D"].mean()),
                "NoBreakoutRate": float(group[f"NoBreakout{window}D"].mean()),
            }
        )

    daily = (
        selected.groupby(["Date", "SignalGroup"], as_index=False)
        .agg(
            AverageReturn=(f"AverageReturn{window}D", "mean"),
            EndReturn=(f"EndReturn{window}D", "mean"),
            MaxReturn=(f"MaxReturn{window}D", "mean"),
            MinReturn=(f"MinReturn{window}D", "mean"),
            PositiveDayRatio=(f"PositiveDayRatio{window}D", "mean"),
            PlateauRate=(f"Plateau{window}D", "mean"),
            NoBreakoutRate=(f"NoBreakout{window}D", "mean"),
            EventCount=("Code", "count"),
            MeanSignal=(SIGNAL_COLUMN, "mean"),
        )
    )

    summary = pd.DataFrame(summary_rows)
    if {GROUP_TOP, GROUP_BOTTOM}.issubset(set(summary["SignalGroup"])):
        top = summary[summary["SignalGroup"].eq(GROUP_TOP)].iloc[0]
        bottom = summary[summary["SignalGroup"].eq(GROUP_BOTTOM)].iloc[0]
        spread = {
            "SignalGroup": "融資大增 - 融資大減",
            "EventCount": None,
            "SignalDateCount": None,
        }
        for column in summary.columns:
            if column in spread or column in {"EventCount", "SignalDateCount", "SignalGroup"}:
                continue
            spread[column] = float(top[column]) - float(bottom[column])
        summary = pd.concat([summary, pd.DataFrame([spread])], ignore_index=True)

    yearly = selected.assign(Year=selected["Date"].dt.year)
    yearly = (
        yearly.groupby(["Year", "SignalGroup"], as_index=False)
        .agg(
            AverageReturn=(f"AverageReturn{window}D", "mean"),
            EndReturn=(f"EndReturn{window}D", "mean"),
            MaxReturn=(f"MaxReturn{window}D", "mean"),
            PlateauRate=(f"Plateau{window}D", "mean"),
            NoBreakoutRate=(f"NoBreakout{window}D", "mean"),
            EventCount=("Code", "count"),
        )
    )
    return summary, daily, yearly


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def integer(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):,}"


def write_html_report(
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    panel: pd.DataFrame,
    config: PlateauConfig,
    output_dir: Path,
    viz_dir: Path,
) -> Path:
    viz_dir.mkdir(parents=True, exist_ok=True)
    selected_events = int(panel["SignalGroup"].ne("").sum())
    stock_count = int(panel["Code"].nunique())
    start_date = panel["Date"].min().strftime("%Y-%m-%d")
    end_date = panel["Date"].max().strftime("%Y-%m-%d")

    display_rows = []
    for row in summary.to_dict("records"):
        display_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['SignalGroup']))}</td>"
            f"<td>{integer(row.get('EventCount'))}</td>"
            f"<td>{pct(row.get('DateWeightedAverageReturn'))}</td>"
            f"<td>{pct(row.get('DateWeightedEndReturn'))}</td>"
            f"<td>{pct(row.get('DateWeightedMaxReturn'))}</td>"
            f"<td>{pct(row.get('DateWeightedMinReturn'))}</td>"
            f"<td>{pct(row.get('PlateauRate'))}</td>"
            f"<td>{pct(row.get('NoBreakoutRate'))}</td>"
            f"<td>{pct(row.get('PositiveDayRatio'))}</td>"
            "</tr>"
        )

    report = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>融資大增後20日平均報酬</title>
<style>
body {{ margin: 0; font-family: Arial, "Microsoft JhengHei", sans-serif; color: #172033; background: #f7f8fb; }}
header {{ background: #172033; color: white; padding: 24px 32px 18px; }}
h1 {{ margin: 0 0 8px; font-size: 24px; }}
.meta {{ color: #cbd5e1; font-size: 13px; line-height: 1.5; }}
main {{ padding: 24px 32px 40px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.card, section {{ background: white; border: 1px solid #dfe5ef; border-radius: 6px; }}
.card {{ padding: 14px 16px; }}
.label {{ color: #5b677a; font-size: 12px; }}
.value {{ margin-top: 5px; font-weight: 700; font-size: 21px; }}
section {{ padding: 18px; margin: 16px 0; }}
h2 {{ font-size: 18px; margin: 0 0 12px; }}
.chart {{ width: 100%; height: 340px; }}
.note {{ color: #59677c; font-size: 13px; line-height: 1.6; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5eaf2; padding: 8px 10px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f2f5f9; color: #334155; }}
.legend {{ display: flex; gap: 16px; font-size: 13px; margin-top: 8px; }}
.legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 6px; border-radius: 2px; }}
.top::before {{ background: #d94b4b; }}
.bottom::before {{ background: #1b8a5a; }}
.spread::before {{ background: #3157d5; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }} main {{ padding: 18px; }} }}
</style>
</head>
<body>
<header>
<h1>融資大增後20日平均報酬</h1>
<div class="meta">主 metric: 後 {config.window} 個交易日平均價格位置 = mean(close_adj[t+1:t+{config.window}]) / close_adj[t+1] - 1。Signal: 每日融資餘額20日變化率 top {config.top_pct:.0%} / bottom {config.bottom_pct:.0%}。</div>
</header>
<main>
<div class="cards">
  <div class="card"><div class="label">股票數</div><div class="value">{stock_count:,}</div></div>
  <div class="card"><div class="label">日期範圍</div><div class="value">{start_date} - {end_date}</div></div>
  <div class="card"><div class="label">Signal events</div><div class="value">{selected_events:,}</div></div>
  <div class="card"><div class="label">高原定義</div><div class="value">±{config.plateau_band:.0%}</div></div>
</div>

<section>
<h2>後20日平均報酬與結束報酬</h2>
<div id="returnChart" class="chart"></div>
<div class="legend"><span class="top">融資大增</span><span class="bottom">融資大減</span><span class="spread">大增 - 大減</span></div>
</section>

<section>
<h2>停在高原率 / 無明顯上攻率</h2>
<div id="plateauChart" class="chart"></div>
<div class="note">停在高原率：後20日平均報酬介於 -{config.plateau_band:.0%} 到 +{config.plateau_band:.0%}。無明顯上攻率：後20日最高報酬不超過 +{config.breakout_threshold:.0%}。</div>
</section>

<section>
<h2>每日 cohort 後20日平均報酬，252 signal-day 滾動平均</h2>
<div id="rollingChart" class="chart"></div>
</section>

<section>
<h2>統計表</h2>
<table>
<thead><tr><th>組別</th><th>Events</th><th>平均報酬</th><th>20日結束報酬</th><th>20日最高報酬</th><th>20日最低報酬</th><th>停在高原率</th><th>無明顯上攻率</th><th>正報酬天數比</th></tr></thead>
<tbody>{''.join(display_rows)}</tbody>
</table>
</section>
</main>
<script>
const summary = {safe_json_records(summary)};
const daily = {safe_json_records(daily)};
const GROUP_TOP = "{GROUP_TOP}";
const GROUP_BOTTOM = "{GROUP_BOTTOM}";
const GROUP_SPREAD = "融資大增 - 融資大減";
function fmtPct(v) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  return (Number(v) * 100).toFixed(2) + "%";
}}
function getRow(group) {{ return summary.find(d => d.SignalGroup === group); }}
function drawBars(targetId, metrics, labels) {{
  const root = document.getElementById(targetId);
  const width = root.clientWidth || 900, height = 320;
  const left = 72, right = 24, top = 22, bottom = 48;
  const rows = [getRow(GROUP_TOP), getRow(GROUP_BOTTOM), getRow(GROUP_SPREAD)];
  const colors = ["#d94b4b", "#1b8a5a", "#3157d5"];
  const values = [];
  metrics.forEach(m => rows.forEach(r => {{ if (r && Number.isFinite(Number(r[m]))) values.push(Number(r[m])); }}));
  const minV = Math.min(0, ...values), maxV = Math.max(0, ...values);
  const pad = Math.max(0.01, (maxV - minV) * 0.18);
  const yMin = minV - pad, yMax = maxV + pad;
  const y = v => top + (yMax - v) / (yMax - yMin) * (height - top - bottom);
  const zeroY = y(0);
  const groupW = (width - left - right) / metrics.length;
  const barW = Math.min(34, groupW / 5);
  let parts = [`<svg viewBox="0 0 ${{width}} ${{height}}" width="100%" height="100%">`];
  parts.push(`<line x1="${{left}}" y1="${{zeroY}}" x2="${{width-right}}" y2="${{zeroY}}" stroke="#6b7280"/>`);
  parts.push(`<text x="8" y="${{top+8}}" fill="#64748b" font-size="12">${{fmtPct(yMax)}}</text>`);
  parts.push(`<text x="8" y="${{height-bottom}}" fill="#64748b" font-size="12">${{fmtPct(yMin)}}</text>`);
  metrics.forEach((metric, i) => {{
    const center = left + groupW * i + groupW / 2;
    rows.forEach((row, j) => {{
      if (!row) return;
      const value = Number(row[metric]);
      if (!Number.isFinite(value)) return;
      const x = center + (j - 1) * barW * 1.25 - barW / 2;
      const yy = y(value), barY = Math.min(yy, zeroY), barH = Math.max(2, Math.abs(zeroY - yy));
      parts.push(`<rect x="${{x}}" y="${{barY}}" width="${{barW}}" height="${{barH}}" fill="${{colors[j]}}"/>`);
      parts.push(`<text x="${{x + barW / 2}}" y="${{barY - 5}}" text-anchor="middle" fill="#334155" font-size="11">${{fmtPct(value)}}</text>`);
    }});
    parts.push(`<text x="${{center}}" y="${{height-16}}" text-anchor="middle" fill="#334155" font-size="12">${{labels[i]}}</text>`);
  }});
  parts.push(`</svg>`);
  root.innerHTML = parts.join("");
}}
function rollingMean(values, windowSize) {{
  const out = [];
  const queue = [];
  let sum = 0;
  for (const item of values) {{
    const v = Number(item.AverageReturn);
    if (!Number.isFinite(v)) continue;
    queue.push(v); sum += v;
    if (queue.length > windowSize) sum -= queue.shift();
    out.push({{date: String(item.Date).slice(0,10), value: sum / queue.length}});
  }}
  return out;
}}
function drawRolling() {{
  const root = document.getElementById("rollingChart");
  const groups = [GROUP_TOP, GROUP_BOTTOM];
  const series = groups.map(group => ({{group, values: rollingMean(daily.filter(d => d.SignalGroup === group).sort((a,b) => String(a.Date).localeCompare(String(b.Date))), 252)}}));
  const byDate = new Map();
  daily.forEach(d => {{
    const key = String(d.Date).slice(0,10);
    if (!byDate.has(key)) byDate.set(key, {{}});
    byDate.get(key)[d.SignalGroup] = Number(d.AverageReturn);
  }});
  const spread = [];
  Array.from(byDate.keys()).sort().forEach(date => {{
    const row = byDate.get(date);
    if (Number.isFinite(row[GROUP_TOP]) && Number.isFinite(row[GROUP_BOTTOM])) spread.push({{Date: date, AverageReturn: row[GROUP_TOP] - row[GROUP_BOTTOM]}});
  }});
  series.push({{group: GROUP_SPREAD, values: rollingMean(spread, 252)}});
  const all = series.flatMap(s => s.values);
  if (!all.length) return;
  const width = root.clientWidth || 900, height = 320;
  const left = 72, right = 24, top = 22, bottom = 48;
  const vals = all.map(d => d.value);
  const yMin = Math.min(0, ...vals) - 0.01, yMax = Math.max(0, ...vals) + 0.01;
  const dates = Array.from(new Set(all.map(d => d.date))).sort();
  const dateIndex = new Map(dates.map((d,i) => [d,i]));
  const x = date => left + (dateIndex.get(date) || 0) / Math.max(1, dates.length - 1) * (width - left - right);
  const y = v => top + (yMax - v) / (yMax - yMin) * (height - top - bottom);
  const colors = new Map([[GROUP_TOP, "#d94b4b"], [GROUP_BOTTOM, "#1b8a5a"], [GROUP_SPREAD, "#3157d5"]]);
  let parts = [`<svg viewBox="0 0 ${{width}} ${{height}}" width="100%" height="100%">`];
  parts.push(`<line x1="${{left}}" y1="${{y(0)}}" x2="${{width-right}}" y2="${{y(0)}}" stroke="#6b7280"/>`);
  parts.push(`<text x="8" y="${{top+8}}" fill="#64748b" font-size="12">${{fmtPct(yMax)}}</text>`);
  parts.push(`<text x="8" y="${{height-bottom}}" fill="#64748b" font-size="12">${{fmtPct(yMin)}}</text>`);
  for (const s of series) {{
    const points = s.values.map(d => `${{x(d.date).toFixed(1)}},${{y(d.value).toFixed(1)}}`).join(" ");
    parts.push(`<polyline points="${{points}}" fill="none" stroke="${{colors.get(s.group)}}" stroke-width="2"/>`);
  }}
  parts.push(`<text x="${{left}}" y="${{height-16}}" fill="#334155" font-size="12">${{dates[0] || ""}}</text>`);
  parts.push(`<text x="${{width-right-72}}" y="${{height-16}}" fill="#334155" font-size="12">${{dates[dates.length-1] || ""}}</text>`);
  parts.push(`</svg>`);
  root.innerHTML = parts.join("");
}}
function render() {{
  drawBars("returnChart", ["DateWeightedAverageReturn", "DateWeightedEndReturn", "DateWeightedMaxReturn"], ["平均位置", "20日結束", "20日最高"]);
  drawBars("plateauChart", ["PlateauRate", "NoBreakoutRate", "PositiveDayRatio"], ["停在高原率", "無明顯上攻率", "正報酬天數比"]);
  drawRolling();
}}
render();
window.addEventListener("resize", render);
</script>
</body>
</html>
"""
    path = viz_dir / "index.html"
    path.write_text(report, encoding="utf-8")
    return path


def write_outputs(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    yearly: pd.DataFrame,
    config: PlateauConfig,
    output_dir: Path,
    viz_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "daily_cohort_metrics.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")

    coverage = (
        panel.groupby("Date", as_index=False)
        .agg(StockCount=("Code", "nunique"), SignalCount=(SIGNAL_COLUMN, "count"), SelectedCount=("SignalGroup", lambda x: int(x.ne("").sum())))
    )
    coverage.to_csv(output_dir / "coverage_by_date.csv", index=False, encoding="utf-8-sig")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "window": config.window,
        "top_pct": config.top_pct,
        "bottom_pct": config.bottom_pct,
        "plateau_band": config.plateau_band,
        "breakout_threshold": config.breakout_threshold,
        "min_stocks_per_date": config.min_stocks_per_date,
        "metric_definition": f"mean(close_adj[t+1:t+{config.window}]) / close_adj[t+1] - 1",
        "signal_column": SIGNAL_COLUMN,
    }
    (output_dir / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return write_html_report(summary, daily, panel, config, output_dir, viz_dir)


def main() -> None:
    args = parse_args()
    config = PlateauConfig(
        window=args.window,
        top_pct=args.top_pct,
        bottom_pct=args.bottom_pct,
        plateau_band=args.plateau_band,
        breakout_threshold=args.breakout_threshold,
        min_stocks_per_date=args.min_stocks_per_date,
    )
    if config.window <= 1:
        raise SystemExit("--window must be greater than 1")

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
    ranking_config = StudyConfig(
        horizons=[config.window],
        top_pct=config.top_pct,
        bottom_pct=config.bottom_pct,
        min_stocks_per_date=config.min_stocks_per_date,
    )
    panel = assign_signal_groups(panel, ranking_config)
    summary, daily, yearly = summarize(panel, config)
    report_path = write_outputs(panel, summary, daily, yearly, config, args.output_dir, args.viz_dir)
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
