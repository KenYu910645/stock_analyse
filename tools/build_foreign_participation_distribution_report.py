"""Build a foreign-investor participation concentration report."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_VIZ_ROOT = PROJECT_ROOT / "data_viz" / "institutional_participation"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "institutional_participation"
LATEST_SUMMARY_PATH = OUTPUT_ROOT / "latest_summary.csv"
MARKET_SUMMARY_PATH = OUTPUT_ROOT / "market_daily_summary.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"

FOREIGN = "\u5916\u8cc7"
FOREIGN_PARTICIPATION = "\u5916\u8cc7\u6210\u4ea4\u53c3\u8207\u7387"
INDUSTRY_GROUP = "\u7522\u696d\u7fa4\u7d44"

TECH_INDUSTRIES = {
    "\u534a\u5c0e\u9ad4\u696d",
    "\u5149\u96fb\u696d",
    "\u96fb\u8166\u53ca\u9031\u908a\u8a2d\u5099\u696d",
    "\u96fb\u5b50\u96f6\u7d44\u4ef6\u696d",
    "\u5176\u4ed6\u96fb\u5b50\u696d",
    "\u96fb\u5b50\u901a\u8def\u696d",
    "\u901a\u4fe1\u7db2\u8def\u696d",
    "\u8cc7\u8a0a\u670d\u52d9\u696d",
    "\u6578\u4f4d\u96f2\u7aef",
}

PETROCHEMICAL_INDUSTRIES = {
    "\u5851\u81a0\u5de5\u696d",
    "\u5316\u5b78\u5de5\u696d",
    "\u6cb9\u96fb\u71c3\u6c23\u696d",
    "\u6a61\u81a0\u5de5\u696d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the latest foreign participation concentration report."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Optional ISO date to analyze; defaults to the latest market summary date.",
    )
    return parser.parse_args()


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def number(value: float | int | None, digits: int = 0) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):,.{digits}f}"


def gini(values: Iterable[float]) -> float:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)) and value >= 0)
    if not clean:
        return 0.0
    total = sum(clean)
    if total <= 0:
        return 0.0
    weighted_sum = sum((index + 1) * value for index, value in enumerate(clean))
    count = len(clean)
    return (2 * weighted_sum) / (count * total) - (count + 1) / count


def stocks_to_reach_share(distribution: pd.DataFrame, target_share: float) -> int:
    reached = distribution["CumulativeForeignVolumeShare"].ge(target_share)
    if not reached.any():
        return len(distribution)
    return int(reached.idxmax()) + 1


def load_distribution(target_date: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    latest = pd.read_csv(LATEST_SUMMARY_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    market = pd.read_csv(MARKET_SUMMARY_PATH, encoding="utf-8-sig")

    if target_date is None:
        target_date = str(market["Date"].iloc[-1])

    market_rows = market[market["Date"].astype(str).eq(target_date)]
    if market_rows.empty:
        raise ValueError(f"date_not_found_in_market_summary:{target_date}")
    market_row = market_rows.iloc[-1]

    stocks = latest[latest["Date"].astype(str).eq(target_date)].copy()
    if stocks.empty:
        raise ValueError(f"date_not_found_in_latest_summary:{target_date}")
    if FOREIGN_PARTICIPATION not in stocks.columns:
        raise ValueError(f"missing_column:{FOREIGN_PARTICIPATION}")

    stocks["Volume"] = pd.to_numeric(stocks["Volume"], errors="coerce")
    stocks["ForeignParticipation"] = pd.to_numeric(stocks[FOREIGN_PARTICIPATION], errors="coerce")
    stocks = stocks.dropna(subset=["Volume", "ForeignParticipation"])
    stocks = stocks[stocks["Volume"].gt(0)]
    stocks["ForeignVolume"] = stocks["Volume"] * stocks["ForeignParticipation"]
    stocks = stocks[stocks["ForeignVolume"].ge(0)]
    stocks = stocks.sort_values(["ForeignVolume", "Code"], ascending=[False, True]).reset_index(drop=True)

    total_foreign_volume = float(stocks["ForeignVolume"].sum())
    if total_foreign_volume <= 0:
        raise ValueError("foreign_volume_total_is_zero")

    stocks["Rank"] = range(1, len(stocks) + 1)
    stocks["ForeignVolumeShare"] = stocks["ForeignVolume"] / total_foreign_volume
    stocks["CumulativeForeignVolumeShare"] = stocks["ForeignVolumeShare"].cumsum()
    return stocks, market_row


def build_bins(distribution: pd.DataFrame) -> pd.DataFrame:
    bins = [
        (0.00, 0.05, "0-5%"),
        (0.05, 0.10, "5-10%"),
        (0.10, 0.20, "10-20%"),
        (0.20, 0.30, "20-30%"),
        (0.30, 0.40, "30-40%"),
        (0.40, 0.50, "40-50%"),
        (0.50, 0.60, "50-60%"),
        (0.60, 0.80, "60-80%"),
        (0.80, 1.01, "80-100%"),
        (1.01, float("inf"), "100%+"),
    ]
    rows: list[dict[str, float | int | str]] = []
    total_stocks = len(distribution)
    total_foreign_volume = float(distribution["ForeignVolume"].sum())
    for lower, upper, label in bins:
        if math.isinf(upper):
            mask = distribution["ForeignParticipation"].ge(lower)
        else:
            mask = distribution["ForeignParticipation"].ge(lower) & distribution["ForeignParticipation"].lt(upper)
        subset = distribution[mask]
        foreign_volume = float(subset["ForeignVolume"].sum())
        rows.append(
            {
                "Bin": label,
                "LowerBound": lower,
                "UpperBound": upper,
                "StockCount": int(len(subset)),
                "StockShare": len(subset) / total_stocks if total_stocks else 0,
                "ForeignVolume": foreign_volume,
                "ForeignVolumeShare": foreign_volume / total_foreign_volume if total_foreign_volume else 0,
            }
        )
    return pd.DataFrame(rows)


def broad_category(industry: str) -> str:
    if industry in TECH_INDUSTRIES:
        return "\u96fb\u5b50\u79d1\u6280"
    if industry == "\u91d1\u878d\u4fdd\u96aa\u696d":
        return "\u91d1\u878d"
    if industry == "\u92fc\u9435\u5de5\u696d":
        return "\u92fc\u9435"
    if industry in PETROCHEMICAL_INDUSTRIES:
        return "\u77f3\u5316/\u5851\u5316"
    if industry == "\u822a\u904b\u696d":
        return "\u822a\u904b"
    return "\u5176\u4ed6"


def add_industry_columns(distribution: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    columns = ["Code", INDUSTRY_GROUP]
    enriched = distribution.merge(metadata[columns], on="Code", how="left")
    enriched[INDUSTRY_GROUP] = enriched[INDUSTRY_GROUP].fillna("\u672a\u5206\u985e")
    enriched["BroadCategory"] = enriched[INDUSTRY_GROUP].map(broad_category)
    return enriched


def summarize_group(distribution: pd.DataFrame, group_column: str, all_foreign_volume: float) -> pd.DataFrame:
    top_80_foreign_volume = float(distribution["ForeignVolume"].sum())
    summary = (
        distribution.groupby(group_column, dropna=False)
        .agg(
            StockCount=("Code", "count"),
            ForeignVolume=("ForeignVolume", "sum"),
            AvgForeignParticipation=("ForeignParticipation", "mean"),
        )
        .reset_index()
        .sort_values("ForeignVolume", ascending=False)
    )
    summary["ShareOfTop80ForeignVolume"] = summary["ForeignVolume"] / top_80_foreign_volume
    summary["ShareOfAllForeignVolume"] = summary["ForeignVolume"] / all_foreign_volume
    top_stocks: dict[str, str] = {}
    for group_value, subset in distribution.groupby(group_column, dropna=False):
        top_stocks[str(group_value)] = "\u3001".join(
            f"{row.Code} {row.Name}" for row in subset.sort_values("ForeignVolume", ascending=False).head(8).itertuples(index=False)
        )
    summary["TopStocks"] = summary[group_column].astype(str).map(top_stocks).fillna("")
    return summary


def line_chart_svg(distribution: pd.DataFrame) -> str:
    width = 820
    height = 360
    left = 58
    right = 18
    top = 18
    bottom = 44
    chart_width = width - left - right
    chart_height = height - top - bottom
    count = max(len(distribution), 1)

    points = []
    for row in distribution.itertuples(index=False):
        x = left + (float(row.Rank) / count) * chart_width
        y = top + (1 - float(row.CumulativeForeignVolumeShare)) * chart_height
        points.append(f"{x:.1f},{y:.1f}")
    curve = " ".join(points)
    equal_line = f"{left},{top + chart_height} {left + chart_width},{top}"

    grid_lines = []
    labels = []
    for share in [0.25, 0.50, 0.75, 1.00]:
        y = top + (1 - share) * chart_height
        grid_lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}" class="grid"/>')
        labels.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{pct(share, 0)}</text>')
    for rank_share in [0.25, 0.50, 0.75, 1.00]:
        x = left + rank_share * chart_width
        grid_lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_height}" class="grid"/>')
        labels.append(f'<text x="{x:.1f}" y="{height - 16}" text-anchor="middle">{pct(rank_share, 0)}</text>')

    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="\u5916\u8cc7\u6210\u4ea4\u91cf\u7d2f\u7a4d\u96c6\u4e2d\u66f2\u7dda">
  <style>
    .axis {{ stroke: #334155; stroke-width: 1; }}
    .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    .equal {{ fill: none; stroke: #94a3b8; stroke-width: 2; stroke-dasharray: 5 5; }}
    .curve {{ fill: none; stroke: #2563eb; stroke-width: 3; }}
    text {{ fill: #475569; font-size: 12px; font-family: "Microsoft JhengHei", Arial, sans-serif; }}
  </style>
  {''.join(grid_lines)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>
  <line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" class="axis"/>
  <polyline points="{equal_line}" class="equal"/>
  <polyline points="{curve}" class="curve"/>
  {''.join(labels)}
  <text x="{left + chart_width / 2}" y="{height - 2}" text-anchor="middle">\u6309\u5916\u8cc7\u6210\u4ea4\u91cf\u6392\u540d\u7684\u7d2f\u7a4d\u80a1\u7968\u6bd4\u4f8b</text>
  <text x="14" y="{top + chart_height / 2}" transform="rotate(-90 14 {top + chart_height / 2})" text-anchor="middle">\u7d2f\u7a4d\u5916\u8cc7\u6210\u4ea4\u91cf\u6bd4\u91cd</text>
</svg>
"""


def histogram_svg(bins: pd.DataFrame) -> str:
    width = 820
    height = 330
    left = 54
    right = 18
    top = 18
    bottom = 68
    chart_width = width - left - right
    chart_height = height - top - bottom
    max_count = max(int(bins["StockCount"].max()), 1)
    bar_gap = 10
    bar_width = (chart_width - bar_gap * (len(bins) - 1)) / len(bins)
    bars = []
    for index, row in bins.iterrows():
        count = int(row["StockCount"])
        bar_height = (count / max_count) * chart_height
        x = left + index * (bar_width + bar_gap)
        y = top + chart_height - bar_height
        bars.append(
            f"""
<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="#0f766e"/>
<text x="{x + bar_width / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle">{count}</text>
<text x="{x + bar_width / 2:.1f}" y="{height - 30}" text-anchor="middle">{html.escape(str(row["Bin"]))}</text>
"""
        )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="\u5916\u8cc7\u53c3\u8207\u7387\u5206\u5e03\u76f4\u65b9\u5716">
  <style>
    .axis {{ stroke: #334155; stroke-width: 1; }}
    .grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    text {{ fill: #475569; font-size: 12px; font-family: "Microsoft JhengHei", Arial, sans-serif; }}
  </style>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>
  <line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" class="axis"/>
  {''.join(bars)}
  <text x="{left + chart_width / 2}" y="{height - 5}" text-anchor="middle">\u55ae\u6a94\u80a1\u7968\u5916\u8cc7\u6210\u4ea4\u53c3\u8207\u7387\u5340\u9593</text>
  <text x="14" y="{top + chart_height / 2}" transform="rotate(-90 14 {top + chart_height / 2})" text-anchor="middle">\u80a1\u7968\u6a94\u6578</text>
</svg>
"""


def table_rows(distribution: pd.DataFrame, limit: int | None = 20) -> str:
    rows = []
    table_distribution = distribution if limit is None else distribution.head(limit)
    for row in table_distribution.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{int(row.Rank)}</td>"
            f"<td>{html.escape(str(row.Code))}</td>"
            f"<td>{html.escape(str(row.Name))}</td>"
            f"<td>{pct(float(row.ForeignParticipation))}</td>"
            f"<td>{number(float(row.ForeignVolume))}</td>"
            f"<td>{pct(float(row.ForeignVolumeShare))}</td>"
            f"<td>{pct(float(row.CumulativeForeignVolumeShare))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def summary_table_rows(summary: pd.DataFrame, group_column: str) -> str:
    rows = []
    for row in summary.itertuples(index=False):
        group_value = getattr(row, group_column)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(group_value))}</td>"
            f"<td>{int(row.StockCount):,}</td>"
            f"<td>{pct(float(row.ShareOfAllForeignVolume))}</td>"
            f"<td>{pct(float(row.ShareOfTop80ForeignVolume))}</td>"
            f"<td>{pct(float(row.AvgForeignParticipation))}</td>"
            f"<td>{html.escape(str(row.TopStocks))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def metric_cards(metrics: dict[str, float | int | str]) -> str:
    card_specs = [
        ("\u5e02\u5834\u65e5\u671f", str(metrics["date"])),
        ("\u6a23\u672c\u6a94\u6578", f"{int(metrics['stock_count']):,}"),
        ("\u5e02\u5834\u5916\u8cc7\u53c3\u8207\u7387", pct(float(metrics["market_foreign_participation"]))),
        ("\u524d10\u6a94\u5916\u8cc7\u91cf\u5360\u6bd4", pct(float(metrics["top_10_share"]))),
        ("\u524d20\u6a94\u5916\u8cc7\u91cf\u5360\u6bd4", pct(float(metrics["top_20_share"]))),
        ("\u905450%\u9700\u8981\u6a94\u6578", f"{int(metrics['stocks_for_50_share']):,}"),
        ("\u905480%\u9700\u8981\u6a94\u6578", f"{int(metrics['stocks_for_80_share']):,}"),
        ("\u7b49\u6548\u5206\u6563\u6a94\u6578", number(float(metrics["effective_stock_count"]), 1)),
    ]
    return "\n".join(
        f'<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in card_specs
    )


def write_report(
    distribution: pd.DataFrame,
    market_row: pd.Series,
    bins: pd.DataFrame,
    broad_summary: pd.DataFrame,
    industry_summary: pd.DataFrame,
    metrics: dict[str, float | int | str],
) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / "foreign_distribution.html"

    supports_hypothesis = float(metrics["top_20_share"]) >= 0.40 and int(metrics["stocks_for_50_share"]) <= 30
    conclusion = (
        "\u7d50\u8ad6\uff1a\u9019\u7d44\u6578\u64da\u652f\u6301\u4f60\u7684\u731c\u60f3\u3002"
        if supports_hypothesis
        else "\u7d50\u8ad6\uff1a\u9019\u7d44\u6578\u64da\u6c92\u6709\u660e\u986f\u652f\u6301\u300c\u5c11\u6578\u80a1\u96c6\u4e2d\u300d\u7684\u731c\u60f3\u3002"
    )
    evidence = (
        f"{conclusion}\u6700\u65b0\u4ea4\u6613\u65e5 {html.escape(str(metrics['date']))} "
        f"\u5e02\u5834\u6574\u9ad4\u5916\u8cc7\u6210\u4ea4\u53c3\u8207\u7387\u70ba {pct(float(metrics['market_foreign_participation']))}\uff0c"
        f"\u4f46\u5916\u8cc7\u6210\u4ea4\u91cf\u524d 10 \u6a94\u5df2\u5360 {pct(float(metrics['top_10_share']))}\uff0c"
        f"\u524d 20 \u6a94\u5360 {pct(float(metrics['top_20_share']))}\uff0c"
        f"\u53ea\u9700 {int(metrics['stocks_for_50_share'])} \u6a94\u5c31\u7d2f\u7a4d\u5230\u5916\u8cc7\u6210\u4ea4\u91cf\u7684 50%\u3002"
    )

    top_80_count = int(metrics["stocks_for_80_share"])
    top_table = table_rows(distribution, 20)
    top_80_table = table_rows(distribution.head(top_80_count), None)
    broad_rows = summary_table_rows(broad_summary, "BroadCategory")
    industry_rows = summary_table_rows(industry_summary, INDUSTRY_GROUP)
    bins_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.Bin))}</td>"
        f"<td>{int(row.StockCount):,}</td>"
        f"<td>{pct(float(row.StockShare))}</td>"
        f"<td>{pct(float(row.ForeignVolumeShare))}</td>"
        "</tr>"
        for row in bins.itertuples(index=False)
    )

    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>\u5916\u8cc7\u53c3\u8207\u96c6\u4e2d\u5ea6\u5831\u544a</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 22px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin: 28px 0 10px; font-size: 18px; }}
p {{ line-height: 1.65; }}
.meta {{ color: #64748b; font-size: 13px; margin-bottom: 14px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }}
.metric {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 10px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
.panel {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 14px; margin: 14px 0; }}
.chart {{ width: 100%; height: auto; display: block; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee9; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: right; font-size: 13px; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
td:nth-child(2), td:nth-child(3), th:nth-child(2), th:nth-child(3) {{ text-align: left; }}
a {{ color: #1d4ed8; text-decoration: none; }}
@media (max-width: 760px) {{ .summary {{ grid-template-columns: 1fr 1fr; }} main {{ padding: 14px; }} }}
</style>
</head>
<body>
<main>
<h1>\u5916\u8cc7\u53c3\u8207\u96c6\u4e2d\u5ea6\u5831\u544a</h1>
<div class="meta">\u4f86\u6e90\uff1a<code>output/institutional_participation/latest_summary.csv</code> \u8207 <code>market_daily_summary.csv</code></div>
<p>{evidence}</p>
<section class="summary">{metric_cards(metrics)}</section>
<section class="panel">
<h2>\u5916\u8cc7\u6210\u4ea4\u91cf\u96c6\u4e2d\u66f2\u7dda</h2>
<p>\u85cd\u7dda\u8d8a\u9760\u5de6\u4e0a\uff0c\u4ee3\u8868\u8d8a\u5c11\u6578\u80a1\u7968\u8ca2\u737b\u8d8a\u591a\u5916\u8cc7\u6210\u4ea4\u91cf\uff1b\u7070\u8272\u865b\u7dda\u662f\u5b8c\u5168\u5e73\u5747\u5206\u5e03\u7684\u5c0d\u7167\u3002</p>
{line_chart_svg(distribution)}
</section>
<section class="panel">
<h2>\u55ae\u6a94\u5916\u8cc7\u6210\u4ea4\u53c3\u8207\u7387\u5206\u5e03</h2>
{histogram_svg(bins)}
</section>
<h2>\u5916\u8cc7\u6210\u4ea4\u91cf Top 20</h2>
<table>
<thead><tr><th>\u6392\u540d</th><th>\u4ee3\u865f</th><th>\u540d\u7a31</th><th>\u5916\u8cc7\u53c3\u8207\u7387</th><th>\u4f30\u7b97\u5916\u8cc7\u6210\u4ea4\u91cf</th><th>\u5360\u5168\u5e02\u5916\u8cc7\u91cf</th><th>\u7d2f\u7a4d\u5360\u6bd4</th></tr></thead>
<tbody>
{top_table}
</tbody>
</table>
<h2>\u7d2f\u7a4d\u9054 80% \u5916\u8cc7\u6210\u4ea4\u91cf\u80a1\u7968\u6392\u540d</h2>
<p>\u4e0b\u8868\u5217\u51fa\u5f9e\u7b2c 1 \u540d\u7d2f\u7a4d\u5230\u9054\u5168\u5e02\u5916\u8cc7\u6210\u4ea4\u91cf 80% \u7684\u6240\u6709\u80a1\u7968\uff1b\u672c\u6b21\u5171 {top_80_count:,} \u6a94\u3002</p>
<table>
<thead><tr><th>\u6392\u540d</th><th>\u4ee3\u865f</th><th>\u540d\u7a31</th><th>\u5916\u8cc7\u53c3\u8207\u7387</th><th>\u4f30\u7b97\u5916\u8cc7\u6210\u4ea4\u91cf</th><th>\u5360\u5168\u5e02\u5916\u8cc7\u91cf</th><th>\u7d2f\u7a4d\u5360\u6bd4</th></tr></thead>
<tbody>
{top_80_table}
</tbody>
</table>
<h2>\u5916\u8cc7\u504f\u597d\u985e\u578b\u7d71\u6574</h2>
<p>\u4e0b\u8868\u53ea\u770b\u7d2f\u7a4d\u9054 80% \u5916\u8cc7\u6210\u4ea4\u91cf\u7684\u80a1\u7968\uff0c\u4e26\u4f9d\u64da <code>data/metadata.csv</code> \u7684\u7522\u696d\u7fa4\u7d44\u5206\u985e\u3002</p>
<table>
<thead><tr><th>\u5927\u985e</th><th>\u6a94\u6578</th><th>\u5360\u5168\u5e02\u5916\u8cc7\u91cf</th><th>\u5360Top80%\u6e05\u55ae\u5916\u8cc7\u91cf</th><th>\u5e73\u5747\u5916\u8cc7\u53c3\u8207\u7387</th><th>\u4ee3\u8868\u80a1</th></tr></thead>
<tbody>
{broad_rows}
</tbody>
</table>
<h2>\u539f\u59cb\u7522\u696d\u7fa4\u7d44\u660e\u7d30</h2>
<table>
<thead><tr><th>\u7522\u696d\u7fa4\u7d44</th><th>\u6a94\u6578</th><th>\u5360\u5168\u5e02\u5916\u8cc7\u91cf</th><th>\u5360Top80%\u6e05\u55ae\u5916\u8cc7\u91cf</th><th>\u5e73\u5747\u5916\u8cc7\u53c3\u8207\u7387</th><th>\u4ee3\u8868\u80a1</th></tr></thead>
<tbody>
{industry_rows}
</tbody>
</table>
<h2>\u5916\u8cc7\u53c3\u8207\u7387\u5340\u9593\u6458\u8981</h2>
<table>
<thead><tr><th>\u5340\u9593</th><th>\u80a1\u7968\u6a94\u6578</th><th>\u6a94\u6578\u5360\u6bd4</th><th>\u5916\u8cc7\u6210\u4ea4\u91cf\u5360\u6bd4</th></tr></thead>
<tbody>
{bins_rows}
</tbody>
</table>
<p><a href="index.html">\u56de\u5230\u6cd5\u4eba\u53c3\u8207\u7e3d\u89bd</a></p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def build_report(target_date: str | None = None) -> tuple[Path, Path, Path, Path, Path, dict[str, float | int | str]]:
    distribution, market_row = load_distribution(target_date)
    bins = build_bins(distribution)

    shares = distribution["ForeignVolumeShare"]
    hhi = float((shares**2).sum())
    metrics: dict[str, float | int | str] = {
        "date": str(market_row["Date"]),
        "stock_count": int(len(distribution)),
        "market_capacity": float(market_row["Capacity"]),
        "total_foreign_volume": float(market_row["foreign_volume"]),
        "market_foreign_participation": float(market_row["foreign_participation"]),
        "top_1_share": float(distribution.head(1)["ForeignVolumeShare"].sum()),
        "top_3_share": float(distribution.head(3)["ForeignVolumeShare"].sum()),
        "top_5_share": float(distribution.head(5)["ForeignVolumeShare"].sum()),
        "top_10_share": float(distribution.head(10)["ForeignVolumeShare"].sum()),
        "top_20_share": float(distribution.head(20)["ForeignVolumeShare"].sum()),
        "top_50_share": float(distribution.head(50)["ForeignVolumeShare"].sum()),
        "top_100_share": float(distribution.head(100)["ForeignVolumeShare"].sum()),
        "stocks_for_50_share": stocks_to_reach_share(distribution, 0.50),
        "stocks_for_80_share": stocks_to_reach_share(distribution, 0.80),
        "stocks_for_90_share": stocks_to_reach_share(distribution, 0.90),
        "hhi": hhi,
        "effective_stock_count": 1 / hhi if hhi else 0,
        "foreign_volume_gini": gini(distribution["ForeignVolume"]),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    top_path = OUTPUT_ROOT / "foreign_distribution_top_stocks.csv"
    top_80_path = OUTPUT_ROOT / "foreign_distribution_top_80pct_stocks.csv"
    broad_summary_path = OUTPUT_ROOT / "foreign_distribution_broad_category_summary.csv"
    industry_summary_path = OUTPUT_ROOT / "foreign_distribution_industry_summary.csv"
    bins_path = OUTPUT_ROOT / "foreign_distribution_bins.csv"
    metrics_path = OUTPUT_ROOT / "foreign_distribution_metrics.csv"

    output_columns = [
        "Rank",
        "Code",
        "Name",
        "Date",
        "Volume",
        "ForeignParticipation",
        "ForeignVolume",
        "ForeignVolumeShare",
        "CumulativeForeignVolumeShare",
        "ReportPath",
    ]
    distribution[output_columns].to_csv(top_path, index=False, encoding="utf-8-sig")
    top_80_distribution = add_industry_columns(distribution.head(int(metrics["stocks_for_80_share"])).copy())
    top_80_distribution[output_columns + [INDUSTRY_GROUP, "BroadCategory"]].to_csv(
        top_80_path, index=False, encoding="utf-8-sig"
    )
    broad_summary = summarize_group(top_80_distribution, "BroadCategory", float(market_row["foreign_volume"]))
    industry_summary = summarize_group(top_80_distribution, INDUSTRY_GROUP, float(market_row["foreign_volume"]))
    broad_summary.to_csv(broad_summary_path, index=False, encoding="utf-8-sig")
    industry_summary.to_csv(industry_summary_path, index=False, encoding="utf-8-sig")
    report_path = write_report(distribution, market_row, bins, broad_summary, industry_summary, metrics)
    bins.to_csv(bins_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Metric": key, "Value": value} for key, value in metrics.items()]).to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )
    return report_path, top_path, top_80_path, bins_path, metrics_path, metrics


def main() -> None:
    args = parse_args()
    report_path, top_path, top_80_path, bins_path, metrics_path, metrics = build_report(args.date)
    print("Foreign participation distribution report:")
    print(f"date={metrics['date']}")
    print(f"stock_count={metrics['stock_count']}")
    print(f"market_foreign_participation={pct(float(metrics['market_foreign_participation']))}")
    print(f"top_10_share={pct(float(metrics['top_10_share']))}")
    print(f"top_20_share={pct(float(metrics['top_20_share']))}")
    print(f"stocks_for_50_share={metrics['stocks_for_50_share']}")
    print(f"report={report_path}")
    print(f"top_stocks={top_path}")
    print(f"top_80pct_stocks={top_80_path}")
    print(f"bins={bins_path}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
