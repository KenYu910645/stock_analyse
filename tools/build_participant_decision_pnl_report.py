"""Estimate future payoff of participant net-buy decisions."""

from __future__ import annotations

import argparse
import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from column_schema import read_csv_canonical
from build_institutional_participation_report import compute_metrics


DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "price"
INSTITUTIONAL_DIR = DATA_DIR / "institutional"
METADATA_PATH = DATA_DIR / "metadata.csv"
DATA_VIZ_ROOT = PROJECT_ROOT / "data_viz" / "institutional_participation"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "institutional_participation"

INDUSTRY_GROUP = "\u7522\u696d\u7fa4\u7d44"
HORIZONS = (1, 5, 10, 20, 30, 60)


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


@dataclass(frozen=True)
class ParticipantSpec:
    key: str
    label: str
    color: str


PARTICIPANTS = [
    ParticipantSpec("foreign", "\u5916\u8cc7", "#2563eb"),
    ParticipantSpec("trust", "\u6295\u4fe1", "#d97706"),
    ParticipantSpec("dealer", "\u81ea\u71df\u5546", "#7c3aed"),
    ParticipantSpec("other", "\u5176\u4ed6\uff08\u975e\u4e09\u5927\u6cd5\u4eba\uff09", "#64748b"),
]
PARTICIPANT_ORDER = {participant.key: index for index, participant in enumerate(PARTICIPANTS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate participant decision payoff after net-buy/net-sell days."
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N stock limit for testing.")
    return parser.parse_args()


def code_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def stock_name_from_path(path: Path, fallback: str = "") -> str:
    if "_" in path.stem:
        return path.stem.split("_", 1)[1]
    return fallback or code_from_path(path)


def path_by_code(directory: Path) -> dict[str, Path]:
    return {
        code_from_path(path): path
        for path in sorted(directory.glob("*.csv"))
        if not path.name.startswith("twse_")
    }


def number_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value) * 100:.{digits}f}%"


def number(value: float | int | None, digits: int = 0) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):,.{digits}f}"


def money(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    value = float(value)
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    if abs_value >= 100_000_000:
        return f"{sign}{abs_value / 100_000_000:,.2f}\u5104"
    if abs_value >= 10_000:
        return f"{sign}{abs_value / 10_000:,.1f}\u842c"
    return f"{value:,.0f}"


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


def listed_common_codes() -> set[str]:
    metadata = pd.read_csv(METADATA_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    listed = metadata[metadata["\u985e\u578b"].eq("\u80a1\u7968") & metadata["\u5e02\u5834"].eq("\u4e0a\u5e02")]
    return set(listed["Code"].astype(str))


def metadata_frame() -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_PATH, dtype={"Code": str}, encoding="utf-8-sig")
    metadata = metadata[["Code", "Name", INDUSTRY_GROUP]].copy()
    metadata[INDUSTRY_GROUP] = metadata[INDUSTRY_GROUP].fillna("\u672a\u5206\u985e")
    metadata["BroadCategory"] = metadata[INDUSTRY_GROUP].map(broad_category)
    return metadata


def iso_date_strings(frame: pd.DataFrame, column: str = "Date") -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")


def prepare_metrics(price_path: Path, institutional_path: Path) -> pd.DataFrame:
    price = read_csv_canonical(price_path, dtype={"Code": str})
    institutional = read_csv_canonical(institutional_path, dtype={"Code": str})
    institutional_dates = set(iso_date_strings(institutional).dropna())
    if not institutional_dates:
        return pd.DataFrame()

    price_dates = iso_date_strings(price)
    price = price[price_dates.isin(institutional_dates)].copy()
    if price.empty:
        return pd.DataFrame()

    price_extra = price[["Date", "Turnover"]].copy()
    price_extra["Date"] = iso_date_strings(price_extra)
    price_extra["Turnover"] = pd.to_numeric(price_extra["Turnover"], errors="coerce")
    price_extra = price_extra.dropna(subset=["Date"]).drop_duplicates("Date", keep="last")

    metrics = compute_metrics(price, institutional)
    if metrics.empty:
        return metrics
    metrics = metrics[metrics["Date"].isin(institutional_dates)].copy()
    metrics = metrics.merge(price_extra, on="Date", how="left")
    metrics = metrics.sort_values("Date").reset_index(drop=True)

    for column in ["Capacity", "Close", "close_adj", "Turnover"]:
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    if "close_adj" not in metrics.columns:
        metrics["close_adj"] = metrics["Close"]
    metrics["close_adj"] = metrics["close_adj"].where(metrics["close_adj"].gt(0), metrics["Close"])

    average_price = metrics["Turnover"] / metrics["Capacity"]
    metrics["EntryPrice"] = average_price.where(average_price.gt(0), metrics["Close"])
    adjustment_ratio = metrics["close_adj"] / metrics["Close"]
    metrics["EntryAdjustedPrice"] = metrics["EntryPrice"] * adjustment_ratio.where(adjustment_ratio.gt(0), 1.0)

    metrics["foreign_net"] = number_series(metrics, "foreign_buy") - number_series(metrics, "foreign_sell")
    metrics["trust_net"] = number_series(metrics, "trust_buy") - number_series(metrics, "trust_sell")
    metrics["dealer_net"] = number_series(metrics, "dealer_buy") - number_series(metrics, "dealer_sell")
    metrics["other_net"] = -(metrics["foreign_net"] + metrics["trust_net"] + metrics["dealer_net"])
    metrics["Year"] = metrics["Date"].str.slice(0, 4)
    return metrics


def summarize_chunk(
    chunk: pd.DataFrame,
    participant: ParticipantSpec,
    horizon: int,
    group_values: dict[str, str | int],
) -> dict[str, float | int | str]:
    net = chunk[f"{participant.key}_net"]
    notional = chunk["DecisionNotional"]
    pnl = chunk["DecisionPnl"]
    buy_mask = net.gt(0)
    sell_mask = net.lt(0)
    gross_notional = float(notional.sum())
    estimated_pnl = float(pnl.sum())
    row: dict[str, float | int | str] = {
        **group_values,
        "ParticipantKey": participant.key,
        "Participant": participant.label,
        "HorizonDays": horizon,
        "DecisionCount": int(len(chunk)),
        "BuyDecisionCount": int(buy_mask.sum()),
        "SellDecisionCount": int(sell_mask.sum()),
        "WinCount": int(pnl.gt(0).sum()),
        "LossCount": int(pnl.lt(0).sum()),
        "GrossDecisionNotional": gross_notional,
        "SignedDecisionNotional": float((net * chunk["EntryPrice"]).sum()),
        "EstimatedPnl": estimated_pnl,
        "BuyEstimatedPnl": float(pnl[buy_mask].sum()),
        "SellEstimatedPnl": float(pnl[sell_mask].sum()),
        "BuyGrossNotional": float(notional[buy_mask].sum()),
        "SellGrossNotional": float(notional[sell_mask].sum()),
    }
    row["PnlPerNotional"] = estimated_pnl / gross_notional if gross_notional else 0.0
    row["PnlPercent"] = row["PnlPerNotional"] * 100
    row["PnlBps"] = row["PnlPerNotional"] * 10_000
    row["PnlPer100m"] = row["PnlPerNotional"] * 100_000_000
    row["WinRate"] = row["WinCount"] / row["DecisionCount"] if row["DecisionCount"] else 0.0
    row["BuyPnlPerNotional"] = row["BuyEstimatedPnl"] / row["BuyGrossNotional"] if row["BuyGrossNotional"] else 0.0
    row["SellPnlPerNotional"] = row["SellEstimatedPnl"] / row["SellGrossNotional"] if row["SellGrossNotional"] else 0.0
    return row


def stock_horizon_rows(
    code: str,
    name: str,
    industry: str,
    category: str,
    metrics: pd.DataFrame,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    stock_rows: list[dict[str, float | int | str]] = []
    year_rows: list[dict[str, float | int | str]] = []
    for horizon in HORIZONS:
        frame = metrics.copy()
        frame["FutureAdjustedClose"] = frame["close_adj"].shift(-horizon)
        frame["ForwardReturn"] = frame["FutureAdjustedClose"] / frame["EntryAdjustedPrice"] - 1.0
        valid_base = (
            frame["FutureAdjustedClose"].gt(0)
            & frame["EntryAdjustedPrice"].gt(0)
            & frame["EntryPrice"].gt(0)
            & frame["ForwardReturn"].map(math.isfinite)
        )
        for participant in PARTICIPANTS:
            net_column = f"{participant.key}_net"
            valid = valid_base & frame[net_column].ne(0)
            chunk = frame.loc[valid, ["Date", "Year", net_column, "EntryPrice", "ForwardReturn"]].copy()
            if chunk.empty:
                continue
            chunk["DecisionNotional"] = chunk[net_column].abs() * chunk["EntryPrice"]
            chunk["DecisionPnl"] = chunk[net_column] * chunk["EntryPrice"] * chunk["ForwardReturn"]
            stock_rows.append(
                summarize_chunk(
                    chunk,
                    participant,
                    horizon,
                    {
                        "Code": code,
                        "Name": name,
                        INDUSTRY_GROUP: industry,
                        "BroadCategory": category,
                        "StartDate": str(chunk["Date"].iloc[0]),
                        "EndDate": str(chunk["Date"].iloc[-1]),
                    },
                )
            )
            for year, year_chunk in chunk.groupby("Year", sort=True):
                year_rows.append(
                    summarize_chunk(
                        year_chunk,
                        participant,
                        horizon,
                        {
                            "Year": str(year),
                            "Code": code,
                            "Name": name,
                            INDUSTRY_GROUP: industry,
                            "BroadCategory": category,
                        },
                    )
                )
    return stock_rows, year_rows


def aggregate_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    aggregated = (
        frame.groupby(group_columns, dropna=False)
        .agg(
            StockCount=("Code", "nunique") if "Code" in frame.columns else ("DecisionCount", "size"),
            DecisionCount=("DecisionCount", "sum"),
            BuyDecisionCount=("BuyDecisionCount", "sum"),
            SellDecisionCount=("SellDecisionCount", "sum"),
            WinCount=("WinCount", "sum"),
            LossCount=("LossCount", "sum"),
            GrossDecisionNotional=("GrossDecisionNotional", "sum"),
            SignedDecisionNotional=("SignedDecisionNotional", "sum"),
            EstimatedPnl=("EstimatedPnl", "sum"),
            BuyEstimatedPnl=("BuyEstimatedPnl", "sum"),
            SellEstimatedPnl=("SellEstimatedPnl", "sum"),
            BuyGrossNotional=("BuyGrossNotional", "sum"),
            SellGrossNotional=("SellGrossNotional", "sum"),
        )
        .reset_index()
    )
    aggregated["PnlPerNotional"] = aggregated["EstimatedPnl"] / aggregated["GrossDecisionNotional"]
    aggregated["PnlPercent"] = aggregated["PnlPerNotional"] * 100
    aggregated["PnlBps"] = aggregated["PnlPerNotional"] * 10_000
    aggregated["PnlPer100m"] = aggregated["PnlPerNotional"] * 100_000_000
    aggregated["WinRate"] = aggregated["WinCount"] / aggregated["DecisionCount"]
    aggregated["BuyPnlPerNotional"] = aggregated["BuyEstimatedPnl"] / aggregated["BuyGrossNotional"]
    aggregated["SellPnlPerNotional"] = aggregated["SellEstimatedPnl"] / aggregated["SellGrossNotional"]
    return aggregated.replace([float("inf"), -float("inf")], 0.0).fillna(0.0)


def ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.assign(_ParticipantOrder=summary["ParticipantKey"].map(PARTICIPANT_ORDER).fillna(99))
        .sort_values(["HorizonDays", "_ParticipantOrder"])
        .drop(columns=["_ParticipantOrder"])
    )


def legend_svg(width: int, y: int) -> str:
    items = []
    start_x = 84
    for index, participant in enumerate(PARTICIPANTS):
        x = start_x + index * 168
        items.append(
            f'<rect x="{x}" y="{y - 10}" width="10" height="10" fill="{participant.color}"/>'
            f'<text x="{x + 16}" y="{y - 1}">{html.escape(participant.label)}</text>'
        )
    return "".join(items)


def pnl_percent_chart_svg(summary: pd.DataFrame) -> str:
    width = 1120
    height = 390
    left = 66
    right = 24
    top = 28
    bottom = 72
    chart_width = width - left - right
    chart_height = height - top - bottom
    ordered = ordered_summary(summary)
    max_abs = max(float(ordered["PnlPerNotional"].abs().max()), 0.001)
    zero_y = top + chart_height / 2
    group_width = chart_width / len(HORIZONS)
    bar_width = min(26, (group_width - 32) / len(PARTICIPANTS))
    bars = []
    for h_index, horizon in enumerate(HORIZONS):
        subset = ordered[ordered["HorizonDays"].eq(horizon)].set_index("ParticipantKey")
        group_x = left + h_index * group_width
        for p_index, participant in enumerate(PARTICIPANTS):
            if participant.key not in subset.index:
                continue
            value = float(subset.loc[participant.key, "PnlPerNotional"])
            bar_height = abs(value) / max_abs * (chart_height / 2 - 18)
            x = group_x + 15 + p_index * (bar_width + 5)
            y = zero_y - bar_height if value >= 0 else zero_y
            label_y = y - 5 if value >= 0 else y + bar_height + 14
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{participant.color}"/>'
                f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle">{pct(value, 2)}</text>'
            )
        bars.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="{height - 32}" text-anchor="middle">{horizon}\u65e5</text>'
        )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="\u56db\u7fa4\u4eba\u640d\u76ca\u7387\u76f4\u65b9\u5716">
  <style>
    .axis {{ stroke: #334155; stroke-width: 1; }}
    .baseline {{ stroke: #475569; stroke-width: 1; stroke-dasharray: 5 5; }}
    text {{ fill: #475569; font-size: 11px; font-family: "Microsoft JhengHei", Arial, sans-serif; }}
  </style>
  <line x1="{left}" y1="{zero_y:.1f}" x2="{left + chart_width}" y2="{zero_y:.1f}" class="baseline"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>
  {''.join(bars)}
  <text x="{left + chart_width / 2}" y="16" text-anchor="middle">\u640d\u76ca / \u6295\u6ce8\u91d1\u984d</text>
  <text x="{left - 8}" y="{zero_y + 4:.1f}" text-anchor="end">0%</text>
  {legend_svg(width, height - 8)}
</svg>
"""


def win_rate_chart_svg(summary: pd.DataFrame) -> str:
    width = 1120
    height = 390
    left = 66
    right = 24
    top = 28
    bottom = 72
    chart_width = width - left - right
    chart_height = height - top - bottom
    ordered = ordered_summary(summary)
    min_value = min(0.45, float(ordered["WinRate"].min()) - 0.01)
    max_value = max(0.55, float(ordered["WinRate"].max()) + 0.01)
    if max_value <= min_value:
        max_value = min_value + 0.1
    baseline = 0.5

    def y_for(value: float) -> float:
        return top + (max_value - value) / (max_value - min_value) * chart_height

    baseline_y = y_for(baseline)
    group_width = chart_width / len(HORIZONS)
    bar_width = min(26, (group_width - 32) / len(PARTICIPANTS))
    bars = []
    for h_index, horizon in enumerate(HORIZONS):
        subset = ordered[ordered["HorizonDays"].eq(horizon)].set_index("ParticipantKey")
        group_x = left + h_index * group_width
        for p_index, participant in enumerate(PARTICIPANTS):
            if participant.key not in subset.index:
                continue
            value = float(subset.loc[participant.key, "WinRate"])
            y = min(y_for(value), baseline_y)
            bar_height = abs(y_for(value) - baseline_y)
            x = group_x + 15 + p_index * (bar_width + 5)
            label_y = y - 5 if value >= baseline else y + bar_height + 14
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{participant.color}"/>'
                f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle">{pct(value, 1)}</text>'
            )
        bars.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="{height - 32}" text-anchor="middle">{horizon}\u65e5</text>'
        )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="\u56db\u7fa4\u4eba\u52dd\u7387\u76f4\u65b9\u5716">
  <style>
    .axis {{ stroke: #334155; stroke-width: 1; }}
    .baseline {{ stroke: #475569; stroke-width: 1; stroke-dasharray: 5 5; }}
    text {{ fill: #475569; font-size: 11px; font-family: "Microsoft JhengHei", Arial, sans-serif; }}
  </style>
  <line x1="{left}" y1="{baseline_y:.1f}" x2="{left + chart_width}" y2="{baseline_y:.1f}" class="baseline"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>
  {''.join(bars)}
  <text x="{left + chart_width / 2}" y="16" text-anchor="middle">\u52dd\u7387\uff0850% \u70ba\u865b\u7dda\u57fa\u6e96\uff09</text>
  <text x="{left - 8}" y="{baseline_y + 4:.1f}" text-anchor="end">50%</text>
  {legend_svg(width, height - 8)}
</svg>
"""


def summary_table(summary: pd.DataFrame) -> str:
    rows = []
    ordered = ordered_summary(summary)
    for row in ordered.itertuples(index=False):
        css_class = "pos" if float(row.EstimatedPnl) >= 0 else "neg"
        rows.append(
            "<tr>"
            f"<td>{int(row.HorizonDays)}\u65e5</td>"
            f"<td>{html.escape(str(row.Participant))}</td>"
            f"<td>{int(row.StockCount):,}</td>"
            f"<td>{int(row.DecisionCount):,}</td>"
            f"<td>{pct(float(row.WinRate))}</td>"
            f"<td>{money(float(row.GrossDecisionNotional))}</td>"
            f'<td class="{css_class}">{money(float(row.EstimatedPnl))}</td>'
            f'<td class="{css_class}">{pct(float(row.PnlPerNotional), 3)}</td>'
            f'<td class="{css_class}">{money(float(row.PnlPer100m))}</td>'
            f'<td class="{css_class}">{float(row.PnlBps):,.1f}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def top_bottom_table(stock_summary: pd.DataFrame, participant_key: str, horizon: int, largest: bool) -> str:
    subset = stock_summary[
        stock_summary["ParticipantKey"].eq(participant_key) & stock_summary["HorizonDays"].eq(horizon)
    ].copy()
    subset = subset.sort_values("EstimatedPnl", ascending=not largest).head(15)
    rows = []
    for rank, row in enumerate(subset.itertuples(index=False), start=1):
        css_class = "pos" if float(row.EstimatedPnl) >= 0 else "neg"
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(str(row.Code))}</td>"
            f"<td>{html.escape(str(row.Name))}</td>"
            f"<td>{html.escape(str(getattr(row, INDUSTRY_GROUP)))}</td>"
            f"<td>{int(row.DecisionCount):,}</td>"
            f"<td>{pct(float(row.WinRate))}</td>"
            f"<td>{money(float(row.GrossDecisionNotional))}</td>"
            f'<td class="{css_class}">{money(float(row.EstimatedPnl))}</td>'
            f'<td class="{css_class}">{money(float(row.PnlPer100m))}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def industry_table(category_summary: pd.DataFrame, horizon: int) -> str:
    subset = category_summary[category_summary["HorizonDays"].eq(horizon)].copy()
    subset["_ParticipantOrder"] = subset["ParticipantKey"].map(PARTICIPANT_ORDER).fillna(99)
    subset = subset.sort_values(["_ParticipantOrder", "EstimatedPnl"], ascending=[True, False])
    rows = []
    for row in subset.itertuples(index=False):
        css_class = "pos" if float(row.EstimatedPnl) >= 0 else "neg"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.Participant))}</td>"
            f"<td>{html.escape(str(row.BroadCategory))}</td>"
            f"<td>{int(row.StockCount):,}</td>"
            f"<td>{int(row.DecisionCount):,}</td>"
            f"<td>{pct(float(row.WinRate))}</td>"
            f'<td class="{css_class}">{money(float(row.EstimatedPnl))}</td>'
            f'<td class="{css_class}">{money(float(row.PnlPer100m))}</td>'
            "</tr>"
        )
    return "\n".join(rows)


def write_report(
    summary: pd.DataFrame,
    stock_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / "participant_decision_pnl.html"
    h20 = summary[summary["HorizonDays"].eq(20)].sort_values("PnlPer100m", ascending=False)
    best = h20.iloc[0]
    worst = h20.iloc[-1]
    coverage_start = str(stock_summary["StartDate"].min())
    coverage_end = str(stock_summary["EndDate"].max())
    participant_sections = []
    for participant in PARTICIPANTS:
        participant_sections.append(
            f"""
<section class="panel">
<h2>{html.escape(participant.label)} 20 \u65e5\u5f8c\u4f30\u7b97\u640d\u76ca\u500b\u80a1\u6392\u540d</h2>
<div class="two-col">
<div>
<h3>\u8ca2\u737b\u6700\u591a</h3>
<table>
<thead><tr><th>\u6392\u540d</th><th>\u4ee3\u865f</th><th>\u540d\u7a31</th><th>\u7522\u696d</th><th>\u6c7a\u7b56\u6578</th><th>\u52dd\u7387</th><th>\u540d\u76ee\u91d1\u984d</th><th>\u4f30\u7b97\u640d\u76ca</th><th>\u6bcf1\u5104\u640d\u76ca</th></tr></thead>
<tbody>{top_bottom_table(stock_summary, participant.key, 20, True)}</tbody>
</table>
</div>
<div>
<h3>\u62d6\u7d2f\u6700\u591a</h3>
<table>
<thead><tr><th>\u6392\u540d</th><th>\u4ee3\u865f</th><th>\u540d\u7a31</th><th>\u7522\u696d</th><th>\u6c7a\u7b56\u6578</th><th>\u52dd\u7387</th><th>\u540d\u76ee\u91d1\u984d</th><th>\u4f30\u7b97\u640d\u76ca</th><th>\u6bcf1\u5104\u640d\u76ca</th></tr></thead>
<tbody>{top_bottom_table(stock_summary, participant.key, 20, False)}</tbody>
</table>
</div>
</div>
</section>
"""
        )

    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>\u56db\u7fa4\u4eba\u8cb7\u8ce3\u8d85\u5f8c\u7e8c\u4f30\u7b97\u640d\u76ca</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
main {{ max-width: 1320px; margin: 0 auto; padding: 22px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin: 26px 0 10px; font-size: 18px; }}
h3 {{ margin: 8px 0; font-size: 15px; }}
p {{ line-height: 1.65; }}
.meta {{ color: #64748b; font-size: 13px; margin-bottom: 14px; }}
.summary {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 16px 0; }}
.metric {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 10px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
.panel {{ background: white; border: 1px solid #d7dee9; border-radius: 6px; padding: 14px; margin: 14px 0; }}
.chart {{ width: 100%; height: auto; display: block; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: start; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d7dee9; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: right; font-size: 13px; }}
th {{ background: #f1f5f9; position: sticky; top: 0; }}
td:nth-child(2), td:nth-child(3), td:nth-child(4), th:nth-child(2), th:nth-child(3), th:nth-child(4) {{ text-align: left; }}
.pos {{ color: #047857; font-weight: 700; }}
.neg {{ color: #b91c1c; font-weight: 700; }}
a {{ color: #1d4ed8; text-decoration: none; }}
code {{ background: #eef2ff; padding: 1px 4px; border-radius: 3px; }}
@media (max-width: 900px) {{ .summary {{ grid-template-columns: 1fr 1fr; }} .two-col {{ grid-template-columns: 1fr; }} main {{ padding: 14px; }} }}
</style>
</head>
<body>
<main>
<h1>\u56db\u7fa4\u4eba\u8cb7\u8ce3\u8d85\u5f8c\u7e8c\u4f30\u7b97\u640d\u76ca</h1>
<div class="meta">\u4f86\u6e90\uff1a<code>data/price/</code> \u8207 <code>data/institutional/</code>\uff1b\u6a23\u672c\u671f\u9593 {html.escape(coverage_start)} ~ {html.escape(coverage_end)}</div>
<p>\u9019\u4efd\u5831\u544a\u4e0d\u662f\u771f\u5be6\u5eab\u5b58\u6703\u8a08\u640d\u76ca\uff0c\u800c\u662f\u628a\u6bcf\u5929\u6bcf\u6a94\u80a1\u7968\u7684\u8cb7\u8ce3\u8d85\u8996\u70ba\u4e00\u500b\u65b9\u5411\u6c7a\u7b56\uff1a\u6de8\u8cb7\u5f8c\u4e0a\u6f32\u7b97\u8cfa\uff0c\u6de8\u8ce3\u5f8c\u4e0b\u8dcc\u4e5f\u7b97\u8cfa\u3002\u4f30\u7b97\u6210\u4ea4\u50f9\u7528\u7576\u65e5 <code>Turnover / Capacity</code>\uff0c\u672a\u4f86\u50f9\u683c\u7528\u5fa9\u6b0a\u6536\u76e4\u50f9\uff0c\u907f\u514d\u9664\u6b0a\u606f\u9020\u6210\u865b\u5047\u8dcc\u50f9\u3002</p>
<p>\u300c\u5176\u4ed6\u300d\u5728\u9019\u88e1\u7528\u4e09\u5927\u6cd5\u4eba\u6de8\u8cb7\u8ce3\u8d85\u7684\u53cd\u5411\u4f30\u7b97\uff0c\u6240\u4ee5\u5b83\u662f\u975e\u4e09\u5927\u6cd5\u4eba\u7684\u65b9\u5411\u6027\u8fd1\u4f3c\uff0c\u4e0d\u662f\u9010\u7b46\u8eab\u5206\u8a8d\u5b9a\u7684\u7d14\u6563\u6236\u640d\u76ca\u3002\u672c\u5831\u544a\u672a\u6263\u624b\u7e8c\u8cbb\u3001\u4ea4\u6613\u7a05\u8207\u501f\u5238\u6210\u672c\u3002</p>
<section class="summary">
<div class="metric"><div class="label">20\u65e5\u5f8c\u6bcf1\u5104\u6700\u597d</div><div class="value">{html.escape(str(best.Participant))} {money(float(best.PnlPer100m))}</div></div>
<div class="metric"><div class="label">20\u65e5\u5f8c\u6bcf1\u5104\u6700\u5dee</div><div class="value">{html.escape(str(worst.Participant))} {money(float(worst.PnlPer100m))}</div></div>
<div class="metric"><div class="label">\u6a23\u672c\u6a94\u6578</div><div class="value">{int(summary["StockCount"].max()):,}</div></div>
<div class="metric"><div class="label">\u8a55\u4f30\u671f\u9593</div><div class="value">1 / 5 / 10 / 20 / 30 / 60 \u65e5</div></div>
</section>
<section class="panel">
<h2>\u76f4\u65b9\u5716\uff1a\u640d\u76ca\u7387\u8207\u52dd\u7387</h2>
<p>\u640d\u76ca\u7387\u662f\u300c\u4f30\u7b97\u640d\u76ca / \u6295\u6ce8\u91d1\u984d\u300d\uff0c\u6295\u6ce8\u91d1\u984d\u70ba\u8cb7\u8ce3\u8d85\u7684\u7d55\u5c0d\u540d\u76ee\u91d1\u984d\u3002\u52dd\u7387\u5716\u4e2d\u7684\u865b\u7dda\u662f 50% \u57fa\u6e96\u3002</p>
{pnl_percent_chart_svg(summary)}
{win_rate_chart_svg(summary)}
</section>
<h2>\u56db\u7fa4\u4eba\u5f8c\u7e8c\u4f30\u7b97\u640d\u76ca\u7e3d\u8868</h2>
<table>
<thead><tr><th>\u671f\u9593</th><th>\u7fa4\u7d44</th><th>\u6a94\u6578</th><th>\u6c7a\u7b56\u6578</th><th>\u52dd\u7387</th><th>\u6295\u6ce8\u91d1\u984d</th><th>\u4f30\u7b97\u640d\u76ca</th><th>\u640d\u76ca/\u6295\u6ce8\u91d1\u984d</th><th>\u6bcf1\u5104\u640d\u76ca</th><th>\u57fa\u9ede</th></tr></thead>
<tbody>
{summary_table(summary)}
</tbody>
</table>
<section class="panel">
<h2>20 \u65e5\u5f8c\u7522\u696d\u5927\u985e\u6458\u8981</h2>
<table>
<thead><tr><th>\u7fa4\u7d44</th><th>\u7522\u696d\u5927\u985e</th><th>\u6a94\u6578</th><th>\u6c7a\u7b56\u6578</th><th>\u52dd\u7387</th><th>\u4f30\u7b97\u640d\u76ca</th><th>\u6bcf1\u5104\u640d\u76ca</th></tr></thead>
<tbody>
{industry_table(category_summary, 20)}
</tbody>
</table>
</section>
{''.join(participant_sections)}
<section class="panel">
<h2>\u8f38\u51fa\u6a94\u6848</h2>
<p>\u7e3d\u8868 CSV\uff1a<code>{html.escape(str(paths["summary"].relative_to(PROJECT_ROOT)))}</code><br>
\u500b\u80a1 CSV\uff1a<code>{html.escape(str(paths["stock"].relative_to(PROJECT_ROOT)))}</code><br>
\u5e74\u5ea6 CSV\uff1a<code>{html.escape(str(paths["year"].relative_to(PROJECT_ROOT)))}</code><br>
\u7522\u696d CSV\uff1a<code>{html.escape(str(paths["category"].relative_to(PROJECT_ROOT)))}</code></p>
</section>
<p><a href="index.html">\u56de\u5230\u6cd5\u4eba\u53c3\u8207\u7e3d\u89bd</a></p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return report_path


def build_report(limit: int | None = None) -> dict[str, Path]:
    price_paths = path_by_code(PRICE_DIR)
    institutional_paths = path_by_code(INSTITUTIONAL_DIR)
    allowed_codes = listed_common_codes()
    codes = sorted((set(price_paths) & set(institutional_paths)) & allowed_codes)
    if limit is not None:
        codes = codes[:limit]

    metadata = metadata_frame().set_index("Code")
    stock_rows: list[dict[str, float | int | str]] = []
    year_rows: list[dict[str, float | int | str]] = []

    for index, code in enumerate(codes, start=1):
        price_path = price_paths[code]
        institutional_path = institutional_paths[code]
        name = stock_name_from_path(institutional_path, stock_name_from_path(price_path, code))
        industry = str(metadata.loc[code, INDUSTRY_GROUP]) if code in metadata.index else "\u672a\u5206\u985e"
        category = str(metadata.loc[code, "BroadCategory"]) if code in metadata.index else "\u5176\u4ed6"
        try:
            metrics = prepare_metrics(price_path, institutional_path)
            if metrics.empty:
                continue
            new_stock_rows, new_year_rows = stock_horizon_rows(code, name, industry, category, metrics)
            stock_rows.extend(new_stock_rows)
            year_rows.extend(new_year_rows)
        except Exception as exc:
            print(f"skipped {code}: {exc}")
        if index % 100 == 0 or index == len(codes):
            print(f"processed {index}/{len(codes)}")

    if not stock_rows:
        raise SystemExit("no participant decision rows found")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stock_summary = pd.DataFrame(stock_rows)
    year_stock_summary = pd.DataFrame(year_rows)
    summary = aggregate_summary(stock_summary, ["ParticipantKey", "Participant", "HorizonDays"])
    year_summary = aggregate_summary(year_stock_summary, ["Year", "ParticipantKey", "Participant", "HorizonDays"])
    category_summary = aggregate_summary(
        stock_summary,
        ["ParticipantKey", "Participant", "HorizonDays", "BroadCategory"],
    )

    sort_columns = ["HorizonDays", "ParticipantKey"]
    summary = summary.sort_values(sort_columns).reset_index(drop=True)
    stock_summary = stock_summary.sort_values(["HorizonDays", "ParticipantKey", "EstimatedPnl"], ascending=[True, True, False])
    year_summary = year_summary.sort_values(["Year", "HorizonDays", "ParticipantKey"]).reset_index(drop=True)
    category_summary = category_summary.sort_values(["HorizonDays", "ParticipantKey", "EstimatedPnl"], ascending=[True, True, False])

    paths = {
        "summary": OUTPUT_ROOT / "participant_decision_pnl_summary.csv",
        "stock": OUTPUT_ROOT / "participant_decision_pnl_by_stock.csv",
        "year": OUTPUT_ROOT / "participant_decision_pnl_by_year.csv",
        "category": OUTPUT_ROOT / "participant_decision_pnl_by_category.csv",
    }
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    stock_summary.to_csv(paths["stock"], index=False, encoding="utf-8-sig")
    year_summary.to_csv(paths["year"], index=False, encoding="utf-8-sig")
    category_summary.to_csv(paths["category"], index=False, encoding="utf-8-sig")
    report_path = write_report(summary, stock_summary, category_summary, paths)
    paths["report"] = report_path
    return paths


def main() -> None:
    args = parse_args()
    paths = build_report(args.limit)
    print(f"report={paths['report']}")
    print(f"summary={paths['summary']}")
    summary = pd.read_csv(paths["summary"], encoding="utf-8-sig")
    for row in ordered_summary(summary).itertuples(index=False):
        print(
            f"{row.Participant} h={int(row.HorizonDays)} "
            f"pnl={money(float(row.EstimatedPnl))} "
            f"pnl_pct={pct(float(row.PnlPerNotional), 3)} "
            f"per100m={money(float(row.PnlPer100m))} "
            f"win_rate={pct(float(row.WinRate))}"
        )


if __name__ == "__main__":
    main()
