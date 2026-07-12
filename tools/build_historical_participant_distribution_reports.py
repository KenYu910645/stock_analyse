"""Build full-history concentration reports for participant groups."""

from __future__ import annotations

import argparse
import html
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical
from build_institutional_participation_report import compute_metrics


DATA_DIR = PROJECT_ROOT / "data"
PRICE_DIR = DATA_DIR / "price"
INSTITUTIONAL_DIR = DATA_DIR / "institutional"
METADATA_PATH = DATA_DIR / "metadata.csv"
DATA_VIZ_ROOT = PROJECT_ROOT / "data_viz" / "institutional_participation"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "institutional_participation"

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


@dataclass(frozen=True)
class ParticipantSpec:
    key: str
    label: str
    short_label: str
    volume_column: str
    buy_column: str
    sell_column: str
    output_prefix: str
    report_filename: str
    note: str = ""


PARTICIPANTS: dict[str, ParticipantSpec] = {
    "foreign": ParticipantSpec(
        key="foreign",
        label="\u5916\u8cc7",
        short_label="\u5916\u8cc7",
        volume_column="foreign_volume",
        buy_column="foreign_buy",
        sell_column="foreign_sell",
        output_prefix="foreign_distribution",
        report_filename="foreign_distribution.html",
    ),
    "trust": ParticipantSpec(
        key="trust",
        label="\u6295\u4fe1",
        short_label="\u6295\u4fe1",
        volume_column="trust_volume",
        buy_column="trust_buy",
        sell_column="trust_sell",
        output_prefix="trust_distribution",
        report_filename="trust_distribution.html",
    ),
    "dealer": ParticipantSpec(
        key="dealer",
        label="\u81ea\u71df\u5546",
        short_label="\u81ea\u71df\u5546",
        volume_column="dealer_volume",
        buy_column="dealer_buy",
        sell_column="dealer_sell",
        output_prefix="dealer_distribution",
        report_filename="dealer_distribution.html",
    ),
    "other": ParticipantSpec(
        key="other",
        label="\u5176\u4ed6\uff08\u6563\u6236\u8fd1\u4f3c\uff09",
        short_label="\u5176\u4ed6",
        volume_column="other_volume",
        buy_column="other_buy",
        sell_column="other_sell",
        output_prefix="other_distribution",
        report_filename="other_distribution.html",
        note=(
            "\u300c\u5176\u4ed6\u300d\u662f\u4ee5\u5168\u5e02\u6210\u4ea4\u91cf\u6263\u9664\u5916\u8cc7\u3001"
            "\u6295\u4fe1\u3001\u81ea\u71df\u5546\u5f8c\u4f30\u7b97\uff0c\u53ef\u8996\u70ba\u6563\u6236\u8207"
            "\u5176\u4ed6\u975e\u4e09\u5927\u6cd5\u4eba\u4ea4\u6613\u7684\u8fd1\u4f3c\uff0c\u4e0d\u7b49\u540c\u65bc"
            "\u9010\u7b46\u8eab\u5206\u8a8d\u5b9a\u7684\u7d14\u6563\u6236\u3002"
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-history participant concentration reports."
    )
    parser.add_argument(
        "--participants",
        default="foreign,trust,dealer,other",
        help="Comma-separated participant keys: foreign,trust,dealer,other, or all.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N stock limit for testing.")
    return parser.parse_args()


def selected_specs(raw: str) -> list[ParticipantSpec]:
    keys = list(PARTICIPANTS) if raw.strip().lower() == "all" else [item.strip() for item in raw.split(",")]
    specs: list[ParticipantSpec] = []
    for key in keys:
        if key not in PARTICIPANTS:
            raise ValueError(f"unknown_participant:{key}")
        specs.append(PARTICIPANTS[key])
    return specs


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
    reached = distribution["CumulativeParticipantVolumeShare"].ge(target_share)
    if not reached.any():
        return len(distribution)
    return int(reached.idxmax()) + 1


def iso_date_strings(frame: pd.DataFrame, column: str = "Date") -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d")


def read_overlap_metrics(price_path: Path, institutional_path: Path) -> pd.DataFrame:
    price = read_csv_canonical(price_path, dtype={"Code": str})
    institutional = read_csv_canonical(institutional_path, dtype={"Code": str})
    institutional_dates = set(iso_date_strings(institutional).dropna())
    if not institutional_dates:
        return pd.DataFrame()
    price_dates = iso_date_strings(price)
    price = price[price_dates.isin(institutional_dates)].copy()
    if price.empty:
        return pd.DataFrame()
    metrics = compute_metrics(price, institutional)
    if metrics.empty:
        return metrics
    metrics = metrics[metrics["Date"].isin(institutional_dates)].copy()
    return metrics.sort_values("Date")


def sum_numeric(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def build_stock_totals(limit: int | None = None) -> pd.DataFrame:
    price_paths = path_by_code(PRICE_DIR)
    institutional_paths = path_by_code(INSTITUTIONAL_DIR)
    codes = sorted(set(price_paths) & set(institutional_paths))
    if limit is not None:
        codes = codes[:limit]

    rows: list[dict[str, float | int | str]] = []
    for index, code in enumerate(codes, start=1):
        price_path = price_paths[code]
        institutional_path = institutional_paths[code]
        name = stock_name_from_path(institutional_path, stock_name_from_path(price_path, code))
        try:
            metrics = read_overlap_metrics(price_path, institutional_path)
            if metrics.empty:
                continue
            capacity = sum_numeric(metrics, "Capacity")
            if capacity <= 0:
                continue
            row: dict[str, float | int | str] = {
                "Code": code,
                "Name": name,
                "StartDate": str(metrics["Date"].iloc[0]),
                "EndDate": str(metrics["Date"].iloc[-1]),
                "TradingDays": int(len(metrics)),
                "TotalVolume": capacity,
                "ReportPath": str((DATA_VIZ_ROOT / "stocks" / f"{code}_{name}.html").relative_to(PROJECT_ROOT)),
            }
            for spec in PARTICIPANTS.values():
                participant_volume = sum_numeric(metrics, spec.volume_column)
                buy = sum_numeric(metrics, spec.buy_column)
                sell = sum_numeric(metrics, spec.sell_column)
                row[f"{spec.key}_volume"] = participant_volume
                row[f"{spec.key}_participation"] = participant_volume / capacity if capacity else 0.0
                row[f"{spec.key}_buy"] = buy
                row[f"{spec.key}_sell"] = sell
                row[f"{spec.key}_net_ratio"] = (buy - sell) / capacity if capacity else 0.0
                row[f"{spec.key}_purity"] = (buy - sell) / (buy + sell) if (buy + sell) else 0.0
            rows.append(row)
        except Exception as exc:
            print(f"skipped {code}: {exc}")
        if index % 100 == 0 or index == len(codes):
            print(f"processed {index}/{len(codes)}")
    return pd.DataFrame(rows)


def distribution_for(spec: ParticipantSpec, stock_totals: pd.DataFrame) -> pd.DataFrame:
    distribution = stock_totals[
        [
            "Code",
            "Name",
            "StartDate",
            "EndDate",
            "TradingDays",
            "TotalVolume",
            "ReportPath",
            f"{spec.key}_volume",
            f"{spec.key}_participation",
            f"{spec.key}_buy",
            f"{spec.key}_sell",
            f"{spec.key}_net_ratio",
            f"{spec.key}_purity",
        ]
    ].copy()
    distribution = distribution.rename(
        columns={
            f"{spec.key}_volume": "ParticipantVolume",
            f"{spec.key}_participation": "Participation",
            f"{spec.key}_buy": "ParticipantBuy",
            f"{spec.key}_sell": "ParticipantSell",
            f"{spec.key}_net_ratio": "ParticipantNetRatio",
            f"{spec.key}_purity": "ParticipantPurity",
        }
    )
    distribution = distribution[distribution["ParticipantVolume"].ge(0)]
    distribution = distribution.sort_values(["ParticipantVolume", "Code"], ascending=[False, True]).reset_index(drop=True)
    total_volume = float(distribution["ParticipantVolume"].sum())
    if total_volume <= 0:
        raise ValueError(f"{spec.key}_participant_volume_total_is_zero")
    distribution["Participant"] = spec.short_label
    distribution["Rank"] = range(1, len(distribution) + 1)
    distribution["ParticipantVolumeShare"] = distribution["ParticipantVolume"] / total_volume
    distribution["CumulativeParticipantVolumeShare"] = distribution["ParticipantVolumeShare"].cumsum()
    return distribution


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
    total_participant_volume = float(distribution["ParticipantVolume"].sum())
    for lower, upper, label in bins:
        if math.isinf(upper):
            mask = distribution["Participation"].ge(lower)
        else:
            mask = distribution["Participation"].ge(lower) & distribution["Participation"].lt(upper)
        subset = distribution[mask]
        participant_volume = float(subset["ParticipantVolume"].sum())
        rows.append(
            {
                "Bin": label,
                "LowerBound": lower,
                "UpperBound": upper,
                "StockCount": int(len(subset)),
                "StockShare": len(subset) / total_stocks if total_stocks else 0,
                "ParticipantVolume": participant_volume,
                "ParticipantVolumeShare": participant_volume / total_participant_volume if total_participant_volume else 0,
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
    enriched = distribution.merge(metadata[["Code", INDUSTRY_GROUP]], on="Code", how="left")
    enriched[INDUSTRY_GROUP] = enriched[INDUSTRY_GROUP].fillna("\u672a\u5206\u985e")
    enriched["BroadCategory"] = enriched[INDUSTRY_GROUP].map(broad_category)
    return enriched


def summarize_group(distribution: pd.DataFrame, group_column: str, all_participant_volume: float) -> pd.DataFrame:
    top_80_participant_volume = float(distribution["ParticipantVolume"].sum())
    summary = (
        distribution.groupby(group_column, dropna=False)
        .agg(
            StockCount=("Code", "count"),
            ParticipantVolume=("ParticipantVolume", "sum"),
            AvgParticipation=("Participation", "mean"),
        )
        .reset_index()
        .sort_values("ParticipantVolume", ascending=False)
    )
    summary["ShareOfTop80ParticipantVolume"] = summary["ParticipantVolume"] / top_80_participant_volume
    summary["ShareOfAllParticipantVolume"] = summary["ParticipantVolume"] / all_participant_volume
    top_stocks: dict[str, str] = {}
    for group_value, subset in distribution.groupby(group_column, dropna=False):
        top_stocks[str(group_value)] = "\u3001".join(
            f"{row.Code} {row.Name}" for row in subset.sort_values("ParticipantVolume", ascending=False).head(8).itertuples(index=False)
        )
    summary["TopStocks"] = summary[group_column].astype(str).map(top_stocks).fillna("")
    return summary


def line_chart_svg(distribution: pd.DataFrame, spec: ParticipantSpec) -> str:
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
        y = top + (1 - float(row.CumulativeParticipantVolumeShare)) * chart_height
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
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(spec.label)}\u5168\u6b77\u53f2\u6210\u4ea4\u91cf\u7d2f\u7a4d\u96c6\u4e2d\u66f2\u7dda">
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
  <text x="{left + chart_width / 2}" y="{height - 2}" text-anchor="middle">\u6309{html.escape(spec.short_label)}\u5168\u6b77\u53f2\u6210\u4ea4\u91cf\u6392\u540d\u7684\u7d2f\u7a4d\u80a1\u7968\u6bd4\u4f8b</text>
  <text x="14" y="{top + chart_height / 2}" transform="rotate(-90 14 {top + chart_height / 2})" text-anchor="middle">\u7d2f\u7a4d{html.escape(spec.short_label)}\u6210\u4ea4\u91cf\u6bd4\u91cd</text>
</svg>
"""


def histogram_svg(bins: pd.DataFrame, spec: ParticipantSpec) -> str:
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
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(spec.label)}\u5168\u6b77\u53f2\u53c3\u8207\u7387\u5206\u5e03\u76f4\u65b9\u5716">
  <style>
    .axis {{ stroke: #334155; stroke-width: 1; }}
    text {{ fill: #475569; font-size: 12px; font-family: "Microsoft JhengHei", Arial, sans-serif; }}
  </style>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" class="axis"/>
  <line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}" class="axis"/>
  {''.join(bars)}
  <text x="{left + chart_width / 2}" y="{height - 5}" text-anchor="middle">\u55ae\u6a94\u80a1\u7968{html.escape(spec.short_label)}\u5168\u6b77\u53f2\u6210\u4ea4\u53c3\u8207\u7387\u5340\u9593</text>
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
            f"<td>{html.escape(str(row.StartDate))}</td>"
            f"<td>{html.escape(str(row.EndDate))}</td>"
            f"<td>{int(row.TradingDays):,}</td>"
            f"<td>{pct(float(row.Participation))}</td>"
            f"<td>{number(float(row.ParticipantVolume))}</td>"
            f"<td>{pct(float(row.ParticipantVolumeShare))}</td>"
            f"<td>{pct(float(row.CumulativeParticipantVolumeShare))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def summary_table_rows(summary: pd.DataFrame, group_column: str) -> str:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row[group_column]))}</td>"
            f"<td>{int(row['StockCount']):,}</td>"
            f"<td>{pct(float(row['ShareOfAllParticipantVolume']))}</td>"
            f"<td>{pct(float(row['ShareOfTop80ParticipantVolume']))}</td>"
            f"<td>{pct(float(row['AvgParticipation']))}</td>"
            f"<td>{html.escape(str(row['TopStocks']))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def metric_cards(spec: ParticipantSpec, metrics: dict[str, float | int | str]) -> str:
    card_specs = [
        ("\u7d71\u8a08\u671f\u9593", f"{metrics['coverage_start']} ~ {metrics['coverage_end']}"),
        ("\u6a23\u672c\u6a94\u6578", f"{int(metrics['stock_count']):,}"),
        (f"\u5168\u6b77\u53f2{spec.short_label}\u53c3\u8207\u7387", pct(float(metrics["market_participation"]))),
        ("\u524d10\u6a94\u91cf\u5360\u6bd4", pct(float(metrics["top_10_share"]))),
        ("\u524d20\u6a94\u91cf\u5360\u6bd4", pct(float(metrics["top_20_share"]))),
        ("\u905450%\u9700\u8981\u6a94\u6578", f"{int(metrics['stocks_for_50_share']):,}"),
        ("\u905480%\u9700\u8981\u6a94\u6578", f"{int(metrics['stocks_for_80_share']):,}"),
        ("\u7b49\u6548\u5206\u6563\u6a94\u6578", number(float(metrics["effective_stock_count"]), 1)),
    ]
    return "\n".join(
        f'<div class="metric"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in card_specs
    )


def write_report(
    spec: ParticipantSpec,
    distribution: pd.DataFrame,
    bins: pd.DataFrame,
    broad_summary: pd.DataFrame,
    industry_summary: pd.DataFrame,
    metrics: dict[str, float | int | str],
) -> Path:
    DATA_VIZ_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = DATA_VIZ_ROOT / spec.report_filename
    top_80_count = int(metrics["stocks_for_80_share"])
    note_html = f"<p>{html.escape(spec.note)}</p>" if spec.note else ""
    evidence = (
        f"\u672c\u5831\u544a\u6539\u7528\u5168\u6b77\u53f2\u7d2f\u7a4d\u53e3\u5f91\uff1a"
        f"\u53ea\u7d71\u8a08\u6bcf\u6a94\u80a1\u7968\u50f9\u683c\u8207\u6cd5\u4eba\u8cc7\u6599\u90fd\u5b58\u5728\u7684\u65e5\u671f\u3002"
        f"\u5168\u6a23\u672c\u671f\u9593\u70ba {html.escape(str(metrics['coverage_start']))} ~ {html.escape(str(metrics['coverage_end']))}\uff0c"
        f"\u7d2f\u7a4d{html.escape(spec.short_label)}\u6210\u4ea4\u53c3\u8207\u7387\u70ba {pct(float(metrics['market_participation']))}\uff0c"
        f"\u524d 10 \u6a94\u5360 {pct(float(metrics['top_10_share']))}\uff0c"
        f"\u524d 20 \u6a94\u5360 {pct(float(metrics['top_20_share']))}\uff0c"
        f"\u7d2f\u7a4d\u5230 80% \u9700\u8981 {top_80_count} \u6a94\u3002"
    )
    broad_rows = summary_table_rows(broad_summary, "BroadCategory")
    industry_rows = summary_table_rows(industry_summary, INDUSTRY_GROUP)
    bins_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.Bin))}</td>"
        f"<td>{int(row.StockCount):,}</td>"
        f"<td>{pct(float(row.StockShare))}</td>"
        f"<td>{pct(float(row.ParticipantVolumeShare))}</td>"
        "</tr>"
        for row in bins.itertuples(index=False)
    )
    table_head = (
        f"<thead><tr><th>\u6392\u540d</th><th>\u4ee3\u865f</th><th>\u540d\u7a31</th>"
        f"<th>\u8d77\u59cb</th><th>\u6700\u65b0</th><th>\u7d71\u8a08\u65e5\u6578</th>"
        f"<th>\u53c3\u8207\u7387</th><th>\u4f30\u7b97\u6210\u4ea4\u91cf</th>"
        f"<th>\u5360\u5168\u5e02{html.escape(spec.short_label)}\u91cf</th><th>\u7d2f\u7a4d\u5360\u6bd4</th></tr></thead>"
    )
    report_path.write_text(
        f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(spec.label)}\u5168\u6b77\u53f2\u53c3\u8207\u96c6\u4e2d\u5ea6\u5831\u544a</title>
<style>
body {{ margin: 0; font-family: "Microsoft JhengHei", "Noto Sans CJK TC", Arial, sans-serif; color: #172033; background: #f8fafc; }}
main {{ max-width: 1240px; margin: 0 auto; padding: 22px; }}
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
td:nth-child(2), td:nth-child(3), th:nth-child(2), th:nth-child(3), td:nth-child(6), th:nth-child(6) {{ text-align: left; }}
a {{ color: #1d4ed8; text-decoration: none; }}
@media (max-width: 760px) {{ .summary {{ grid-template-columns: 1fr 1fr; }} main {{ padding: 14px; }} }}
</style>
</head>
<body>
<main>
<h1>{html.escape(spec.label)}\u5168\u6b77\u53f2\u53c3\u8207\u96c6\u4e2d\u5ea6\u5831\u544a</h1>
<div class="meta">\u4f86\u6e90\uff1a<code>data/price/</code> \u8207 <code>data/institutional/</code>\uff1b\u50c5\u7d71\u8a08\u5169\u8005\u91cd\u758a\u65e5\u671f</div>
<p>{evidence}</p>
{note_html}
<section class="summary">{metric_cards(spec, metrics)}</section>
<section class="panel">
<h2>{html.escape(spec.short_label)}\u5168\u6b77\u53f2\u6210\u4ea4\u91cf\u96c6\u4e2d\u66f2\u7dda</h2>
<p>\u85cd\u7dda\u8d8a\u9760\u5de6\u4e0a\uff0c\u4ee3\u8868\u8d8a\u5c11\u6578\u80a1\u7968\u8ca2\u737b\u8d8a\u591a{html.escape(spec.short_label)}\u6210\u4ea4\u91cf\uff1b\u7070\u8272\u865b\u7dda\u662f\u5b8c\u5168\u5e73\u5747\u5206\u5e03\u7684\u5c0d\u7167\u3002</p>
{line_chart_svg(distribution, spec)}
</section>
<section class="panel">
<h2>\u55ae\u6a94{html.escape(spec.short_label)}\u5168\u6b77\u53f2\u53c3\u8207\u7387\u5206\u5e03</h2>
{histogram_svg(bins, spec)}
</section>
<h2>{html.escape(spec.short_label)}\u5168\u6b77\u53f2\u6210\u4ea4\u91cf\u524d 20 \u540d</h2>
<table>
{table_head}
<tbody>
{table_rows(distribution, 20)}
</tbody>
</table>
<h2>\u7d2f\u7a4d\u9054 80% {html.escape(spec.short_label)}\u5168\u6b77\u53f2\u6210\u4ea4\u91cf\u80a1\u7968\u6392\u540d</h2>
<p>\u4e0b\u8868\u5217\u51fa\u5f9e\u7b2c 1 \u540d\u7d2f\u7a4d\u5230\u9054\u5168\u5e02{html.escape(spec.short_label)}\u5168\u6b77\u53f2\u6210\u4ea4\u91cf 80% \u7684\u6240\u6709\u80a1\u7968\uff1b\u672c\u6b21\u5171 {top_80_count:,} \u6a94\u3002</p>
<table>
{table_head}
<tbody>
{table_rows(distribution.head(top_80_count), None)}
</tbody>
</table>
<h2>{html.escape(spec.short_label)}\u5168\u6b77\u53f2\u504f\u597d\u985e\u578b\u7d71\u6574</h2>
<table>
<thead><tr><th>\u5927\u985e</th><th>\u6a94\u6578</th><th>\u5360\u5168\u5e02{html.escape(spec.short_label)}\u91cf</th><th>\u536080%\u6e05\u55ae{html.escape(spec.short_label)}\u91cf</th><th>\u5e73\u5747\u53c3\u8207\u7387</th><th>\u4ee3\u8868\u80a1</th></tr></thead>
<tbody>
{broad_rows}
</tbody>
</table>
<h2>\u539f\u59cb\u7522\u696d\u7fa4\u7d44\u660e\u7d30</h2>
<table>
<thead><tr><th>\u7522\u696d\u7fa4\u7d44</th><th>\u6a94\u6578</th><th>\u5360\u5168\u5e02{html.escape(spec.short_label)}\u91cf</th><th>\u536080%\u6e05\u55ae{html.escape(spec.short_label)}\u91cf</th><th>\u5e73\u5747\u53c3\u8207\u7387</th><th>\u4ee3\u8868\u80a1</th></tr></thead>
<tbody>
{industry_rows}
</tbody>
</table>
<h2>{html.escape(spec.short_label)}\u5168\u6b77\u53f2\u53c3\u8207\u7387\u5340\u9593\u6458\u8981</h2>
<table>
<thead><tr><th>\u5340\u9593</th><th>\u80a1\u7968\u6a94\u6578</th><th>\u6a94\u6578\u5360\u6bd4</th><th>{html.escape(spec.short_label)}\u6210\u4ea4\u91cf\u5360\u6bd4</th></tr></thead>
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


def build_report_for(spec: ParticipantSpec, stock_totals: pd.DataFrame) -> dict[str, object]:
    distribution = distribution_for(spec, stock_totals)
    bins = build_bins(distribution)
    total_participant_volume = float(distribution["ParticipantVolume"].sum())
    total_market_volume = float(stock_totals["TotalVolume"].sum())
    shares = distribution["ParticipantVolumeShare"]
    hhi = float((shares**2).sum())
    metrics: dict[str, float | int | str] = {
        "coverage_start": str(distribution["StartDate"].min()),
        "coverage_end": str(distribution["EndDate"].max()),
        "stock_count": int(len(distribution)),
        "total_stock_days": int(distribution["TradingDays"].sum()),
        "market_capacity": total_market_volume,
        "total_participant_volume": total_participant_volume,
        "market_participation": total_participant_volume / total_market_volume if total_market_volume else 0.0,
        "top_1_share": float(distribution.head(1)["ParticipantVolumeShare"].sum()),
        "top_3_share": float(distribution.head(3)["ParticipantVolumeShare"].sum()),
        "top_5_share": float(distribution.head(5)["ParticipantVolumeShare"].sum()),
        "top_10_share": float(distribution.head(10)["ParticipantVolumeShare"].sum()),
        "top_20_share": float(distribution.head(20)["ParticipantVolumeShare"].sum()),
        "top_50_share": float(distribution.head(50)["ParticipantVolumeShare"].sum()),
        "top_100_share": float(distribution.head(100)["ParticipantVolumeShare"].sum()),
        "stocks_for_50_share": stocks_to_reach_share(distribution, 0.50),
        "stocks_for_80_share": stocks_to_reach_share(distribution, 0.80),
        "stocks_for_90_share": stocks_to_reach_share(distribution, 0.90),
        "hhi": hhi,
        "effective_stock_count": 1 / hhi if hhi else 0,
        "participant_volume_gini": gini(distribution["ParticipantVolume"]),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    top_path = OUTPUT_ROOT / f"{spec.output_prefix}_top_stocks.csv"
    top_80_path = OUTPUT_ROOT / f"{spec.output_prefix}_top_80pct_stocks.csv"
    broad_summary_path = OUTPUT_ROOT / f"{spec.output_prefix}_broad_category_summary.csv"
    industry_summary_path = OUTPUT_ROOT / f"{spec.output_prefix}_industry_summary.csv"
    bins_path = OUTPUT_ROOT / f"{spec.output_prefix}_bins.csv"
    metrics_path = OUTPUT_ROOT / f"{spec.output_prefix}_metrics.csv"

    output_columns = [
        "Rank",
        "Code",
        "Name",
        "StartDate",
        "EndDate",
        "TradingDays",
        "TotalVolume",
        "Participant",
        "Participation",
        "ParticipantVolume",
        "ParticipantBuy",
        "ParticipantSell",
        "ParticipantNetRatio",
        "ParticipantPurity",
        "ParticipantVolumeShare",
        "CumulativeParticipantVolumeShare",
        "ReportPath",
    ]
    distribution[output_columns].to_csv(top_path, index=False, encoding="utf-8-sig")
    top_80_distribution = add_industry_columns(distribution.head(int(metrics["stocks_for_80_share"])).copy())
    top_80_distribution[output_columns + [INDUSTRY_GROUP, "BroadCategory"]].to_csv(
        top_80_path, index=False, encoding="utf-8-sig"
    )
    broad_summary = summarize_group(top_80_distribution, "BroadCategory", total_participant_volume)
    industry_summary = summarize_group(top_80_distribution, INDUSTRY_GROUP, total_participant_volume)
    broad_summary.to_csv(broad_summary_path, index=False, encoding="utf-8-sig")
    industry_summary.to_csv(industry_summary_path, index=False, encoding="utf-8-sig")
    bins.to_csv(bins_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"Metric": key, "Value": value} for key, value in metrics.items()]).to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )
    report_path = write_report(spec, distribution, bins, broad_summary, industry_summary, metrics)
    return {
        "spec": spec,
        "report_path": report_path,
        "top_path": top_path,
        "top_80_path": top_80_path,
        "broad_summary_path": broad_summary_path,
        "industry_summary_path": industry_summary_path,
        "bins_path": bins_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    specs = selected_specs(args.participants)
    stock_totals = build_stock_totals(args.limit)
    if stock_totals.empty:
        raise SystemExit("no overlapping price/institutional rows found")
    stock_totals_path = OUTPUT_ROOT / "historical_participant_stock_totals.csv"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stock_totals.to_csv(stock_totals_path, index=False, encoding="utf-8-sig")
    print(f"stock_totals={stock_totals_path}")
    for spec in specs:
        result = build_report_for(spec, stock_totals)
        metrics = result["metrics"]
        print(f"{spec.key} historical participant distribution report:")
        print(f"coverage={metrics['coverage_start']}..{metrics['coverage_end']}")
        print(f"stock_count={metrics['stock_count']}")
        print(f"market_participation={pct(float(metrics['market_participation']))}")
        print(f"top_10_share={pct(float(metrics['top_10_share']))}")
        print(f"top_20_share={pct(float(metrics['top_20_share']))}")
        print(f"stocks_for_80_share={metrics['stocks_for_80_share']}")
        print(f"report={result['report_path']}")
        print(f"top_80pct_stocks={result['top_80_path']}")


if __name__ == "__main__":
    main()
