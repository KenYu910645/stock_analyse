"""Rank TWSE listed stocks by daily day-trading heat.

The report answers which stocks day traders repeatedly concentrate in.  It
uses the canonical per-stock files under data/day_trading and writes a
self-contained HTML report plus CSV outputs.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from column_schema import read_csv_canonical
from downloader import price as price_downloader

DAY_TRADING_DIR = PROJECT_ROOT / "data" / "day_trading"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "day_trading_heat_rank"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "day_trading_heat_rank"

METRIC_COLUMNS = [
    "DayTradingVolume",
    "DayTradingBuyAmount",
    "DayTradingSellAmount",
    "DayTradingVolumeRatio",
    "DayTradingBuyAmountRatio",
    "DayTradingSellAmountRatio",
    "DayTradingTurnover",
    "DayTradingTurnoverRatio",
    "DayTradingAvgBuyPrice",
    "DayTradingAvgSellPrice",
    "DayTradingAvgSpreadRate",
    "DayTradingAmountImbalanceRatio",
    "IntradayRangeRate",
    "OpenCloseReturn",
    "DayTradingVolumeRatio20DayZScore",
    "DayTradingTurnover20DayZScore",
]

HEAT_WEIGHTS = {
    "DayTradingVolumeRatioRankPct": 0.35,
    "DayTradingTurnoverRatioRankPct": 0.35,
    "DayTradingTurnoverRankPct": 0.20,
    "DayTradingVolumeRatio20DayZScoreRankPct": 0.10,
}


@dataclass
class ReportConfig:
    top_n: int
    recent_days: int
    output_dir: Path
    viz_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily day-trading heat rankings.")
    parser.add_argument("--top-n", type=int, default=20, help="daily top N stocks to keep")
    parser.add_argument("--recent-days", type=int, default=60, help="trading days for recent summary")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    return parser.parse_args()


def load_universe() -> pd.DataFrame:
    metadata = read_csv_canonical(METADATA_PATH, dtype={"Code": str}).fillna("")
    required = {"Code", "Name", "Type", "Market", "Group"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {sorted(missing)}")

    mask = (
        metadata["Type"].astype(str).isin([price_downloader.COMMON_STOCK_TYPE, "STOCK"])
        & metadata["Market"].astype(str).eq(price_downloader.TWSE_MARKET)
    )
    universe = metadata.loc[mask, ["Code", "Name", "Group"]].copy()
    for column in ["Code", "Name", "Group"]:
        universe[column] = universe[column].astype(str).str.strip()
    universe["Group"] = universe["Group"].replace("", "未分類")
    return universe[universe["Code"].ne("")].drop_duplicates("Code")


def day_trading_path_for_code(code: str) -> Path | None:
    matches = sorted(DAY_TRADING_DIR.glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def load_stock_frame(code: str, name: str, group: str) -> pd.DataFrame | None:
    path = day_trading_path_for_code(code)
    if path is None:
        return None
    try:
        frame = read_csv_canonical(path, dtype=str).fillna("")
    except Exception as exc:
        print(f"skip {code}: {exc}")
        return None
    if frame.empty or "Date" not in frame.columns:
        return None

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).copy()
    if frame.empty:
        return None

    if "Code" not in frame.columns:
        frame["Code"] = code
    if "Name" not in frame.columns:
        frame["Name"] = name
    frame["Code"] = frame["Code"].astype(str).str.strip().replace("", code)
    frame["Name"] = frame["Name"].astype(str).str.strip().replace("", name)
    frame["Group"] = group or "未分類"

    for column in METRIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["DayTradingTurnover"].isna().all():
        frame["DayTradingTurnover"] = (
            frame["DayTradingBuyAmount"].fillna(0) + frame["DayTradingSellAmount"].fillna(0)
        ) / 2
    return frame[["Date", "Code", "Name", "Group"] + METRIC_COLUMNS]


def load_panel() -> pd.DataFrame:
    frames = []
    universe = load_universe()
    for _, row in universe.iterrows():
        frame = load_stock_frame(str(row["Code"]), str(row["Name"]), str(row["Group"]))
        if frame is not None:
            frames.append(frame)
        if frames and len(frames) % 100 == 0:
            print(f"loaded {len(frames)} stock panels")
    if not frames:
        raise RuntimeError("no day-trading panels loaded")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["Date", "Code"]).reset_index(drop=True)
    return panel


def compute_heat_ranks(panel: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = panel.copy()
    ranking_inputs = [
        "DayTradingVolumeRatio",
        "DayTradingTurnoverRatio",
        "DayTradingTurnover",
        "DayTradingVolumeRatio20DayZScore",
    ]
    for column in ranking_inputs:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        rank_column = f"{column}RankPct"
        valid_count = data.groupby("Date")[column].transform(lambda values: values.notna().sum())
        spread = data.groupby("Date")[column].transform(lambda values: values.max(skipna=True) - values.min(skipna=True))
        available = valid_count.gt(1) & spread.gt(0)
        data[rank_column] = data.groupby("Date")[column].rank(method="average", pct=True)
        data[rank_column] = data[rank_column].where(available)

    data["HeatScoreNumerator"] = 0.0
    data["HeatScoreWeight"] = 0.0
    data["HeatScoreMetricCount"] = 0
    for column, weight in HEAT_WEIGHTS.items():
        usable = data[column].notna()
        data.loc[usable, "HeatScoreNumerator"] += data.loc[usable, column] * weight
        data.loc[usable, "HeatScoreWeight"] += weight
        data.loc[usable, "HeatScoreMetricCount"] += 1
    data["HeatScore"] = safe_ratio(data["HeatScoreNumerator"], data["HeatScoreWeight"]) * 100
    data["HeatScore"] = data["HeatScore"].fillna(0)

    data["DailyStockCount"] = data.groupby("Date")["Code"].transform("nunique")
    data = data.sort_values(
        ["Date", "HeatScore", "DayTradingTurnover", "DayTradingVolumeRatio", "Code"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    data["HeatRank"] = data.groupby("Date").cumcount() + 1
    daily_top = data[data["HeatRank"].le(top_n)].copy()
    return data, daily_top


def summarize_daily(ranked: pd.DataFrame, daily_top: pd.DataFrame) -> pd.DataFrame:
    market = (
        ranked.groupby("Date", as_index=False)
        .agg(
            StockCount=("Code", "nunique"),
            MarketDayTradingTurnover=("DayTradingTurnover", "sum"),
            MarketAvgDayTradingVolumeRatio=("DayTradingVolumeRatio", "mean"),
            MarketAvgDayTradingTurnoverRatio=("DayTradingTurnoverRatio", "mean"),
        )
    )
    top = (
        daily_top.groupby("Date", as_index=False)
        .agg(
            TopCount=("Code", "size"),
            TopAvgHeatScore=("HeatScore", "mean"),
            TopAvgDayTradingVolumeRatio=("DayTradingVolumeRatio", "mean"),
            TopAvgDayTradingTurnoverRatio=("DayTradingTurnoverRatio", "mean"),
            TopDayTradingTurnover=("DayTradingTurnover", "sum"),
        )
    )
    summary = market.merge(top, on="Date", how="left")
    summary["TopDayTradingTurnoverShare"] = safe_ratio(
        summary["TopDayTradingTurnover"], summary["MarketDayTradingTurnover"]
    )
    return summary.sort_values("Date")


def summarize_stocks(ranked: pd.DataFrame, daily_top: pd.DataFrame) -> pd.DataFrame:
    observed = (
        ranked.groupby(["Code", "Name", "Group"], as_index=False)
        .agg(
            ObservedDayCount=("Date", "nunique"),
            AvgHeatScore=("HeatScore", "mean"),
            AvgDayTradingVolumeRatio=("DayTradingVolumeRatio", "mean"),
            AvgDayTradingTurnoverRatio=("DayTradingTurnoverRatio", "mean"),
            AvgDayTradingTurnover=("DayTradingTurnover", "mean"),
            TotalDayTradingTurnover=("DayTradingTurnover", "sum"),
        )
    )
    top = (
        daily_top.groupby(["Code", "Name", "Group"], as_index=False)
        .agg(
            Top20Count=("Date", "size"),
            Top1Count=("HeatRank", lambda values: int((values == 1).sum())),
            AvgTop20Rank=("HeatRank", "mean"),
            BestTop20Rank=("HeatRank", "min"),
            AvgTop20HeatScore=("HeatScore", "mean"),
            LatestTop20Date=("Date", "max"),
        )
    )
    latest_date = ranked["Date"].max()
    latest = ranked[ranked["Date"].eq(latest_date)][["Code", "HeatRank", "HeatScore"]].rename(
        columns={"HeatRank": "LatestHeatRank", "HeatScore": "LatestHeatScore"}
    )
    summary = observed.merge(top, on=["Code", "Name", "Group"], how="left").merge(latest, on="Code", how="left")
    fill_zero = ["Top20Count", "Top1Count"]
    summary[fill_zero] = summary[fill_zero].fillna(0).astype(int)
    summary["Top20Rate"] = safe_ratio(summary["Top20Count"], summary["ObservedDayCount"])
    summary["LatestTop20Date"] = pd.to_datetime(summary["LatestTop20Date"], errors="coerce")
    return summary.sort_values(
        ["Top20Count", "Top1Count", "AvgTop20Rank", "TotalDayTradingTurnover"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def summarize_recent_stocks(daily_top: pd.DataFrame, recent_days: int) -> pd.DataFrame:
    dates = sorted(daily_top["Date"].dropna().unique())
    recent_dates = set(dates[-recent_days:])
    recent = daily_top[daily_top["Date"].isin(recent_dates)].copy()
    if recent.empty:
        return pd.DataFrame()
    summary = (
        recent.groupby(["Code", "Name", "Group"], as_index=False)
        .agg(
            RecentTop20Count=("Date", "size"),
            RecentTop1Count=("HeatRank", lambda values: int((values == 1).sum())),
            RecentAvgTop20Rank=("HeatRank", "mean"),
            RecentAvgHeatScore=("HeatScore", "mean"),
            RecentAvgDayTradingVolumeRatio=("DayTradingVolumeRatio", "mean"),
            RecentAvgDayTradingTurnover=("DayTradingTurnover", "mean"),
            LatestTop20Date=("Date", "max"),
        )
    )
    summary["RecentTop20Rate"] = summary["RecentTop20Count"] / max(1, len(recent_dates))
    return summary.sort_values(
        ["RecentTop20Count", "RecentTop1Count", "RecentAvgTop20Rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def summarize_groups(daily_top: pd.DataFrame) -> pd.DataFrame:
    summary = (
        daily_top.groupby("Group", as_index=False)
        .agg(
            Top20Count=("Code", "size"),
            StockCount=("Code", "nunique"),
            Top1Count=("HeatRank", lambda values: int((values == 1).sum())),
            AvgTop20Rank=("HeatRank", "mean"),
            AvgHeatScore=("HeatScore", "mean"),
            AvgDayTradingVolumeRatio=("DayTradingVolumeRatio", "mean"),
            AvgDayTradingTurnoverRatio=("DayTradingTurnoverRatio", "mean"),
            AvgDayTradingTurnover=("DayTradingTurnover", "mean"),
        )
    )
    summary["Top20CountPerStock"] = safe_ratio(summary["Top20Count"], summary["StockCount"])
    return summary.sort_values(["Top20Count", "Top1Count"], ascending=[False, False]).reset_index(drop=True)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def write_outputs(
    output_dir: Path,
    daily_top: pd.DataFrame,
    stock_summary: pd.DataFrame,
    recent_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_columns = [
        "Date",
        "Code",
        "Name",
        "Group",
        "DayTradingVolume",
        "DayTradingBuyAmount",
        "DayTradingSellAmount",
        "DayTradingVolumeRatio",
        "DayTradingBuyAmountRatio",
        "DayTradingSellAmountRatio",
        "DayTradingTurnover",
        "DayTradingTurnoverRatio",
        "DayTradingAvgBuyPrice",
        "DayTradingAvgSellPrice",
        "DayTradingAvgSpreadRate",
        "DayTradingAmountImbalanceRatio",
        "IntradayRangeRate",
        "OpenCloseReturn",
        "DayTradingVolumeRatio20DayZScore",
        "DayTradingTurnover20DayZScore",
        "DayTradingVolumeRatioRankPct",
        "DayTradingTurnoverRatioRankPct",
        "DayTradingTurnoverRankPct",
        "DayTradingVolumeRatio20DayZScoreRankPct",
        "HeatScore",
        "HeatScoreWeight",
        "HeatScoreMetricCount",
        "DailyStockCount",
        "HeatRank",
    ]
    daily_top_out = daily_top[daily_columns].copy()
    daily_top_out.to_csv(output_dir / "daily_top20_heat_rank.csv", index=False, encoding="utf-8-sig")
    latest_date = daily_top["Date"].max()
    daily_top_out[daily_top_out["Date"].eq(latest_date)].to_csv(
        output_dir / "latest_top20_heat_rank.csv", index=False, encoding="utf-8-sig"
    )
    stock_summary.to_csv(output_dir / "stock_heat_summary.csv", index=False, encoding="utf-8-sig")
    recent_summary.to_csv(output_dir / "recent_stock_heat_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(output_dir / "group_heat_summary.csv", index=False, encoding="utf-8-sig")
    daily_summary.to_csv(output_dir / "daily_heat_summary.csv", index=False, encoding="utf-8-sig")


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


def compact(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    units = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]
    for scale, suffix in units:
        if value >= scale:
            return f"{sign}{value / scale:.{digits}f}{suffix}"
    return f"{sign}{value:,.0f}"


def format_cell(column: str, value: object, percent_columns: set[str], compact_columns: set[str]) -> str:
    if value is None or value == "" or pd.isna(value):
        return ""
    if column == "Date" or column.endswith("Date"):
        timestamp = pd.to_datetime(value, errors="coerce")
        return timestamp.strftime("%Y-%m-%d") if pd.notna(timestamp) else str(value)
    if column in percent_columns:
        return pct(float(value))
    if column in compact_columns:
        return compact(float(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return num(float(value))
    return str(value)


def html_table(
    df: pd.DataFrame,
    columns: Iterable[str],
    labels: dict[str, str],
    percent_columns: set[str] | None = None,
    compact_columns: set[str] | None = None,
    max_rows: int | None = None,
) -> str:
    percent_columns = percent_columns or set()
    compact_columns = compact_columns or set()
    clean = df[list(columns)].copy()
    if max_rows is not None:
        clean = clean.head(max_rows)
    rows = []
    for row in clean.replace([np.inf, -np.inf], np.nan).to_dict("records"):
        cells = []
        for column in columns:
            text = format_cell(column, row[column], percent_columns, compact_columns)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{html.escape(labels.get(column, column))}</th>" for column in columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def svg_horizontal_bars(
    df: pd.DataFrame,
    label_column: str,
    value_column: str,
    title: str,
    max_rows: int = 20,
    percent: bool = False,
) -> str:
    data = df.head(max_rows).copy()
    if data.empty:
        return ""
    width = 980
    row_h = 28
    left = 190
    right = 70
    top = 36
    bottom = 24
    height = top + bottom + row_h * len(data)
    max_v = max(0.01, float(data[value_column].max()))
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">']
    parts.append(f'<text x="0" y="20" font-size="16" font-weight="700">{html.escape(title)}</text>')
    for i, row in enumerate(data.to_dict("records")):
        y = top + i * row_h
        label = str(row[label_column])
        value = float(row[value_column]) if pd.notna(row[value_column]) else 0.0
        bar_w = (width - left - right) * value / max_v
        fill = "#2563eb" if i % 2 == 0 else "#16a34a"
        value_text = pct(value / 100, 1) if percent else num(value, 1)
        parts.append(f'<text x="0" y="{y + 16}" font-size="12">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="18" fill="{fill}" />')
        parts.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 14}" font-size="12">{html.escape(value_text)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def daily_payload(daily_top: pd.DataFrame) -> str:
    columns = [
        "HeatRank",
        "Code",
        "Name",
        "Group",
        "HeatScore",
        "DayTradingVolumeRatio",
        "DayTradingTurnoverRatio",
        "DayTradingTurnover",
        "DayTradingVolumeRatio20DayZScore",
        "DayTradingAvgSpreadRate",
        "IntradayRangeRate",
        "OpenCloseReturn",
    ]
    payload: dict[str, list[list[object]]] = {}
    for date, data in daily_top.groupby("Date"):
        key = pd.Timestamp(date).strftime("%Y-%m-%d")
        rows = []
        for row in data.sort_values("HeatRank")[columns].to_dict("records"):
            rows.append(
                [
                    int(row["HeatRank"]),
                    str(row["Code"]),
                    str(row["Name"]),
                    str(row["Group"]),
                    round(float(row["HeatScore"]), 2) if pd.notna(row["HeatScore"]) else None,
                    clean_number(row["DayTradingVolumeRatio"]),
                    clean_number(row["DayTradingTurnoverRatio"]),
                    clean_number(row["DayTradingTurnover"]),
                    clean_number(row["DayTradingVolumeRatio20DayZScore"]),
                    clean_number(row["DayTradingAvgSpreadRate"]),
                    clean_number(row["IntradayRangeRate"]),
                    clean_number(row["OpenCloseReturn"]),
                ]
            )
        payload[key] = rows
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def clean_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def write_html_report(
    viz_dir: Path,
    output_dir: Path,
    ranked: pd.DataFrame,
    daily_top: pd.DataFrame,
    stock_summary: pd.DataFrame,
    recent_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
    config: ReportConfig,
) -> Path:
    viz_dir.mkdir(parents=True, exist_ok=True)
    report_path = viz_dir / "index.html"
    start = ranked["Date"].min().strftime("%Y-%m-%d")
    end = ranked["Date"].max().strftime("%Y-%m-%d")
    latest_date = ranked["Date"].max().strftime("%Y-%m-%d")
    stock_count = int(ranked["Code"].nunique())
    date_count = int(ranked["Date"].nunique())
    top_rows = int(len(daily_top))
    labels = {
        "Date": "日期",
        "HeatRank": "熱度排名",
        "Code": "股票代碼",
        "Name": "公司簡稱",
        "Group": "產業",
        "HeatScore": "熱度分數",
        "DayTradingVolume": "當沖股數",
        "DayTradingVolumeRatio": "當沖股數占比",
        "DayTradingTurnover": "當沖成交值",
        "DayTradingTurnoverRatio": "當沖成交值占比",
        "DayTradingAvgSpreadRate": "當沖平均價差率",
        "DayTradingVolumeRatio20DayZScore": "當沖股數占比20日Z分數",
        "IntradayRangeRate": "日內振幅",
        "OpenCloseReturn": "開收報酬",
        "Top20Count": f"進入每日前{config.top_n}名次數",
        "Top20Rate": "進榜率",
        "Top1Count": "每日第1名次數",
        "AvgTop20Rank": "進榜平均名次",
        "BestTop20Rank": "最佳名次",
        "AvgTop20HeatScore": "進榜平均熱度",
        "AvgHeatScore": "全期間平均熱度",
        "AvgDayTradingVolumeRatio": "平均當沖股數占比",
        "AvgDayTradingTurnoverRatio": "平均當沖成交值占比",
        "AvgDayTradingTurnover": "平均當沖成交值",
        "TotalDayTradingTurnover": "累計當沖成交值",
        "LatestHeatRank": "最新日排名",
        "LatestHeatScore": "最新日熱度",
        "LatestTop20Date": "最近進榜日",
        "RecentTop20Count": f"近{config.recent_days}日進榜次數",
        "RecentTop20Rate": f"近{config.recent_days}日進榜率",
        "RecentTop1Count": f"近{config.recent_days}日第1名次數",
        "RecentAvgTop20Rank": f"近{config.recent_days}日平均名次",
        "RecentAvgHeatScore": f"近{config.recent_days}日平均熱度",
        "RecentAvgDayTradingVolumeRatio": f"近{config.recent_days}日平均當沖股數占比",
        "RecentAvgDayTradingTurnover": f"近{config.recent_days}日平均當沖成交值",
        "StockCount": "股票數",
        "Top20CountPerStock": "每檔平均進榜次數",
    }
    percent_columns = {
        "DayTradingVolumeRatio",
        "DayTradingTurnoverRatio",
        "DayTradingAvgSpreadRate",
        "IntradayRangeRate",
        "OpenCloseReturn",
        "Top20Rate",
        "RecentTop20Rate",
        "AvgDayTradingVolumeRatio",
        "AvgDayTradingTurnoverRatio",
    }
    compact_columns = {
        "DayTradingVolume",
        "DayTradingTurnover",
        "AvgDayTradingTurnover",
        "TotalDayTradingTurnover",
        "RecentAvgDayTradingTurnover",
    }
    daily_columns = [
        "HeatRank",
        "Code",
        "Name",
        "Group",
        "HeatScore",
        "DayTradingVolumeRatio",
        "DayTradingTurnoverRatio",
        "DayTradingTurnover",
        "DayTradingVolumeRatio20DayZScore",
        "DayTradingAvgSpreadRate",
    ]
    stock_columns = [
        "Code",
        "Name",
        "Group",
        "Top20Count",
        "Top20Rate",
        "Top1Count",
        "AvgTop20Rank",
        "AvgTop20HeatScore",
        "AvgDayTradingVolumeRatio",
        "AvgDayTradingTurnover",
        "TotalDayTradingTurnover",
        "LatestHeatRank",
    ]
    recent_columns = [
        "Code",
        "Name",
        "Group",
        "RecentTop20Count",
        "RecentTop20Rate",
        "RecentTop1Count",
        "RecentAvgTop20Rank",
        "RecentAvgHeatScore",
        "RecentAvgDayTradingVolumeRatio",
        "RecentAvgDayTradingTurnover",
        "LatestTop20Date",
    ]
    group_columns = [
        "Group",
        "Top20Count",
        "StockCount",
        "Top1Count",
        "AvgTop20Rank",
        "AvgHeatScore",
        "AvgDayTradingVolumeRatio",
        "AvgDayTradingTurnoverRatio",
        "AvgDayTradingTurnover",
        "Top20CountPerStock",
    ]
    latest_top = daily_top[daily_top["Date"].eq(ranked["Date"].max())].sort_values("HeatRank")
    latest_chart_data = latest_top.assign(Label=latest_top["Code"].astype(str) + " " + latest_top["Name"].astype(str))
    stock_chart_data = stock_summary.head(20).assign(
        Label=stock_summary.head(20)["Code"].astype(str) + " " + stock_summary.head(20)["Name"].astype(str)
    )
    group_chart_data = group_summary.head(15).rename(columns={"Group": "Label"})
    latest_chart = svg_horizontal_bars(latest_chart_data, "Label", "HeatScore", f"{latest_date} 每日前{config.top_n}名熱度分數")
    stock_chart = svg_horizontal_bars(stock_chart_data, "Label", "Top20Count", f"全期間最常進入每日前{config.top_n}名")
    group_chart = svg_horizontal_bars(group_chart_data, "Label", "Top20Count", "進榜次數最多的產業")
    payload = daily_payload(daily_top)

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>當沖熱度排行報告</title>
<style>
body {{ font-family: Arial, "Microsoft JhengHei", sans-serif; margin: 0; background: #f6f7f9; color: #172033; }}
header {{ background: #172033; color: #fff; padding: 24px 32px 18px; }}
h1 {{ margin: 0 0 8px; font-size: 25px; }}
.meta {{ color: #cbd5e1; font-size: 13px; line-height: 1.6; }}
main {{ padding: 24px 32px 40px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.card {{ background: #fff; border: 1px solid #dfe5ef; border-radius: 6px; padding: 14px 16px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
section {{ background: #fff; border: 1px solid #dfe5ef; border-radius: 6px; margin: 16px 0; padding: 18px; overflow-x: auto; }}
h2 {{ font-size: 18px; margin: 0 0 12px; }}
h3 {{ font-size: 15px; margin: 18px 0 10px; }}
p {{ line-height: 1.7; color: #334155; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5eaf2; padding: 8px 10px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4) {{ text-align: left; }}
th {{ background: #f2f5f9; color: #334155; }}
select {{ font: inherit; padding: 6px 10px; border: 1px solid #cbd5e1; border-radius: 4px; background: white; }}
.note {{ color: #59677c; font-size: 13px; line-height: 1.6; }}
.toolbar {{ display: flex; gap: 12px; align-items: center; margin: 8px 0 14px; flex-wrap: wrap; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }} main {{ padding: 18px; }} }}
</style>
</head>
<body>
<header>
<h1>當沖熱度排行報告</h1>
<div class="meta">資料範圍：{html.escape(start)} 到 {html.escape(end)}。每日排名只使用 TWSE 上市普通股且有當沖資料的股票。</div>
</header>
<main>
<div class="cards">
  <div class="card"><div class="label">股票數</div><div class="value">{stock_count:,}</div></div>
  <div class="card"><div class="label">交易日數</div><div class="value">{date_count:,}</div></div>
  <div class="card"><div class="label">每日前{config.top_n}名筆數</div><div class="value">{top_rows:,}</div></div>
  <div class="card"><div class="label">最新交易日</div><div class="value">{html.escape(latest_date)}</div></div>
</div>
<section>
<h2>熱度定義</h2>
<p>熱度分數是同一交易日內的橫斷面排名分數，不直接拿不同日期的原始數字硬比。基礎公式為：35% 當沖股數占比排名 + 35% 當沖成交值占比排名 + 20% 當沖成交值排名 + 10% 當沖股數占比20日Z分數排名。分數越高，代表該股票在當天同市場股票中越受當沖交易集中。</p>
<p class="note">若某交易日的某項指標全缺或所有股票數值相同，該指標會在當日排除，剩餘可用指標會重新分配權重。這個定義同時看「占比」和「絕對成交值」，避免只有低成交股票因占比高而排到最前面；CSV 仍保留各項原始指標，方便改用單一指標重排。</p>
</section>
<section>
<h2>每日前{config.top_n}名排行</h2>
<div class="toolbar">
  <label for="dateSelect">選擇交易日</label>
  <select id="dateSelect"></select>
</div>
<div id="dailyTable"></div>
</section>
<section>
<h2>最新交易日排行</h2>
{latest_chart}
{html_table(latest_top, daily_columns, labels, percent_columns, compact_columns)}
</section>
<section>
<h2>全期間最受當沖偏好的股票</h2>
{stock_chart}
{html_table(stock_summary, stock_columns, labels, percent_columns, compact_columns, max_rows=50)}
</section>
<section>
<h2>近{config.recent_days}個交易日熱度</h2>
{html_table(recent_summary, recent_columns, labels, percent_columns, compact_columns, max_rows=50)}
</section>
<section>
<h2>產業集中度</h2>
{group_chart}
{html_table(group_summary, group_columns, labels, percent_columns, compact_columns, max_rows=30)}
</section>
<section>
<h2>輸出檔案</h2>
<div class="note">
<a href="../../output/day_trading_heat_rank/daily_top20_heat_rank.csv">下載每日前{config.top_n}名完整排行</a><br>
<a href="../../output/day_trading_heat_rank/latest_top20_heat_rank.csv">下載最新交易日前{config.top_n}名</a><br>
<a href="../../output/day_trading_heat_rank/stock_heat_summary.csv">下載全期間股票熱度總表</a><br>
<a href="../../output/day_trading_heat_rank/recent_stock_heat_summary.csv">下載近{config.recent_days}日股票熱度總表</a><br>
<a href="../../output/day_trading_heat_rank/group_heat_summary.csv">下載產業集中度總表</a><br>
<a href="../../output/day_trading_heat_rank/daily_heat_summary.csv">下載每日市場熱度摘要</a>
</div>
</section>
</main>
<script id="dailyData" type="application/json">{payload}</script>
<script>
const dailyData = JSON.parse(document.getElementById("dailyData").textContent);
const dates = Object.keys(dailyData).sort();
const select = document.getElementById("dateSelect");
for (const date of dates) {{
  const option = document.createElement("option");
  option.value = date;
  option.textContent = date;
  select.appendChild(option);
}}
select.value = dates[dates.length - 1];

function fmtPct(value) {{
  if (value === null || Number.isNaN(value)) return "";
  return (value * 100).toFixed(2) + "%";
}}
function fmtNum(value, digits = 2) {{
  if (value === null || Number.isNaN(value)) return "";
  return Number(value).toLocaleString("zh-TW", {{ maximumFractionDigits: digits, minimumFractionDigits: digits }});
}}
function fmtCompact(value) {{
  if (value === null || Number.isNaN(value)) return "";
  return Number(value).toLocaleString("zh-TW", {{ notation: "compact", maximumFractionDigits: 2 }});
}}
function renderDaily() {{
  const rows = dailyData[select.value] || [];
  const headers = ["排名", "股票代碼", "公司簡稱", "產業", "熱度分數", "當沖股數占比", "當沖成交值占比", "當沖成交值", "20日Z分數", "平均價差率", "日內振幅", "開收報酬"];
  let html = "<table><thead><tr>" + headers.map(h => `<th>${{h}}</th>`).join("") + "</tr></thead><tbody>";
  for (const row of rows) {{
    html += "<tr>";
    html += `<td>${{row[0]}}</td><td>${{row[1]}}</td><td>${{row[2]}}</td><td>${{row[3]}}</td>`;
    html += `<td>${{fmtNum(row[4])}}</td><td>${{fmtPct(row[5])}}</td><td>${{fmtPct(row[6])}}</td><td>${{fmtCompact(row[7])}}</td>`;
    html += `<td>${{fmtNum(row[8])}}</td><td>${{fmtPct(row[9])}}</td><td>${{fmtPct(row[10])}}</td><td>${{fmtPct(row[11])}}</td>`;
    html += "</tr>";
  }}
  html += "</tbody></table>";
  document.getElementById("dailyTable").innerHTML = html;
}}
select.addEventListener("change", renderDaily);
renderDaily();
</script>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.recent_days <= 0:
        raise ValueError("--recent-days must be positive")
    config = ReportConfig(
        top_n=args.top_n,
        recent_days=args.recent_days,
        output_dir=args.output_dir,
        viz_dir=args.viz_dir,
    )

    panel = load_panel()
    ranked, daily_top = compute_heat_ranks(panel, config.top_n)
    daily_summary = summarize_daily(ranked, daily_top)
    stock_summary = summarize_stocks(ranked, daily_top)
    recent_summary = summarize_recent_stocks(daily_top, config.recent_days)
    group_summary = summarize_groups(daily_top)
    write_outputs(config.output_dir, daily_top, stock_summary, recent_summary, group_summary, daily_summary)
    report_path = write_html_report(
        config.viz_dir,
        config.output_dir,
        ranked,
        daily_top,
        stock_summary,
        recent_summary,
        group_summary,
        daily_summary,
        config,
    )

    print(f"panel rows: {len(panel):,}")
    print(f"daily top rows: {len(daily_top):,}")
    print(f"report: {report_path}")
    print(f"output: {config.output_dir}")


if __name__ == "__main__":
    main()
