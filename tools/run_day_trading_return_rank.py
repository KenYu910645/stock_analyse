"""Rank daily positive and negative aggregate day-trading returns.

The return proxy is DayTradingAvgSpreadRate, calculated from aggregate TWSE
day-trading average sell price versus average buy price.  It is a gross
aggregate spread, not individual trader P/L and not net of fees or tax.
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
OUTPUT_DIR = PROJECT_ROOT / "output" / "day_trading_return_rank"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "day_trading_return_rank"

METRIC_COLUMNS = [
    "DayTradingVolume",
    "DayTradingBuyAmount",
    "DayTradingSellAmount",
    "DayTradingVolumeRatio",
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


@dataclass
class ReportConfig:
    top_n: int
    recent_days: int
    output_dir: Path
    viz_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily positive/negative day-trading return rankings.")
    parser.add_argument("--top-n", type=int, default=20, help="daily top N stocks to keep for each side")
    parser.add_argument("--recent-days", type=int, default=60, help="trading days for recent summaries")
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


def rank_daily_returns(panel: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = panel.copy()
    ranked = ranked.dropna(subset=["DayTradingAvgSpreadRate"]).copy()
    ranked = ranked[ranked["DayTradingVolume"].fillna(0).gt(0)].copy()
    ranked["ReturnSide"] = np.select(
        [ranked["DayTradingAvgSpreadRate"].gt(0), ranked["DayTradingAvgSpreadRate"].lt(0)],
        ["正報酬", "負報酬"],
        default="零報酬",
    )

    positive = ranked[ranked["DayTradingAvgSpreadRate"].gt(0)].copy()
    positive = positive.sort_values(
        ["Date", "DayTradingAvgSpreadRate", "DayTradingTurnover", "DayTradingVolumeRatio", "Code"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    positive["ReturnRank"] = positive.groupby("Date").cumcount() + 1
    positive_top = positive[positive["ReturnRank"].le(top_n)].copy()

    negative = ranked[ranked["DayTradingAvgSpreadRate"].lt(0)].copy()
    negative = negative.sort_values(
        ["Date", "DayTradingAvgSpreadRate", "DayTradingTurnover", "DayTradingVolumeRatio", "Code"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)
    negative["ReturnRank"] = negative.groupby("Date").cumcount() + 1
    negative_top = negative[negative["ReturnRank"].le(top_n)].copy()
    return ranked, positive_top, negative_top


def summarize_daily(ranked: pd.DataFrame, positive_top: pd.DataFrame, negative_top: pd.DataFrame) -> pd.DataFrame:
    summary = (
        ranked.groupby("Date", as_index=False)
        .agg(
            StockCount=("Code", "nunique"),
            PositiveStockCount=("ReturnSide", lambda values: int((values == "正報酬").sum())),
            NegativeStockCount=("ReturnSide", lambda values: int((values == "負報酬").sum())),
            ZeroStockCount=("ReturnSide", lambda values: int((values == "零報酬").sum())),
            MeanDayTradingAvgSpreadRate=("DayTradingAvgSpreadRate", "mean"),
            MedianDayTradingAvgSpreadRate=("DayTradingAvgSpreadRate", "median"),
            MarketDayTradingTurnover=("DayTradingTurnover", "sum"),
        )
        .sort_values("Date")
    )
    summary["PositiveRate"] = safe_ratio(summary["PositiveStockCount"], summary["StockCount"])
    summary["NegativeRate"] = safe_ratio(summary["NegativeStockCount"], summary["StockCount"])
    top_counts = (
        positive_top.groupby("Date", as_index=False)
        .agg(PositiveTopCount=("Code", "size"), PositiveTopAvgReturn=("DayTradingAvgSpreadRate", "mean"))
        .merge(
            negative_top.groupby("Date", as_index=False).agg(
                NegativeTopCount=("Code", "size"), NegativeTopAvgReturn=("DayTradingAvgSpreadRate", "mean")
            ),
            on="Date",
            how="outer",
        )
    )
    return summary.merge(top_counts, on="Date", how="left")


def summarize_stocks(ranked: pd.DataFrame, positive_top: pd.DataFrame, negative_top: pd.DataFrame) -> pd.DataFrame:
    base = (
        ranked.groupby(["Code", "Name", "Group"], as_index=False)
        .agg(
            ObservedDayCount=("Date", "nunique"),
            PositiveDayCount=("ReturnSide", lambda values: int((values == "正報酬").sum())),
            NegativeDayCount=("ReturnSide", lambda values: int((values == "負報酬").sum())),
            AvgDayTradingAvgSpreadRate=("DayTradingAvgSpreadRate", "mean"),
            MedianDayTradingAvgSpreadRate=("DayTradingAvgSpreadRate", "median"),
            AvgDayTradingTurnover=("DayTradingTurnover", "mean"),
            TotalDayTradingTurnover=("DayTradingTurnover", "sum"),
            AvgDayTradingVolumeRatio=("DayTradingVolumeRatio", "mean"),
        )
    )
    base["PositiveRate"] = safe_ratio(base["PositiveDayCount"], base["ObservedDayCount"])
    base["NegativeRate"] = safe_ratio(base["NegativeDayCount"], base["ObservedDayCount"])

    pos = (
        positive_top.groupby(["Code", "Name", "Group"], as_index=False)
        .agg(
            PositiveTop20Count=("Date", "size"),
            PositiveTop1Count=("ReturnRank", lambda values: int((values == 1).sum())),
            AvgPositiveTop20Rank=("ReturnRank", "mean"),
            AvgPositiveTop20Return=("DayTradingAvgSpreadRate", "mean"),
            LatestPositiveTop20Date=("Date", "max"),
        )
    )
    neg = (
        negative_top.groupby(["Code", "Name", "Group"], as_index=False)
        .agg(
            NegativeTop20Count=("Date", "size"),
            NegativeTop1Count=("ReturnRank", lambda values: int((values == 1).sum())),
            AvgNegativeTop20Rank=("ReturnRank", "mean"),
            AvgNegativeTop20Return=("DayTradingAvgSpreadRate", "mean"),
            LatestNegativeTop20Date=("Date", "max"),
        )
    )
    summary = base.merge(pos, on=["Code", "Name", "Group"], how="left").merge(neg, on=["Code", "Name", "Group"], how="left")
    count_columns = ["PositiveTop20Count", "PositiveTop1Count", "NegativeTop20Count", "NegativeTop1Count"]
    summary[count_columns] = summary[count_columns].fillna(0).astype(int)
    summary["PositiveTop20Rate"] = safe_ratio(summary["PositiveTop20Count"], summary["ObservedDayCount"])
    summary["NegativeTop20Rate"] = safe_ratio(summary["NegativeTop20Count"], summary["ObservedDayCount"])
    return summary.sort_values(
        ["PositiveTop20Count", "PositiveTop1Count", "AvgPositiveTop20Rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def summarize_recent(top: pd.DataFrame, recent_days: int, side_prefix: str) -> pd.DataFrame:
    dates = sorted(top["Date"].dropna().unique())
    recent_dates = set(dates[-recent_days:])
    recent = top[top["Date"].isin(recent_dates)].copy()
    if recent.empty:
        return pd.DataFrame()
    summary = (
        recent.groupby(["Code", "Name", "Group"], as_index=False)
        .agg(
            RecentTop20Count=("Date", "size"),
            RecentTop1Count=("ReturnRank", lambda values: int((values == 1).sum())),
            RecentAvgTop20Rank=("ReturnRank", "mean"),
            RecentAvgReturn=("DayTradingAvgSpreadRate", "mean"),
            RecentAvgTurnover=("DayTradingTurnover", "mean"),
            LatestTop20Date=("Date", "max"),
        )
        .sort_values(["RecentTop20Count", "RecentTop1Count", "RecentAvgTop20Rank"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    summary["RecentTop20Rate"] = summary["RecentTop20Count"] / max(1, len(recent_dates))
    summary.insert(0, "Side", side_prefix)
    return summary


def summarize_groups(top: pd.DataFrame, side: str) -> pd.DataFrame:
    summary = (
        top.groupby("Group", as_index=False)
        .agg(
            Top20Count=("Code", "size"),
            StockCount=("Code", "nunique"),
            Top1Count=("ReturnRank", lambda values: int((values == 1).sum())),
            AvgTop20Rank=("ReturnRank", "mean"),
            AvgReturn=("DayTradingAvgSpreadRate", "mean"),
            AvgTurnover=("DayTradingTurnover", "mean"),
            AvgDayTradingVolumeRatio=("DayTradingVolumeRatio", "mean"),
        )
        .sort_values(["Top20Count", "Top1Count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    summary.insert(0, "Side", side)
    return summary


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def output_columns() -> list[str]:
    return [
        "Date",
        "ReturnRank",
        "Code",
        "Name",
        "Group",
        "DayTradingAvgSpreadRate",
        "DayTradingAvgBuyPrice",
        "DayTradingAvgSellPrice",
        "DayTradingVolume",
        "DayTradingTurnover",
        "DayTradingVolumeRatio",
        "DayTradingTurnoverRatio",
        "DayTradingAmountImbalanceRatio",
        "IntradayRangeRate",
        "OpenCloseReturn",
        "DayTradingVolumeRatio20DayZScore",
        "DayTradingTurnover20DayZScore",
    ]


def write_outputs(
    output_dir: Path,
    positive_top: pd.DataFrame,
    negative_top: pd.DataFrame,
    stock_summary: pd.DataFrame,
    recent_positive: pd.DataFrame,
    recent_negative: pd.DataFrame,
    group_summary: pd.DataFrame,
    daily_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = output_columns()
    positive_top[columns].to_csv(output_dir / "daily_positive_return_top20.csv", index=False, encoding="utf-8-sig")
    negative_top[columns].to_csv(output_dir / "daily_negative_return_top20.csv", index=False, encoding="utf-8-sig")
    latest_date = max(positive_top["Date"].max(), negative_top["Date"].max())
    positive_top[positive_top["Date"].eq(latest_date)][columns].to_csv(
        output_dir / "latest_positive_return_top20.csv", index=False, encoding="utf-8-sig"
    )
    negative_top[negative_top["Date"].eq(latest_date)][columns].to_csv(
        output_dir / "latest_negative_return_top20.csv", index=False, encoding="utf-8-sig"
    )
    stock_summary.to_csv(output_dir / "stock_return_rank_summary.csv", index=False, encoding="utf-8-sig")
    recent_positive.to_csv(output_dir / "recent_positive_return_summary.csv", index=False, encoding="utf-8-sig")
    recent_negative.to_csv(output_dir / "recent_negative_return_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(output_dir / "group_return_rank_summary.csv", index=False, encoding="utf-8-sig")
    daily_summary.to_csv(output_dir / "daily_return_rank_summary.csv", index=False, encoding="utf-8-sig")


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
    for scale, suffix in [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]:
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
    data = df[list(columns)].copy()
    if max_rows is not None:
        data = data.head(max_rows)
    rows = []
    for row in data.replace([np.inf, -np.inf], np.nan).to_dict("records"):
        cells = []
        for column in columns:
            text = format_cell(column, row[column], percent_columns, compact_columns)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{html.escape(labels.get(column, column))}</th>" for column in columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def svg_horizontal_bars(df: pd.DataFrame, label_column: str, value_column: str, title: str, positive: bool) -> str:
    data = df.head(20).copy()
    if data.empty:
        return ""
    width = 980
    row_h = 28
    left = 190
    right = 80
    top = 36
    bottom = 24
    height = top + bottom + row_h * len(data)
    values = data[value_column].astype(float).abs()
    max_v = max(0.001, float(values.max()))
    fill = "#16a34a" if positive else "#dc2626"
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">']
    parts.append(f'<text x="0" y="20" font-size="16" font-weight="700">{html.escape(title)}</text>')
    for i, row in enumerate(data.to_dict("records")):
        y = top + i * row_h
        label = str(row[label_column])
        value = float(row[value_column]) if pd.notna(row[value_column]) else 0.0
        bar_w = (width - left - right) * abs(value) / max_v
        parts.append(f'<text x="0" y="{y + 16}" font-size="12">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="18" fill="{fill}" />')
        parts.append(f'<text x="{left + bar_w + 6:.1f}" y="{y + 14}" font-size="12">{html.escape(pct(value))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def clean_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def daily_payload(positive_top: pd.DataFrame, negative_top: pd.DataFrame) -> str:
    columns = [
        "ReturnRank",
        "Code",
        "Name",
        "Group",
        "DayTradingAvgSpreadRate",
        "DayTradingAvgBuyPrice",
        "DayTradingAvgSellPrice",
        "DayTradingVolume",
        "DayTradingTurnover",
        "DayTradingVolumeRatio",
        "DayTradingTurnoverRatio",
        "IntradayRangeRate",
        "OpenCloseReturn",
    ]
    payload: dict[str, dict[str, list[list[object]]]] = {}
    for side, frame in [("positive", positive_top), ("negative", negative_top)]:
        for date, data in frame.groupby("Date"):
            key = pd.Timestamp(date).strftime("%Y-%m-%d")
            payload.setdefault(key, {"positive": [], "negative": []})
            for row in data.sort_values("ReturnRank")[columns].to_dict("records"):
                payload[key][side].append(
                    [
                        int(row["ReturnRank"]),
                        str(row["Code"]),
                        str(row["Name"]),
                        str(row["Group"]),
                        clean_number(row["DayTradingAvgSpreadRate"]),
                        clean_number(row["DayTradingAvgBuyPrice"]),
                        clean_number(row["DayTradingAvgSellPrice"]),
                        clean_number(row["DayTradingVolume"]),
                        clean_number(row["DayTradingTurnover"]),
                        clean_number(row["DayTradingVolumeRatio"]),
                        clean_number(row["DayTradingTurnoverRatio"]),
                        clean_number(row["IntradayRangeRate"]),
                        clean_number(row["OpenCloseReturn"]),
                    ]
                )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_html_report(
    viz_dir: Path,
    output_dir: Path,
    ranked: pd.DataFrame,
    positive_top: pd.DataFrame,
    negative_top: pd.DataFrame,
    stock_summary: pd.DataFrame,
    recent_positive: pd.DataFrame,
    recent_negative: pd.DataFrame,
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
    labels = {
        "Date": "日期",
        "ReturnRank": "排名",
        "Code": "股票代碼",
        "Name": "公司簡稱",
        "Group": "產業",
        "Side": "方向",
        "DayTradingAvgSpreadRate": "當沖平均價差率",
        "DayTradingAvgBuyPrice": "當沖平均買進價",
        "DayTradingAvgSellPrice": "當沖平均賣出價",
        "DayTradingVolume": "當沖股數",
        "DayTradingTurnover": "當沖成交值",
        "DayTradingVolumeRatio": "當沖股數占比",
        "DayTradingTurnoverRatio": "當沖成交值占比",
        "IntradayRangeRate": "日內振幅",
        "OpenCloseReturn": "開收報酬",
        "ObservedDayCount": "樣本日數",
        "PositiveDayCount": "正報酬日數",
        "NegativeDayCount": "負報酬日數",
        "PositiveRate": "正報酬比例",
        "NegativeRate": "負報酬比例",
        "AvgDayTradingAvgSpreadRate": "平均價差率",
        "MedianDayTradingAvgSpreadRate": "價差率中位數",
        "AvgDayTradingTurnover": "平均當沖成交值",
        "TotalDayTradingTurnover": "累計當沖成交值",
        "AvgDayTradingVolumeRatio": "平均當沖股數占比",
        "PositiveTop20Count": "正報酬前20名次數",
        "PositiveTop1Count": "正報酬第1名次數",
        "AvgPositiveTop20Rank": "正報酬平均名次",
        "AvgPositiveTop20Return": "正報酬進榜平均價差率",
        "LatestPositiveTop20Date": "最近正報酬進榜日",
        "NegativeTop20Count": "負報酬前20名次數",
        "NegativeTop1Count": "負報酬第1名次數",
        "AvgNegativeTop20Rank": "負報酬平均名次",
        "AvgNegativeTop20Return": "負報酬進榜平均價差率",
        "LatestNegativeTop20Date": "最近負報酬進榜日",
        "RecentTop20Count": f"近{config.recent_days}日進榜次數",
        "RecentTop1Count": f"近{config.recent_days}日第1名次數",
        "RecentAvgTop20Rank": f"近{config.recent_days}日平均名次",
        "RecentAvgReturn": f"近{config.recent_days}日進榜平均價差率",
        "RecentAvgTurnover": f"近{config.recent_days}日平均當沖成交值",
        "LatestTop20Date": "最近進榜日",
        "RecentTop20Rate": f"近{config.recent_days}日進榜率",
        "Top20Count": "進榜次數",
        "StockCount": "股票數",
        "Top1Count": "第1名次數",
        "AvgTop20Rank": "平均名次",
        "AvgReturn": "平均價差率",
        "AvgTurnover": "平均當沖成交值",
    }
    percent_columns = {
        "DayTradingAvgSpreadRate",
        "DayTradingVolumeRatio",
        "DayTradingTurnoverRatio",
        "IntradayRangeRate",
        "OpenCloseReturn",
        "PositiveRate",
        "NegativeRate",
        "AvgDayTradingAvgSpreadRate",
        "MedianDayTradingAvgSpreadRate",
        "AvgDayTradingVolumeRatio",
        "AvgPositiveTop20Return",
        "AvgNegativeTop20Return",
        "RecentAvgReturn",
        "RecentTop20Rate",
        "AvgReturn",
    }
    compact_columns = {
        "DayTradingVolume",
        "DayTradingTurnover",
        "AvgDayTradingTurnover",
        "TotalDayTradingTurnover",
        "RecentAvgTurnover",
        "AvgTurnover",
    }
    daily_columns = [
        "ReturnRank",
        "Code",
        "Name",
        "Group",
        "DayTradingAvgSpreadRate",
        "DayTradingAvgBuyPrice",
        "DayTradingAvgSellPrice",
        "DayTradingVolume",
        "DayTradingTurnover",
        "DayTradingVolumeRatio",
        "IntradayRangeRate",
        "OpenCloseReturn",
    ]
    stock_columns = [
        "Code",
        "Name",
        "Group",
        "ObservedDayCount",
        "PositiveRate",
        "AvgDayTradingAvgSpreadRate",
        "PositiveTop20Count",
        "PositiveTop1Count",
        "AvgPositiveTop20Rank",
        "AvgPositiveTop20Return",
        "NegativeTop20Count",
        "AvgNegativeTop20Return",
        "TotalDayTradingTurnover",
    ]
    negative_stock_columns = [
        "Code",
        "Name",
        "Group",
        "ObservedDayCount",
        "NegativeRate",
        "AvgDayTradingAvgSpreadRate",
        "NegativeTop20Count",
        "NegativeTop1Count",
        "AvgNegativeTop20Rank",
        "AvgNegativeTop20Return",
        "PositiveTop20Count",
        "AvgPositiveTop20Return",
        "TotalDayTradingTurnover",
    ]
    recent_columns = [
        "Code",
        "Name",
        "Group",
        "RecentTop20Count",
        "RecentTop20Rate",
        "RecentTop1Count",
        "RecentAvgTop20Rank",
        "RecentAvgReturn",
        "RecentAvgTurnover",
        "LatestTop20Date",
    ]
    group_columns = [
        "Side",
        "Group",
        "Top20Count",
        "StockCount",
        "Top1Count",
        "AvgTop20Rank",
        "AvgReturn",
        "AvgTurnover",
        "AvgDayTradingVolumeRatio",
    ]
    latest_positive = positive_top[positive_top["Date"].eq(ranked["Date"].max())].sort_values("ReturnRank")
    latest_negative = negative_top[negative_top["Date"].eq(ranked["Date"].max())].sort_values("ReturnRank")
    positive_long = stock_summary.sort_values(
        ["PositiveTop20Count", "PositiveTop1Count", "AvgPositiveTop20Rank"], ascending=[False, False, True]
    )
    negative_long = stock_summary.sort_values(
        ["NegativeTop20Count", "NegativeTop1Count", "AvgNegativeTop20Rank"], ascending=[False, False, True]
    )
    latest_positive_chart = svg_horizontal_bars(
        latest_positive.assign(Label=latest_positive["Code"].astype(str) + " " + latest_positive["Name"].astype(str)),
        "Label",
        "DayTradingAvgSpreadRate",
        f"{latest_date} 正報酬前{config.top_n}名",
        positive=True,
    )
    latest_negative_chart = svg_horizontal_bars(
        latest_negative.assign(Label=latest_negative["Code"].astype(str) + " " + latest_negative["Name"].astype(str)),
        "Label",
        "DayTradingAvgSpreadRate",
        f"{latest_date} 負報酬前{config.top_n}名",
        positive=False,
    )
    payload = daily_payload(positive_top, negative_top)

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>當沖正負報酬排行</title>
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
.split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 1000px) {{ .cards, .split {{ grid-template-columns: 1fr; }} main {{ padding: 18px; }} }}
</style>
</head>
<body>
<header>
<h1>當沖正負報酬排行</h1>
<div class="meta">資料範圍：{html.escape(start)} 到 {html.escape(end)}。每日分別列出當沖平均價差率最高與最低的前{config.top_n}名。</div>
</header>
<main>
<div class="cards">
  <div class="card"><div class="label">股票數</div><div class="value">{stock_count:,}</div></div>
  <div class="card"><div class="label">交易日數</div><div class="value">{date_count:,}</div></div>
  <div class="card"><div class="label">最新交易日</div><div class="value">{html.escape(latest_date)}</div></div>
  <div class="card"><div class="label">排行口徑</div><div class="value">前{config.top_n}名</div></div>
</div>
<section>
<h2>報酬定義</h2>
<p>本報告用「當沖平均價差率」作為當天當沖交易的粗估毛報酬：當沖平均賣出價格高於平均買進價格時為正，低於平均買進價格時為負。這不是逐筆交易者真實損益，也不含手續費、證交稅與滑價。</p>
<p class="note">每日正報酬榜按當沖平均價差率由高到低排序；每日負報酬榜按當沖平均價差率由低到高排序。同分時用當沖成交值與當沖股數占比排序。</p>
</section>
<section>
<h2>每日正負報酬前{config.top_n}名</h2>
<div class="toolbar">
  <label for="dateSelect">選擇交易日</label>
  <select id="dateSelect"></select>
</div>
<div class="split">
  <div><h3>正報酬前{config.top_n}名</h3><div id="positiveTable"></div></div>
  <div><h3>負報酬前{config.top_n}名</h3><div id="negativeTable"></div></div>
</div>
</section>
<section>
<h2>最新交易日排行</h2>
<div class="split">
  <div>{latest_positive_chart}{html_table(latest_positive, daily_columns, labels, percent_columns, compact_columns)}</div>
  <div>{latest_negative_chart}{html_table(latest_negative, daily_columns, labels, percent_columns, compact_columns)}</div>
</div>
</section>
<section>
<h2>全期間正報酬常勝股</h2>
{html_table(positive_long, stock_columns, labels, percent_columns, compact_columns, max_rows=50)}
</section>
<section>
<h2>全期間負報酬常見股</h2>
{html_table(negative_long, negative_stock_columns, labels, percent_columns, compact_columns, max_rows=50)}
</section>
<section>
<h2>近{config.recent_days}個交易日正報酬排行</h2>
{html_table(recent_positive, recent_columns, labels, percent_columns, compact_columns, max_rows=50)}
</section>
<section>
<h2>近{config.recent_days}個交易日負報酬排行</h2>
{html_table(recent_negative, recent_columns, labels, percent_columns, compact_columns, max_rows=50)}
</section>
<section>
<h2>產業分布</h2>
{html_table(group_summary, group_columns, labels, percent_columns, compact_columns, max_rows=50)}
</section>
<section>
<h2>輸出檔案</h2>
<div class="note">
<a href="../../output/day_trading_return_rank/daily_positive_return_top20.csv">下載每日正報酬前{config.top_n}名</a><br>
<a href="../../output/day_trading_return_rank/daily_negative_return_top20.csv">下載每日負報酬前{config.top_n}名</a><br>
<a href="../../output/day_trading_return_rank/latest_positive_return_top20.csv">下載最新日正報酬前{config.top_n}名</a><br>
<a href="../../output/day_trading_return_rank/latest_negative_return_top20.csv">下載最新日負報酬前{config.top_n}名</a><br>
<a href="../../output/day_trading_return_rank/stock_return_rank_summary.csv">下載股票正負報酬總表</a><br>
<a href="../../output/day_trading_return_rank/recent_positive_return_summary.csv">下載近{config.recent_days}日正報酬總表</a><br>
<a href="../../output/day_trading_return_rank/recent_negative_return_summary.csv">下載近{config.recent_days}日負報酬總表</a><br>
<a href="../../output/day_trading_return_rank/group_return_rank_summary.csv">下載產業分布總表</a><br>
<a href="../../output/day_trading_return_rank/daily_return_rank_summary.csv">下載每日市場正負報酬摘要</a>
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
function renderTable(targetId, rows) {{
  const headers = ["排名", "股票代碼", "公司簡稱", "產業", "平均價差率", "平均買進價", "平均賣出價", "當沖股數", "當沖成交值", "當沖股數占比", "當沖成交值占比", "日內振幅", "開收報酬"];
  let html = "<table><thead><tr>" + headers.map(h => `<th>${{h}}</th>`).join("") + "</tr></thead><tbody>";
  for (const row of rows) {{
    html += "<tr>";
    html += `<td>${{row[0]}}</td><td>${{row[1]}}</td><td>${{row[2]}}</td><td>${{row[3]}}</td>`;
    html += `<td>${{fmtPct(row[4])}}</td><td>${{fmtNum(row[5])}}</td><td>${{fmtNum(row[6])}}</td>`;
    html += `<td>${{fmtCompact(row[7])}}</td><td>${{fmtCompact(row[8])}}</td>`;
    html += `<td>${{fmtPct(row[9])}}</td><td>${{fmtPct(row[10])}}</td><td>${{fmtPct(row[11])}}</td><td>${{fmtPct(row[12])}}</td>`;
    html += "</tr>";
  }}
  html += "</tbody></table>";
  document.getElementById(targetId).innerHTML = html;
}}
function renderDaily() {{
  const data = dailyData[select.value] || {{ positive: [], negative: [] }};
  renderTable("positiveTable", data.positive || []);
  renderTable("negativeTable", data.negative || []);
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
    config = ReportConfig(args.top_n, args.recent_days, args.output_dir, args.viz_dir)
    panel = load_panel()
    ranked, positive_top, negative_top = rank_daily_returns(panel, config.top_n)
    daily_summary = summarize_daily(ranked, positive_top, negative_top)
    stock_summary = summarize_stocks(ranked, positive_top, negative_top)
    recent_positive = summarize_recent(positive_top, config.recent_days, "正報酬")
    recent_negative = summarize_recent(negative_top, config.recent_days, "負報酬")
    group_summary = pd.concat(
        [summarize_groups(positive_top, "正報酬"), summarize_groups(negative_top, "負報酬")],
        ignore_index=True,
    )
    write_outputs(
        config.output_dir,
        positive_top,
        negative_top,
        stock_summary,
        recent_positive,
        recent_negative,
        group_summary,
        daily_summary,
    )
    report_path = write_html_report(
        config.viz_dir,
        config.output_dir,
        ranked,
        positive_top,
        negative_top,
        stock_summary,
        recent_positive,
        recent_negative,
        group_summary,
        daily_summary,
        config,
    )
    print(f"panel rows: {len(panel):,}")
    print(f"ranked rows: {len(ranked):,}")
    print(f"positive top rows: {len(positive_top):,}")
    print(f"negative top rows: {len(negative_top):,}")
    print(f"report: {report_path}")
    print(f"output: {config.output_dir}")


if __name__ == "__main__":
    main()
