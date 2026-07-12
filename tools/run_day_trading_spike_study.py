"""Study whether day-trading spikes coincide with sharp price moves.

The signal is based on TWSE day-trading data enriched with local price-derived
features.  Forward returns use adjusted closes and enter from the next trading
day's close, so the event day's close is not used as a tradable fill.
"""

from __future__ import annotations

import argparse
import html
import math
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

PRICE_DIR = PROJECT_ROOT / "data" / "price"
DAY_TRADING_DIR = PROJECT_ROOT / "data" / "day_trading"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "day_trading_spike_study"
VIZ_DIR = PROJECT_ROOT / "data_viz" / "day_trading_spike_study"

SIGNAL_Z = "DayTradingVolumeRatio20DayZScore"
SIGNAL_RATIO = "DayTradingVolumeRatio"
TURNOVER_Z = "DayTradingTurnover20DayZScore"

DISPLAY_LABELS = {
    "Cohort": "組別",
    "DisplayCohort": "組別",
    "ObservationCount": "樣本數",
    "StockCount": "股票數",
    "DateCount": "交易日數",
    "MeanDayTradingVolumeRatio": "平均當沖成交股數占比",
    "MedianDayTradingVolumeRatio": "當沖成交股數占比中位數",
    "MeanDayTradingZ": "平均當沖占比20日Z分數",
    "MeanCloseToCloseReturn": "平均收盤對前收報酬",
    "MedianCloseToCloseReturn": "收盤對前收報酬中位數",
    "BigUpRate": "大漲日比例",
    "BigDownRate": "大跌日比例",
    "BigAbsMoveRate": "大漲或大跌比例",
    "HighIntradayRangeRate": "高日內振幅比例",
    "AnyHighlightedDayRate": "至少符合一項比例",
    "OtherDayRate": "其他日比例",
    "MeanIntradayRangeRate": "平均日內振幅",
    "MeanOpenCloseReturn": "平均開收報酬",
    "Horizon": "未來交易日",
    "Count": "樣本數",
    "Mean": "平均報酬",
    "Median": "報酬中位數",
    "WinRate": "勝率",
    "TStat": "t值",
    "P25": "第25百分位",
    "P75": "第75百分位",
    "Date": "日期",
    "Code": "股票代號",
    "Name": "公司簡稱",
    SIGNAL_RATIO: "當沖成交股數占比",
    SIGNAL_Z: "當沖成交股數占比20日Z分數",
    TURNOVER_Z: "當沖成交值20日Z分數",
    "CloseToCloseReturn": "收盤對前收報酬",
    "OpenCloseReturn": "開收報酬",
    "IntradayRangeRate": "日內振幅",
    "DayTradingAvgSpreadRate": "當沖平均價差率",
    "ForwardReturn1D": "未來1日報酬",
    "ForwardReturn5D": "未來5日報酬",
    "ForwardReturn20D": "未來20日報酬",
    "1D": "未來1日",
    "5D": "未來5日",
    "20D": "未來20日",
}


@dataclass
class StudyConfig:
    horizons: list[int]
    z_threshold: float
    extreme_z_threshold: float
    min_ratio: float
    big_move_threshold: float
    high_range_threshold: float


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
    parser = argparse.ArgumentParser(description="Day-trading spike event study.")
    parser.add_argument("--horizons", type=parse_horizons, default=parse_horizons("1,5,20"))
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--extreme-z-threshold", type=float, default=3.0)
    parser.add_argument("--min-ratio", type=float, default=0.10)
    parser.add_argument("--big-move-threshold", type=float, default=0.03)
    parser.add_argument("--high-range-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--viz-dir", type=Path, default=VIZ_DIR)
    return parser.parse_args()


def price_path_for_code(code: str) -> Path | None:
    matches = sorted(PRICE_DIR.glob(f"{code}_*.csv"))
    return matches[0] if matches else None


def day_trading_path_for_code(code: str) -> Path | None:
    matches = sorted(DAY_TRADING_DIR.glob(f"{code}_*.csv"))
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
    day_path = day_trading_path_for_code(code)
    if not price_path or not day_path:
        return None

    price_columns = ["Date", "Open", "High", "Low", "Close", "close_adj", "Capacity", "Turnover"]
    day_columns = [
        "Date",
        "DayTradingVolume",
        "DayTradingBuyAmount",
        "DayTradingSellAmount",
        SIGNAL_RATIO,
        "DayTradingTurnoverRatio",
        "DayTradingAvgSpreadRate",
        "DayTradingAmountImbalanceRatio",
        "IntradayRangeRate",
        "OpenCloseReturn",
        SIGNAL_Z,
        TURNOVER_Z,
    ]
    try:
        price_df = read_csv_canonical(price_path, dtype=str, usecols=price_columns).fillna("")
        day_df = read_csv_canonical(day_path, dtype=str, usecols=day_columns).fillna("")
    except Exception as exc:
        print(f"skip {code}: {exc}")
        return None

    if price_df.empty or day_df.empty:
        return None

    price_df["Date"] = pd.to_datetime(price_df["Date"], errors="coerce")
    for column in price_columns:
        if column != "Date":
            price_df[column] = pd.to_numeric(price_df[column], errors="coerce")
    price_df = (
        price_df.dropna(subset=["Date", "close_adj", "Open", "Close"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    price_df = price_df[price_df["close_adj"].gt(0)].copy()
    if price_df.empty:
        return None

    price_df["CloseToCloseReturn"] = price_df["close_adj"] / price_df["close_adj"].shift(1) - 1
    for horizon in config.horizons:
        entry = price_df["close_adj"].shift(-1)
        exit_ = price_df["close_adj"].shift(-(horizon + 1))
        price_df[f"ForwardReturn{horizon}D"] = exit_ / entry - 1

    day_df["Date"] = pd.to_datetime(day_df["Date"], errors="coerce")
    for column in day_columns:
        if column != "Date":
            day_df[column] = pd.to_numeric(day_df[column], errors="coerce")
    day_df = (
        day_df.dropna(subset=["Date", SIGNAL_Z, SIGNAL_RATIO])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if day_df.empty:
        return None

    merged = day_df.merge(price_df, on="Date", how="inner")
    if merged.empty:
        return None

    merged.insert(0, "Code", code)
    merged.insert(1, "Name", name)
    return merged


def load_panel(config: StudyConfig) -> pd.DataFrame:
    frames = []
    universe = load_universe()
    for index, row in universe.iterrows():
        code = str(row["Code"]).strip()
        name = str(row["Name"]).strip()
        frame = load_stock_panel(code, name, config)
        if frame is not None:
            frames.append(frame)
        if (len(frames) + 1) % 100 == 0:
            print(f"loaded {len(frames)} stock panels")
    if not frames:
        raise RuntimeError("no stock panels loaded")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["Date", "Code"]).reset_index(drop=True)
    return panel


def assign_event_groups(panel: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    result = panel.copy()
    result["IsSpike"] = (result[SIGNAL_Z] >= config.z_threshold) & (result[SIGNAL_RATIO] >= config.min_ratio)
    result["IsExtremeSpike"] = (result[SIGNAL_Z] >= config.extreme_z_threshold) & (result[SIGNAL_RATIO] >= config.min_ratio)
    result["BigUp"] = result["CloseToCloseReturn"] >= config.big_move_threshold
    result["BigDown"] = result["CloseToCloseReturn"] <= -config.big_move_threshold
    result["BigAbsMove"] = result["CloseToCloseReturn"].abs() >= config.big_move_threshold
    result["HighIntradayRange"] = result["IntradayRangeRate"] >= config.high_range_threshold

    group = np.full(len(result), "非高點", dtype=object)
    spike = result["IsSpike"].to_numpy()
    group[spike] = "當沖高點"
    group[spike & result["BigUp"].to_numpy()] = "當沖高點+當日大漲"
    group[spike & result["BigDown"].to_numpy()] = "當沖高點+當日大跌"
    group[spike & ~(result["BigUp"] | result["BigDown"]).to_numpy()] = "當沖高點+無大漲跌"
    result["EventGroup"] = group

    exclusive = np.full(len(result), "非當沖高點", dtype=object)
    exclusive[spike & result["BigUp"].to_numpy()] = "大漲"
    exclusive[spike & result["BigDown"].to_numpy()] = "大跌"
    exclusive[spike & ~(result["BigUp"] | result["BigDown"]).to_numpy() & result["HighIntradayRange"].to_numpy()] = (
        "非大漲非大跌但高日內振幅"
    )
    exclusive[spike & ~(result["BigUp"] | result["BigDown"] | result["HighIntradayRange"]).to_numpy()] = "其他"
    result["ExclusiveSpikeGroup"] = exclusive
    return result


def describe_series(values: pd.Series) -> dict[str, float | int]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "Count": 0,
            "Mean": np.nan,
            "Median": np.nan,
            "WinRate": np.nan,
            "TStat": np.nan,
            "P10": np.nan,
            "P25": np.nan,
            "P75": np.nan,
            "P90": np.nan,
        }
    std = clean.std(ddof=1)
    t_stat = clean.mean() / (std / math.sqrt(len(clean))) if len(clean) > 1 and std > 0 else np.nan
    return {
        "Count": int(len(clean)),
        "Mean": float(clean.mean()),
        "Median": float(clean.median()),
        "WinRate": float((clean > 0).mean()),
        "TStat": float(t_stat) if pd.notna(t_stat) else np.nan,
        "P10": float(clean.quantile(0.10)),
        "P25": float(clean.quantile(0.25)),
        "P75": float(clean.quantile(0.75)),
        "P90": float(clean.quantile(0.90)),
    }


def summarize_same_day(panel: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []
    cohorts = {
        "全樣本": panel,
        "非高點": panel[~panel["IsSpike"]],
        "當沖高點": panel[panel["IsSpike"]],
        "極端當沖高點": panel[panel["IsExtremeSpike"]],
        "當沖高點+當日大漲": panel[panel["EventGroup"].eq("當沖高點+當日大漲")],
        "當沖高點+當日大跌": panel[panel["EventGroup"].eq("當沖高點+當日大跌")],
        "當沖高點+無大漲跌": panel[panel["EventGroup"].eq("當沖高點+無大漲跌")],
    }
    for label, data in cohorts.items():
        rows.append(
            {
                "Cohort": label,
                "ObservationCount": int(len(data)),
                "StockCount": int(data["Code"].nunique()) if not data.empty else 0,
                "DateCount": int(data["Date"].nunique()) if not data.empty else 0,
                "MeanDayTradingVolumeRatio": float(data[SIGNAL_RATIO].mean()) if not data.empty else np.nan,
                "MedianDayTradingVolumeRatio": float(data[SIGNAL_RATIO].median()) if not data.empty else np.nan,
                "MeanDayTradingZ": float(data[SIGNAL_Z].mean()) if not data.empty else np.nan,
                "MeanCloseToCloseReturn": float(data["CloseToCloseReturn"].mean()) if not data.empty else np.nan,
                "MedianCloseToCloseReturn": float(data["CloseToCloseReturn"].median()) if not data.empty else np.nan,
                "BigUpRate": float(data["BigUp"].mean()) if not data.empty else np.nan,
                "BigDownRate": float(data["BigDown"].mean()) if not data.empty else np.nan,
                "BigAbsMoveRate": float(data["BigAbsMove"].mean()) if not data.empty else np.nan,
                "HighIntradayRangeRate": float(data["HighIntradayRange"].mean()) if not data.empty else np.nan,
                "AnyHighlightedDayRate": float((data["BigUp"] | data["BigDown"] | data["HighIntradayRange"]).mean()) if not data.empty else np.nan,
                "OtherDayRate": float((~(data["BigUp"] | data["BigDown"] | data["HighIntradayRange"])).mean()) if not data.empty else np.nan,
                "MeanIntradayRangeRate": float(data["IntradayRangeRate"].mean()) if not data.empty else np.nan,
                "MeanOpenCloseReturn": float(data["OpenCloseReturn"].mean()) if not data.empty else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    base = summary[summary["Cohort"].eq("非高點")].iloc[0]
    for column in ["BigUpRate", "BigDownRate", "BigAbsMoveRate", "HighIntradayRangeRate"]:
        base_value = base[column]
        summary[f"{column}LiftVsNonSpike"] = summary[column] / base_value if base_value and pd.notna(base_value) else np.nan
    return summary


def summarize_forward_returns(panel: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []
    cohorts = {
        "非高點": panel[~panel["IsSpike"]],
        "當沖高點": panel[panel["IsSpike"]],
        "當沖高點+當日大漲": panel[panel["EventGroup"].eq("當沖高點+當日大漲")],
        "當沖高點+當日大跌": panel[panel["EventGroup"].eq("當沖高點+當日大跌")],
        "當沖高點+無大漲跌": panel[panel["EventGroup"].eq("當沖高點+無大漲跌")],
    }
    for horizon in config.horizons:
        column = f"ForwardReturn{horizon}D"
        for group, data in cohorts.items():
            stats = describe_series(data[column])
            stats.update({"Cohort": group, "Horizon": horizon})
            rows.append(stats)
    return pd.DataFrame(rows)


def summarize_by_year(panel: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    rows = []
    event_panel = panel[panel["IsSpike"]].copy()
    if event_panel.empty:
        return pd.DataFrame()
    event_panel["Year"] = event_panel["Date"].dt.year
    for (year, group), data in event_panel.groupby(["Year", "EventGroup"]):
        row = {
            "Year": int(year),
            "EventGroup": group,
            "EventCount": int(len(data)),
            "MeanCloseToCloseReturn": float(data["CloseToCloseReturn"].mean()),
            "BigAbsMoveRate": float(data["BigAbsMove"].mean()),
            "HighIntradayRangeRate": float(data["HighIntradayRange"].mean()),
        }
        for horizon in config.horizons:
            row[f"ForwardReturn{horizon}DMean"] = float(data[f"ForwardReturn{horizon}D"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["Year", "EventGroup"])


def top_events(panel: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    columns = [
        "Date",
        "Code",
        "Name",
        SIGNAL_RATIO,
        SIGNAL_Z,
        TURNOVER_Z,
        "CloseToCloseReturn",
        "OpenCloseReturn",
        "IntradayRangeRate",
        "DayTradingAvgSpreadRate",
    ] + [f"ForwardReturn{h}D" for h in config.horizons]
    return (
        panel[panel["IsSpike"]]
        .sort_values([SIGNAL_Z, SIGNAL_RATIO], ascending=False)
        .head(200)[columns]
        .reset_index(drop=True)
    )


def load_metadata_groups() -> pd.DataFrame:
    metadata = read_csv_canonical(METADATA_PATH, dtype={"Code": str}).fillna("")
    if "Group" not in metadata.columns:
        return pd.DataFrame(columns=["Code", "Group"])
    groups = metadata[["Code", "Group"]].copy()
    groups["Code"] = groups["Code"].astype(str).str.strip()
    groups["Group"] = groups["Group"].astype(str).str.strip().replace("", "未分類")
    return groups.drop_duplicates("Code")


def bucket_summary(
    data: pd.DataFrame,
    source_column: str,
    bucket_column: str,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    result = data.copy()
    result[bucket_column] = pd.cut(result[source_column], bins=bins, labels=labels, include_lowest=True)
    grouped = result.dropna(subset=[bucket_column]).groupby(bucket_column, observed=False)
    summary = grouped.agg(
        EventCount=("Code", "size"),
        MeanDayTradingVolumeRatio=(SIGNAL_RATIO, "mean"),
        MeanDayTradingZ=(SIGNAL_Z, "mean"),
        MeanCloseToCloseReturn=("CloseToCloseReturn", "mean"),
        MeanIntradayRangeRate=("IntradayRangeRate", "mean"),
        MeanOpenCloseReturn=("OpenCloseReturn", "mean"),
        MeanDayTradingAvgSpreadRate=("DayTradingAvgSpreadRate", "mean"),
    ).reset_index()
    summary["EventShare"] = summary["EventCount"] / len(data)
    return summary


def summarize_other_day_drivers(panel: pd.DataFrame, config: StudyConfig) -> dict[str, pd.DataFrame]:
    spike = panel[panel["IsSpike"]].copy()
    other = spike[~(spike["BigUp"] | spike["BigDown"] | spike["HighIntradayRange"])].copy()
    if other.empty:
        empty = pd.DataFrame()
        return {
            "overview": empty,
            "return_buckets": empty,
            "intraday_buckets": empty,
            "spread_buckets": empty,
            "top_stocks": empty,
            "groups": empty,
            "years": empty,
        }

    overview = pd.DataFrame(
        [
            {
                "Metric": "其他日當沖高點",
                "EventCount": int(len(other)),
                "ShareOfSpikeEvents": float(len(other) / len(spike)) if len(spike) else np.nan,
                "StockCount": int(other["Code"].nunique()),
                "DateCount": int(other["Date"].nunique()),
                "MeanDayTradingVolumeRatio": float(other[SIGNAL_RATIO].mean()),
                "MedianDayTradingVolumeRatio": float(other[SIGNAL_RATIO].median()),
                "MeanDayTradingZ": float(other[SIGNAL_Z].mean()),
                "MeanCloseToCloseReturn": float(other["CloseToCloseReturn"].mean()),
                "PositiveCloseToCloseRate": float((other["CloseToCloseReturn"] > 0).mean()),
                "MeanOpenCloseReturn": float(other["OpenCloseReturn"].mean()),
                "PositiveOpenCloseRate": float((other["OpenCloseReturn"] > 0).mean()),
                "MeanIntradayRangeRate": float(other["IntradayRangeRate"].mean()),
                "MeanDayTradingTurnoverRatio": float(other["DayTradingTurnoverRatio"].mean()),
                "MeanDayTradingAvgSpreadRate": float(other["DayTradingAvgSpreadRate"].mean()),
                "PositiveAvgSpreadRate": float((other["DayTradingAvgSpreadRate"] > 0).mean()),
                "MeanDayTradingAmountImbalanceRatio": float(other["DayTradingAmountImbalanceRatio"].mean()),
            }
        ]
    )

    return_buckets = bucket_summary(
        other,
        "CloseToCloseReturn",
        "CloseToCloseBucket",
        [-np.inf, -0.02, -0.01, 0, 0.01, 0.02, np.inf],
        ["-3%~-2%", "-2%~-1%", "-1%~0%", "0%~1%", "1%~2%", "2%~3%"],
    )
    intraday_buckets = bucket_summary(
        other,
        "IntradayRangeRate",
        "IntradayRangeBucket",
        [-np.inf, 0.01, 0.02, 0.03, 0.04, np.inf],
        ["0%~1%", "1%~2%", "2%~3%", "3%~4%", "4%~5%"],
    )
    spread_buckets = bucket_summary(
        other,
        "DayTradingAvgSpreadRate",
        "AvgSpreadBucket",
        [-np.inf, -0.005, 0, 0.005, 0.01, np.inf],
        ["<-0.5%", "-0.5%~0%", "0%~0.5%", "0.5%~1%", "1%~3%"],
    )

    top_stocks = (
        other.groupby(["Code", "Name"], as_index=False)
        .agg(
            EventCount=("Date", "size"),
            DateCount=("Date", "nunique"),
            MeanDayTradingVolumeRatio=(SIGNAL_RATIO, "mean"),
            MeanDayTradingZ=(SIGNAL_Z, "mean"),
            MeanCloseToCloseReturn=("CloseToCloseReturn", "mean"),
            MeanIntradayRangeRate=("IntradayRangeRate", "mean"),
            MeanDayTradingAvgSpreadRate=("DayTradingAvgSpreadRate", "mean"),
        )
        .sort_values(["EventCount", "MeanDayTradingVolumeRatio"], ascending=False)
        .reset_index(drop=True)
    )
    top_stocks["EventShare"] = top_stocks["EventCount"] / len(other)

    groups_data = other.merge(load_metadata_groups(), on="Code", how="left")
    groups_data["Group"] = groups_data["Group"].fillna("未分類").replace("", "未分類")
    groups = (
        groups_data.groupby("Group", as_index=False)
        .agg(
            EventCount=("Code", "size"),
            StockCount=("Code", "nunique"),
            MeanDayTradingVolumeRatio=(SIGNAL_RATIO, "mean"),
            MeanDayTradingZ=(SIGNAL_Z, "mean"),
            MeanCloseToCloseReturn=("CloseToCloseReturn", "mean"),
            MeanIntradayRangeRate=("IntradayRangeRate", "mean"),
        )
        .sort_values(["EventCount", "MeanDayTradingVolumeRatio"], ascending=False)
        .reset_index(drop=True)
    )
    groups["EventShare"] = groups["EventCount"] / len(other)

    years_data = other.copy()
    years_data["Year"] = years_data["Date"].dt.year
    years = (
        years_data.groupby("Year", as_index=False)
        .agg(
            EventCount=("Code", "size"),
            StockCount=("Code", "nunique"),
            MeanDayTradingVolumeRatio=(SIGNAL_RATIO, "mean"),
            MeanDayTradingZ=(SIGNAL_Z, "mean"),
            MeanCloseToCloseReturn=("CloseToCloseReturn", "mean"),
            MeanIntradayRangeRate=("IntradayRangeRate", "mean"),
        )
        .sort_values("Year")
        .reset_index(drop=True)
    )
    years["EventShare"] = years["EventCount"] / len(other)

    return {
        "overview": overview,
        "return_buckets": return_buckets,
        "intraday_buckets": intraday_buckets,
        "spread_buckets": spread_buckets,
        "top_stocks": top_stocks,
        "groups": groups,
        "years": years,
    }


def summarize_exclusive_spike_groups(panel: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    spike = panel[panel["IsSpike"]].copy()
    if spike.empty:
        return pd.DataFrame()
    order = ["大漲", "大跌", "非大漲非大跌但高日內振幅", "其他"]
    rows = []
    for group in order:
        data = spike[spike["ExclusiveSpikeGroup"].eq(group)]
        row = {
            "ExclusiveSpikeGroup": group,
            "EventCount": int(len(data)),
            "EventShare": float(len(data) / len(spike)),
            "StockCount": int(data["Code"].nunique()) if not data.empty else 0,
            "MeanDayTradingVolumeRatio": float(data[SIGNAL_RATIO].mean()) if not data.empty else np.nan,
            "MeanDayTradingZ": float(data[SIGNAL_Z].mean()) if not data.empty else np.nan,
            "MeanCloseToCloseReturn": float(data["CloseToCloseReturn"].mean()) if not data.empty else np.nan,
            "MeanIntradayRangeRate": float(data["IntradayRangeRate"].mean()) if not data.empty else np.nan,
            "MeanDayTradingAvgSpreadRate": float(data["DayTradingAvgSpreadRate"].mean()) if not data.empty else np.nan,
            "PositiveAvgSpreadRate": float((data["DayTradingAvgSpreadRate"] > 0).mean()) if not data.empty else np.nan,
        }
        for horizon in config.horizons:
            row[f"ForwardReturn{horizon}DMean"] = (
                float(data[f"ForwardReturn{horizon}D"].mean()) if not data.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_day_trader_return(panel: pd.DataFrame) -> pd.DataFrame:
    cohorts = {
        "非當沖高點": panel[~panel["IsSpike"]],
        "當沖高點": panel[panel["IsSpike"]],
        "極端當沖高點": panel[panel["IsExtremeSpike"]],
        "大漲": panel[panel["ExclusiveSpikeGroup"].eq("大漲")],
        "大跌": panel[panel["ExclusiveSpikeGroup"].eq("大跌")],
        "非大漲非大跌但高日內振幅": panel[panel["ExclusiveSpikeGroup"].eq("非大漲非大跌但高日內振幅")],
        "其他": panel[panel["ExclusiveSpikeGroup"].eq("其他")],
    }
    rows = []
    for cohort, data in cohorts.items():
        spread = data["DayTradingAvgSpreadRate"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "Cohort": cohort,
                "ObservationCount": int(len(data)),
                "MeanDayTradingAvgSpreadRate": float(spread.mean()) if not spread.empty else np.nan,
                "MedianDayTradingAvgSpreadRate": float(spread.median()) if not spread.empty else np.nan,
                "PositiveAvgSpreadRate": float((spread > 0).mean()) if not spread.empty else np.nan,
                "P25DayTradingAvgSpreadRate": float(spread.quantile(0.25)) if not spread.empty else np.nan,
                "P75DayTradingAvgSpreadRate": float(spread.quantile(0.75)) if not spread.empty else np.nan,
                "MeanOpenCloseReturn": float(data["OpenCloseReturn"].mean()) if not data.empty else np.nan,
                "MeanCloseToCloseReturn": float(data["CloseToCloseReturn"].mean()) if not data.empty else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    base = summary.loc[summary["Cohort"].eq("非當沖高點"), "MeanDayTradingAvgSpreadRate"]
    base_value = float(base.iloc[0]) if not base.empty and pd.notna(base.iloc[0]) else np.nan
    summary["MeanSpreadLiftVsNonSpike"] = summary["MeanDayTradingAvgSpreadRate"] - base_value
    return summary


def histogram_summary(
    panel: pd.DataFrame,
    column: str,
    bucket_column: str,
    bins: list[float],
    labels: list[str],
) -> pd.DataFrame:
    cohorts = {
        "非當沖高點": panel[~panel["IsSpike"]],
        "當沖高點": panel[panel["IsSpike"]],
    }
    rows = []
    for cohort, data in cohorts.items():
        clean = data[[column]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        clean[bucket_column] = pd.cut(clean[column], bins=bins, labels=labels, include_lowest=True)
        total = int(clean[bucket_column].notna().sum())
        counts = clean[bucket_column].value_counts(sort=False, dropna=False)
        for label in labels:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "Cohort": cohort,
                    bucket_column: label,
                    "ObservationCount": count,
                    "Share": float(count / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_histograms(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "close_return": histogram_summary(
            panel,
            "CloseToCloseReturn",
            "Bucket",
            [-np.inf, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05, np.inf],
            ["<-5%", "-5%~-3%", "-3%~-2%", "-2%~-1%", "-1%~0%", "0%~1%", "1%~2%", "2%~3%", "3%~5%", ">5%"],
        ),
        "intraday_range": histogram_summary(
            panel,
            "IntradayRangeRate",
            "Bucket",
            [-np.inf, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, np.inf],
            ["0%~1%", "1%~2%", "2%~3%", "3%~4%", "4%~5%", "5%~7%", "7%~10%", ">10%"],
        ),
        "avg_spread": histogram_summary(
            panel,
            "DayTradingAvgSpreadRate",
            "Bucket",
            [-np.inf, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, np.inf],
            ["<-2%", "-2%~-1%", "-1%~-0.5%", "-0.5%~0%", "0%~0.5%", "0.5%~1%", "1%~2%", ">2%"],
        ),
    }


def write_csvs(
    output_dir: Path,
    panel: pd.DataFrame,
    same_day: pd.DataFrame,
    forward: pd.DataFrame,
    yearly: pd.DataFrame,
    top: pd.DataFrame,
    other_analysis: dict[str, pd.DataFrame],
    exclusive_summary: pd.DataFrame,
    day_trader_return: pd.DataFrame,
    histograms: dict[str, pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    same_day.to_csv(output_dir / "same_day_summary.csv", index=False, encoding="utf-8-sig")
    forward.to_csv(output_dir / "forward_return_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output_dir / "yearly_spike_summary.csv", index=False, encoding="utf-8-sig")
    top.to_csv(output_dir / "top_day_trading_spike_events.csv", index=False, encoding="utf-8-sig")
    exclusive_summary.to_csv(output_dir / "exclusive_spike_group_summary.csv", index=False, encoding="utf-8-sig")
    day_trader_return.to_csv(output_dir / "day_trader_return_summary.csv", index=False, encoding="utf-8-sig")
    histograms["close_return"].to_csv(output_dir / "hist_close_to_close_return.csv", index=False, encoding="utf-8-sig")
    histograms["intraday_range"].to_csv(output_dir / "hist_intraday_range.csv", index=False, encoding="utf-8-sig")
    histograms["avg_spread"].to_csv(output_dir / "hist_day_trading_avg_spread.csv", index=False, encoding="utf-8-sig")
    other_analysis["overview"].to_csv(output_dir / "other_day_overview.csv", index=False, encoding="utf-8-sig")
    other_analysis["return_buckets"].to_csv(output_dir / "other_day_return_buckets.csv", index=False, encoding="utf-8-sig")
    other_analysis["intraday_buckets"].to_csv(output_dir / "other_day_intraday_buckets.csv", index=False, encoding="utf-8-sig")
    other_analysis["spread_buckets"].to_csv(output_dir / "other_day_avg_spread_buckets.csv", index=False, encoding="utf-8-sig")
    other_analysis["top_stocks"].to_csv(output_dir / "other_day_top_stocks.csv", index=False, encoding="utf-8-sig")
    other_analysis["groups"].to_csv(output_dir / "other_day_group_summary.csv", index=False, encoding="utf-8-sig")
    other_analysis["years"].to_csv(output_dir / "other_day_yearly_summary.csv", index=False, encoding="utf-8-sig")
    event_columns = [
        "Date",
        "Code",
        "Name",
        "EventGroup",
        SIGNAL_RATIO,
        SIGNAL_Z,
        "DayTradingTurnoverRatio",
        "DayTradingAvgSpreadRate",
        "DayTradingAmountImbalanceRatio",
        TURNOVER_Z,
        "CloseToCloseReturn",
        "OpenCloseReturn",
        "IntradayRangeRate",
    ]
    event_columns += [f"ForwardReturn{h}D" for h in sorted({int(c.replace("ForwardReturn", "").replace("D", "")) for c in panel.columns if c.startswith("ForwardReturn")})]
    events = panel.loc[panel["IsSpike"], event_columns].copy()
    events.to_csv(output_dir / "spike_events.csv", index=False, encoding="utf-8-sig")


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


def html_table(
    df: pd.DataFrame,
    columns: Iterable[str],
    percent_columns: set[str] | None = None,
    labels: dict[str, str] | None = None,
) -> str:
    percent_columns = percent_columns or set()
    labels = labels or {}
    rows = []
    clean = df[list(columns)].replace([np.inf, -np.inf], np.nan)
    clean = clean.where(pd.notna(clean), "")
    for row in clean.to_dict("records"):
        cells = []
        for column in columns:
            value = row[column]
            if column in percent_columns and value != "":
                text = pct(value)
            elif isinstance(value, (int, float, np.integer, np.floating)) and value != "":
                text = num(value)
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{html.escape(labels.get(column, column))}</th>" for column in columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def svg_bar_chart(
    df: pd.DataFrame,
    label_column: str,
    value_columns: list[str],
    title: str,
    percent: bool = True,
    labels: dict[str, str] | None = None,
    show_title: bool = True,
    show_y_axis_labels: bool = True,
    legend_below_labels: bool = False,
) -> str:
    labels_map = labels or {}
    labels = df[label_column].astype(str).tolist()
    values = df[value_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    flat = values.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return ""
    min_v = min(0.0, float(flat.min()))
    max_v = max(0.0, float(flat.max()))
    pad = max(0.01, (max_v - min_v) * 0.15)
    y_min = min_v - pad
    y_max = max_v + pad
    width = 980
    height = 390 if legend_below_labels else 360
    left = 48 if not show_y_axis_labels else 160
    right = 28
    top = 26 if not show_title else 34
    bottom = 94 if legend_below_labels else 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / max(1, len(labels))
    bar_w = min(24, group_w / (len(value_columns) + 1))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    def x_for(i: int, j: int) -> float:
        center = left + group_w * i + group_w / 2
        offset = (j - (len(value_columns) - 1) / 2) * bar_w * 1.25
        return center + offset - bar_w / 2

    def y_for(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    zero_y = y_for(0)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">']
    if show_title and title:
        parts.append(f'<text x="{left}" y="20" font-size="16" font-weight="700">{html.escape(title)}</text>')
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#64748b" />')
    if show_y_axis_labels:
        parts.append(f'<text x="8" y="{top + 8}" font-size="12" fill="#64748b">{pct(y_max) if percent else num(y_max)}</text>')
        parts.append(f'<text x="8" y="{height - bottom}" font-size="12" fill="#64748b">{pct(y_min) if percent else num(y_min)}</text>')
    for i, label in enumerate(labels):
        label_x = left + group_w * i + group_w / 2
        label_y = height - (54 if legend_below_labels else 20)
        label_parts = label.split(" (", 1)
        if len(label_parts) == 2:
            first_line = label_parts[0]
            second_line = "(" + label_parts[1]
            parts.append(
                f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-size="12" text-anchor="middle">'
                f'<tspan x="{label_x:.1f}" dy="0">{html.escape(first_line)}</tspan>'
                f'<tspan x="{label_x:.1f}" dy="15">{html.escape(second_line)}</tspan>'
                "</text>"
            )
        else:
            parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" font-size="12" text-anchor="middle">{html.escape(label)}</text>')
        for j, column in enumerate(value_columns):
            v = values.iloc[i][column]
            if pd.isna(v):
                continue
            yy = y_for(float(v))
            bar_y = min(yy, zero_y)
            bar_h = max(2, abs(zero_y - yy))
            x = x_for(i, j)
            color = colors[j % len(colors)]
            label_text = pct(v) if percent else num(v)
            parts.append(f'<rect x="{x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" />')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{bar_y - 5:.1f}" font-size="10" text-anchor="middle" fill="#334155">{html.escape(label_text)}</text>')
    legend_width = len(value_columns) * 180
    legend_x = left + max(0, (plot_w - legend_width) / 2) if legend_below_labels else left
    legend_y = height - 18 if legend_below_labels else height - 44
    legend_text_y = legend_y + 9
    for j, column in enumerate(value_columns):
        x = legend_x + j * 180
        parts.append(f'<rect x="{x}" y="{legend_y}" width="10" height="10" fill="{colors[j % len(colors)]}" />')
        parts.append(f'<text x="{x + 16}" y="{legend_text_y}" font-size="12">{html.escape(labels_map.get(column, column))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_histogram_chart(
    df: pd.DataFrame,
    bucket_column: str,
    cohort_column: str,
    value_column: str,
    title: str,
) -> str:
    if df.empty:
        return ""
    buckets = df[bucket_column].drop_duplicates().astype(str).tolist()
    cohorts = df[cohort_column].drop_duplicates().astype(str).tolist()
    if not buckets or not cohorts:
        return ""
    width = 980
    height = 360
    left = 64
    right = 24
    top = 38
    bottom = 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = df[value_column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    y_max = max(0.01, float(values.max()) * 1.18)
    group_w = plot_w / max(1, len(buckets))
    bar_w = min(22, group_w / (len(cohorts) + 0.8))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    def y_for(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">']
    parts.append(f'<text x="{left}" y="22" font-size="16" font-weight="700">{html.escape(title)}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h:.1f}" x2="{width-right}" y2="{top + plot_h:.1f}" stroke="#64748b" />')
    parts.append(f'<text x="8" y="{top + 6}" font-size="12" fill="#64748b">{pct(y_max)}</text>')
    parts.append(f'<text x="8" y="{top + plot_h}" font-size="12" fill="#64748b">0%</text>')
    for i, bucket in enumerate(buckets):
        label_x = left + group_w * i + group_w / 2
        parts.append(
            f'<text x="{label_x:.1f}" y="{height - 38}" font-size="11" text-anchor="middle" transform="rotate(-35 {label_x:.1f} {height - 38})">{html.escape(bucket)}</text>'
        )
        for j, cohort in enumerate(cohorts):
            match = df[(df[bucket_column].astype(str) == bucket) & (df[cohort_column].astype(str) == cohort)]
            value = float(match[value_column].iloc[0]) if not match.empty and pd.notna(match[value_column].iloc[0]) else 0.0
            x = left + group_w * i + (group_w - bar_w * len(cohorts)) / 2 + j * bar_w
            y = y_for(value)
            bar_h = max(1, top + plot_h - y)
            color = colors[j % len(colors)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" height="{bar_h:.1f}" fill="{color}" />')
    legend_x = left
    legend_y = height - 18
    for j, cohort in enumerate(cohorts):
        x = legend_x + j * 160
        parts.append(f'<rect x="{x}" y="{legend_y}" width="10" height="10" fill="{colors[j % len(colors)]}" />')
        parts.append(f'<text x="{x + 16}" y="{legend_y + 9}" font-size="12">{html.escape(cohort)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def add_display_cohort_with_ratio(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    def display_label(row: pd.Series) -> str:
        cohort = "非當沖高點" if row["Cohort"] == "非高點" else str(row["Cohort"])
        ratio = row.get("MeanDayTradingVolumeRatio")
        suffix = f" (當沖占比{pct(ratio, 1)})" if pd.notna(ratio) else ""
        return cohort + suffix

    result["DisplayCohort"] = result.apply(display_label, axis=1)
    return result


def write_html_report(
    viz_dir: Path,
    output_dir: Path,
    panel: pd.DataFrame,
    same_day: pd.DataFrame,
    forward: pd.DataFrame,
    top: pd.DataFrame,
    other_analysis: dict[str, pd.DataFrame],
    exclusive_summary: pd.DataFrame,
    day_trader_return: pd.DataFrame,
    histograms: dict[str, pd.DataFrame],
    config: StudyConfig,
) -> Path:
    viz_dir.mkdir(parents=True, exist_ok=True)
    report_path = viz_dir / "index.html"
    start = panel["Date"].min().strftime("%Y-%m-%d")
    end = panel["Date"].max().strftime("%Y-%m-%d")
    spike_count = int(panel["IsSpike"].sum())
    stock_count = int(panel["Code"].nunique())
    event_dates = int(panel.loc[panel["IsSpike"], "Date"].nunique())
    same_day_display = same_day[
        same_day["Cohort"].isin(["非高點", "當沖高點", "極端當沖高點"])
    ].copy()
    same_day_display = add_display_cohort_with_ratio(same_day_display)
    forward_display = forward[forward["Cohort"].isin(["當沖高點"])].copy()
    forward_pivot = (
        forward_display.pivot(index="Cohort", columns="Horizon", values="Mean")
        .reset_index()
        .rename(columns={1: "1D", 5: "5D", 20: "20D"})
    )
    forward_cols = [c for c in ["1D", "5D", "20D"] if c in forward_pivot.columns]
    same_day_chart = svg_bar_chart(
        same_day_display,
        "DisplayCohort",
        ["BigUpRate", "BigDownRate", "HighIntradayRangeRate"],
        "",
        labels=DISPLAY_LABELS,
        show_title=False,
        show_y_axis_labels=False,
        legend_below_labels=True,
    )
    forward_chart = svg_bar_chart(
        forward_pivot,
        "Cohort",
        forward_cols,
        "事件後平均報酬",
        percent=True,
        labels=DISPLAY_LABELS,
    )

    same_day_columns = [
        "DisplayCohort",
        "ObservationCount",
        "MeanDayTradingVolumeRatio",
        "MeanDayTradingZ",
        "MeanCloseToCloseReturn",
        "BigUpRate",
        "BigDownRate",
        "BigAbsMoveRate",
        "HighIntradayRangeRate",
        "OtherDayRate",
    ]
    same_day_percent = {
        "MeanDayTradingVolumeRatio",
        "MeanCloseToCloseReturn",
        "BigUpRate",
        "BigDownRate",
        "BigAbsMoveRate",
        "HighIntradayRangeRate",
        "OtherDayRate",
    }
    forward_columns = ["Cohort", "Horizon", "Count", "Mean", "Median", "WinRate", "TStat", "P25", "P75"]
    forward_percent = {"Mean", "Median", "WinRate", "P25", "P75"}
    top_columns = [
        "Date",
        "Code",
        "Name",
        SIGNAL_RATIO,
        SIGNAL_Z,
        "CloseToCloseReturn",
        "OpenCloseReturn",
        "IntradayRangeRate",
    ]
    top_columns += [f"ForwardReturn{h}D" for h in config.horizons]
    top_percent = {SIGNAL_RATIO, "CloseToCloseReturn", "OpenCloseReturn", "IntradayRangeRate"} | {
        f"ForwardReturn{h}D" for h in config.horizons
    }
    top_for_html = top.head(30).copy()
    other_labels = {
        "Metric": "項目",
        "EventCount": "事件數",
        "EventShare": "占其他日比例",
        "ShareOfSpikeEvents": "占當沖高點比例",
        "StockCount": "股票數",
        "DateCount": "交易日數",
        "MeanDayTradingVolumeRatio": "平均當沖股數占比",
        "MedianDayTradingVolumeRatio": "當沖股數占比中位數",
        "MeanDayTradingZ": "平均當沖股數占比Z分數",
        "MeanCloseToCloseReturn": "平均收盤報酬",
        "PositiveCloseToCloseRate": "收盤上漲比例",
        "MeanOpenCloseReturn": "平均開收報酬",
        "PositiveOpenCloseRate": "開收上漲比例",
        "MeanIntradayRangeRate": "平均日內振幅",
        "MeanDayTradingTurnoverRatio": "平均當沖成交值占比",
        "MeanDayTradingAvgSpreadRate": "平均當沖價差率",
        "PositiveAvgSpreadRate": "當沖平均價差為正比例",
        "MeanDayTradingAmountImbalanceRatio": "買賣金額差率",
        "CloseToCloseBucket": "收盤報酬區間",
        "IntradayRangeBucket": "日內振幅區間",
        "AvgSpreadBucket": "當沖平均價差區間",
        "Code": "股票代碼",
        "Name": "公司簡稱",
        "Group": "產業",
        "Year": "年度",
    }
    other_percent = {
        "EventShare",
        "ShareOfSpikeEvents",
        "MeanDayTradingVolumeRatio",
        "MedianDayTradingVolumeRatio",
        "MeanCloseToCloseReturn",
        "PositiveCloseToCloseRate",
        "MeanOpenCloseReturn",
        "PositiveOpenCloseRate",
        "MeanIntradayRangeRate",
        "MeanDayTradingTurnoverRatio",
        "MeanDayTradingAvgSpreadRate",
        "PositiveAvgSpreadRate",
        "MeanDayTradingAmountImbalanceRatio",
    }
    other_overview_columns = [
        "Metric",
        "EventCount",
        "ShareOfSpikeEvents",
        "StockCount",
        "MeanDayTradingVolumeRatio",
        "MeanDayTradingZ",
        "MeanCloseToCloseReturn",
        "PositiveCloseToCloseRate",
        "MeanIntradayRangeRate",
        "MeanDayTradingTurnoverRatio",
        "MeanDayTradingAvgSpreadRate",
        "PositiveAvgSpreadRate",
    ]
    bucket_columns = [
        "EventCount",
        "EventShare",
        "MeanDayTradingVolumeRatio",
        "MeanDayTradingZ",
        "MeanCloseToCloseReturn",
        "MeanIntradayRangeRate",
        "MeanDayTradingAvgSpreadRate",
    ]
    stock_columns = [
        "Code",
        "Name",
        "EventCount",
        "EventShare",
        "MeanDayTradingVolumeRatio",
        "MeanDayTradingZ",
        "MeanCloseToCloseReturn",
        "MeanIntradayRangeRate",
        "MeanDayTradingAvgSpreadRate",
    ]
    group_columns = [
        "Group",
        "EventCount",
        "EventShare",
        "StockCount",
        "MeanDayTradingVolumeRatio",
        "MeanDayTradingZ",
        "MeanCloseToCloseReturn",
        "MeanIntradayRangeRate",
    ]
    exclusive_labels = {
        **other_labels,
        "ExclusiveSpikeGroup": "互斥分類",
        "ForwardReturn1DMean": "未來1日平均報酬",
        "ForwardReturn5DMean": "未來5日平均報酬",
        "ForwardReturn20DMean": "未來20日平均報酬",
    }
    exclusive_columns = [
        "ExclusiveSpikeGroup",
        "EventCount",
        "EventShare",
        "StockCount",
        "MeanDayTradingVolumeRatio",
        "MeanDayTradingZ",
        "MeanCloseToCloseReturn",
        "MeanIntradayRangeRate",
        "MeanDayTradingAvgSpreadRate",
        "PositiveAvgSpreadRate",
    ] + [f"ForwardReturn{h}DMean" for h in config.horizons]
    exclusive_percent = {
        "EventShare",
        "MeanDayTradingVolumeRatio",
        "MeanCloseToCloseReturn",
        "MeanIntradayRangeRate",
        "MeanDayTradingAvgSpreadRate",
        "PositiveAvgSpreadRate",
    } | {f"ForwardReturn{h}DMean" for h in config.horizons}
    return_labels = {
        **exclusive_labels,
        "Cohort": "組別",
        "ObservationCount": "樣本數",
        "MedianDayTradingAvgSpreadRate": "當沖價差率中位數",
        "P25DayTradingAvgSpreadRate": "當沖價差率第25百分位",
        "P75DayTradingAvgSpreadRate": "當沖價差率第75百分位",
        "MeanSpreadLiftVsNonSpike": "平均價差率差異",
    }
    return_columns = [
        "Cohort",
        "ObservationCount",
        "MeanDayTradingAvgSpreadRate",
        "MedianDayTradingAvgSpreadRate",
        "PositiveAvgSpreadRate",
        "P25DayTradingAvgSpreadRate",
        "P75DayTradingAvgSpreadRate",
        "MeanSpreadLiftVsNonSpike",
        "MeanOpenCloseReturn",
        "MeanCloseToCloseReturn",
    ]
    return_percent = {
        "MeanDayTradingAvgSpreadRate",
        "MedianDayTradingAvgSpreadRate",
        "PositiveAvgSpreadRate",
        "P25DayTradingAvgSpreadRate",
        "P75DayTradingAvgSpreadRate",
        "MeanSpreadLiftVsNonSpike",
        "MeanOpenCloseReturn",
        "MeanCloseToCloseReturn",
    }
    exclusive_chart_data = exclusive_summary.rename(columns={"ExclusiveSpikeGroup": "Cohort"}).copy()
    exclusive_chart = svg_bar_chart(
        exclusive_chart_data,
        "Cohort",
        ["EventShare"],
        "互斥分類占所有當沖高點比例",
        percent=True,
        labels={"EventShare": "占所有當沖高點比例"},
    )
    day_trader_return_chart = svg_bar_chart(
        day_trader_return[day_trader_return["Cohort"].isin(["非當沖高點", "當沖高點", "極端當沖高點"])],
        "Cohort",
        ["MeanDayTradingAvgSpreadRate", "PositiveAvgSpreadRate"],
        "當沖平均價差率比較",
        percent=True,
        labels=return_labels,
    )
    close_return_hist = svg_histogram_chart(
        histograms["close_return"],
        "Bucket",
        "Cohort",
        "Share",
        "收盤報酬分布",
    )
    intraday_hist = svg_histogram_chart(
        histograms["intraday_range"],
        "Bucket",
        "Cohort",
        "Share",
        "日內振幅分布",
    )
    spread_hist = svg_histogram_chart(
        histograms["avg_spread"],
        "Bucket",
        "Cohort",
        "Share",
        "當沖平均價差率分布",
    )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>當沖高點事件研究</title>
<style>
body {{ font-family: Arial, "Microsoft JhengHei", sans-serif; margin: 0; background: #f6f7f9; color: #172033; }}
header {{ background: #172033; color: white; padding: 24px 32px 18px; }}
h1 {{ margin: 0 0 8px; font-size: 25px; }}
.meta {{ color: #cbd5e1; font-size: 13px; line-height: 1.6; }}
main {{ padding: 24px 32px 40px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.card {{ background: white; border: 1px solid #dfe5ef; border-radius: 6px; padding: 14px 16px; }}
.label {{ color: #64748b; font-size: 12px; }}
.value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
section {{ background: white; border: 1px solid #dfe5ef; border-radius: 6px; margin: 16px 0; padding: 18px; overflow-x: auto; }}
h2 {{ font-size: 18px; margin: 0 0 12px; }}
p {{ line-height: 1.7; color: #334155; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5eaf2; padding: 8px 10px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f2f5f9; color: #334155; }}
.note {{ color: #59677c; font-size: 13px; line-height: 1.6; }}
@media (max-width: 900px) {{ .cards {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }} main {{ padding: 18px; }} }}
</style>
</head>
<body>
<header>
<h1>當沖高點事件研究</h1>
<div class="meta">事件定義：當沖成交股數占比20日Z分數 >= {config.z_threshold:g} 且當沖成交股數占比 >= {config.min_ratio:.0%}。極端事件：Z分數 >= {config.extreme_z_threshold:g}。未來報酬使用下一個交易日復權收盤價進場，避免用事件日收盤當作事後成交價。</div>
</header>
<main>
<div class="cards">
  <div class="card"><div class="label">股票數</div><div class="value">{stock_count:,}</div></div>
  <div class="card"><div class="label">樣本期間</div><div class="value">{html.escape(start)} - {html.escape(end)}</div></div>
  <div class="card"><div class="label">當沖高點事件</div><div class="value">{spike_count:,}</div></div>
  <div class="card"><div class="label">事件日期數</div><div class="value">{event_dates:,}</div></div>
</div>
<section>
<h2>結論摘要</h2>
<p>這份研究檢查你的假設：當沖高點通常是不是發生在股價大漲或大跌時。重點不是只看事件後報酬，也先看事件當天是否伴隨大幅波動。</p>
<p>大漲日定義為事件日復權收盤價相對前一個交易日復權收盤價上漲至少 {config.big_move_threshold:.0%}；大跌日定義為下跌至少 {config.big_move_threshold:.0%}；高日內振幅定義為「日內振幅 = (最高價 - 最低價) / 收盤價」至少 {config.high_range_threshold:.0%}。</p>
</section>
<section>
<h2>事件當天型態</h2>
{same_day_chart}
<p class="note">注意：大漲日比例、大跌日比例、高日內振幅比例是三個獨立條件，不是互斥分類，所以不會加總成 100%。高日內振幅可以同時發生在大漲日或大跌日；沒有符合這三個條件的事件，會列在表格的「其他日比例」。</p>
{html_table(same_day_display, same_day_columns, same_day_percent, DISPLAY_LABELS)}
</section>
<section>
<h2>互斥分類：分母為所有當沖高點</h2>
<p>這裡每個當沖高點只會被放進一個分類：先判斷大漲，再判斷大跌，再判斷非大漲非大跌但高日內振幅，最後才是其他。四類比例會加總為 100%。</p>
{exclusive_chart}
{html_table(exclusive_summary, exclusive_columns, exclusive_percent, exclusive_labels)}
</section>
<section>
<h2>其他日 {pct(other_analysis["overview"].iloc[0]["ShareOfSpikeEvents"]) if not other_analysis["overview"].empty else ""} 拆解</h2>
<p>這裡的「其他日」不是缺資料，而是當沖高點中，沒有同時符合大漲、大跌或高日內振幅三個條件的事件。它代表當沖活躍度是相對自身 20 日基準突然升高，但股價當天沒有走到我們設定的極端門檻。</p>
{html_table(other_analysis["overview"], other_overview_columns, other_percent, other_labels)}
<h3>收盤報酬分布</h3>
{html_table(other_analysis["return_buckets"], ["CloseToCloseBucket"] + bucket_columns, other_percent, other_labels)}
<h3>日內振幅分布</h3>
{html_table(other_analysis["intraday_buckets"], ["IntradayRangeBucket"] + bucket_columns, other_percent, other_labels)}
<h3>當沖平均價差分布</h3>
{html_table(other_analysis["spread_buckets"], ["AvgSpreadBucket"] + bucket_columns, other_percent, other_labels)}
<h3>其他日事件最多的股票</h3>
{html_table(other_analysis["top_stocks"].head(15), stock_columns, other_percent, other_labels)}
<h3>其他日事件最多的產業</h3>
{html_table(other_analysis["groups"].head(12), group_columns, other_percent, other_labels)}
</section>
<section>
<h2>當沖客日內粗估報酬</h2>
<p>這裡用「當沖平均價差率」估算當天當沖交易的毛報酬方向：平均賣出價格高於平均買進價格時為正。它不含手續費、證交稅，也不是逐筆交易者真實損益，但可以用來比較當沖高點是否提供較高日內價差。</p>
{day_trader_return_chart}
{html_table(day_trader_return, return_columns, return_percent, return_labels)}
</section>
<section>
<h2>分布直方圖</h2>
{close_return_hist}
{intraday_hist}
{spread_hist}
</section>
<section>
<h2>事件後報酬</h2>
{forward_chart}
{html_table(forward_display, forward_columns, forward_percent, DISPLAY_LABELS)}
</section>
<section>
<h2>最大當沖高點事件前三十名</h2>
{html_table(top_for_html, top_columns, top_percent, DISPLAY_LABELS)}
</section>
<section>
<h2>輸出檔案</h2>
<div class="note">
<a href="../../output/day_trading_spike_study/same_day_summary.csv">下載事件當天型態摘要</a><br>
<a href="../../output/day_trading_spike_study/forward_return_summary.csv">下載事件後報酬摘要</a><br>
<a href="../../output/day_trading_spike_study/spike_events.csv">下載當沖高點事件明細</a><br>
<a href="../../output/day_trading_spike_study/top_day_trading_spike_events.csv">下載最大當沖高點事件清單</a>
<br>
<a href="../../output/day_trading_spike_study/exclusive_spike_group_summary.csv">下載互斥分類摘要</a><br>
<a href="../../output/day_trading_spike_study/day_trader_return_summary.csv">下載當沖價差率摘要</a><br>
<a href="../../output/day_trading_spike_study/hist_close_to_close_return.csv">下載收盤報酬直方圖資料</a><br>
<a href="../../output/day_trading_spike_study/hist_intraday_range.csv">下載日內振幅直方圖資料</a><br>
<a href="../../output/day_trading_spike_study/hist_day_trading_avg_spread.csv">下載當沖價差率直方圖資料</a><br>
<a href="../../output/day_trading_spike_study/other_day_overview.csv">下載其他日拆解總表</a><br>
<a href="../../output/day_trading_spike_study/other_day_top_stocks.csv">下載其他日股票集中度</a><br>
<a href="../../output/day_trading_spike_study/other_day_group_summary.csv">下載其他日產業集中度</a>
</div>
</section>
</main>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    config = StudyConfig(
        horizons=args.horizons,
        z_threshold=args.z_threshold,
        extreme_z_threshold=args.extreme_z_threshold,
        min_ratio=args.min_ratio,
        big_move_threshold=args.big_move_threshold,
        high_range_threshold=args.high_range_threshold,
    )
    output_dir = args.output_dir
    viz_dir = args.viz_dir

    panel = load_panel(config)
    panel = assign_event_groups(panel, config)
    same_day = summarize_same_day(panel, config)
    forward = summarize_forward_returns(panel, config)
    yearly = summarize_by_year(panel, config)
    top = top_events(panel, config)
    other_analysis = summarize_other_day_drivers(panel, config)
    exclusive_summary = summarize_exclusive_spike_groups(panel, config)
    day_trader_return = summarize_day_trader_return(panel)
    histograms = build_histograms(panel)
    write_csvs(
        output_dir,
        panel,
        same_day,
        forward,
        yearly,
        top,
        other_analysis,
        exclusive_summary,
        day_trader_return,
        histograms,
    )
    report_path = write_html_report(
        viz_dir,
        output_dir,
        panel,
        same_day,
        forward,
        top,
        other_analysis,
        exclusive_summary,
        day_trader_return,
        histograms,
        config,
    )

    print(f"panel rows: {len(panel):,}")
    print(f"spike events: {int(panel['IsSpike'].sum()):,}")
    print(f"report: {report_path}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
