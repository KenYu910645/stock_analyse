"""Institutional-flow factor calculations."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from column_schema import read_csv_canonical


TRUST_BUY_COL = "InvestmentTrustBuy"
TRUST_SELL_COL = "InvestmentTrustSell"
TRUST_NET_COL = "InvestmentTrustNet"


def _stock_id_from_path(csv_path: Path) -> str:
    return csv_path.stem.split("_", 1)[0]


def _normalize_institutional_frame(frame: pd.DataFrame, stock_id: str | None = None) -> pd.DataFrame:
    column_map = {
        "Date": "date",
        "Code": "stock_id",
        TRUST_BUY_COL: "trust_buy",
        TRUST_SELL_COL: "trust_sell",
        TRUST_NET_COL: "trust_net",
    }
    available = [column for column in column_map if column in frame.columns]
    df = frame[available].rename(columns=column_map).copy()
    if "date" not in df.columns:
        raise ValueError("institutional frame missing Date column")
    if "stock_id" not in df.columns:
        if stock_id is None:
            raise ValueError("institutional frame missing Code column")
        df["stock_id"] = stock_id

    df["stock_id"] = df["stock_id"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ["trust_buy", "trust_sell", "trust_net"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "trust_buy" not in df.columns:
        df["trust_buy"] = np.nan
    if "trust_sell" not in df.columns:
        df["trust_sell"] = np.nan
    if "trust_net" not in df.columns:
        df["trust_net"] = df["trust_buy"] - df["trust_sell"]
    df["trust_net"] = df["trust_net"].fillna(df["trust_buy"] - df["trust_sell"])
    df = df.dropna(subset=["date", "stock_id"])
    return (
        df[["date", "stock_id", "trust_buy", "trust_sell", "trust_net"]]
        .sort_values(["stock_id", "date"])
        .drop_duplicates(["stock_id", "date"], keep="last")
        .reset_index(drop=True)
    )


def _load_institutional_data(institutional_dir: Path, stock_ids: set[str]) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(institutional_dir.glob("*.csv")):
        stock_id = _stock_id_from_path(csv_path)
        if stock_id not in stock_ids:
            continue
        try:
            raw = read_csv_canonical(csv_path, dtype={"Code": str})
            frames.append(_normalize_institutional_frame(raw, stock_id))
        except Exception as exc:
            print(f"Skipped institutional file {csv_path.name}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["date", "stock_id", "trust_buy", "trust_sell", "trust_net"])
    return pd.concat(frames, ignore_index=True)


def _rolling_ratio(
    grouped: pd.core.groupby.generic.SeriesGroupBy,
    denominator_grouped: pd.core.groupby.generic.SeriesGroupBy,
    window: int,
) -> pd.Series:
    numerator = grouped.transform(lambda values: values.rolling(window, min_periods=window).sum())
    denominator = denominator_grouped.transform(lambda values: values.rolling(window, min_periods=window).sum())
    return numerator / denominator.replace(0, np.nan)


def _positive_streak(values: pd.Series, window: int) -> pd.Series:
    """Return the trailing positive run length, capped at the requested window."""
    valid = values.notna()
    positive = values.gt(0) & valid
    run_groups = (~positive).cumsum()
    streak = positive.groupby(run_groups).cumsum().clip(upper=window).astype(float)
    complete_window = valid.rolling(window, min_periods=window).sum().eq(window)
    return streak.where(complete_window)


def compute_trust_flow_factors(
    price_df: pd.DataFrame,
    factor_names: list[str],
    institutional_dir: str | Path = "data/institutional",
    institutional_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute investment-trust flow factors without using future prices."""
    df = price_df.sort_values(["stock_id", "date"]).copy()
    stock_ids = set(df["stock_id"].astype(str))
    if institutional_df is None:
        institutional = _load_institutional_data(Path(institutional_dir), stock_ids)
    else:
        institutional = _normalize_institutional_frame(institutional_df)

    merged = df.merge(institutional, on=["date", "stock_id"], how="left")
    for column in ["volume", "turnover", "trust_buy", "trust_sell", "trust_net"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    average_price = (merged["turnover"] / merged["volume"]).where(merged["volume"].gt(0))
    merged["trust_net_value"] = merged["trust_net"] * average_price
    merged["trust_sell_value"] = (-merged["trust_net_value"]).clip(lower=0)
    merged["trust_activity"] = merged["trust_buy"] + merged["trust_sell"]

    grouped_net_value = merged.groupby("stock_id", sort=False)["trust_net_value"]
    grouped_sell_value = merged.groupby("stock_id", sort=False)["trust_sell_value"]
    grouped_turnover = merged.groupby("stock_id", sort=False)["turnover"]
    grouped_net = merged.groupby("stock_id", sort=False)["trust_net"]

    for factor_name in factor_names:
        net_value_match = re.fullmatch(r"trust_net_value_(\d+)d_to_turnover", factor_name)
        if net_value_match:
            window = int(net_value_match.group(1))
            merged[factor_name] = _rolling_ratio(grouped_net_value, grouped_turnover, window)
            continue

        if factor_name == "trust_buy_purity_1d":
            merged[factor_name] = merged["trust_net"] / merged["trust_activity"].replace(0, np.nan)
            continue

        if factor_name == "trust_buy_streak_3d":
            merged[factor_name] = grouped_net.transform(lambda values: _positive_streak(values, 3))
            continue

        if factor_name == "trust_sell_pressure_3d":
            merged[factor_name] = _rolling_ratio(grouped_sell_value, grouped_turnover, 3)
            continue

        raise ValueError(f"Unknown trust-flow factor: {factor_name}")

    return merged
