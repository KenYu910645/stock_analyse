"""Forward-return label generation."""

from __future__ import annotations

import pandas as pd


def compute_future_returns(price_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Compute future returns from adjusted close for each horizon."""
    df = price_df.sort_values(["stock_id", "date"]).copy()
    grouped_close = df.groupby("stock_id", sort=False)["adj_close"]
    output = df[["date", "stock_id", "adj_close"]].copy()
    for horizon in horizons:
        future_close = grouped_close.shift(-int(horizon))
        output[f"future_return_{horizon}d"] = future_close / df["adj_close"] - 1
    return output.drop(columns=["adj_close"])


def compute_next_open_future_returns(price_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Compute future returns from next adjusted open to later adjusted open."""
    df = price_df.sort_values(["stock_id", "date"]).copy()
    grouped_open = df.groupby("stock_id", sort=False)["adj_open"]
    entry_open = grouped_open.shift(-1)
    output = df[["date", "stock_id"]].copy()
    for horizon in horizons:
        exit_open = grouped_open.shift(-(int(horizon) + 1))
        output[f"future_return_{horizon}d"] = exit_open / entry_open - 1
    return output

