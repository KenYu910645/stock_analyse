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

