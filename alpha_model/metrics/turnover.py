"""Portfolio turnover and coverage metrics."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_turnover(factor_values: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """Compute top-quantile turnover by factor/date."""
    rows = []
    for factor_name, factor_group in factor_values.groupby("factor_name", sort=True):
        previous_top: set[str] | None = None
        for date_value, date_group in factor_group.groupby("date", sort=True):
            valid = date_group.dropna(subset=["factor_zscore"]).copy()
            if len(valid) < 2:
                continue
            ranks = valid["factor_zscore"].rank(method="first")
            valid["quantile"] = np.ceil(ranks * quantiles / len(valid)).clip(1, quantiles)
            top = set(valid.loc[valid["quantile"] == valid["quantile"].max(), "stock_id"].astype(str))
            if previous_top is None or not top:
                turnover = 0.0
            else:
                turnover = 1 - len(top & previous_top) / len(top)
            rows.append(
                {
                    "date": date_value,
                    "factor_name": factor_name,
                    "top_count": len(top),
                    "turnover": turnover,
                }
            )
            previous_top = top
    return pd.DataFrame(rows)


def summarize_turnover(turnover_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize average and median turnover by factor and year."""
    if turnover_df.empty:
        return pd.DataFrame()
    df = turnover_df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return (
        df.groupby(["factor_name", "year"], as_index=False)["turnover"]
        .agg(average_turnover="mean", median_turnover="median")
    )


def compute_coverage(price_universe_df: pd.DataFrame, factor_values: pd.DataFrame) -> pd.DataFrame:
    """Compute factor coverage relative to the tradable universe."""
    universe_counts = (
        price_universe_df[price_universe_df["in_universe"]]
        .groupby("date")["stock_id"]
        .nunique()
        .rename("universe_count")
        .reset_index()
    )
    factor_counts = (
        factor_values.groupby(["factor_name", "date"])["stock_id"]
        .nunique()
        .rename("valid_factor_count")
        .reset_index()
    )
    coverage = factor_counts.merge(universe_counts, on="date", how="left")
    coverage["coverage"] = coverage["valid_factor_count"] / coverage["universe_count"]
    return coverage
