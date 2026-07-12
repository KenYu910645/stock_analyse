"""Portfolio turnover and coverage metrics."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_turnover(factor_values: pd.DataFrame, quantiles: int) -> pd.DataFrame:
    """Compute top-quantile turnover by factor/date."""
    valid = factor_values.dropna(subset=["factor_zscore"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["date", "factor_name", "top_count", "turnover"])

    grouped = valid.groupby(["factor_name", "date"], sort=True)["factor_zscore"]
    valid["_rank"] = grouped.rank(method="first")
    valid["_count"] = grouped.transform("size")
    valid = valid[valid["_count"].ge(2)].copy()
    if valid.empty:
        return pd.DataFrame(columns=["date", "factor_name", "top_count", "turnover"])

    date_order = valid[["factor_name", "date"]].drop_duplicates().sort_values(["factor_name", "date"])
    date_order["_date_seq"] = date_order.groupby("factor_name").cumcount()
    valid = valid.merge(date_order, on=["factor_name", "date"], how="left")
    valid["_quantile"] = np.ceil(valid["_rank"] * quantiles / valid["_count"]).clip(1, quantiles)
    top = valid[valid["_quantile"].eq(quantiles)][["factor_name", "date", "_date_seq", "stock_id"]].copy()
    if top.empty:
        return pd.DataFrame(columns=["date", "factor_name", "top_count", "turnover"])

    previous_top = top[["factor_name", "_date_seq", "stock_id"]].copy()
    previous_top["_date_seq"] = previous_top["_date_seq"] + 1
    previous_top["_was_previous_top"] = True
    top = top.merge(previous_top, on=["factor_name", "_date_seq", "stock_id"], how="left")
    top["_overlap"] = top["_was_previous_top"].fillna(False).astype(int)
    result = (
        top.groupby(["factor_name", "date", "_date_seq"], as_index=False)
        .agg(top_count=("stock_id", "size"), overlap=("_overlap", "sum"))
        .sort_values(["factor_name", "date"])
    )
    result["turnover"] = 1 - result["overlap"] / result["top_count"]
    result.loc[result["_date_seq"].eq(0), "turnover"] = 0.0
    return result[["date", "factor_name", "top_count", "turnover"]]


def compute_coverage(
    price_universe_df: pd.DataFrame,
    factor_values: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute factor coverage, retaining dates with zero valid factor rows."""
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
    names = list(dict.fromkeys(factor_names or factor_values["factor_name"].dropna().tolist()))
    if universe_counts.empty or not names:
        return pd.DataFrame(
            columns=["factor_name", "date", "valid_factor_count", "universe_count", "coverage"]
        )

    coverage_grid = pd.MultiIndex.from_product(
        [names, universe_counts["date"]],
        names=["factor_name", "date"],
    ).to_frame(index=False)
    coverage = coverage_grid.merge(factor_counts, on=["factor_name", "date"], how="left")
    coverage["valid_factor_count"] = coverage["valid_factor_count"].fillna(0).astype(int)
    coverage = coverage.merge(universe_counts, on="date", how="left")
    coverage["coverage"] = coverage["valid_factor_count"] / coverage["universe_count"]
    return coverage.sort_values(["factor_name", "date"]).reset_index(drop=True)
