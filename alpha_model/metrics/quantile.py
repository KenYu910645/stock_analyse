"""Quantile portfolio analysis."""

from __future__ import annotations

import pandas as pd
import numpy as np

from alpha_model.metrics.performance import summarize_returns


def compute_quantile_returns(
    factor_values: pd.DataFrame,
    future_returns: pd.DataFrame,
    horizon: int,
    quantiles: int,
    transaction_cost_enabled: bool = False,
    round_trip_cost: float = 0.006,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute quantile returns, cumulative returns, and summary metrics."""
    label_col = f"future_return_{horizon}d"
    merged = factor_values.merge(future_returns[["date", "stock_id", label_col]], on=["date", "stock_id"], how="left")
    merged = merged.dropna(subset=["factor_zscore", label_col]).copy()
    grouped = merged.groupby(["factor_name", "date"])["factor_zscore"]
    ranks = grouped.rank(method="first")
    counts = grouped.transform("size")
    merged["quantile"] = np.ceil(ranks * quantiles / counts).clip(1, quantiles)
    merged.loc[counts < 2, "quantile"] = pd.NA
    merged = merged.dropna(subset=["quantile"]).copy()
    merged["quantile"] = merged["quantile"].astype(int)

    quantile_returns = (
        merged.groupby(["factor_name", "date", "quantile"], as_index=False)[label_col]
        .mean()
        .rename(columns={label_col: "return"})
    )

    long_short = []
    for (factor_name, date_value), group in quantile_returns.groupby(["factor_name", "date"], sort=True):
        top = group[group["quantile"] == group["quantile"].max()]["return"].mean()
        bottom = group[group["quantile"] == group["quantile"].min()]["return"].mean()
        before_cost = top - bottom
        after_cost = before_cost - round_trip_cost if transaction_cost_enabled else before_cost
        long_short.append(
            {
                "factor_name": factor_name,
                "date": date_value,
                "quantile": "long_short",
                "return": before_cost,
                "return_after_cost": after_cost,
            }
        )
    long_short_df = pd.DataFrame(long_short)
    quantile_returns["return_after_cost"] = quantile_returns["return"]
    quantile_returns = pd.concat([quantile_returns, long_short_df], ignore_index=True)

    cumulative = quantile_returns.sort_values(["factor_name", "quantile", "date"]).copy()
    cumulative["cumulative_return"] = (
        cumulative.groupby(["factor_name", "quantile"])["return"]
        .transform(lambda returns: (1 + returns.fillna(0)).cumprod() - 1)
    )
    cumulative["cumulative_return_after_cost"] = (
        cumulative.groupby(["factor_name", "quantile"])["return_after_cost"]
        .transform(lambda returns: (1 + returns.fillna(0)).cumprod() - 1)
    )

    summary_rows = []
    for (factor_name, quantile), group in quantile_returns.groupby(["factor_name", "quantile"], sort=True):
        stats = summarize_returns(group.sort_values("date")["return"])
        stats_after_cost = summarize_returns(group.sort_values("date")["return_after_cost"])
        summary_rows.append(
            {
                "factor_name": factor_name,
                "quantile": quantile,
                "mean_return": float(group["return"].mean()),
                **stats,
                "annualized_return_after_cost": stats_after_cost["annualized_return"],
                "sharpe_ratio_after_cost": stats_after_cost["sharpe_ratio"],
            }
        )

    return quantile_returns, cumulative, pd.DataFrame(summary_rows)
