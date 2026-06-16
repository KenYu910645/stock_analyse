"""Factor stability metrics."""

from __future__ import annotations

import pandas as pd


def compute_factor_stability(factor_values: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Compute average per-stock factor autocorrelation for each lag."""
    lags = lags or [1, 5, 20]
    rows = []
    for factor_name, factor_group in factor_values.groupby("factor_name", sort=True):
        for lag in lags:
            correlations = []
            for _, stock_group in factor_group.sort_values("date").groupby("stock_id", sort=False):
                sample = stock_group[["factor_zscore"]].copy()
                sample["lagged"] = sample["factor_zscore"].shift(lag)
                corr = sample["factor_zscore"].corr(sample["lagged"])
                if pd.notna(corr):
                    correlations.append(corr)
            rows.append(
                {
                    "factor_name": factor_name,
                    "lag": lag,
                    "average_autocorrelation": float(pd.Series(correlations).mean()) if correlations else float("nan"),
                    "stock_count": len(correlations),
                }
            )
    return pd.DataFrame(rows)

