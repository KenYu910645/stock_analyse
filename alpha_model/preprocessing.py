"""Cross-sectional factor preprocessing."""

from __future__ import annotations

import pandas as pd
import numpy as np


def preprocess_factor(
    df: pd.DataFrame,
    factor_name: str,
    lower_q: float,
    upper_q: float,
    zscore: bool = True,
) -> pd.DataFrame:
    """Add raw, winsorized, and z-score factor columns for one factor."""
    result = df.copy()
    raw_col = f"{factor_name}_raw"
    winsor_col = f"{factor_name}_winsorized"
    z_col = f"{factor_name}_zscore"
    result[raw_col] = pd.to_numeric(result[factor_name], errors="coerce")

    result[raw_col] = result[raw_col].replace([np.inf, -np.inf], np.nan)
    grouped_raw = result.groupby("date")[raw_col]
    lower = grouped_raw.transform(lambda values: values.quantile(lower_q))
    upper = grouped_raw.transform(lambda values: values.quantile(upper_q))
    result[winsor_col] = result[raw_col].clip(lower=lower, upper=upper)

    if zscore:
        grouped = result.groupby("date")[winsor_col]
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        result[z_col] = (result[winsor_col] - mean) / std.replace(0, pd.NA)
    else:
        result[z_col] = result[winsor_col]

    return result


def build_factor_values(
    factor_df: pd.DataFrame,
    factor_names: list[str],
    lower_q: float,
    upper_q: float,
    zscore: bool = True,
) -> pd.DataFrame:
    """Return one long factor-value table filtered to the tradable universe."""
    frames = []
    universe_df = factor_df[factor_df["in_universe"]].copy()
    for factor_name in factor_names:
        processed = preprocess_factor(universe_df, factor_name, lower_q, upper_q, zscore)
        frames.append(
            processed[
                [
                    "date",
                    "stock_id",
                    "in_universe",
                    f"{factor_name}_raw",
                    f"{factor_name}_winsorized",
                    f"{factor_name}_zscore",
                ]
            ]
            .rename(
                columns={
                    f"{factor_name}_raw": "factor_raw",
                    f"{factor_name}_winsorized": "factor_winsorized",
                    f"{factor_name}_zscore": "factor_zscore",
                }
            )
            .assign(factor_name=factor_name)
        )

    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["factor_raw", "factor_winsorized", "factor_zscore"])
        .sort_values(["factor_name", "date", "stock_id"])
        .reset_index(drop=True)
    )
