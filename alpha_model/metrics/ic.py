"""Information coefficient metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def _grouped_corr(sample: pd.DataFrame, x_col: str, y_col: str, value_col: str) -> pd.DataFrame:
    """Compute Pearson correlation per factor/date using aggregate sums."""
    work = sample[["factor_name", "date", x_col, y_col]].dropna().copy()
    work["_x2"] = work[x_col] * work[x_col]
    work["_y2"] = work[y_col] * work[y_col]
    work["_xy"] = work[x_col] * work[y_col]
    grouped = (
        work.groupby(["factor_name", "date"], as_index=False)
        .agg(
            sample_count=(x_col, "size"),
            sum_x=(x_col, "sum"),
            sum_y=(y_col, "sum"),
            sum_x2=("_x2", "sum"),
            sum_y2=("_y2", "sum"),
            sum_xy=("_xy", "sum"),
        )
    )
    numerator = grouped["sample_count"] * grouped["sum_xy"] - grouped["sum_x"] * grouped["sum_y"]
    x_denom = grouped["sample_count"] * grouped["sum_x2"] - grouped["sum_x"] ** 2
    y_denom = grouped["sample_count"] * grouped["sum_y2"] - grouped["sum_y"] ** 2
    denominator = np.sqrt(x_denom * y_denom)
    grouped[value_col] = numerator / denominator.replace(0, np.nan)
    grouped.loc[grouped["sample_count"] < 3, value_col] = np.nan
    return grouped[["factor_name", "date", "sample_count", value_col]]


def compute_ic_timeseries(
    factor_values: pd.DataFrame,
    future_returns: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Compute per-date IC and Rank IC for each factor and horizon."""
    merged = factor_values.merge(future_returns, on=["date", "stock_id"], how="left")
    frames = []
    for horizon in horizons:
        label_col = f"future_return_{horizon}d"
        sample = merged[["factor_name", "date", "factor_zscore", label_col]].dropna().copy()
        if sample.empty:
            continue

        ic = _grouped_corr(sample, "factor_zscore", label_col, "ic")
        sample["factor_rank"] = sample.groupby(["factor_name", "date"])["factor_zscore"].rank(method="average")
        sample["return_rank"] = sample.groupby(["factor_name", "date"])[label_col].rank(method="average")
        rank_ic = _grouped_corr(sample, "factor_rank", "return_rank", "rank_ic").drop(columns=["sample_count"])
        frame = ic.merge(rank_ic, on=["factor_name", "date"], how="outer")
        frame["horizon"] = horizon
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "factor_name", "horizon", "ic", "rank_ic", "sample_count"])

    return (
        pd.concat(frames, ignore_index=True)
        [["date", "factor_name", "horizon", "ic", "rank_ic", "sample_count"]]
        .dropna(subset=["ic", "rank_ic"], how="all")
    )


def _summarize_series(values: pd.Series, prefix: str) -> dict[str, float]:
    clean = values.dropna()
    count = int(len(clean))
    if count == 0:
        return {
            f"mean_{prefix}": float("nan"),
            f"std_{prefix}": float("nan"),
            f"{prefix}_ir": float("nan"),
            f"positive_{prefix}_ratio": float("nan"),
        }

    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    return {
        f"mean_{prefix}": mean,
        f"std_{prefix}": std,
        f"{prefix}_ir": mean / std if std else float("nan"),
        f"positive_{prefix}_ratio": float((clean > 0).mean()),
    }


def summarize_ic(ic_timeseries: pd.DataFrame) -> pd.DataFrame:
    """Summarize IC and Rank IC by factor/horizon."""
    rows = []
    for (factor_name, horizon), group in ic_timeseries.groupby(["factor_name", "horizon"], sort=True):
        clean_ic = group["ic"].dropna()
        mean_ic = float(clean_ic.mean()) if not clean_ic.empty else float("nan")
        std_ic = float(clean_ic.std(ddof=1)) if len(clean_ic) > 1 else float("nan")
        t_stat = mean_ic / (std_ic / math.sqrt(len(clean_ic))) if len(clean_ic) > 1 and std_ic else float("nan")
        p_value = float(stats.ttest_1samp(clean_ic, 0, nan_policy="omit").pvalue) if len(clean_ic) > 1 else float("nan")
        row = {
            "factor_name": factor_name,
            "horizon": horizon,
            "sample_count": int(group["sample_count"].sum()),
            "date_count": int(group["date"].nunique()),
            "t_stat": t_stat,
            "p_value": p_value,
        }
        row.update(_summarize_series(group["ic"], "ic"))
        row.update(_summarize_series(group["rank_ic"], "rank_ic"))
        rows.append(row)
    return pd.DataFrame(rows)
