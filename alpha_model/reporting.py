"""Output writing and plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dirs(output_dir: Path, plot_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)


def write_table_outputs(
    output_dir: Path,
    factor_values: pd.DataFrame,
    future_returns: pd.DataFrame,
    ic_summary: pd.DataFrame,
    ic_timeseries: pd.DataFrame,
    quantile_returns: pd.DataFrame,
    quantile_cumulative: pd.DataFrame,
    turnover: pd.DataFrame,
    coverage: pd.DataFrame,
    factor_stability: pd.DataFrame,
) -> None:
    """Persist public table outputs."""
    factor_values.to_parquet(output_dir / "factor_values.parquet", index=False)
    future_returns.to_parquet(output_dir / "future_returns.parquet", index=False)
    ic_summary.to_csv(output_dir / "ic_summary.csv", index=False, encoding="utf-8-sig")
    ic_timeseries.to_csv(output_dir / "ic_timeseries.csv", index=False, encoding="utf-8-sig")
    quantile_returns.to_csv(output_dir / "quantile_returns.csv", index=False, encoding="utf-8-sig")
    quantile_cumulative.to_csv(output_dir / "quantile_cumulative_returns.csv", index=False, encoding="utf-8-sig")
    turnover.to_csv(output_dir / "turnover.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "coverage.csv", index=False, encoding="utf-8-sig")
    factor_stability.to_csv(output_dir / "factor_stability.csv", index=False, encoding="utf-8-sig")


def _primary_factor(df: pd.DataFrame) -> str | None:
    if df.empty or "factor_name" not in df.columns:
        return None
    return str(sorted(df["factor_name"].dropna().unique())[0])


def plot_ic_timeseries(ic_timeseries: pd.DataFrame, plot_dir: Path) -> None:
    factor_name = _primary_factor(ic_timeseries)
    if not factor_name:
        return
    data = ic_timeseries[ic_timeseries["factor_name"] == factor_name].sort_values("date")
    for column, filename, title in [
        ("ic", "ic_timeseries.png", "IC Timeseries"),
        ("rank_ic", "rank_ic_timeseries.png", "Rank IC Timeseries"),
    ]:
        plt.figure(figsize=(12, 5))
        for horizon, group in data.groupby("horizon"):
            plt.plot(group["date"], group[column], label=f"{horizon}d", linewidth=1)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(f"{title} - {factor_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=150)
        plt.close()

    plt.figure(figsize=(8, 5))
    data["ic"].dropna().hist(bins=40)
    plt.title(f"IC Histogram - {factor_name}")
    plt.tight_layout()
    plt.savefig(plot_dir / "ic_histogram.png", dpi=150)
    plt.close()


def plot_quantile_returns(quantile_cumulative: pd.DataFrame, plot_dir: Path) -> None:
    factor_name = _primary_factor(quantile_cumulative)
    if not factor_name:
        return
    data = quantile_cumulative[quantile_cumulative["factor_name"] == factor_name].sort_values("date")
    plt.figure(figsize=(12, 6))
    for quantile, group in data[data["quantile"] != "long_short"].groupby("quantile"):
        plt.plot(group["date"], group["cumulative_return"], label=f"Q{quantile}", linewidth=1)
    plt.title(f"Quantile Cumulative Returns - {factor_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "quantile_cumulative_returns.png", dpi=150)
    plt.close()

    long_short = data[data["quantile"] == "long_short"]
    plt.figure(figsize=(12, 5))
    plt.plot(long_short["date"], long_short["cumulative_return"], label="before cost")
    if "cumulative_return_after_cost" in long_short.columns:
        plt.plot(long_short["date"], long_short["cumulative_return_after_cost"], label="after cost")
    plt.title(f"Long-Short Cumulative Returns - {factor_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "long_short_cumulative_returns.png", dpi=150)
    plt.close()


def plot_turnover_coverage(turnover: pd.DataFrame, coverage: pd.DataFrame, plot_dir: Path) -> None:
    factor_name = _primary_factor(turnover)
    if factor_name:
        data = turnover[turnover["factor_name"] == factor_name].sort_values("date")
        plt.figure(figsize=(12, 5))
        plt.plot(data["date"], data["turnover"], linewidth=1)
        plt.title(f"Top Quantile Turnover - {factor_name}")
        plt.tight_layout()
        plt.savefig(plot_dir / "turnover_timeseries.png", dpi=150)
        plt.close()

    factor_name = _primary_factor(coverage)
    if factor_name:
        data = coverage[coverage["factor_name"] == factor_name].sort_values("date")
        plt.figure(figsize=(12, 5))
        plt.plot(data["date"], data["coverage"], linewidth=1)
        plt.title(f"Coverage - {factor_name}")
        plt.tight_layout()
        plt.savefig(plot_dir / "coverage_timeseries.png", dpi=150)
        plt.close()


def write_plots(
    plot_dir: Path,
    ic_timeseries: pd.DataFrame,
    quantile_cumulative: pd.DataFrame,
    turnover: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    """Write public PNG plots."""
    plot_ic_timeseries(ic_timeseries, plot_dir)
    plot_quantile_returns(quantile_cumulative, plot_dir)
    plot_turnover_coverage(turnover, coverage, plot_dir)

