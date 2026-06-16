"""CLI entrypoint for momentum alpha factor evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpha_model.config import load_config
from alpha_model.data import build_universe, filter_date_range, load_price_data
from alpha_model.factors.momentum import compute_momentum_factors
from alpha_model.labels import compute_future_returns
from alpha_model.metrics.ic import compute_ic_timeseries, summarize_ic
from alpha_model.metrics.quantile import compute_quantile_returns
from alpha_model.metrics.stability import compute_factor_stability
from alpha_model.metrics.turnover import compute_coverage, compute_turnover
from alpha_model.preprocessing import build_factor_values
from alpha_model.reporting import ensure_dirs, write_plots, write_table_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Taiwan stock momentum alpha factors.")
    parser.add_argument("--config", default="alpha_model/config/momentum.yaml")
    parser.add_argument("--output-dir", default=None, help="Override table output directory.")
    parser.add_argument("--plot-dir", default=None, help="Override plot output directory.")
    parser.add_argument("--stock-limit", type=int, default=None, help="Load only the first N CSVs for smoke tests.")
    parser.add_argument("--start", default=None, help="Optional inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Optional inclusive end date, YYYY-MM-DD.")
    return parser.parse_args()


def run_pipeline(config: dict, stock_limit: int | None = None, start: str | None = None, end: str | None = None) -> dict:
    """Run the full alpha model pipeline and return output dataframes."""
    factor_names = list(config["factor"]["names"])
    horizons = [int(horizon) for horizon in config["labels"]["horizons"]]
    primary_horizon = int(config["evaluation"]["primary_horizon"])
    quantiles = int(config["evaluation"]["quantiles"])

    price_df = load_price_data(config, stock_limit=stock_limit)
    price_df = filter_date_range(price_df, start=start, end=end)
    universe_df = build_universe(price_df, config["universe"])
    factor_df = compute_momentum_factors(universe_df, factor_names)
    factor_values = build_factor_values(
        factor_df,
        factor_names,
        float(config["factor"]["winsorize_lower"]),
        float(config["factor"]["winsorize_upper"]),
        bool(config["factor"].get("zscore", True)),
    )
    future_returns = compute_future_returns(price_df, horizons)
    ic_timeseries = compute_ic_timeseries(factor_values, future_returns, horizons)
    ic_summary = summarize_ic(ic_timeseries)
    tc_config = config["evaluation"].get("transaction_cost", {})
    quantile_returns, quantile_cumulative, quantile_summary = compute_quantile_returns(
        factor_values,
        future_returns,
        primary_horizon,
        quantiles,
        bool(tc_config.get("enabled", False)),
        float(tc_config.get("round_trip_cost", 0.006)),
    )
    turnover = compute_turnover(factor_values, quantiles)
    coverage = compute_coverage(universe_df, factor_values)
    factor_stability = compute_factor_stability(factor_values)
    ic_summary = ic_summary.merge(
        quantile_summary[quantile_summary["quantile"].eq("long_short")],
        on="factor_name",
        how="left",
        suffixes=("", "_long_short"),
    )

    return {
        "factor_values": factor_values,
        "future_returns": future_returns,
        "ic_summary": ic_summary,
        "ic_timeseries": ic_timeseries,
        "quantile_returns": quantile_returns,
        "quantile_cumulative": quantile_cumulative,
        "turnover": turnover,
        "coverage": coverage,
        "factor_stability": factor_stability,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["output"]["dir"] = args.output_dir
    if args.plot_dir:
        config["output"]["plot_dir"] = args.plot_dir

    output_dir = Path(config["output"]["dir"])
    plot_dir = Path(config["output"]["plot_dir"])
    ensure_dirs(output_dir, plot_dir)

    outputs = run_pipeline(config, stock_limit=args.stock_limit, start=args.start, end=args.end)
    write_table_outputs(output_dir=output_dir, **outputs)
    write_plots(
        plot_dir,
        outputs["ic_timeseries"],
        outputs["quantile_cumulative"],
        outputs["turnover"],
        outputs["coverage"],
    )
    print(f"Alpha model tables written to {output_dir}.")
    print(f"Alpha model plots written to {plot_dir}.")


if __name__ == "__main__":
    main()

