from __future__ import annotations

import math

import pandas as pd

from alpha_model.data import build_universe
from alpha_model.factors.momentum import compute_momentum_factors
from alpha_model.labels import compute_future_returns
from alpha_model.metrics.ic import compute_ic_timeseries
from alpha_model.metrics.quantile import compute_quantile_returns
from alpha_model.metrics.turnover import compute_turnover
from alpha_model.preprocessing import build_factor_values, preprocess_factor


def make_price_df(stock_ids=("A", "B", "C", "D"), periods=140) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rows = []
    for stock_index, stock_id in enumerate(stock_ids):
        for day_index, date in enumerate(dates):
            price = 10 + stock_index * 2 + day_index * (0.1 + stock_index * 0.01)
            rows.append(
                {
                    "date": date,
                    "stock_id": stock_id,
                    "adj_open": price,
                    "adj_high": price * 1.01,
                    "adj_low": price * 0.99,
                    "adj_close": price,
                    "volume": 1000 + stock_index,
                    "turnover": 10_000_000,
                }
            )
    return pd.DataFrame(rows)


def test_momentum_uses_current_and_past_prices_only() -> None:
    df = make_price_df(stock_ids=("A",), periods=70)
    factors = compute_momentum_factors(df, ["momentum_20d", "momentum_60_5"])
    row_20 = factors.iloc[20]
    row_65 = factors.iloc[65]

    assert math.isclose(row_20["momentum_20d"], row_20["adj_close"] / factors.iloc[0]["adj_close"] - 1)
    assert math.isclose(row_65["momentum_60_5"], factors.iloc[60]["adj_close"] / factors.iloc[0]["adj_close"] - 1)


def test_future_returns_are_shifted_forward() -> None:
    df = make_price_df(stock_ids=("A",), periods=8)
    labels = compute_future_returns(df, [1, 5])

    assert math.isclose(labels.iloc[0]["future_return_1d"], df.iloc[1]["adj_close"] / df.iloc[0]["adj_close"] - 1)
    assert math.isclose(labels.iloc[0]["future_return_5d"], df.iloc[5]["adj_close"] / df.iloc[0]["adj_close"] - 1)
    assert pd.isna(labels.iloc[-1]["future_return_1d"])


def test_universe_excludes_insufficient_history() -> None:
    df = make_price_df(stock_ids=("A",), periods=5)
    universe = build_universe(
        df,
        {
            "min_price": 5,
            "min_avg_turnover_20d": None,
            "min_history_days": 3,
        },
    )

    assert universe["in_universe"].tolist() == [False, False, True, True, True]


def test_preprocess_factor_is_cross_sectional_by_date() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3,
            "stock_id": ["A", "B", "C"] * 2,
            "in_universe": [True] * 6,
            "momentum_20d": [1.0, 2.0, 100.0, 3.0, 4.0, 5.0],
        }
    )
    processed = preprocess_factor(df, "momentum_20d", 0.01, 0.99)

    for _, group in processed.groupby("date"):
        assert abs(group["momentum_20d_zscore"].mean()) < 1e-12


def test_ic_compares_same_date_factor_with_future_return() -> None:
    factor_values = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 4,
            "stock_id": ["A", "B", "C", "D"],
            "factor_name": ["momentum_20d"] * 4,
            "factor_zscore": [1, 2, 3, 4],
        }
    )
    future_returns = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 4,
            "stock_id": ["A", "B", "C", "D"],
            "future_return_5d": [0.01, 0.02, 0.03, 0.04],
        }
    )
    ic = compute_ic_timeseries(factor_values, future_returns, [5])

    assert math.isclose(ic.iloc[0]["ic"], 1.0)
    assert math.isclose(ic.iloc[0]["rank_ic"], 1.0)


def test_quantile_groups_are_per_date() -> None:
    dates = [pd.Timestamp("2024-01-01")] * 5 + [pd.Timestamp("2024-01-02")] * 5
    factor_values = pd.DataFrame(
        {
            "date": dates,
            "stock_id": list("ABCDE") * 2,
            "factor_name": ["momentum_20d"] * 10,
            "factor_zscore": list(range(5)) + list(range(5)),
        }
    )
    future_returns = factor_values[["date", "stock_id"]].copy()
    future_returns["future_return_5d"] = factor_values["factor_zscore"] / 100
    quantile_returns, _, _ = compute_quantile_returns(factor_values, future_returns, 5, 5)

    date_counts = quantile_returns[quantile_returns["quantile"] != "long_short"].groupby("date")["quantile"].nunique()
    assert date_counts.tolist() == [5, 5]


def test_turnover_uses_top_quantile_overlap() -> None:
    factor_values = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 4 + [pd.Timestamp("2024-01-02")] * 4,
            "stock_id": ["A", "B", "C", "D"] * 2,
            "factor_name": ["momentum_20d"] * 8,
            "factor_zscore": [1, 2, 3, 4, 4, 3, 2, 1],
        }
    )
    turnover = compute_turnover(factor_values, quantiles=2)

    assert turnover.iloc[0]["turnover"] == 0
    assert turnover.iloc[1]["turnover"] == 1


def test_build_factor_values_drops_missing_future_unrelated_invalid_factors() -> None:
    df = make_price_df(periods=125)
    universe = build_universe(
        df,
        {
            "min_price": 5,
            "min_avg_turnover_20d": None,
            "min_history_days": 120,
        },
    )
    factors = compute_momentum_factors(universe, ["momentum_20d"])
    values = build_factor_values(factors, ["momentum_20d"], 0.01, 0.99)

    assert values["date"].min() >= df["date"].unique()[119]
    assert values["factor_zscore"].notna().all()
