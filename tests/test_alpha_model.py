from __future__ import annotations

import math

import pandas as pd
import pytest

from alpha_model.data import build_universe, load_price_data
from alpha_model.factors.institutional_flow import compute_trust_flow_factors
from alpha_model.factors.momentum import compute_momentum_factors
from alpha_model.labels import compute_future_returns, compute_next_open_future_returns
from alpha_model.metrics.ic import compute_ic_timeseries
from alpha_model.metrics.quantile import compute_quantile_returns
from alpha_model.metrics.turnover import compute_coverage, compute_turnover
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


def make_price_loader_config(tmp_path, *, allow_unfiltered_universe: bool = False) -> dict:
    price_dir = tmp_path / "price"
    price_dir.mkdir()
    for stock_id in ["0050", "2330"]:
        pd.DataFrame(
            [
                {
                    "Date": "2024-01-02",
                    "open_adj": 10,
                    "high_adj": 11,
                    "low_adj": 9,
                    "close_adj": 10,
                    "Capacity": 100,
                    "Turnover": 1000,
                }
            ]
        ).to_csv(price_dir / f"{stock_id}_test.csv", index=False)
    return {
        "data": {
            "price_dir": str(price_dir),
            "metadata_path": str(tmp_path / "metadata.csv"),
            "allow_unfiltered_universe": allow_unfiltered_universe,
            "date_col": "Date",
            "open_col": "open_adj",
            "high_col": "high_adj",
            "low_col": "low_adj",
            "close_col": "close_adj",
            "volume_col": "Capacity",
            "turnover_col": "Turnover",
        }
    }


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


def test_next_open_future_returns_start_after_signal_date() -> None:
    df = make_price_df(stock_ids=("A",), periods=8)
    labels = compute_next_open_future_returns(df, [1, 5])

    assert math.isclose(labels.iloc[0]["future_return_1d"], df.iloc[2]["adj_open"] / df.iloc[1]["adj_open"] - 1)
    assert math.isclose(labels.iloc[0]["future_return_5d"], df.iloc[6]["adj_open"] / df.iloc[1]["adj_open"] - 1)
    assert pd.isna(labels.iloc[-2]["future_return_1d"])


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


def test_price_loader_fails_closed_when_metadata_is_missing(tmp_path) -> None:
    config = make_price_loader_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="Metadata catalog does not exist"):
        load_price_data(config)


def test_price_loader_fails_closed_when_metadata_schema_is_malformed(tmp_path) -> None:
    config = make_price_loader_config(tmp_path)
    pd.DataFrame([{"Code": "2330", "Name": "test"}]).to_csv(
        config["data"]["metadata_path"],
        index=False,
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_price_data(config)


def test_price_loader_requires_explicit_opt_in_for_unfiltered_universe(tmp_path) -> None:
    config = make_price_loader_config(tmp_path, allow_unfiltered_universe=True)

    loaded = load_price_data(config)

    assert set(loaded["stock_id"]) == {"0050", "2330"}


def test_price_loader_defaults_to_twse_listed_common_stocks(tmp_path) -> None:
    config = make_price_loader_config(tmp_path)
    pd.DataFrame(
        [
            {"Code": "0050", "Type": "ETF", "Market": "上市"},
            {"Code": "2330", "Type": "股票", "Market": "上市"},
        ]
    ).to_csv(config["data"]["metadata_path"], index=False, encoding="utf-8-sig")

    loaded = load_price_data(config)

    assert loaded["stock_id"].tolist() == ["2330"]


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


def test_trust_flow_factors_use_net_value_and_no_future_data() -> None:
    price = make_price_df(stock_ids=("A",), periods=5)
    price["volume"] = 100
    price["turnover"] = 1000
    institutional = pd.DataFrame(
        {
            "Date": price["date"],
            "Code": ["A"] * 5,
            "InvestmentTrustBuy": [20, 30, 10, 0, 0],
            "InvestmentTrustSell": [10, 10, 20, 20, 0],
        }
    )
    factors = compute_trust_flow_factors(
        price,
        [
            "trust_net_value_1d_to_turnover",
            "trust_net_value_3d_to_turnover",
            "trust_buy_purity_1d",
            "trust_buy_streak_3d",
            "trust_sell_pressure_3d",
        ],
        institutional_df=institutional,
    )

    assert math.isclose(factors.iloc[0]["trust_net_value_1d_to_turnover"], 0.1)
    assert math.isclose(factors.iloc[2]["trust_net_value_3d_to_turnover"], (100 + 200 - 100) / 3000)
    assert math.isclose(factors.iloc[0]["trust_buy_purity_1d"], 10 / 30)
    assert factors.iloc[2]["trust_buy_streak_3d"] == 0
    assert math.isclose(factors.iloc[3]["trust_sell_pressure_3d"], 300 / 3000)


def test_trust_buy_streak_is_consecutive_and_capped_at_three_days() -> None:
    price = make_price_df(stock_ids=("A",), periods=7)
    institutional = pd.DataFrame(
        {
            "Date": price["date"],
            "Code": ["A"] * 7,
            "InvestmentTrustBuy": [10, 10, 10, 10, 0, 10, 10],
            "InvestmentTrustSell": [0, 0, 0, 0, 10, 0, 0],
        }
    )

    factors = compute_trust_flow_factors(
        price,
        ["trust_buy_streak_3d"],
        institutional_df=institutional,
    )

    expected = [float("nan"), float("nan"), 3.0, 3.0, 0.0, 1.0, 2.0]
    pd.testing.assert_series_equal(
        factors["trust_buy_streak_3d"],
        pd.Series(expected, name="trust_buy_streak_3d"),
    )


def test_missing_institutional_row_does_not_become_a_zero_factor() -> None:
    price = make_price_df(stock_ids=("A",), periods=5)
    institutional = pd.DataFrame(
        {
            "Date": price.loc[[0, 1, 3, 4], "date"].tolist(),
            "Code": ["A"] * 4,
            "InvestmentTrustBuy": [10, 10, 10, 10],
            "InvestmentTrustSell": [0, 0, 0, 0],
        }
    )
    factor_names = [
        "trust_net_value_1d_to_turnover",
        "trust_buy_purity_1d",
        "trust_buy_streak_3d",
        "trust_sell_pressure_3d",
    ]

    factors = compute_trust_flow_factors(
        price,
        factor_names,
        institutional_df=institutional,
    )

    missing_row = factors.iloc[2]
    assert missing_row[factor_names].isna().all()


def test_coverage_keeps_missing_factor_dates_and_zero_count_factors() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    universe = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "stock_id": ["A", "B", "A", "B"],
            "in_universe": [True, True, True, True],
        }
    )
    factor_values = pd.DataFrame(
        {
            "date": [dates[0]],
            "stock_id": ["A"],
            "factor_name": ["trust_buy_purity_1d"],
            "factor_zscore": [1.0],
        }
    )

    coverage = compute_coverage(
        universe,
        factor_values,
        factor_names=["trust_buy_purity_1d", "trust_buy_streak_3d"],
    ).set_index(["factor_name", "date"])

    assert coverage.loc[("trust_buy_purity_1d", dates[0]), "coverage"] == 0.5
    assert coverage.loc[("trust_buy_purity_1d", dates[1]), "coverage"] == 0.0
    assert coverage.loc[("trust_buy_streak_3d", dates[0]), "valid_factor_count"] == 0
    assert coverage.loc[("trust_buy_streak_3d", dates[1]), "coverage"] == 0.0


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
