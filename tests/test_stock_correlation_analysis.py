from __future__ import annotations

import math

import pandas as pd

from stock_correlation_analysis import (
    COMMON_STOCK_TYPE,
    TWSE_MARKET,
    calculate_correlation_matrix,
    calculate_log_returns,
    calculate_market_residual_returns,
    calculate_rolling_correlation,
    calculate_turnover_weighted_group_returns,
    cluster_correlation_matrix,
    clean_price_data,
    get_industry_representative_stock_groups,
    get_representative_stock_list,
    get_top_correlated_peers,
    load_metadata,
    make_price_matrix,
    make_turnover_matrix,
    order_by_mean_peer_correlation,
    rank_correlation_to_target,
)


def make_raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Date": "2024-01-01", "stock_id": "2330", "Close_adj": 100},
            {"Date": "2024-01-02", "stock_id": "2330", "Close_adj": 110},
            {"Date": "2024-01-03", "stock_id": "2330", "Close_adj": 125},
            {"Date": "2024-01-01", "stock_id": "2454", "Close_adj": 200},
            {"Date": "2024-01-02", "stock_id": "2454", "Close_adj": 220},
            {"Date": "2024-01-03", "stock_id": "2454", "Close_adj": 250},
            {"Date": "2024-01-01", "stock_id": "2881", "Close_adj": 50},
            {"Date": "2024-01-02", "stock_id": "2881", "Close_adj": 49},
            {"Date": "2024-01-03", "stock_id": "2881", "Close_adj": 51},
        ]
    )


def test_clean_price_data_filters_to_twse_common_metadata(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"Code": "2330", "Name": "TSMC", "Type": COMMON_STOCK_TYPE, "Market": TWSE_MARKET, "Group": "Semi"},
            {"Code": "0050", "Name": "ETF", "Type": "ETF", "Market": TWSE_MARKET, "Group": "ETF"},
        ]
    ).to_csv(metadata_path, index=False)
    raw = pd.DataFrame(
        [
            {"Date": "2024-01-01", "stock_id": "2330", "Close_adj": 100},
            {"Date": "2024-01-01", "stock_id": "0050", "Close_adj": 100},
        ]
    )

    cleaned = clean_price_data(raw, metadata=load_metadata(metadata_path))

    assert cleaned["stock_id"].tolist() == ["2330"]
    assert cleaned["stock_name"].tolist() == ["TSMC"]
    assert cleaned["industry"].tolist() == ["Semi"]


def test_log_returns_use_log_difference_not_raw_prices() -> None:
    cleaned = clean_price_data(make_raw_df(), metadata=None, common_only=False)
    prices = make_price_matrix(cleaned)
    returns = calculate_log_returns(prices, min_valid_ratio=0.5)

    assert math.isclose(
        returns.loc[pd.Timestamp("2024-01-02"), "2330"],
        math.log(110) - math.log(100),
    )


def test_turnover_weighted_group_returns_weight_high_turnover_more() -> None:
    clean_df = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "stock_id": "A", "adj_close": 100, "turnover": 100, "stock_name": "A", "industry": "Semi"},
            {"date": pd.Timestamp("2024-01-02"), "stock_id": "A", "adj_close": 110, "turnover": 900, "stock_name": "A", "industry": "Semi"},
            {"date": pd.Timestamp("2024-01-01"), "stock_id": "B", "adj_close": 100, "turnover": 100, "stock_name": "B", "industry": "Semi"},
            {"date": pd.Timestamp("2024-01-02"), "stock_id": "B", "adj_close": 90, "turnover": 100, "stock_name": "B", "industry": "Semi"},
            {"date": pd.Timestamp("2024-01-01"), "stock_id": "C", "adj_close": 100, "turnover": 100, "stock_name": "C", "industry": "Finance"},
            {"date": pd.Timestamp("2024-01-02"), "stock_id": "C", "adj_close": 105, "turnover": 100, "stock_name": "C", "industry": "Finance"},
        ]
    )
    returns = calculate_log_returns(make_price_matrix(clean_df), min_valid_ratio=0.5)
    turnover = make_turnover_matrix(clean_df)
    stock_info = clean_df[["stock_id", "stock_name", "industry"]].drop_duplicates("stock_id").set_index("stock_id", drop=False)

    group_returns = calculate_turnover_weighted_group_returns(
        returns,
        turnover,
        stock_info,
        min_valid_members=1,
    )

    expected = (math.log(110 / 100) * 900 + math.log(90 / 100) * 100) / 1000
    assert math.isclose(group_returns.loc[pd.Timestamp("2024-01-02"), "Semi"], expected)
    assert math.isclose(group_returns.loc[pd.Timestamp("2024-01-02"), "Finance"], math.log(105 / 100))


def test_market_residual_returns_estimate_beta_and_remove_market_component() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    market = pd.Series([0.01, -0.02, 0.03, 0.00, 0.02], index=dates)
    stock_returns = pd.DataFrame(
        {
            "A": 0.001 + 2.0 * market,
            "B": -0.002 + 0.5 * market + pd.Series([0.01, -0.01, 0.0, 0.01, -0.01], index=dates),
        },
        index=dates,
    )

    residuals, beta = calculate_market_residual_returns(
        stock_returns,
        market,
        min_observations=5,
    )

    beta_by_stock = beta.set_index("stock_id")
    assert math.isclose(beta_by_stock.loc["A", "beta_market"], 2.0)
    assert abs(residuals["A"]).max() < 1e-12
    assert "r_squared" in beta.columns


def test_representative_stock_list_keeps_available_manual_order() -> None:
    returns = pd.DataFrame(columns=["2454", "2330", "2881"])

    assert get_representative_stock_list(returns) == ["2330", "2454", "2881"]


def test_industry_representatives_pick_each_industry_by_turnover() -> None:
    returns = pd.DataFrame(columns=["2330", "2454", "2881", "2603"])
    stock_info = pd.DataFrame(
        [
            {"stock_id": "2330", "stock_name": "TSMC", "industry": "Semi", "avg_turnover": 10},
            {"stock_id": "2454", "stock_name": "MTK", "industry": "Semi", "avg_turnover": 20},
            {"stock_id": "2881", "stock_name": "Fubon", "industry": "Finance", "avg_turnover": 5},
            {"stock_id": "2603", "stock_name": "Evergreen", "industry": "Shipping", "avg_turnover": 7},
        ]
    ).set_index("stock_id", drop=False)

    groups = get_industry_representative_stock_groups(
        returns,
        stock_info,
        representatives_per_industry=1,
    )

    assert groups == {
        "Finance": ["2881"],
        "Semi": ["2454"],
        "Shipping": ["2603"],
    }


def test_correlation_and_target_ranking() -> None:
    cleaned = clean_price_data(make_raw_df(), metadata=None, common_only=False)
    returns = calculate_log_returns(make_price_matrix(cleaned), min_valid_ratio=0.5)
    corr = calculate_correlation_matrix(returns, ["2330", "2454", "2881"])
    top, bottom = rank_correlation_to_target(returns, target="2330", top_n=1)

    assert math.isclose(corr.loc["2330", "2454"], 1.0)
    assert top.iloc[0]["stock_id"] == "2454"
    assert bottom.iloc[0]["stock_id"] == "2881"


def test_rolling_correlation_only_uses_available_compare_stocks() -> None:
    cleaned = clean_price_data(make_raw_df(), metadata=None, common_only=False)
    returns = calculate_log_returns(make_price_matrix(cleaned), min_valid_ratio=0.5)

    rolling = calculate_rolling_correlation(
        returns,
        target="2330",
        compare_stocks=["2454", "9999"],
        window=2,
    )

    assert rolling.columns.tolist() == ["2454"]


def test_order_by_mean_peer_correlation_places_core_stock_first() -> None:
    corr = pd.DataFrame(
        [
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.1],
            [0.8, 0.1, 1.0],
        ],
        index=["core", "edge_a", "edge_b"],
        columns=["core", "edge_a", "edge_b"],
    )

    ordered = order_by_mean_peer_correlation(corr)

    assert ordered.index.tolist()[0] == "core"
    assert ordered.columns.tolist()[0] == "core"


def test_cluster_correlation_matrix_keeps_related_items_adjacent() -> None:
    corr = pd.DataFrame(
        [
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )

    clustered = cluster_correlation_matrix(corr)
    order = clustered.index.tolist()

    assert abs(order.index("A") - order.index("B")) == 1
    assert abs(order.index("C") - order.index("D")) == 1


def test_get_top_correlated_peers_excludes_target_and_sorts_descending() -> None:
    corr = pd.DataFrame(
        [
            [1.0, 0.2, 0.8, 0.5],
            [0.2, 1.0, 0.1, 0.3],
            [0.8, 0.1, 1.0, 0.4],
            [0.5, 0.3, 0.4, 1.0],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )

    peers = get_top_correlated_peers(corr, "A", top_n=2)

    assert peers.index.tolist() == ["C", "D"]
    assert peers.tolist() == [0.8, 0.5]


def test_top_peer_matrix_includes_target_and_peer_correlations() -> None:
    corr = pd.DataFrame(
        [
            [1.0, 0.8, 0.5],
            [0.8, 1.0, 0.2],
            [0.5, 0.2, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    peers = get_top_correlated_peers(corr, "A", top_n=2)
    matrix = corr.loc[["A"] + peers.index.tolist(), ["A"] + peers.index.tolist()]

    assert matrix.index.tolist() == ["A", "B", "C"]
    assert matrix.loc["A", "A"] == 1.0
    assert matrix.loc["B", "C"] == 0.2
