from __future__ import annotations

import math

import pandas as pd

from stock_regime_analysis import (
    COMMON_STOCK_TYPE,
    TWSE_MARKET,
    calculate_atr_percent,
    compute_regime_dataframe,
    generate_reports,
    get_trend_axis_range_from_regimes,
    get_trend_axis_range_from_members,
    get_trend_axis_range_from_records,
    load_listed_common_metadata,
    minmax_normalize,
    regression_log_slope,
)


def make_adjusted_price_df(periods: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rows = []
    for index, date in enumerate(dates):
        close = 100 * math.exp(index * 0.01)
        rows.append(
            {
                "Date": date,
                "Open_adj": close * 0.99,
                "High_adj": close * 1.02,
                "Low_adj": close * 0.98,
                "Close_adj": close,
                "Capacity": 1000,
            }
        )
    return pd.DataFrame(rows)


def test_regression_log_slope_uses_current_and_past_prices_only() -> None:
    df = make_adjusted_price_df(periods=40)
    slopes = regression_log_slope(df["Close_adj"], 30)

    assert pd.isna(slopes.iloc[28])
    assert math.isclose(slopes.iloc[29], 0.01, rel_tol=1e-10)


def test_flat_price_maps_trend_to_zero() -> None:
    df = make_adjusted_price_df(periods=40)
    df["Close_adj"] = 100
    regime = compute_regime_dataframe(df, trend_window=30, atr_window=10)

    assert abs(regime["trend_slope"].iloc[0]) < 1e-12


def test_atr_percent_uses_adjusted_ohlc_and_previous_adjusted_close() -> None:
    df = pd.DataFrame(
        {
            "High_adj": [11.0, 13.0, 14.0],
            "Low_adj": [9.0, 10.0, 11.0],
            "Close_adj": [10.0, 12.0, 13.0],
        }
    )
    atr_pct = calculate_atr_percent(df, window=2)

    assert math.isclose(atr_pct.iloc[1], (2.0 + 3.0) / 2 / 12.0 * 100)


def test_volatility_minmax_normalization_maps_min_and_max() -> None:
    normalized = minmax_normalize(pd.Series([2.0, 4.0, 6.0]))

    assert normalized.tolist() == [0.0, 0.5, 1.0]


def test_member_trend_axis_range_uses_all_member_records() -> None:
    members = [
        {"regimes": [{"records": [{"x": -0.03}, {"x": 0.01}]}]},
        {"regimes": [{"records": [{"x": 0.08}, {"x": -0.02}]}]},
    ]

    assert get_trend_axis_range_from_members(members) == (-0.03, 0.08)


def test_regime_trend_axis_range_uses_all_horizons() -> None:
    regimes = [
        {"records": [{"x": -0.01}, {"x": 0.02}]},
        {"records": [{"x": -0.04}, {"x": 0.03}]},
    ]

    assert get_trend_axis_range_from_regimes(regimes) == (-0.04, 0.03)


def test_record_trend_axis_range_uses_one_plot_history() -> None:
    records = [{"x": -0.02}, {"x": 0.04}, {"x": 0.01}]

    assert get_trend_axis_range_from_records(records) == (-0.02, 0.04)


def test_metadata_filter_keeps_only_twse_common_stocks(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"Code": "1101", "Name": "A", "Type": COMMON_STOCK_TYPE, "Market": TWSE_MARKET, "Group": "G"},
            {"Code": "0050", "Name": "ETF", "Type": "ETF", "Market": TWSE_MARKET, "Group": "ETF"},
            {"Code": "3105", "Name": "OTC", "Type": COMMON_STOCK_TYPE, "Market": "\u4e0a\u6ac3", "Group": "G"},
        ]
    ).to_csv(metadata_path, index=False)

    filtered = load_listed_common_metadata(metadata_path)

    assert filtered.index.tolist() == ["1101"]


def test_generate_reports_smoke(tmp_path) -> None:
    price_dir = tmp_path / "adj_price"
    output_dir = tmp_path / "regime"
    beta_output_dir = tmp_path / "regime_beta"
    price_dir.mkdir()
    metadata_path = tmp_path / "metadata.csv"
    taiex_path = tmp_path / "TAIEX_202001_to_202401.csv"
    pd.DataFrame(
        [
            {"Code": "1101", "Name": "A", "Type": COMMON_STOCK_TYPE, "Market": TWSE_MARKET, "Group": "G"},
            {"Code": "1102", "Name": "B", "Type": COMMON_STOCK_TYPE, "Market": TWSE_MARKET, "Group": "G"},
        ]
    ).to_csv(metadata_path, index=False)

    make_adjusted_price_df(150).to_csv(price_dir / "1101_202001_to_202401.csv", index=False)
    make_adjusted_price_df(150).to_csv(price_dir / "1102_202001_to_202401.csv", index=False)
    taiex_df = make_adjusted_price_df(150).rename(
        columns={
            "Open_adj": "Open",
            "High_adj": "High",
            "Low_adj": "Low",
            "Close_adj": "Close",
        }
    )
    taiex_df[["Date", "Open", "High", "Low", "Close"]].to_csv(taiex_path, index=False)

    result = generate_reports(
        price_dir=price_dir,
        metadata_path=metadata_path,
        output_dir=output_dir,
        taiex_path=taiex_path,
        beta_output_dir=beta_output_dir,
    )

    assert result["stocks"] == 2
    assert result["groups"] == 1
    assert result["beta_stocks"] == 2
    assert (output_dir / "index.html").exists()
    assert (beta_output_dir / "index.html").exists()
    stock_html = (output_dir / "stocks" / "1101.html").read_text(encoding="utf-8")
    beta_html = (beta_output_dir / "stocks" / "1101.html").read_text(encoding="utf-8")
    assert "Plotly.newPlot" in stock_html
    assert "Short-term" in stock_html
    assert "Medium-term" in stock_html
    assert "Long-term" in stock_html
    assert "const trendAxisMin =" in stock_html
    assert "const trendAxisMax =" in stock_html
    assert "Regime Analysis vs TAIEX" in beta_html
    assert "stock regime minus TAIEX regime" in beta_html
