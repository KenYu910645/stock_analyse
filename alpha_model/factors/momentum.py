"""Momentum factor calculations."""

from __future__ import annotations

import re

import pandas as pd


def _simple_momentum(close: pd.Series, lookback: int, skip: int = 0) -> pd.Series:
    current = close.shift(skip) if skip else close
    past = close.shift(lookback + skip)
    return current / past - 1


def _vol_adjusted_momentum(close: pd.Series, lookback: int) -> pd.Series:
    returns = close.pct_change()
    momentum = _simple_momentum(close, lookback)
    volatility = returns.rolling(lookback, min_periods=lookback).std()
    return momentum / volatility


def compute_momentum_factors(price_df: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    """Compute configured momentum factors without using future prices."""
    df = price_df.sort_values(["stock_id", "date"]).copy()
    grouped_close = df.groupby("stock_id", sort=False)["adj_close"]

    for factor_name in factor_names:
        if factor_name == "vol_adj_momentum_60d":
            df[factor_name] = grouped_close.transform(lambda close: _vol_adjusted_momentum(close, 60))
            continue

        skip_match = re.fullmatch(r"momentum_(\d+)_(\d+)", factor_name)
        if skip_match:
            lookback = int(skip_match.group(1))
            skip = int(skip_match.group(2))
            df[factor_name] = grouped_close.transform(
                lambda close, lookback=lookback, skip=skip: _simple_momentum(close, lookback, skip)
            )
            continue

        simple_match = re.fullmatch(r"momentum_(\d+)d", factor_name)
        if simple_match:
            lookback = int(simple_match.group(1))
            df[factor_name] = grouped_close.transform(
                lambda close, lookback=lookback: _simple_momentum(close, lookback)
            )
            continue

        raise ValueError(f"Unknown momentum factor: {factor_name}")

    return df

