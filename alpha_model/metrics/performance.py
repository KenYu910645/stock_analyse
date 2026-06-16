"""Portfolio performance helpers."""

from __future__ import annotations

import math

import pandas as pd


def max_drawdown(return_series: pd.Series) -> float:
    """Calculate max drawdown from a periodic return series."""
    cumulative = (1 + return_series.fillna(0)).cumprod()
    drawdown = cumulative / cumulative.cummax() - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def summarize_returns(return_series: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Return annualized return, volatility, Sharpe, drawdown, and win rate."""
    returns = return_series.dropna()
    if returns.empty:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }

    cumulative = float((1 + returns).prod())
    annualized_return = cumulative ** (periods_per_year / len(returns)) - 1
    annualized_volatility = float(returns.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = annualized_return / annualized_volatility if annualized_volatility else 0.0
    return {
        "annualized_return": float(annualized_return),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_drawdown(returns),
        "win_rate": float((returns > 0).mean()),
    }

