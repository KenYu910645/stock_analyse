"""Rolling z-score pair trading strategy.

The strategy estimates the pair relationship with a rolling lookback window:

    log(price_a) = alpha + beta * log(price_b) + spread

Only past rows are used to estimate alpha, beta, spread mean, and spread std.
The current row is then traded by the current spread z-score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.trade_cost import calculate_trade_cost


class PairTradingStrategy:
    """Market-neutral long/short pair strategy with rolling spread estimates."""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        lookback: int = 252,
        entry_z: float = 2.0,
        exit_z: float = 0.25,
        stop_z: float = 3.5,
        gross_exposure: float = 1.0,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.lookback = int(lookback)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.stop_z = float(stop_z)
        self.gross_exposure = float(gross_exposure)

    def backtest(self, df_pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run the pair strategy on Date, price_a, price_b columns."""
        required = {"Date", "price_a", "price_b"}
        missing = required - set(df_pair.columns)
        if missing:
            raise ValueError(f"Pair data missing columns: {sorted(missing)}")

        df = df_pair.copy().sort_values("Date").reset_index(drop=True)
        df["log_a"] = np.log(pd.to_numeric(df["price_a"], errors="coerce"))
        df["log_b"] = np.log(pd.to_numeric(df["price_b"], errors="coerce"))
        df = df.dropna(subset=["Date", "price_a", "price_b", "log_a", "log_b"]).reset_index(drop=True)

        cash = self.initial_capital
        shares_a = 0.0
        shares_b = 0.0
        position = "flat"
        entry_equity = None
        trades = []
        equity_rows = []

        for idx, row in enumerate(df.itertuples(index=False)):
            price_a = float(row.price_a)
            price_b = float(row.price_b)
            equity_before = self._equity(cash, shares_a, shares_b, price_a, price_b)
            action = "hold"
            alpha = beta = spread = spread_mean = spread_std = zscore = np.nan

            if idx >= self.lookback:
                history = df.iloc[idx - self.lookback:idx]
                alpha, beta, spread_mean, spread_std = self._estimate(history)
                spread = float(row.log_a - alpha - beta * row.log_b)
                zscore = (spread - spread_mean) / spread_std if spread_std > 0 else np.nan

                if position == "flat" and pd.notna(zscore):
                    if zscore >= self.entry_z:
                        cash, shares_a, shares_b, costs = self._open_short_a_long_b(
                            cash,
                            equity_before,
                            price_a,
                            price_b,
                        )
                        position = "short_a_long_b"
                        entry_equity = self._equity(cash, shares_a, shares_b, price_a, price_b)
                        action = "enter_short_a_long_b"
                        trades.append(self._trade_row(row.Date, action, price_a, price_b, shares_a, shares_b, costs, entry_equity, zscore))
                    elif zscore <= -self.entry_z:
                        cash, shares_a, shares_b, costs = self._open_long_a_short_b(
                            cash,
                            equity_before,
                            price_a,
                            price_b,
                        )
                        position = "long_a_short_b"
                        entry_equity = self._equity(cash, shares_a, shares_b, price_a, price_b)
                        action = "enter_long_a_short_b"
                        trades.append(self._trade_row(row.Date, action, price_a, price_b, shares_a, shares_b, costs, entry_equity, zscore))
                elif position != "flat" and pd.notna(zscore):
                    should_exit = abs(zscore) <= self.exit_z or abs(zscore) >= self.stop_z
                    if should_exit:
                        old_position = position
                        cash, costs = self._close(cash, shares_a, shares_b, price_a, price_b)
                        shares_a = 0.0
                        shares_b = 0.0
                        position = "flat"
                        equity_after = self._equity(cash, shares_a, shares_b, price_a, price_b)
                        action = f"exit_{old_position}"
                        trade = self._trade_row(row.Date, action, price_a, price_b, shares_a, shares_b, costs, equity_after, zscore)
                        trade["round_trip_pnl"] = equity_after - entry_equity if entry_equity is not None else np.nan
                        trades.append(trade)
                        entry_equity = None

            equity = self._equity(cash, shares_a, shares_b, price_a, price_b)
            equity_rows.append(
                {
                    "Date": row.Date,
                    "price_a": price_a,
                    "price_b": price_b,
                    "cash": cash,
                    "shares_a": shares_a,
                    "shares_b": shares_b,
                    "position": position,
                    "action": action,
                    "equity": equity,
                    "alpha": alpha,
                    "beta": beta,
                    "spread": spread,
                    "spread_mean": spread_mean,
                    "spread_std": spread_std,
                    "zscore": zscore,
                }
            )

        if position != "flat" and not df.empty:
            row = df.iloc[-1]
            price_a = float(row.price_a)
            price_b = float(row.price_b)
            cash, costs = self._close(cash, shares_a, shares_b, price_a, price_b)
            equity = self._equity(cash, 0.0, 0.0, price_a, price_b)
            trades.append(self._trade_row(row["Date"], "forced_exit", price_a, price_b, 0.0, 0.0, costs, equity, np.nan))
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["shares_a"] = 0.0
            equity_rows[-1]["shares_b"] = 0.0
            equity_rows[-1]["position"] = "flat"
            equity_rows[-1]["action"] = "forced_exit"
            equity_rows[-1]["equity"] = equity

        return pd.DataFrame(equity_rows), pd.DataFrame(trades)

    def _estimate(self, history: pd.DataFrame) -> tuple[float, float, float, float]:
        x = history["log_b"].to_numpy(dtype=float)
        y = history["log_a"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        alpha, beta = np.linalg.lstsq(design, y, rcond=None)[0]
        spread = y - (alpha + beta * x)
        return float(alpha), float(beta), float(spread.mean()), float(spread.std(ddof=0))

    def _open_short_a_long_b(self, cash, equity, price_a, price_b):
        leg_value = equity * self.gross_exposure / 2
        shares_a = -leg_value / price_a
        shares_b = leg_value / price_b
        costs = self._combined_cost(
            [
                ("sell", abs(shares_a) * price_a),
                ("buy", abs(shares_b) * price_b),
            ]
        )
        cash = cash + abs(shares_a) * price_a - abs(shares_b) * price_b - costs["transaction_cost"]
        return cash, shares_a, shares_b, costs

    def _open_long_a_short_b(self, cash, equity, price_a, price_b):
        leg_value = equity * self.gross_exposure / 2
        shares_a = leg_value / price_a
        shares_b = -leg_value / price_b
        costs = self._combined_cost(
            [
                ("buy", abs(shares_a) * price_a),
                ("sell", abs(shares_b) * price_b),
            ]
        )
        cash = cash - abs(shares_a) * price_a + abs(shares_b) * price_b - costs["transaction_cost"]
        return cash, shares_a, shares_b, costs

    def _close(self, cash, shares_a, shares_b, price_a, price_b):
        legs = []
        if shares_a > 0:
            legs.append(("sell", abs(shares_a) * price_a))
            cash += abs(shares_a) * price_a
        elif shares_a < 0:
            legs.append(("buy", abs(shares_a) * price_a))
            cash -= abs(shares_a) * price_a

        if shares_b > 0:
            legs.append(("sell", abs(shares_b) * price_b))
            cash += abs(shares_b) * price_b
        elif shares_b < 0:
            legs.append(("buy", abs(shares_b) * price_b))
            cash -= abs(shares_b) * price_b

        costs = self._combined_cost(legs)
        cash -= costs["transaction_cost"]
        return cash, costs

    def _combined_cost(self, legs):
        total = {"fee": 0.0, "tax": 0.0, "transaction_cost": 0.0}
        for action, value in legs:
            costs = calculate_trade_cost(action, value)
            for key in total:
                total[key] += float(costs[key])
        return total

    def _equity(self, cash, shares_a, shares_b, price_a, price_b):
        return cash + shares_a * price_a + shares_b * price_b

    def _trade_row(self, date, action, price_a, price_b, shares_a, shares_b, costs, equity, zscore):
        return {
            "Date": date,
            "action": action,
            "price_a": price_a,
            "price_b": price_b,
            "shares_a": shares_a,
            "shares_b": shares_b,
            "fee": costs["fee"],
            "tax": costs["tax"],
            "transaction_cost": costs["transaction_cost"],
            "equity": equity,
            "zscore": zscore,
        }
