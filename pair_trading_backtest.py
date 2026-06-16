"""Backtest pair trading shortlist against TAIEX buy-and-hold."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stock_correlation_analysis import DEFAULT_PRICE_PATH, DEFAULT_TAIEX_PATH
from strategies.pair_trading import PairTradingStrategy


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SHORTLIST_PATH = PROJECT_ROOT / "output" / "pair_trading" / "cointegration_gt_0_5" / "manual_pair_trading_shortlist_15.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "pair_trading" / "backtest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest pair trading shortlist.")
    parser.add_argument("--shortlist", default=str(DEFAULT_SHORTLIST_PATH))
    parser.add_argument("--price-dir", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--taiex", default=str(DEFAULT_TAIEX_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--initial-capital", type=float, default=1_000_000)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.25)
    parser.add_argument("--stop-z", type=float, default=3.5)
    return parser.parse_args()


def latest_price_path(price_dir: Path, stock_id: str) -> Path:
    paths = sorted(price_dir.glob(f"{stock_id}_*_to_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No adjusted price CSV found for {stock_id}.")
    return paths[-1]


def load_adjusted_close(price_dir: Path, stock_id: str) -> pd.DataFrame:
    path = latest_price_path(price_dir, stock_id)
    df = pd.read_csv(path)
    close_col = "Close_adj" if "Close_adj" in df.columns else "Close"
    df = df[["Date", close_col]].copy()
    df.columns = ["Date", stock_id]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[stock_id] = pd.to_numeric(df[stock_id], errors="coerce")
    return df.dropna().sort_values("Date")


def build_pair_price_df(price_dir: Path, stock_a: str, stock_b: str, start_date: pd.Timestamp) -> pd.DataFrame:
    a = load_adjusted_close(price_dir, stock_a)
    b = load_adjusted_close(price_dir, stock_b)
    df = a.merge(b, on="Date", how="inner")
    df = df[df["Date"] >= start_date].copy()
    df = df.rename(columns={stock_a: "price_a", stock_b: "price_b"})
    return df.dropna().sort_values("Date").reset_index(drop=True)


def load_taiex(taiex_path: Path, start_date: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(taiex_path)
    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return (
        df.dropna()
        .sort_values("Date")
        .loc[lambda data: data["Date"] >= start_date]
        .reset_index(drop=True)
    )


def calculate_metrics(equity_df: pd.DataFrame, initial_capital: float) -> dict:
    if equity_df.empty:
        return {}

    equity = equity_df["equity"].astype(float)
    daily_returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / initial_capital - 1
    years = max((equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).days / 365.25, 1 / 365.25)
    annual_return = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
    annual_volatility = daily_returns.std(ddof=0) * math.sqrt(252) if not daily_returns.empty else np.nan
    sharpe = annual_return / annual_volatility if annual_volatility and pd.notna(annual_volatility) else np.nan
    running_peak = equity.cummax()
    max_drawdown = (equity / running_peak - 1).min()
    return {
        "final_equity": equity.iloc[-1],
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def build_equal_weight_portfolio(equity_frames: dict[str, pd.DataFrame], initial_capital: float) -> tuple[pd.DataFrame, dict]:
    """Combine pair equity curves as an equal-weight portfolio on common dates."""
    normalized = []
    for pair_name, equity_df in equity_frames.items():
        if equity_df.empty:
            continue
        series = equity_df[["Date", "equity"]].copy()
        series = series.rename(columns={"equity": pair_name})
        series[pair_name] = series[pair_name].astype(float) / initial_capital
        normalized.append(series)

    if not normalized:
        return pd.DataFrame(columns=["Date", "equity"]), {}

    merged = normalized[0]
    for series in normalized[1:]:
        merged = merged.merge(series, on="Date", how="inner")

    pair_cols = [col for col in merged.columns if col != "Date"]
    merged["equity"] = merged[pair_cols].mean(axis=1) * initial_capital
    portfolio = merged[["Date", "equity", *pair_cols]].copy()
    metrics = calculate_metrics(portfolio[["Date", "equity"]], initial_capital)
    return portfolio, metrics


def buy_and_hold_taiex(taiex_df: pd.DataFrame, initial_capital: float) -> tuple[pd.DataFrame, dict]:
    df = taiex_df.copy()
    first_close = float(df["Close"].iloc[0])
    shares = initial_capital / first_close
    df["equity"] = shares * df["Close"]
    return df[["Date", "Close", "equity"]], calculate_metrics(df[["Date", "equity"]], initial_capital)


def summarize_trades(trade_df: pd.DataFrame) -> dict:
    if trade_df.empty:
        return {
            "trade_count": 0,
            "round_trips": 0,
            "win_rate": np.nan,
            "total_transaction_cost": 0.0,
        }
    exits = trade_df[trade_df["action"].str.startswith("exit") | trade_df["action"].eq("forced_exit")]
    pnl = pd.to_numeric(exits.get("round_trip_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "trade_count": len(trade_df),
        "round_trips": len(exits),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "total_transaction_cost": float(trade_df["transaction_cost"].sum()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equity_dir = output_dir / "equity_curves"
    trade_dir = output_dir / "trades"
    equity_dir.mkdir(exist_ok=True)
    trade_dir.mkdir(exist_ok=True)

    shortlist = pd.read_csv(args.shortlist, dtype={"stock_id_1": str, "stock_id_2": str})
    price_dir = Path(args.price_dir)
    start_date = pd.Timestamp.today().normalize() - pd.DateOffset(years=args.years)

    taiex_df = load_taiex(Path(args.taiex), start_date)
    taiex_equity, taiex_metrics = buy_and_hold_taiex(taiex_df, args.initial_capital)
    taiex_equity.to_csv(output_dir / "taiex_buy_and_hold_equity.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    equity_frames = {}
    for pair in shortlist.itertuples(index=False):
        stock_a = str(pair.stock_id_1)
        stock_b = str(pair.stock_id_2)
        name = f"{stock_a}_{stock_b}"
        df_pair = build_pair_price_df(price_dir, stock_a, stock_b, start_date)
        strategy = PairTradingStrategy(
            initial_capital=args.initial_capital,
            lookback=args.lookback,
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
        )
        equity_df, trade_df = strategy.backtest(df_pair)
        metrics = calculate_metrics(equity_df, args.initial_capital)
        trade_summary = summarize_trades(trade_df)

        equity_df.to_csv(equity_dir / f"{name}_equity.csv", index=False, encoding="utf-8-sig")
        trade_df.to_csv(trade_dir / f"{name}_trades.csv", index=False, encoding="utf-8-sig")
        equity_frames[name] = equity_df

        row = {
            "stock_id_1": stock_a,
            "stock_name_1": pair.stock_name_1,
            "industry_1": pair.industry_1,
            "stock_id_2": stock_b,
            "stock_name_2": pair.stock_name_2,
            "industry_2": pair.industry_2,
            "raw_correlation": pair.raw_correlation,
            "adf_pvalue_approx": pair.adf_pvalue_approx,
            "half_life_days": pair.half_life_days,
            "hurst_exponent": pair.hurst_exponent,
            "start_date": equity_df["Date"].min() if not equity_df.empty else pd.NaT,
            "end_date": equity_df["Date"].max() if not equity_df.empty else pd.NaT,
            **metrics,
            **trade_summary,
            "taiex_total_return": taiex_metrics["total_return"],
            "taiex_annual_return": taiex_metrics["annual_return"],
            "taiex_max_drawdown": taiex_metrics["max_drawdown"],
            "excess_total_return_vs_taiex": metrics.get("total_return", np.nan) - taiex_metrics["total_return"],
            "excess_annual_return_vs_taiex": metrics.get("annual_return", np.nan) - taiex_metrics["annual_return"],
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["annual_return", "sharpe"], ascending=False)
    summary.to_csv(output_dir / "pair_trading_backtest_summary.csv", index=False, encoding="utf-8-sig")

    portfolio_equity, portfolio_metrics = build_equal_weight_portfolio(equity_frames, args.initial_capital)
    portfolio_equity.to_csv(output_dir / "equal_weight_15_pair_portfolio_equity.csv", index=False, encoding="utf-8-sig")
    portfolio_vs_taiex = pd.DataFrame(
        [
            {
                "strategy": "15_pair_equal_weight_portfolio",
                **portfolio_metrics,
                "taiex_total_return": taiex_metrics["total_return"],
                "taiex_annual_return": taiex_metrics["annual_return"],
                "taiex_max_drawdown": taiex_metrics["max_drawdown"],
                "excess_total_return_vs_taiex": portfolio_metrics.get("total_return", np.nan) - taiex_metrics["total_return"],
                "excess_annual_return_vs_taiex": portfolio_metrics.get("annual_return", np.nan) - taiex_metrics["annual_return"],
            },
            {
                "strategy": "taiex_buy_and_hold",
                **taiex_metrics,
                "taiex_total_return": taiex_metrics["total_return"],
                "taiex_annual_return": taiex_metrics["annual_return"],
                "taiex_max_drawdown": taiex_metrics["max_drawdown"],
                "excess_total_return_vs_taiex": 0.0,
                "excess_annual_return_vs_taiex": 0.0,
            },
        ]
    )
    portfolio_vs_taiex.to_csv(output_dir / "portfolio_vs_taiex_summary.csv", index=False, encoding="utf-8-sig")

    report = build_report(summary, taiex_metrics, portfolio_metrics, args)
    (output_dir / "pair_trading_backtest_report.md").write_text(report, encoding="utf-8-sig")

    print(f"Pair trading backtest outputs written to {output_dir}.")
    print(f"Pairs tested: {len(summary)}")
    print(f"TAIEX total return: {taiex_metrics['total_return']:.2%}")
    print(f"Equal-weight 15-pair portfolio total return: {portfolio_metrics.get('total_return', np.nan):.2%}")
    print(summary[["stock_name_1", "stock_name_2", "total_return", "annual_return", "max_drawdown", "round_trips"]].to_string(index=False))


def build_report(summary: pd.DataFrame, taiex_metrics: dict, portfolio_metrics: dict, args: argparse.Namespace) -> str:
    show_cols = [
        "stock_name_1",
        "stock_name_2",
        "total_return",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "round_trips",
        "win_rate",
        "excess_total_return_vs_taiex",
    ]
    table = summary[show_cols].copy()
    table.columns = [
        "股票A",
        "股票B",
        "總報酬",
        "年化報酬",
        "Sharpe",
        "最大回撤",
        "來回次數",
        "勝率",
        "相對TAIEX總報酬差",
    ]
    for col in ["總報酬", "年化報酬", "Sharpe", "最大回撤", "勝率", "相對TAIEX總報酬差"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(4)

    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    return f"""# Pair Trading 回測報告

資料來源：`manual_pair_trading_shortlist_15.csv`

回測設定：
- 期間：最近 `{args.years}` 年可取得資料。
- 模型：rolling OLS，使用過去 `{args.lookback}` 個交易日估計 `log(A) = alpha + beta * log(B) + spread`。
- 進場：`zscore >= {args.entry_z}` 時 short A / long B；`zscore <= -{args.entry_z}` 時 long A / short B。
- 出場：`|zscore| <= {args.exit_z}` 回到均值附近，或 `|zscore| >= {args.stop_z}` 觸發停損。
- 部位：每組 pair 使用 100% gross exposure，兩邊各約 50%。
- 交易成本：使用 `strategies.trade_cost` 的手續費與交易稅模型。

## 大盤 Buy & Hold

- TAIEX 總報酬：`{taiex_metrics['total_return']:.2%}`
- TAIEX 年化報酬：`{taiex_metrics['annual_return']:.2%}`
- TAIEX 最大回撤：`{taiex_metrics['max_drawdown']:.2%}`

## 15 組 Pair 等權組合

- 等權組合總報酬：`{portfolio_metrics.get('total_return', np.nan):.2%}`
- 等權組合年化報酬：`{portfolio_metrics.get('annual_return', np.nan):.2%}`
- 等權組合 Sharpe：`{portfolio_metrics.get('sharpe', np.nan):.2f}`
- 等權組合最大回撤：`{portfolio_metrics.get('max_drawdown', np.nan):.2%}`
- 相對 TAIEX 總報酬差：`{portfolio_metrics.get('total_return', np.nan) - taiex_metrics['total_return']:.2%}`
- 相對 TAIEX 年化報酬差：`{portfolio_metrics.get('annual_return', np.nan) - taiex_metrics['annual_return']:.2%}`

## 單組 Pair 回測結果

{chr(10).join(lines)}

## 解讀

這版結果偏向檢查 pair trading 訊號本身，而不是最佳化策略。若一組 pair 有 cointegration，但回測報酬不好，常見原因是交易成本吃掉均值回歸利潤、spread 回歸太慢、或進出場門檻不適合該產業。這份回測允許放空與零股/小數股，尚未加入融券限制、借券成本、滑價、流動性限制與停牌風險。
"""


if __name__ == "__main__":
    main()
