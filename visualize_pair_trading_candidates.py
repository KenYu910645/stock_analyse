"""Visualize pair-trading candidates with separate calculation and display windows."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from column_schema import read_csv_canonical
from stock_correlation_analysis import DEFAULT_PRICE_PATH
from pair_trading_backtest import calculate_metrics, summarize_trades
from strategies.pair_trading import PairTradingStrategy


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CANDIDATES_PATH = PROJECT_ROOT / "output" / "pair_trading" / "cointegration_gt_0_5" / "pair_trading_candidates.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_viz" / "pair_trading" / "candidate_visuals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visuals for pair-trading candidates.")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--price-dir", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--chart-subdir", default="pair_charts_1y")
    parser.add_argument("--calc-years", type=int, default=5)
    parser.add_argument("--display-years", type=int, default=1)
    parser.add_argument("--rolling-corr-window", type=int, default=60)
    parser.add_argument("--initial-capital", type=float, default=1_000_000)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.25)
    parser.add_argument("--stop-z", type=float, default=3.5)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def configure_plot_fonts() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "Noto Sans CJK TC",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def latest_price_path(price_dir: Path, stock_id: str) -> Path:
    paths = sorted(
        path for path in price_dir.glob(f"{stock_id}_*.csv")
        if not path.name.startswith("twse_price_")
    )
    if not paths:
        raise FileNotFoundError(f"No adjusted price CSV found for {stock_id}.")
    return paths[-1]


def load_adjusted_close(price_dir: Path, stock_id: str) -> pd.DataFrame:
    path = latest_price_path(price_dir, stock_id)
    df = read_csv_canonical(path)
    close_col = "close_adj" if "close_adj" in df.columns else "Close_adj" if "Close_adj" in df.columns else "Close"
    df = df[["Date", close_col]].copy()
    df.columns = ["Date", stock_id]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[stock_id] = pd.to_numeric(df[stock_id], errors="coerce")
    return df.dropna().sort_values("Date")


def build_pair_data(
    price_dir: Path,
    row: pd.Series,
    calc_start: pd.Timestamp,
    display_start: pd.Timestamp,
    rolling_corr_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stock_a = str(row["stock_id_1"])
    stock_b = str(row["stock_id_2"])
    price_a = load_adjusted_close(price_dir, stock_a)
    price_b = load_adjusted_close(price_dir, stock_b)
    df = price_a.merge(price_b, on="Date", how="inner")
    df = df[df["Date"] >= calc_start].copy()
    df = df.rename(columns={stock_a: "price_a", stock_b: "price_b"})
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    alpha = float(row["eg_alpha"])
    beta = float(row["eg_beta"])
    df["log_a"] = np.log(df["price_a"].where(df["price_a"] > 0))
    df["log_b"] = np.log(df["price_b"].where(df["price_b"] > 0))
    df["spread"] = df["log_a"] - alpha - beta * df["log_b"]

    spread_mean_5y = float(df["spread"].mean())
    spread_std_5y = float(df["spread"].std(ddof=0))
    df["zscore"] = (df["spread"] - spread_mean_5y) / spread_std_5y if spread_std_5y > 0 else np.nan
    df["equilibrium_price_a"] = np.exp(alpha + beta * df["log_b"] + spread_mean_5y)
    df["return_a"] = df["price_a"].pct_change()
    df["return_b"] = df["price_b"].pct_change()
    df["rolling_corr"] = df["return_a"].rolling(rolling_corr_window).corr(df["return_b"])

    display = df[df["Date"] >= display_start].copy().reset_index(drop=True)
    if display.empty:
        display = df.copy()
    display_base_a = float(display["price_a"].iloc[0])
    display_base_b = float(display["price_b"].iloc[0])
    display["norm_a"] = display["price_a"] / display_base_a * 100
    display["norm_b"] = display["price_b"] / display_base_b * 100
    display["norm_equilibrium_a"] = display["equilibrium_price_a"] / display_base_a * 100

    calc_stats = {
        "calc_start_date": df["Date"].iloc[0].date().isoformat(),
        "calc_end_date": df["Date"].iloc[-1].date().isoformat(),
        "display_start_date": display["Date"].iloc[0].date().isoformat(),
        "display_end_date": display["Date"].iloc[-1].date().isoformat(),
        "spread_mean_5y": spread_mean_5y,
        "spread_std_5y": spread_std_5y,
    }
    return df, display, calc_stats


def run_backtest_overlay(
    calc_df: pd.DataFrame,
    display_start: pd.Timestamp,
    initial_capital: float,
    lookback: int,
    entry_z: float,
    exit_z: float,
    stop_z: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    strategy = PairTradingStrategy(
        initial_capital=initial_capital,
        lookback=lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
    )
    equity_df, trade_df = strategy.backtest(calc_df[["Date", "price_a", "price_b"]])
    metrics = calculate_metrics(equity_df, initial_capital)
    metrics.update(summarize_trades(trade_df))
    equity_display = equity_df[equity_df["Date"] >= display_start].copy()
    trades_display = trade_df[trade_df["Date"] >= display_start].copy() if not trade_df.empty else trade_df.copy()
    return equity_display, trades_display, metrics


def find_extreme_windows(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    zones = []
    active_start = None
    active_side = None
    for date, zscore in zip(df["Date"], df["zscore"]):
        side = None
        if pd.notna(zscore) and zscore >= 2:
            side = "high"
        elif pd.notna(zscore) and zscore <= -2:
            side = "low"

        if side and active_start is None:
            active_start = date
            active_side = side
        elif side != active_side and active_start is not None:
            zones.append((active_start, date, active_side))
            active_start = date if side else None
            active_side = side

    if active_start is not None:
        zones.append((active_start, df["Date"].iloc[-1], active_side))
    return zones


def find_zero_crossings(df: pd.DataFrame) -> pd.DataFrame:
    z = df["zscore"].copy()
    signs = np.sign(z)
    crossing = (signs.shift(1) * signs < 0) & signs.notna() & signs.shift(1).notna()
    return df.loc[crossing, ["Date", "zscore"]]


def find_position_windows(equity_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    if equity_df.empty or "position" not in equity_df:
        return []
    windows = []
    active_start = None
    active_position = None
    for row in equity_df.itertuples(index=False):
        position = getattr(row, "position")
        date = getattr(row, "Date")
        if position != "flat" and active_start is None:
            active_start = date
            active_position = position
        elif active_start is not None and position != active_position:
            windows.append((active_start, date, active_position))
            active_start = date if position != "flat" else None
            active_position = position if position != "flat" else None
    if active_start is not None:
        windows.append((active_start, equity_df["Date"].iloc[-1], active_position))
    return windows


def plot_backtest_overlay(axes, trades_df: pd.DataFrame, equity_df: pd.DataFrame, stock_a: str, stock_b: str) -> None:
    position_colors = {
        "long_a_short_b": "#d9f0d3",
        "short_a_long_b": "#fee0d2",
    }
    position_labels = {
        "long_a_short_b": f"持倉: 買 {stock_a} / 空 {stock_b}",
        "short_a_long_b": f"持倉: 空 {stock_a} / 買 {stock_b}",
    }
    used_position_labels = set()
    for start, end, position in find_position_windows(equity_df):
        color = position_colors.get(position, "#f0f0f0")
        label = position_labels.get(position, "持倉")
        for ax in axes[:3]:
            ax.axvspan(
                start,
                end,
                color=color,
                alpha=0.28,
                linewidth=0,
                label=label if ax is axes[0] and label not in used_position_labels else None,
            )
        used_position_labels.add(label)

    if trades_df.empty:
        return

    marker_styles = {
        "enter_long_a_short_b": ("^", "#238b45", f"建倉 買 {stock_a} / 空 {stock_b}"),
        "enter_short_a_long_b": ("v", "#cb181d", f"建倉 空 {stock_a} / 買 {stock_b}"),
        "exit_long_a_short_b": ("o", "#08519c", "平倉"),
        "exit_short_a_long_b": ("o", "#08519c", "平倉"),
        "forced_exit": ("X", "#54278f", "強制平倉"),
    }
    used_trade_labels = set()
    for trade in trades_df.itertuples(index=False):
        action = getattr(trade, "action")
        if action not in marker_styles:
            continue
        marker, color, label = marker_styles[action]
        date = getattr(trade, "Date")
        zscore = getattr(trade, "zscore", np.nan)
        for ax in axes[:2]:
            ax.axvline(date, color=color, linestyle=":", linewidth=1.0, alpha=0.65)
        if pd.notna(zscore):
            axes[2].scatter(
                [date],
                [zscore],
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                s=72,
                zorder=5,
                label=label if label not in used_trade_labels else None,
            )
            used_trade_labels.add(label)


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def add_backtest_metrics_box(ax, metrics: dict) -> None:
    text = "\n".join(
        [
            "Backtest",
            f"總報酬：{format_pct(metrics.get('total_return', np.nan))}",
            f"年化報酬：{format_pct(metrics.get('annual_return', np.nan))}",
            f"最大回撤：{format_pct(metrics.get('max_drawdown', np.nan))}",
            f"來回交易：{int(metrics.get('round_trips', 0))} 次",
            f"勝率：{format_pct(metrics.get('win_rate', np.nan))}",
        ]
    )
    ax.text(
        0.985,
        0.965,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#bdbdbd",
            "alpha": 0.88,
        },
    )


def plot_pair(
    row: pd.Series,
    df: pd.DataFrame,
    output_path: Path,
    rolling_corr_window: int,
    calc_stats: dict,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    backtest_metrics: dict,
) -> dict:
    stock_a = f"{row['stock_name_1']}({row['stock_id_1']})"
    stock_b = f"{row['stock_name_2']}({row['stock_id_2']})"
    stock_a_name = str(row["stock_name_1"])
    stock_b_name = str(row["stock_name_2"])
    title = f"{stock_a} / {stock_b}"
    zones = find_extreme_windows(df)
    crossings = find_zero_crossings(df)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.35, 1.35, 1.0]},
        constrained_layout=True,
    )

    axes[0].plot(df["Date"], df["norm_a"], label=stock_a, linewidth=1.7, color="#1f77b4")
    axes[0].plot(df["Date"], df["norm_b"], label=stock_b, linewidth=1.7, color="#ff7f0e")
    axes[0].plot(
        df["Date"],
        df["norm_equilibrium_a"],
        label=f"{stock_a} 均值回歸線",
        linewidth=1.5,
        linestyle="--",
        color="#2ca25f",
    )
    axes[0].set_title(
        f"{title} | corr={row['raw_correlation']:.3f}, ADF p={row['adf_pvalue_approx']:.3g}, "
        f"half-life={row['half_life_days']:.1f}d, Hurst={row['hurst_exponent']:.3f}"
    )
    axes[0].set_ylabel("標準化股價\n顯示起點=100")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.25)
    add_backtest_metrics_box(axes[0], backtest_metrics)

    spread_mean = calc_stats["spread_mean_5y"]
    spread_std = calc_stats["spread_std_5y"]
    axes[1].plot(df["Date"], df["spread"], color="#4c4c4c", linewidth=1.3)
    axes[1].axhline(spread_mean, color="#222222", linewidth=1.0, label="5年 spread mean")
    for multiple, color, label in [(1, "#9ecae1", "±1σ"), (2, "#fdae6b", "±2σ")]:
        axes[1].axhline(spread_mean + multiple * spread_std, color=color, linestyle="--", linewidth=1.0, label=label)
        axes[1].axhline(spread_mean - multiple * spread_std, color=color, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Spread")
    axes[1].legend(loc="upper left", ncols=3)
    axes[1].grid(alpha=0.25)

    axes[2].plot(df["Date"], df["zscore"], color="#6a3d9a", linewidth=1.2)
    if not equity_df.empty and "zscore" in equity_df:
        strategy_z = equity_df[["Date", "zscore"]].dropna()
        axes[2].plot(
            strategy_z["Date"],
            strategy_z["zscore"],
            color="#08519c",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
            label="策略rolling z-score",
        )
    axes[2].axhline(0, color="#222222", linewidth=1.0)
    axes[2].axhline(2, color="#d95f0e", linestyle="--", linewidth=1.0)
    axes[2].axhline(-2, color="#d95f0e", linestyle="--", linewidth=1.0)
    axes[2].axhline(0.25, color="#31a354", linestyle=":", linewidth=1.0)
    axes[2].axhline(-0.25, color="#31a354", linestyle=":", linewidth=1.0)
    axes[2].set_ylabel("Z-score")
    axes[2].grid(alpha=0.25)

    for start, end, side in zones:
        color = "#fee8c8" if side == "high" else "#deebf7"
        for ax in axes[:3]:
            ax.axvspan(start, end, color=color, alpha=0.45, linewidth=0)

    plot_backtest_overlay(axes, trades_df, equity_df, stock_a_name, stock_b_name)
    axes[0].legend(loc="upper left")
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(loc="upper left")

    axes[3].plot(df["Date"], df["rolling_corr"], color="#08519c", linewidth=1.2)
    axes[3].axhline(float(row["raw_correlation"]), color="#636363", linestyle="--", linewidth=1.0, label="全期 raw corr")
    axes[3].set_ylabel(f"{rolling_corr_window}日\nrolling corr")
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].legend(loc="upper left")
    axes[3].grid(alpha=0.25)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in axes[-1].get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    latest = df.iloc[-1]
    return {
        "plot_path": str(output_path),
        **calc_stats,
        "latest_price_a": float(latest["price_a"]),
        "latest_price_b": float(latest["price_b"]),
        "latest_equilibrium_price_a": float(latest["equilibrium_price_a"]),
        "latest_price_a_vs_equilibrium_pct": float(latest["price_a"] / latest["equilibrium_price_a"] - 1),
        "latest_zscore_5y": float(latest["zscore"]),
        "extreme_z_days_display": int((df["zscore"].abs() >= 2).sum()),
        "zero_crossings_display": int(len(crossings)),
        "backtest_trade_count_display": int(len(trades_df)),
        "backtest_round_trips_display": int(
            trades_df["action"].str.startswith("exit").sum() + trades_df["action"].eq("forced_exit").sum()
        )
        if not trades_df.empty
        else 0,
        "backtest_total_return": float(backtest_metrics.get("total_return", np.nan)),
        "backtest_annual_return": float(backtest_metrics.get("annual_return", np.nan)),
        "backtest_max_drawdown": float(backtest_metrics.get("max_drawdown", np.nan)),
        "backtest_round_trips": int(backtest_metrics.get("round_trips", 0)),
        "backtest_win_rate": float(backtest_metrics.get("win_rate", np.nan)),
    }


def plot_overview(index_df: pd.DataFrame, output_dir: Path) -> None:
    if index_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    scatter = axes[0, 0].scatter(
        index_df["half_life_days"],
        index_df["hit_rate_z2_cross_mean_20d"],
        c=index_df["raw_correlation"],
        s=28,
        cmap="viridis",
        alpha=0.8,
    )
    axes[0, 0].set_xlabel("Half-life days")
    axes[0, 0].set_ylabel("20日回歸均值命中率")
    axes[0, 0].set_title("回歸速度 vs 命中率")
    axes[0, 0].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[0, 0], label="raw correlation")

    axes[0, 1].scatter(index_df["hurst_exponent"], index_df["half_life_days"], s=28, color="#756bb1", alpha=0.75)
    axes[0, 1].axvline(0.5, color="#d95f0e", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Hurst exponent")
    axes[0, 1].set_ylabel("Half-life days")
    axes[0, 1].set_title("Hurst < 0.5 越偏均值回歸")
    axes[0, 1].grid(alpha=0.25)

    top = index_df.sort_values("latest_zscore_5y", key=lambda s: s.abs(), ascending=False).head(25)
    labels = top["stock_name_1"] + "/" + top["stock_name_2"]
    axes[1, 0].barh(labels[::-1], top["latest_zscore_5y"][::-1], color="#3182bd")
    axes[1, 0].axvline(2, color="#d95f0e", linestyle="--", linewidth=1)
    axes[1, 0].axvline(-2, color="#d95f0e", linestyle="--", linewidth=1)
    axes[1, 0].set_title("目前偏離最大的候選 pair")
    axes[1, 0].set_xlabel("latest z-score, 5年mean/std")

    industry_counts = (
        index_df.assign(pair_industry=index_df["industry_1"] + " / " + index_df["industry_2"])
        ["pair_industry"]
        .value_counts()
        .head(20)
    )
    axes[1, 1].barh(industry_counts.index[::-1], industry_counts.values[::-1], color="#31a354")
    axes[1, 1].set_title("候選 pair 產業分布 Top 20")
    axes[1, 1].set_xlabel("pair count")

    fig.savefig(output_dir / "candidate_overview_metrics_1y_5ycalc.png", dpi=150)
    plt.close(fig)


def build_report(index_df: pd.DataFrame, output_dir: Path, chart_subdir: str, calc_years: int, display_years: int) -> None:
    top = index_df.sort_values(
        ["adf_pvalue_approx", "half_life_days", "hit_rate_z2_cross_mean_20d"],
        ascending=[True, True, False],
    ).head(30)
    show_cols = [
        "stock_name_1",
        "stock_name_2",
        "industry_1",
        "industry_2",
        "raw_correlation",
        "adf_pvalue_approx",
        "half_life_days",
        "hurst_exponent",
        "hit_rate_z2_cross_mean_20d",
        "latest_zscore_5y",
        "plot_path",
    ]
    table = top[show_cols].copy()
    for col in [
        "raw_correlation",
        "adf_pvalue_approx",
        "half_life_days",
        "hurst_exponent",
        "hit_rate_z2_cross_mean_20d",
        "latest_zscore_5y",
    ]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(4)
    table["plot_path"] = table["plot_path"].map(lambda path: Path(path).name)
    table.columns = [
        "股票A",
        "股票B",
        "產業A",
        "產業B",
        "Raw corr",
        "ADF p",
        "Half-life",
        "Hurst",
        "20日命中率",
        "最新Z(5年)",
        "圖檔",
    ]

    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    report = f"""# Pair Trading Candidate 視覺化報告

這份版本使用近 `{calc_years}` 年資料計算 spread mean、spread std、z-score、rolling correlation，圖上只顯示近 `{display_years}` 年。

每一張 pair 圖包含四個區塊：
- 標準化股價：兩檔股票都從顯示期間起點 100 開始；綠色虛線是由 B 推算出的 A 均值回歸價格線。
- Spread：使用 `log(A) - alpha - beta * log(B)`，並標出近 `{calc_years}` 年平均、正負 1/2 個標準差。
- Z-score：紫色線使用近 `{calc_years}` 年 mean/std，深藍虛線是 backtest 實際使用的 rolling z-score；建倉/平倉點是依照深藍虛線觸發。
- Rolling correlation：在近 `{calc_years}` 年資料上計算，再只顯示最近 `{display_years}` 年。
- Backtest overlay：淡綠背景代表策略持有「買第一檔 / 空第二檔」，淡紅背景代表策略持有「空第一檔 / 買第二檔」；三角形是建倉，圓點是平倉，X 是最後強制平倉。

綠色虛線的公式：

```text
A_equilibrium = exp(alpha + beta * log(B) + spread_mean_5y)
```

## 輸出檔

- 每組 pair 圖：`{chart_subdir}/*.png`
- 索引表：`candidate_visual_index_1y_5ycalc.csv`
- 總覽圖：`candidate_overview_metrics_1y_5ycalc.png`

## 排名前 30 的候選 pair

{chr(10).join(lines)}
"""
    (output_dir / "candidate_visual_report_1y_5ycalc.md").write_text(report, encoding="utf-8-sig")


def safe_chart_filename(rank: int, row_data: pd.Series) -> str:
    filename = f"{rank:03d}_{row_data['stock_id_1']}_{row_data['stock_id_2']}_{row_data['stock_name_1']}_{row_data['stock_name_2']}.png"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in filename)


def main() -> None:
    args = parse_args()
    configure_plot_fonts()

    candidates = pd.read_csv(args.candidates, dtype={"stock_id_1": str, "stock_id_2": str})
    candidates = candidates.sort_values(
        ["adf_pvalue_approx", "half_life_days", "hit_rate_z2_cross_mean_20d"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    if args.max_pairs:
        candidates = candidates.head(args.max_pairs).copy()

    output_dir = Path(args.output_dir)
    chart_dir = output_dir / args.chart_subdir
    chart_dir.mkdir(parents=True, exist_ok=True)
    price_dir = Path(args.price_dir)
    today = pd.Timestamp.today().normalize()
    calc_start = today - pd.DateOffset(years=args.calc_years)
    display_start = today - pd.DateOffset(years=args.display_years)

    rows = []
    for rank, row in enumerate(candidates.itertuples(index=False), start=1):
        row_data = pd.Series(row._asdict())
        output_path = chart_dir / safe_chart_filename(rank, row_data)
        calc_df, display_df, calc_stats = build_pair_data(
            price_dir,
            row_data,
            calc_start,
            display_start,
            args.rolling_corr_window,
        )
        equity_df, trades_df, backtest_metrics = run_backtest_overlay(
            calc_df,
            display_start,
            args.initial_capital,
            args.lookback,
            args.entry_z,
            args.exit_z,
            args.stop_z,
        )
        stats = plot_pair(
            row_data,
            display_df,
            output_path,
            args.rolling_corr_window,
            calc_stats,
            equity_df,
            trades_df,
            backtest_metrics,
        )
        rows.append({**row_data.to_dict(), "visual_rank": rank, **stats})

    index_df = pd.DataFrame(rows)
    index_df.to_csv(output_dir / "candidate_visual_index_1y_5ycalc.csv", index=False, encoding="utf-8-sig")
    plot_overview(index_df, output_dir)
    build_report(index_df, output_dir, args.chart_subdir, args.calc_years, args.display_years)

    print(f"Pair candidate visuals written to {output_dir}.")
    print(f"Chart folder: {chart_dir}")
    print(f"Pairs visualized: {len(index_df)}")
    print(f"Index: {output_dir / 'candidate_visual_index_1y_5ycalc.csv'}")
    print(f"Report: {output_dir / 'candidate_visual_report_1y_5ycalc.md'}")


if __name__ == "__main__":
    main()
